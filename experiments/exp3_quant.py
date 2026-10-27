"""Experiment 3: quantization stability and theory-alignment stats for M2."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import sys


import numpy as np
import torch

from group_auth_dnn.config import (
    EXP3_QUANT_MODES,
    EXP_NUM_SAMPLES,
    MLP_HIDDEN_DIM,
    MLP_INPUT_DIM,
    MLP_NUM_LAYERS,
    MLP_OUTPUT_DIM,
    RESNET_INPUT_CHANNELS,
    RESNET_NUM_CLASSES,
    SEED,
    TOPOLOGY_DEPTHS,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_NUM_CLASSES,
    TRANSFORMER_SEQ_LEN,
    generate_key,
)
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.models.resnet import create_resnet_shards
from group_auth_dnn.models.transformer import create_transformer_shards
from group_auth_dnn.protocol.chain import run_authenticated_inference, run_plain_inference, verify_proof
from group_auth_dnn.protocol.encoding import encode_activation
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json


MODEL_ORDER = ("mlp", "resnet", "transformer")
PHASE_BINS = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-samples", type=int, default=EXP_NUM_SAMPLES)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _build_shards(model_type: str, num_shards: int, seed: int, device: torch.device):
    if model_type == "mlp":
        shards = create_mlp_shards(
            MLP_INPUT_DIM,
            MLP_HIDDEN_DIM,
            MLP_OUTPUT_DIM,
            max(MLP_NUM_LAYERS, num_shards),
            num_shards,
            seed=seed,
        )
    elif model_type == "resnet":
        shards = create_resnet_shards(RESNET_NUM_CLASSES, RESNET_INPUT_CHANNELS, num_shards, seed=seed)
    elif model_type == "transformer":
        shards = create_transformer_shards(
            TRANSFORMER_DIM,
            TRANSFORMER_HEADS,
            max(TRANSFORMER_LAYERS, num_shards),
            TRANSFORMER_SEQ_LEN,
            TRANSFORMER_NUM_CLASSES,
            num_shards,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return [shard.to(device).eval() for shard in shards]


def _make_input(model_type: str, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if model_type == "mlp":
        return torch.randn(1, MLP_INPUT_DIM, generator=generator, device=device)
    if model_type == "resnet":
        return torch.randn(1, RESNET_INPUT_CHANNELS, 32, 32, generator=generator, device=device)
    if model_type == "transformer":
        return torch.randn(1, TRANSFORMER_SEQ_LEN, TRANSFORMER_DIM, generator=generator, device=device)
    raise ValueError(f"Unsupported model type: {model_type}")


def _collect_plain_activations(shards: list[torch.nn.Module], x_input: torch.Tensor) -> list[torch.Tensor]:
    activations: list[torch.Tensor] = []
    current = x_input
    for shard in shards:
        with torch.no_grad():
            current = shard(current)
        activations.append(current.detach().cpu())
    return activations


def _fp16_phase_values(arr: np.ndarray) -> np.ndarray:
    q16 = arr.astype(np.float16)
    q32 = q16.astype(np.float32)
    neg_inf = np.full_like(q16, -np.inf, dtype=np.float16)
    pos_inf = np.full_like(q16, np.inf, dtype=np.float16)
    prev32 = np.nextafter(q16, neg_inf).astype(np.float32)
    next32 = np.nextafter(q16, pos_inf).astype(np.float32)
    lower = (prev32 + q32) * 0.5
    upper = (next32 + q32) * 0.5
    width = np.maximum(upper - lower, 1e-8)
    return np.clip((arr - lower) / width, 0.0, 1.0 - 1e-7)


def _int8_phase_values(arr: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    return np.mod(arr / scale, 1.0)


def _phase_values(tensor: torch.Tensor, encoding: str) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if encoding == "int8":
        return _int8_phase_values(arr)
    if encoding == "fp16":
        return _fp16_phase_values(arr)
    if encoding == "mixed":
        return _int8_phase_values(arr) if arr.ndim >= 3 else _fp16_phase_values(arr)
    raise ValueError(f"Unsupported encoding mode: {encoding}")


def _matrix_for_correlation(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.ndim == 4:
        return np.transpose(arr, (0, 2, 3, 1)).reshape(-1, arr.shape[1])
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    return arr.reshape(1, -1)


def _mean_abs_offdiag_correlation(matrix: np.ndarray) -> float | None:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    std = matrix.std(axis=0)
    matrix = matrix[:, std > 1e-12]
    if matrix.shape[1] < 2:
        return None
    corr = np.corrcoef(matrix, rowvar=False)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(vals.mean())


def _cell_width_mean(output: torch.Tensor, encoding: str) -> float:
    arr = output.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.size == 0:
        return 1.0
    if encoding == "int8" or (encoding == "mixed" and arr.ndim >= 3):
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
        return max_abs / 127.0 if max_abs > 0 else 1.0
    q16 = arr.astype(np.float16)
    q32 = q16.astype(np.float32)
    neg_inf = np.full_like(q16, -np.inf, dtype=np.float16)
    pos_inf = np.full_like(q16, np.inf, dtype=np.float16)
    prev32 = np.nextafter(q16, neg_inf).astype(np.float32)
    next32 = np.nextafter(q16, pos_inf).astype(np.float32)
    lower = (prev32 + q32) * 0.5
    upper = (next32 + q32) * 0.5
    return float(np.maximum(upper - lower, 1e-8).mean())


def _theory_bound(reference_output: torch.Tensor, auth_output: torch.Tensor, encoding: str) -> float:
    epsilon = float((reference_output - auth_output).detach().abs().max().item())
    if epsilon <= 0.0:
        return 1.0
    delta = _cell_width_mean(reference_output, encoding)
    hazard = min(1.0, (2.0 * epsilon) / max(delta, 1e-8))
    return max(0.0, 1.0 - hazard) ** reference_output.numel()


def _run_single_config(
    model_type: str,
    num_shards: int,
    encoding: str,
    base_seed: int,
    num_samples: int,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, str]]:
    seed_offset = MODEL_ORDER.index(model_type) * 10_000 + num_shards * 100 + EXP3_QUANT_MODES.index(encoding) * 10
    generator = _make_generator(base_seed + seed_offset, device)
    route_id = f"route_exp3_{model_type}_{num_shards}_{encoding}"
    model_version = f"exp3-{model_type}-v1"

    shards = _build_shards(model_type, num_shards, base_seed + seed_offset, device)
    keys = [generate_key(seed=base_seed + seed_offset + idx) for idx in range(1, num_shards + 1)]
    nodes = create_node_chain(shards, keys, route_id=route_id, quantization=encoding)
    registry = build_registry(nodes)

    qmatch_count = 0
    reject_count = 0
    bound_values: list[float] = []
    phase_hist = np.zeros(PHASE_BINS, dtype=np.float64)
    phase_total = 0.0
    channel_scores: list[float] = []
    pending_rows: dict[int, list[np.ndarray]] = {}

    for _ in range(num_samples):
        x_input = _make_input(model_type, generator, device)
        session = create_session(route_id, model_version, x_input)
        proof = run_authenticated_inference(nodes, x_input, session)
        reference_output = run_plain_inference(nodes, x_input)
        verification = verify_proof(proof, registry, session, x_input)

        qmatch = encode_activation(proof.output, mode=encoding) == encode_activation(reference_output, mode=encoding)
        if qmatch:
            qmatch_count += 1
        if not (verification.accepted and qmatch):
            reject_count += 1

        bound_values.append(_theory_bound(reference_output, proof.output, encoding))

        for activation in _collect_plain_activations(shards, x_input):
            phases = _phase_values(activation, encoding).reshape(-1)
            hist, _ = np.histogram(phases, bins=PHASE_BINS, range=(0.0, 1.0))
            phase_hist += hist
            phase_total += phases.size

            matrix = _matrix_for_correlation(activation)
            score = _mean_abs_offdiag_correlation(matrix)
            if score is not None:
                channel_scores.append(score)
            elif matrix.ndim == 2 and matrix.shape[1] >= 2:
                pending_rows.setdefault(matrix.shape[1], []).append(matrix)

    for matrices in pending_rows.values():
        stacked = np.concatenate(matrices, axis=0)
        score = _mean_abs_offdiag_correlation(stacked)
        if score is not None:
            channel_scores.append(score)

    qmatch_rate = qmatch_count / num_samples
    frr = reject_count / num_samples
    bound_mean = statistics.fmean(bound_values) if bound_values else 0.0
    if phase_total > 0:
        phase_dist = phase_hist / phase_total
        uniform = np.full(PHASE_BINS, 1.0 / PHASE_BINS, dtype=np.float64)
        sa1 = float(np.mean(np.abs(phase_dist - uniform)))
    else:
        sa1 = 0.0
    sa2 = statistics.fmean(channel_scores) if channel_scores else 0.0

    stability_row = {
        "model": model_type,
        "N": str(num_shards),
        "encoding": encoding,
        "QMatch": f"{qmatch_rate:.6f}",
        "FRR": f"{frr:.6f}",
    }
    theory_row = {
        "model": model_type,
        "N": str(num_shards),
        "encoding": encoding,
        "Bound_theory": f"{bound_mean:.6f}",
        "delta_QMatch": f"{(qmatch_rate - bound_mean):.6f}",
        "SA1_phase_deviation": f"{sa1:.6f}",
        "SA2_channel_correlation": f"{sa2:.6f}",
    }
    return stability_row, theory_row


def run_exp3(base_seed: int = SEED, num_samples: int = EXP_NUM_SAMPLES, device_arg: str = "auto") -> list[dict[str, str]]:
    stability_rows: list[dict[str, str]] = []
    theory_rows: list[dict[str, str]] = []
    device = _resolve_device(device_arg)

    print("=" * 72)
    print("Experiment 3: Quantization Stability and Theory Alignment")
    print("=" * 72)
    print(f"Using device: {device}")

    for model_type in MODEL_ORDER:
        for num_shards in TOPOLOGY_DEPTHS:
            for encoding in EXP3_QUANT_MODES:
                print(f"  model={model_type}, N={num_shards}, encoding={encoding}")
                stability_row, theory_row = _run_single_config(
                    model_type,
                    num_shards,
                    encoding,
                    base_seed,
                    num_samples,
                    device,
                )
                stability_rows.append(stability_row)
                theory_rows.append(theory_row)
                print(
                    "    "
                    f"QMatch={stability_row['QMatch']}, FRR={stability_row['FRR']}, "
                    f"Bound={theory_row['Bound_theory']}, delta_QMatch={theory_row['delta_QMatch']}"
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stability_versioned, stability_latest = write_versioned_csv(
        Path("results") / "exp3_quant_stability.csv",
        stability_rows,
        fieldnames=["model", "N", "encoding", "QMatch", "FRR"],
        timestamp=timestamp,
    )
    theory_versioned, theory_latest = write_versioned_csv(
        Path("results") / "exp3_quant_theory.csv",
        theory_rows,
        fieldnames=[
            "model",
            "N",
            "encoding",
            "Bound_theory",
            "delta_QMatch",
            "SA1_phase_deviation",
            "SA2_channel_correlation",
        ],
        timestamp=timestamp,
    )
    summary_versioned, summary_latest = write_versioned_json(
        Path("results") / "exp3_quant_summary.json",
        {
            "milestone": "M2",
            "experiment": "exp3_quant",
            "seed": base_seed,
            "num_samples": num_samples,
            "device": str(device),
            "stability_rows": stability_rows,
            "theory_rows": theory_rows,
        },
        timestamp=timestamp,
    )

    manifest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (
                manifest_timestamp,
                "/experiment-bridge",
                stability_versioned.as_posix(),
                "results",
                "M2 exp3 stability table with model, depth, encoding, QMatch, and FRR",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                stability_latest.as_posix(),
                "results",
                "latest copy",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                theory_versioned.as_posix(),
                "results",
                "M2 exp3 theory-alignment stats: Bound_theory, delta_QMatch, SA-1, SA-2",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                theory_latest.as_posix(),
                "results",
                "latest copy",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_versioned.as_posix(),
                "results",
                "M2 exp3 JSON summary",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_latest.as_posix(),
                "results",
                "latest copy",
            ),
        ]
    )

    print(f"\nSaved {stability_latest.as_posix()}")
    print(f"Saved {theory_latest.as_posix()}")
    print(f"Saved {summary_latest.as_posix()}")
    return stability_rows


if __name__ == "__main__":
    args = _parse_args()
    run_exp3(base_seed=args.seed, num_samples=args.num_samples, device_arg=args.device)

