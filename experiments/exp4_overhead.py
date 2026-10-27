"""Experiment 4: controlled-environment overhead measurements for M2."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import sys
import time


import torch

from group_auth_dnn.config import (
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
from group_auth_dnn.protocol.chain import proof_size_breakdown, run_authenticated_inference, run_plain_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json


MODEL_ORDER = ("mlp", "resnet", "transformer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-samples", type=int, default=EXP_NUM_SAMPLES)
    parser.add_argument("--warmup-runs", type=int, default=3)
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


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_exp4(
    base_seed: int = SEED,
    num_samples: int = EXP_NUM_SAMPLES,
    warmup_runs: int = 3,
    device_arg: str = "auto",
) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    per_run_rows: list[dict[str, str]] = []
    device = _resolve_device(device_arg)

    print("=" * 72)
    print("Experiment 4: System Overhead")
    print("=" * 72)
    print(f"Using device: {device}")

    for model_type in MODEL_ORDER:
        for num_shards in TOPOLOGY_DEPTHS:
            seed_offset = MODEL_ORDER.index(model_type) * 10_000 + num_shards * 100
            route_id = f"route_exp4_{model_type}_{num_shards}"
            model_version = f"exp4-{model_type}-v1"
            generator = _make_generator(base_seed + seed_offset, device)

            shards = _build_shards(model_type, num_shards, base_seed + seed_offset, device)
            keys = [generate_key(seed=base_seed + seed_offset + idx) for idx in range(1, num_shards + 1)]
            nodes = create_node_chain(shards, keys, route_id=route_id, quantization="int8")
            registry = build_registry(nodes)

            print(f"  model={model_type}, N={num_shards}")

            for _ in range(warmup_runs):
                x_input = _make_input(model_type, generator, device)
                session = create_session(route_id, model_version, x_input)
                _ = run_plain_inference(nodes, x_input)
                _ = run_authenticated_inference(nodes, x_input, session)

            plain_times_ms: list[float] = []
            auth_times_ms: list[float] = []
            delta_times_ms: list[float] = []
            delta_protocol_bytes: list[int] = []
            proof_total_bytes: list[int] = []
            y_bytes: list[int] = []
            z_bytes: list[int] = []
            proof_items_bytes: list[int] = []

            for run_index in range(num_samples):
                x_input = _make_input(model_type, generator, device)
                session = create_session(route_id, model_version, x_input)

                _sync_device(device)
                plain_start = time.perf_counter_ns()
                plain_output = run_plain_inference(nodes, x_input)
                _sync_device(device)
                plain_elapsed_ms = (time.perf_counter_ns() - plain_start) / 1_000_000.0

                _sync_device(device)
                auth_start = time.perf_counter_ns()
                proof = run_authenticated_inference(nodes, x_input, session)
                _sync_device(device)
                auth_elapsed_ms = (time.perf_counter_ns() - auth_start) / 1_000_000.0

                verification = verify_proof(proof, registry, session, x_input)
                if not verification.accepted or not torch.allclose(proof.output, plain_output):
                    raise RuntimeError(
                        f"exp4 honest run failed for {model_type} N={num_shards}: {verification.reason}"
                    )

                breakdown = proof_size_breakdown(proof)
                protocol_bytes = breakdown["z_bytes"] + breakdown["proof_items_bytes"]

                plain_times_ms.append(plain_elapsed_ms)
                auth_times_ms.append(auth_elapsed_ms)
                delta_times_ms.append(auth_elapsed_ms - plain_elapsed_ms)
                delta_protocol_bytes.append(protocol_bytes)
                proof_total_bytes.append(breakdown["total_bytes"])
                y_bytes.append(breakdown["y_bytes"])
                z_bytes.append(breakdown["z_bytes"])
                proof_items_bytes.append(breakdown["proof_items_bytes"])

                per_run_rows.append(
                    {
                        "model": model_type,
                        "N": str(num_shards),
                        "run_index": str(run_index),
                        "T_plain_ms": f"{plain_elapsed_ms:.6f}",
                        "T_auth_ms": f"{auth_elapsed_ms:.6f}",
                        "delta_T_ms": f"{(auth_elapsed_ms - plain_elapsed_ms):.6f}",
                        "delta_B_protocol_bytes": str(protocol_bytes),
                        "delta_B_bytes": str(protocol_bytes),
                        "P_serialized_bytes": str(breakdown["total_bytes"]),
                        "y_bytes": str(breakdown["y_bytes"]),
                        "z_bytes": str(breakdown["z_bytes"]),
                        "proof_items_bytes": str(breakdown["proof_items_bytes"]),
                    }
                )

            row = {
                "model": model_type,
                "N": str(num_shards),
                "T_plain_ms": f"{statistics.fmean(plain_times_ms):.6f}",
                "T_plain_ms_std": f"{_std(plain_times_ms):.6f}",
                "T_auth_ms": f"{statistics.fmean(auth_times_ms):.6f}",
                "T_auth_ms_std": f"{_std(auth_times_ms):.6f}",
                "delta_T_ms": f"{statistics.fmean(delta_times_ms):.6f}",
                "delta_T_ms_std": f"{_std(delta_times_ms):.6f}",
                "delta_B_protocol_bytes": f"{statistics.fmean(delta_protocol_bytes):.2f}",
                "delta_B_protocol_bytes_std": f"{_std([float(v) for v in delta_protocol_bytes]):.6f}",
                "delta_B_bytes": f"{statistics.fmean(delta_protocol_bytes):.2f}",
                "delta_B_bytes_std": f"{_std([float(v) for v in delta_protocol_bytes]):.6f}",
                "P_serialized_bytes": f"{statistics.fmean(proof_total_bytes):.2f}",
                "P_serialized_bytes_std": f"{_std([float(v) for v in proof_total_bytes]):.6f}",
                "y_bytes": f"{statistics.fmean(y_bytes):.2f}",
                "z_bytes": f"{statistics.fmean(z_bytes):.2f}",
                "proof_items_bytes": f"{statistics.fmean(proof_items_bytes):.2f}",
            }
            summary_rows.append(row)
            print(
                "    "
                f"T_plain={row['T_plain_ms']}ms, T_auth={row['T_auth_ms']}ms, "
                f"delta_T={row['delta_T_ms']}ms, delta_B_protocol={row['delta_B_protocol_bytes']}B, "
                f"|P|={row['P_serialized_bytes']}B"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_versioned, summary_latest = write_versioned_csv(
        Path("results") / "exp4_overhead.csv",
        summary_rows,
        fieldnames=[
            "model",
            "N",
            "T_plain_ms",
            "T_plain_ms_std",
            "T_auth_ms",
            "T_auth_ms_std",
            "delta_T_ms",
            "delta_T_ms_std",
            "delta_B_protocol_bytes",
            "delta_B_protocol_bytes_std",
            "delta_B_bytes",
            "delta_B_bytes_std",
            "P_serialized_bytes",
            "P_serialized_bytes_std",
            "y_bytes",
            "z_bytes",
            "proof_items_bytes",
        ],
        timestamp=timestamp,
    )
    runs_versioned, runs_latest = write_versioned_csv(
        Path("results") / "exp4_overhead_runs.csv",
        per_run_rows,
        fieldnames=[
            "model",
            "N",
            "run_index",
            "T_plain_ms",
            "T_auth_ms",
            "delta_T_ms",
            "delta_B_protocol_bytes",
            "delta_B_bytes",
            "P_serialized_bytes",
            "y_bytes",
            "z_bytes",
            "proof_items_bytes",
        ],
        timestamp=timestamp,
    )
    summary_json_versioned, summary_json_latest = write_versioned_json(
        Path("results") / "exp4_overhead_summary.json",
        {
            "milestone": "M2",
            "experiment": "exp4_overhead",
            "seed": base_seed,
            "num_samples": num_samples,
            "warmup_runs": warmup_runs,
            "device": str(device),
            "summary_rows": summary_rows,
        },
        timestamp=timestamp,
    )

    manifest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_versioned.as_posix(),
                "results",
                "M2 exp4 summary table with latency and serialized-proof overhead metrics",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_latest.as_posix(),
                "results",
                "latest copy",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                runs_versioned.as_posix(),
                "results",
                "M2 exp4 per-run overhead measurements",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                runs_latest.as_posix(),
                "results",
                "latest copy",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_json_versioned.as_posix(),
                "results",
                "M2 exp4 JSON summary",
            ),
            (
                manifest_timestamp,
                "/experiment-bridge",
                summary_json_latest.as_posix(),
                "results",
                "latest copy",
            ),
        ]
    )

    print(f"\nSaved {summary_latest.as_posix()}")
    print(f"Saved {runs_latest.as_posix()}")
    print(f"Saved {summary_json_latest.as_posix()}")
    return summary_rows


if __name__ == "__main__":
    args = _parse_args()
    run_exp4(base_seed=args.seed, num_samples=args.num_samples, warmup_runs=args.warmup_runs, device_arg=args.device)

