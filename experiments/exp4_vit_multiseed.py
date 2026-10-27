"""Experiment 4 supplement: real ViT/CIFAR-10 overhead and activation-encoding cost."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import math
import statistics
import time

import numpy as np
import torch

from group_auth_dnn.config import (
    RESNET_INPUT_CHANNELS,
    VIT_DIM,
    VIT_HEADS,
    VIT_IMAGE_SIZE,
    VIT_LAYERS,
    VIT_MLP_DIM,
    VIT_NUM_CLASSES,
    VIT_PATCH_SIZE,
    generate_key,
)
from group_auth_dnn.models.vit import create_vit_shards
from group_auth_dnn.protocol.chain import proof_size_breakdown, run_authenticated_inference, run_plain_inference, verify_proof
from group_auth_dnn.protocol.encoding import encode_activation, resolve_quant_config
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json

if __package__ in (None, ""):
    from m1_runners import build_cifar10_inputs, resolve_experiment_device
else:
    from .m1_runners import build_cifar10_inputs, resolve_experiment_device


DEFAULT_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_TOPOLOGIES = [5, 10]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--topologies", type=int, nargs="+", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="exp4-vit-v1")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument("--output-prefix", type=str, default="multiseed_exp4_vit")
    return parser.parse_args()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_nodes(*, seed: int, num_shards: int, route_id: str, quant_mode: str, device: torch.device):
    shards = create_vit_shards(
        VIT_IMAGE_SIZE,
        VIT_PATCH_SIZE,
        RESNET_INPUT_CHANNELS,
        VIT_DIM,
        VIT_HEADS,
        VIT_LAYERS,
        VIT_MLP_DIM,
        VIT_NUM_CLASSES,
        num_shards,
        seed=seed,
    )
    shards = [shard.to(device).eval() for shard in shards]
    keys = [generate_key(seed=seed + idx) for idx in range(1, num_shards + 1)]
    return create_node_chain(shards, keys, route_id=route_id, quantization=quant_mode)


def _measure_activation_encoding(nodes, x_input: torch.Tensor, quant_mode: str) -> tuple[float, float, int, str]:
    activation = x_input
    per_hop_ms: list[float] = []
    boundary_bytes: list[int] = []
    boundary_shapes: list[str] = []
    quantization = resolve_quant_config(quant_mode)
    for node in nodes:
        with torch.no_grad():
            activation = node.model_shard(activation)
        started = time.perf_counter_ns()
        payload = encode_activation(activation, mode=quantization)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        per_hop_ms.append(elapsed_ms)
        boundary_bytes.append(len(payload))
        boundary_shapes.append("x".join(str(dim) for dim in activation.shape))
    total_ms = float(sum(per_hop_ms))
    mean_ms = float(statistics.fmean(per_hop_ms)) if per_hop_ms else 0.0
    max_bytes = max(boundary_bytes) if boundary_bytes else 0
    representative_shape = boundary_shapes[0] if boundary_shapes else ""
    return total_ms, mean_ms, max_bytes, representative_shape


def _run_one_seed(
    *,
    seed: int,
    topologies: list[int],
    num_samples: int,
    warmup_runs: int,
    quant_mode: str,
    model_version: str,
    device: torch.device,
    data_root: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []
    input_samples = build_cifar10_inputs(num_samples + warmup_runs, seed, root=data_root)

    for num_shards in topologies:
        route_id = f"route_exp4_vit_{num_shards}"
        nodes = _build_nodes(seed=seed, num_shards=num_shards, route_id=route_id, quant_mode=quant_mode, device=device)
        registry = build_registry(nodes)

        for warmup_idx in range(warmup_runs):
            x_input = input_samples[warmup_idx].to(device)
            session = create_session(route_id, model_version, x_input)
            _ = run_plain_inference(nodes, x_input)
            _ = run_authenticated_inference(nodes, x_input, session)
            _ = _measure_activation_encoding(nodes, x_input, quant_mode)

        plain_times_ms: list[float] = []
        auth_times_ms: list[float] = []
        delta_times_ms: list[float] = []
        encode_total_ms: list[float] = []
        encode_mean_ms: list[float] = []
        max_boundary_bytes: list[int] = []
        delta_b_bytes: list[int] = []
        proof_total_bytes: list[int] = []
        representative_shape = ""

        for sample_idx in range(num_samples):
            x_input = input_samples[warmup_runs + sample_idx].to(device)
            session = create_session(route_id, model_version, x_input)

            _sync_device(device)
            plain_started = time.perf_counter_ns()
            plain_output = run_plain_inference(nodes, x_input)
            _sync_device(device)
            plain_ms = (time.perf_counter_ns() - plain_started) / 1_000_000.0

            _sync_device(device)
            auth_started = time.perf_counter_ns()
            proof = run_authenticated_inference(nodes, x_input, session)
            _sync_device(device)
            auth_ms = (time.perf_counter_ns() - auth_started) / 1_000_000.0

            verdict = verify_proof(proof, registry, session, x_input)
            if not verdict.accepted or not torch.allclose(proof.output, plain_output, atol=1e-5, rtol=1e-4):
                raise RuntimeError(f"ViT honest overhead run failed at seed={seed}, N={num_shards}: {verdict.reason}")

            enc_total, enc_mean, boundary_bytes, sample_shape = _measure_activation_encoding(nodes, x_input, quant_mode)
            if not representative_shape:
                representative_shape = sample_shape
            breakdown = proof_size_breakdown(proof)
            protocol_bytes = breakdown["z_bytes"] + breakdown["proof_items_bytes"]

            plain_times_ms.append(plain_ms)
            auth_times_ms.append(auth_ms)
            delta_times_ms.append(auth_ms - plain_ms)
            encode_total_ms.append(enc_total)
            encode_mean_ms.append(enc_mean)
            max_boundary_bytes.append(boundary_bytes)
            delta_b_bytes.append(protocol_bytes)
            proof_total_bytes.append(breakdown["total_bytes"])

            raw_rows.append(
                {
                    "seed": seed,
                    "model": "vit",
                    "N": num_shards,
                    "sample_idx": sample_idx,
                    "T_plain_ms": round(plain_ms, 6),
                    "T_auth_ms": round(auth_ms, 6),
                    "delta_T_ms": round(auth_ms - plain_ms, 6),
                    "encode_total_ms": round(enc_total, 6),
                    "encode_mean_per_hop_ms": round(enc_mean, 6),
                    "max_boundary_bytes": boundary_bytes,
                    "delta_B_bytes": protocol_bytes,
                    "P_serialized_bytes": breakdown["total_bytes"],
                    "representative_boundary_shape": sample_shape,
                }
            )

        seed_summary_rows.append(
            {
                "seed": seed,
                "model": "vit",
                "N": num_shards,
                "num_samples": num_samples,
                "T_plain_ms": round(statistics.fmean(plain_times_ms), 6),
                "T_auth_ms": round(statistics.fmean(auth_times_ms), 6),
                "delta_T_ms": round(statistics.fmean(delta_times_ms), 6),
                "throughput_sessions_s": round(1000.0 / statistics.fmean(auth_times_ms), 6),
                "encode_total_ms": round(statistics.fmean(encode_total_ms), 6),
                "encode_mean_per_hop_ms": round(statistics.fmean(encode_mean_ms), 6),
                "max_boundary_bytes": round(statistics.fmean([float(value) for value in max_boundary_bytes]), 2),
                "delta_B_bytes": round(statistics.fmean([float(value) for value in delta_b_bytes]), 2),
                "P_serialized_bytes": round(statistics.fmean([float(value) for value in proof_total_bytes]), 2),
                "representative_boundary_shape": representative_shape,
            }
        )

    return raw_rows, seed_summary_rows


def _summarize(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_keys = [
        "T_plain_ms",
        "T_auth_ms",
        "delta_T_ms",
        "throughput_sessions_s",
        "encode_total_ms",
        "encode_mean_per_hop_ms",
        "max_boundary_bytes",
        "delta_B_bytes",
        "P_serialized_bytes",
        "num_samples",
    ]
    grouped: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in seed_rows:
        grouped.setdefault((row["model"], row["N"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (model, depth), items in sorted(grouped.items(), key=lambda item: int(item[0][1])):
        out_row: dict[str, object] = {
            "model": model,
            "N": depth,
            "n_seeds": len(items),
            "representative_boundary_shape": items[0]["representative_boundary_shape"],
        }
        for metric in metric_keys:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            out_row[f"{metric}_mean"] = round(mean, 6)
            out_row[f"{metric}_std"] = round(std, 6)
            out_row[f"{metric}_ci95"] = round(ci95, 6)
        summary_rows.append(out_row)
    return summary_rows


def main() -> None:
    args = _parse_args()
    device = resolve_experiment_device(args.device)

    print("=" * 72)
    print("Experiment 4 Supplement: ViT/CIFAR-10 Overhead")
    print("=" * 72)
    print(
        f"device={device} seeds={args.seeds} topologies={args.topologies} "
        f"num_samples={args.num_samples}"
    )

    raw_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"  seed={seed}")
        seed_raw, seed_summary = _run_one_seed(
            seed=seed,
            topologies=args.topologies,
            num_samples=args.num_samples,
            warmup_runs=args.warmup_runs,
            quant_mode=args.quant_mode,
            model_version=args.model_version,
            device=device,
            data_root=args.data_root,
        )
        raw_rows.extend(seed_raw)
        seed_summary_rows.extend(seed_summary)

    summary_rows = _summarize(seed_summary_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_versioned, raw_latest = write_versioned_csv(
        Path("results") / f"{args.output_prefix}_raw.csv",
        raw_rows,
        fieldnames=list(raw_rows[0].keys()),
        timestamp=timestamp,
    )
    seed_versioned, seed_latest = write_versioned_csv(
        Path("results") / f"{args.output_prefix}_seed_summary.csv",
        seed_summary_rows,
        fieldnames=list(seed_summary_rows[0].keys()),
        timestamp=timestamp,
    )
    summary_versioned, summary_latest = write_versioned_csv(
        Path("results") / f"{args.output_prefix}_summary.csv",
        summary_rows,
        fieldnames=list(summary_rows[0].keys()),
        timestamp=timestamp,
    )
    bundle_versioned, bundle_latest = write_versioned_json(
        Path("results") / f"{args.output_prefix}_bundle.json",
        {
            "experiment": "exp4_vit_multiseed",
            "seeds": args.seeds,
            "topologies": args.topologies,
            "num_samples": args.num_samples,
            "device": str(device),
            "raw_rows": raw_rows,
            "seed_summary_rows": seed_summary_rows,
            "summary_rows": summary_rows,
        },
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/run-experiment", raw_versioned.as_posix(), "results", "exp4 ViT raw overhead rows"),
            (manifest_ts, "/run-experiment", raw_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/run-experiment", seed_versioned.as_posix(), "results", "exp4 ViT per-seed summary"),
            (manifest_ts, "/run-experiment", seed_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/run-experiment", summary_versioned.as_posix(), "results", "exp4 ViT official summary"),
            (manifest_ts, "/run-experiment", summary_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/run-experiment", bundle_versioned.as_posix(), "results", "exp4 ViT bundle"),
            (manifest_ts, "/run-experiment", bundle_latest.as_posix(), "results", "latest copy"),
        ]
    )

    print(f"\nSaved {summary_latest.as_posix()}")
    for row in summary_rows:
        print(
            f"N={row['N']} "
            f"delta_T={row['delta_T_ms_mean']}ms "
            f"encode_total={row['encode_total_ms_mean']}ms "
            f"throughput={row['throughput_sessions_s_mean']} sessions/s"
        )


if __name__ == "__main__":
    main()
