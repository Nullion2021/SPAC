"""M4 / R021: compare HMAC and Ed25519 signature overhead on honest MLP runs."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import statistics
import sys
import time


import torch

from group_auth_dnn.baselines.signature_mode import (
    build_signature_registry,
    create_signature_node_chain,
    run_signature_authenticated_inference,
    verify_signature_proof,
)
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import run_authenticated_inference, run_plain_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json

if __package__ in (None, ""):
    from m1_runners import build_random_vector_inputs, resolve_experiment_device
else:
    from .m1_runners import build_random_vector_inputs, resolve_experiment_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--base-layers", type=int, default=6)
    parser.add_argument("--topologies", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m4-signature-v1")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = resolve_experiment_device(args.device)
    input_samples = build_random_vector_inputs(
        args.num_samples + args.warmup_runs,
        args.input_dim,
        args.seed,
        device=device,
    )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M4",
        "block": "Block 6",
        "run_id": "R021",
        "model": "MLP",
        "data_source": "synthetic Gaussian vectors",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "warmup_runs": args.warmup_runs,
        "device": str(device),
        "signature_scheme": "Ed25519",
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m4_signature_mlp_{depth}"
        shards = create_mlp_shards(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            max(args.base_layers, depth),
            depth,
            seed=args.seed,
        )
        shards = [shard.to(device).eval() for shard in shards]
        hmac_nodes = create_node_chain(
            [copy.deepcopy(shard) for shard in shards],
            [generate_key(seed=args.seed + idx) for idx in range(depth)],
            route_id,
            quantization=args.quant_mode,
        )
        signature_nodes = create_signature_node_chain(
            [copy.deepcopy(shard) for shard in shards],
            [args.seed + 10_000 + idx for idx in range(depth)],
            route_id,
            quantization=args.quant_mode,
        )
        hmac_registry = build_registry(hmac_nodes)
        signature_registry = build_signature_registry(signature_nodes)

        plain_times: list[float] = []
        hmac_times: list[float] = []
        signature_times: list[float] = []
        hmac_accepts = 0
        signature_accepts = 0

        for x_input in input_samples[: args.warmup_runs]:
            session = create_session(route_id, args.model_version, x_input)
            _ = run_plain_inference(hmac_nodes, x_input)
            _ = run_authenticated_inference(hmac_nodes, x_input, session)
            _ = run_signature_authenticated_inference(signature_nodes, x_input, session)

        for x_input in input_samples[args.warmup_runs :]:
            session = create_session(route_id, args.model_version, x_input)

            _sync_device(device)
            plain_start = time.perf_counter_ns()
            plain_output = run_plain_inference(hmac_nodes, x_input)
            _sync_device(device)
            plain_elapsed_ms = (time.perf_counter_ns() - plain_start) / 1_000_000.0

            _sync_device(device)
            hmac_start = time.perf_counter_ns()
            hmac_proof = run_authenticated_inference(hmac_nodes, x_input, session)
            hmac_verdict = verify_proof(hmac_proof, hmac_registry, session, x_input)
            _sync_device(device)
            hmac_elapsed_ms = (time.perf_counter_ns() - hmac_start) / 1_000_000.0

            _sync_device(device)
            signature_start = time.perf_counter_ns()
            signature_proof = run_signature_authenticated_inference(signature_nodes, x_input, session)
            signature_verdict = verify_signature_proof(signature_proof, signature_registry, session, x_input)
            _sync_device(device)
            signature_elapsed_ms = (time.perf_counter_ns() - signature_start) / 1_000_000.0

            if not hmac_verdict.accepted or not torch.allclose(hmac_proof.output, plain_output):
                raise RuntimeError(f"HMAC honest run failed for N={depth}: {hmac_verdict.reason}")
            if not signature_verdict.accepted or not torch.allclose(signature_proof.output, plain_output):
                raise RuntimeError(f"Signature honest run failed for N={depth}: {signature_verdict.reason}")

            plain_times.append(plain_elapsed_ms)
            hmac_times.append(hmac_elapsed_ms)
            signature_times.append(signature_elapsed_ms)
            hmac_accepts += int(hmac_verdict.accepted)
            signature_accepts += int(signature_verdict.accepted)

        topology_summary = {
            "plain_ms": statistics.fmean(plain_times),
            "plain_ms_std": _std(plain_times),
            "hmac_ms": statistics.fmean(hmac_times),
            "hmac_ms_std": _std(hmac_times),
            "signature_ms": statistics.fmean(signature_times),
            "signature_ms_std": _std(signature_times),
            "delta_t_hmac_ms": statistics.fmean(hmac_times) - statistics.fmean(plain_times),
            "delta_t_signature_ms": statistics.fmean(signature_times) - statistics.fmean(plain_times),
            "signature_over_hmac_ratio": (
                statistics.fmean(signature_times) / statistics.fmean(hmac_times)
                if statistics.fmean(hmac_times) > 0
                else 0.0
            ),
            "hmac_accept_rate": hmac_accepts / args.num_samples,
            "signature_accept_rate": signature_accepts / args.num_samples,
        }
        summary["topologies"][str(depth)] = topology_summary
        rows.append(
            {
                "model": "mlp",
                "N": depth,
                "plain_ms": round(topology_summary["plain_ms"], 6),
                "hmac_ms": round(topology_summary["hmac_ms"], 6),
                "signature_ms": round(topology_summary["signature_ms"], 6),
                "delta_t_hmac_ms": round(topology_summary["delta_t_hmac_ms"], 6),
                "delta_t_signature_ms": round(topology_summary["delta_t_signature_ms"], 6),
                "signature_over_hmac_ratio": round(topology_summary["signature_over_hmac_ratio"], 6),
                "hmac_accept_rate": round(topology_summary["hmac_accept_rate"], 6),
                "signature_accept_rate": round(topology_summary["signature_accept_rate"], 6),
                "num_samples": args.num_samples,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m4_ablation_signature_mode_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m4_ablation_signature_mode_mlp.csv",
        rows,
        fieldnames=list(rows[0].keys()),
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (
                manifest_ts,
                "/experiment-bridge",
                json_versioned.as_posix(),
                "implementation",
                "M4 R021 HMAC vs signature overhead JSON",
            ),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (
                manifest_ts,
                "/experiment-bridge",
                csv_versioned.as_posix(),
                "implementation",
                "M4 R021 HMAC vs signature overhead CSV",
            ),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print("summary_json=results/m4_ablation_signature_mode_mlp.json")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]
        print(
            f"N={depth} "
            f"hmac={info['hmac_ms']:.3f}ms "
            f"sig={info['signature_ms']:.3f}ms "
            f"ratio={info['signature_over_hmac_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()

