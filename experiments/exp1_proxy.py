"""M1 / Block 1: proxy-computation and model-substitution attacks on MLP."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import sys


import torch

from group_auth_dnn.attacks.proxy_model import run_proxy_attack_suite
from group_auth_dnn.baselines.baseline_a import evaluate_delta_scan
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import run_plain_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--base-layers", type=int, default=6)
    parser.add_argument("--topologies", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--proxy-seed", type=int, default=999)
    parser.add_argument("--deltas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m1-mlp-v1")
    return parser.parse_args()


def create_mlp_factory(args: argparse.Namespace):
    def factory(*, num_shards: int, seed: int):
        return create_mlp_shards(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            max(args.base_layers, num_shards),
            num_shards,
            seed=seed,
        )

    return factory


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 100)
    create_shards_fn = create_mlp_factory(args)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M1",
        "block": "Block 1",
        "model": "MLP",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "deltas": args.deltas,
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m1_mlp_{depth}"
        honest_shards = create_shards_fn(num_shards=depth, seed=args.seed)
        keys = [generate_key(seed=args.seed + idx) for idx in range(depth)]
        honest_nodes = create_node_chain(honest_shards, keys, route_id, quantization=args.quant_mode)
        registry = build_registry(honest_nodes)

        attack_names = ["random_proxy", "single_machine_full_model"]
        baseline_counts = {attack_name: {delta: 0 for delta in args.deltas} for attack_name in attack_names}
        method_counts = {attack_name: 0 for attack_name in attack_names}
        distances = {attack_name: [] for attack_name in attack_names}

        for sample_idx in range(args.num_samples):
            x = torch.randn(1, args.input_dim, generator=generator)
            session = create_session(route_id, args.model_version, x)
            honest_output = run_plain_inference(honest_nodes, x)

            for attack_case in run_proxy_attack_suite(
                x,
                session,
                honest_nodes,
                create_shards_fn,
                depth,
                keys,
                proxy_seed=args.proxy_seed + sample_idx,
                quantization=args.quant_mode,
            ):
                delta_hits = evaluate_delta_scan(attack_case.output, honest_output, args.deltas)
                for delta, accepted in delta_hits.items():
                    baseline_counts[attack_case.name][delta] += int(accepted)
                verification = verify_proof(attack_case.forged_bundle, registry, session, x)
                method_counts[attack_case.name] += int(verification.accepted)
                distances[attack_case.name].append(float((attack_case.output - honest_output).norm(p=2).item()))

        topology_summary = {}
        for attack_name in attack_names:
            topology_summary[attack_name] = {
                "baseline_asr_by_delta": {
                    str(delta): baseline_counts[attack_name][delta] / args.num_samples for delta in args.deltas
                },
                "method_far": method_counts[attack_name] / args.num_samples,
                "distance_mean": statistics.mean(distances[attack_name]),
            }
            for delta in args.deltas:
                rows.append(
                    {
                        "model": "mlp",
                        "N": depth,
                        "attack_type": attack_name,
                        "delta": delta,
                        "baseline_asr": round(baseline_counts[attack_name][delta] / args.num_samples, 6),
                        "method_far": round(method_counts[attack_name] / args.num_samples, 6),
                        "distance_mean": round(statistics.mean(distances[attack_name]), 6),
                        "num_samples": args.num_samples,
                    }
                )
        summary["topologies"][str(depth)] = topology_summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m1_block1_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m1_block1_mlp.csv",
        rows,
        fieldnames=list(rows[0].keys()),
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", json_versioned.as_posix(), "implementation", "M1 Block 1 MLP 浠ｇ悊鏀诲嚮缁撴灉 JSON"),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (manifest_ts, "/experiment-bridge", csv_versioned.as_posix(), "implementation", "M1 Block 1 MLP 浠ｇ悊鏀诲嚮閫愭潯 CSV"),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print(f"summary_json={json_latest.as_posix()}")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]["single_machine_full_model"]
        print(
            f"N={depth} single_machine_full_model "
            f"ASR@1e-3={info['baseline_asr_by_delta']['0.001']:.3f} "
            f"FAR={info['method_far']:.3f}"
        )


if __name__ == "__main__":
    main()

