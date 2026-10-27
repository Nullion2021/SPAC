"""M4 / R017: ablation study for removing measurement binding against inject-params on MLP."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


import torch

from group_auth_dnn.attacks.topology import attack_inject_params, create_unauthorized_shard_like
from group_auth_dnn.baselines.no_measurement import verify_proof_wo_measurement
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import verify_proof
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--base-layers", type=int, default=6)
    parser.add_argument("--topologies", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m4-mlp-v1")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_experiment_device(args.device)
    input_samples = build_random_vector_inputs(args.num_samples, args.input_dim, args.seed + 700, device=device)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M4",
        "block": "Block 6",
        "run_id": "R017",
        "model": "MLP",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "device": str(device),
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m4_measurement_mlp_{depth}"
        shards = create_mlp_shards(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            max(args.base_layers, depth),
            depth,
            seed=args.seed,
        )
        shards = [shard.to(device).eval() for shard in shards]
        keys = [generate_key(seed=args.seed + idx) for idx in range(depth)]
        honest_nodes = create_node_chain(shards, keys, route_id, quantization=args.quant_mode)
        registry = build_registry(honest_nodes)
        target_index = min(2, depth)

        stats = {
            "method_c": {"accepts": 0, "valid": 0},
            "wo_measurement": {"accepts": 0, "valid": 0},
        }

        for sample_idx, x_input in enumerate(input_samples):
            session = create_session(route_id, args.model_version, x_input)
            fake_shard = create_unauthorized_shard_like(
                honest_nodes[target_index - 1].model_shard,
                seed=args.seed + 2000 + sample_idx,
            )
            attack = attack_inject_params(
                honest_nodes,
                x_input,
                session,
                inject_index=target_index,
                fake_shard=fake_shard,
            )
            if attack.executable and attack.bundle is not None:
                stats["method_c"]["valid"] += 1
                stats["wo_measurement"]["valid"] += 1
                method_verdict = verify_proof(attack.bundle, registry, session, x_input)
                ablation_verdict = verify_proof_wo_measurement(attack.bundle, registry, session, x_input)
                stats["method_c"]["accepts"] += int(method_verdict.accepted)
                stats["wo_measurement"]["accepts"] += int(ablation_verdict.accepted)

        topology_summary = {}
        for variant_name, values in stats.items():
            valid = values["valid"]
            far = values["accepts"] / valid if valid else 0.0
            topology_summary[variant_name] = {
                "far": far,
                "valid_samples": valid,
                "failed_samples": args.num_samples - valid,
            }
            rows.append(
                {
                    "model": "mlp",
                    "N": depth,
                    "variant": variant_name,
                    "far": round(far, 6),
                    "valid_samples": valid,
                    "failed_samples": args.num_samples - valid,
                    "num_samples": args.num_samples,
                }
            )
        summary["topologies"][str(depth)] = topology_summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m4_ablation_no_measurement_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m4_ablation_no_measurement_mlp.csv",
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
                "M4 R017 MLP ablation without measurement binding JSON",
            ),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (
                manifest_ts,
                "/experiment-bridge",
                csv_versioned.as_posix(),
                "implementation",
                "M4 R017 MLP ablation without measurement binding CSV",
            ),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print("summary_json=results/m4_ablation_no_measurement_mlp.json")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]
        print(
            f"N={depth} "
            f"methodC={info['method_c']['far']:.3f} "
            f"wo_mu={info['wo_measurement']['far']:.3f}"
        )


if __name__ == "__main__":
    main()

