"""M1 / Block 2: topology-tampering attacks on MLP."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


import torch

from group_auth_dnn.attacks.topology import (
    attack_inject_params,
    attack_replace_node,
    attack_skip_node,
    attack_swap_nodes,
    create_unauthorized_shard_like,
)
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import verify_proof
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
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m1-mlp-v1")
    return parser.parse_args()


def _pick_swap_pair(nodes, x_input, session):
    if len(nodes) < 2:
        raise ValueError("swap attack requires at least two nodes")
    for idx1 in range(1, len(nodes)):
        idx2 = idx1 + 1
        result = attack_swap_nodes(nodes, x_input, session, idx1=idx1, idx2=idx2)
        if result.executable:
            return result
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 500)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M1",
        "block": "Block 2",
        "model": "MLP",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m1_topology_mlp_{depth}"
        shards = create_mlp_shards(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            max(args.base_layers, depth),
            depth,
            seed=args.seed,
        )
        keys = [generate_key(seed=args.seed + idx) for idx in range(depth)]
        honest_nodes = create_node_chain(shards, keys, route_id, quantization=args.quant_mode)
        registry = build_registry(honest_nodes)

        attack_stats = {
            "skip_node": {"accepts": 0, "valid": 0},
            "swap_order": {"accepts": 0, "valid": 0},
            "replace_node": {"accepts": 0, "valid": 0},
            "inject_params": {"accepts": 0, "valid": 0},
        }

        for sample_idx in range(args.num_samples):
            x = torch.randn(1, args.input_dim, generator=generator)
            session = create_session(route_id, args.model_version, x)
            target_index = min(2, depth)
            fake_shard = create_unauthorized_shard_like(honest_nodes[target_index - 1].model_shard, seed=args.seed + 1000 + sample_idx)

            attack_results = [
                attack_skip_node(honest_nodes, x, session, skip_index=target_index),
                _pick_swap_pair(honest_nodes, x, session),
                attack_replace_node(honest_nodes, x, session, replace_index=target_index, fake_shard=fake_shard),
                attack_inject_params(honest_nodes, x, session, inject_index=target_index, fake_shard=fake_shard),
            ]

            for attack_result in attack_results:
                if attack_result.executable and attack_result.bundle is not None:
                    attack_stats[attack_result.attack_name]["valid"] += 1
                    verdict = verify_proof(attack_result.bundle, registry, session, x)
                    attack_stats[attack_result.attack_name]["accepts"] += int(verdict.accepted)

        summary["topologies"][str(depth)] = {}
        for attack_name, values in attack_stats.items():
            valid = values["valid"]
            far = values["accepts"] / valid if valid else 0.0
            summary["topologies"][str(depth)][attack_name] = {
                "far": far,
                "valid_samples": valid,
                "failed_samples": args.num_samples - valid,
            }
            rows.append(
                {
                    "model": "mlp",
                    "N": depth,
                    "attack_type": attack_name,
                    "method_far": round(far, 6),
                    "valid_samples": valid,
                    "failed_samples": args.num_samples - valid,
                    "num_samples": args.num_samples,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m1_block2_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m1_block2_mlp.csv",
        rows,
        fieldnames=list(rows[0].keys()),
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", json_versioned.as_posix(), "implementation", "M1 Block 2 MLP 鎷撴墤鏀诲嚮缁撴灉 JSON"),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (manifest_ts, "/experiment-bridge", csv_versioned.as_posix(), "implementation", "M1 Block 2 MLP 鎷撴墤鏀诲嚮閫愭潯 CSV"),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print(f"summary_json={json_latest.as_posix()}")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]
        print(
            f"N={depth} "
            f"skip={info['skip_node']['far']:.3f} "
            f"swap={info['swap_order']['far']:.3f} "
            f"replace={info['replace_node']['far']:.3f} "
            f"inject={info['inject_params']['far']:.3f}"
        )


if __name__ == "__main__":
    main()

