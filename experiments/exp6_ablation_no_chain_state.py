"""M4 / R020: ablation study for removing z_i against reordered-route attacks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


import torch
import torch.nn as nn

from group_auth_dnn.attacks.order_splicing import forge_logical_order_transcript, skip_then_backfill_order, swap_execution_order
from group_auth_dnn.baselines.no_chain_state import run_inference_wo_chain_state, verify_proof_wo_chain_state
from group_auth_dnn.config import generate_key
from group_auth_dnn.protocol.chain import run_authenticated_inference, verify_proof
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
    parser.add_argument("--topologies", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m4-chain-v1")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_experiment_device(args.device)
    input_samples = build_random_vector_inputs(args.num_samples, args.input_dim, args.seed, device=device)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M4",
        "block": "Block 6",
        "run_id": "R020",
        "model": "controlled identity-compatible MLP chain",
        "data_source": "synthetic Gaussian vectors",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "device": str(device),
        "attack_note": (
            "All hops are activation-preserving identity shards so local u_i/v_i checks remain satisfied; "
            "the only remaining order signal is whether tags are recursively chained through z_i."
        ),
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m4_chain_state_{depth}"
        shards = [nn.Identity().to(device).eval() for _ in range(depth)]
        keys = [generate_key(seed=args.seed + idx) for idx in range(depth)]
        nodes = create_node_chain(shards, keys, route_id, quantization=args.quant_mode)
        registry = build_registry(nodes)

        attack_orders = {
            "swap_order": lambda current_nodes: swap_execution_order(current_nodes, 2, 3),
            "skip_then_backfill": lambda current_nodes: skip_then_backfill_order(current_nodes, 2),
        }
        stats = {
            attack_name: {"method_c": 0, "wo_chain_state": 0}
            for attack_name in attack_orders
        }

        for x_input in input_samples:
            session = create_session(route_id, args.model_version, x_input)
            for attack_name, make_order in attack_orders.items():
                physical_nodes = make_order(nodes)
                method_bundle = forge_logical_order_transcript(
                    run_authenticated_inference(physical_nodes, x_input, session)
                )
                no_chain_bundle = forge_logical_order_transcript(
                    run_inference_wo_chain_state(physical_nodes, x_input, session)
                )
                method_verdict = verify_proof(method_bundle, registry, session, x_input)
                no_chain_verdict = verify_proof_wo_chain_state(no_chain_bundle, registry, session, x_input)
                stats[attack_name]["method_c"] += int(method_verdict.accepted)
                stats[attack_name]["wo_chain_state"] += int(no_chain_verdict.accepted)

        topology_summary: dict[str, object] = {}
        for attack_name, values in stats.items():
            topology_summary[attack_name] = {
                "method_c_far": values["method_c"] / args.num_samples,
                "wo_chain_state_far": values["wo_chain_state"] / args.num_samples,
            }
            rows.append(
                {
                    "model": "identity_chain",
                    "N": depth,
                    "attack_type": attack_name,
                    "method_c_far": round(values["method_c"] / args.num_samples, 6),
                    "wo_chain_state_far": round(values["wo_chain_state"] / args.num_samples, 6),
                    "num_samples": args.num_samples,
                }
            )
        summary["topologies"][str(depth)] = topology_summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m4_ablation_no_chain_state_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m4_ablation_no_chain_state_mlp.csv",
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
                "M4 R020 controlled chain-state ablation JSON",
            ),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (
                manifest_ts,
                "/experiment-bridge",
                csv_versioned.as_posix(),
                "implementation",
                "M4 R020 controlled chain-state ablation CSV",
            ),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print("summary_json=results/m4_ablation_no_chain_state_mlp.json")
    for depth in args.topologies:
        for attack_name, values in summary["topologies"][str(depth)].items():
            print(
                f"N={depth} attack={attack_name} "
                f"methodC={values['method_c_far']:.3f} "
                f"wo_z={values['wo_chain_state_far']:.3f}"
            )


if __name__ == "__main__":
    main()

