"""M4 / R018: ablation study for removing sid/nonce freshness against replay on MLP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


from group_auth_dnn.attacks.replay import nonce_from_int, prepare_cross_session_replay
from group_auth_dnn.baselines.no_freshness import verify_proof_wo_freshness
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain

if __package__ in (None, ""):
    from m1_runners import build_random_vector_inputs, resolve_experiment_device, write_experiment_outputs
else:
    from .m1_runners import build_random_vector_inputs, resolve_experiment_device, write_experiment_outputs


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
    parser.add_argument("--issued-at", type=float, default=1234.0)
    parser.add_argument("--lifetime", type=float, default=3600.0)
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
        "run_id": "R018",
        "model": "MLP",
        "data_source": "synthetic Gaussian vectors",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "device": str(device),
        "replay_setup": (
            "same input / route / model / issued_at, but different nonce so the "
            "replayed proof is evaluated under a fresh sid"
        ),
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m4_freshness_mlp_{depth}"
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
        nodes = create_node_chain(shards, keys, route_id, quantization=args.quant_mode)
        registry = build_registry(nodes)

        stats = {
            "method_c": {"accepts": 0},
            "wo_freshness": {"accepts": 0},
        }
        sid_changed = 0

        for sample_idx, x_input in enumerate(input_samples):
            issued_at = args.issued_at + sample_idx
            attack = prepare_cross_session_replay(
                nodes,
                x_input,
                route_id,
                args.model_version,
                issued_at=issued_at,
                lifetime=args.lifetime,
                original_nonce=nonce_from_int((sample_idx * 2) + 1),
                replay_nonce=nonce_from_int((sample_idx * 2) + 2),
            )
            verify_now = issued_at + (args.lifetime / 2.0)
            sid_changed += int(attack.original_session.sid != attack.replay_session.sid)
            method_verdict = verify_proof(
                attack.bundle,
                registry,
                attack.replay_session,
                x_input,
                now=verify_now,
            )
            ablation_verdict = verify_proof_wo_freshness(
                attack.bundle,
                registry,
                attack.replay_session,
                x_input,
                now=verify_now,
            )
            stats["method_c"]["accepts"] += int(method_verdict.accepted)
            stats["wo_freshness"]["accepts"] += int(ablation_verdict.accepted)

        topology_summary = {
            "sid_changed_rate": sid_changed / args.num_samples,
            "method_c": {
                "replay_accept_rate": stats["method_c"]["accepts"] / args.num_samples,
            },
            "wo_freshness": {
                "replay_accept_rate": stats["wo_freshness"]["accepts"] / args.num_samples,
            },
        }
        summary["topologies"][str(depth)] = topology_summary

        for variant_name, values in stats.items():
            rows.append(
                {
                    "model": "mlp",
                    "N": depth,
                    "variant": variant_name,
                    "replay_accept_rate": round(values["accepts"] / args.num_samples, 6),
                    "sid_changed_rate": round(sid_changed / args.num_samples, 6),
                    "num_samples": args.num_samples,
                }
            )

    write_experiment_outputs(
        Path("results") / "m4_ablation_no_freshness_mlp.json",
        Path("results") / "m4_ablation_no_freshness_mlp.csv",
        summary,
        rows,
        (
            "M4 R018 MLP ablation without sid/nonce freshness JSON",
            "M4 R018 MLP ablation without sid/nonce freshness CSV",
        ),
    )

    print("summary_json=results/m4_ablation_no_freshness_mlp.json")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]
        print(
            f"N={depth} "
            f"sid_changed={info['sid_changed_rate']:.3f} "
            f"methodC={info['method_c']['replay_accept_rate']:.3f} "
            f"wo_freshness={info['wo_freshness']['replay_accept_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

