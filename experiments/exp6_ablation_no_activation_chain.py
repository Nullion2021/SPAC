"""M4 / R019: ablation study for removing the activation chain under AIA on MLP."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import sys


from group_auth_dnn.attacks.activation_injection import run_activation_injection_attack
from group_auth_dnn.baselines.baseline_e import build_cra_registry, create_cra_node_chain, verify_cra_proof
from group_auth_dnn.baselines.no_activation_chain import verify_proof_wo_activation_chain
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
    parser.add_argument("--topologies", type=int, nargs="+", default=[3])
    parser.add_argument("--target-hops", type=int, nargs="+", default=[2])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m4-mlp-v1")
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
        "run_id": "R019",
        "model": "MLP",
        "data_source": "synthetic Gaussian vectors",
        "quant_mode": args.quant_mode,
        "num_samples": args.num_samples,
        "device": str(device),
        "strengths": args.strengths,
        "target_hops": args.target_hops,
        "topologies": {},
    }

    for depth in args.topologies:
        route_id = f"route_m4_mlp_{depth}"
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
        method_nodes = create_node_chain(shards, keys, route_id, quantization=args.quant_mode)
        baseline_nodes = create_cra_node_chain(shards, keys, route_id)
        method_registry = build_registry(method_nodes)
        baseline_registry = build_cra_registry(baseline_nodes)
        effective_hops = [hop for hop in args.target_hops if 1 <= hop <= depth]

        topology_summary: dict[str, dict[str, object]] = {}
        for target_hop in effective_hops:
            hop_summary: dict[str, object] = {}
            for strength in args.strengths:
                baseline_accepts = 0
                ablation_accepts = 0
                method_accepts = 0
                injection_norms = []
                output_shifts = []
                alternate_gaps = []

                for sample_idx in range(args.num_samples):
                    x_input = input_samples[sample_idx]
                    alternate_input = input_samples[(sample_idx + 1) % len(input_samples)]
                    session = create_session(route_id, args.model_version, x_input)
                    attack = run_activation_injection_attack(
                        method_nodes,
                        baseline_nodes,
                        x_input,
                        alternate_input,
                        session,
                        target_hop=target_hop,
                        strength=strength,
                    )
                    baseline_verdict = verify_cra_proof(attack.baseline_bundle, baseline_registry, session)
                    ablation_verdict = verify_proof_wo_activation_chain(
                        attack.method_bundle,
                        method_registry,
                        session,
                        x_input,
                    )
                    method_verdict = verify_proof(attack.method_bundle, method_registry, session, x_input)
                    baseline_accepts += int(baseline_verdict.accepted)
                    ablation_accepts += int(ablation_verdict.accepted)
                    method_accepts += int(method_verdict.accepted)
                    injection_norms.append(attack.injection_l2)
                    output_shifts.append(attack.output_shift_l2)
                    alternate_gaps.append(attack.alternate_gap_l2)

                hop_summary[str(strength)] = {
                    "baseline_e_air": baseline_accepts / args.num_samples,
                    "wo_activation_chain_air": ablation_accepts / args.num_samples,
                    "method_c_air": method_accepts / args.num_samples,
                    "injection_l2_mean": statistics.mean(injection_norms),
                    "output_shift_l2_mean": statistics.mean(output_shifts),
                    "alternate_gap_l2_mean": statistics.mean(alternate_gaps),
                }
                rows.append(
                    {
                        "model": "mlp",
                        "N": depth,
                        "target_hop": target_hop,
                        "strength": strength,
                        "baseline_e_air": round(baseline_accepts / args.num_samples, 6),
                        "wo_activation_chain_air": round(ablation_accepts / args.num_samples, 6),
                        "method_c_air": round(method_accepts / args.num_samples, 6),
                        "injection_l2_mean": round(statistics.mean(injection_norms), 6),
                        "output_shift_l2_mean": round(statistics.mean(output_shifts), 6),
                        "alternate_gap_l2_mean": round(statistics.mean(alternate_gaps), 6),
                        "num_samples": args.num_samples,
                    }
                )
            topology_summary[str(target_hop)] = hop_summary
        summary["topologies"][str(depth)] = topology_summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(
        Path("results") / "m4_ablation_no_activation_chain_mlp.json",
        summary,
        timestamp=timestamp,
    )
    csv_versioned, csv_latest = write_versioned_csv(
        Path("results") / "m4_ablation_no_activation_chain_mlp.csv",
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
                "M4 R019 MLP ablation without activation chain JSON",
            ),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (
                manifest_ts,
                "/experiment-bridge",
                csv_versioned.as_posix(),
                "implementation",
                "M4 R019 MLP ablation without activation chain CSV",
            ),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )

    print("summary_json=results/m4_ablation_no_activation_chain_mlp.json")
    for depth in args.topologies:
        for target_hop in args.target_hops:
            if str(target_hop) not in summary["topologies"].get(str(depth), {}):
                continue
            for strength in args.strengths:
                info = summary["topologies"][str(depth)][str(target_hop)][str(strength)]
                print(
                    f"N={depth} hop={target_hop} strength={strength:.3f} "
                    f"baselineE={info['baseline_e_air']:.3f} "
                    f"wo_chain={info['wo_activation_chain_air']:.3f} "
                    f"methodC={info['method_c_air']:.3f}"
                )


if __name__ == "__main__":
    main()

