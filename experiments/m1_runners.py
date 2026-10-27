"""Shared runners for M1 Block 1 and Block 2 experiments."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torchvision import datasets, transforms

from group_auth_dnn.attacks.proxy_model import run_proxy_attack_suite
from group_auth_dnn.attacks.topology import (
    attack_inject_params,
    attack_replace_node,
    attack_skip_node,
    attack_swap_nodes,
    create_unauthorized_shard_like,
)
from group_auth_dnn.baselines.baseline_a import evaluate_delta_scan
from group_auth_dnn.config import generate_key
from group_auth_dnn.protocol.chain import run_plain_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json


TensorList = list[torch.Tensor]
ShardFactory = Callable[..., list[torch.nn.Module]]


def resolve_experiment_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def build_random_vector_inputs(
    num_samples: int,
    input_dim: int,
    seed: int,
    device: torch.device | str | None = None,
) -> TensorList:
    resolved_device = device if isinstance(device, torch.device) else resolve_experiment_device(device or "cpu")
    generator = torch.Generator(device=resolved_device.type).manual_seed(seed) if resolved_device.type == "cuda" else torch.Generator().manual_seed(seed)
    return [torch.randn(1, input_dim, generator=generator, device=resolved_device) for _ in range(num_samples)]


def build_cifar10_inputs(num_samples: int, seed: int, root: str | Path = "data/cifar10") -> TensorList:
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    indices = permutation[:num_samples].tolist()
    return [dataset[idx][0].unsqueeze(0) for idx in indices]


def build_toy_sequence_inputs(num_samples: int, seq_len: int, dim: int, seed: int) -> TensorList:
    generator = torch.Generator().manual_seed(seed)
    positions = torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32).view(1, seq_len, 1)
    feature_freq = torch.linspace(1.0, 3.0, steps=dim, dtype=torch.float32).view(1, 1, dim)
    token_bias = torch.linspace(-0.5, 0.5, steps=seq_len, dtype=torch.float32).view(1, seq_len, 1)

    samples: TensorList = []
    for _ in range(num_samples):
        phase = 2.0 * math.pi * torch.rand((1, 1, dim), generator=generator, dtype=torch.float32)
        amplitude = 0.5 + torch.rand((1, 1, dim), generator=generator, dtype=torch.float32)
        harmonic = amplitude * torch.sin((2.0 * math.pi * positions * feature_freq) + phase)
        noise = 0.05 * torch.randn((1, seq_len, dim), generator=generator, dtype=torch.float32)
        samples.append((harmonic + noise + token_bias).contiguous())
    return samples


def pick_swap_attack(nodes, x_input: torch.Tensor, session):
    result = None
    for idx1 in range(1, len(nodes)):
        idx2 = idx1 + 1
        result = attack_swap_nodes(nodes, x_input, session, idx1=idx1, idx2=idx2)
        if result.executable:
            return result
    if result is None:
        raise ValueError("swap attack requires at least two nodes")
    return result


def write_experiment_outputs(
    latest_json_path: str | Path,
    latest_csv_path: str | Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
    manifest_descriptions: tuple[str, str],
) -> tuple[Path, Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_versioned, json_latest = write_versioned_json(latest_json_path, summary, timestamp=timestamp)
    csv_versioned, csv_latest = write_versioned_csv(
        latest_csv_path,
        rows,
        fieldnames=list(rows[0].keys()),
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    json_desc, csv_desc = manifest_descriptions
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", json_versioned.as_posix(), "implementation", json_desc),
            (manifest_ts, "/experiment-bridge", json_latest.as_posix(), "implementation", "latest copy"),
            (manifest_ts, "/experiment-bridge", csv_versioned.as_posix(), "implementation", csv_desc),
            (manifest_ts, "/experiment-bridge", csv_latest.as_posix(), "implementation", "latest copy"),
        ]
    )
    return json_versioned, json_latest, csv_versioned, csv_latest


def run_block1_experiment(
    *,
    model_name: str,
    model_slug: str,
    num_samples: int,
    seed: int,
    topologies: list[int],
    deltas: list[float],
    quant_mode: str,
    model_version: str,
    proxy_seed: int,
    input_samples: TensorList,
    create_shards_fn: ShardFactory,
    latest_json_path: str | Path,
    latest_csv_path: str | Path,
    data_source: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M1",
        "block": "Block 1",
        "model": model_name,
        "data_source": data_source,
        "quant_mode": quant_mode,
        "num_samples": num_samples,
        "deltas": deltas,
        "topologies": {},
    }

    attack_names = ["random_proxy", "single_machine_full_model"]
    for depth in topologies:
        route_id = f"route_m1_{model_slug}_{depth}"
        honest_shards = create_shards_fn(num_shards=depth, seed=seed)
        keys = [generate_key(seed=seed + idx) for idx in range(depth)]
        honest_nodes = create_node_chain(honest_shards, keys, route_id, quantization=quant_mode)
        registry = build_registry(honest_nodes)

        baseline_counts = {attack_name: {delta: 0 for delta in deltas} for attack_name in attack_names}
        method_counts = {attack_name: 0 for attack_name in attack_names}
        distances = {attack_name: [] for attack_name in attack_names}

        for sample_idx, x_input in enumerate(input_samples):
            session = create_session(route_id, model_version, x_input)
            honest_output = run_plain_inference(honest_nodes, x_input)
            attack_cases = run_proxy_attack_suite(
                x_input,
                session,
                honest_nodes,
                create_shards_fn,
                depth,
                keys,
                proxy_seed=proxy_seed + sample_idx,
                quantization=quant_mode,
            )
            for attack_case in attack_cases:
                delta_hits = evaluate_delta_scan(attack_case.output, honest_output, deltas)
                for delta, accepted in delta_hits.items():
                    baseline_counts[attack_case.name][delta] += int(accepted)
                verdict = verify_proof(attack_case.forged_bundle, registry, session, x_input)
                method_counts[attack_case.name] += int(verdict.accepted)
                distances[attack_case.name].append(float((attack_case.output - honest_output).norm(p=2).item()))

        topology_summary = {}
        for attack_name in attack_names:
            topology_summary[attack_name] = {
                "baseline_asr_by_delta": {
                    str(delta): baseline_counts[attack_name][delta] / num_samples for delta in deltas
                },
                "method_far": method_counts[attack_name] / num_samples,
                "distance_mean": statistics.mean(distances[attack_name]),
            }
            for delta in deltas:
                rows.append(
                    {
                        "model": model_slug,
                        "N": depth,
                        "attack_type": attack_name,
                        "delta": delta,
                        "baseline_asr": round(baseline_counts[attack_name][delta] / num_samples, 6),
                        "method_far": round(method_counts[attack_name] / num_samples, 6),
                        "distance_mean": round(statistics.mean(distances[attack_name]), 6),
                        "num_samples": num_samples,
                    }
                )
        summary["topologies"][str(depth)] = topology_summary

    write_experiment_outputs(
        latest_json_path,
        latest_csv_path,
        summary,
        rows,
        (
            f"M1 Block 1 {model_name} proxy attack results JSON",
            f"M1 Block 1 {model_name} proxy attack per-case CSV",
        ),
    )
    return summary


def run_block2_experiment(
    *,
    model_name: str,
    model_slug: str,
    num_samples: int,
    seed: int,
    topologies: list[int],
    quant_mode: str,
    model_version: str,
    input_samples: TensorList,
    create_shards_fn: ShardFactory,
    latest_json_path: str | Path,
    latest_csv_path: str | Path,
    data_source: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "milestone": "M1",
        "block": "Block 2",
        "model": model_name,
        "data_source": data_source,
        "quant_mode": quant_mode,
        "num_samples": num_samples,
        "topologies": {},
    }

    for depth in topologies:
        route_id = f"route_m1_topology_{model_slug}_{depth}"
        shards = create_shards_fn(num_shards=depth, seed=seed)
        keys = [generate_key(seed=seed + idx) for idx in range(depth)]
        honest_nodes = create_node_chain(shards, keys, route_id, quantization=quant_mode)
        registry = build_registry(honest_nodes)
        target_index = min(2, depth)

        attack_stats = {
            "skip_node": {"accepts": 0, "valid": 0},
            "swap_order": {"accepts": 0, "valid": 0},
            "replace_node": {"accepts": 0, "valid": 0},
            "inject_params": {"accepts": 0, "valid": 0},
        }

        for sample_idx, x_input in enumerate(input_samples):
            session = create_session(route_id, model_version, x_input)
            fake_shard = create_unauthorized_shard_like(
                honest_nodes[target_index - 1].model_shard,
                seed=seed + 1000 + sample_idx,
            )
            attack_results = [
                attack_skip_node(honest_nodes, x_input, session, skip_index=target_index),
                pick_swap_attack(honest_nodes, x_input, session),
                attack_replace_node(
                    honest_nodes,
                    x_input,
                    session,
                    replace_index=target_index,
                    fake_shard=fake_shard,
                ),
                attack_inject_params(
                    honest_nodes,
                    x_input,
                    session,
                    inject_index=target_index,
                    fake_shard=fake_shard,
                ),
            ]
            for attack_result in attack_results:
                if attack_result.executable and attack_result.bundle is not None:
                    attack_stats[attack_result.attack_name]["valid"] += 1
                    verdict = verify_proof(attack_result.bundle, registry, session, x_input)
                    attack_stats[attack_result.attack_name]["accepts"] += int(verdict.accepted)

        topology_summary = {}
        for attack_name, values in attack_stats.items():
            valid = values["valid"]
            far = values["accepts"] / valid if valid else 0.0
            topology_summary[attack_name] = {
                "far": far,
                "valid_samples": valid,
                "failed_samples": num_samples - valid,
            }
            rows.append(
                {
                    "model": model_slug,
                    "N": depth,
                    "attack_type": attack_name,
                    "method_far": round(far, 6),
                    "valid_samples": valid,
                    "failed_samples": num_samples - valid,
                    "num_samples": num_samples,
                }
            )
        summary["topologies"][str(depth)] = topology_summary

    write_experiment_outputs(
        latest_json_path,
        latest_csv_path,
        summary,
        rows,
        (
            f"M1 Block 2 {model_name} topology attack results JSON",
            f"M1 Block 2 {model_name} topology attack per-case CSV",
        ),
    )
    return summary

