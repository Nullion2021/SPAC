"""Topology-tampering attacks for M1."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch

from group_auth_dnn.config import generate_key
from group_auth_dnn.protocol.chain import ProofBundle, run_authenticated_inference
from group_auth_dnn.protocol.node import AuthenticatedNode


@dataclass(frozen=True)
class TopologyAttackResult:
    attack_name: str
    bundle: ProofBundle | None
    executable: bool


def create_unauthorized_shard_like(shard, seed: int = 9999):
    fake_shard = copy.deepcopy(shard)
    try:
        first_parameter = next(fake_shard.parameters())
        rng = torch.Generator(device=first_parameter.device.type)
    except StopIteration:
        rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        for parameter in fake_shard.parameters():
            sampled = torch.randn(parameter.shape, generator=rng, dtype=parameter.dtype, device=parameter.device)
            parameter.copy_(sampled)
    return fake_shard


def _try_run_attack(nodes, x_input, session) -> ProofBundle | None:
    try:
        return run_authenticated_inference(nodes, x_input, session)
    except RuntimeError:
        return None


def attack_skip_node(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session,
    *,
    skip_index: int,
) -> TopologyAttackResult:
    modified_nodes = [node for node in nodes if node.index != skip_index]
    bundle = _try_run_attack(modified_nodes, x_input, session)
    if bundle is None:
        honest_bundle = run_authenticated_inference(nodes, x_input, session)
        proof_items = [item for item in honest_bundle.proof_items if item.index != skip_index]
        bundle = replace(honest_bundle, proof_items=tuple(proof_items))
    return TopologyAttackResult("skip_node", bundle=bundle, executable=bundle is not None)


def attack_swap_nodes(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session,
    *,
    idx1: int,
    idx2: int,
) -> TopologyAttackResult:
    modified_nodes = list(nodes)
    modified_nodes[idx1 - 1], modified_nodes[idx2 - 1] = modified_nodes[idx2 - 1], modified_nodes[idx1 - 1]
    bundle = _try_run_attack(modified_nodes, x_input, session)
    if bundle is None:
        honest_bundle = run_authenticated_inference(nodes, x_input, session)
        proof_items = list(honest_bundle.proof_items)
        proof_items[idx1 - 1], proof_items[idx2 - 1] = proof_items[idx2 - 1], proof_items[idx1 - 1]
        bundle = replace(honest_bundle, proof_items=tuple(proof_items))
    return TopologyAttackResult("swap_order", bundle=bundle, executable=bundle is not None)


def attack_replace_node(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session,
    *,
    replace_index: int,
    fake_shard,
) -> TopologyAttackResult:
    modified_nodes = list(nodes)
    original = modified_nodes[replace_index - 1]
    modified_nodes[replace_index - 1] = AuthenticatedNode(
        index=original.index,
        device_id="fake_device",
        model_shard=fake_shard,
        key=generate_key(seed=9999),
        route_id=original.route_id,
        quantization=original.quantization,
    )
    bundle = _try_run_attack(modified_nodes, x_input, session)
    return TopologyAttackResult("replace_node", bundle=bundle, executable=bundle is not None)


def attack_inject_params(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session,
    *,
    inject_index: int,
    fake_shard,
) -> TopologyAttackResult:
    modified_nodes = list(nodes)
    original = modified_nodes[inject_index - 1]
    modified_nodes[inject_index - 1] = AuthenticatedNode(
        index=original.index,
        device_id=original.device_id,
        model_shard=fake_shard,
        key=original.key,
        route_id=original.route_id,
        quantization=original.quantization,
    )
    bundle = _try_run_attack(modified_nodes, x_input, session)
    return TopologyAttackResult("inject_params", bundle=bundle, executable=bundle is not None)
