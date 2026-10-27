"""Proxy/model-substitution attacks for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from group_auth_dnn.protocol.chain import ProofBundle, run_authenticated_inference
from group_auth_dnn.protocol.node import AuthenticatedNode, create_node_chain
from group_auth_dnn.protocol.session import SessionContext


@dataclass(frozen=True)
class ProxyAttackCase:
    name: str
    output: torch.Tensor
    forged_bundle: ProofBundle


def create_proxy_shards(
    create_shards_fn: Callable[..., list[nn.Module]],
    num_shards: int,
    proxy_seed: int = 999,
) -> list[nn.Module]:
    return create_shards_fn(num_shards=num_shards, seed=proxy_seed)


def compose_full_model(shards: list[nn.Module]) -> nn.Sequential:
    """Assemble a single-machine model from honest shards."""
    modules: list[nn.Module] = []
    for shard in shards:
        if isinstance(shard, nn.Sequential):
            modules.extend(list(shard.children()))
        else:
            modules.append(shard)
    return nn.Sequential(*modules).eval()


def build_empty_forgery(
    session: SessionContext,
    output: torch.Tensor,
    quantization,
) -> ProofBundle:
    """Attack that can match the output but cannot provide a chain proof."""
    return ProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=output.detach().cpu().clone(),
        chain_state="0" * 64,
        proof_items=tuple(),
        quantization=quantization,
    )


def run_random_proxy_attack(
    x_input: torch.Tensor,
    session: SessionContext,
    route_id: str,
    create_shards_fn: Callable[..., list[nn.Module]],
    num_shards: int,
    keys: list[bytes],
    *,
    proxy_seed: int = 999,
    quantization: str = "int8",
) -> ProxyAttackCase:
    proxy_shards = create_proxy_shards(create_shards_fn, num_shards=num_shards, proxy_seed=proxy_seed)
    proxy_nodes = create_node_chain(proxy_shards, keys, route_id, quantization=quantization)
    output_bundle = run_authenticated_inference(proxy_nodes, x_input, session)
    return ProxyAttackCase(name="random_proxy", output=output_bundle.output, forged_bundle=output_bundle)


def run_single_machine_full_model_attack(
    x_input: torch.Tensor,
    session: SessionContext,
    honest_nodes: list[AuthenticatedNode],
) -> ProxyAttackCase:
    full_model = compose_full_model([node.model_shard for node in honest_nodes])
    with torch.no_grad():
        output = full_model(x_input).detach().cpu().clone()
    forged_bundle = build_empty_forgery(session, output, honest_nodes[0].quantization)
    return ProxyAttackCase(
        name="single_machine_full_model",
        output=output,
        forged_bundle=forged_bundle,
    )


def run_proxy_attack_suite(
    x_input: torch.Tensor,
    session: SessionContext,
    honest_nodes: list[AuthenticatedNode],
    create_shards_fn: Callable[..., list[nn.Module]],
    num_shards: int,
    keys: list[bytes],
    *,
    proxy_seed: int = 999,
    quantization: str = "int8",
) -> list[ProxyAttackCase]:
    return [
        run_random_proxy_attack(
            x_input,
            session,
            honest_nodes[0].route_id,
            create_shards_fn,
            num_shards,
            keys,
            proxy_seed=proxy_seed,
            quantization=quantization,
        ),
        run_single_machine_full_model_attack(x_input, session, honest_nodes),
    ]
