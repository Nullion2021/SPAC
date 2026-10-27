"""Activation-injection attacks used by M3 / Block 5."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from group_auth_dnn.baselines.baseline_e import CRAProofBundle, CRABaselineNode, init_cra_chain_state
from group_auth_dnn.protocol.chain import ProofBundle
from group_auth_dnn.protocol.node import AuthenticatedNode
from group_auth_dnn.protocol.session import SessionContext


@dataclass(frozen=True)
class ActivationInjectionAttackResult:
    target_hop: int
    strength: float
    injected_activation: torch.Tensor
    method_bundle: ProofBundle
    baseline_bundle: CRAProofBundle
    injection_l2: float
    output_shift_l2: float
    alternate_gap_l2: float


def _collect_activations(modules: list[torch.nn.Module], x_input: torch.Tensor) -> list[torch.Tensor]:
    activations = [x_input.detach().clone()]
    activation = x_input
    with torch.no_grad():
        for module in modules:
            activation = module(activation)
            activations.append(activation.detach().clone())
    return activations


def build_injected_activation(
    legit_activation: torch.Tensor,
    alternate_activation: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")
    return legit_activation + strength * (alternate_activation - legit_activation)


def _execute_method_attack(
    nodes: list[AuthenticatedNode],
    session: SessionContext,
    x_input: torch.Tensor,
    injected_activation: torch.Tensor,
    *,
    target_hop: int,
) -> ProofBundle:
    activation = injected_activation if target_hop == 1 else x_input
    z_prev = session.z0
    proof_items = []
    for hop_idx, node in enumerate(nodes, start=1):
        input_digest_override = session.input_hash if hop_idx == 1 else None
        activation, proof_item, z_prev = node.forward(
            activation,
            session,
            z_prev,
            input_digest_override=input_digest_override,
        )
        proof_items.append(proof_item)
        if hop_idx == target_hop - 1:
            activation = injected_activation
    return ProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=activation.detach().cpu().clone(),
        chain_state=z_prev,
        proof_items=tuple(proof_items),
        quantization=nodes[0].quantization,
    )


def _execute_baseline_attack(
    nodes: list[CRABaselineNode],
    session: SessionContext,
    x_input: torch.Tensor,
    injected_activation: torch.Tensor,
    *,
    target_hop: int,
) -> CRAProofBundle:
    activation = injected_activation if target_hop == 1 else x_input
    z_prev = init_cra_chain_state(session.sid, session.route_id)
    proof_items = []
    for hop_idx, node in enumerate(nodes, start=1):
        activation, proof_item, z_prev = node.forward(activation, session, z_prev)
        proof_items.append(proof_item)
        if hop_idx == target_hop - 1:
            activation = injected_activation
    return CRAProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=activation.detach().cpu().clone(),
        chain_state=z_prev,
        proof_items=tuple(proof_items),
    )


def run_activation_injection_attack(
    method_nodes: list[AuthenticatedNode],
    baseline_nodes: list[CRABaselineNode],
    x_input: torch.Tensor,
    alternate_input: torch.Tensor,
    session: SessionContext,
    *,
    target_hop: int,
    strength: float,
) -> ActivationInjectionAttackResult:
    if len(method_nodes) != len(baseline_nodes):
        raise ValueError("method_nodes and baseline_nodes must have the same length")
    if not 1 <= target_hop <= len(method_nodes):
        raise ValueError("target_hop out of range")

    modules = [node.model_shard for node in method_nodes]
    legit_acts = _collect_activations(modules, x_input)
    alternate_acts = _collect_activations(modules, alternate_input)
    legit_input = legit_acts[target_hop - 1]
    alternate_input_at_hop = alternate_acts[target_hop - 1]
    injected_activation = build_injected_activation(legit_input, alternate_input_at_hop, strength)

    method_bundle = _execute_method_attack(
        method_nodes,
        session,
        x_input,
        injected_activation,
        target_hop=target_hop,
    )
    baseline_bundle = _execute_baseline_attack(
        baseline_nodes,
        session,
        x_input,
        injected_activation,
        target_hop=target_hop,
    )

    honest_output = legit_acts[-1].detach().cpu()
    alternate_output = alternate_acts[-1].detach().cpu()
    return ActivationInjectionAttackResult(
        target_hop=target_hop,
        strength=strength,
        injected_activation=injected_activation.detach().cpu().clone(),
        method_bundle=method_bundle,
        baseline_bundle=baseline_bundle,
        injection_l2=float((injected_activation - legit_input).norm(p=2).item()),
        output_shift_l2=float((method_bundle.output - honest_output).norm(p=2).item()),
        alternate_gap_l2=float((alternate_output - honest_output).norm(p=2).item()),
    )
