"""M4 ablation: remove recursive chain-state binding from Method C."""

from __future__ import annotations

import torch

from group_auth_dnn.protocol.chain import ProofBundle, VerificationResult
from group_auth_dnn.protocol.crypto import hmac_sha256_hex
from group_auth_dnn.protocol.encoding import hash_encode_activation, hash_encode_input
from group_auth_dnn.protocol.node import AuthenticatedNode, HopProof, RegistryEntry
from group_auth_dnn.protocol.session import SessionContext, is_session_expired


def build_hop_message_wo_chain_state(
    *,
    sid: str,
    route_id: str,
    index: int,
    device_id: str,
    measurement: str,
    u: str,
    v: str,
) -> dict[str, str | int]:
    return {
        "sid": sid,
        "route_id": route_id,
        "index": index,
        "device_id": device_id,
        "measurement": measurement,
        "u": u,
        "v": v,
    }


def run_inference_wo_chain_state(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session: SessionContext,
) -> ProofBundle:
    if not nodes:
        raise ValueError("nodes must not be empty")
    activation = x_input
    proof_items: list[HopProof] = []
    quantization = nodes[0].quantization

    for execution_pos, node in enumerate(nodes):
        u_i = session.input_hash if execution_pos == 0 else hash_encode_activation(activation, mode=quantization)
        with torch.no_grad():
            activation_out = node.model_shard(activation)
        v_i = hash_encode_activation(activation_out, mode=quantization)
        tag = hmac_sha256_hex(
            node.key,
            build_hop_message_wo_chain_state(
                sid=session.sid,
                route_id=session.route_id,
                index=node.index,
                device_id=node.device_id,
                measurement=node.measurement,
                u=u_i,
                v=v_i,
            ),
        )
        proof_items.append(
            HopProof(
                index=node.index,
                device_id=node.device_id,
                measurement=node.measurement,
                u=u_i,
                v=v_i,
                tag=tag,
            )
        )
        activation = activation_out

    return ProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=activation.detach().cpu().clone(),
        chain_state="no_chain_state",
        proof_items=tuple(proof_items),
        quantization=quantization,
    )


def verify_proof_wo_chain_state(
    bundle: ProofBundle,
    registry: list[RegistryEntry],
    session: SessionContext,
    x_input: torch.Tensor | None = None,
    *,
    now: float | None = None,
) -> VerificationResult:
    if bundle.session_id != session.sid:
        return VerificationResult(False, "sid mismatch")
    if bundle.route_id != session.route_id:
        return VerificationResult(False, "route mismatch")
    if is_session_expired(session.expires_at, now=now):
        return VerificationResult(False, "session expired")
    if x_input is not None and hash_encode_input(x_input) != session.input_hash:
        return VerificationResult(False, "input hash mismatch")
    if len(bundle.proof_items) != len(registry):
        return VerificationResult(False, "proof length mismatch")

    expected_u = session.input_hash
    for proof_item, record in zip(bundle.proof_items, registry):
        if proof_item.index != record.index:
            return VerificationResult(False, f"unexpected hop index at {record.index}")
        if proof_item.device_id != record.device_id:
            return VerificationResult(False, f"device mismatch at hop {record.index}")
        if proof_item.measurement != record.measurement:
            return VerificationResult(False, f"measurement mismatch at hop {record.index}")
        if proof_item.u != expected_u:
            return VerificationResult(False, f"activation chain mismatch at hop {record.index}")

        expected_tag = hmac_sha256_hex(
            record.key,
            build_hop_message_wo_chain_state(
                sid=session.sid,
                route_id=session.route_id,
                index=record.index,
                device_id=record.device_id,
                measurement=record.measurement,
                u=proof_item.u,
                v=proof_item.v,
            ),
        )
        if proof_item.tag != expected_tag:
            return VerificationResult(False, f"hmac mismatch at hop {record.index}")
        expected_u = proof_item.v

    output_digest = hash_encode_activation(bundle.output, mode=bundle.quantization)
    if output_digest != bundle.proof_items[-1].v:
        return VerificationResult(False, "final output digest mismatch")

    return VerificationResult(True, "accept")
