"""M4 ablation: keep Method C structure but disable measurement-binding enforcement."""

from __future__ import annotations

import torch

from group_auth_dnn.protocol.chain import ProofBundle, VerificationResult
from group_auth_dnn.protocol.crypto import hmac_sha256_hex
from group_auth_dnn.protocol.encoding import hash_encode_activation, hash_encode_input
from group_auth_dnn.protocol.node import RegistryEntry, advance_chain_state, build_hop_message
from group_auth_dnn.protocol.session import SessionContext, is_session_expired


def verify_proof_wo_measurement(
    bundle: ProofBundle,
    registry: list[RegistryEntry],
    session: SessionContext,
    x_input: torch.Tensor | None = None,
    *,
    now: float | None = None,
) -> VerificationResult:
    """Verify Method C proofs without checking mu_i against the registry."""
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
    z_prev = session.z0
    for proof_item, record in zip(bundle.proof_items, registry):
        if proof_item.index != record.index:
            return VerificationResult(False, f"unexpected hop index at {record.index}")
        if proof_item.device_id != record.device_id:
            return VerificationResult(False, f"device mismatch at hop {record.index}")
        if proof_item.u != expected_u:
            return VerificationResult(False, f"activation chain mismatch at hop {record.index}")

        expected_tag = hmac_sha256_hex(
            record.key,
            build_hop_message(
                sid=session.sid,
                route_id=session.route_id,
                index=record.index,
                device_id=record.device_id,
                measurement=proof_item.measurement,
                u=proof_item.u,
                v=proof_item.v,
                z_prev=z_prev,
            ),
        )
        if proof_item.tag != expected_tag:
            return VerificationResult(False, f"hmac mismatch at hop {record.index}")
        z_prev = advance_chain_state(z_prev, proof_item.tag)
        expected_u = proof_item.v

    if z_prev != bundle.chain_state:
        return VerificationResult(False, "terminal chain state mismatch")

    output_digest = hash_encode_activation(bundle.output, mode=bundle.quantization)
    if output_digest != bundle.proof_items[-1].v:
        return VerificationResult(False, "final output digest mismatch")

    return VerificationResult(True, "accept")
