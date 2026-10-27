"""Method C proof execution and verification."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from group_auth_dnn.protocol.crypto import canonical_json_bytes, hmac_sha256_hex
from group_auth_dnn.protocol.encoding import QuantizationConfig, encode_tensor_fp32, hash_encode_activation, hash_encode_input, resolve_quant_config
from group_auth_dnn.protocol.node import AuthenticatedNode, HopProof, RegistryEntry, advance_chain_state, build_hop_message
from group_auth_dnn.protocol.session import SessionContext, is_session_expired


@dataclass(frozen=True)
class ProofBundle:
    session_id: str
    route_id: str
    model_version: str
    expires_at: float
    output: torch.Tensor
    chain_state: str
    proof_items: tuple[HopProof, ...]
    quantization: QuantizationConfig

    def metadata_dict(self) -> dict[str, object]:
        return {
            "sid": self.session_id,
            "route_id": self.route_id,
            "model_version": self.model_version,
            "expires_at": self.expires_at,
            "chain_state": self.chain_state,
            "proof_items": [item.to_dict() for item in self.proof_items],
        }


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: str


def run_authenticated_inference(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    session: SessionContext,
) -> ProofBundle:
    if not nodes:
        raise ValueError("nodes must not be empty")
    activation = x_input
    z_prev = session.z0
    proof_items: list[HopProof] = []
    quantization = nodes[0].quantization
    for hop_idx, node in enumerate(nodes):
        input_digest_override = session.input_hash if hop_idx == 0 else None
        activation, proof_item, z_prev = node.forward(
            activation,
            session,
            z_prev,
            input_digest_override=input_digest_override,
        )
        proof_items.append(proof_item)
    return ProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=activation.detach().cpu().clone(),
        chain_state=z_prev,
        proof_items=tuple(proof_items),
        quantization=quantization,
    )


def run_plain_inference(nodes: list[AuthenticatedNode], x_input: torch.Tensor) -> torch.Tensor:
    activation = x_input
    for node in nodes:
        with torch.no_grad():
            activation = node.model_shard(activation)
    return activation.detach().cpu().clone()


def verify_proof(
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
    z_prev = session.z0
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
            build_hop_message(
                sid=session.sid,
                route_id=session.route_id,
                index=record.index,
                device_id=record.device_id,
                measurement=record.measurement,
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


def proof_size_breakdown(bundle: ProofBundle) -> dict[str, int]:
    y_bytes = len(encode_tensor_fp32(bundle.output))
    z_bytes = len(bundle.chain_state.encode("ascii"))
    proof_items_bytes = len(canonical_json_bytes([item.to_dict() for item in bundle.proof_items]))
    return {
        "y_bytes": y_bytes,
        "z_bytes": z_bytes,
        "proof_items_bytes": proof_items_bytes,
        "total_bytes": y_bytes + z_bytes + proof_items_bytes,
    }


def serialized_proof_size(bundle: ProofBundle) -> int:
    return proof_size_breakdown(bundle)["total_bytes"]


def verify_chain(
    y_out: torch.Tensor,
    z_N: str,
    y_ref: torch.Tensor,
    z_N_ref: str,
    enc_mode: str = "int8",
    t_exp: float | None = None,
    now_ts: float | None = None,
) -> bool:
    if t_exp is not None:
        now = time.time() if now_ts is None else now_ts
        if now >= t_exp:
            return False
    return (
        hash_encode_activation(y_out, mode=resolve_quant_config(enc_mode))
        == hash_encode_activation(y_ref, mode=resolve_quant_config(enc_mode))
        and z_N == z_N_ref
    )
