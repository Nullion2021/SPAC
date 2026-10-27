"""M4 ablation: replace HMAC authentication with Ed25519 signatures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn as nn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from group_auth_dnn.protocol.chain import ProofBundle, VerificationResult
from group_auth_dnn.protocol.crypto import canonical_json_bytes
from group_auth_dnn.protocol.encoding import QuantizationConfig, hash_encode_activation, hash_encode_input, resolve_quant_config
from group_auth_dnn.protocol.measurement import compute_measurement
from group_auth_dnn.protocol.node import HopProof, advance_chain_state, build_hop_message
from group_auth_dnn.protocol.session import SessionContext, is_session_expired


def _seed_bytes(seed: int) -> bytes:
    return hashlib.sha256(f"signature-seed:{seed}".encode("utf-8")).digest()


def generate_ed25519_keypair(seed: int) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    private_key = Ed25519PrivateKey.from_private_bytes(_seed_bytes(seed))
    return private_key, private_key.public_key()


def sign_payload(private_key: Ed25519PrivateKey, payload: object) -> str:
    return private_key.sign(canonical_json_bytes(payload)).hex()


def verify_payload_signature(public_key: Ed25519PublicKey, payload: object, signature_hex: str) -> bool:
    try:
        public_key.verify(bytes.fromhex(signature_hex), canonical_json_bytes(payload))
        return True
    except InvalidSignature:
        return False


@dataclass(frozen=True)
class SignatureRegistryEntry:
    index: int
    device_id: str
    public_key: Ed25519PublicKey
    measurement: str


class SignatureNode:
    def __init__(
        self,
        *,
        index: int,
        device_id: str,
        model_shard: nn.Module,
        private_key: Ed25519PrivateKey,
        route_id: str,
        quantization: QuantizationConfig | str = "int8",
        code_hash: str | None = None,
    ) -> None:
        self.index = index
        self.device_id = device_id
        self.model_shard = model_shard.eval()
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.route_id = route_id
        self.quantization = resolve_quant_config(quantization)
        self.measurement = compute_measurement(
            route_id=route_id,
            index=index,
            model_shard=model_shard,
            quantization=self.quantization,
            code_hash=code_hash,
        )

    @property
    def registry_entry(self) -> SignatureRegistryEntry:
        return SignatureRegistryEntry(
            index=self.index,
            device_id=self.device_id,
            public_key=self.public_key,
            measurement=self.measurement,
        )

    def forward(
        self,
        activation_in: torch.Tensor,
        session: SessionContext,
        z_prev: str,
        *,
        input_digest_override: str | None = None,
    ) -> tuple[torch.Tensor, HopProof, str]:
        u_i = input_digest_override or hash_encode_activation(activation_in, mode=self.quantization)
        with torch.no_grad():
            activation_out = self.model_shard(activation_in)
        v_i = hash_encode_activation(activation_out, mode=self.quantization)
        message = build_hop_message(
            sid=session.sid,
            route_id=session.route_id,
            index=self.index,
            device_id=self.device_id,
            measurement=self.measurement,
            u=u_i,
            v=v_i,
            z_prev=z_prev,
        )
        tag = sign_payload(self.private_key, message)
        z_next = advance_chain_state(z_prev, tag)
        proof = HopProof(
            index=self.index,
            device_id=self.device_id,
            measurement=self.measurement,
            u=u_i,
            v=v_i,
            tag=tag,
        )
        return activation_out, proof, z_next


def create_signature_node_chain(
    shards: list[nn.Module],
    signing_seeds: list[int],
    route_id: str,
    *,
    quantization: QuantizationConfig | str = "int8",
    code_hashes: list[str] | None = None,
) -> list[SignatureNode]:
    if len(shards) != len(signing_seeds):
        raise ValueError("shards and signing_seeds must have the same length")
    if code_hashes is not None and len(code_hashes) != len(shards):
        raise ValueError("code_hashes and shards must have the same length")
    effective_hashes = code_hashes or [None] * len(shards)
    nodes: list[SignatureNode] = []
    for idx, (shard, seed, code_hash) in enumerate(zip(shards, signing_seeds, effective_hashes), start=1):
        private_key, _ = generate_ed25519_keypair(seed)
        nodes.append(
            SignatureNode(
                index=idx,
                device_id=f"device_{idx}",
                model_shard=shard,
                private_key=private_key,
                route_id=route_id,
                quantization=quantization,
                code_hash=code_hash,
            )
        )
    return nodes


def build_signature_registry(nodes: list[SignatureNode]) -> list[SignatureRegistryEntry]:
    return [node.registry_entry for node in nodes]


def run_signature_authenticated_inference(
    nodes: list[SignatureNode],
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


def verify_signature_proof(
    bundle: ProofBundle,
    registry: list[SignatureRegistryEntry],
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

        message = build_hop_message(
            sid=session.sid,
            route_id=session.route_id,
            index=record.index,
            device_id=record.device_id,
            measurement=record.measurement,
            u=proof_item.u,
            v=proof_item.v,
            z_prev=z_prev,
        )
        if not verify_payload_signature(record.public_key, message, proof_item.tag):
            return VerificationResult(False, f"signature mismatch at hop {record.index}")
        z_prev = advance_chain_state(z_prev, proof_item.tag)
        expected_u = proof_item.v

    if z_prev != bundle.chain_state:
        return VerificationResult(False, "terminal chain state mismatch")

    output_digest = hash_encode_activation(bundle.output, mode=bundle.quantization)
    if output_digest != bundle.proof_items[-1].v:
        return VerificationResult(False, "final output digest mismatch")

    return VerificationResult(True, "accept")
