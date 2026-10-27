"""Baseline E: a CRA-style chain that omits activation binding and quantization-aware measurement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from group_auth_dnn.protocol.crypto import hmac_sha256_hex, sha256_hex
from group_auth_dnn.protocol.measurement import module_structure_hash
from group_auth_dnn.protocol.encoding import hash_encode_params
from group_auth_dnn.protocol.node import advance_chain_state
from group_auth_dnn.protocol.session import SessionContext, is_session_expired


def build_cra_hop_message(
    *,
    sid: str,
    route_id: str,
    index: int,
    device_id: str,
    measurement: str,
    z_prev: str,
) -> dict[str, str | int]:
    return {
        "sid": sid,
        "route_id": route_id,
        "index": index,
        "device_id": device_id,
        "measurement": measurement,
        "z_prev": z_prev,
    }


def init_cra_chain_state(sid: str, route_id: str) -> str:
    return sha256_hex({"sid": sid, "route_id": route_id})


def compute_cra_measurement(
    route_id: str,
    index: int,
    model_shard: nn.Module,
    code_hash: str | None = None,
) -> str:
    firmware_hash = sha256_hex(
        {
            "route_id": route_id,
            "index": index,
            "theta_hash": hash_encode_params(model_shard),
            "code_hash": code_hash or module_structure_hash(model_shard),
        }
    )
    return sha256_hex(firmware_hash)


@dataclass(frozen=True)
class CRARegistryEntry:
    index: int
    device_id: str
    key: bytes
    measurement: str


@dataclass(frozen=True)
class CRAHopProof:
    index: int
    device_id: str
    measurement: str
    tag: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "device_id": self.device_id,
            "measurement": self.measurement,
            "tag": self.tag,
        }


@dataclass(frozen=True)
class CRAProofBundle:
    session_id: str
    route_id: str
    model_version: str
    expires_at: float
    output: torch.Tensor
    chain_state: str
    proof_items: tuple[CRAHopProof, ...]


@dataclass(frozen=True)
class CRAVerificationResult:
    accepted: bool
    reason: str


class CRABaselineNode:
    """A CRA baseline node that authenticates only device identity and measurement."""

    def __init__(
        self,
        *,
        index: int,
        device_id: str,
        model_shard: nn.Module,
        key: bytes,
        route_id: str,
        code_hash: str | None = None,
    ) -> None:
        self.index = index
        self.device_id = device_id
        self.model_shard = model_shard.eval()
        self.key = key
        self.route_id = route_id
        self.measurement = compute_cra_measurement(
            route_id=route_id,
            index=index,
            model_shard=model_shard,
            code_hash=code_hash,
        )

    @property
    def registry_entry(self) -> CRARegistryEntry:
        return CRARegistryEntry(
            index=self.index,
            device_id=self.device_id,
            key=self.key,
            measurement=self.measurement,
        )

    def forward(
        self,
        activation_in: torch.Tensor,
        session: SessionContext,
        z_prev: str,
    ) -> tuple[torch.Tensor, CRAHopProof, str]:
        with torch.no_grad():
            activation_out = self.model_shard(activation_in)
        message = build_cra_hop_message(
            sid=session.sid,
            route_id=session.route_id,
            index=self.index,
            device_id=self.device_id,
            measurement=self.measurement,
            z_prev=z_prev,
        )
        tag = hmac_sha256_hex(self.key, message)
        z_next = advance_chain_state(z_prev, tag)
        proof = CRAHopProof(
            index=self.index,
            device_id=self.device_id,
            measurement=self.measurement,
            tag=tag,
        )
        return activation_out, proof, z_next


def create_cra_node_chain(
    shards: list[nn.Module],
    keys: list[bytes],
    route_id: str,
    *,
    code_hashes: list[str] | None = None,
) -> list[CRABaselineNode]:
    if len(shards) != len(keys):
        raise ValueError("shards and keys must have the same length")
    if code_hashes is not None and len(code_hashes) != len(shards):
        raise ValueError("code_hashes and shards must have the same length")
    effective_hashes = code_hashes or [None] * len(shards)
    nodes = []
    for idx, (shard, key, code_hash) in enumerate(zip(shards, keys, effective_hashes), start=1):
        nodes.append(
            CRABaselineNode(
                index=idx,
                device_id=f"device_{idx}",
                model_shard=shard,
                key=key,
                route_id=route_id,
                code_hash=code_hash,
            )
        )
    return nodes


def build_cra_registry(nodes: list[CRABaselineNode]) -> list[CRARegistryEntry]:
    return [node.registry_entry for node in nodes]


def run_cra_inference(
    nodes: list[CRABaselineNode],
    x_input: torch.Tensor,
    session: SessionContext,
) -> CRAProofBundle:
    if not nodes:
        raise ValueError("nodes must not be empty")
    activation = x_input
    z_prev = init_cra_chain_state(session.sid, session.route_id)
    proof_items: list[CRAHopProof] = []
    for node in nodes:
        activation, proof_item, z_prev = node.forward(activation, session, z_prev)
        proof_items.append(proof_item)
    return CRAProofBundle(
        session_id=session.sid,
        route_id=session.route_id,
        model_version=session.model_version,
        expires_at=session.expires_at,
        output=activation.detach().cpu().clone(),
        chain_state=z_prev,
        proof_items=tuple(proof_items),
    )


def verify_cra_proof(
    bundle: CRAProofBundle,
    registry: list[CRARegistryEntry],
    session: SessionContext,
    *,
    now: float | None = None,
) -> CRAVerificationResult:
    if bundle.session_id != session.sid:
        return CRAVerificationResult(False, "sid mismatch")
    if bundle.route_id != session.route_id:
        return CRAVerificationResult(False, "route mismatch")
    if is_session_expired(session.expires_at, now=now):
        return CRAVerificationResult(False, "session expired")
    if len(bundle.proof_items) != len(registry):
        return CRAVerificationResult(False, "proof length mismatch")

    z_prev = init_cra_chain_state(session.sid, session.route_id)
    for proof_item, record in zip(bundle.proof_items, registry):
        if proof_item.index != record.index:
            return CRAVerificationResult(False, f"unexpected hop index at {record.index}")
        if proof_item.device_id != record.device_id:
            return CRAVerificationResult(False, f"device mismatch at hop {record.index}")
        if proof_item.measurement != record.measurement:
            return CRAVerificationResult(False, f"measurement mismatch at hop {record.index}")

        expected_tag = hmac_sha256_hex(
            record.key,
            build_cra_hop_message(
                sid=session.sid,
                route_id=session.route_id,
                index=record.index,
                device_id=record.device_id,
                measurement=record.measurement,
                z_prev=z_prev,
            ),
        )
        if proof_item.tag != expected_tag:
            return CRAVerificationResult(False, f"hmac mismatch at hop {record.index}")
        z_prev = advance_chain_state(z_prev, proof_item.tag)

    if z_prev != bundle.chain_state:
        return CRAVerificationResult(False, "terminal chain state mismatch")

    return CRAVerificationResult(True, "accept")
