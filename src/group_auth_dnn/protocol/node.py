"""Authenticated nodes and proof items for Method C."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from group_auth_dnn.protocol.crypto import hmac_sha256_hex, sha256_hex
from group_auth_dnn.protocol.encoding import QuantizationConfig, hash_encode_activation, resolve_quant_config
from group_auth_dnn.protocol.measurement import compute_measurement
from group_auth_dnn.protocol.session import SessionContext


def build_hop_message(
    *,
    sid: str,
    route_id: str,
    index: int,
    device_id: str,
    measurement: str,
    u: str,
    v: str,
    z_prev: str,
) -> dict[str, str | int]:
    return {
        "sid": sid,
        "route_id": route_id,
        "index": index,
        "device_id": device_id,
        "measurement": measurement,
        "u": u,
        "v": v,
        "z_prev": z_prev,
    }


def advance_chain_state(z_prev: str, tag: str) -> str:
    return sha256_hex({"z_prev": z_prev, "tag": tag})


@dataclass(frozen=True)
class RegistryEntry:
    index: int
    device_id: str
    key: bytes
    measurement: str


@dataclass(frozen=True)
class HopProof:
    index: int
    device_id: str
    measurement: str
    u: str
    v: str
    tag: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "device_id": self.device_id,
            "measurement": self.measurement,
            "u": self.u,
            "v": self.v,
            "tag": self.tag,
        }


class AuthenticatedNode:
    """A model shard with registered metadata and an HMAC key."""

    def __init__(
        self,
        *,
        index: int,
        device_id: str,
        model_shard: nn.Module,
        key: bytes,
        route_id: str,
        quantization: QuantizationConfig | str = "int8",
        code_hash: str | None = None,
    ) -> None:
        self.index = index
        self.device_id = device_id
        self.model_shard = model_shard.eval()
        self.key = key
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
    def registry_entry(self) -> RegistryEntry:
        return RegistryEntry(
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
        tag = hmac_sha256_hex(self.key, message)
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


def create_node_chain(
    shards: list[nn.Module],
    keys: list[bytes],
    route_id: str,
    *,
    quantization: QuantizationConfig | str = "int8",
    code_hashes: list[str] | None = None,
) -> list[AuthenticatedNode]:
    if len(shards) != len(keys):
        raise ValueError("shards and keys must have the same length")
    if code_hashes is not None and len(code_hashes) != len(shards):
        raise ValueError("code_hashes and shards must have the same length")
    effective_hashes = code_hashes or [None] * len(shards)
    nodes = []
    for idx, (shard, key, code_hash) in enumerate(zip(shards, keys, effective_hashes), start=1):
        nodes.append(
            AuthenticatedNode(
                index=idx,
                device_id=f"device_{idx}",
                model_shard=shard,
                key=key,
                route_id=route_id,
                quantization=quantization,
                code_hash=code_hash,
            )
        )
    return nodes


def build_registry(nodes: list[AuthenticatedNode]) -> list[RegistryEntry]:
    return [node.registry_entry for node in nodes]
