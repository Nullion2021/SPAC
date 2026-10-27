"""Session helpers for Method C."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import struct
import time

import torch

from group_auth_dnn.protocol.crypto import sha256_hex
from group_auth_dnn.protocol.encoding import hash_encode_input


@dataclass(frozen=True)
class SessionContext:
    route_id: str
    model_version: str
    sid: str
    expires_at: float
    nonce_hex: str
    input_hash: str
    z0: str


def generate_session_id(
    nonce: bytes,
    expires_at: float,
    route_id: str,
    model_version: str,
    input_hash: str,
) -> str:
    header = (
        nonce
        + struct.pack("!d", float(expires_at))
        + route_id.encode("utf-8")
        + model_version.encode("utf-8")
        + bytes.fromhex(input_hash)
    )
    return sha256_hex(header)


def init_chain_state(sid: str, route_id: str, input_hash: str) -> str:
    return sha256_hex({"sid": sid, "route_id": route_id, "u1": input_hash})


def is_session_expired(expires_at: float, *, now: float | None = None) -> bool:
    current = time.time() if now is None else now
    return current >= expires_at


def create_session(
    route_id: str,
    model_version: str,
    x_task: torch.Tensor,
    *,
    lifetime: float = 3600.0,
    now: float | None = None,
    nonce: bytes | None = None,
) -> SessionContext:
    issued_at = time.time() if now is None else now
    expires_at = issued_at + lifetime
    nonce_value = secrets.token_bytes(16) if nonce is None else nonce
    input_hash = hash_encode_input(x_task)
    sid = generate_session_id(nonce_value, expires_at, route_id, model_version, input_hash)
    z0 = init_chain_state(sid, route_id, input_hash)
    return SessionContext(
        route_id=route_id,
        model_version=model_version,
        sid=sid,
        expires_at=expires_at,
        nonce_hex=nonce_value.hex(),
        input_hash=input_hash,
        z0=z0,
    )
