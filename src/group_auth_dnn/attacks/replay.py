"""Replay helpers for the M4 freshness ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from group_auth_dnn.protocol.chain import ProofBundle, run_authenticated_inference
from group_auth_dnn.protocol.node import AuthenticatedNode
from group_auth_dnn.protocol.session import SessionContext, create_session


@dataclass(frozen=True)
class ReplayAttackResult:
    original_session: SessionContext
    replay_session: SessionContext
    bundle: ProofBundle


def nonce_from_int(value: int) -> bytes:
    if value < 0:
        raise ValueError("nonce value must be non-negative")
    return value.to_bytes(16, byteorder="big", signed=False)


def prepare_cross_session_replay(
    nodes: list[AuthenticatedNode],
    x_input: torch.Tensor,
    route_id: str,
    model_version: str,
    *,
    issued_at: float,
    lifetime: float = 3600.0,
    original_nonce: bytes,
    replay_nonce: bytes,
) -> ReplayAttackResult:
    """Reuse an honest proof under a second request that differs only in nonce / sid."""
    if original_nonce == replay_nonce:
        raise ValueError("replay_nonce must differ from original_nonce")

    original_session = create_session(
        route_id,
        model_version,
        x_input,
        lifetime=lifetime,
        now=issued_at,
        nonce=original_nonce,
    )
    replay_session = create_session(
        route_id,
        model_version,
        x_input,
        lifetime=lifetime,
        now=issued_at,
        nonce=replay_nonce,
    )
    bundle = run_authenticated_inference(nodes, x_input, original_session)
    return ReplayAttackResult(
        original_session=original_session,
        replay_session=replay_session,
        bundle=bundle,
    )
