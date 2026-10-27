"""Deterministic cryptographic helpers for Method C."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize payload with stable key order for cross-host reproducibility."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_hex(payload: bytes | str | Any) -> str:
    """Hash raw bytes, text, or a JSON-serializable object to a hex digest."""
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = canonical_json_bytes(payload)
    return hashlib.sha256(data).hexdigest()


def hmac_sha256_hex(key: bytes, payload: bytes | str | Any) -> str:
    """Compute an HMAC-SHA256 hex digest over a deterministic payload."""
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = canonical_json_bytes(payload)
    return hmac.new(key, data, hashlib.sha256).hexdigest()
