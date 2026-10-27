"""Baseline B: multi-device compute but only the terminal node emits HMAC_K(sid || y)."""

import hashlib
import hmac
import torch

from group_auth_dnn.protocol.encoding import encode_activation


def compute_end_mac(key: bytes, sid: str, y: torch.Tensor, enc_mode: str = "int8") -> str:
    """Terminal device computes HMAC_K(sid || E_a(y))."""
    y_bytes = encode_activation(y, mode=enc_mode)
    msg = sid.encode() + y_bytes
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_baseline_b(
    y_out: torch.Tensor,
    mac_out: str,
    y_ref: torch.Tensor,
    mac_ref: str,
    enc_mode: str = "int8",
) -> bool:
    """Accept when the encoded outputs match and the terminal MAC matches."""
    y_enc = encode_activation(y_out, mode=enc_mode)
    y_ref_enc = encode_activation(y_ref, mode=enc_mode)
    return y_enc == y_ref_enc and mac_out == mac_ref
