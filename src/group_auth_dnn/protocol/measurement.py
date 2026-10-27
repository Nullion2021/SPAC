"""Measurement helpers for registered model shards."""

from __future__ import annotations

import torch.nn as nn

from group_auth_dnn.protocol.crypto import sha256_hex
from group_auth_dnn.protocol.encoding import QuantizationConfig, hash_encode_params, quant_config_fingerprint, resolve_quant_config


def module_structure_hash(module: nn.Module) -> str:
    structure = []
    for name, child in module.named_modules():
        structure.append({"name": name or "__root__", "type": child.__class__.__name__})
    return sha256_hex(structure)


def compute_measurement(
    route_id: str,
    index: int,
    model_shard: nn.Module,
    quantization: QuantizationConfig | str = "int8",
    code_hash: str | None = None,
) -> str:
    """Bind route, shard parameters, code identity, and quantization config."""
    config = resolve_quant_config(quantization)
    payload = {
        "route_id": route_id,
        "index": index,
        "theta_hash": hash_encode_params(model_shard),
        "code_hash": code_hash or module_structure_hash(model_shard),
        "quant_config": quant_config_fingerprint(config),
    }
    return sha256_hex(payload)
