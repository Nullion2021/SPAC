"""Deterministic encoding functions for inputs, activations, and parameters."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
import torch

from group_auth_dnn.protocol.crypto import canonical_json_bytes, sha256_hex


@dataclass(frozen=True)
class QuantizationConfig:
    """Unified activation encoding config used by Method C."""

    mode: str = "int8"
    num_bits: int = 8
    scale: float | None = None
    zero_point: int = 0
    granularity: str = "per_tensor"
    channel_axis: int = 1

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "mode": self.mode,
            "num_bits": self.num_bits,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "granularity": self.granularity,
            "channel_axis": self.channel_axis,
        }


def resolve_quant_config(
    quantization: QuantizationConfig | str = "int8",
    *,
    scale: float | None = None,
    zero_point: int = 0,
    granularity: str = "per_tensor",
    channel_axis: int = 1,
) -> QuantizationConfig:
    if isinstance(quantization, QuantizationConfig):
        return quantization
    if quantization == "int8":
        return QuantizationConfig(
            mode="int8",
            num_bits=8,
            scale=scale,
            zero_point=zero_point,
            granularity=granularity,
            channel_axis=channel_axis,
        )
    if quantization in {"int8_pc", "int8_per_channel"}:
        return QuantizationConfig(
            mode="int8",
            num_bits=8,
            scale=scale,
            zero_point=zero_point,
            granularity="per_channel",
            channel_axis=channel_axis,
        )
    if quantization == "fp16":
        return QuantizationConfig(
            mode="fp16",
            num_bits=16,
            scale=None,
            zero_point=0,
            granularity="per_tensor",
            channel_axis=channel_axis,
        )
    if quantization == "fp32":
        return QuantizationConfig(
            mode="fp32",
            num_bits=32,
            scale=None,
            zero_point=0,
            granularity="per_tensor",
            channel_axis=channel_axis,
        )
    if quantization == "mixed":
        return QuantizationConfig(
            mode="mixed",
            num_bits=16,
            scale=scale,
            zero_point=zero_point,
            granularity=granularity,
            channel_axis=channel_axis,
        )
    if quantization in {"mixed_pc", "mixed_per_channel"}:
        return QuantizationConfig(
            mode="mixed",
            num_bits=16,
            scale=scale,
            zero_point=zero_point,
            granularity="per_channel",
            channel_axis=channel_axis,
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")


def _normalize_channel_axis(axis: int, ndim: int) -> int:
    resolved = axis if axis >= 0 else ndim + axis
    if resolved < 0 or resolved >= ndim:
        raise ValueError(f"channel_axis={axis} is invalid for ndim={ndim}")
    return resolved


def _per_channel_scales(arr: np.ndarray, config: QuantizationConfig) -> tuple[np.ndarray, int, list[float]]:
    axis = _normalize_channel_axis(config.channel_axis, arr.ndim)
    reduce_axes = tuple(dim for dim in range(arr.ndim) if dim != axis)
    if reduce_axes:
        max_vals = np.max(np.abs(arr), axis=reduce_axes, keepdims=True)
    else:
        max_vals = np.abs(arr)
    scales = np.where(max_vals > 0, max_vals / 127.0, 1.0).astype(np.float32, copy=False)
    if reduce_axes:
        scale_values = np.squeeze(scales, axis=reduce_axes)
    else:
        scale_values = scales
    flattened = np.reshape(scale_values, -1)
    serialized_scales = [float(value) for value in flattened]
    return scales, axis, serialized_scales


def encode_tensor_fp32(tensor: torch.Tensor) -> bytes:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return arr.astype("<f4", copy=False).tobytes(order="C")


def encode_tensor_fp16(tensor: torch.Tensor) -> bytes:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float16)
    return arr.astype("<f2", copy=False).tobytes(order="C")


def encode_tensor_int8(tensor: torch.Tensor, config: QuantizationConfig) -> bytes:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    header_payload: dict[str, object] = {
        "mode": config.mode,
        "num_bits": config.num_bits,
        "zero_point": config.zero_point,
        "shape": list(arr.shape),
        "granularity": config.granularity,
    }
    if config.granularity == "per_channel" and arr.ndim >= 2:
        scales, axis, serialized_scales = _per_channel_scales(arr, config)
        quantized = np.clip(np.round(arr / scales + config.zero_point), -128, 127).astype(np.int8)
        header_payload["channel_axis"] = axis
        header_payload["scales"] = serialized_scales
    else:
        max_val = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = config.scale
        if scale is None:
            scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized = np.clip(np.round(arr / scale + config.zero_point), -128, 127).astype(np.int8)
        header_payload["scale"] = scale
    header = canonical_json_bytes(header_payload)
    return struct.pack("!I", len(header)) + header + quantized.tobytes(order="C")


def encode_tensor_mixed(tensor: torch.Tensor, config: QuantizationConfig) -> bytes:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    if arr.ndim >= 3:
        header_payload: dict[str, object] = {
            "mode": config.mode,
            "backend": "int8",
            "num_bits": 8,
            "zero_point": config.zero_point,
            "shape": list(arr.shape),
            "granularity": config.granularity,
        }
        if config.granularity == "per_channel":
            scales, axis, serialized_scales = _per_channel_scales(arr, config)
            quantized = np.clip(np.round(arr / scales + config.zero_point), -128, 127).astype(np.int8)
            header_payload["channel_axis"] = axis
            header_payload["scales"] = serialized_scales
        else:
            max_val = float(np.max(np.abs(arr))) if arr.size else 0.0
            scale = config.scale
            if scale is None:
                scale = max_val / 127.0 if max_val > 0 else 1.0
            quantized = np.clip(np.round(arr / scale + config.zero_point), -128, 127).astype(np.int8)
            header_payload["scale"] = scale
        header = canonical_json_bytes(header_payload)
        payload = quantized.tobytes(order="C")
    else:
        header = canonical_json_bytes(
            {
                "mode": config.mode,
                "backend": "fp16",
                "shape": list(arr.shape),
            }
        )
        payload = arr.astype("<f2", copy=False).tobytes(order="C")
    return struct.pack("!I", len(header)) + header + payload


def encode_activation(
    tensor: torch.Tensor,
    mode: QuantizationConfig | str = "int8",
    scale: float | None = None,
    zero_point: int = 0,
) -> bytes:
    config = resolve_quant_config(mode, scale=scale, zero_point=zero_point)
    if config.mode == "fp32":
        return encode_tensor_fp32(tensor)
    if config.mode == "fp16":
        return encode_tensor_fp16(tensor)
    if config.mode == "int8":
        return encode_tensor_int8(tensor, config)
    if config.mode == "mixed":
        return encode_tensor_mixed(tensor, config)
    raise ValueError(f"Unsupported activation encoding mode: {config.mode}")


def encode_input(tensor: torch.Tensor) -> bytes:
    return encode_tensor_fp32(tensor)


def encode_params(module: torch.nn.Module) -> bytes:
    parts: list[bytes] = []
    for name, tensor in sorted(module.state_dict().items()):
        metadata = canonical_json_bytes(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "float32",
            }
        )
        parts.append(struct.pack("!I", len(metadata)))
        parts.append(metadata)
        parts.append(encode_tensor_fp32(tensor))
    return b"".join(parts)


def hash_encode_activation(
    tensor: torch.Tensor,
    mode: QuantizationConfig | str = "int8",
    scale: float | None = None,
    zero_point: int = 0,
) -> str:
    return sha256_hex(encode_activation(tensor, mode=mode, scale=scale, zero_point=zero_point))


def hash_encode_input(tensor: torch.Tensor) -> str:
    return sha256_hex(encode_input(tensor))


def hash_encode_params(module: torch.nn.Module) -> str:
    return sha256_hex(encode_params(module))


def quant_config_fingerprint(quantization: QuantizationConfig | str = "int8") -> str:
    config = resolve_quant_config(quantization)
    return sha256_hex(config.as_dict())
