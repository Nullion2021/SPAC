"""Model checkpoint save/load helpers."""

from __future__ import annotations

import os
from typing import Any

import torch

from group_auth_dnn.config import CHECKPOINT_PATHS, CHECKPOINTS_DIR


def _normalize_model_name(model_name: str) -> str:
    name = model_name.lower()
    if name not in CHECKPOINT_PATHS:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Expected one of {list(CHECKPOINT_PATHS.keys())}"
        )
    return name


def get_checkpoint_path(model_name: str) -> str:
    return CHECKPOINT_PATHS[_normalize_model_name(model_name)]


def save_checkpoint(
    model_name: str,
    model_state_dict: dict[str, torch.Tensor],
    epoch: int,
    best_acc: float,
    optimizer_state_dict: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | None = None,
) -> str:
    ckpt_path = path or get_checkpoint_path(model_name)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    payload = {
        "model_name": _normalize_model_name(model_name),
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "metadata": metadata or {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    model_name: str,
    path: str | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    ckpt_path = path or get_checkpoint_path(model_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
    if "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}: missing model_state_dict")
    return payload


def load_model_state_dict(
    model_name: str,
    path: str | None = None,
    map_location: str = "cpu",
) -> dict[str, torch.Tensor]:
    payload = load_checkpoint(model_name=model_name, path=path, map_location=map_location)
    return payload["model_state_dict"]


def checkpoint_exists(model_name: str, path: str | None = None) -> bool:
    ckpt_path = path or get_checkpoint_path(model_name)
    return os.path.exists(ckpt_path)
