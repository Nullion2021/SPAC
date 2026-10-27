"""Baseline A: accept when the final output stays within a tolerance."""

from __future__ import annotations

import torch


def output_distance(y_out: torch.Tensor, y_ref: torch.Tensor, p: int = 2) -> float:
    """Return the output-space distance used by Baseline A."""
    return float((y_out - y_ref).norm(p=p).item())


def verify_baseline_a(
    y_out: torch.Tensor,
    y_ref: torch.Tensor,
    delta: float = 1e-3,
    p: int = 2,
) -> bool:
    return output_distance(y_out, y_ref, p=p) <= delta


def evaluate_delta_scan(
    y_out: torch.Tensor,
    y_ref: torch.Tensor,
    deltas: list[float],
    p: int = 2,
) -> dict[float, bool]:
    dist = output_distance(y_out, y_ref, p=p)
    return {delta: dist <= delta for delta in deltas}
