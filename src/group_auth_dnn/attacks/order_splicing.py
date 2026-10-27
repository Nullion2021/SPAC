"""Transcript-splicing helpers for the M4 chain-state ablation."""

from __future__ import annotations

from dataclasses import replace

from group_auth_dnn.protocol.chain import ProofBundle


def swap_execution_order(nodes: list[object], idx1: int, idx2: int) -> list[object]:
    reordered = list(nodes)
    reordered[idx1 - 1], reordered[idx2 - 1] = reordered[idx2 - 1], reordered[idx1 - 1]
    return reordered


def skip_then_backfill_order(nodes: list[object], skip_index: int) -> list[object]:
    skipped = None
    reordered: list[object] = []
    for node in nodes:
        if getattr(node, "index", None) == skip_index:
            skipped = node
        else:
            reordered.append(node)
    if skipped is None:
        raise ValueError(f"skip_index {skip_index} not found")
    reordered.append(skipped)
    return reordered


def forge_logical_order_transcript(bundle: ProofBundle) -> ProofBundle:
    return replace(
        bundle,
        proof_items=tuple(sorted(bundle.proof_items, key=lambda item: item.index)),
    )
