"""Helpers for timestamped artifacts and manifest updates."""

from __future__ import annotations

from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Iterable


MANIFEST_HEADER = """# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
"""


def make_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d_%H%M%S")


def make_versioned_paths(latest_path: str | Path, timestamp: str | None = None) -> tuple[Path, Path, str]:
    latest = Path(latest_path)
    latest.parent.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or make_timestamp()
    versioned = latest.with_name(f"{latest.stem}_{stamp}{latest.suffix}")
    return versioned, latest, stamp


def write_versioned_text(
    latest_path: str | Path,
    content: str,
    *,
    timestamp: str | None = None,
    encoding: str = "utf-8",
) -> tuple[Path, Path]:
    versioned, latest, _ = make_versioned_paths(latest_path, timestamp)
    versioned.write_text(content, encoding=encoding)
    latest.write_text(content, encoding=encoding)
    return versioned, latest


def write_versioned_json(
    latest_path: str | Path,
    payload: object,
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return write_versioned_text(latest_path, text + "\n", timestamp=timestamp)


def write_versioned_csv(
    latest_path: str | Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    versioned, latest, _ = make_versioned_paths(latest_path, timestamp)
    row_list = list(rows)
    for path in (versioned, latest):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_list)
    return versioned, latest


def append_manifest(entries: list[tuple[str, str, str, str]], manifest_path: str | Path = "MANIFEST.md") -> None:
    manifest = Path(manifest_path)
    if not manifest.exists():
        manifest.write_text(MANIFEST_HEADER, encoding="utf-8")
    with manifest.open("a", encoding="utf-8") as handle:
        for timestamp, skill, file_path, stage, description in entries:
            handle.write(f"| {timestamp} | {skill} | {file_path} | {stage} | {description} |\n")
