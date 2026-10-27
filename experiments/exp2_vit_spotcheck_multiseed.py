"""Targeted multi-seed runner and aggregator for Block 2 ViT/CIFAR-10 spot-checks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys


import numpy as np

from group_auth_dnn.config import (
    RESNET_INPUT_CHANNELS,
    RESULTS_DIR,
    VIT_DIM,
    VIT_HEADS,
    VIT_IMAGE_SIZE,
    VIT_LAYERS,
    VIT_MLP_DIM,
    VIT_NUM_CLASSES,
    VIT_PATCH_SIZE,
)
from group_auth_dnn.models.vit import create_vit_shards
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json

if __package__ in (None, ""):
    from m1_runners import build_cifar10_inputs, run_block2_experiment
else:
    from .m1_runners import build_cifar10_inputs, run_block2_experiment


DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_run_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--num-samples", type=int, default=64)
        target.add_argument("--artifact-prefix", type=str, default="exp2_vit_spotcheck_multiseed")
        target.add_argument("--topologies", type=int, nargs="+", default=[5])
        target.add_argument("--quant-mode", type=str, default="int8")
        target.add_argument("--model-version", type=str, default="m1-vit-v1")
        target.add_argument("--data-root", type=str, default="data/cifar10")

    run_seed = subparsers.add_parser("run-seed", help="run Block 2 ViT spot-check for one seed")
    add_shared_run_args(run_seed)
    run_seed.add_argument("--seed", type=int, required=True)

    aggregate = subparsers.add_parser("aggregate", help="aggregate seed-specific ViT spot-check snapshots")
    aggregate.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    aggregate.add_argument("--artifact-prefix", type=str, default="exp2_vit_spotcheck_multiseed")
    aggregate.add_argument("--summary-prefix", type=str, default="multiseed_exp2_vit")

    run_all = subparsers.add_parser("run-all", help="run ViT spot-check across multiple seeds then aggregate")
    run_all.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    add_shared_run_args(run_all)
    run_all.add_argument("--summary-prefix", type=str, default="multiseed_exp2_vit")

    return parser.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(rows: list[dict[str, object]], *, group_keys: list[str], metric_keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for group_key, items in groups.items():
        out_row = {key: value for key, value in zip(group_keys, group_key)}
        n = len(items)
        out_row["n_seeds"] = n
        for metric in metric_keys:
            values = np.array([float(item[metric]) for item in items], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
            out_row[f"{metric}_mean"] = round(mean, 6)
            out_row[f"{metric}_std"] = round(std, 6)
            out_row[f"{metric}_ci95"] = round(ci95, 6)
        summary_rows.append(out_row)

    summary_rows.sort(key=lambda row: tuple(row[key] for key in group_keys))
    return summary_rows


def _rows_from_latest_csv(path: str | Path, *, run_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(path):
        rows.append(
            {
                "run_id": run_id,
                "model": row["model"],
                "N": int(row["N"]),
                "attack_type": row["attack_type"],
                "FAR_Method_C": float(row["method_far"]),
                "valid_samples": float(row["valid_samples"]),
                "failed_samples": float(row["failed_samples"]),
                "num_samples": float(row["num_samples"]),
            }
        )
    return rows


def _materialize_seed_snapshot(*, seed: int, artifact_prefix: str) -> tuple[Path, Path]:
    vit_rows = _rows_from_latest_csv(Path(RESULTS_DIR) / "m1_block2_vit.csv", run_id="R009V")
    bundle_payload = {
        "seed": seed,
        "artifact_prefix": artifact_prefix,
        "rows": vit_rows,
        "vit_summary": _read_json(Path(RESULTS_DIR) / "m1_block2_vit.json"),
        "source_latest_files": [
            (Path(RESULTS_DIR) / "m1_block2_vit.json").as_posix(),
            (Path(RESULTS_DIR) / "m1_block2_vit.csv").as_posix(),
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_versioned, rows_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_rows.csv",
        vit_rows,
        fieldnames=list(vit_rows[0].keys()),
        timestamp=timestamp,
    )
    bundle_versioned, bundle_latest = write_versioned_json(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_bundle.json",
        bundle_payload,
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", rows_versioned.as_posix(), "results", f"exp2 ViT seed {seed} rows"),
            (manifest_ts, "/experiment-bridge", rows_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", bundle_versioned.as_posix(), "results", f"exp2 ViT seed {seed} bundle"),
            (manifest_ts, "/experiment-bridge", bundle_latest.as_posix(), "results", "latest copy"),
        ]
    )
    return rows_latest, bundle_latest


def run_one_seed(
    *,
    seed: int,
    artifact_prefix: str,
    num_samples: int,
    topologies: list[int],
    quant_mode: str,
    model_version: str,
    data_root: str,
) -> tuple[Path, Path]:
    print(f"[exp2-vit-multiseed] running seed={seed} num_samples={num_samples}")
    vit_inputs = build_cifar10_inputs(num_samples, seed, root=data_root)
    run_block2_experiment(
        model_name="ViT",
        model_slug="vit",
        num_samples=num_samples,
        seed=seed,
        topologies=topologies,
        quant_mode=quant_mode,
        model_version=model_version,
        input_samples=vit_inputs,
        create_shards_fn=lambda *, num_shards, seed: create_vit_shards(
            VIT_IMAGE_SIZE,
            VIT_PATCH_SIZE,
            RESNET_INPUT_CHANNELS,
            VIT_DIM,
            VIT_HEADS,
            VIT_LAYERS,
            VIT_MLP_DIM,
            VIT_NUM_CLASSES,
            num_shards,
            seed=seed,
        ),
        latest_json_path=Path(RESULTS_DIR) / "m1_block2_vit.json",
        latest_csv_path=Path(RESULTS_DIR) / "m1_block2_vit.csv",
        data_source="CIFAR-10 test subset",
    )
    return _materialize_seed_snapshot(seed=seed, artifact_prefix=artifact_prefix)


def aggregate_seed_snapshots(*, seeds: list[int], artifact_prefix: str, summary_prefix: str) -> tuple[Path, Path, Path]:
    raw_rows: list[dict[str, object]] = []
    source_files: list[str] = []

    for seed in seeds:
        bundle_path = Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_bundle.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing seed snapshot: {bundle_path.as_posix()}")
        payload = _read_json(bundle_path)
        source_files.append(bundle_path.as_posix())
        for row in payload["rows"]:
            raw_rows.append({"seed": seed, **row})

    summary_rows = _summarize(
        raw_rows,
        group_keys=["run_id", "model", "N", "attack_type"],
        metric_keys=["FAR_Method_C", "valid_samples", "failed_samples", "num_samples"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_versioned, raw_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{summary_prefix}_raw.csv",
        raw_rows,
        fieldnames=list(raw_rows[0].keys()),
        timestamp=timestamp,
    )
    summary_versioned, summary_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{summary_prefix}_summary.csv",
        summary_rows,
        fieldnames=list(summary_rows[0].keys()),
        timestamp=timestamp,
    )
    bundle_versioned, bundle_latest = write_versioned_json(
        Path(RESULTS_DIR) / f"{summary_prefix}_bundle.json",
        {
            "seeds": seeds,
            "artifact_prefix": artifact_prefix,
            "source_files": source_files,
            "raw_rows": raw_rows,
            "summary_rows": summary_rows,
        },
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", raw_versioned.as_posix(), "results", "exp2 ViT multiseed raw table"),
            (manifest_ts, "/experiment-bridge", raw_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", summary_versioned.as_posix(), "results", "exp2 ViT multiseed summary"),
            (manifest_ts, "/experiment-bridge", summary_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", bundle_versioned.as_posix(), "results", "exp2 ViT multiseed bundle"),
            (manifest_ts, "/experiment-bridge", bundle_latest.as_posix(), "results", "latest copy"),
        ]
    )
    return raw_latest, summary_latest, bundle_latest


def main() -> None:
    args = _parse_args()

    if args.command == "run-seed":
        paths = run_one_seed(
            seed=args.seed,
            artifact_prefix=args.artifact_prefix,
            num_samples=args.num_samples,
            topologies=args.topologies,
            quant_mode=args.quant_mode,
            model_version=args.model_version,
            data_root=args.data_root,
        )
        print("seed_snapshot_files=" + ",".join(path.as_posix() for path in paths))
        return

    if args.command == "aggregate":
        paths = aggregate_seed_snapshots(
            seeds=args.seeds,
            artifact_prefix=args.artifact_prefix,
            summary_prefix=args.summary_prefix,
        )
        print("aggregate_files=" + ",".join(path.as_posix() for path in paths))
        return

    if args.command == "run-all":
        for seed in args.seeds:
            run_one_seed(
                seed=seed,
                artifact_prefix=args.artifact_prefix,
                num_samples=args.num_samples,
                topologies=args.topologies,
                quant_mode=args.quant_mode,
                model_version=args.model_version,
                data_root=args.data_root,
            )
        paths = aggregate_seed_snapshots(
            seeds=args.seeds,
            artifact_prefix=args.artifact_prefix,
            summary_prefix=args.summary_prefix,
        )
        print("aggregate_files=" + ",".join(path.as_posix() for path in paths))


if __name__ == "__main__":
    main()

