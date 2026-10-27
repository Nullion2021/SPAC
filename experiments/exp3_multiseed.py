"""Targeted multi-seed runner and aggregator for Block 3 / exp3 quantization results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys


import numpy as np

from group_auth_dnn.config import RESULTS_DIR
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json

if __package__ in (None, ""):
    from exp3_quant import run_exp3
else:
    from .exp3_quant import run_exp3


DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_seed = subparsers.add_parser("run-seed", help="run exp3 for one seed and save a seed-specific snapshot")
    run_seed.add_argument("--seed", type=int, required=True)
    run_seed.add_argument("--num-samples", type=int, default=64)
    run_seed.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    run_seed.add_argument("--artifact-prefix", type=str, default="exp3_multiseed")

    aggregate = subparsers.add_parser("aggregate", help="aggregate seed-specific exp3 snapshots into multiseed tables")
    aggregate.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    aggregate.add_argument("--artifact-prefix", type=str, default="exp3_multiseed")
    aggregate.add_argument("--summary-prefix", type=str, default="multiseed_exp3")

    run_all = subparsers.add_parser("run-all", help="run exp3 sequentially for multiple seeds and aggregate locally")
    run_all.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    run_all.add_argument("--num-samples", type=int, default=64)
    run_all.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    run_all.add_argument("--artifact-prefix", type=str, default="exp3_multiseed")
    run_all.add_argument("--summary-prefix", type=str, default="multiseed_exp3")

    return parser.parse_args()


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(
    rows: list[dict[str, object]],
    *,
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict[str, object]]:
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


def _materialize_seed_snapshot(*, seed: int, artifact_prefix: str) -> tuple[Path, Path, Path]:
    summary_payload = _read_json(Path(RESULTS_DIR) / "exp3_quant_summary.json")
    stability_rows = summary_payload["stability_rows"]
    theory_rows = summary_payload["theory_rows"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stability_versioned, stability_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_stability.csv",
        stability_rows,
        fieldnames=list(stability_rows[0].keys()),
        timestamp=timestamp,
    )
    theory_versioned, theory_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_theory.csv",
        theory_rows,
        fieldnames=list(theory_rows[0].keys()),
        timestamp=timestamp,
    )
    summary_versioned, summary_latest = write_versioned_json(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_summary.json",
        summary_payload,
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", stability_versioned.as_posix(), "results", f"exp3 seed {seed} stability snapshot"),
            (manifest_ts, "/experiment-bridge", stability_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", theory_versioned.as_posix(), "results", f"exp3 seed {seed} theory snapshot"),
            (manifest_ts, "/experiment-bridge", theory_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", summary_versioned.as_posix(), "results", f"exp3 seed {seed} JSON snapshot"),
            (manifest_ts, "/experiment-bridge", summary_latest.as_posix(), "results", "latest copy"),
        ]
    )
    return stability_latest, theory_latest, summary_latest


def run_one_seed(*, seed: int, num_samples: int, device: str, artifact_prefix: str) -> tuple[Path, Path, Path]:
    print(f"[exp3-multiseed] running seed={seed} num_samples={num_samples} device={device}")
    run_exp3(base_seed=seed, num_samples=num_samples, device_arg=device)
    return _materialize_seed_snapshot(seed=seed, artifact_prefix=artifact_prefix)


def aggregate_seed_snapshots(*, seeds: list[int], artifact_prefix: str, summary_prefix: str) -> tuple[Path, Path, Path, Path, Path]:
    raw_rows: list[dict[str, object]] = []
    theory_raw_rows: list[dict[str, object]] = []
    source_files: list[str] = []

    for seed in seeds:
        summary_path = Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing seed snapshot: {summary_path.as_posix()}")
        payload = _read_json(summary_path)
        source_files.append(summary_path.as_posix())
        for row in payload["stability_rows"]:
            raw_rows.append(
                {
                    "seed": seed,
                    "model": row["model"],
                    "N": int(row["N"]),
                    "encoding": row["encoding"],
                    "QMatch": float(row["QMatch"]),
                    "FRR": float(row["FRR"]),
                }
            )
        for row in payload["theory_rows"]:
            theory_raw_rows.append(
                {
                    "seed": seed,
                    "model": row["model"],
                    "N": int(row["N"]),
                    "encoding": row["encoding"],
                    "Bound_theory": float(row["Bound_theory"]),
                    "delta_QMatch": float(row["delta_QMatch"]),
                    "SA1_phase_deviation": float(row["SA1_phase_deviation"]),
                    "SA2_channel_correlation": float(row["SA2_channel_correlation"]),
                }
            )

    summary_rows = _summarize(
        raw_rows,
        group_keys=["model", "N", "encoding"],
        metric_keys=["QMatch", "FRR"],
    )
    theory_summary_rows = _summarize(
        theory_raw_rows,
        group_keys=["model", "N", "encoding"],
        metric_keys=["Bound_theory", "delta_QMatch", "SA1_phase_deviation", "SA2_channel_correlation"],
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
    theory_raw_versioned, theory_raw_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{summary_prefix}_theory_raw.csv",
        theory_raw_rows,
        fieldnames=list(theory_raw_rows[0].keys()),
        timestamp=timestamp,
    )
    theory_summary_versioned, theory_summary_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{summary_prefix}_theory_summary.csv",
        theory_summary_rows,
        fieldnames=list(theory_summary_rows[0].keys()),
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
            "theory_raw_rows": theory_raw_rows,
            "theory_summary_rows": theory_summary_rows,
        },
        timestamp=timestamp,
    )

    manifest_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_manifest(
        [
            (manifest_ts, "/experiment-bridge", raw_versioned.as_posix(), "results", "exp3 multiseed raw table"),
            (manifest_ts, "/experiment-bridge", raw_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", summary_versioned.as_posix(), "results", "exp3 multiseed summary table"),
            (manifest_ts, "/experiment-bridge", summary_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", theory_raw_versioned.as_posix(), "results", "exp3 multiseed theory raw table"),
            (manifest_ts, "/experiment-bridge", theory_raw_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", theory_summary_versioned.as_posix(), "results", "exp3 multiseed theory summary table"),
            (manifest_ts, "/experiment-bridge", theory_summary_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", bundle_versioned.as_posix(), "results", "exp3 multiseed bundle JSON"),
            (manifest_ts, "/experiment-bridge", bundle_latest.as_posix(), "results", "latest copy"),
        ]
    )
    return raw_latest, summary_latest, theory_raw_latest, theory_summary_latest, bundle_latest


def main() -> None:
    args = _parse_args()
    if args.command == "run-seed":
        paths = run_one_seed(seed=args.seed, num_samples=args.num_samples, device=args.device, artifact_prefix=args.artifact_prefix)
        print(f"saved {paths[0].as_posix()}")
        print(f"saved {paths[1].as_posix()}")
        print(f"saved {paths[2].as_posix()}")
        return
    if args.command == "aggregate":
        paths = aggregate_seed_snapshots(seeds=args.seeds, artifact_prefix=args.artifact_prefix, summary_prefix=args.summary_prefix)
        print(f"saved {paths[0].as_posix()}")
        print(f"saved {paths[1].as_posix()}")
        print(f"saved {paths[2].as_posix()}")
        print(f"saved {paths[3].as_posix()}")
        print(f"saved {paths[4].as_posix()}")
        return
    if args.command == "run-all":
        for seed in args.seeds:
            run_one_seed(seed=seed, num_samples=args.num_samples, device=args.device, artifact_prefix=args.artifact_prefix)
        paths = aggregate_seed_snapshots(seeds=args.seeds, artifact_prefix=args.artifact_prefix, summary_prefix=args.summary_prefix)
        print(f"saved {paths[1].as_posix()}")
        print(f"saved {paths[3].as_posix()}")
        print(f"saved {paths[4].as_posix()}")
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

