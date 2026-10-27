"""Targeted multi-seed runner and aggregator for Block 6 / ablation results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[1]
import numpy as np

from group_auth_dnn.config import RESULTS_DIR
from group_auth_dnn.utils.artifacts import append_manifest, write_versioned_csv, write_versioned_json


DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_run_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--num-samples", type=int, default=64)
        target.add_argument("--warmup-runs", type=int, default=3)
        target.add_argument("--artifact-prefix", type=str, default="exp6_multiseed")
        target.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
        target.add_argument("--topologies", type=int, nargs="+", default=[3, 5, 10])
        target.add_argument("--target-hops", type=int, nargs="+", default=[2])
        target.add_argument("--strengths", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])

    run_seed = subparsers.add_parser("run-seed", help="run Block 6 for one seed and save a seed-specific snapshot")
    add_shared_run_args(run_seed)
    run_seed.add_argument("--seed", type=int, required=True)

    aggregate = subparsers.add_parser("aggregate", help="aggregate seed-specific Block 6 snapshots into multiseed tables")
    aggregate.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    aggregate.add_argument("--artifact-prefix", type=str, default="exp6_multiseed")
    aggregate.add_argument("--summary-prefix", type=str, default="multiseed_exp6")

    run_all = subparsers.add_parser("run-all", help="run Block 6 sequentially for multiple seeds and aggregate locally")
    run_all.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    add_shared_run_args(run_all)
    run_all.add_argument("--summary-prefix", type=str, default="multiseed_exp6")

    return parser.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _run_script(script_name: str, *script_args: object) -> None:
    command = [sys.executable, str(RELEASE_ROOT / "experiments" / script_name)]
    command.extend(str(arg) for arg in script_args)
    completed = subprocess.run(
        command,
        cwd=str(RELEASE_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")


def _build_block6_rows_from_latest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    measurement_rows = _read_csv(Path(RESULTS_DIR) / "m4_ablation_no_measurement_mlp.csv")
    for row in measurement_rows:
        if row["variant"] != "wo_measurement":
            continue
        method_row = next(
            item for item in measurement_rows if item["N"] == row["N"] and item["variant"] == "method_c"
        )
        rows.append(
            {
                "run_id": "R017",
                "variant": "w/o mu_i",
                "model": "mlp",
                "N": int(row["N"]),
                "condition": "inject_params",
                "metric": "FAR",
                "baseline_e_value": None,
                "method_c_value": float(method_row["far"]),
                "ablation_value": float(row["far"]),
                "extra_key": "",
                "extra_value": None,
                "num_samples": float(row["num_samples"]),
            }
        )

    freshness_rows = _read_csv(Path(RESULTS_DIR) / "m4_ablation_no_freshness_mlp.csv")
    for row in freshness_rows:
        if row["variant"] != "wo_freshness":
            continue
        method_row = next(
            item for item in freshness_rows if item["N"] == row["N"] and item["variant"] == "method_c"
        )
        rows.append(
            {
                "run_id": "R018",
                "variant": "w/o sid/nonce",
                "model": "mlp",
                "N": int(row["N"]),
                "condition": "cross_session_replay",
                "metric": "replay_accept_rate",
                "baseline_e_value": None,
                "method_c_value": float(method_row["replay_accept_rate"]),
                "ablation_value": float(row["replay_accept_rate"]),
                "extra_key": "sid_changed_rate",
                "extra_value": float(row["sid_changed_rate"]),
                "num_samples": float(row["num_samples"]),
            }
        )

    activation_rows = _read_csv(Path(RESULTS_DIR) / "m4_ablation_no_activation_chain_mlp.csv")
    for row in activation_rows:
        rows.append(
            {
                "run_id": "R019",
                "variant": "w/o u_i/v_i",
                "model": "mlp",
                "N": int(row["N"]),
                "condition": f"target_hop={row['target_hop']};strength={row['strength']}",
                "metric": "AIR",
                "baseline_e_value": float(row["baseline_e_air"]),
                "method_c_value": float(row["method_c_air"]),
                "ablation_value": float(row["wo_activation_chain_air"]),
                "extra_key": "injection_l2_mean",
                "extra_value": float(row["injection_l2_mean"]),
                "num_samples": float(row["num_samples"]),
            }
        )

    chain_state_rows = _read_csv(Path(RESULTS_DIR) / "m4_ablation_no_chain_state_mlp.csv")
    for row in chain_state_rows:
        rows.append(
            {
                "run_id": "R020",
                "variant": "w/o z_i",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": row["attack_type"],
                "metric": "FAR",
                "baseline_e_value": None,
                "method_c_value": float(row["method_c_far"]),
                "ablation_value": float(row["wo_chain_state_far"]),
                "extra_key": "",
                "extra_value": None,
                "num_samples": float(row["num_samples"]),
            }
        )

    signature_rows = _read_csv(Path(RESULTS_DIR) / "m4_ablation_signature_mode_mlp.csv")
    for row in signature_rows:
        rows.append(
            {
                "run_id": "R021",
                "variant": "HMAC -> Signature",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": "honest_inference",
                "metric": "signature_over_hmac_ratio",
                "baseline_e_value": None,
                "method_c_value": float(row["delta_t_hmac_ms"]),
                "ablation_value": float(row["delta_t_signature_ms"]),
                "extra_key": "signature_over_hmac_ratio",
                "extra_value": float(row["signature_over_hmac_ratio"]),
                "num_samples": float(row["num_samples"]),
            }
        )

    return rows


def _materialize_seed_snapshot(*, seed: int, artifact_prefix: str) -> tuple[Path, Path]:
    seed_rows = _build_block6_rows_from_latest()
    bundle_payload = {
        "seed": seed,
        "artifact_prefix": artifact_prefix,
        "rows": seed_rows,
        "source_latest_files": [
            (Path(RESULTS_DIR) / "m4_ablation_no_measurement_mlp.json").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_measurement_mlp.csv").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_freshness_mlp.json").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_freshness_mlp.csv").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_activation_chain_mlp.json").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_activation_chain_mlp.csv").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_chain_state_mlp.json").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_no_chain_state_mlp.csv").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_signature_mode_mlp.json").as_posix(),
            (Path(RESULTS_DIR) / "m4_ablation_signature_mode_mlp.csv").as_posix(),
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_versioned, rows_latest = write_versioned_csv(
        Path(RESULTS_DIR) / f"{artifact_prefix}_seed{seed}_rows.csv",
        seed_rows,
        fieldnames=list(seed_rows[0].keys()),
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
            (manifest_ts, "/experiment-bridge", rows_versioned.as_posix(), "results", f"exp6 seed {seed} unified rows"),
            (manifest_ts, "/experiment-bridge", rows_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", bundle_versioned.as_posix(), "results", f"exp6 seed {seed} bundle JSON"),
            (manifest_ts, "/experiment-bridge", bundle_latest.as_posix(), "results", "latest copy"),
        ]
    )
    return rows_latest, bundle_latest


def _summarize_metric(values: list[float | None]) -> tuple[object, object, object]:
    present = [value for value in values if value is not None]
    if not present:
        return "", "", ""
    array = np.array(present, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return round(mean, 6), round(std, 6), round(ci95, 6)


def _summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    group_keys = ["run_id", "variant", "model", "N", "condition", "metric", "extra_key"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for group_key, items in groups.items():
        out_row = {key: value for key, value in zip(group_keys, group_key)}
        out_row["n_seeds"] = len(items)
        for metric_key in ("baseline_e_value", "method_c_value", "ablation_value", "extra_value", "num_samples"):
            mean, std, ci95 = _summarize_metric([_parse_float_or_none(item.get(metric_key)) for item in items])
            out_row[f"{metric_key}_mean"] = mean
            out_row[f"{metric_key}_std"] = std
            out_row[f"{metric_key}_ci95"] = ci95
        summary_rows.append(out_row)

    summary_rows.sort(key=lambda row: tuple(row[key] for key in group_keys))
    return summary_rows


def run_one_seed(
    *,
    seed: int,
    artifact_prefix: str,
    num_samples: int,
    warmup_runs: int,
    device: str,
    topologies: list[int],
    target_hops: list[int],
    strengths: list[float],
) -> tuple[Path, Path]:
    print(f"[exp6-multiseed] running seed={seed} num_samples={num_samples} device={device}")

    topo_args = ["--topologies", *topologies]
    target_hop_args = ["--target-hops", *target_hops]
    strength_args = ["--strengths", *strengths]

    _run_script(
        "exp6_ablation_no_measurement.py",
        "--num-samples", num_samples,
        "--seed", seed,
        *topo_args,
        "--device", device,
    )
    _run_script(
        "exp6_ablation_no_freshness.py",
        "--num-samples", num_samples,
        "--seed", seed,
        *topo_args,
        "--device", device,
    )
    _run_script(
        "exp6_ablation_no_activation_chain.py",
        "--num-samples", num_samples,
        "--seed", seed,
        *topo_args,
        *target_hop_args,
        *strength_args,
        "--device", device,
    )
    _run_script(
        "exp6_ablation_no_chain_state.py",
        "--num-samples", num_samples,
        "--seed", seed,
        *topo_args,
        "--device", device,
    )
    _run_script(
        "exp6_ablation_signature_mode.py",
        "--num-samples", num_samples,
        "--warmup-runs", warmup_runs,
        "--seed", seed,
        *topo_args,
        "--device", device,
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

    summary_rows = _summarize_rows(raw_rows)

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
            (manifest_ts, "/experiment-bridge", raw_versioned.as_posix(), "results", "exp6 multiseed raw table"),
            (manifest_ts, "/experiment-bridge", raw_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", summary_versioned.as_posix(), "results", "exp6 multiseed summary table"),
            (manifest_ts, "/experiment-bridge", summary_latest.as_posix(), "results", "latest copy"),
            (manifest_ts, "/experiment-bridge", bundle_versioned.as_posix(), "results", "exp6 multiseed bundle JSON"),
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
            warmup_runs=args.warmup_runs,
            device=args.device,
            topologies=args.topologies,
            target_hops=args.target_hops,
            strengths=args.strengths,
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
                warmup_runs=args.warmup_runs,
                device=args.device,
                topologies=args.topologies,
                target_hops=args.target_hops,
                strengths=args.strengths,
            )
        paths = aggregate_seed_snapshots(
            seeds=args.seeds,
            artifact_prefix=args.artifact_prefix,
            summary_prefix=args.summary_prefix,
        )
        print("aggregate_files=" + ",".join(path.as_posix() for path in paths))


if __name__ == "__main__":
    main()

