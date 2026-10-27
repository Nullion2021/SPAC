"""Generate release-oriented summary tables from the retained experiment artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from group_auth_dnn.utils.artifacts import write_versioned_csv, write_versioned_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", type=str, default="release_summary")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_if_exists(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def build_main_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for row in _read_if_exists(results_dir / "m1_block1_mlp.csv"):
        rows.append(
            {
                "block": "exp1",
                "artifact": "proxy_attack",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": f"{row['attack_type']} delta={row['delta']}",
                "metric_a": _to_float(row, "baseline_asr"),
                "metric_b": _to_float(row, "method_far"),
                "source_file": "m1_block1_mlp.csv",
            }
        )
    for row in _read_if_exists(results_dir / "m1_block1_resnet.csv"):
        rows.append(
            {
                "block": "exp1",
                "artifact": "proxy_attack",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": f"{row['attack_type']} delta={row['delta']}",
                "metric_a": _to_float(row, "baseline_asr"),
                "metric_b": _to_float(row, "method_far"),
                "source_file": "m1_block1_resnet.csv",
            }
        )
    for row in _read_if_exists(results_dir / "multiseed_exp1_vit_summary.csv"):
        rows.append(
            {
                "block": "exp1",
                "artifact": "proxy_attack",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": f"{row['attack']} delta={row['delta']}",
                "metric_a": _to_float(row, "ASR_Baseline_A_mean"),
                "metric_b": _to_float(row, "FAR_Method_C_mean"),
                "source_file": "multiseed_exp1_vit_summary.csv",
            }
        )

    for source_name in ("m1_block2_mlp.csv", "multiseed_exp2_spotcheck_summary.csv", "multiseed_exp2_vit_summary.csv"):
        for row in _read_if_exists(results_dir / source_name):
            rows.append(
                {
                    "block": "exp2",
                    "artifact": "topology_tampering",
                    "model": row["model"],
                    "N": int(row["N"]),
                    "condition": row["attack_type"],
                    "metric_a": _to_float(row, "FAR_Method_C_mean", _to_float(row, "method_far")),
                    "metric_b": _to_float(row, "valid_samples_mean", _to_float(row, "valid_samples")),
                    "source_file": source_name,
                }
            )

    for row in _read_if_exists(results_dir / "multiseed_exp3_summary.csv"):
        rows.append(
            {
                "block": "exp3",
                "artifact": "quantization_stability",
                "model": row["model"],
                "N": int(row["N"]),
                "condition": row["encoding"],
                "metric_a": _to_float(row, "QMatch_mean"),
                "metric_b": _to_float(row, "FRR_mean"),
                "source_file": "multiseed_exp3_summary.csv",
            }
        )

    for source_name in ("exp4_overhead.csv", "multiseed_exp4_vit_summary.csv"):
        for row in _read_if_exists(results_dir / source_name):
            rows.append(
                {
                    "block": "exp4",
                    "artifact": "system_overhead",
                    "model": row["model"],
                    "N": int(row["N"]),
                    "condition": "honest_inference",
                    "metric_a": _to_float(row, "delta_T_ms_mean", _to_float(row, "delta_T_ms")),
                    "metric_b": _to_float(row, "delta_B_bytes_mean", _to_float(row, "delta_B_bytes")),
                    "source_file": source_name,
                }
            )

    return rows


def build_ablation_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_if_exists(results_dir / "multiseed_exp6_summary.csv"):
        rows.append(
            {
                "run_id": row["run_id"],
                "variant": row["variant"],
                "model": row["model"],
                "N": int(row["N"]),
                "condition": row["condition"],
                "metric": row["metric"],
                "baseline_e_value": row["baseline_e_value_mean"],
                "method_c_value": row["method_c_value_mean"],
                "ablation_value": row["ablation_value_mean"],
                "extra_key": row["extra_key"],
                "extra_value": row["extra_value_mean"],
                "source_file": "multiseed_exp6_summary.csv",
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    main_rows = build_main_rows(results_dir)
    ablation_rows = build_ablation_rows(results_dir)
    if not main_rows and not ablation_rows:
        raise FileNotFoundError(f"No retained result tables found under {results_dir.as_posix()}")

    bundle: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": results_dir.as_posix(),
        "outputs": {},
    }

    if main_rows:
        _, latest = write_versioned_csv(
            results_dir / f"{args.output_prefix}_main_results.csv",
            main_rows,
            fieldnames=list(main_rows[0].keys()),
            timestamp=timestamp,
        )
        bundle["outputs"]["main_results"] = latest.as_posix()

    if ablation_rows:
        _, latest = write_versioned_csv(
            results_dir / f"{args.output_prefix}_ablations.csv",
            ablation_rows,
            fieldnames=list(ablation_rows[0].keys()),
            timestamp=timestamp,
        )
        bundle["outputs"]["ablations"] = latest.as_posix()

    _, bundle_latest = write_versioned_json(
        results_dir / f"{args.output_prefix}_bundle.json",
        bundle,
        timestamp=timestamp,
    )

    print(f"bundle={bundle_latest.as_posix()}")
    for label, path in bundle["outputs"].items():
        print(f"{label}={path}")


if __name__ == "__main__":
    main()
