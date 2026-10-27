"""Generate release figures from the retained experiment artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from group_auth_dnn.config import RESULTS_DIR

if __package__ in (None, ""):
    from plot_style import ATTACK_COLORS, HEATMAP_CMAP, PALETTE, annotate_heatmap, setup_publication_style, style_axis, style_heatmap_axis
else:
    from .plot_style import ATTACK_COLORS, HEATMAP_CMAP, PALETTE, annotate_heatmap, setup_publication_style, style_axis, style_heatmap_axis


ATTACK_ORDER = ["random_proxy", "single_machine_full_model"]
ATTACK_LABELS = {
    "random_proxy": "Random Proxy",
    "single_machine_full_model": "Single-Machine Full Model",
}
ATTACK_TYPE_ORDER = ["skip_node", "swap_order", "replace_node", "inject_params"]
ATTACK_TYPE_LABELS = {
    "skip_node": "Skip Node",
    "swap_order": "Swap Order",
    "replace_node": "Replace Node",
    "inject_params": "Inject Params",
}
MODEL_LABELS = {"mlp": "MLP", "resnet": "ResNet-18", "vit": "ViT/CIFAR-10"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--output-dir", type=Path, default=None)
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


def _save(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_exp1_rows(results_dir: Path) -> list[dict[str, str]]:
    rows = _read_if_exists(results_dir / "m1_block1_mlp.csv")
    rows.extend(_read_if_exists(results_dir / "m1_block1_resnet.csv"))
    rows.extend(_read_if_exists(results_dir / "multiseed_exp1_vit_summary.csv"))
    return rows


def _build_exp2_rows(results_dir: Path) -> list[dict[str, str]]:
    rows = _read_if_exists(results_dir / "m1_block2_mlp.csv")
    rows.extend(_read_if_exists(results_dir / "multiseed_exp2_spotcheck_summary.csv"))
    rows.extend(_read_if_exists(results_dir / "multiseed_exp2_vit_summary.csv"))
    return rows


def plot_exp1(results_dir: Path, output_dir: Path) -> Path | None:
    rows = _build_exp1_rows(results_dir)
    available_models = [model for model in ("mlp", "resnet", "vit") if any(row["model"] == model for row in rows)]
    if not available_models:
        return None

    fig, axes = plt.subplots(1, len(available_models), figsize=(5.5 * len(available_models), 4.6), squeeze=False)
    for axis, model in zip(axes[0], available_models):
        style_axis(axis, grid_axis="both")
        model_rows = [row for row in rows if row["model"] == model]
        target_n = 10 if any(int(row["N"]) == 10 for row in model_rows) else max(int(row["N"]) for row in model_rows)
        for attack in ATTACK_ORDER:
            attack_rows = [row for row in model_rows if row.get("attack_type", row.get("attack", "")) == attack and int(row["N"]) == target_n]
            if not attack_rows:
                continue
            attack_rows.sort(key=lambda item: _to_float(item, "delta"))
            xs = np.array([_to_float(row, "delta") for row in attack_rows], dtype=np.float64)
            y_key = "ASR_Baseline_A_mean" if "ASR_Baseline_A_mean" in attack_rows[0] else "baseline_asr"
            ci_key = "ASR_Baseline_A_ci95" if "ASR_Baseline_A_ci95" in attack_rows[0] else ""
            ys = np.array([_to_float(row, y_key) for row in attack_rows], dtype=np.float64)
            axis.plot(xs, ys, marker="o", linewidth=2.2, markersize=5.5, color=ATTACK_COLORS[attack], label=ATTACK_LABELS[attack])
            if ci_key:
                ci = np.array([_to_float(row, ci_key) for row in attack_rows], dtype=np.float64)
                axis.fill_between(xs, np.clip(ys - ci, 0.0, 1.0), np.clip(ys + ci, 0.0, 1.0), alpha=0.14, color=ATTACK_COLORS[attack])
        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Tolerance delta")
        axis.set_title(f"{MODEL_LABELS[model]} (N={target_n})")
    axes[0][0].set_ylabel("Attack success rate")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("Release Figure 1: Proxy Attack Stress Test")
    return _save(fig, output_dir, "figure1_proxy_attack_curves.png")


def plot_exp2(results_dir: Path, output_dir: Path) -> Path | None:
    rows = _build_exp2_rows(results_dir)
    available_models = [model for model in ("mlp", "resnet", "vit") if any(row["model"] == model for row in rows)]
    if not available_models:
        return None

    fig, axes = plt.subplots(1, len(available_models), figsize=(5.2 * len(available_models), 4.8), squeeze=False)
    palette = [PALETTE["teal"], PALETTE["blue"], PALETTE["gold"]]
    for axis, model, color in zip(axes[0], available_models, palette):
        style_axis(axis, grid_axis="y")
        model_rows = [row for row in rows if row["model"] == model]
        candidate_depths = sorted({int(row["N"]) for row in model_rows})
        target_n = 5 if 5 in candidate_depths else candidate_depths[0]
        target_rows = {row["attack_type"]: row for row in model_rows if int(row["N"]) == target_n}
        xs = np.arange(len(ATTACK_TYPE_ORDER))
        y_key = "FAR_Method_C_mean" if any("FAR_Method_C_mean" in row for row in target_rows.values()) else "method_far"
        ci_key = "FAR_Method_C_ci95" if any("FAR_Method_C_ci95" in row for row in target_rows.values()) else ""
        ys = np.array([_to_float(target_rows.get(attack, {}), y_key) for attack in ATTACK_TYPE_ORDER], dtype=np.float64)
        axis.bar(xs, ys, color=color, edgecolor="white", linewidth=0.9)
        if ci_key:
            ci = np.array([_to_float(target_rows.get(attack, {}), ci_key) for attack in ATTACK_TYPE_ORDER], dtype=np.float64)
            axis.errorbar(xs, ys, yerr=ci, fmt="none", capsize=2, color=PALETTE["ink"], linewidth=0.8)
        axis.set_xticks(xs)
        axis.set_xticklabels([ATTACK_TYPE_LABELS[attack] for attack in ATTACK_TYPE_ORDER], rotation=15)
        axis.set_ylim(0.0, max(0.03, float(ys.max()) + 0.01))
        axis.set_title(f"{MODEL_LABELS[model]} (N={target_n})")
        axis.set_ylabel("Method C FAR")
    fig.suptitle("Release Figure 2: Topology Tampering FAR")
    return _save(fig, output_dir, "figure2_topology_far.png")


def plot_exp3(results_dir: Path, output_dir: Path) -> Path | None:
    rows = _read_if_exists(results_dir / "multiseed_exp3_summary.csv")
    if not rows:
        return None

    available_models = [model for model in ("mlp", "resnet", "transformer") if any(row["model"] == model for row in rows)]
    fig, axes = plt.subplots(len(available_models), 2, figsize=(12.5, 4.0 * len(available_models)), squeeze=False)
    encodings = ["fp16", "int8", "mixed"]
    colors = {"fp16": PALETTE["blue"], "int8": PALETTE["teal"], "mixed": PALETTE["gold"]}

    for row_index, model in enumerate(available_models):
        model_rows = [row for row in rows if row["model"] == model]
        depths = sorted({int(row["N"]) for row in model_rows})
        xs = np.arange(len(depths))
        for axis in axes[row_index]:
            style_axis(axis, grid_axis="y")
        for enc_index, encoding in enumerate(encodings):
            enc_rows = {(int(row["N"]), row["encoding"]): row for row in model_rows}
            qmatch = np.array([_to_float(enc_rows.get((depth, encoding), {}), "QMatch_mean") for depth in depths], dtype=np.float64)
            qmatch_ci = np.array([_to_float(enc_rows.get((depth, encoding), {}), "QMatch_ci95") for depth in depths], dtype=np.float64)
            frr = np.array([_to_float(enc_rows.get((depth, encoding), {}), "FRR_mean") for depth in depths], dtype=np.float64)
            frr_ci = np.array([_to_float(enc_rows.get((depth, encoding), {}), "FRR_ci95") for depth in depths], dtype=np.float64)
            shift = (enc_index - 1) * 0.22
            axes[row_index, 0].bar(xs + shift, qmatch, width=0.22, color=colors[encoding], label=encoding.upper())
            axes[row_index, 0].errorbar(xs + shift, qmatch, yerr=qmatch_ci, fmt="none", capsize=2, color=PALETTE["ink"], linewidth=0.8)
            axes[row_index, 1].bar(xs + shift, frr, width=0.22, color=colors[encoding], label=encoding.upper())
            axes[row_index, 1].errorbar(xs + shift, frr, yerr=frr_ci, fmt="none", capsize=2, color=PALETTE["ink"], linewidth=0.8)
        axes[row_index, 0].set_xticks(xs)
        axes[row_index, 0].set_xticklabels([str(depth) for depth in depths])
        axes[row_index, 1].set_xticks(xs)
        axes[row_index, 1].set_xticklabels([str(depth) for depth in depths])
        axes[row_index, 0].set_ylabel(f"{model.upper()} QMatch")
        axes[row_index, 1].set_ylabel(f"{model.upper()} FRR")
        axes[row_index, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Release Figure 3: Quantization Stability")
    return _save(fig, output_dir, "figure3_quantization_stability.png")


def plot_exp4(results_dir: Path, output_dir: Path) -> Path | None:
    rows = _read_if_exists(results_dir / "exp4_overhead.csv")
    rows.extend(_read_if_exists(results_dir / "multiseed_exp4_vit_summary.csv"))
    if not rows:
        return None

    available_models = [model for model in ("mlp", "resnet", "vit") if any(row["model"] == model for row in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    colors = {"mlp": PALETTE["teal"], "resnet": PALETTE["blue"], "vit": PALETTE["gold"]}
    for axis in axes:
        style_axis(axis, grid_axis="y")
    for model in available_models:
        model_rows = [row for row in rows if row["model"] == model]
        model_rows.sort(key=lambda row: int(row["N"]))
        xs = np.array([int(row["N"]) for row in model_rows], dtype=np.int64)
        if "delta_T_ms_mean" in model_rows[0]:
            delta_t = np.array([_to_float(row, "delta_T_ms_mean") for row in model_rows], dtype=np.float64)
            delta_b = np.array([_to_float(row, "delta_B_bytes_mean") for row in model_rows], dtype=np.float64)
        else:
            delta_t = np.array([_to_float(row, "delta_T_ms") for row in model_rows], dtype=np.float64)
            delta_b = np.array([_to_float(row, "delta_B_bytes") for row in model_rows], dtype=np.float64)
        axes[0].plot(xs, delta_t, marker="o", linewidth=2.2, color=colors[model], label=MODEL_LABELS[model])
        axes[1].plot(xs, delta_b, marker="o", linewidth=2.2, color=colors[model], label=MODEL_LABELS[model])
    axes[0].set_xlabel("Topology depth N")
    axes[0].set_ylabel("delta_T (ms)")
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("Topology depth N")
    axes[1].set_ylabel("delta_B (bytes)")
    fig.suptitle("Release Figure 4: System Overhead")
    return _save(fig, output_dir, "figure4_system_overhead.png")


def plot_exp6(results_dir: Path, output_dir: Path) -> Path | None:
    rows = _read_if_exists(results_dir / "multiseed_exp6_summary.csv")
    if not rows:
        return None

    run_ids = ["R017", "R018", "R019", "R020", "R021"]
    labels = {
        "R017": "No measurement",
        "R018": "No freshness",
        "R019": "No act. chain",
        "R020": "No chain state",
        "R021": "Signature mode",
    }
    heatmap = []
    y_labels = []
    for run_id in run_ids:
        subset = [row for row in rows if row["run_id"] == run_id]
        if not subset:
            continue
        method_values = [_to_float(row, "method_c_value_mean") for row in subset]
        ablation_values = [_to_float(row, "ablation_value_mean") for row in subset]
        heatmap.append([max(method_values), max(ablation_values)])
        y_labels.append(labels[run_id])

    if not heatmap:
        return None

    data = np.array(heatmap, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    image = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=float(data.min()), vmax=float(data.max()) or 1.0, aspect="auto")
    style_heatmap_axis(ax, data.shape[0], data.shape[1])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Method C", "Ablation"])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title("Release Figure 5: Key Ablation Summary")
    annotate_heatmap(ax, data)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Acceptance / FAR metric")
    return _save(fig, output_dir, "figure5_ablation_summary.png")


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "figures")).resolve()

    setup_publication_style()
    generated: list[Path] = []
    for builder in (plot_exp1, plot_exp2, plot_exp3, plot_exp4, plot_exp6):
        figure = builder(results_dir, output_dir)
        if figure is not None:
            generated.append(figure)

    if not generated:
        raise FileNotFoundError(f"No retained result files found under {results_dir.as_posix()}")

    print(f"output_dir={output_dir.as_posix()}")
    for path in generated:
        print(f"generated={path.as_posix()}")


if __name__ == "__main__":
    main()
