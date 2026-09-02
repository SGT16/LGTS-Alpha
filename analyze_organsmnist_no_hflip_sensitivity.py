from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


MODELS = ["ResNet18", "MobileNetV2", "EfficientNetV2S"]
ACTIVATIONS = ["LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native"]
SEEDS = [936137, 529212, 448132, 942529, 110107]
KEYS = ["model", "activation", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired OrganSMNIST no-horizontal-flip sensitivity analysis")
    parser.add_argument("--sensitivity-root", type=Path, default=None)
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--sensitivity-csv", type=Path, default=None)
    parser.add_argument("--original-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=75)
    args = parser.parse_args()
    if args.sensitivity_root is None and args.sensitivity_csv is None:
        parser.error("provide --sensitivity-root or --sensitivity-csv")
    if args.original_root is None and args.original_csv is None:
        parser.error("provide --original-root or --original-csv")
    return args


def load_done(
    root: Path | None,
    expected_augmentation: str | None = None,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    rows = []
    if csv_path is not None:
        source_rows = pd.read_csv(csv_path).to_dict(orient="records")
        source_label = csv_path
    else:
        if root is None:
            raise RuntimeError("A results root or CSV file is required")
        source_rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("done.json")]
        source_label = root
    for row in source_rows:
        if row.get("dataset") != "organsmnist":
            continue
        if row.get("model") not in MODELS or row.get("activation") not in ACTIVATIONS:
            continue
        if int(row.get("seed", -1)) not in SEEDS:
            continue
        if expected_augmentation is not None and row.get("augmentation") != expected_augmentation:
            continue
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No matching completed runs in {source_label}")
    if frame.duplicated(KEYS).any():
        duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
        raise RuntimeError(f"Duplicate completed keys in {source_label}:\n{duplicates}")
    return frame


def main() -> None:
    args = parse_args()
    sensitivity = load_done(args.sensitivity_root, "organsmnist_no_hflip", args.sensitivity_csv)
    original = load_done(args.original_root, "standard", args.original_csv)
    if len(sensitivity) != args.expected_runs:
        raise RuntimeError(f"Expected {args.expected_runs} sensitivity runs, found {len(sensitivity)}")
    expected_keys = {(model, activation, seed) for model in MODELS for activation in ACTIVATIONS for seed in SEEDS}
    actual_keys = set(map(tuple, sensitivity[KEYS].itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        raise RuntimeError(f"Sensitivity key mismatch: missing={sorted(expected_keys - actual_keys)}")

    columns = KEYS + ["test_f1_macro", "test_accuracy"]
    paired = original[columns].merge(
        sensitivity[columns], on=KEYS, how="inner", validate="one_to_one",
        suffixes=("_standard", "_no_hflip"),
    )
    if len(paired) != args.expected_runs:
        raise RuntimeError(f"Expected {args.expected_runs} paired runs, found {len(paired)}")
    paired["delta_f1_no_hflip_minus_standard"] = (
        paired["test_f1_macro_no_hflip"] - paired["test_f1_macro_standard"]
    )
    paired["delta_accuracy_no_hflip_minus_standard"] = (
        paired["test_accuracy_no_hflip"] - paired["test_accuracy_standard"]
    )

    summary_rows = []
    for (model, activation), block in paired.groupby(["model", "activation"], sort=False):
        delta = block["delta_f1_no_hflip_minus_standard"].to_numpy(float)
        n = len(delta)
        delta_sd = float(delta.std(ddof=1))
        margin = float(t.ppf(0.975, n - 1) * delta_sd / math.sqrt(n))
        summary_rows.append({
            "model": model,
            "activation": activation,
            "n_pairs": n,
            "standard_f1_mean": block["test_f1_macro_standard"].mean(),
            "standard_f1_sd": block["test_f1_macro_standard"].std(ddof=1),
            "no_hflip_f1_mean": block["test_f1_macro_no_hflip"].mean(),
            "no_hflip_f1_sd": block["test_f1_macro_no_hflip"].std(ddof=1),
            "paired_delta_f1_mean": delta.mean(),
            "paired_delta_f1_sd": delta_sd,
            "paired_delta_f1_ci95_low": delta.mean() - margin,
            "paired_delta_f1_ci95_high": delta.mean() + margin,
        })
    summary = pd.DataFrame(summary_rows)
    model_order = {value: index for index, value in enumerate(MODELS)}
    activation_order = {value: index for index, value in enumerate(ACTIVATIONS)}
    summary = summary.sort_values(
        ["model", "activation"],
        key=lambda series: series.map(model_order if series.name == "model" else activation_order),
    )

    output_dir = args.output_dir or args.sensitivity_root or Path("reproduced/organsmnist_no_hflip")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_run_results.csv", index=False)
    summary.to_csv(output_dir / "paired_sensitivity_summary.csv", index=False)
    integrity = {
        "status": "complete",
        "expected_runs": args.expected_runs,
        "sensitivity_runs": len(sensitivity),
        "paired_runs": len(paired),
        "models": MODELS,
        "activations": ACTIVATIONS,
        "seeds": SEEDS,
        "augmentation": "organsmnist_no_hflip",
    }
    (output_dir / "sensitivity_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    labels = ["LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native"]
    for ax, model in zip(axes, MODELS):
        block = summary[summary.model == model].set_index("activation").loc[ACTIVATIONS]
        mean = block["paired_delta_f1_mean"].to_numpy(float)
        low = block["paired_delta_f1_ci95_low"].to_numpy(float)
        high = block["paired_delta_f1_ci95_high"].to_numpy(float)
        x = np.arange(len(labels))
        ax.errorbar(x, mean, yerr=np.vstack((mean - low, high - mean)), fmt="o", capsize=3, color="#D55E00")
        ax.axhline(0, color="#555555", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(model.replace("EfficientNetV2S", "EfficientNetV2-S"))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Paired macro-F1 change\n(no horizontal flip - standard)")
    fig.suptitle("Post-confirmatory OrganSMNIST augmentation sensitivity", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / "organ_no_hflip_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensitivity analysis complete: {len(paired)} paired runs")


if __name__ == "__main__":
    main()
