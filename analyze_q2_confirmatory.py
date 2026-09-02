from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


DATASETS = ("cifar100", "dermamnist", "organsmnist")
MODELS = ("ResNet18", "MobileNetV2", "EfficientNetV2S")
ACTIVATIONS = ("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")
SEEDS = (936137, 529212, 448132, 942529, 110107)
PRIMARY = "LGTS-f4-Alpha"
NONINFERIORITY_MARGIN = 0.01
BOOTSTRAP_SEED = 684731
BOOTSTRAP_REPEATS = 20_000


def normalized_activation(value: str) -> str:
    for name in ("LGTS-f4-Alpha", "LGTS-f2-Slope"):
        if value.startswith(name):
            return name
    if value.startswith("Native"):
        return "Native"
    return value


def finite_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_rows(output_dir: Path, input_csv: Path | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = [input_csv] if input_csv is not None else sorted(output_dir.glob("exp_*/results.csv"))
    for path in paths:
        if path is None:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("protocol") != "full_train":
                    continue
                item = dict(row)
                item["activation_normalized"] = normalized_activation(
                    item.get("activation_variant") or item.get("activation") or ""
                )
                item["source_results_csv"] = path.name
                rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def integrity_report(rows: list[dict[str, str]], expected_runs: int) -> tuple[dict, dict[tuple, dict[str, str]]]:
    by_key: dict[tuple, dict[str, str]] = {}
    duplicates = []
    unexpected = []
    for row in rows:
        try:
            key = (
                row["dataset"],
                row["model"],
                row["activation_normalized"],
                int(row["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            unexpected.append({"reason": "invalid_key", "row": row})
            continue
        if key[0] not in DATASETS or key[1] not in MODELS or key[2] not in ACTIVATIONS or key[3] not in SEEDS:
            unexpected.append({"reason": "outside_locked_panel", "key": key})
            continue
        if row.get("evaluation_split") != "test" or row.get("status") != "confirmatory_done":
            unexpected.append({"reason": "not_confirmatory_test", "key": key})
            continue
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row

    expected_keys = {
        (dataset, model, activation, seed)
        for dataset in DATASETS
        for model in MODELS
        for activation in ACTIVATIONS
        for seed in SEEDS
    }
    missing = sorted(expected_keys - set(by_key))
    report = {
        "expected_runs_argument": expected_runs,
        "locked_expected_runs": len(expected_keys),
        "raw_full_train_rows": len(rows),
        "unique_locked_rows": len(by_key),
        "missing_count": len(missing),
        "duplicate_count": len(duplicates),
        "unexpected_count": len(unexpected),
        "complete": (
            expected_runs == len(expected_keys)
            and len(by_key) == len(expected_keys)
            and not missing
            and not duplicates
            and not unexpected
        ),
        "missing_keys": [list(item) for item in missing],
        "duplicate_keys": [list(item) for item in duplicates],
        "unexpected": unexpected,
    }
    return report, by_key


def condition_summary(rows: list[dict[str, str]]) -> list[dict]:
    metrics = (
        "test_f1_macro",
        "test_accuracy",
        "test_error_rate_percent",
        "test_precision_macro",
        "test_recall_macro",
        "test_mcc",
        "test_balanced_accuracy",
        "test_specificity_macro",
        "test_cohen_kappa",
        "test_roc_auc_macro_ovr",
        "test_pr_auc_macro",
        "test_nll",
        "test_brier_score",
        "test_ece_15",
        "mean_epoch_time_sec",
        "train_time_sec",
        "inference_ms_per_image",
        "throughput_images_sec",
        "peak_training_gpu_memory_mb",
        "peak_inference_gpu_memory_mb",
        "lgts_alpha",
    )
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], row["activation_normalized"])].append(row)
    summaries = []
    for key in sorted(groups, key=lambda item: (DATASETS.index(item[0]), MODELS.index(item[1]), ACTIVATIONS.index(item[2]))):
        items = groups[key]
        out: dict[str, object] = {
            "dataset": key[0],
            "model": key[1],
            "activation": key[2],
            "n_seeds": len(items),
        }
        for metric in metrics:
            values = [value for item in items if (value := finite_float(item, metric)) is not None]
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        summaries.append(out)
    return summaries


def block_scores(by_key: dict[tuple, dict[str, str]]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for model in MODELS:
            row: dict[str, object] = {"dataset": dataset, "model": model}
            scores = []
            for activation in ACTIVATIONS:
                values = [
                    float(by_key[(dataset, model, activation, seed)]["test_f1_macro"])
                    for seed in SEEDS
                ]
                value = mean(values)
                row[activation] = value
                scores.append(value)
            ranks = rankdata(-np.asarray(scores), method="average")
            for activation, rank in zip(ACTIVATIONS, ranks):
                row[f"rank_{activation}"] = float(rank)
            output.append(row)
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for index, original_index in enumerate(order):
        candidate = min(1.0, (total - index) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def bootstrap_mean_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(differences, size=(BOOTSTRAP_REPEATS, len(differences)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def statistical_analysis(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    arrays = [np.asarray([float(block[activation]) for block in blocks]) for activation in ACTIVATIONS]
    statistic, p_value = friedmanchisquare(*arrays)
    average_ranks = {
        activation: mean(float(block[f"rank_{activation}"]) for block in blocks)
        for activation in ACTIVATIONS
    }
    omnibus = [{
        "test": "Friedman",
        "metric": "mean test macro-F1 over five seeds",
        "n_dataset_model_blocks": len(blocks),
        "n_activations": len(ACTIVATIONS),
        "statistic": float(statistic),
        "p_value": float(p_value),
        **{f"average_rank_{activation}": average_ranks[activation] for activation in ACTIVATIONS},
    }]

    primary_values = np.asarray([float(block[PRIMARY]) for block in blocks])
    pairwise = []
    raw_p = []
    for comparator in ACTIVATIONS:
        if comparator == PRIMARY:
            continue
        comparator_values = np.asarray([float(block[comparator]) for block in blocks])
        differences = primary_values - comparator_values
        try:
            test = wilcoxon(differences, zero_method="zsplit", alternative="two-sided", method="auto")
            statistic_value, p = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic_value, p = 0.0, 1.0
        low, high = bootstrap_mean_ci(differences)
        tolerance = 1e-12
        pairwise.append({
            "primary": PRIMARY,
            "comparator": comparator,
            "n_dataset_model_blocks": len(differences),
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "wins": int(np.sum(differences > tolerance)),
            "ties": int(np.sum(np.abs(differences) <= tolerance)),
            "losses": int(np.sum(differences < -tolerance)),
            "wilcoxon_statistic": statistic_value,
            "p_value_raw": p,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "noninferior": bool(low > -NONINFERIORITY_MARGIN),
        })
        raw_p.append(p)
    for row, adjusted in zip(pairwise, holm_adjust(raw_p)):
        row["p_value_holm"] = adjusted
        row["superiority_supported"] = bool(
            row["mean_paired_difference"] > 0 and adjusted < 0.05
        )
    return omnibus, pairwise


def make_plots(output_dir: Path, summaries: list[dict], blocks: list[dict], rows: list[dict[str, str]]) -> None:
    plot_dir = output_dir / "q2_confirmatory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "LGTS-f4-Alpha": "#D62728",
        "LGTS-f2-Slope": "#FF7F0E",
        "ACON-C": "#1F77B4",
        "GELU": "#2CA02C",
        "Native": "#7F7F7F",
    }

    labels = [f"{block['dataset']}\n{block['model']}" for block in blocks]
    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(14, 6))
    for activation in ACTIVATIONS:
        ax.plot(x, [block[activation] for block in blocks], marker="o", label=activation, color=colors[activation])
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Mean official-test macro-F1 (5 seeds)")
    ax.set_title("Q2 confirmatory performance by dataset–model block")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(plot_dir / "macro_f1_by_block.png", dpi=220)
    plt.close(fig)

    ranks = [mean(float(block[f"rank_{activation}"]) for block in blocks) for activation in ACTIVATIONS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ACTIVATIONS, ranks, color=[colors[item] for item in ACTIVATIONS])
    ax.set_ylabel("Average rank (lower is better)")
    ax.set_title("Confirmatory average activation rank")
    ax.set_ylim(0.8, max(ranks) + 0.5)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "average_rank.png", dpi=220)
    plt.close(fig)

    activation_groups: dict[str, list[dict]] = defaultdict(list)
    for item in summaries:
        activation_groups[str(item["activation"])].append(item)
    fig, ax = plt.subplots(figsize=(8, 6))
    for activation in ACTIVATIONS:
        items = activation_groups[activation]
        score = mean(float(item["test_f1_macro_mean"]) for item in items)
        epoch = mean(float(item["mean_epoch_time_sec_mean"]) for item in items)
        ax.scatter(epoch, score, s=90, color=colors[activation], label=activation)
        ax.annotate(activation, (epoch, score), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean epoch time (s)")
    ax.set_ylabel("Mean official-test macro-F1")
    ax.set_title("Accuracy–efficiency trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "performance_efficiency.png", dpi=220)
    plt.close(fig)

    alpha_by_dataset = {
        dataset: [
            value
            for row in rows
            if row["activation_normalized"] == PRIMARY and row["dataset"] == dataset
            if (value := finite_float(row, "lgts_alpha")) is not None
        ]
        for dataset in DATASETS
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([alpha_by_dataset[item] for item in DATASETS], tick_labels=DATASETS, showmeans=True)
    ax.axhline(0.4, color="#555555", linestyle="--", linewidth=1, label="initial alpha=0.40")
    ax.set_ylabel("Learned layer-mean alpha")
    ax.set_title("LGTS-f4-Alpha adaptation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "learned_alpha.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked Q2 confirmatory integrity and statistics")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/q2_confirmatory"))
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help="Optional public run-level CSV; otherwise exp_*/results.csv files are discovered under output_dir",
    )
    parser.add_argument("--expected_runs", type=int, default=225)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.output_dir, args.input_csv)
    integrity, by_key = integrity_report(rows, args.expected_runs)
    (args.output_dir / "q2_integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not integrity["complete"]:
        print(
            "INCOMPLETE | "
            f"unique={integrity['unique_locked_rows']}/{integrity['locked_expected_runs']} | "
            f"missing={integrity['missing_count']} | duplicates={integrity['duplicate_count']} | "
            f"unexpected={integrity['unexpected_count']}"
        )
        raise SystemExit(2)

    locked_rows = list(by_key.values())
    summaries = condition_summary(locked_rows)
    blocks = block_scores(by_key)
    omnibus, pairwise = statistical_analysis(blocks)
    write_csv(args.output_dir / "q2_summary_by_condition.csv", summaries)
    write_csv(args.output_dir / "q2_dataset_model_blocks.csv", blocks)
    write_csv(args.output_dir / "q2_friedman.csv", omnibus)
    write_csv(args.output_dir / "q2_pairwise_holm.csv", pairwise)
    make_plots(args.output_dir, summaries, blocks, locked_rows)
    print("Q2 ANALYSIS COMPLETE | 225/225 unique official-test runs verified")


if __name__ == "__main__":
    main()
