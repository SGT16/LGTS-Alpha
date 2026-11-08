import argparse
import csv
import gzip
import json
import math
from itertools import groupby, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from study_data import SEEDS, rows, write_csv


ACTIVATIONS = ("LGTS-f4-Alpha", "ReLU")
LAYERS = tuple(f"layer{i}.1.conv2.weight" for i in range(1, 5))
RECIPE = ("code_sha256", "dataset_fingerprint_sha256", "batch_size", "optimizer",
          "learning_rate_initial", "weight_decay")


def read_run(source, records):
    observations = {}
    sizes = {}
    for row in records:
        if (row["batch_index"] != "0" or row["sampling"] != "first_training_minibatch_per_epoch"
                or row["gradient_scaling"] != "unscaled"):
            raise ValueError(f"Unexpected gradient sampling in {source}")
        name = row["parameter"]
        key = int(row["epoch"]), name
        size = int(row["numel"])
        if key in observations or sizes.setdefault(name, size) != size:
            raise ValueError(f"Duplicate parameter or inconsistent dimensions for {key}")
        value = float(row["gradient_l2"]) if row["gradient_finite"].lower() == "true" else math.nan
        if row["gradient_finite"].lower() == "true" and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid recorded finite norm for {key}")
        observations[key] = value
    if not sizes or observations.keys() != set(product(range(1, 101), sizes)):
        raise ValueError(f"Expected every trainable parameter in every epoch of {source}")
    return observations, sizes


def engine_runs(directory):
    runs = {}
    protocols = set()
    for path in sorted(directory.rglob("done.json")):
        meta = json.loads(path.read_bytes())
        if (meta["dataset"], meta["model"], meta["activation"]) not in {
            ("cifar100", "ResNet18", a) for a in ACTIVATIONS
        }:
            continue
        if (meta.get("protocol"), meta.get("augmentation"), meta.get("activation_sharing"),
                meta.get("epochs_run"), meta.get("evaluation_split")) != ("full_train", "standard", "site", 100, "test"):
            continue
        key = meta["activation"], int(meta["seed"])
        if key in runs:
            raise ValueError(f"Multiple gradient runs for {key}")
        protocols.add(tuple(meta[k] for k in RECIPE))
        source = path.with_name("gradient_norms.csv")
        runs[key] = read_run(source, rows(source))
    if len(protocols) != 1:
        raise ValueError("Expected a single matched training recipe")
    return runs, {}


def archived_runs(directory):
    source = directory / "diagnostics" / "gradient_parameters.csv.gz"
    identities = {}
    protocols = set()
    for row in rows(directory / "run_settings.csv"):
        if (row["panel"], row["dataset"], row["model"]) != ("components_baselines", "cifar100", "ResNet18"):
            continue
        if row["activation"] not in ACTIVATIONS:
            continue
        if tuple(row[k] for k in ("protocol", "augmentation", "activation_sharing", "epochs", "evaluation_split")) != (
                "full_train", "standard", "site", "100", "test"):
            raise ValueError(f"Unexpected gradient training conditions for run {row['run_id']}")
        identities[row["run_id"]] = row["activation"], int(row["seed"])
        protocols.add(tuple(row[k] for k in RECIPE))
    if len(protocols) != 1 or len(identities) != len(set(identities.values())):
        raise ValueError("Invalid training recipe or repeated seed in the gradient cohort")
    runs = {}
    with gzip.open(source, "rt", encoding="utf-8", newline="") as archive:
        for run_id, group in groupby(csv.DictReader(archive), key=lambda r: r["run_id"]):
            key = identities[run_id]
            records = list(group)
            if key in runs or any((r["activation"], int(r["seed"])) != key for r in records):
                raise ValueError(f"Unexpected group identity for run {run_id}")
            runs[key] = read_run(run_id, records)
    if len(runs) != len(identities):
        raise ValueError("The archived gradient logs and training settings have different run IDs")
    return runs, {key: run_id for run_id, key in identities.items()}


def plot_profiles(results, output):
    epochs = np.arange(1, 101)
    values = {(r["activation"], r["seed"], r["epoch"]): r for r in results}
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), layout="constrained", sharex=True, sharey=True)
    for layer, ax in zip(LAYERS, axes.flat):
        for act, label, color in zip(ACTIVATIONS, ("LGTS-Alpha", "ReLU"), ("#2166a1", "#b75428")):
            norms = np.array([[values[act, seed, epoch][layer] for epoch in epochs] for seed in SEEDS])
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError(f"Logarithmic gradient plot requires positive finite observations for {layer}")
            ax.fill_between(epochs, norms.min(axis=0), norms.max(axis=0), color=color, alpha=.13)
            ax.plot(epochs, norms.mean(axis=0), color=color, lw=1.3, label=label)
        ax.set(yscale="log", title=layer, xlim=(1, 100))
        ax.grid(axis="y", alpha=.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("CIFAR-100 · ResNet18 · first training minibatch of each epoch\n"
                 "Five paired seeds · lines show means and shading spans the observed range", fontsize=11)
    fig.supxlabel("Epoch")
    fig.supylabel("Unscaled gradient L2 norm")
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100/ResNet18 gradient profiles for five paired seeds.")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    sources.add_argument("--runs-dir", type=Path, help="Read original training-engine run directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    runs, run_ids = engine_runs(args.runs_dir) if args.runs_dir else archived_runs(args.results_dir)
    if runs.keys() != set(product(ACTIVATIONS, SEEDS)):
        raise ValueError("Expected two matched sets of five runs under one training recipe")
    shared = set.intersection(*(set(sizes) for _, sizes in runs.values()))
    if not set(LAYERS) <= shared or any("raw_alpha" in name for name in shared):
        raise ValueError("The shared parameter set does not match the two ResNet18 architectures")
    ordered = [name for name in runs[ACTIVATIONS[0], SEEDS[0]][1] if name in shared]
    if any(len({sizes[name] for _, sizes in runs.values()}) != 1 for name in ordered):
        raise ValueError("Mismatched parameter shapes")
    result = []
    for (act, seed), (obs, _) in sorted(runs.items()):
        for epoch in range(1, 101):
            norm = float(np.sqrt(np.sum(np.square([obs[epoch, name] for name in ordered]))))
            row = {"run_id": run_ids.get((act, seed), ""), "activation": act, "seed": seed,
                   "epoch": epoch, "global_l2_shared": norm}
            row.update((name, obs[epoch, name]) for name in LAYERS)
            result.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gradient_norms_by_epoch.csv", result)
    plot_profiles(result, args.output_dir / "gradient_profiles.svg")


if __name__ == "__main__":
    main()
