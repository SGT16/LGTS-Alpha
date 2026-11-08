import argparse
import csv
from pathlib import Path


MODELS = ("ResNet18", "MobileNetV2", "EfficientNetV2S")
SEEDS = (936137, 529212, 448132, 942529, 110107)
DATASETS = ("cifar100", "dermamnist", "organsmnist")
ACTIVATIONS = ("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def indexed(path):
    indexed_rows = {}
    for row in rows(path):
        key = row["run_id"]
        if not key or key in indexed_rows:
            raise ValueError(f"Missing or duplicate run ID in {path.name}: {key}")
        indexed_rows[key] = row
    return indexed_rows


def load_runs(directory):
    settings = indexed(directory / "run_settings.csv")
    scores = indexed(directory / "seed_results.csv")
    if settings.keys() != scores.keys():
        raise ValueError("Training settings and outcomes do not cover the same run IDs")
    for key, run in settings.items():
        outcome = scores[key]
        if any(run[field] != outcome[field] for field in ("panel", "condition_id", "seed", "fold")):
            raise ValueError(f"Condition or pairing differs for run {key}")
        yield run, outcome


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
