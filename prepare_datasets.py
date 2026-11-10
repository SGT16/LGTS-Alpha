from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lgts_benchmark.data import dataset_manifest, load_bundle


def merge_dataset_manifests(target: Path, updates: list[dict]) -> list[dict]:
    existing: list[dict] = []
    if target.exists():
        loaded = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(item, dict) and "dataset" in item for item in loaded):
            raise ValueError(f"Invalid dataset manifest format: {target}")
        existing = loaded
    by_dataset = {item["dataset"]: item for item in existing}
    for item in updates:
        by_dataset[item["dataset"]] = item
    merged = list(by_dataset.values())
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description="Download and verify benchmark datasets before training")
    p.add_argument("--datasets", nargs="+", default=["cifar100", "dermamnist", "organsmnist"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--med_size", type=int, default=64, choices=[28, 64, 128, 224])
    args = p.parse_args()
    root = Path(args.data_dir); root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for name in args.datasets:
        print(f"Preparing | {name}", flush=True)
        bundle = load_bundle(name, root, args.med_size, download=True)
        manifest = dataset_manifest(bundle); manifests.append(manifest)
        print(f"Prepared | {bundle.name} | train={manifest['train']} | validation={manifest['validation']} | test={manifest['test']} | total={manifest['train'] + manifest['validation'] + manifest['test']} | source={manifest['source']}")
    target = root / "dataset_manifest.json"
    merged = merge_dataset_manifests(target, manifests)
    print(f"Manifest | {target.resolve()}")
    print(f"Manifest datasets | {', '.join(item['dataset'] for item in merged)}")


if __name__ == "__main__":
    main()
