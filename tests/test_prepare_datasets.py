import json
import tempfile
import unittest
from pathlib import Path

from prepare_datasets import merge_dataset_manifests


class DatasetManifestMergeTests(unittest.TestCase):
    def test_separate_preparation_preserves_existing_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset_manifest.json"
            target.write_text(
                json.dumps([{"dataset": "cifar100", "fingerprint": "old"}]),
                encoding="utf-8",
            )
            merged = merge_dataset_manifests(
                target,
                [{"dataset": "imagenet100", "fingerprint": "new"}],
            )
            self.assertEqual([item["dataset"] for item in merged], ["cifar100", "imagenet100"])
            persisted = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(persisted, merged)

    def test_reprepared_dataset_replaces_only_its_own_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset_manifest.json"
            target.write_text(
                json.dumps([
                    {"dataset": "cifar100", "fingerprint": "v1"},
                    {"dataset": "pathmnist", "fingerprint": "keep"},
                ]),
                encoding="utf-8",
            )
            merged = merge_dataset_manifests(
                target,
                [{"dataset": "cifar100", "fingerprint": "v2"}],
            )
            self.assertEqual(merged, [
                {"dataset": "cifar100", "fingerprint": "v2"},
                {"dataset": "pathmnist", "fingerprint": "keep"},
            ])


if __name__ == "__main__":
    unittest.main()
