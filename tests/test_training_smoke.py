import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lgts_benchmark.data import DatasetBundle
from lgts_benchmark import __version__
from train_benchmark import is_duplicate_control, parse_args, resolve_class_weight_mode, train_one


class TinyImages(Dataset):
    def __init__(self, count: int):
        self.labels = np.arange(count, dtype=int) % 2
        rng = np.random.default_rng(7)
        self.images = rng.integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return Image.fromarray(self.images[index]), int(self.labels[index])


class TrainingPipelineSmokeTests(unittest.TestCase):
    @staticmethod
    def make_args(root: Path, protocol: str) -> SimpleNamespace:
        return SimpleNamespace(
            activation_scope="full_backbone", protocol=protocol, output_dir=str(root),
            rerun_done=False, image_size=32, imagenet_size=160, med_size=64,
            augmentation="none", normalization="dataset", small_input_stem="auto",
            batch_size=4, med_batch_size=4, imagenet_batch_size=4,
            num_workers=0, prefetch_factor=2, class_weight="none",
            performance_mode=False, cudnn_benchmark=False, channels_last=False,
            log_t=0.6, lgts_fixed=False, lgts_beta=1.0,
            lgts_alpha_min=0.05, lgts_alpha_max=1.5,
            lgts_beta_min=0.5, lgts_beta_max=3.0,
            no_pretrained=True, optimizer="adamw", fused_optimizer=False,
            learning_rate=1e-3, weight_decay=1e-4, momentum=0.9,
            scheduler="none", epochs=1, min_learning_rate=1e-6,
            no_amp=True, min_delta=0.0, patience=0, min_epochs=0,
            best_state_device="device", plots="none", tf32=False,
            seeds=[42, 123], seed_order=None,
            live_workbook_path=str(root / "live_results.xlsx"),
            experiment_id="smoke-exp", code_sha256="abc123", git_commit="unavailable",
            runtime_metadata={},
        )

    def test_fixed_protocol_seals_test_and_writes_validation_logits(self):
        train, val, test = TinyImages(20), TinyImages(8), TinyImages(8)
        bundle = DatasetBundle(
            "smoke", train, val, test,
            train.labels, val.labels, test.labels,
            ["zero", "one"], 3, 32, "synthetic",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, "fixed")
            train_one(
                args, bundle, "PaperCNN", "LGTS-f2-Slope", 0.4,
                seed=42, fold_no=1, repeat_no=1,
                train_idx=None, val_idx=None, device=torch.device("cpu"),
            )
            done = next(root.rglob("done.json"))
            predictions = next(root.rglob("validation_predictions.npz"))
            self.assertTrue(done.exists())
            self.assertFalse(any(root.rglob("test_predictions.npz")))
            with np.load(predictions) as archive:
                self.assertIn("logits", archive.files)
            with (root / "results.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["protocol"], "fixed")
            self.assertEqual(row["evaluation_split"], "validation")
            self.assertEqual(row["nll_source"], "stable_logits_float64")
            self.assertAlmostEqual(float(row["validation_nll"]), float(row["validation_loss"]), places=5)
            self.assertNotIn("test_f1_macro", row)
            self.assertEqual(row["test_count"], "sealed")
            self.assertEqual(row["training_seed"], "42")
            self.assertEqual(row["experiment_id"], "smoke-exp")
            self.assertIn("validation_top5_accuracy", row)
            self.assertIn("peak_training_gpu_memory_mb", row)
            self.assertIn("peak_inference_gpu_memory_mb", row)
            self.assertTrue((root / "live_results.xlsx").exists())
            self.assertFalse(any(root.rglob("in_progress.json")))

    def test_full_train_uses_train_and_validation_without_model_selection(self):
        train, val, test = TinyImages(20), TinyImages(8), TinyImages(8)
        bundle = DatasetBundle(
            "smoke", train, val, test,
            train.labels, val.labels, test.labels,
            ["zero", "one"], 3, 32, "synthetic",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, "full_train")
            train_one(
                args, bundle, "PaperCNN", "LGTS-f2-Slope", 0.4,
                seed=123, fold_no=1, repeat_no=1,
                train_idx=None, val_idx=None, device=torch.device("cpu"),
            )
            with (root / "results.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["protocol"], "full_train")
            self.assertEqual(row["train_count_before_aug"], "28")
            self.assertEqual(row["validation_count"], "0")
            self.assertEqual(row["best_validation_f1"], "")
            self.assertEqual(row["selection_policy"], "fixed_budget_final_epoch")
            self.assertEqual(row["evaluation_split"], "test")
            self.assertTrue(any(root.rglob("test_predictions.npz")))
            self.assertIn("test_f1_macro", row)

    def test_incomplete_fold_restarts_from_epoch_one_without_checkpoint(self):
        train, val, test = TinyImages(20), TinyImages(8), TinyImages(8)
        bundle = DatasetBundle(
            "smoke", train, val, test,
            train.labels, val.labels, test.labels,
            ["zero", "one"], 3, 32, "synthetic",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, "fixed")
            train_one(
                args, bundle, "PaperCNN", "LGTS-f2-Slope", 0.4,
                seed=42, fold_no=1, repeat_no=1,
                train_idx=None, val_idx=None, device=torch.device("cpu"),
            )
            run_dir = next(root.rglob("fixed"))
            (run_dir / "done.json").unlink()
            (run_dir / "history.json").write_text('[{"epoch": 99}]', encoding="utf-8")
            torch.save({"epoch": 99}, run_dir / "last.pt")
            train_one(
                args, bundle, "PaperCNN", "LGTS-f2-Slope", 0.4,
                seed=42, fold_no=1, repeat_no=1,
                train_idx=None, val_idx=None, device=torch.device("cpu"),
            )
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([item["epoch"] for item in history], [1])
            self.assertFalse((run_dir / "last.pt").exists())
            self.assertFalse((run_dir / "in_progress.json").exists())

    def test_locked_default_min_delta_and_package_version(self):
        with patch("sys.argv", ["train_benchmark.py"]):
            args = parse_args()
        self.assertEqual(args.min_delta, 0.0)
        self.assertFalse(args.shared_live_workbook)
        self.assertEqual(__version__, "2.2.0")

    def test_shared_live_workbook_is_explicit_opt_in(self):
        with patch("sys.argv", ["train_benchmark.py", "--shared_live_workbook"]):
            args = parse_args()
        self.assertTrue(args.shared_live_workbook)

    def test_auto_class_weight_is_dataset_specific(self):
        self.assertEqual(resolve_class_weight_mode("auto", "cifar100"), "none")
        self.assertEqual(resolve_class_weight_mode("auto", "imagenet100"), "none")
        self.assertEqual(resolve_class_weight_mode("auto", "pathmnist"), "balanced")
        self.assertEqual(resolve_class_weight_mode("auto", "dermamnist"), "balanced")

    def test_model_native_duplicates_are_skipped_by_default(self):
        activations = ["Native", "ReLU", "Swish"]
        self.assertTrue(is_duplicate_control("ResNet18", "ReLU", activations, keep=False))
        self.assertTrue(is_duplicate_control("EfficientNetV2S", "Swish", activations, keep=False))
        self.assertFalse(is_duplicate_control("MobileNetV2", "ReLU", activations, keep=False))


if __name__ == "__main__":
    unittest.main()
