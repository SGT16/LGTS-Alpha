import unittest

import numpy as np
from torchvision import transforms

from lgts_benchmark.data import (
    IMAGENET100_REVISION, MEDMNIST_2D_DATASETS, MEDMNIST_MD5,
    DatasetBundle, _stratified_indices, build_transforms, canonical_name,
    dataset_manifest, declared_dataset_contract,
)


class DataProtocolTests(unittest.TestCase):
    def test_cifar_standard_profile_uses_crop_flip_and_normalization(self):
        train, evaluation = build_transforms("cifar100", 32, 3, "standard", "dataset")
        self.assertTrue(any(isinstance(op, transforms.RandomCrop) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.RandomHorizontalFlip) for op in train.transforms))
        self.assertFalse(any(isinstance(op, transforms.RandomVerticalFlip) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.Normalize) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.Normalize) for op in evaluation.transforms))

    def test_imagenet100_profile_uses_160_crop_by_default_contract(self):
        train, evaluation = build_transforms("imagenet100", 160, 3, "standard", "dataset")
        crop = next(op for op in train.transforms if isinstance(op, transforms.RandomResizedCrop))
        self.assertEqual(crop.size, (160, 160))
        self.assertTrue(any(isinstance(op, transforms.CenterCrop) for op in evaluation.transforms))

    def test_fixed_holdout_is_stratified_and_deterministic(self):
        labels = np.repeat(np.arange(10), 100)
        train_a, val_a = _stratified_indices(labels)
        train_b, val_b = _stratified_indices(labels)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(val_a, val_b))
        self.assertTrue(np.array_equal(np.bincount(labels[val_a]), np.full(10, 10)))

    def test_dataset_manifest_fingerprint_is_idempotent(self):
        bundle = DatasetBundle(
            "demo", [0, 1], [2], [3],
            np.asarray([0, 1]), np.asarray([0]), np.asarray([1]),
            ["zero", "one"], 3, 32, "synthetic", {"revision": "abc"},
        )
        first = dataset_manifest(bundle)["dataset_fingerprint_sha256"]
        second = dataset_manifest(bundle)["dataset_fingerprint_sha256"]
        self.assertEqual(first, second)

    def test_imagenet_revision_is_pinned_in_contract(self):
        self.assertEqual(declared_dataset_contract()["imagenet100"]["revision"], IMAGENET100_REVISION)

    def test_organsmnist_alias_checksum_and_grayscale_profile(self):
        self.assertEqual(canonical_name("organ_s_mnist"), "organsmnist")
        self.assertIn("organsmnist", MEDMNIST_2D_DATASETS)
        self.assertEqual(MEDMNIST_MD5["organsmnist"][64], "53a6d115339d874c25e309a994ff46d3")
        train, evaluation = build_transforms("organsmnist", 64, 1, "standard", "dataset")
        self.assertTrue(any(isinstance(op, transforms.Grayscale) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.RandomHorizontalFlip) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.RandomVerticalFlip) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.Normalize) for op in evaluation.transforms))

    def test_organsmnist_sensitivity_profile_removes_only_horizontal_flip(self):
        train, _ = build_transforms(
            "organsmnist", 64, 1, "organsmnist_no_hflip", "dataset",
        )
        self.assertFalse(any(isinstance(op, transforms.RandomHorizontalFlip) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.RandomVerticalFlip) for op in train.transforms))
        self.assertTrue(any(isinstance(op, transforms.RandomRotation) for op in train.transforms))

    def test_organsmnist_sensitivity_profile_is_dataset_restricted(self):
        with self.assertRaises(ValueError):
            build_transforms("dermamnist", 64, 3, "organsmnist_no_hflip", "dataset")


if __name__ == "__main__":
    unittest.main()
