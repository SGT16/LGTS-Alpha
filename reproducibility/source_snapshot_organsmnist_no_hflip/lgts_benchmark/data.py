from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split


DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "fashionmnist": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "svhn": "svhn",
    "beans": "beans",
    "pathmnist": "pathmnist",
    "dermamnist": "dermamnist",
    "organsmnist": "organsmnist",
    "organ_s_mnist": "organsmnist",
    "organ_smnist": "organsmnist",
    "imagenet100": "imagenet100",
    "imagenet_100": "imagenet100",
}

HOLDOUT_SEED = 2026
IMAGENET100_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"
MEDMNIST_2D_DATASETS = frozenset({"pathmnist", "dermamnist", "organsmnist"})
AUGMENTATION_PROFILES = frozenset({"none", "standard", "organsmnist_no_hflip"})
MEDMNIST_MD5 = {
    "pathmnist": {64: "55aa9c1e0525abe5a6b9d8343a507616"},
    "dermamnist": {64: "b70a2f5635c6199aeaa28c31d7202e1f"},
    "organsmnist": {64: "53a6d115339d874c25e309a994ff46d3"},
}


def declared_dataset_contract() -> dict[str, Any]:
    """Immutable upstream identifiers included in every experiment identity."""
    return {
        "holdout_seed": HOLDOUT_SEED,
        "imagenet100": {
            "repo_id": "clane9/imagenet-100",
            "revision": IMAGENET100_REVISION,
            "subset": "CMC ImageNet-100",
        },
        "medmnist_md5": MEDMNIST_MD5,
        "torchvision": {"cifar10": "CIFAR10", "cifar100": "CIFAR100"},
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DatasetBundle:
    name: str
    train: Dataset
    val: Dataset
    test: Dataset
    labels_train: np.ndarray
    labels_val: np.ndarray
    labels_test: np.ndarray
    class_names: list[str]
    in_channels: int
    native_size: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train_val(self) -> Dataset:
        return ConcatDataset([self.train, self.val])

    @property
    def labels_train_val(self) -> np.ndarray:
        return np.concatenate([self.labels_train, self.labels_val])


def canonical_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(set(DATASET_ALIASES.values())))}")
    return DATASET_ALIASES[key]


def _targets(dataset: Any) -> np.ndarray:
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            return np.asarray(values).reshape(-1).astype(int)
    return np.asarray([int(np.asarray(dataset[i][1]).squeeze()) for i in range(len(dataset))], dtype=int)


class MedMNISTView(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, int(np.asarray(label).squeeze())


class HFBeansView(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        item = self.split[index]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return image.convert("RGB"), int(item["labels"])


class HFImageView(Dataset):
    """Pickle-safe view over a Hugging Face image-classification split."""

    def __init__(self, split, image_key: str = "image", label_key: str = "label"):
        self.split = split
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        item = self.split[index]
        image = item[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return image.convert("RGB"), int(item[self.label_key])


def _stratified_indices(labels: np.ndarray, validation_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=HOLDOUT_SEED,
        shuffle=True,
        stratify=labels,
    )
    return np.asarray(train_indices), np.asarray(validation_indices)


def load_bundle(name: str, root: str | Path, med_size: int = 64, download: bool = True) -> DatasetBundle:
    name = canonical_name(name)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if name == "cifar10":
        train_all = tv_datasets.CIFAR10(root / "torchvision", train=True, download=download)
        test = tv_datasets.CIFAR10(root / "torchvision", train=False, download=download)
        # The official dataset has no validation split. Keep an auditable 90/10
        # HOLDOUT_SEED=2026 defines the fixed stratified validation split.
        labels = _targets(train_all)
        train_idx, val_idx = _stratified_indices(labels)
        train = torch.utils.data.Subset(train_all, train_idx.tolist())
        val = torch.utils.data.Subset(train_all, val_idx.tolist())
        archive = root / "torchvision" / "cifar-10-python.tar.gz"
        metadata = {
            "train_indices_sha256": _array_sha256(train_idx),
            "validation_indices_sha256": _array_sha256(val_idx),
            "archive_sha256": _file_hash(archive, "sha256") if archive.exists() else "archive_not_present",
        }
        return DatasetBundle(name, train, val, test, labels[train_idx], labels[val_idx], _targets(test), list(train_all.classes), 3, 32, "torchvision/CIFAR10", metadata)
    if name == "cifar100":
        train_all = tv_datasets.CIFAR100(root / "torchvision", train=True, download=download)
        test = tv_datasets.CIFAR100(root / "torchvision", train=False, download=download)
        labels = _targets(train_all)
        train_idx, val_idx = _stratified_indices(labels)
        train = torch.utils.data.Subset(train_all, train_idx.tolist()); val = torch.utils.data.Subset(train_all, val_idx.tolist())
        archive = root / "torchvision" / "cifar-100-python.tar.gz"
        metadata = {
            "train_indices_sha256": _array_sha256(train_idx),
            "validation_indices_sha256": _array_sha256(val_idx),
            "archive_sha256": _file_hash(archive, "sha256") if archive.exists() else "archive_not_present",
        }
        return DatasetBundle(name, train, val, test, labels[train_idx], labels[val_idx], _targets(test), list(train_all.classes), 3, 32, "torchvision/CIFAR100", metadata)
    if name == "fashion_mnist":
        train_all = tv_datasets.FashionMNIST(root / "torchvision", train=True, download=download)
        test = tv_datasets.FashionMNIST(root / "torchvision", train=False, download=download)
        rng = np.random.default_rng(42); idx = rng.permutation(len(train_all)); cut = int(0.9 * len(idx))
        train = torch.utils.data.Subset(train_all, idx[:cut].tolist()); val = torch.utils.data.Subset(train_all, idx[cut:].tolist())
        labels = _targets(train_all)
        return DatasetBundle(name, train, val, test, labels[idx[:cut]], labels[idx[cut:]], _targets(test), list(train_all.classes), 1, 28, "torchvision/FashionMNIST")
    if name == "svhn":
        train_all = tv_datasets.SVHN(root / "torchvision", split="train", download=download)
        test = tv_datasets.SVHN(root / "torchvision", split="test", download=download)
        rng = np.random.default_rng(42); idx = rng.permutation(len(train_all)); cut = int(0.9 * len(idx))
        train = torch.utils.data.Subset(train_all, idx[:cut].tolist()); val = torch.utils.data.Subset(train_all, idx[cut:].tolist())
        labels = _targets(train_all)
        return DatasetBundle(name, train, val, test, labels[idx[:cut]], labels[idx[cut:]], _targets(test), [str(i) for i in range(10)], 3, 32, "torchvision/SVHN")
    if name == "beans":
        from datasets import load_dataset
        ds = load_dataset("AI-Lab-Makerere/beans", cache_dir=str(root / "huggingface"))
        class_names = list(ds["train"].features["labels"].names)
        tr, va, te = HFBeansView(ds["train"]), HFBeansView(ds["validation"]), HFBeansView(ds["test"])
        return DatasetBundle(name, tr, va, te, _targets(tr), _targets(va), _targets(te), class_names, 3, 160, "AI-Lab-Makerere/beans")
    if name in MEDMNIST_2D_DATASETS:
        import medmnist
        from medmnist import INFO
        flag = name
        info = INFO[flag]
        cls = getattr(medmnist, info["python_class"])
        med_root = root / "medmnist"
        med_root.mkdir(parents=True, exist_ok=True)
        kwargs = {"root": str(med_root), "download": download, "size": med_size}
        tr = MedMNISTView(cls(split="train", **kwargs)); va = MedMNISTView(cls(split="val", **kwargs)); te = MedMNISTView(cls(split="test", **kwargs))
        class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
        npz_path = med_root / f"{flag}_{med_size}.npz"
        actual_md5 = _file_hash(npz_path, "md5") if npz_path.exists() else "missing"
        expected_md5 = MEDMNIST_MD5.get(flag, {}).get(med_size)
        if expected_md5 is not None and actual_md5 != expected_md5:
            raise RuntimeError(f"{npz_path.name} MD5 mismatch: expected {expected_md5}, got {actual_md5}")
        metadata = {
            "file": npz_path.name,
            "md5": actual_md5,
            "expected_md5": expected_md5 or "not_registered",
            "medmnist_version": getattr(medmnist, "__version__", "unknown"),
            "task": info.get("task", "unknown"),
            "license": info.get("license", "unknown"),
        }
        return DatasetBundle(
            name, tr, va, te,
            _targets(tr), _targets(va), _targets(te),
            class_names, int(info.get("n_channels", 3)), med_size,
            f"MedMNIST/{flag}", metadata,
        )
    if name == "imagenet100":
        from datasets import DownloadConfig, load_dataset
        dataset_id = "clane9/imagenet-100"
        download_config = DownloadConfig(max_retries=20)
        ds = load_dataset(
            dataset_id,
            revision=IMAGENET100_REVISION,
            cache_dir=str(root / "huggingface"),
            download_config=download_config,
        )
        if "train" not in ds or "validation" not in ds:
            raise RuntimeError(f"{dataset_id} must provide train and validation splits.")
        labels = np.asarray(ds["train"]["label"], dtype=int)
        train_idx, val_idx = _stratified_indices(labels)
        train_split = ds["train"].select(train_idx.tolist())
        val_split = ds["train"].select(val_idx.tolist())
        test_split = ds["validation"]
        label_feature = ds["train"].features["label"]
        class_names = list(label_feature.names) if getattr(label_feature, "names", None) else [str(i) for i in range(100)]
        from huggingface_hub import hf_hub_download
        classes_file = hf_hub_download(
            repo_id=dataset_id, repo_type="dataset", filename="scripts/classes.py",
            revision=IMAGENET100_REVISION, cache_dir=str(root / "huggingface_hub"),
        )
        synset_ids = re.findall(r'"(n\d{8})"\s*:', Path(classes_file).read_text(encoding="utf-8"))
        if len(synset_ids) != len(class_names):
            raise RuntimeError(f"Expected {len(class_names)} ImageNet synsets, found {len(synset_ids)}")
        train = HFImageView(train_split)
        val = HFImageView(val_split)
        test = HFImageView(test_split)
        metadata = {
            "huggingface_revision": IMAGENET100_REVISION,
            "train_split_fingerprint": getattr(ds["train"], "_fingerprint", "unavailable"),
            "validation_split_fingerprint": getattr(ds["validation"], "_fingerprint", "unavailable"),
            "train_indices_sha256": _array_sha256(train_idx),
            "validation_indices_sha256": _array_sha256(val_idx),
            "synset_ids": synset_ids,
            "source_resolution_contract": "short-side resized to 160 pixels by upstream CMC distribution",
        }
        return DatasetBundle(
            name, train, val, test,
            labels[train_idx], labels[val_idx], np.asarray(test_split["label"], dtype=int),
            class_names, 3, 160, f"HuggingFace/{dataset_id}; CMC ImageNet-100 subset",
            metadata,
        )
    raise AssertionError(name)


class TransformSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray | list[int], transform):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[int(self.indices[index])]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return self.transform(image), int(np.asarray(label).squeeze())


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return self.transform(image), int(np.asarray(label).squeeze())


class ConvertToRGB:
    """Pickle-safe RGB conversion for Windows DataLoader workers."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def normalization_stats(dataset_name: str, in_channels: int, profile: str = "dataset") -> tuple[list[float], list[float]] | None:
    if profile == "none":
        return None
    if in_channels == 1:
        return ([0.5], [0.5])
    if profile == "imagenet" or dataset_name == "imagenet100":
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if dataset_name == "cifar10":
        return ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    if dataset_name == "cifar100":
        return ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    # MedMNIST's reference preprocessing maps each RGB channel to [-1, 1].
    return ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def build_transforms(
    dataset_name: str,
    image_size: int,
    in_channels: int,
    augmentation: str,
    normalization: str = "dataset",
):
    if augmentation not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unknown augmentation profile '{augmentation}'. "
            f"Available: {', '.join(sorted(AUGMENTATION_PROFILES))}"
        )
    if augmentation == "organsmnist_no_hflip" and dataset_name != "organsmnist":
        raise ValueError(
            "The organsmnist_no_hflip profile is restricted to OrganSMNIST "
            "post-confirmatory sensitivity runs."
        )
    convert = [transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else [ConvertToRGB()]
    if dataset_name == "imagenet100":
        if augmentation == "standard":
            train_spatial = [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            train_spatial = [transforms.Resize((image_size, image_size), antialias=True)]
        eval_size = int(round(image_size / 0.875))
        eval_spatial = [transforms.Resize(eval_size, antialias=True), transforms.CenterCrop(image_size)]
    elif dataset_name in {"cifar10", "cifar100"}:
        train_spatial = []
        if image_size != 32:
            train_spatial.append(transforms.Resize((image_size, image_size), antialias=True))
        if augmentation == "standard":
            train_spatial += [transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()]
        eval_spatial = [] if image_size == 32 else [transforms.Resize((image_size, image_size), antialias=True)]
    elif dataset_name in MEDMNIST_2D_DATASETS:
        train_spatial = [transforms.Resize((image_size, image_size), antialias=True)]
        if augmentation in {"standard", "organsmnist_no_hflip"}:
            # The post-confirmatory OrganSMNIST profile changes exactly one
            # factor: horizontal flipping is omitted because it exchanges
            # left/right anatomy while retaining the original class label.
            if augmentation == "standard":
                train_spatial.append(transforms.RandomHorizontalFlip())
            train_spatial += [transforms.RandomVerticalFlip(), transforms.RandomRotation(15)]
        eval_spatial = [transforms.Resize((image_size, image_size), antialias=True)]
    else:
        train_spatial = [transforms.Resize((image_size, image_size), antialias=True)]
        eval_spatial = list(train_spatial)
    tensor_ops: list[Any] = [transforms.ToTensor()]
    stats = normalization_stats(dataset_name, in_channels, normalization)
    if stats is not None:
        tensor_ops.append(transforms.Normalize(*stats))
    train = transforms.Compose(convert + train_spatial + tensor_ops)
    evaluation = transforms.Compose(convert + eval_spatial + tensor_ops)
    return train, evaluation


def dataset_manifest(bundle: DatasetBundle) -> dict[str, Any]:
    upstream_metadata = dict(bundle.metadata)
    upstream_metadata.pop("dataset_fingerprint_sha256", None)
    manifest = {
        "dataset": bundle.name,
        "source": bundle.source,
        "classes": bundle.class_names,
        "in_channels": bundle.in_channels,
        "native_size": bundle.native_size,
        "train": len(bundle.train),
        "validation": len(bundle.val),
        "test": len(bundle.test),
        "train_class_counts": np.bincount(bundle.labels_train, minlength=len(bundle.class_names)).tolist(),
        "validation_class_counts": np.bincount(bundle.labels_val, minlength=len(bundle.class_names)).tolist(),
        "test_class_counts": np.bincount(bundle.labels_test, minlength=len(bundle.class_names)).tolist(),
        "recommended_normalization": normalization_stats(bundle.name, bundle.in_channels, "dataset"),
        "holdout_seed": HOLDOUT_SEED,
        "train_labels_sha256": _array_sha256(bundle.labels_train),
        "validation_labels_sha256": _array_sha256(bundle.labels_val),
        "test_labels_sha256": _array_sha256(bundle.labels_test),
        "upstream_metadata": upstream_metadata,
    }
    fingerprint_payload = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    manifest["dataset_fingerprint_sha256"] = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    bundle.metadata["dataset_fingerprint_sha256"] = manifest["dataset_fingerprint_sha256"]
    return manifest
