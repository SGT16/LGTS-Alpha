import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace
import real_epoch as epoch
from runtime import FILES, ROOT

CHUNK_BYTES = 1024 * 1024
TRAIN_IMAGES, BATCH, SIZE, SEED = 126689, 64, 160, 936137
FINGERPRINT = "1aef1aacf312a5e763363e93dfd79b1488fb3c3223c4c3ff58990e88b8cd9f3d"
def manifest_entries(manifest_path: Path) -> tuple[dict, list[dict], str]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    entries = manifest["files"]
    paths = set()
    for item in entries:
        name = item["path"]
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or "\\" in name
            or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or name in paths
        ):
            raise ValueError(f"Unsafe or duplicated manifest path: {name!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError(f"Invalid SHA256 for {name}")
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError(f"Invalid size for {name}")
        paths.add(name)
    if manifest["file_count"] != len(entries):
        raise ValueError("Manifest file count mismatch")
    if manifest["total_bytes"] != sum(item["size"] for item in entries):
        raise ValueError("Manifest total bytes mismatch")
    return manifest, entries, hashlib.sha256(raw).hexdigest()


def file_under(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"File escapes checked data root: {candidate}")
    return candidate


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            byte_count += len(chunk)

    return byte_count, digest.hexdigest()


def verify_file(path: Path, item: dict) -> None:
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise RuntimeError(f"Missing/non-file/wrong-size data file (not changed): {path}")
    size, sha = file_digest(path)
    if size != item["size"] or sha != item["sha256"]:
        raise RuntimeError(f"SHA256 mismatch (not changed): {path}")

def validate_unit(unit):
    epoch.require(unit["dataset"] == "imagenet100" and unit["seed"] == SEED
                  and unit["model"] in epoch.MODELS and unit["activation"] in epoch.ACTIVATIONS
                  and unit["backend"] in epoch.BACKENDS and unit["dataset_fingerprint_sha256"] == FINGERPRINT,
                  "Unexpected ImageNet training comparison")
    epoch.require((unit["image_size"], unit["batch_size"], unit["expected_images"], unit["expected_steps"])
                  == (SIZE, BATCH, TRAIN_IMAGES, math.ceil(TRAIN_IMAGES / BATCH)),
                  "Wrong ImageNet training input geometry or split")
    epoch.require(re.fullmatch(r"ep-[a-f0-9]{20}", unit["id"]) and
                  re.fullmatch(r"[a-f0-9]{64}", unit["protocol_sha256"]), "Invalid epoch identity")


def readonly_bundle(root):


    import datasets
    from datasets.arrow_reader import ArrowReader
    from datasets.fingerprint import Hasher
    from lgts_benchmark import data
    import numpy as np

    epoch.require(Path(data.__file__).resolve().is_relative_to(ROOT / "engine") and
                  datasets.__version__ == "4.8.5", "Unverified ImageNet dataset runtime")
    pool, root = FILES, Path(root).resolve()
    inventory, entries, _ = manifest_entries(pool / "source_data_manifest.json")
    revision = data.IMAGENET100_REVISION
    epoch.require(inventory["revision"] == revision and inventory["dataset"] == "imagenet100",
                  "Sealed Arrow inventory differs from the training revision")


    for item in entries:
        verify_file(file_under(root, item["path"]), item)
    cache = root / "huggingface"
    directory = cache / "clane9___imagenet-100" / "default" / "0.0.0" / revision
    info = datasets.DatasetInfo.from_directory(str(directory))
    reader = ArrowReader(str(directory), info)

    def split(name):


        fingerprint = Hasher()
        fingerprint.update(directory.relative_to(cache).as_posix())
        fingerprint.update(name)
        kwargs = reader.read(name="imagenet-100", instructions=name, split_infos=info.splits.values())
        return datasets.Dataset(**kwargs, fingerprint=fingerprint.hexdigest())

    original, test = split("train"), split("validation")
    labels = np.asarray(original["label"], dtype=int)
    train_idx, val_idx = data._stratified_indices(labels)
    train = original.select(train_idx.tolist(), keep_in_memory=True)
    val = original.select(val_idx.tolist(), keep_in_memory=True)
    classes = list(original.features["label"].names)
    synsets_file = root / "huggingface_hub" / "datasets--clane9--imagenet-100" / "snapshots" / revision / "scripts/classes.py"
    synsets = re.findall(r'"(n\d{8})"\s*:', synsets_file.read_text(encoding="utf-8"))
    bundle = data.DatasetBundle(
        "imagenet100", data.HFImageView(train), data.HFImageView(val), data.HFImageView(test),
        labels[train_idx], labels[val_idx], np.asarray(test["label"], dtype=int), classes, 3, SIZE,
        "HuggingFace/clane9/imagenet-100; CMC ImageNet-100 subset",
        dict(huggingface_revision=revision, train_split_fingerprint=original._fingerprint,
             validation_split_fingerprint=test._fingerprint, train_indices_sha256=data._array_sha256(train_idx),
             validation_indices_sha256=data._array_sha256(val_idx), synset_ids=synsets,
             source_resolution_contract="short-side resized to 160 pixels by upstream CMC distribution"))
    manifest = data.dataset_manifest(bundle)
    expected = json.loads((pool / "source_expected_dataset_manifest.json").read_bytes())
    epoch.require(json.loads(json.dumps(manifest)) == expected and manifest["dataset_fingerprint_sha256"] == FINGERPRINT,
                  "Read-only ImageNet loader differs from the original training dataset")
    return bundle


def prepare(runtime, unit):
    import train_benchmark as engine
    from lgts_benchmark import data

    for module in (engine, data, runtime.models, runtime.activations):
        epoch.require(Path(module.__file__).resolve().is_relative_to(ROOT / "engine"), "Unpinned training engine")
    bundle = readonly_bundle(runtime.data_dir)
    transform, _ = data.build_transforms(bundle.name, SIZE, bundle.in_channels, "standard", "dataset")
    train = data.TransformDataset(bundle.train_val, transform)
    cfg = SimpleNamespace(batch_size=BATCH, num_workers=4, prefetch_factor=4, learning_rate=.001,
                          weight_decay=.0001, fused_optimizer=True, performance_mode=False,
                          cudnn_benchmark=True, channels_last=False, tf32=True)
    classes = len(bundle.class_names)
    cell = dict(id=f"epoch-gate:imagenet100:{unit['model']}:{SEED}", source_seed=SEED, dataset=bundle.name,
                config=vars(cfg), input_shape=[bundle.in_channels, SIZE, SIZE], batch_size=BATCH,
                expected_steps=unit["expected_steps"], num_classes=classes,
                precision=dict(channels_last=False, tf32=True, cudnn_benchmark=True))
    runtime.set_policy(cell)
    loader = engine.loader(train, cfg, True, BATCH, data_seed=SEED)
    epoch.require(len(train) == TRAIN_IMAGES and len(loader) == unit["expected_steps"], "Incomplete ImageNet loader")
    sampler = epoch.ObservedSampler(loader.batch_sampler.sampler)
    loader.batch_sampler.sampler = sampler
    engine.seed_everything(SEED, cfg.performance_mode, cfg.cudnn_benchmark)
    cpu = runtime.models.build_model(
        unit["model"], bundle.in_channels, classes, unit["activation"], lgts_alpha=.4, lgts_learn_alpha=True,
        lgts_learn_beta=False, pretrained=False, small_input_stem=False, activation_scope="full_backbone",
        activation_sharing="site")
    runtime.criterion = lambda _: runtime.torch.nn.CrossEntropyLoss()
    return engine, cpu, cell, loader, sampler
