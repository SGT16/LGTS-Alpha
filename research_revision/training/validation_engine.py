from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

VERSION = "major_revision_lr_validation_final_epoch_v1"
ACTIVATIONS = ("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")
LEARNING_RATES = (0.0003, 0.001, 0.003)
SEEDS = (936137, 529212)
HOLDOUT_SEED = 2026
EPOCHS = 100
EXPECTED_ENGINE_SHA256 = "3d27d05c235891599db8bf6de774de9ba13f04a581b8dae7332386b1f14c7ad9"
SOURCE_HASHES_SHA256 = "ce26eb8e8566ec6d7a5f844e98440ebf960397ce77c532513fefb36c52e8ab77"
PLAN_SHA256 = "3c582ec51f7b17cf41bfd08019465bd8f508baa4beb8f62b18faef3ef9608263"
REQUIRED_SEALED_FILES = (
    "train_benchmark.py", "lgts_benchmark/__init__.py",
    "lgts_benchmark/activations.py", "lgts_benchmark/data.py",
    "lgts_benchmark/metrics.py", "lgts_benchmark/models.py",
    "lgts_benchmark/numerics.py", "lgts_benchmark/reporting.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def extended_path(path):
    text = os.path.abspath(os.fspath(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def verify_sealed_engine(revision_root=None):
    root = Path(revision_root or Path(__file__).resolve().parent.parent)
    engine_dir = root / "engine"
    pin_file = root / "training" / "validation_source_hashes.json"
    if sha256(pin_file) != SOURCE_HASHES_SHA256:
        raise RuntimeError("Validation source hash manifest changed")
    expected = json.loads(pin_file.read_text(encoding="utf-8"))["release_engine_files"]
    observed = {name: sha256(engine_dir / name) for name in REQUIRED_SEALED_FILES}
    if observed != expected or observed["train_benchmark.py"] != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Validation source files differ from the release hashes")
    return engine_dir, observed


def validate_job(plan, job_id):
    candidates = [job for job in plan.get("jobs", []) if job.get("id", job.get("job_id")) == job_id]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one matching job in the LR-sweep plan")
    job = copy.deepcopy(candidates[0])
    if not isinstance(plan.get("extension_sha256"), str) or len(plan["extension_sha256"]) != 64:
        raise ValueError("Missing extension identity")
    if job.get("dataset") != "cifar100" or job.get("model") != "EfficientNetV2S":
        raise ValueError("The LR sweep is restricted to CIFAR-100 / EfficientNetV2-S")
    if job.get("activation") not in ACTIVATIONS or job.get("seed") not in SEEDS:
        raise ValueError("Unapproved activation or paired seed")
    if job.get("learning_rate") not in LEARNING_RATES or job.get("epochs") != EPOCHS:
        raise ValueError("Unapproved learning-rate or epoch budget")
    return job


def import_engine(engine_dir):
    engine_dir = Path(engine_dir).resolve()
    sys.path.insert(0, str(engine_dir))


    already_loaded = sys.modules.get("train_benchmark")
    if already_loaded is not None and Path(already_loaded.__file__).resolve() != engine_dir / "train_benchmark.py":
        raise RuntimeError("A different train_benchmark module is already loaded")
    module = importlib.import_module("train_benchmark")
    if Path(module.__file__).resolve() != engine_dir / "train_benchmark.py":
        raise RuntimeError("Imported training module is not the sealed engine")
    module.ENGINE_VERSION = VERSION
    return module


class SealedOfficialTest:
    def __len__(self):
        raise RuntimeError("Official test access is forbidden during LR selection")

    def __getitem__(self, index):
        raise RuntimeError("Official test access is forbidden during LR selection")


class CIFARTrainingOnly:

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        from PIL import Image
        return Image.fromarray(self.data[index]), int(self.targets[index])


def load_training_only_bundle(data_dir, engine):
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR100
    from sklearn.model_selection import train_test_split
    from lgts_benchmark.data import DatasetBundle

    folder = Path(data_dir) / "torchvision" / CIFAR100.base_folder
    train_name, expected_md5 = CIFAR100.train_list[0]
    train_file = folder / train_name
    meta_file = folder / CIFAR100.meta["filename"]


    for path, digest in ((train_file, expected_md5), (meta_file, CIFAR100.meta["md5"])):
        hasher = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise RuntimeError(f"Official CIFAR training/meta bytes failed integrity: {path.name}")
    with train_file.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    with meta_file.open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    labels = np.asarray(raw["fine_labels"], dtype=np.int64)
    if len(labels) != 50000 or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 500)):
        raise RuntimeError("Unexpected CIFAR-100 official training class counts")
    data = raw["data"].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    base = CIFARTrainingOnly(data, labels)
    train_idx, val_idx = train_test_split(np.arange(50000), test_size=0.1, random_state=HOLDOUT_SEED, shuffle=True, stratify=labels)
    train_counts = np.bincount(labels[train_idx], minlength=100)
    val_counts = np.bincount(labels[val_idx], minlength=100)
    if not np.all(train_counts == 450) or not np.all(val_counts == 50):
        raise RuntimeError("Validation split must contain 450/50 train/validation examples per class")
    def index_hash(values):
        return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()
    metadata = {
        "source": "official_cifar100_training_file_only",
        "training_file_sha256": sha256(train_file), "metadata_file_sha256": sha256(meta_file),
        "holdout_seed": HOLDOUT_SEED, "train_count": 45000, "validation_count": 5000,
        "train_indices_sha256": index_hash(train_idx), "validation_indices_sha256": index_hash(val_idx),
        "train_class_counts": train_counts.tolist(), "validation_class_counts": val_counts.tolist(),
        "official_test_accessed": False,
    }
    metadata["dataset_fingerprint_sha256"] = canonical_hash(metadata)
    if metadata["dataset_fingerprint_sha256"] != "b588c77570783f332281f73ed0d12868158b6bbd9fd21448d02075fa2e3d6efa":
        raise ValueError("CIFAR-100 validation split differs from the recorded split")
    bundle = DatasetBundle("cifar100", torch.utils.data.Subset(base, train_idx.tolist()),
        torch.utils.data.Subset(base, val_idx.tolist()), SealedOfficialTest(),
        labels[train_idx], labels[val_idx], np.empty(0, dtype=np.int64), list(meta[CIFAR100.meta["key"]]),
        3, 32, "torchvision/CIFAR100 training only", metadata)
    return bundle, {**metadata, "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist()}


def adapt_train_one(engine, decorate_row, write_json):


    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "train_one")
    node = copy.deepcopy(node)
    reload_blocks = []
    for index, statement in enumerate(node.body):
        if isinstance(statement, ast.If) and isinstance(statement.test, ast.Name) and statement.test.id == "uses_validation":
            calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "load_state_dict"]
            if len(calls) == 1 and isinstance(calls[0].func.value, ast.Name) and calls[0].func.value.id == "model":
                reload_blocks.append(index)
    if len(reload_blocks) != 1:
        raise RuntimeError("Expected exactly one final best-weight reload block")
    node.body[reload_blocks[0]] = ast.parse("if best_state is not None:\n    del best_state").body[0]
    recoveries = [index for index, statement in enumerate(node.body) if isinstance(statement, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "recovery_metadata" for target in statement.targets)]
    if len(recoveries) != 1:
        raise RuntimeError("Expected exactly one final-weight metadata construction")
    node.body.insert(recoveries[0] + 1, ast.parse(
        "recovery_metadata.update(selected_epoch=len(history), selection_policy='fixed_budget_final_epoch_validation', "
        "validation_best_epoch_diagnostic=best_epoch, official_test_accessed=False)"
    ).body[0])
    rows = [index for index, statement in enumerate(node.body) if isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "row" for target in statement.targets)]
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one result-row construction")
    node.body.insert(rows[0] + 1, ast.parse("row = _decorate_validation_row(row)").body[0])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(engine))
    namespace.update(_decorate_validation_row=decorate_row, write_json=write_json, ENGINE_VERSION=VERSION)
    exec(compile(module, str(Path(__file__).resolve()) + ":sealed_train_one_adapter", "exec"), namespace)
    return namespace["train_one"], namespace


def make_engine_args(engine, job, data_dir, output_dir, device, num_workers, pause_file):
    cli = [str(engine.__file__), "--datasets", "cifar100", "--models", "EfficientNetV2S",
        "--activations", job["activation"], "--seeds", str(job["seed"]), "--seed_order", *map(str, SEEDS),
        "--protocol", "fixed", "--epochs", str(EPOCHS), "--patience", "0", "--learning_rate", str(job["learning_rate"]),
        "--batch_size", "128", "--num_workers", str(num_workers), "--prefetch_factor", "4",
        "--optimizer", "adamw", "--weight_decay", "0.0001", "--scheduler", "cosine", "--min_learning_rate", "0.000001",
        "--no_pretrained", "--tf32", "--cudnn_benchmark", "--best_state_device", "cpu", "--gradient_log", "--activation_scope", "full_backbone",
        "--activation_sharing", "site", "--lgts_alphas", "0.40", "--lgts_beta", "1.0", "--lgts_learning", "both",
        "--augmentation", "standard", "--normalization", "dataset", "--class_weight", "auto",
        "--data_dir", str(data_dir), "--output_dir", str(output_dir), "--device", device, "--plots", "none"]
    if pause_file:
        cli.extend(["--pause_file", str(pause_file)])
    old_argv = sys.argv
    try:
        sys.argv = cli
        return engine.parse_args()
    finally:
        sys.argv = old_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pause-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    if sha256(options.plan) != PLAN_SHA256:
        raise ValueError("Validation plan differs from the released 30-run plan")
    plan = json.loads(Path(options.plan).read_text(encoding="utf-8-sig"))
    job = validate_job(plan, options.job_id)
    engine_dir, base_hashes = verify_sealed_engine()
    adapter_hash = sha256(__file__)
    source_identity = {"adapter_sha256": adapter_hash, "base_source_file_sha256": base_hashes, "version": VERSION}
    if options.dry_run:
        print(json.dumps({"status": "dry_run_ok", "job": job, **source_identity,
            "selection_policy": "fixed_budget_final_epoch_validation", "official_test_accessed": False}, ensure_ascii=False))
        return
    engine = import_engine(engine_dir)
    import torch
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is not permitted")
    output = extended_path(options.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = make_engine_args(engine, job, options.data_dir, output / "artifacts", options.device, options.num_workers, options.pause_file)
    args.lr_sweep_job = job
    args.lr_sweep_extension_sha256 = plan["extension_sha256"]
    args.lr_sweep_selection_policy = "fixed_budget_final_epoch_validation"
    args.lr_sweep_source_identity = source_identity
    identity = engine.prepare_experiment_identity(args, engine_dir)
    base_engine_code_sha256 = identity["code_sha256"]
    args.code_sha256 = canonical_hash({"base_engine_code_sha256": base_engine_code_sha256, **source_identity})
    args.experiment_id = canonical_hash({"base_identity": identity, "scientific_code_sha256": args.code_sha256, "job": job})[:16]
    args.git_commit = identity["git_commit"]
    args.runtime_metadata = identity["runtime"]
    args.live_workbook_path = str(output / "validation_results.xlsx")
    identity.update(experiment_id=args.experiment_id, code_sha256=args.code_sha256,
        base_engine_code_sha256=base_engine_code_sha256, **source_identity)
    existing_identity = output / "experiment_identity.json"
    if existing_identity.exists():
        prior = json.loads(existing_identity.read_text(encoding="utf-8"))
        if prior.get("experiment_id") != args.experiment_id:
            raise RuntimeError("Existing sweep output belongs to different scientific settings; refusing overwrite")
    engine.write_json(existing_identity, identity)
    if (output / "done.json").exists():
        done = json.loads((output / "done.json").read_text(encoding="utf-8"))
        if done.get("experiment_id") != args.experiment_id:
            raise RuntimeError("Completed result identity mismatch")
        print(f"SKIP | {options.job_id} | validation-only run complete", flush=True)
        return
    bundle, manifest = load_training_only_bundle(options.data_dir, engine)
    engine.write_json(output / "validation_split.json", manifest)
    engine.write_json(output / "run_config.json", vars(args))
    engine.configure_runtime(args)
    engine.initialize_live_workbook(Path(args.live_workbook_path))
    base_write_json = engine.write_json
    def decorate_row(row):
        if row.get("epochs_run") != EPOCHS or row.get("evaluation_split") != "validation":
            raise RuntimeError("LR selection requires the full final-epoch validation result")
        row = {key: value for key, value in row.items() if not key.startswith("test_")}
        row.update(status="lr_validation_done", study_stage="post_confirmatory_lr_validation_sweep",
            job_id=options.job_id, extension_sha256=plan["extension_sha256"], activation=job["activation"],
            learning_rate=job["learning_rate"], protocol="validation_only", selection_policy="fixed_budget_final_epoch_validation",
            selected_epoch=EPOCHS, evaluation_split="validation", official_test_accessed=False,
            best_epoch_role="diagnostic_only_not_used_for_lr_selection",
            validation_macro_f1=row["validation_f1_macro"], validation_accuracy=row["validation_accuracy"],
            adapter_sha256=adapter_hash, base_engine_code_sha256=base_engine_code_sha256,
            split_fingerprint=bundle.metadata["dataset_fingerprint_sha256"],
            train_count=45000, validation_count=5000, split_seed=HOLDOUT_SEED,
            training_precision="AMP_FP16_with_sealed_FP32_stable_LGTS",
            precision_policy="AMP_FP16_with_sealed_FP32_stable_LGTS", final_validation_score_for_lr_selection=True)
        return row
    latest_training_seconds = 0.0
    def mirrored_write(path, payload):
        nonlocal latest_training_seconds
        base_write_json(path, payload)
        filename = Path(path).name
        if filename == "in_progress.json":
            root_progress = {**payload, "job_id": options.job_id, "extension_sha256": plan["extension_sha256"],
                "learning_rate": job["learning_rate"], "official_test_accessed": False,
                "train_time_sec": latest_training_seconds}
            base_write_json(output / filename, root_progress)
        elif filename in {"done.json", "history.json", "numerical_failure.json"}:
            if filename == "history.json":
                latest_training_seconds = sum(float(item["time_sec"]) for item in payload)
            base_write_json(output / filename, payload)
            if filename == "done.json":
                (output / "in_progress.json").unlink(missing_ok=True)
    train_one, _ = adapt_train_one(engine, decorate_row, mirrored_write)
    print(f"LR VALIDATION | job={options.job_id} | lr={job['learning_rate']} | seed={job['seed']} | train=45000 | val=5000 | official_test=NOT_LOADED | final_epoch=100", flush=True)
    train_one(args, bundle, "EfficientNetV2S", job["activation"], 0.4 if job["activation"].startswith("LGTS") else None,
        job["seed"], 1, SEEDS.index(job["seed"]) + 1, None, None, torch.device(options.device))


    if not (output / "done.json").exists():
        markers = list((output / "artifacts").rglob("done.json"))
        if len(markers) != 1:
            raise RuntimeError("Expected one completed nested validation result")
        result = json.loads(markers[0].read_text(encoding="utf-8"))
        if result.get("experiment_id") != args.experiment_id or result.get("job_id") != options.job_id:
            raise RuntimeError("Nested completion identity mismatch")
        base_write_json(output / "done.json", result)
        (output / "in_progress.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
