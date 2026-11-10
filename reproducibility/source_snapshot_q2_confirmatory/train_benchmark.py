from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from lgts_benchmark.activations import (
    adaptive_activation_parameter_rows,
    adaptive_activation_parameter_stats,
    canonical_activation_name,
    is_lgts_activation,
    lgts_parameter_rows,
    lgts_parameter_snapshot,
    lgts_parameter_stats,
)
from lgts_benchmark.data import HOLDOUT_SEED, MEDMNIST_2D_DATASETS, TransformDataset, TransformSubset, build_transforms, dataset_manifest, declared_dataset_contract, load_bundle
from lgts_benchmark.metrics import classification_metrics, plot_confusion, plot_learning_curves, plot_lgts_diagnostics, plot_reliability, plot_roc_pr
from lgts_benchmark.models import (
    build_model,
    count_parameters,
    is_native_activation,
    native_activation_variant,
)
from lgts_benchmark.reporting import append_csv, build_aggregate_reports, initialize_live_workbook, update_live_workbook, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LGTS v2 validation-first full-backbone benchmark")
    p.add_argument("--datasets", nargs="+", default=["cifar100", "dermamnist", "organsmnist"])
    p.add_argument("--models", nargs="+", default=["ResNet18", "MobileNetV2", "EfficientNetV2S"], choices=["PaperCNN", "ResNet18", "MobileNetV2", "ResNet50", "EfficientNetB0", "EfficientNetV2S", "ConvNeXtTiny"])
    p.add_argument("--activations", nargs="+", default=["LGTS-f2-Slope", "Native", "Swish", "Qian-Gated", "ACON-C", "ReLU", "ELU", "Softplus", "GELU", "Mish", "PReLU"], help="LGTS-f2-Slope is the locked primary method; Native preserves each backbone's original activation")
    p.add_argument("--activation_scope", default="full_backbone", choices=["last_stage", "full_backbone"], help="Head-only is intentionally disabled; PaperCNN always uses all_layers")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0, help="0 gives every activation the full epoch budget (confirmatory default)")
    p.add_argument("--min_epochs", type=int, default=0, help="Do not early-stop before this epoch; 0 applies patience immediately")
    p.add_argument(
        "--protocol", default="fixed", choices=["fixed", "full_train", "kfold"],
        help="fixed selects on a locked validation split; full_train uses train+validation for a fixed epoch budget after method selection",
    )
    p.add_argument("--k_folds", type=int, default=5, help="Used only with --protocol kfold")
    p.add_argument("--batch_size", type=int, default=128, help="CIFAR/default batch size")
    p.add_argument("--med_batch_size", type=int, default=128)
    p.add_argument("--imagenet_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    fused = p.add_mutually_exclusive_group()
    fused.add_argument("--fused_optimizer", dest="fused_optimizer", action="store_true", help="Use CUDA fused Adam/AdamW when supported")
    fused.add_argument("--no_fused_optimizer", dest="fused_optimizer", action="store_false")
    p.set_defaults(fused_optimizer=True)
    p.add_argument("--momentum", type=float, default=0.9, help="Used only by SGD")
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--min_learning_rate", type=float, default=1e-6)
    p.add_argument("--min_delta", type=float, default=0.0, help="Minimum validation macro-F1 improvement; 0 selects the exact maximum")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    p.add_argument(
        "--seed_order",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional study-wide seed order used for stable Run i/N labels when "
            "--seeds launches only an unfinished subset"
        ),
    )
    p.add_argument("--med_size", type=int, default=64, choices=[28, 64, 128, 224])
    p.add_argument("--imagenet_size", type=int, default=160, choices=[128, 160, 192, 224])
    p.add_argument("--image_size", type=int, default=0, help="Optional global override; 0 uses CIFAR=32, medical=med_size, ImageNet-100=imagenet_size")
    p.add_argument("--augmentation", default="standard", choices=["none", "standard"])
    p.add_argument("--normalization", default="dataset", choices=["dataset", "imagenet", "none"])
    p.add_argument("--small_input_stem", default="auto", choices=["auto", "always", "never",], help="auto uses stride-1 small-input stems for supported backbones at <=64 px")
    p.add_argument("--class_weight", default="auto", choices=["auto", "none", "balanced"], help="auto: balanced for Path/DermaMNIST, none otherwise")
    p.add_argument("--log_t", type=float, default=0.6)
    p.add_argument("--lgts_alphas", nargs="+", type=float, default=[0.40], help="Locked primary initialization is 0.40; use a grid only in a separately labeled ablation")
    p.add_argument("--lgts_beta", type=float, default=1.0, help="Initial beta; f3 initializes beta+ and beta- to this value")
    p.add_argument("--lgts_alpha_min", type=float, default=0.05)
    p.add_argument("--lgts_alpha_max", type=float, default=1.5)
    p.add_argument("--lgts_beta_min", type=float, default=0.5)
    p.add_argument("--lgts_beta_max", type=float, default=3.0)
    p.add_argument("--lgts_learning", default="both", choices=["both", "alpha", "beta", "none"], help="LGTS learnability ablation")
    p.add_argument("--lgts_fixed", action="store_true", help="Deprecated alias for --lgts_learning none")
    pretraining = p.add_mutually_exclusive_group()
    pretraining.add_argument("--pretrained", dest="no_pretrained", action="store_false", help="Use torchvision ImageNet weights")
    pretraining.add_argument("--no_pretrained", dest="no_pretrained", action="store_true", help="Use random initialization (default)")
    p.set_defaults(no_pretrained=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--live_workbook", default="", help="Optional tracker path; default is <output_dir>/live_results.xlsx")
    p.add_argument(
        "--shared_live_workbook",
        action="store_true",
        help="Use the exact --live_workbook path across sequential experiment processes",
    )
    p.add_argument("--plots", default="all", choices=["all", "none"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--best_state_device", default="device", choices=["device", "cpu"], help="device avoids a GPU-to-CPU copy at every validation improvement")
    p.add_argument("--performance_mode", action="store_true", help="Enable TF32, cuDNN benchmark and channels-last; faster but not strictly deterministic")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 for CUDA matmul and cuDNN")
    p.add_argument("--cudnn_benchmark", action="store_true", help="Let cuDNN select the fastest kernels for fixed input shapes")
    p.add_argument("--channels_last", action="store_true", help="Use NHWC/channels-last memory layout for 4D image tensors")
    p.add_argument("--rerun_done", action="store_true")
    p.add_argument("--keep_duplicate_controls", action="store_true", help="Also run Native-ReLU vs ReLU and Native-SiLU vs Swish duplicates")
    p.add_argument("--dry_run", action="store_true", help="Print the planned run count without loading data or training")
    args = p.parse_args()
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        p.error("--min_epochs must be between 0 and --epochs")
    if args.patience < 0:
        p.error("--patience must be >=0; use 0 to disable early stopping")
    if args.protocol == "full_train" and args.patience != 0:
        p.error("--protocol full_train requires --patience 0 because it uses a fixed epoch budget")
    if not args.lgts_alpha_min < args.lgts_alpha_max:
        p.error("--lgts_alpha_min must be smaller than --lgts_alpha_max")
    if not args.lgts_beta_min < args.lgts_beta_max:
        p.error("--lgts_beta_min must be smaller than --lgts_beta_max")
    if any(not args.lgts_alpha_min < alpha < args.lgts_alpha_max for alpha in args.lgts_alphas):
        p.error("Every --lgts_alphas value must lie strictly inside the configured alpha range")
    if not args.lgts_beta_min < args.lgts_beta < args.lgts_beta_max:
        p.error("--lgts_beta must lie strictly inside the configured beta range")
    if args.performance_mode:
        args.tf32 = True
        args.cudnn_benchmark = True
        args.channels_last = True
    if args.lgts_fixed:
        if args.lgts_learning != "both":
            p.error("Use either --lgts_fixed or --lgts_learning, not both")
        args.lgts_learning = "none"
    return args


FINGERPRINT_FILES = (
    "train_benchmark.py",
    "protocol_v2.json",
    "protocol_f4_ablation.json",
    "protocol_q2_confirmatory.json",
    "lgts_benchmark/activations.py",
    "lgts_benchmark/data.py",
    "lgts_benchmark/metrics.py",
    "lgts_benchmark/models.py",
    "lgts_benchmark/reporting.py",
)


def source_fingerprint(project_root: Path) -> tuple[str, dict[str, str]]:
    per_file: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in FINGERPRINT_FILES:
        path = project_root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        per_file[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
    return combined.hexdigest(), per_file


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def runtime_metadata() -> dict[str, str | bool | None]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": gpu,
    }


def package_lock_snapshot() -> tuple[str, str]:
    packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    text = "\n".join(packages) + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_experiment_identity(args: argparse.Namespace, project_root: Path) -> dict:
    code_sha256, file_hashes = source_fingerprint(project_root)
    runtime = runtime_metadata()
    package_lock, package_lock_sha256 = package_lock_snapshot()
    excluded = {
        "datasets", "models", "activations", "lgts_alphas", "seeds", "seed_order",
        "data_dir", "output_dir", "live_workbook", "shared_live_workbook", "plots", "rerun_done", "dry_run",
    }
    canonical_config = {
        key: value for key, value in sorted(vars(args).items()) if key not in excluded
    }
    environment_identity = {
        key: runtime.get(key) for key in (
            "python_version", "torch_version", "torchvision_version",
            "cuda_runtime_version", "cudnn_version", "gpu_model",
        )
    }
    identity_payload = {
        "canonical_config": canonical_config,
        "code_sha256": code_sha256,
        "dataset_contract": declared_dataset_contract(),
        "environment_identity": environment_identity,
        "package_lock_sha256": package_lock_sha256,
    }
    canonical_json = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    experiment_id = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]
    return {
        "experiment_id": experiment_id,
        "code_sha256": code_sha256,
        "source_file_sha256": file_hashes,
        "git_commit": git_commit(project_root),
        "canonical_config": canonical_config,
        "dataset_contract": declared_dataset_contract(),
        "runtime": runtime,
        "package_lock_sha256": package_lock_sha256,
        "package_lock": package_lock.splitlines(),
    }


def seed_everything(seed: int, performance_mode: bool = False, cudnn_benchmark: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not performance_mode
    torch.backends.cudnn.benchmark = cudnn_benchmark


def configure_runtime(args) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.deterministic = not bool(args.performance_mode)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")


def move_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device, channels_last: bool):
    if channels_last and x.ndim == 4:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        x = x.to(device, non_blocking=True)
    return x, y.to(device, non_blocking=True)


def seed_loader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader(dataset, args, shuffle: bool, batch_size: int | None = None, data_seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(data_seed))
    kwargs = dict(
        batch_size=batch_size or args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_loader_worker,
    )
    if args.num_workers > 0:
        kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, **kwargs)


def resolve_class_weight_mode(requested: str, dataset_name: str) -> str:
    if requested == "auto":
        return "balanced" if dataset_name in MEDMNIST_2D_DATASETS else "none"
    return requested


def is_duplicate_control(model_name: str, activation: str, all_activations: list[str], keep: bool) -> bool:
    if keep or not any(is_native_activation(item) for item in all_activations):
        return False
    model_key = model_name.lower().replace("-", "").replace("_", "")
    activation_key = canonical_activation_name(activation).lower().replace("-", "").replace("_", "")
    if model_key.startswith("resnet") and activation_key == "relu":
        return True
    if model_key in {"efficientnetb0", "efficientnetv2s"} and activation_key in {"swish", "silu"}:
        return True
    return False


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16) if enabled else nullcontext()


def evaluate_basic(model: nn.Module, data_loader: DataLoader, criterion, device: torch.device, amp: bool, channels_last: bool = False):
    model.eval(); total_loss = 0.0; labels = []; predictions = []
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = move_batch(x, y, device, channels_last)
            with amp_context(device, amp): logits = model(x); loss = criterion(logits, y)
            total_loss += loss.detach().item() * y.size(0); labels.append(y.cpu().numpy()); predictions.append(logits.argmax(1).cpu().numpy())
    true = np.concatenate(labels); pred = np.concatenate(predictions)
    return total_loss / len(data_loader.dataset), f1_score(true, pred, average="macro", zero_division=0), accuracy_score(true, pred)


def predict(model: nn.Module, data_loader: DataLoader, criterion, device: torch.device, amp: bool, channels_last: bool = False):
    model.eval(); losses = 0.0; ys = []; probs = []; logits_rows = []
    if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = move_batch(x, y, device, channels_last)
            with amp_context(device, amp): logits = model(x); loss = criterion(logits, y)
            logits_float = logits.float()
            losses += loss.detach().item() * y.size(0)
            ys.append(y.cpu().numpy())
            logits_rows.append(logits_float.cpu().numpy())
            probs.append(torch.softmax(logits_float, dim=1).cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(logits_rows), losses / len(data_loader.dataset), elapsed, peak_mb


def balanced_weights(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = len(labels) / np.maximum(n_classes * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_optimizer(model: nn.Module, args):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith("bias") else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.learning_rate, momentum=args.momentum, nesterov=True)
    use_fused = bool(args.fused_optimizer and next(model.parameters()).is_cuda)
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.learning_rate, fused=use_fused)
    return torch.optim.AdamW(groups, lr=args.learning_rate, fused=use_fused)


def resolve_image_size(args, dataset_name: str, native_size: int) -> int:
    if args.image_size > 0:
        return int(args.image_size)
    if dataset_name == "imagenet100":
        return int(args.imagenet_size)
    if dataset_name in MEDMNIST_2D_DATASETS:
        return int(args.med_size)
    return int(native_size)


def resolve_batch_size(args, dataset_name: str) -> int:
    if dataset_name == "imagenet100":
        return int(args.imagenet_batch_size)
    if dataset_name in MEDMNIST_2D_DATASETS:
        return int(args.med_batch_size)
    return int(args.batch_size)


PARTIAL_RUN_PATTERNS = (
    "history.json",
    "in_progress.json",
    "best.pt",
    "last.pt",
    "*_predictions.npz",
    "*_per_class_metrics.csv",
    "lgts_layer_parameters.csv",
    "adaptive_activation_parameters.csv",
    "*.png",
)


def reset_partial_run(run_dir: Path) -> list[str]:
    removed: list[str] = []
    for pattern in PARTIAL_RUN_PATTERNS:
        for artifact in run_dir.glob(pattern):
            if artifact.is_file():
                artifact.unlink()
                removed.append(artifact.name)
    return sorted(set(removed))


def release_accelerator_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_one(
    args,
    bundle,
    model_name: str,
    activation: str,
    lgts_alpha: float | None,
    seed: int,
    fold_no: int,
    repeat_no: int,
    train_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    device: torch.device,
):
    release_accelerator_memory(device)
    effective_scope = "all_layers" if model_name == "PaperCNN" else args.activation_scope
    activation_name = canonical_activation_name(activation)
    if is_native_activation(activation):
        activation_name = "Native"
        activation_variant = native_activation_variant(model_name)
    else:
        activation_variant = f"{activation_name}_a{lgts_alpha:.2f}" if is_lgts_activation(activation) and lgts_alpha is not None else activation_name
    study_seed_order = args.seed_order or args.seeds
    repeat_total = len(study_seed_order)
    if args.protocol in {"fixed", "full_train"}:
        terminal_unit = (
            f"Run {repeat_no}/{repeat_total} | seed={seed} | Fold 1/1"
        )
    else:
        terminal_unit = (
            f"Run {repeat_no}/{repeat_total} | seed={seed} | "
            f"Fold {fold_no}/{args.k_folds}"
        )
    unit_dir = args.protocol if args.protocol in {"fixed", "full_train"} else f"fold_{fold_no}"
    dataset_fingerprint = bundle.metadata.get("dataset_fingerprint_sha256", "unversioned")
    run_dir = Path(args.output_dir) / bundle.name / f"data_{dataset_fingerprint[:16]}" / model_name / effective_scope / activation_variant / f"seed_{seed}" / unit_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.json"
    in_progress_path = run_dir / "in_progress.json"
    if done_path.exists() and not args.rerun_done:
        in_progress_path.unlink(missing_ok=True)
        print(f"SKIP | {bundle.name} | {terminal_unit} | {model_name}-{activation_variant}-{effective_scope} | completed")
        return
    rerunning_completed = done_path.exists()
    if rerunning_completed:
        done_path.unlink()
    partial_artifacts = reset_partial_run(run_dir)
    if partial_artifacts and not rerunning_completed:
        print(
            f"RESTART | {bundle.name} | {terminal_unit} | "
            f"{model_name}-{activation_variant}-{effective_scope} | "
            "incomplete fold detected; restarting at epoch 1",
            flush=True,
        )
    elif rerunning_completed:
        print(
            f"RERUN | {bundle.name} | {terminal_unit} | "
            f"{model_name}-{activation_variant}-{effective_scope} | restarting at epoch 1",
            flush=True,
        )
    write_json(in_progress_path, {
        "status": "running_from_epoch_1",
        "dataset": bundle.name,
        "model": model_name,
        "activation_variant": activation_variant,
        "seed": seed,
        "fold": fold_no,
        "protocol": args.protocol,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    image_size = resolve_image_size(args, bundle.name, bundle.native_size)
    batch_size = resolve_batch_size(args, bundle.name)
    train_tf, eval_tf = build_transforms(
        bundle.name, image_size, bundle.in_channels, args.augmentation, args.normalization,
    )
    if args.protocol == "fixed":
        train_ds = TransformDataset(bundle.train, train_tf)
        val_ds = TransformDataset(bundle.val, eval_tf)
        training_labels = bundle.labels_train
        effective_seed = seed
    elif args.protocol == "full_train":
        train_ds = TransformDataset(bundle.train_val, train_tf)
        val_ds = None
        training_labels = bundle.labels_train_val
        effective_seed = seed
    else:
        if train_idx is None or val_idx is None:
            raise ValueError("kfold protocol requires train and validation indices")
        train_ds = TransformSubset(bundle.train_val, train_idx, train_tf)
        val_ds = TransformSubset(bundle.train_val, val_idx, eval_tf)
        training_labels = bundle.labels_train_val[train_idx]
        effective_seed = seed + fold_no * 1009
    test_ds = TransformDataset(bundle.test, eval_tf) if args.protocol == "full_train" else None
    validation_count = len(val_ds) if val_ds is not None else 0
    test_count_display = len(test_ds) if test_ds is not None else "sealed"
    print(f"Counts | {bundle.name} | {terminal_unit} | train={len(train_ds)} | validation={validation_count} | test={test_count_display}")
    print(f"Augmentation | profile={args.augmentation} | before={len(train_ds)} | after={len(train_ds)} | online_count_unchanged=true")
    print(f"Normalization | profile={args.normalization} | image_size={image_size} | batch_size={batch_size}")
    model_seed = effective_seed
    data_seed = effective_seed
    train_loader = loader(train_ds, args, True, batch_size, data_seed=data_seed)
    val_loader = loader(val_ds, args, False, batch_size, data_seed=data_seed + 1) if val_ds is not None else None

    seed_everything(effective_seed, args.performance_mode, args.cudnn_benchmark)
    small_input_stem = args.small_input_stem == "always" or (
        args.small_input_stem == "auto" and image_size <= 64
    )
    model = build_model(
        model_name,
        bundle.in_channels,
        len(bundle.class_names),
        activation,
        log_t=args.log_t,
        lgts_learnable=getattr(args, "lgts_learning", "none" if args.lgts_fixed else "both") != "none",
        lgts_learn_alpha=getattr(args, "lgts_learning", "none" if args.lgts_fixed else "both") in {"both", "alpha"},
        lgts_learn_beta=getattr(args, "lgts_learning", "none" if args.lgts_fixed else "both") in {"both", "beta"},
        lgts_alpha=lgts_alpha if lgts_alpha is not None else 0.40,
        lgts_beta=args.lgts_beta,
        lgts_alpha_min=args.lgts_alpha_min,
        lgts_alpha_max=args.lgts_alpha_max,
        lgts_beta_min=args.lgts_beta_min,
        lgts_beta_max=args.lgts_beta_max,
        activation_scope=args.activation_scope,
        pretrained=not args.no_pretrained,
        small_input_stem=small_input_stem,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    class_weight_mode = resolve_class_weight_mode(args.class_weight, bundle.name)
    weights = balanced_weights(training_labels, len(bundle.class_names), device) if class_weight_mode == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate,
    ) if args.scheduler == "cosine" else None
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    uses_validation = val_loader is not None
    best_f1, best_epoch, best_state, esb, history = (-math.inf if uses_validation else None), 0, None, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter(); model.train(); running = 0.0
        for x, y in train_loader:
            x, y = move_batch(x, y, device, args.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp): logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); running += loss.detach().item() * y.size(0)
        if uses_validation:
            val_loss, val_f1, val_acc = evaluate_basic(model, val_loader, criterion, device, amp, args.channels_last)
        else:
            val_loss, val_f1, val_acc = None, None, None
        improved = bool(uses_validation and val_f1 > best_f1 + args.min_delta)
        if improved:
            best_f1, best_epoch, esb = val_f1, epoch, 0
            if args.best_state_device == "cpu":
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            else:
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        elif uses_validation:
            esb += 1
        else:
            best_epoch = epoch
        parameter_snapshot = lgts_parameter_snapshot(model)
        epoch_time = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_ds),
            "val_f1": val_f1,
            "acc": val_acc,
            "loss": val_loss,
            "time_sec": epoch_time,
            "esb": esb,
            "best": best_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **parameter_snapshot,
        }
        history.append(record); write_json(run_dir / "history.json", history)
        if uses_validation:
            print(f"{epoch:03d}/{args.epochs:03d} | {bundle.name} | {terminal_unit} | {model_name}-{activation_variant}-{effective_scope} | val_f1={val_f1:.4f} | acc={val_acc:.4f} | loss={val_loss:.4f} | time={epoch_time:.1f}s | esb={esb} | best={best_f1:.4f}", flush=True)
        else:
            print(f"{epoch:03d}/{args.epochs:03d} | {bundle.name} | {terminal_unit} | {model_name}-{activation_variant}-{effective_scope} | val_f1=NA | acc=NA | loss={record['train_loss']:.4f} | time={epoch_time:.1f}s | esb=0 | best=NA", flush=True)
        if scheduler is not None:
            scheduler.step()
        if uses_validation and args.patience > 0 and epoch >= args.min_epochs and esb >= args.patience:
            break

    train_time = float(sum(float(item["time_sec"]) for item in history))
    peak_training_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    if uses_validation:
        if best_state is None:
            raise RuntimeError("No best model was selected in this run.")
        model.load_state_dict(best_state)
        del best_state
    del optimizer, scaler, train_loader
    if scheduler is not None:
        del scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate_official_test = args.protocol == "full_train"
    evaluation_split = "test" if evaluate_official_test else "validation"
    if evaluate_official_test:
        if test_ds is None:
            raise RuntimeError("Confirmatory full_train requires the official test dataset")
        evaluation_ds = test_ds
        evaluation_loader = loader(evaluation_ds, args, False, batch_size, data_seed=data_seed + 2)
    else:
        if val_loader is None or val_ds is None:
            raise RuntimeError("A sealed selection run requires a validation split")
        evaluation_ds = val_ds
        evaluation_loader = val_loader
    y_true, probs, logits, evaluation_loss, inference_time, peak_mb = predict(
        model, evaluation_loader, criterion, device, amp, args.channels_last,
    )
    if val_loader is not None:
        del val_loader
    metrics, per_class, cm = classification_metrics(y_true, probs, bundle.class_names, logits=logits)
    total_params, trainable_params = count_parameters(model); parameter_stats = lgts_parameter_stats(model)
    adaptive_stats = adaptive_activation_parameter_stats(model)
    alpha, beta = parameter_stats["alpha_mean"], parameter_stats["beta_mean"]
    epoch_times = [float(item["time_sec"]) for item in history]
    mean_epoch_time = float(np.mean(epoch_times))
    sd_epoch_time = float(np.std(epoch_times, ddof=1)) if len(epoch_times) > 1 else 0.0
    np.savez_compressed(run_dir / f"{evaluation_split}_predictions.npz", y_true=y_true, probabilities=probs, logits=logits)
    with (run_dir / f"{evaluation_split}_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0])); w.writeheader(); w.writerows(per_class)
    parameter_rows = lgts_parameter_rows(model)
    if parameter_rows:
        parameter_fields = sorted(set().union(*(item.keys() for item in parameter_rows)))
        with (run_dir / "lgts_layer_parameters.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=parameter_fields); w.writeheader(); w.writerows(parameter_rows)
    adaptive_rows = adaptive_activation_parameter_rows(model)
    if adaptive_rows:
        adaptive_fields = sorted(set().union(*(item.keys() for item in adaptive_rows)))
        with (run_dir / "adaptive_activation_parameters.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=adaptive_fields); w.writeheader(); w.writerows(adaptive_rows)
    if args.plots == "all":
        plot_learning_curves(history, run_dir / "learning_curves.png")
        plot_confusion(cm, bundle.class_names, run_dir / f"{evaluation_split}_confusion_matrix.png")
        plot_roc_pr(y_true, probs, bundle.class_names, run_dir / f"{evaluation_split}_roc_curves.png", run_dir / f"{evaluation_split}_pr_curves.png")
        plot_reliability(y_true, probs, run_dir / f"{evaluation_split}_reliability_diagram.png")
        plot_lgts_diagnostics(model, history, run_dir / "lgts_shape_and_derivative.png", run_dir / "lgts_parameter_trajectories.png")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "dataset": bundle.name, "source": bundle.source,
        "model": model_name, "activation_scope": effective_scope, "activation": activation_name, "activation_variant": activation_variant,
        "experiment_id": getattr(args, "experiment_id", "unversioned"),
        "code_sha256": getattr(args, "code_sha256", "unversioned"),
        "git_commit": getattr(args, "git_commit", "unavailable"),
        "protocol": args.protocol,
        "split_seed": HOLDOUT_SEED if args.protocol == "fixed" else seed if args.protocol == "kfold" else "",
        "training_seed": effective_seed, "model_seed": model_seed, "data_seed": data_seed,
        "repeat": repeat_no, "seed": seed, "fold": fold_no,
        "pretrained": model_name != "PaperCNN" and not args.no_pretrained,
        "weight_initialization": "he_random" if model_name == "PaperCNN" else "imagenet" if not args.no_pretrained else "random",
        "epochs_run": len(history), "best_epoch": best_epoch,
        "best_validation_f1": best_f1 if uses_validation else "",
        "selection_policy": "best_validation_macro_f1" if uses_validation else "fixed_budget_final_epoch",
        "evaluation_split": evaluation_split,
        f"{evaluation_split}_loss": evaluation_loss,
        "nll_source": "stable_logits_float64",
        **{f"{evaluation_split}_{k}": v for k, v in metrics.items()},
        "mean_epoch_time_sec": mean_epoch_time, "sd_epoch_time_sec": sd_epoch_time,
        "train_time_sec": train_time, "inference_time_sec": inference_time,
        "inference_ms_per_image": inference_time * 1000 / len(evaluation_ds), "throughput_images_sec": len(evaluation_ds) / inference_time,
        "peak_training_gpu_memory_mb": peak_training_mb,
        "peak_inference_gpu_memory_mb": peak_mb,
        "total_parameters": total_params, "trainable_parameters": trainable_params,
        "lgts_alpha_initial": lgts_alpha if lgts_alpha is not None else "",
        "lgts_alpha": alpha if alpha is not None else "", "lgts_alpha_layer_sd": parameter_stats["alpha_sd"] if alpha is not None else "",
        "lgts_alpha_layer_min": parameter_stats["alpha_min"] if alpha is not None else "", "lgts_alpha_layer_max": parameter_stats["alpha_max"] if alpha is not None else "",
        "lgts_alpha_near_lower_fraction": parameter_stats["alpha_near_lower_fraction"] if alpha is not None else "",
        "lgts_beta_initial": args.lgts_beta if (
            beta is not None or parameter_stats["beta_plus_mean"] is not None
        ) else "", "lgts_beta": beta if beta is not None else "",
        "lgts_beta_layer_sd": parameter_stats["beta_sd"] if beta is not None else "",
        "lgts_beta_min": parameter_stats["beta_min"] if beta is not None else "", "lgts_beta_max": parameter_stats["beta_max"] if beta is not None else "",
        "lgts_beta_plus": parameter_stats["beta_plus_mean"] if parameter_stats["beta_plus_mean"] is not None else "",
        "lgts_beta_plus_layer_sd": parameter_stats["beta_plus_sd"] if parameter_stats["beta_plus_mean"] is not None else "",
        "lgts_beta_plus_min": parameter_stats["beta_plus_min"] if parameter_stats["beta_plus_mean"] is not None else "",
        "lgts_beta_plus_max": parameter_stats["beta_plus_max"] if parameter_stats["beta_plus_mean"] is not None else "",
        "lgts_beta_minus": parameter_stats["beta_minus_mean"] if parameter_stats["beta_minus_mean"] is not None else "",
        "lgts_beta_minus_layer_sd": parameter_stats["beta_minus_sd"] if parameter_stats["beta_minus_mean"] is not None else "",
        "lgts_beta_minus_min": parameter_stats["beta_minus_min"] if parameter_stats["beta_minus_mean"] is not None else "",
        "lgts_beta_minus_max": parameter_stats["beta_minus_max"] if parameter_stats["beta_minus_mean"] is not None else "",
        "lgts_beta_gap": parameter_stats["beta_gap_mean"] if parameter_stats["beta_gap_mean"] is not None else "",
        "lgts_beta_gap_layer_sd": parameter_stats["beta_gap_sd"] if parameter_stats["beta_gap_mean"] is not None else "",
        "lgts_beta_ratio": parameter_stats["beta_ratio_mean"] if parameter_stats["beta_ratio_mean"] is not None else "",
        "lgts_beta_ratio_layer_sd": parameter_stats["beta_ratio_sd"] if parameter_stats["beta_ratio_mean"] is not None else "",
        "lgts_layer_count": parameter_stats["count"],
        "lgts_learning": parameter_stats["learning_mode"] or "",
        "lgts_ablation_role": parameter_stats["ablation_role"] or "",
        "lgts_formula": parameter_stats["formula"] or "",
        "lgts_alpha_constraint": f"[{args.lgts_alpha_min},{args.lgts_alpha_max}]" if alpha is not None else "",
        "lgts_beta_constraint": f"[{args.lgts_beta_min},{args.lgts_beta_max}]" if (
            beta is not None or parameter_stats["beta_plus_mean"] is not None
        ) else "",
        "acon_p1": adaptive_stats["acon_p1_mean"] if adaptive_stats["acon_p1_mean"] is not None else "",
        "acon_p1_sd": adaptive_stats["acon_p1_sd"] if adaptive_stats["acon_p1_mean"] is not None else "",
        "acon_p2": adaptive_stats["acon_p2_mean"] if adaptive_stats["acon_p2_mean"] is not None else "",
        "acon_p2_sd": adaptive_stats["acon_p2_sd"] if adaptive_stats["acon_p2_mean"] is not None else "",
        "acon_beta": adaptive_stats["acon_beta_mean"] if adaptive_stats["acon_beta_mean"] is not None else "",
        "acon_beta_sd": adaptive_stats["acon_beta_sd"] if adaptive_stats["acon_beta_mean"] is not None else "",
        "qian_omega": adaptive_stats["qian_omega_mean"] if adaptive_stats["qian_omega_mean"] is not None else "",
        "qian_omega_sd": adaptive_stats["qian_omega_sd"] if adaptive_stats["qian_omega_mean"] is not None else "",
        "adaptive_activation_layer_count": adaptive_stats["adaptive_layer_count"],
        "augmentation": args.augmentation, "normalization": args.normalization,
        "class_weight_requested": args.class_weight, "class_weight": class_weight_mode,
        "loss_name": "cross_entropy", "evaluation_loss_weighting": class_weight_mode,
        f"{evaluation_split}_loss_weighting": class_weight_mode,
        "stem_type": getattr(model, "stem_type", "model_default"),
        "activation_site_policy": getattr(model, "activation_site_policy", "model_default"),
        "train_count_before_aug": len(train_ds), "train_count_after_aug": len(train_ds),
        "optimizer": args.optimizer, "scheduler": args.scheduler, "learning_rate_initial": args.learning_rate,
        "learning_rate_final": history[-1]["learning_rate"], "weight_decay": args.weight_decay,
        "fused_optimizer": args.fused_optimizer, "best_state_device": args.best_state_device,
        "performance_mode": args.performance_mode, "tf32": args.tf32,
        "cudnn_benchmark": args.cudnn_benchmark, "channels_last": args.channels_last,
        "validation_count": validation_count, "test_count": len(test_ds) if test_ds is not None else "sealed",
        "image_size": image_size, "batch_size": batch_size,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "python_version": getattr(args, "runtime_metadata", {}).get("python_version", platform.python_version()),
        "torch_version": getattr(args, "runtime_metadata", {}).get("torch_version", torch.__version__),
        "torchvision_version": getattr(args, "runtime_metadata", {}).get("torchvision_version", "unavailable"),
        "cuda_version": getattr(args, "runtime_metadata", {}).get("cuda_runtime_version", torch.version.cuda),
        "cudnn_version": getattr(args, "runtime_metadata", {}).get("cudnn_version", torch.backends.cudnn.version()),
        "gpu_model": getattr(args, "runtime_metadata", {}).get("gpu_model", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"),
        "prelu_parameterization": "scalar_shared" if canonical_activation_name(activation_name).lower().replace("-", "") == "prelu" else "",
        "status": "confirmatory_done" if evaluate_official_test else "selection_done_test_sealed",
    }
    append_csv(Path(args.output_dir) / "results.csv", row)
    update_live_workbook(Path(args.live_workbook_path), row, per_class)
    write_json(done_path, row)
    in_progress_path.unlink(missing_ok=True)
    label = "TEST" if evaluate_official_test else "VALIDATION"
    print(f"{label} | {bundle.name} | {terminal_unit} | {model_name}-{activation_variant}-{effective_scope} | {evaluation_split}_f1={metrics['f1_macro']:.4f} | acc={metrics['accuracy']:.4f} | error={metrics['error_rate_percent']:.2f}% | auc={metrics['roc_auc_macro_ovr']:.4f} | ece={metrics['ece_15']:.4f}")
    del model, criterion, evaluation_loader
    release_accelerator_memory(device)


def main() -> None:
    args = parse_args()
    study_seed_order = args.seed_order or args.seeds
    if len(study_seed_order) != len(set(study_seed_order)):
        raise ValueError("--seed_order must contain unique seed values")
    missing_seed_labels = [seed for seed in args.seeds if seed not in study_seed_order]
    if missing_seed_labels:
        raise ValueError(
            "Every --seeds value must appear in --seed_order; missing: "
            + ", ".join(str(seed) for seed in missing_seed_labels)
        )
    project_root = Path(__file__).resolve().parent
    identity = prepare_experiment_identity(args, project_root)
    output_root = Path(args.output_dir)
    args.output_root = str(output_root)
    args.experiment_id = identity["experiment_id"]
    args.code_sha256 = identity["code_sha256"]
    args.git_commit = identity["git_commit"]
    args.runtime_metadata = identity["runtime"]
    args.output_dir = str(output_root / f"exp_{args.experiment_id}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    configure_runtime(args)
    if args.shared_live_workbook:
        requested_live_workbook = Path(args.live_workbook or "live_results.xlsx")
        live_workbook = (
            requested_live_workbook
            if requested_live_workbook.is_absolute()
            else project_root / requested_live_workbook
        )
    else:
        live_name = Path(args.live_workbook).name if args.live_workbook else "live_results.xlsx"
        live_workbook = Path(args.output_dir) / live_name
    if live_workbook.suffix.lower() != ".xlsx":
        live_workbook = live_workbook / "live_results.xlsx"
    args.live_workbook_path = str(live_workbook)
    write_json(Path(args.output_dir) / "experiment_identity.json", identity)
    (Path(args.output_dir) / "environment_lock.txt").write_text(
        "\n".join(identity["package_lock"]) + "\n", encoding="utf-8",
    )
    priority = {
        "LGTS-f4-Base-Squareplus": 0,
        "LGTS-f4-Fixed": 1,
        "LGTS-f4-Alpha": 2,
        "LGTS-f4-Beta": 3,
        "LGTS-f4-Full": 4,
        "LGTS-f2-Slope": 5,
        "LGTS-f1": 6,
        "LGTS-f3-Dual": 7,
        "Native": 8,
        "Swish": 9,
    }
    args.activations = sorted(
        enumerate(args.activations),
        key=lambda item: (priority.get(canonical_activation_name(item[1]), 100), item[0]),
    )
    args.activations = [name for _, name in args.activations]
    conditions_by_model = {
        model_name: sum(
            len(args.lgts_alphas) if is_lgts_activation(name) else 1
            for name in args.activations
            if not is_duplicate_control(model_name, name, args.activations, args.keep_duplicate_controls)
        )
        for model_name in args.models
    }
    split_multiplier = 1 if args.protocol in {"fixed", "full_train"} else args.k_folds
    planned_runs = (
        len(args.datasets) * len(args.seeds) * split_multiplier
        * sum(conditions_by_model.values())
    )
    condition_text = ",".join(f"{name}:{count}" for name, count in conditions_by_model.items())
    print(f"Plan | datasets={len(args.datasets)} | models={len(args.models)} | seeds={len(args.seeds)} | conditions={condition_text} | planned_runs={planned_runs}")
    print(f"Experiment | id={args.experiment_id} | code_sha256={args.code_sha256} | output={Path(args.output_dir).resolve()}")
    if args.dry_run:
        return
    legacy_checkpoints = [*Path(args.output_dir).rglob("best.pt"), *Path(args.output_dir).rglob("last.pt")]
    for checkpoint in legacy_checkpoints:
        checkpoint.unlink(missing_ok=True)
    if legacy_checkpoints:
        print(f"Legacy epoch checkpoints removed | count={len(legacy_checkpoints)}")
    initialize_live_workbook(Path(args.live_workbook_path))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable in the installed PyTorch build.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"Device | {device} | CUDA available={torch.cuda.is_available()}")
    print(f"Initialization | pretrained={not args.no_pretrained} | mode={'imagenet' if not args.no_pretrained else 'random'}")
    print(f"Runtime | performance_mode={args.performance_mode} | tf32={args.tf32} | cudnn_benchmark={args.cudnn_benchmark} | channels_last={args.channels_last} | amp={not args.no_amp} | fused_optimizer={args.fused_optimizer} | best_state={args.best_state_device}")
    print(
        f"Protocol | {args.protocol} | stage_seeds={len(args.seeds)} | "
        f"study_seed_order={','.join(str(seed) for seed in study_seed_order)} | "
        f"augmentation={args.augmentation} | normalization={args.normalization}"
    )
    print(f"Activation order | {' '.join(args.activations)}")
    write_json(Path(args.output_dir) / "run_config.json", vars(args))
    config_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(Path(args.output_dir) / "run_configs" / f"run_config_{config_stamp}.json", vars(args))
    for dataset_name in args.datasets:
        bundle = load_bundle(dataset_name, args.data_dir, args.med_size, download=True)
        manifest = dataset_manifest(bundle); write_json(Path(args.output_dir) / f"manifest_{bundle.name}.json", manifest)
        print(f"Dataset | {bundle.name} | raw_train={len(bundle.train)} | raw_validation={len(bundle.val)} | raw_test={len(bundle.test)} | classes={len(bundle.class_names)}")
        for seed in args.seeds:
            repeat_no = study_seed_order.index(seed) + 1
            if args.protocol in {"fixed", "full_train"}:
                split_specs = [(1, None, None)]
            else:
                splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
                split_specs = [
                    (fold_no, train_idx, val_idx)
                    for fold_no, (train_idx, val_idx) in enumerate(
                        splitter.split(np.zeros(len(bundle.labels_train_val)), bundle.labels_train_val),
                        start=1,
                    )
                ]
            for model_name in args.models:
                for activation in args.activations:
                    if is_duplicate_control(model_name, activation, args.activations, args.keep_duplicate_controls):
                        print(f"SKIP DUPLICATE | {model_name} | {activation} is represented by {native_activation_variant(model_name)}")
                        continue
                    alpha_grid = args.lgts_alphas if is_lgts_activation(activation) else [None]
                    for lgts_alpha in alpha_grid:
                        for fold_no, train_idx, val_idx in split_specs:
                            train_one(
                                args, bundle, model_name, activation, lgts_alpha,
                                seed, fold_no, repeat_no, train_idx, val_idx, device,
                            )
    build_aggregate_reports(Path(args.output_dir) / "results.csv", Path(args.output_dir))
    print(f"Completed | results={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
