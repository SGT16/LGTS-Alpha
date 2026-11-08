from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

from .activations import AsymmetricDualGateLGTS, LGTSBase
from .numerics import NumericalError, require_finite


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            score += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(score)


def nll_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))
    selected = values[np.arange(len(y_true)), np.asarray(y_true, dtype=int)]
    return float(np.mean(log_normalizer - selected))


def classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    logits: np.ndarray | None = None,
) -> tuple[dict, list[dict], np.ndarray]:
    n_classes = len(class_names)
    require_finite(probs, "classification probabilities")
    if logits is not None:
        require_finite(logits, "classification logits")
    if probs.shape != (len(y_true), n_classes):
        raise NumericalError("Classification probability shape does not match labels/classes")
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise NumericalError("Classification probabilities are outside the probability simplex")
    labels = np.arange(n_classes)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=labels)
    p, r, f, support = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    onehot = label_binarize(y_true, classes=labels)
    if n_classes == 2 and onehot.shape[1] == 1:
        onehot = np.column_stack((1 - onehot[:, 0], onehot[:, 0]))
    specificity = []
    for i in labels:
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        specificity.append(tn / (tn + fp) if tn + fp else 0.0)
    try:
        auc_per = roc_auc_score(onehot, probs, average=None, multi_class="ovr")
        auc_macro = roc_auc_score(onehot, probs, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(onehot, probs, average="weighted", multi_class="ovr")
    except ValueError:
        auc_per = np.full(n_classes, np.nan); auc_macro = auc_weighted = np.nan
    try:
        ap_per = average_precision_score(onehot, probs, average=None)
        ap_macro = average_precision_score(onehot, probs, average="macro")
    except ValueError:
        ap_per = np.full(n_classes, np.nan); ap_macro = np.nan
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    accuracy = float(accuracy_score(y_true, pred))
    top_k = min(5, n_classes)
    top_k_indices = np.argpartition(probs, -top_k, axis=1)[:, -top_k:]
    top5_accuracy = float(np.mean(np.any(top_k_indices == np.asarray(y_true)[:, None], axis=1)))
    overall = {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "error_rate": 1.0 - accuracy,
        "error_rate_percent": 100.0 * (1.0 - accuracy),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "specificity_macro": float(np.mean(specificity)),
        "mcc": matthews_corrcoef(y_true, pred),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "roc_auc_macro_ovr": auc_macro,
        "roc_auc_weighted_ovr": auc_weighted,
        "pr_auc_macro": ap_macro,
        "nll": nll_from_logits(y_true, logits) if logits is not None else log_loss(
            y_true,
            np.asarray(probs, dtype=np.float64) / np.asarray(probs, dtype=np.float64).sum(axis=1, keepdims=True),
            labels=labels,
        ),
        "brier_score": brier,
        "ece_15": expected_calibration_error(y_true, probs, 15),
    }
    per_class = [{
        "class_index": int(i), "class_name": class_names[i], "precision": float(p[i]),
        "recall_sensitivity": float(r[i]), "specificity": float(specificity[i]),
        "f1": float(f[i]), "support": int(support[i]), "roc_auc_ovr": float(auc_per[i]),
        "pr_auc": float(ap_per[i]),
    } for i in labels]
    return {k: float(v) for k, v in overall.items()}, per_class, cm


def plot_learning_curves(history: list[dict], path: Path) -> None:
    epochs = [x["epoch"] for x in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [x["train_loss"] for x in history], label="train")
    validation_loss = [x.get("loss") for x in history]
    has_validation = any(value is not None for value in validation_loss)
    if has_validation:
        axes[0].plot(epochs, [np.nan if value is None else value for value in validation_loss], label="validation")
    axes[0].set_title("Loss"); axes[0].legend()
    if has_validation:
        axes[1].plot(epochs, [x.get("val_f1", x.get("f1")) for x in history], color="#B91C1C")
        axes[1].set_title("Validation macro-F1")
        axes[2].plot(epochs, [x.get("acc", x.get("val")) for x in history], color="#1D4ED8")
        axes[2].set_title("Validation accuracy")
    else:
        for ax, label in zip(axes[1:], ("No validation split", "Fixed-budget final model")):
            ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(6, min(12, len(class_names) * 0.9))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], roc_path: Path, pr_path: Path) -> None:
    labels = np.arange(len(class_names)); onehot = label_binarize(y_true, classes=labels)
    if len(class_names) == 2 and onehot.shape[1] == 1:
        onehot = np.column_stack((1 - onehot[:, 0], onehot[:, 0]))
    fig_r, ax_r = plt.subplots(figsize=(7, 6)); fig_p, ax_p = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        if len(np.unique(onehot[:, i])) < 2: continue
        fpr, tpr, _ = roc_curve(onehot[:, i], probs[:, i]); pr, rc, _ = precision_recall_curve(onehot[:, i], probs[:, i])
        ax_r.plot(fpr, tpr, label=name); ax_p.plot(rc, pr, label=name)
    ax_r.plot([0, 1], [0, 1], "--", color="gray"); ax_r.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
    ax_p.set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall")
    for ax in (ax_r, ax_p): ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.2)
    fig_r.tight_layout(); fig_r.savefig(roc_path, dpi=180); plt.close(fig_r)
    fig_p.tight_layout(); fig_p.savefig(pr_path, dpi=180); plt.close(fig_p)


def plot_reliability(y_true: np.ndarray, probs: np.ndarray, path: Path) -> None:
    confidence = probs.max(axis=1); correct = (probs.argmax(axis=1) == y_true).astype(int)
    observed, predicted = calibration_curve(correct, confidence, n_bins=15, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(predicted, observed, marker="o", color="#B91C1C")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", title="Reliability diagram"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_lgts_diagnostics(model: nn.Module, history: list[dict], shape_path: Path, trajectory_path: Path) -> None:
    layers = [module for module in model.modules() if isinstance(module, LGTSBase)]
    if not layers:
        return
    layer = layers[0]
    anchor = next(layer.parameters(), None)
    if anchor is None:
        anchor = next(layer.buffers())
    device = anchor.device
    x = torch.linspace(-8.0, 8.0, 2001, device=device, requires_grad=True)
    y = layer(x)
    derivative = torch.autograd.grad(y.sum(), x)[0]
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    d_np = derivative.detach().cpu().numpy()

    columns = 3 if isinstance(layer, AsymmetricDualGateLGTS) else 2
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 4.5))
    axes[0].plot(x_np, y_np, color="#17365D", label=layer.variant)
    axes[0].plot(x_np, x_np, "--", color="gray", alpha=0.6, label="identity")
    axes[0].axhline(0, color="black", linewidth=0.6); axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].set(title="Activation function", xlabel="x", ylabel="f(x)"); axes[0].legend()
    axes[1].plot(x_np, d_np, color="#B91C1C")
    axes[1].axhline(1, color="gray", linestyle="--", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.6); axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set(title="Activation derivative", xlabel="x", ylabel="f'(x)")
    if isinstance(layer, AsymmetricDualGateLGTS):
        with torch.no_grad():
            positive = torch.sigmoid(layer.beta_plus * x.detach())
            negative = torch.sigmoid(-layer.beta_minus * x.detach())
            axes[2].plot(x_np, positive.cpu(), label="positive gate")
            axes[2].plot(x_np, negative.cpu(), label="negative gate")
            axes[2].plot(x_np, (positive + negative).cpu(), "--", label="gate sum")
        axes[2].axhline(1, color="gray", linestyle=":")
        axes[2].set(title="Oppositely oriented gates", xlabel="x", ylabel="gate value"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(shape_path, dpi=180); plt.close(fig)

    epochs = [row["epoch"] for row in history]
    series = [
        ("lgts_alpha", "alpha"),
        ("lgts_beta", "beta"),
        ("lgts_beta_plus", "beta+"),
        ("lgts_beta_minus", "beta-"),
        ("lgts_beta_gap", "beta+ - beta-"),
    ]
    present = [(key, label) for key, label in series if any(row.get(key) is not None for row in history)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in present:
        ax.plot(epochs, [np.nan if row.get(key) is None else row[key] for row in history], label=label)
    ax.set(title="Learned LGTS parameters", xlabel="Epoch", ylabel="Layer mean")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout(); fig.savefig(trajectory_path, dpi=180); plt.close(fig)
