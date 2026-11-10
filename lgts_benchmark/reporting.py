from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from scipy.stats import friedmanchisquare, t, wilcoxon


def _with_derived_metrics(row: dict) -> dict:
    item = dict(row)
    accuracy = item.get("test_accuracy", "")
    if accuracy not in {"", None, "nan"}:
        try:
            accuracy_value = float(accuracy)
            if item.get("test_error_rate", "") in {"", None, "nan"}:
                item["test_error_rate"] = 1.0 - accuracy_value
            if item.get("test_error_rate_percent", "") in {"", None, "nan"}:
                item["test_error_rate_percent"] = 100.0 * (1.0 - accuracy_value)
        except (TypeError, ValueError):
            pass
    return item


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict] = []
    old_fields: list[str] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f); old_fields = list(reader.fieldnames or []); old_rows = [_with_derived_metrics(item) for item in reader]
    row = _with_derived_metrics(row)
    key_fields = ("experiment_id", "protocol", "dataset", "model", "activation_scope", "activation_variant", "seed", "fold")
    key = tuple(str(row.get(field, "")) for field in key_fields)
    old_rows = [item for item in old_rows if tuple(str(item.get(field, "")) for field in key_fields) != key]
    old_rows.append(row)
    fields = list(row) + [field for field in old_fields if field not in row]
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(old_rows)
    os.replace(temp, path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


RUN_PRIORITY = [
    "test_f1_macro", "validation_f1_macro", "mean_epoch_time_sec", "best_validation_f1",
    "dataset", "model", "activation_scope", "activation", "activation_variant", "experiment_id", "code_sha256", "git_commit",
    "protocol", "split_seed", "training_seed", "model_seed", "data_seed", "repeat", "seed", "fold", "pretrained", "weight_initialization", "selection_policy",
    "test_precision_macro", "test_recall_macro", "test_mcc", "train_time_sec", "lgts_beta",
    "test_accuracy", "test_top5_accuracy", "test_error_rate", "test_error_rate_percent",
    "validation_accuracy", "validation_top5_accuracy", "validation_error_rate", "validation_error_rate_percent",
    "test_balanced_accuracy", "test_precision_weighted", "test_recall_weighted",
    "test_f1_weighted", "test_specificity_macro", "test_cohen_kappa", "test_roc_auc_macro_ovr",
    "test_roc_auc_weighted_ovr", "test_pr_auc_macro", "test_nll", "test_brier_score", "test_ece_15",
    "test_loss", "sd_epoch_time_sec", "inference_time_sec", "inference_ms_per_image",
    "throughput_images_sec", "peak_training_gpu_memory_mb", "peak_inference_gpu_memory_mb", "total_parameters", "trainable_parameters",
    "lgts_alpha_initial", "lgts_alpha", "lgts_alpha_layer_sd", "lgts_alpha_layer_min", "lgts_alpha_layer_max", "lgts_alpha_near_lower_fraction",
    "lgts_beta_initial", "lgts_beta_layer_sd", "lgts_beta_min", "lgts_beta_max",
    "lgts_beta_plus", "lgts_beta_plus_layer_sd", "lgts_beta_plus_min", "lgts_beta_plus_max",
    "lgts_beta_minus", "lgts_beta_minus_layer_sd", "lgts_beta_minus_min", "lgts_beta_minus_max",
    "lgts_beta_gap", "lgts_beta_gap_layer_sd", "lgts_beta_ratio", "lgts_beta_ratio_layer_sd",
    "lgts_layer_count", "lgts_learning", "lgts_ablation_role", "lgts_formula", "lgts_alpha_constraint", "lgts_beta_constraint",
    "acon_p1", "acon_p1_sd", "acon_p2", "acon_p2_sd", "acon_beta", "acon_beta_sd",
    "qian_omega", "qian_omega_sd", "adaptive_activation_layer_count",
    "best_epoch", "epochs_run", "evaluation_split",
    "optimizer", "scheduler", "learning_rate_initial", "learning_rate_final", "weight_decay", "fused_optimizer", "best_state_device",
    "performance_mode", "tf32", "cudnn_benchmark", "channels_last",
    "augmentation", "normalization", "class_weight_requested", "class_weight", "loss_name", "evaluation_loss_weighting",
    "stem_type", "activation_site_policy", "dataset_fingerprint_sha256",
    "train_count_before_aug", "train_count_after_aug", "validation_count", "test_count",
    "image_size", "batch_size", "python_version", "torch_version", "torchvision_version", "cuda_version", "cudnn_version", "gpu_model",
    "prelu_parameterization", "source", "status", "timestamp",
]

SUMMARY_METRICS = [
    "best_validation_f1", "validation_f1_macro", "validation_accuracy", "validation_top5_accuracy", "validation_error_rate", "validation_error_rate_percent",
    "validation_balanced_accuracy", "validation_precision_macro", "validation_recall_macro", "validation_mcc", "validation_specificity_macro",
    "validation_roc_auc_macro_ovr", "validation_pr_auc_macro", "validation_nll", "validation_brier_score", "validation_ece_15", "validation_loss",
    "test_f1_macro", "test_accuracy", "test_top5_accuracy", "mean_epoch_time_sec", "test_precision_macro", "test_recall_macro", "test_mcc",
    "train_time_sec", "lgts_beta", "test_error_rate", "test_error_rate_percent", "test_balanced_accuracy", "test_precision_weighted",
    "test_recall_weighted", "test_f1_weighted", "test_specificity_macro", "test_cohen_kappa",
    "test_roc_auc_macro_ovr", "test_roc_auc_weighted_ovr", "test_pr_auc_macro", "test_nll",
    "test_brier_score", "test_ece_15", "test_loss", "inference_ms_per_image", "throughput_images_sec",
    "peak_training_gpu_memory_mb", "peak_inference_gpu_memory_mb", "lgts_alpha", "lgts_alpha_layer_min", "lgts_alpha_layer_max", "lgts_alpha_near_lower_fraction", "lgts_beta_layer_sd", "lgts_beta_plus",
    "lgts_beta_minus", "lgts_beta_gap", "lgts_beta_ratio",
    "acon_p1", "acon_p2", "acon_beta", "qian_omega",
]

EFFICIENCY_PRIORITY = [
    "model", "activation", "image_size", "batch_size", "num_classes", "mode", "amp",
    "forward_ms_mean", "forward_ms_sd", "inference_images_sec", "forward_peak_mb",
    "forward_backward_ms_mean", "forward_backward_ms_sd", "training_images_sec",
    "forward_backward_peak_mb", "stem_type", "warmup_repeats", "measured_repeats",
    "gpu_model", "torch_version", "cuda_version", "timestamp_utc",
]


def _ordered_columns(rows: list[dict]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set(RUN_PRIORITY)
    return [key for key in RUN_PRIORITY if key in keys] + sorted(keys - set(RUN_PRIORITY))


def _numeric_values(items: list[dict], metric: str) -> list[float]:
    values = []
    for item in items:
        value = item.get(metric, "")
        if value in {"", None, "nan"}: continue
        try:
            number = float(value)
            if np.isfinite(number): values.append(number)
        except (TypeError, ValueError):
            continue
    return values


def _live_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in rows:
        key = (item.get("dataset"), item.get("model"), item.get("activation_scope"), item.get("activation_variant") or item.get("activation"))
        groups[key].append(item)
    result = []
    for key, items in groups.items():
        validation_values = _numeric_values(items, "best_validation_f1")
        test_f1_values = _numeric_values(items, "test_f1_macro")
        epoch_values = _numeric_values(items, "mean_epoch_time_sec")
        out = {
            "validation_f1_mean": mean(validation_values) if validation_values else "",
            "validation_f1_sd": stdev(validation_values) if len(validation_values) > 1 else 0.0,
            "test_f1_mean": mean(test_f1_values) if test_f1_values else "",
            "epoch_time_mean_sec": mean(epoch_values),
            "test_f1_sd": stdev(test_f1_values) if len(test_f1_values) > 1 else 0.0,
            "epoch_time_sd_sec": stdev(epoch_values) if len(epoch_values) > 1 else 0.0,
            "dataset": key[0], "model": key[1], "activation_scope": key[2], "activation_variant": key[3],
            "completed_runs": len(items),
        }
        for metric in SUMMARY_METRICS:
            values = _numeric_values(items, metric)
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        result.append(out)
    return sorted(
        result,
        key=lambda item: (
            float(item["validation_f1_mean"])
            if item["validation_f1_mean"] != ""
            else float(item["test_f1_mean"]) if item["test_f1_mean"] != "" else -np.inf
        ),
        reverse=True,
    )


def _style_table(ws, freeze: str = "A2") -> None:
    ws.freeze_panes = freeze; ws.auto_filter.ref = ws.dimensions; ws.sheet_view.showGridLines = False
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 34
    for column_cells in ws.columns:
        letter = column_cells[0].column_letter
        width = min(28, max(11, max(len(str(cell.value or "")) for cell in column_cells) + 2))
        ws.column_dimensions[letter].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float): cell.number_format = "0.0000"


MODEL_DISPLAY_ORDER = {"ResNet18": 0, "MobileNetV2": 1, "EfficientNetV2S": 2}


def _summary_rank_value(item: dict) -> float:
    value = item.get("validation_f1_mean", "")
    if value == "":
        value = item.get("test_f1_mean", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return -np.inf


def _model_sort_key(item: dict) -> tuple:
    model = str(item.get("model", ""))
    return (
        str(item.get("dataset", "")),
        MODEL_DISPLAY_ORDER.get(model, len(MODEL_DISPLAY_ORDER)),
        model,
        -_summary_rank_value(item),
    )


def _add_model_color_scales(ws, items: list[dict], first_data_row: int = 2) -> None:
    start = first_data_row
    while start < first_data_row + len(items):
        group = (
            items[start - first_data_row].get("dataset"),
            items[start - first_data_row].get("model"),
        )
        end = start
        while (
            end + 1 < first_data_row + len(items)
            and (
                items[end + 1 - first_data_row].get("dataset"),
                items[end + 1 - first_data_row].get("model"),
            ) == group
        ):
            end += 1
        ws.conditional_formatting.add(
            f"A{start}:A{end}",
            ColorScaleRule(
                start_type="min", start_color="F8696B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="63BE7B",
            ),
        )
        start = end + 1


def _write_model_summary(ws, summaries: list[dict], confirmatory_view: bool) -> None:
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Model-Based Activation Summary"
    ws["A1"].font = Font(size=18, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="17365D")
    ws.merge_cells("A1:I1")
    ws.append([])
    headers = (
        ["Test F1 mean", "Epoch time mean (s)", "Test F1 SD", "Validation F1 mean", "Validation F1 SD", "Error rate mean (%)", "Activation", "Runs", "Scope"]
        if confirmatory_view else
        ["Validation F1 mean", "Epoch time mean (s)", "Test F1 mean", "Validation F1 SD", "Test F1 SD", "Error rate mean (%)", "Activation", "Runs", "Scope"]
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in summaries:
        grouped[(item.get("dataset"), item.get("model"))].append(item)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda pair: (
            str(pair[0][0]),
            MODEL_DISPLAY_ORDER.get(str(pair[0][1]), len(MODEL_DISPLAY_ORDER)),
            str(pair[0][1]),
        ),
    )
    for (dataset, model), items in ordered_groups:
        items = sorted(items, key=_summary_rank_value, reverse=True)
        ws.append([f"{dataset} | {model}"])
        title_row = ws.max_row
        ws.merge_cells(start_row=title_row, start_column=1, end_row=title_row, end_column=9)
        ws.cell(title_row, 1).font = Font(bold=True, color="FFFFFF")
        ws.cell(title_row, 1).fill = PatternFill("solid", fgColor="5B9BD5")
        ws.append(headers)
        header_row = ws.max_row
        for cell in ws[header_row]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="4472C4")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        data_start = ws.max_row + 1
        for item in items:
            error_rate = item.get("test_error_rate_percent_mean", "")
            if error_rate == "":
                error_rate = item.get("validation_error_rate_percent_mean", "")
            score_cells = (
                [item["test_f1_mean"], item["epoch_time_mean_sec"], item["test_f1_sd"], item["validation_f1_mean"], item["validation_f1_sd"]]
                if confirmatory_view else
                [item["validation_f1_mean"], item["epoch_time_mean_sec"], item["test_f1_mean"], item["validation_f1_sd"], item["test_f1_sd"]]
            )
            ws.append(score_cells + [error_rate, item["activation_variant"], item["completed_runs"], item["activation_scope"]])
        data_end = ws.max_row
        if data_end >= data_start:
            ws.conditional_formatting.add(
                f"A{data_start}:A{data_end}",
                ColorScaleRule(
                    start_type="min", start_color="F8696B",
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color="63BE7B",
                ),
            )
        ws.append([])
    ws.freeze_panes = "A4"
    for column, width in {"A": 20, "B": 18, "C": 16, "D": 18, "E": 16, "F": 20, "G": 38, "H": 11, "I": 20}.items():
        ws.column_dimensions[column].width = width
    for row in ws.iter_rows(min_row=3):
        for cell in row[:6]:
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"


def _write_live_sheets(
    wb: Workbook,
    rows: list[dict],
    per_class_rows: list[dict],
    efficiency_rows: list[dict] | None = None,
    protocol_rows: list[dict] | None = None,
    tracker_title: str = "LGTS v2 Validation-First Experiment Tracker",
) -> None:
    for name in list(wb.sheetnames): del wb[name]
    dashboard = wb.create_sheet("Live_Dashboard")
    model_summary_ws = wb.create_sheet("Model_Summary")
    summary_ws = wb.create_sheet("Summary")
    runs_ws = wb.create_sheet("Runs")
    per_class_ws = wb.create_sheet("Per_Class")
    efficiency_ws = wb.create_sheet("Efficiency_Benchmark")
    protocol_ws = wb.create_sheet("Protocol")
    summaries = _live_summary(rows)

    dashboard.sheet_view.showGridLines = False
    dashboard["A1"] = tracker_title; dashboard["A1"].font = Font(size=18, bold=True, color="FFFFFF")
    dashboard["A1"].fill = PatternFill("solid", fgColor="17365D"); dashboard.merge_cells("A1:K1")
    dashboard["A3"] = "Completed runs"; dashboard["B3"] = len(rows)
    dashboard["A4"] = "Last update"; dashboard["B4"] = rows[-1].get("timestamp", "") if rows else ""
    confirmatory_view = (
        tracker_title.startswith("LGTS Q2 Confirmatory")
        or bool(rows) and all(item.get("protocol") == "full_train" for item in rows)
    )
    dashboard["A6"] = "Confirmatory view — ranked by Test Macro-F1" if confirmatory_view else "Selection view — ranked only by Validation Macro-F1"
    dashboard["A6"].font = Font(bold=True, color="17365D")
    dash_headers = (
        ["Test F1 mean", "Epoch time mean (s)", "Test F1 SD", "Validation F1 mean", "Validation F1 SD", "Error rate mean (%)", "Dataset", "Model", "Scope", "Activation", "Runs"]
        if confirmatory_view else
        ["Validation F1 mean", "Epoch time mean (s)", "Test F1 mean", "Validation F1 SD", "Test F1 SD", "Error rate mean (%)", "Dataset", "Model", "Scope", "Activation", "Runs"]
    )
    dashboard.append([]); dashboard.append(dash_headers)
    for item in summaries[:20]:
        error_rate = item.get("test_error_rate_percent_mean", "")
        if error_rate == "":
            error_rate = item.get("validation_error_rate_percent_mean", "")
        score_cells = (
            [item["test_f1_mean"], item["epoch_time_mean_sec"], item["test_f1_sd"], item["validation_f1_mean"], item["validation_f1_sd"]]
            if confirmatory_view else
            [item["validation_f1_mean"], item["epoch_time_mean_sec"], item["test_f1_mean"], item["validation_f1_sd"], item["test_f1_sd"]]
        )
        dashboard.append(score_cells + [error_rate, item["dataset"], item["model"], item["activation_scope"], item["activation_variant"], item["completed_runs"]])
    for cell in dashboard[8]:
        cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="4472C4"); cell.alignment = Alignment(wrap_text=True)
    dashboard.freeze_panes = "A8"; dashboard.column_dimensions["A"].width = 18; dashboard.column_dimensions["B"].width = 16
    for col in "CDEFGHIJK": dashboard.column_dimensions[col].width = 20
    for row_index in range(9, dashboard.max_row + 1):
        dashboard.cell(row_index, 5).number_format = "0.00"
        dashboard.cell(row_index, 6).number_format = "0.00"

    _write_model_summary(model_summary_ws, summaries, confirmatory_view)

    model_sorted_summaries = sorted(summaries, key=_model_sort_key)
    summary_headers = list(summaries[0]) if summaries else ["validation_f1_mean", "validation_f1_sd", "test_f1_mean", "epoch_time_mean_sec", "test_f1_sd", "epoch_time_sd_sec", "test_error_rate_percent_mean", "dataset", "model", "activation_scope", "activation_variant", "completed_runs"]
    summary_ws.append(summary_headers)
    for item in model_sorted_summaries: summary_ws.append([item.get(header, "") for header in summary_headers])
    _style_table(summary_ws, "C2")
    _add_model_color_scales(summary_ws, model_sorted_summaries)

    model_sorted_rows = sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset", "")),
            MODEL_DISPLAY_ORDER.get(str(item.get("model", "")), len(MODEL_DISPLAY_ORDER)),
            str(item.get("model", "")),
            int(item.get("repeat") or 0),
            int(item.get("seed") or 0),
            str(item.get("activation_variant") or item.get("activation") or ""),
        ),
    )
    run_headers = _ordered_columns(rows)
    runs_ws.append(run_headers)
    for item in model_sorted_rows: runs_ws.append([item.get(header, "") for header in run_headers])
    _style_table(runs_ws, "C2")
    _add_model_color_scales(runs_ws, model_sorted_rows)

    per_class_headers = [
        "dataset", "model", "activation_scope", "activation_variant", "experiment_id", "protocol", "evaluation_split", "training_seed", "repeat", "seed", "fold",
        "class_index", "class_name", "precision", "recall_sensitivity", "specificity",
        "f1", "support", "roc_auc_ovr", "pr_auc",
    ]
    per_class_ws.append(per_class_headers)
    model_sorted_per_class = sorted(
        per_class_rows,
        key=lambda item: (
            str(item.get("dataset", "")),
            MODEL_DISPLAY_ORDER.get(str(item.get("model", "")), len(MODEL_DISPLAY_ORDER)),
            str(item.get("model", "")),
            int(item.get("repeat") or 0),
            int(item.get("seed") or 0),
            str(item.get("activation_variant") or ""),
            int(item.get("class_index") or 0),
        ),
    )
    for item in model_sorted_per_class: per_class_ws.append([item.get(header, "") for header in per_class_headers])
    _style_table(per_class_ws, "G2")

    efficiency_rows = efficiency_rows or []
    efficiency_keys = set().union(*(item.keys() for item in efficiency_rows)) if efficiency_rows else set()
    efficiency_headers = (
        [key for key in EFFICIENCY_PRIORITY if key in efficiency_keys]
        + sorted(efficiency_keys - set(EFFICIENCY_PRIORITY))
    ) if efficiency_rows else [
        "model", "activation", "image_size", "batch_size",
        "forward_ms_mean", "forward_backward_ms_mean",
        "inference_images_sec", "training_images_sec",
        "forward_backward_peak_mb", "gpu_model",
    ]
    efficiency_ws.append(efficiency_headers)
    for item in sorted(
        efficiency_rows,
        key=lambda row: (
            int(float(row.get("image_size") or 0)),
            MODEL_DISPLAY_ORDER.get(str(row.get("model", "")), len(MODEL_DISPLAY_ORDER)),
            str(row.get("activation", "")),
        ),
    ):
        efficiency_ws.append([item.get(header, "") for header in efficiency_headers])
    _style_table(efficiency_ws, "C2")

    protocol_rows = protocol_rows or []
    protocol_ws.append(["Field", "Locked value"])
    for item in protocol_rows:
        protocol_ws.append([item.get("field", ""), item.get("value", "")])
    _style_table(protocol_ws, "A2")
    protocol_ws.column_dimensions["A"].width = 34
    protocol_ws.column_dimensions["B"].width = 90
    for row in protocol_ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(vertical="top", wrap_text=True)


def _read_efficiency_rows(workbook_path: Path) -> list[dict]:
    rows: list[dict] = []
    for source in sorted((workbook_path.parent / "efficiency").glob("efficiency_*.csv")):
        with source.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(item) for item in csv.DictReader(handle))
    return rows


def _read_protocol_rows(workbook_path: Path) -> list[dict]:
    if "q2" not in workbook_path.stem.lower():
        return []
    protocol_path = Path(__file__).resolve().parents[1] / "protocol_q2_confirmatory.json"
    if not protocol_path.exists():
        return []
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    selected = {
        "protocol_version": protocol.get("protocol_version"),
        "status": protocol.get("status"),
        "purpose": protocol.get("purpose"),
        "datasets": protocol.get("datasets"),
        "models": protocol.get("models"),
        "activations": protocol.get("activations"),
        "seeds": protocol.get("seeds"),
        "planned_runs": protocol.get("planned_runs"),
        "primary_metric": protocol.get("primary_metric"),
        "primary_comparator": protocol.get("primary_comparator"),
        "noninferiority_margin_macro_f1": protocol.get("analysis", {}).get("noninferiority_margin_macro_f1"),
        "resume_policy": protocol.get("resume_policy"),
        "claim_boundary": protocol.get("claim_boundary"),
    }
    return [
        {
            "field": key,
            "value": ", ".join(str(value) for value in item) if isinstance(item, list) else item,
        }
        for key, item in selected.items()
    ]


def update_live_workbook(path: Path, row: dict, per_class: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.stem}_pending{path.suffix}")
    candidates = [p for p in (path, pending) if p.exists()]
    source = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    rows: list[dict] = []
    per_class_rows: list[dict] = []
    if source:
        old = load_workbook(source, read_only=True, data_only=True)
        if "Runs" in old.sheetnames:
            values = old["Runs"].iter_rows(values_only=True)
            headers = next(values, ())
            rows = [_with_derived_metrics(dict(zip(headers, values_row))) for values_row in values]
        if "Per_Class" in old.sheetnames:
            values = old["Per_Class"].iter_rows(values_only=True)
            headers = next(values, ())
            per_class_rows = [dict(zip(headers, values_row)) for values_row in values]
        old.close()
    unique = (row.get("experiment_id"), row.get("dataset"), row.get("model"), row.get("activation_scope"), row.get("activation_variant"), row.get("seed"), row.get("fold"))
    rows = [item for item in rows if (item.get("experiment_id"), item.get("dataset"), item.get("model"), item.get("activation_scope"), item.get("activation_variant"), item.get("seed"), item.get("fold")) != unique]
    rows.append(_with_derived_metrics(row))
    per_class_rows = [item for item in per_class_rows if (item.get("experiment_id"), item.get("dataset"), item.get("model"), item.get("activation_scope"), item.get("activation_variant"), item.get("seed"), item.get("fold")) != unique]
    for class_row in per_class or []:
        per_class_rows.append({
            "dataset": row.get("dataset"), "model": row.get("model"), "activation_scope": row.get("activation_scope"),
            "activation_variant": row.get("activation_variant"), "experiment_id": row.get("experiment_id"),
            "protocol": row.get("protocol"), "evaluation_split": row.get("evaluation_split"),
            "training_seed": row.get("training_seed"), "repeat": row.get("repeat"),
            "seed": row.get("seed"), "fold": row.get("fold"), **class_row,
        })
    tracker_title = (
        "LGTS Q2 Confirmatory Official-Test Tracker"
        if "q2" in path.stem.lower()
        else "LGTS v2 150-Epoch Validation-First Experiment Tracker"
        if path.stem.lower().endswith("_150")
        else "LGTS v2 Validation-First Experiment Tracker"
    )
    wb = Workbook(); _write_live_sheets(
        wb, rows, per_class_rows,
        _read_efficiency_rows(path), _read_protocol_rows(path), tracker_title,
    )
    temp = path.with_name(f"{path.stem}_tmp{path.suffix}")
    wb.save(temp)
    try:
        os.replace(temp, path)
        if pending.exists(): pending.unlink()
    except PermissionError:
        os.replace(temp, pending)
        print(f"LIVE EXCEL LOCKED | wrote pending workbook: {pending}", flush=True)


def initialize_live_workbook(path: Path) -> None:
    if path.exists() or path.with_name(f"{path.stem}_pending{path.suffix}").exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tracker_title = (
        "LGTS Q2 Confirmatory Official-Test Tracker"
        if "q2" in path.stem.lower()
        else "LGTS v2 150-Epoch Validation-First Experiment Tracker"
        if path.stem.lower().endswith("_150")
        else "LGTS v2 Validation-First Experiment Tracker"
    )
    wb = Workbook(); _write_live_sheets(
        wb, [], [], _read_efficiency_rows(path), _read_protocol_rows(path), tracker_title,
    )
    wb.save(path)


def build_aggregate_reports(results_csv: Path, output_dir: Path) -> None:
    if not results_csv.exists(): return
    with results_csv.open(encoding="utf-8") as f: rows = [_with_derived_metrics(item) for item in csv.DictReader(f)]
    metrics = SUMMARY_METRICS
    score_key = "test_f1_macro" if any(r.get("test_f1_macro", "") not in {"", "nan"} for r in rows) else "best_validation_f1"
    score_label = "Test macro-F1" if score_key == "test_f1_macro" else "Best validation macro-F1"
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows: groups[(r["dataset"], r["model"], r["activation_scope"], r.get("activation_variant") or r["activation"], r.get("protocol"), r.get("gpu_model"))].append(r)
    summary = []
    for key, items in groups.items():
        out = dict(zip(("dataset", "model", "activation_scope", "activation", "protocol", "gpu_model"), key)); out["n"] = len(items)
        for metric in metrics:
            vals = [float(x[metric]) for x in items if x.get(metric, "") not in {"", "nan"}]
            metric_mean = mean(vals) if vals else ""
            metric_sd = stdev(vals) if len(vals) > 1 else 0.0 if vals else ""
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_sd"] = metric_sd
            if len(vals) > 1:
                half_width = float(t.ppf(0.975, len(vals) - 1) * metric_sd / np.sqrt(len(vals)))
                out[f"{metric}_ci95_low"] = metric_mean - half_width
                out[f"{metric}_ci95_high"] = metric_mean + half_width
            elif vals:
                out[f"{metric}_ci95_low"] = metric_mean
                out[f"{metric}_ci95_high"] = metric_mean
            else:
                out[f"{metric}_ci95_low"] = ""
                out[f"{metric}_ci95_high"] = ""
        summary.append(out)
    summary_path = output_dir / "summary.csv"
    if summary:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    stats = []
    scopes = sorted(set((r["dataset"], r["model"], r["activation_scope"], r.get("protocol"), r.get("gpu_model")) for r in rows))
    for dataset, model, scope, protocol, gpu_model in scopes:
        block = [r for r in rows if (r["dataset"], r["model"], r["activation_scope"], r.get("protocol"), r.get("gpu_model")) == (dataset, model, scope, protocol, gpu_model)]
        paired: dict[str, dict[tuple, float]] = defaultdict(dict)
        for r in block: paired[r.get("activation_variant") or r["activation"]][(r["seed"], r["fold"])] = float(r[score_key])
        lgts_variants = [a for a in paired if a.startswith("LGTS")]
        if not lgts_variants: continue
        common_all = set.intersection(*(set(v) for v in paired.values())) if paired else set()
        if len(paired) >= 3 and len(common_all) >= 2:
            arrays = [[paired[a][k] for k in sorted(common_all)] for a in sorted(paired)]
            try: stat, p = friedmanchisquare(*arrays)
            except ValueError: stat = p = np.nan
            stats.append({"dataset": dataset, "model": model, "scope": scope, "protocol": protocol, "gpu_model": gpu_model, "metric": score_key, "test": "Friedman", "comparison": "all", "n": len(common_all), "statistic": stat, "p_value": p})
        for lgts_variant in lgts_variants:
            for competitor, values in paired.items():
                if competitor.startswith("LGTS"): continue
                common = sorted(set(values) & set(paired[lgts_variant]));
                if len(common) < 2: continue
                x = [paired[lgts_variant][k] for k in common]; y = [values[k] for k in common]
                try: stat, p = wilcoxon(x, y, zero_method="zsplit")
                except ValueError: stat = p = np.nan
                stats.append({"dataset": dataset, "model": model, "scope": scope, "protocol": protocol, "gpu_model": gpu_model, "metric": score_key, "test": "Wilcoxon", "comparison": f"{lgts_variant} vs {competitor}", "n": len(common), "statistic": stat, "p_value": p, "mean_difference": mean(np.asarray(x)-np.asarray(y))})
        for f1_name in sorted(name for name in paired if name.startswith("LGTS-f1")):
            alpha_suffix = f1_name[len("LGTS-f1"):]
            planned = [
                (f1_name, "Swish", "tanh_branch"),
                (f"LGTS-f2-Slope{alpha_suffix}", f1_name, "slope_matching"),
                (f"LGTS-f3-Dual{alpha_suffix}", f"LGTS-f2-Slope{alpha_suffix}", "untied_gate_rates"),
            ]
            for treatment, control, contribution in planned:
                if treatment not in paired or control not in paired:
                    continue
                common = sorted(set(paired[treatment]) & set(paired[control]))
                if len(common) < 2:
                    continue
                x = np.asarray([paired[treatment][key] for key in common], dtype=float)
                y = np.asarray([paired[control][key] for key in common], dtype=float)
                differences = x - y
                try: stat, p = wilcoxon(x, y, zero_method="zsplit")
                except ValueError: stat = p = np.nan
                difference_sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                stats.append({
                    "dataset": dataset, "model": model, "scope": scope,
                    "protocol": protocol, "gpu_model": gpu_model, "metric": score_key,
                    "test": "Wilcoxon_planned", "comparison": f"{treatment} vs {control}",
                    "contribution": contribution, "n": len(common), "statistic": stat,
                    "p_value": p, "mean_difference": float(np.mean(differences)),
                    "paired_cohen_dz": float(np.mean(differences) / difference_sd) if difference_sd > 0 else np.nan,
                })
    families: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(stats):
        if str(row.get("test", "")).startswith("Wilcoxon") and np.isfinite(row.get("p_value", np.nan)):
            families[(row["dataset"], row["model"], row["scope"], row.get("protocol"), row.get("gpu_model"))].append(index)
    for indices in families.values():
        ordered = sorted(indices, key=lambda index: float(stats[index]["p_value"]))
        running = 0.0; total = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * float(stats[index]["p_value"]))
            running = max(running, adjusted)
            stats[index]["p_value_holm"] = running
    if stats:
        with (output_dir / "statistical_tests.csv").open("w", newline="", encoding="utf-8") as f:
            priority = ["dataset", "model", "scope", "protocol", "gpu_model", "metric", "test", "comparison", "contribution", "n", "statistic", "p_value", "p_value_holm", "mean_difference", "paired_cohen_dz"]
            keys = set().union(*(row.keys() for row in stats))
            fields = [field for field in priority if field in keys] + sorted(keys - set(priority))
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(stats)
    _aggregate_plots(rows, output_dir, score_key, score_label)


def _aggregate_plots(rows: list[dict], output_dir: Path, score_key: str, score_label: str) -> None:
    if not rows: return
    plot_dir = output_dir / "aggregate_plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(set(r["dataset"] for r in rows))
    for dataset in datasets:
        block = [r for r in rows if r["dataset"] == dataset]
        activations = sorted(set(r.get("activation_variant") or r["activation"] for r in block))
        values = [[float(r[score_key]) for r in block if (r.get("activation_variant") or r["activation"]) == a] for a in activations]
        fig, ax = plt.subplots(figsize=(max(8, len(activations) * 0.8), 5))
        ax.boxplot(values, tick_labels=activations, showmeans=True); ax.set_ylabel(score_label); ax.set_title(f"{dataset}: activation comparison")
        ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(plot_dir / f"{dataset}_activation_boxplot.png", dpi=180); plt.close(fig)
    activations = sorted(set(r.get("activation_variant") or r["activation"] for r in rows))
    matrix = np.full((len(datasets), len(activations)), np.nan)
    for i, d in enumerate(datasets):
        for j, a in enumerate(activations):
            vals = [float(r[score_key]) for r in rows if r["dataset"] == d and (r.get("activation_variant") or r["activation"]) == a]
            if vals: matrix[i, j] = mean(vals)
    fig, ax = plt.subplots(figsize=(max(9, len(activations) * 0.85), max(4, len(datasets) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", xticklabels=activations, yticklabels=datasets, ax=ax)
    ax.set_title(f"Mean {score_label.lower()}"); fig.tight_layout(); fig.savefig(plot_dir / "dataset_activation_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for a in activations:
        part = [r for r in rows if (r.get("activation_variant") or r["activation"]) == a]
        ax.scatter(mean([float(r["inference_ms_per_image"]) for r in part]), mean([float(r[score_key]) for r in part]), label=a, s=55)
    ax.set(xlabel="Evaluation inference time (ms/image; lower is better)", ylabel=score_label, title="Accuracy-cost trade-off"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "accuracy_cost_pareto.png", dpi=180); plt.close(fig)
    dual = [r for r in rows if r.get("lgts_beta_plus", "") not in {"", None} and r.get("lgts_beta_minus", "") not in {"", None}]
    if dual:
        beta_plus = np.asarray([float(r["lgts_beta_plus"]) for r in dual])
        beta_minus = np.asarray([float(r["lgts_beta_minus"]) for r in dual])
        low = min(beta_plus.min(), beta_minus.min()); high = max(beta_plus.max(), beta_minus.max())
        margin = max(0.05, 0.08 * (high - low if high > low else 1.0))
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for activation in sorted(set(r.get("activation_variant", r.get("activation")) for r in dual)):
            part = [r for r in dual if r.get("activation_variant", r.get("activation")) == activation]
            ax.scatter([float(r["lgts_beta_minus"]) for r in part], [float(r["lgts_beta_plus"]) for r in part], label=activation, alpha=0.8)
        ax.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="gray", label="beta+ = beta-")
        ax.set(xlim=(low - margin, high + margin), ylim=(low - margin, high + margin), xlabel="Learned beta-", ylabel="Learned beta+", title="Dual-gate parameter divergence")
        ax.grid(alpha=0.2); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot_dir / "dual_gate_beta_divergence.png", dpi=180); plt.close(fig)
