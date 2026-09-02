from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from lgts_benchmark.activations import AconC, LGTSBase
from lgts_benchmark.models import build_model, count_parameters


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_channels: int
    num_classes: int
    image_size: int


DATASETS = (
    DatasetSpec("cifar100", 3, 100, 32),
    DatasetSpec("dermamnist", 3, 7, 64),
    DatasetSpec("organsmnist", 1, 11, 64),
)
MODELS = ("ResNet18", "MobileNetV2", "EfficientNetV2S")
ACTIVATIONS = ("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")


def is_activation_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            LGTSBase,
            AconC,
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
        ),
    )


def profile(model: nn.Module, sample: torch.Tensor) -> dict[str, int]:
    """Count conventional Conv2d/Linear MACs for one forward pass.

    A multiply-accumulate is reported as one MAC and two FLOPs. Bias,
    normalization, pooling, residual addition and element-wise activation
    operations are deliberately excluded. Their treatment differs between
    profiling packages, particularly for sigmoid/tanh/sqrt activations, so
    measured latency is retained as the hardware-aware efficiency measure.
    """

    totals = {
        "conv_macs": 0,
        "linear_macs": 0,
        "conv_layers": 0,
        "linear_layers": 0,
        "activation_sites": 0,
        "activation_elements": 0,
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        totals["conv_macs"] += int(output.numel() * kernel_ops)
        totals["conv_layers"] += 1

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        totals["linear_macs"] += int(output.numel() * module.in_features)
        totals["linear_layers"] += 1

    def activation_hook(_module: nn.Module, inputs, _output) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            totals["activation_elements"] += int(inputs[0].numel())
            totals["activation_sites"] += 1

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        if is_activation_module(module):
            handles.append(module.register_forward_hook(activation_hook))

    try:
        model.eval()
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()

    return totals


def build_row(dataset: DatasetSpec, model_name: str, activation: str) -> dict[str, object]:
    model = build_model(
        model_name,
        dataset.in_channels,
        dataset.num_classes,
        activation,
        activation_scope="full_backbone",
        pretrained=False,
        small_input_stem=True,
        lgts_alpha=0.40,
        lgts_beta=1.0,
    ).cpu()
    sample = torch.zeros(1, dataset.in_channels, dataset.image_size, dataset.image_size)
    counts = profile(model, sample)
    total_parameters, trainable_parameters = count_parameters(model)
    total_macs = counts["conv_macs"] + counts["linear_macs"]
    total_flops = 2 * total_macs
    return {
        "dataset": dataset.name,
        "model": model_name,
        "activation": activation,
        "input_shape": f"1x{dataset.in_channels}x{dataset.image_size}x{dataset.image_size}",
        "num_classes": dataset.num_classes,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "conv_layers": counts["conv_layers"],
        "linear_layers": counts["linear_layers"],
        "activation_sites": counts["activation_sites"],
        "activation_elements": counts["activation_elements"],
        "conv_macs": counts["conv_macs"],
        "linear_macs": counts["linear_macs"],
        "total_macs": total_macs,
        "gmacs": total_macs / 1e9,
        "total_flops": total_flops,
        "gflops": total_flops / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile exact confirmatory models without retraining")
    parser.add_argument(
        "--output",
        default="outputs/q2_confirmatory/complexity/model_complexity.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model_name in MODELS:
            for activation in ACTIVATIONS:
                row = build_row(dataset, model_name, activation)
                rows.append(row)
                print(
                    f"{dataset.name} | {model_name} | {activation} | "
                    f"{row['gmacs']:.6f} GMAC | {row['gflops']:.6f} GFLOPs | "
                    f"{row['total_parameters']:,} parameters"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methodology = {
        "input_batch_size": 1,
        "included_operations": ["Conv2d multiply-accumulate", "Linear multiply-accumulate"],
        "excluded_operations": [
            "bias",
            "normalization",
            "pooling",
            "residual addition",
            "element-wise activation arithmetic and transcendental functions",
        ],
        "conversion": "1 MAC = 2 FLOPs",
        "reason_for_exclusions": (
            "Element-wise sigmoid, tanh, erf and sqrt costs are implementation- and hardware-dependent. "
            "They are represented by activation_elements and compared using measured latency."
        ),
        "datasets": [spec.__dict__ for spec in DATASETS],
        "models": list(MODELS),
        "activations": list(ACTIVATIONS),
        "torch_version": torch.__version__,
    }
    metadata_path = output.with_name("model_complexity_method.json")
    metadata_path.write_text(json.dumps(methodology, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved | {output.resolve()}")
    print(f"Saved | {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
