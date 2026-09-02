from __future__ import annotations

import argparse
import csv
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from lgts_benchmark.models import build_model


def configure(mode: str) -> bool:
    tf32 = mode in {"tf32", "all"}
    benchmark_mode = mode in {"cudnn", "all"}
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = benchmark_mode
    torch.backends.cudnn.deterministic = not benchmark_mode
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return mode in {"channels_last", "all"}


def autocast_context(enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16) if enabled else nullcontext()


def summarize(samples: list[float]) -> tuple[float, float]:
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def measure_forward(model: nn.Module, x: torch.Tensor, warmup: int, repeats: int, amp: bool) -> tuple[float, float, float]:
    samples: list[float] = []
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), autocast_context(amp):
            model(x)
        torch.cuda.synchronize()
        if index == warmup - 1:
            torch.cuda.reset_peak_memory_stats()
        if index >= warmup:
            samples.append(1000.0 * (time.perf_counter() - started))
    mean_ms, sd_ms = summarize(samples)
    return mean_ms, sd_ms, torch.cuda.max_memory_allocated() / 1024**2


def measure_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup: int,
    repeats: int,
    amp: bool,
) -> tuple[float, float, float]:
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    samples: list[float] = []
    for index in range(warmup + repeats):
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context(amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        if index == warmup - 1:
            torch.cuda.reset_peak_memory_stats()
        if index >= warmup:
            samples.append(1000.0 * (time.perf_counter() - started))
    mean_ms, sd_ms = summarize(samples)
    return mean_ms, sd_ms, torch.cuda.max_memory_allocated() / 1024**2


def benchmark(
    model_name: str,
    activation: str,
    batch_size: int,
    in_channels: int,
    image_size: int,
    num_classes: int,
    warmup: int,
    repeats: int,
    mode: str,
    amp: bool,
) -> dict:
    channels_last = configure(mode)
    device = torch.device("cuda")
    model = build_model(
        model_name, in_channels, num_classes, activation,
        activation_scope="full_backbone", pretrained=False,
        small_input_stem=image_size <= 64,
        lgts_alpha=0.40, lgts_beta=1.0,
    ).to(device).train()
    x = torch.randn(batch_size, in_channels, image_size, image_size, device=device, requires_grad=True)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last).detach().requires_grad_(True)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    forward = measure_forward(model.eval(), x, warmup, repeats, amp)
    backward = measure_forward_backward(model.train(), x, y, warmup, repeats, amp)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu_model": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": model_name,
        "activation": activation,
        "mode": mode,
        "amp": amp,
        "batch_size": batch_size,
        "in_channels": in_channels,
        "image_size": image_size,
        "num_classes": num_classes,
        "warmup_repeats": warmup,
        "measured_repeats": repeats,
        "stem_type": getattr(model, "stem_type", "model_default"),
        "forward_ms_mean": forward[0],
        "forward_ms_sd": forward[1],
        "inference_images_sec": batch_size * 1000.0 / forward[0],
        "forward_peak_mb": forward[2],
        "forward_backward_ms_mean": backward[0],
        "forward_backward_ms_sd": backward[1],
        "training_images_sec": batch_size * 1000.0 / backward[0],
        "forward_backward_peak_mb": backward[2],
    }
    del model, x, y
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-model activation forward/backward CUDA microbenchmark")
    parser.add_argument("--models", nargs="+", default=["ResNet18", "MobileNetV2", "EfficientNetV2S"])
    parser.add_argument(
        "--activations", nargs="+",
        default=["Native", "ReLU", "ELU", "Softplus", "Qian-Gated", "ACON-C", "LGTS-f2-Slope"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=["baseline"], choices=["baseline", "tf32", "cudnn", "channels_last", "all"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_csv", default="", help="Optional machine-readable benchmark table")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    print(f"GPU | {torch.cuda.get_device_name(0)}")
    print(
        f"Input | batch={args.batch_size} | channels={args.in_channels} | "
        f"image={args.image_size} | classes={args.num_classes}"
    )
    results = []
    for mode in args.modes:
        for model_name in args.models:
            for activation in args.activations:
                result = benchmark(
                    model_name, activation, args.batch_size, args.in_channels, args.image_size,
                    args.num_classes, args.warmup, args.repeats, mode, not args.no_amp,
                )
                results.append(result)
                print(
                    f"{result['model']} | {result['activation']} | {result['mode']} | {result['stem_type']} | "
                    f"forward={result['forward_ms_mean']:.3f}+/-{result['forward_ms_sd']:.3f}ms | "
                    f"fwd+bwd={result['forward_backward_ms_mean']:.3f}+/-{result['forward_backward_ms_sd']:.3f}ms | "
                    f"memory={result['forward_backward_peak_mb']:.0f}MB"
                )
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved | {output.resolve()}")


if __name__ == "__main__":
    main()
