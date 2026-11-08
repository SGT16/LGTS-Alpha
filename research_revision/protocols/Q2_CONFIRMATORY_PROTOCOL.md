# LGTS-f4-Alpha Q2 Confirmatory Protocol

This protocol was frozen before any official test evaluation in the Q2 phase. All earlier Stage 1–3 results are validation-only discovery evidence.

## Locked design

- Datasets: CIFAR-100, DermaMNIST and OrganSMNIST.
- Backbones: ResNet18, MobileNetV2 and EfficientNetV2S.
- Activations: LGTS-f4-Alpha, LGTS-f2-Slope, ACON-C, GELU and each backbone's Native activation.
- Five paired seeds: `936137`, `529212`, `448132`, `942529`, `110107`.
- Seeds were sampled once with Python's `secrets` module before test access and will not be replaced after observing results.
- Random initialization, full-backbone replacement, online standard augmentation, dataset normalization, AdamW, cosine learning-rate schedule and 100 fixed epochs.
- Train and validation partitions are combined for training. The official test partition is evaluated once after epoch 100. There is no test-driven early stopping or checkpoint selection.
- Planned total: 3 datasets × 3 backbones × 5 activations × 5 seeds = 225 independent training runs.

## Primary analysis

The primary outcome is test macro-F1. The confirmatory block is one dataset–model pair; each block score is the mean of its five paired seeds. A Friedman test compares all five activations across the nine blocks. LGTS-f4-Alpha is then compared with ACON-C, LGTS-f2-Slope, GELU and Native using paired two-sided Wilcoxon signed-rank tests with Holm correction.

Non-inferiority uses a predeclared margin of 0.01 macro-F1. The 95% bootstrap lower confidence bound of the paired block difference must be greater than −0.01. Superiority will not be claimed without a positive effect and multiplicity-corrected statistical support.

## Mathematical properties of LGTS-f4-Alpha

For learned `alpha > 0` and fixed `beta = 1`, the activation is

`f(x) = 0.5 * (x + sqrt(x^2 + alpha^2)) - alpha * sigmoid(-beta*x)`.

It satisfies:

- `f(0) = 0`.
- `f(x) -> x` as `x -> +infinity`.
- `f(x) -> -alpha` as `x -> -infinity`.
- `f'(x) = 0.5 * (1 + x/sqrt(x^2+alpha^2)) + alpha*beta*q*(1-q)`, where `q = sigmoid(-beta*x)`. Both terms are positive for finite `x`, so the function is strictly increasing.
- `f'(0) = 0.5 + alpha*beta/4`.
- `0 < f'(x) < 1 + alpha*beta/4`; under the locked alpha range and beta value, the global upper bound is below `1.375`.
- The function is smooth for every allowed alpha.

The CUDA implementation uses the defining Squareplus expression directly. This is algebraically identical to the prior branchwise form while avoiding eager computation of an unused division branch at every activation site.

## Reproducibility and reporting

Each completed run records predictions, logits, per-class metrics, learned activation parameters, dataset fingerprint, package lock, code hash, GPU/runtime versions and all performance/efficiency metrics. The live workbook ranks only official-test results in this confirmatory phase. Interrupted runs restart at epoch 1; completed runs are never repeated unless explicitly requested.
