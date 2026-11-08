# Revision benchmarks

Run these commands from `research_revision`. Listing recorded configurations requires Python only.

```sh
python benchmarks/measure.py list epochs
python benchmarks/measure.py list cells
python benchmarks/measure.py list activation-replays
```

The timings were collected on one NVIDIA RTX A5500 with Python 3.14.6, PyTorch 2.12.1+cu132, torchvision 0.27.1+cu132, CUDA 13.2 and cuDNN 92000. Use the repository requirements and the recorded software versions, an idle GPU visible as CUDA device 0, and a new output directory for each invocation. Each measurement writes its environment and numerical evidence. GPU timing blocks are recorded in `blocks.jsonl`.

## One training epoch with the actual data loader

```sh
python benchmarks/measure.py epoch --dataset imagenet100 --model EfficientNetV2S --activation LGTS-f4-Alpha --backend cudagraphs --amp-policy matched-updates --data-dir ./data --output ./measurements/imagenet-alpha-graphs
python benchmarks/measure.py epoch --dataset imagenet100 --model EfficientNetV2S --activation LGTS-f4-Alpha --backend eager --amp-policy matched-updates --data-dir ./data --output ./measurements/imagenet-alpha-eager
```

The 72 arms cross four datasets, three backbones, three activations (LGTS-Alpha, the native backbone activation and GELU), and two backends. Use `--amp-policy original` for the initial scale of 65,536. The matched-update control starts at 256 and checks for a full count of optimizer updates, an unchanged scale, and the recorded initialization, batch order and augmentation fingerprints. Both protocols start from seeded, untrained weights. The timer includes the first epoch's data fetching, transforms, transfers and training steps. Numerical checks, compilation and CUDA Graph warm-up take place before the timer. The compiled arm requires observed forward and backward graph launches.

| Dataset | Training images | Input size | Batch | Steps |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | 50,000 | 32 | 128 | 391 |
| DermaMNIST | 8,010 | 64 | 128 | 63 |
| OrganSMNIST | 16,384 | 64 | 128 | 128 |
| ImageNet-100 | 126,689 | 160 | 64 | 1,980 |

Prepare the data as described in [Training](../training/README.md). The data root contains `torchvision/` for CIFAR and `medmnist/` for MedMNIST. The ImageNet-100 reader uses `datasets==4.8.5` and verifies all 21 entries in [the cache inventory](protocol_data/source_data_manifest.json) before opening the prepared Arrow shards. The inventory gives relative paths, byte counts and SHA-256 digests for the pinned dataset revision.

## Learned-checkpoint measurements

```sh
python benchmarks/measure.py training-step --cell cost-7aee2faefd89bdfbb9a0 --checkpoint ./weights/imagenet-efficientnet-alpha.pt --backend cudagraphs --output ./measurements/step
python benchmarks/measure.py inference --cell cost-7aee2faefd89bdfbb9a0 --checkpoint ./weights/imagenet-efficientnet-alpha.pt --backend eager --output ./measurements/inference
python benchmarks/measure.py operator --cell cost-7aee2faefd89bdfbb9a0 --checkpoint ./weights/imagenet-efficientnet-alpha.pt --site late --batch 32 --kind activation_forward --output ./measurements/operator
```

The 158 cell records identify the dataset, source model, activation, training seed and checkpoint digest. Supply the original final-weights file with the SHA-256 printed by `list cells`. Learned weights and image data must be supplied separately. The program checks checkpoint metadata and tensors before measuring. Synthetic training steps include forward, loss, backward, gradient scaling, AdamW, and parameter projection, and exclude data loading. Compiled training uses the revision's graph-compatible native cuDNN BatchNorm forward and backward and its calibrated gradient checks. Inference records batches 1, 16, 32, 64 and 128. For a failed inference gradient gate, `--scaled-gate` enables the separately reported validation-only loss-scaling probe and restores gradient units for comparison. CUDA event timings retain the original precision.

The activation operator accepts the early or late recorded layer, learned or declared parameter state, and forward, retained-graph backward, or combined forward and backward. Match the state, batch and kind of the target row in [the timing records](../results/timing_measurements.csv). Each accepted microbenchmark uses 50 warm-up iterations and ten timing blocks with at least 100 iterations and 0.5 seconds per block.

## FP16 activation replays

```sh
python benchmarks/measure.py activation-replay --unit pelu-range-db8bc2d371289a3fa809ef5a --checkpoint ./weights/cifar-resnet-pelu.pt --data-dir ./data --output ./measurements/pelu
python benchmarks/measure.py activation-replay --unit prelu-grad-b642316259cf574211b3c5bc --checkpoint ./weights/source-prelu.pt --data-dir ./data --output ./measurements/prelu-gradient
python benchmarks/measure.py activation-replay --unit legacy-prelu-2391 --checkpoint ./weights/legacy-prelu.pt --output ./measurements/prelu-legacy
```

Select an ID and its required checkpoint digest with `list activation-replays`. The 51 PELU arms use the captured first-batch tensor or its batch prefix, and validate against extrema from eight full training batches. The six PReLU training-gradient arms use paired native FP16 inputs and unscaled derivatives of the original weighted or unweighted mean cross-entropy, as specified for the dataset. The four legacy arms reconstruct the seeded Gaussian FP16 inputs and unit upstream gradient, verify both tensor hashes, and record learned-slope gradient finiteness outside the timer after each block. A nonfinite diagnostic retains its evidence and is excluded from speed comparisons.
