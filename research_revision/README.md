# LGTS-Alpha experimental code and outcomes

The original comparison used three datasets, three CNN backbones, five activations and five paired seeds (225 runs). The subsequent OrganSMNIST sensitivity panel changed horizontal flipping (75 runs). ImageNet-100, the activation controls and timing experiments are separate evidence sets. The [original protocol](protocols/Q2_CONFIRMATORY_PROTOCOL.md) and its [machine-readable specification](protocols/protocol_q2_confirmatory.json) record the design and planned analysis. The date of 19 August 2026 comes from the internal record. An independent timestamp establishing that the protocol preceded official test evaluation is unavailable.

## Code for the revision experiments

- [Training](training/README.md) lists the recorded settings and commands for the 960 completed revision runs, including the ImageNet-100 extension, the validation-only learning-rate sweep and the OrganSMNIST rotation control.
- [Benchmarks](benchmarks/README.md) covers activation microbenchmarks, eager and compiled training steps, and full epochs timed with the data loader.
- [Analysis](analysis/README.md) reproduces the paired activation and orientation comparisons and the uncertainty estimates from the released data.

The training engine records a source fingerprint with each run and checkpoint. The earlier engine is archived in commit `cbc7baf51d662d4ec030fa3ae21cb41ea7ce4ab8` under `research_revision/engine`. See the [training entry point](engine/train_benchmark.py) for the version in this checkout. Reruns from this checkout receive a new source fingerprint.

## Inspect the results

- [Reviewer tables](results/reviewer_tables/README.md) contain the 300 original and flip-sensitivity seed scores, statistical comparisons, activation controls and sagittal organ confusion matrices.
- [Training diagnostics](results/diagnostics/README.md) contain layer gradients, learned activation parameters and validation losses from the learning-rate pilot.
- [Measurement details](results/measurement_details/) accompany the activation, training-step and whole-epoch timing results.
- [Per-run archive index](results/release_index/README.md) maps each of the 960 completed revision runs to its files in the 48 numbered ZIP archives. Reviewers can download those archives through the view-only link in the manuscript. The same files are available under the release tag `major-revision-data-2026-11-08`.

From this directory, with Python, NumPy and SciPy installed:

```sh
python verify.py
python analysis/revision_statistics.py --output-dir reproduced_revision_statistics
```

The first command checks the payload hashes, relational keys and panel sizes. It reconstructs the original Friedman test and mixed-model and whole-seed intervals for Alpha versus ACON-C and checks the PReLU replay and matched-epoch coverage. The second reproduces the five-seed comparisons and crossed sensitivity analyses for the complete 780-run base revision panel from a portable statistical snapshot. It writes its tables to a separate directory. The original fixed comparison and augmentation sensitivity retain their original inference families.

| File in results/ | Contents |
| --- | --- |
| run_settings.csv, seed_results.csv | One record per completed training run, linked by run_id. Use panel, dataset, model, activation, variant, seed and assessment split when selecting a comparison. |
| condition_summaries.csv, training_history.csv, class_metrics.csv | Seed-level aggregate, epoch-level and class-level records, linked by condition_id or run_id. Empty entries identify fields unavailable in that source. The original full-budget fits pooled the training and validation partitions, so they have training-loss curves but no independent validation-loss curves. |
| candidate_validation.csv | 863 exploratory validation records across four development archives, including 257 CIFAR-10 fold records. Development scores were excluded from the confirmatory statistics. |
| four_dataset_accuracy_f1.csv | Accuracy and macro-F1 in the original five-activation panels, with five seeds in each cell. Consult run_settings.csv for the separate eight-activation ImageNet-100 revision panel. |
| timing_measurements.csv, timing_blocks.csv, timing_memory.csv, timing_exclusions.csv, latency_ratios.csv | Scope, backend, stage, batch, precision, validation, dispersion and exclusions for the cost measurements. Join blocks and memory to measurement_id. The unit of throughput depends on the operation being timed. |
| real_epochs.csv, real_epoch_ratios.csv, real_epoch_ratio_members.csv, matched_update_contexts.json | First epoch with the actual loader, six arms per dataset–backbone combination. Original and matched-update protocols remain separate. Initial AMP scale 256 allowed all optimizer updates in the 72 matched arms. A single epoch per arm yields a duration and no estimate of within-arm epoch variability. |
| prelu_fp16_slope_replays.json, prelu_training_gradient.json, pelu_observed_range*.json | FP16 diagnostics and their input/gradient protocols. All four legacy PReLU replays returned a nonfinite learned-slope gradient. Their recorded times are diagnostic. |
| sagittal_*.csv, loss_landscape_grid.csv, mixed_model*.csv or .json | OrganSMNIST class and confusion changes, two checkpoint-specific surfaces, and mixed-model contrasts. |

The 1,335 final run records include the original 225, the historical 75-run flip sensitivity, 75 original ImageNet-100 runs, 780 base revision runs, 30 validation pilots, 120 additional ImageNet-100 runs and two 15-run follow-ups. Grouping by panel preserves differences in training duration, numerical implementation, augmentation and split. The candidate-validation table is separate. Protocol counts are in protocols/panel_counts.json. The 136,000 epoch records include the longer-budget panel. Reported metric scales and missingness follow their original files.

## Run the engine

The engine directory contains the revision implementation in PyTorch. Check the arguments before launching a training run:

```sh
python engine/train_benchmark.py --help
```

A single matching example for the stable CIFAR-100 protocol, with the data prepared in a user-supplied directory, is:

```sh
python engine/train_benchmark.py --datasets cifar100 --models ResNet18 --activations LGTS-f4-Alpha --lgts_alphas 0.40 --seeds 936137 --protocol full_train --epochs 100 --patience 0 --batch_size 128 --optimizer adamw --fused_optimizer --learning_rate 0.001 --weight_decay 0.0001 --scheduler cosine --min_learning_rate 0.000001 --no_pretrained --tf32 --cudnn_benchmark --activation_scope full_backbone --data_dir ./data --output_dir ./reproduced
```

Use the recorded row in run_settings.csv to select the appropriate seed, scope, sharing, augmentation, precision and learning rate for any other condition. The original source snapshots and protocol files are separate from the revised implementation. Running the revised engine does not reconstruct historical FP16 arithmetic. The image data and learned weights must be obtained independently. Data requirements, environment and versioned protocols are included with the archive. Statistical reanalysis of the supplied tables does not require a GPU.
