# Reproducibility record

## Protocol separation

The primary panel and the later augmentation sensitivity analysis are kept as
two distinct experiments:

| Experiment | Protocol date | Runs | Recorded code SHA-256 |
|---|---:|---:|---|
| Primary confirmatory panel | 2026-08-19 | 225 | `988cfad419b8ae4613f37e9c4b4c513e39cd56e278ed0e23fc4bfd88d2bdf301` |
| OrganSMNIST no-horizontal-flip sensitivity | 2026-08-31 | 75 | `cc8753bdf13348a1cf6add379d5ad899da8b6dddada7e9c1f0d81496e3a38d72` |

The complete primary design and analysis plan were internally frozen before the
first official-test evaluation. Each condition was evaluated only after its
fixed 100-epoch budget, and no protocol change was made thereafter. The
protocol was not registered in an external preregistration registry. The
sensitivity protocol was specified after completion of the primary analysis
and before the 75 additional runs. Those runs were not added to the primary
Friedman, Wilcoxon, Holm, or non-inferiority calculations.

## Exact source snapshots

The root source tree is the maintained public interface. Byte-identical source
snapshots corresponding to the two recorded experiment hashes are preserved in:

- `reproducibility/source_snapshot_q2_confirmatory/`
- `reproducibility/source_snapshot_organsmnist_no_hflip/`

The snapshot file hashes reproduce the combined code hashes written into the
experiment records. The original runs were completed before this Git repository
was created; consequently, their recorded `git_commit` field is unavailable.
The per-file SHA-256 map in each `results/**/provenance.json` file is the
authoritative source identity for those runs.

## Data identity

`results/dataset_manifests/` records the dataset source, class names, class
counts, official split sizes, label hashes, source checksums where available,
and the study-specific dataset fingerprint. These files contain metadata only.

The main protocol combined each dataset's official training and validation
partitions for the fixed 100-epoch fit and evaluated the final-epoch model once
on the official test partition. No early stopping or best-checkpoint selection
was used in the primary panel.

## Result integrity

- `results/confirmatory/integrity.json` verifies 225 unique primary records,
  with no missing, duplicated, or unexpected condition.
- `results/organsmnist_no_hflip/sensitivity_integrity.json` verifies 75 unique
  sensitivity records and 75 paired comparisons.
- `results/CHECKSUMS.sha256` contains SHA-256 checksums for every published
  result artifact.

Run-level aggregate metrics are in `run_results.csv`; all 100 epoch records for
the primary panel are in `training_history.csv`; class-level test metrics are in
`per_class_results.csv`. Large probability arrays, checkpoints, and redundant
per-run plots were excluded from the repository.

## Reference environment

The recorded reference environment was:

- Windows 11
- Python 3.14.6
- PyTorch 2.12.1+cu132
- torchvision 0.27.1+cu132
- CUDA runtime 13.2
- cuDNN 92000
- NVIDIA RTX A5500

The complete package list captured with the primary run is available at
`environment/reference_environment.txt`. Timing and GPU-memory values are
hardware- and software-stack-dependent; accuracy metrics are the principal
cross-environment reproducibility targets.
