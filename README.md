# LGTS-Alpha

This repository contains the training code, locked protocols, numerical results,
and analysis scripts for the multi-dataset evaluation of LGTS-Alpha, a
learnable smooth activation function for image classification.

The manuscript names `LGTS-f4-Alpha` as **LGTS-Alpha** and
`LGTS-f2-Slope` as **LGTS-Slope**. The implementation labels are retained in
the files so that the published results remain traceable to the recorded runs.

## Release scope

The primary experiment contains:

- three datasets: CIFAR-100, DermaMNIST, and OrganSMNIST;
- three backbones: ResNet18, MobileNetV2, and EfficientNetV2-S;
- five activation conditions: LGTS-Alpha, LGTS-Slope, ACON-C, GELU, and
  each backbone's native activation;
- five paired random seeds and 100 fixed epochs per condition;
- 225 completed runs with no missing, duplicated, or off-protocol condition.

A separate OrganSMNIST sensitivity experiment contains 75 additional runs in
which horizontal flipping was removed while the other settings were retained.
It is reported separately and does not alter the primary statistical analysis.

This release contains only the experiments and derived results reported in the
manuscript, together with one explicitly identified validation-only exploration
workbook. Dataset files, model checkpoints, prediction arrays, and local training
logs are not included.

## Repository contents

| Path | Contents |
|---|---|
| `lgts_benchmark/` | Activation functions, dataset handling, models, metrics, and reporting |
| `train_benchmark.py` | Main training entry point |
| `protocol_q2_confirmatory.json` | Primary protocol fixed on 2026-08-19 |
| `protocol_organsmnist_no_hflip_sensitivity.json` | Post-confirmatory sensitivity protocol fixed on 2026-08-31 |
| `results/confirmatory/` | Run-level records, summaries, statistical tests, costs, figures, and workbook |
| `results/organsmnist_no_hflip/` | Paired sensitivity results and workbook |
| `results/learned_alpha/` | Layer-level alpha workbook and the corresponding figure |
| `results/dataset_manifests/` | Class counts, split fingerprints, and source metadata; no images |
| `exploration/live_results_f4.xlsx` | Validation-only F4 exploration; excluded from all confirmatory analyses |
| `reproducibility/` | Exact source snapshots identified by the recorded SHA-256 hashes |
| `tests/` | Unit and smoke tests |

`protocol_v2.json` and `protocol_f4_ablation.json` are retained as historical
development-stage records. They are not definitions of the final confirmatory
panel; the final panel is defined by `protocol_q2_confirmatory.json`.

## Installation

Python 3.11–3.14 is supported. The reference runs used Python 3.14.6,
PyTorch 2.12.1+cu132, and torchvision 0.27.1+cu132 on an NVIDIA RTX A5500.
The complete reference package list is provided in
`environment/reference_environment.txt`.

Create an isolated environment and install a PyTorch build suitable for the
available hardware, followed by the remaining dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install torch and torchvision using the command recommended at pytorch.org.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CPU execution is sufficient for unit tests and dry runs. Repeating the full
300-run study requires a CUDA-capable GPU and substantial computation time.

Verify the published result files, exact source snapshots, and run counts with:

```powershell
.\.venv\Scripts\python.exe verify_release.py
```

## Dataset preparation

The repository downloads the datasets from their original distribution
channels. No dataset is redistributed here.

```powershell
.\.venv\Scripts\python.exe prepare_datasets.py `
  --datasets cifar100 dermamnist organsmnist `
  --med_size 64 `
  --data_dir .\data
```

See `DATASETS.md` and `results/dataset_manifests/` for sources, licenses,
class counts, split definitions, and integrity fingerprints.

## Reproducing the protocol

Inspect the complete 225-run plan without starting training:

```powershell
.\run_q2_confirmatory.ps1 -DryRun
```

Start the primary study:

```powershell
.\run_q2_confirmatory.ps1
```

After the primary study has completed, the separate OrganSMNIST sensitivity
study can be run with:

```powershell
.\run_organsmnist_no_hflip_sensitivity.ps1
```

The run scripts skip completed conditions. An interrupted condition restarts
at epoch 1 because epoch checkpoints were not used in the fixed protocol.

## Reproducing the reported analyses

The primary analysis can be regenerated directly from the published run-level
CSV file:

```powershell
.\.venv\Scripts\python.exe analyze_q2_confirmatory.py `
  --input_csv .\results\confirmatory\run_results.csv `
  --output_dir .\reproduced\confirmatory `
  --expected_runs 225
```

The OrganSMNIST paired sensitivity analysis can be regenerated with:

```powershell
.\.venv\Scripts\python.exe analyze_organsmnist_no_hflip_sensitivity.py `
  --original-csv .\results\confirmatory\run_results.csv `
  --sensitivity-csv .\results\organsmnist_no_hflip\run_results.csv `
  --output-dir .\reproduced\organsmnist_no_hflip `
  --expected-runs 75
```

`REPRODUCIBILITY.md` describes the source fingerprints and integrity checks.
`RESULTS.md` provides a file-level guide to the published outputs.

## License

The software is released under the MIT License. Third-party datasets are not
included and remain subject to their original terms.
