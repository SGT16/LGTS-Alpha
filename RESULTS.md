# Published result files

## Primary 225-run panel

| File | Description |
|---|---|
| `results/confirmatory/run_results.csv` | One row per dataset–backbone–activation–seed condition; aggregate test metrics, timings, memory, parameter counts, and learned activation statistics |
| `results/confirmatory/training_history.csv` | One row per epoch for all 225 runs |
| `results/confirmatory/per_class_results.csv` | Per-class test metrics for every run |
| `results/confirmatory/q2_summary_by_condition.csv` | Mean and sample SD over five paired seeds for each of the 45 conditions |
| `results/confirmatory/q2_dataset_model_blocks.csv` | Five-seed macro-F1 means and ranks for the nine dataset–backbone blocks |
| `results/confirmatory/q2_friedman.csv` | Friedman omnibus test and average ranks |
| `results/confirmatory/q2_pairwise_holm.csv` | Paired differences, bootstrap intervals, Wilcoxon tests, Holm adjustment, and non-inferiority decisions |
| `results/confirmatory/model_complexity.csv` | Parameter counts, Conv/Linear MACs and GFLOPs, and activation-site counts |
| `results/confirmatory/efficiency_*.csv` | Measured latency, throughput, and peak GPU memory |
| `results/confirmatory/live_results_q2.xlsx` | Original multi-sheet results workbook |
| `results/confirmatory/provenance.json` | Sanitized source hashes, protocol configuration, runtime, and package-lock hash |

## OrganSMNIST sensitivity panel

| File | Description |
|---|---|
| `results/organsmnist_no_hflip/run_results.csv` | Aggregate metrics for the 75 no-horizontal-flip runs |
| `results/organsmnist_no_hflip/paired_run_results.csv` | Seed-paired primary-versus-sensitivity values and changes |
| `results/organsmnist_no_hflip/paired_sensitivity_summary.csv` | Mean, sample SD, paired change, and 95% t interval for each condition |
| `results/organsmnist_no_hflip/live_results_organsmnist_no_hflip.xlsx` | Original sensitivity workbook |
| `results/organsmnist_no_hflip/sensitivity_integrity.json` | Completion and pairing audit |

## Learned alpha

`results/learned_alpha/LGTS_learned_alpha_all_details.xlsx` contains run-level,
layer-level, epoch-level, block-level, and figure-ready alpha values together
with a data dictionary. The PNG and SVG files in the same directory show the
layer-wise learned-alpha dynamics used in the manuscript.

## Validation-only exploration

`exploration/live_results_f4.xlsx` is the original F4 candidate-screening
workbook. It documents method development but was not included in any
confirmatory test or effect estimate. The four `before_seed_merge` intermediate
workbooks were intentionally excluded.

All means and standard deviations in the summary tables were calculated over
the same five seeds. Standard deviation denotes the sample standard deviation
unless a file states otherwise.
