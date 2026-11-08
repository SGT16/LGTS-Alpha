# Reviewer result tables

## Original tests and the horizontal-flip follow-up

`original_and_flip_seed_results.csv` contains the 225 original runs and the 75 subsequent OrganSMNIST runs with horizontal flipping disabled (H1-Mo3, H2-08). The `panel` column identifies the two cohorts. Each `run_id` joins to `../run_settings.csv` and `../training_history.csv`. Settings and every recorded seed-result column with at least one observation in these cohorts are retained across the linked files. Blank cells indicate unavailable measurements. `original_secondary_metrics.csv` reports mean test accuracy and macro one-vs-rest ROC-AUC, with sample standard deviations across five seeds for each of the 45 original dataset–backbone–activation conditions (H1-Mo2, H3-m3).

## Paired inference

`original_primary_comparisons.csv` gives the four original paired macro-F1 contrasts against LGTS-Alpha, their confidence intervals and Holm-adjusted tests. The primary comparator was ACON-C. `original_sampling_intervals.csv`, `original_margin_sensitivity.csv` and `original_seed_aggregate_tests.csv` give the subsequent checks under alternative sampling assumptions and margins (H1-M5, H3-08, H8-07, H8-08). The margins are 0.005, 0.010 and 0.020. The preselected original margin was 0.010. `../mixed_model_contrasts.csv` gives the mixed-model and paired whole-seed intervals.

`paired_seed_and_gelu_posthoc.csv` contains 36 original within-context macro-F1 contrasts, 36 original accuracy contrasts, 30 revision-panel CIFAR-100 accuracy contrasts and four post-hoc GELU comparisons across the nine original context means (H1-M5, H7-04, H8-08). Each row gives the paired observations, effect, interval, exact signed-rank test and adjusted p value. Holm correction applies within the named family. With five nonzero seed pairs, the smallest two-sided exact p value is 0.0625.

`original_and_flip_substitution.json` gives the Friedman statistic and p value, ranks, paired tests and crossed bootstrap intervals before and after replacing the three original OrganSMNIST blocks with their seed-matched no-horizontal-flip counterparts. The other six blocks are unchanged. This substitution was performed after the primary analysis. The file also contains the 15 matched augmentation effects.

## Components and orientation

`development_component_screen.csv` gives validation macro-F1 for the five historical component candidates on ResNet18 at seeds 42, 123 and 456 (H1-B3). Each cell is the best validation macro-F1 for the run. The original promotion criterion concerned CIFAR-100 at seed 42. The three-seed and medical panels were subsequent development stages. See `../candidate_validation.csv` for all 863 archived development records. `component_and_scale_controls.csv` reports five-seed means and sample standard deviations for the 121 revision conditions covering components, conventional baselines, fixed scales, sharing and the zero-bound control (H1-B4, H2-02, H3-01).

`sagittal_confusion_seed_counts.csv` gives every cell of the 60 held-out, 11-class LGTS-Alpha confusion matrices across three backbones, four flip policies and five seeds. `sagittal_confusion_condition_rates.csv` gives the 12 matrices plotted in Supplementary Figure S2 as mean row proportions and sample standard deviations over seeds. The seed-level counts divide by their true-class support. All four revision policies retained rotation. The original five-activation horizontal-flip comparison uses a separate set of runs. Its 11 class-F1 and six directed femur, kidney and lung confusion changes appear in `../sagittal_seed_pairs.csv`, `../sagittal_model_activation.csv` and `../sagittal_all_contexts.csv` (H1-B1). The same test images recur across seeds.
