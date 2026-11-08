# Gradients, learned α and learning-rate sensitivity

`run_id` identifies the training run in [run_settings.csv](../run_settings.csv) and [seed_results.csv](../seed_results.csv). Accuracy and macro-F1 use the 0–1 scale.

## Gradients

CIFAR-100/ResNet18 used five paired seeds, two activations and 100 epochs. Each sample was recorded on the first training minibatch of an epoch after AMP gradient unscaling.

| File | Contents |
| --- | --- |
| [gradient_parameters.csv.gz](gradient_parameters.csv.gz) | 70,500 original parameter observations, including gradient L2, maximum absolute gradient, parameter L2, finite flag and prior AMP scale. ReLU has 62 parameter tensors and LGTS-Alpha has 79. |
| [gradient_norms_by_epoch.csv](gradient_norms_by_epoch.csv) | 1,000 seed–epoch rows. Each row gives the global L2 across the 62 shared parameters and the second convolution in the final residual block of each of the four stages. |
| [gradient_profiles.svg](gradient_profiles.svg) | Four stage-specific curves. Each line averages five seeds and its band spans their minimum and maximum. |

The global norm equals the square root of the sum of squared parameter-gradient norms. The raw archive includes the 17 LGTS-specific α tensors as well. To reproduce the profile table and plot from these observations, run `python analysis/gradient_profiles.py --output-dir reproduced/gradients` from `research_revision/`.

## Activation parameters

| File | Contents |
| --- | --- |
| [learned_alpha_by_site.csv](learned_alpha_by_site.csv) | All 2,310 final site-specific α values from 45 runs of the revised implementation. Each of the three backbones has five seeds on CIFAR-100, DermaMNIST and OrganSMNIST. The table records the layer name, initialization, bounds and β. |
| [alpha_sensitivity.csv](alpha_sensitivity.csv) | Final test accuracy and macro-F1 on CIFAR-100/ResNet18 for five paired seeds at fixed α = 0.1, 0.5, 1 and 2, and learned per-site α initialized at 0.4. |
| [alpha_ablation.svg](alpha_ablation.svg) | Individual macro-F1 scores and mean ± sample SD for the 25 fits. The activation curves use the minimum, median and maximum of the 85 learned α values from this comparison. |

## Learning-rate validation pilot

The CIFAR-100/EfficientNetV2S pilot used 45,000 training images and 5,000 validation images with a fixed split. Five activations were each trained from scratch with three initial learning rates and the same two seeds. Selection used final-epoch validation macro-F1 averaged over those seeds. The official test split remained held out.

| File | Contents |
| --- | --- |
| [learning_rate_validation.csv](learning_rate_validation.csv) | All 30 final validation results with evaluation precision, attempt count, recorded errors and epochs with nonfinite intermediate validation loss. |
| [learning_rate_groups.csv](learning_rate_groups.csv) | Fifteen activation–rate combinations with both seed scores, mean, sample SD and the rate selected per activation. |
| [common_lr_loss.csv](common_lr_loss.csv) | All 1,000 training and validation loss pairs at initial learning rate 0.001, covering ten runs and 100 epochs each. |

The separate 100- and 200-epoch training-budget comparison can be regenerated from [training_history.csv](../training_history.csv) using [training_budgets.py](../../analysis/training_budgets.py).
