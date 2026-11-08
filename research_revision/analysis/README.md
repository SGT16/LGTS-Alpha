# Statistical analyses

Run these commands from `research_revision/`. The first uses the included 780-run snapshot. The others read the data in `results/`. Use `--results-dir` to select a different copy of those files.

```sh
python analysis/revision_statistics.py --output-dir reproduced/statistics
python analysis/mixed_model.py --output-dir reproduced/mixed
python analysis/paired_controls.py --output-dir reproduced/controls
python analysis/training_budgets.py --output-dir reproduced/budgets
python analysis/plot_landscape.py --output-dir reproduced/landscape
python analysis/gradient_profiles.py --output-dir reproduced/gradients
```

The mixed model estimates the macro-F1 difference between LGTS-Alpha and each comparator over the nine original dataset–backbone combinations. The output includes intervals conditional on those combinations and intervals calculated over the five complete seeds.

The control analysis pairs test scores by model and seed. For CIFAR-100 the difference is network-wide α minus fixed, stage-shared or site-specific α. For OrganSMNIST the difference is rotation off minus rotation on, with both flips disabled. Each interval uses all 3,125 ordered resamples of the five seed differences.

The budget analysis gives per-seed final test scores, mean training losses over the last two ten-epoch windows, and five-seed training-loss trajectories. The 100- and 200-epoch schedules span independent fits. Only LGTS-Alpha has both budgets under the revised implementation. The landscape plots use the recorded 21 × 21 FP32 grid for each checkpoint. Plotting requires Matplotlib.

The gradient analysis reads the bundled parameter observations from ten CIFAR-100/ResNet18 runs. It writes the 1,000 seed–epoch profiles and plots the four stage-specific convolutions. The global norm uses all 62 shared trainable parameters. Each row describes the first training minibatch of one epoch. To analyse original engine outputs, supply `--runs-dir` with a directory containing the ten `done.json` and `gradient_norms.csv` pairs.

The per-site α values, fixed-α sensitivity scores, learning-rate validation outcomes and common-rate loss histories are described with the [diagnostic results](../results/diagnostics/README.md).
