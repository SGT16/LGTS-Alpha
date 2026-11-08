# Reproducing the reviewer training runs

The 960 completed runs comprise 780 baseline and control runs, 120 ImageNet-100 runs, 30 validation-only learning-rate pilots, 15 network-wide α controls and 15 OrganSMNIST rotation controls. The `run_id` in `runs.csv` links directly to `../results/run_settings.csv`, `seed_results.csv` and `training_history.csv`. The per-run arguments and the shared arguments for its stage come from the recorded invocations.

From the repository root, print the command for a run:

```shell
python research_revision/training/reproduce.py 887 --data-dir ./data --output-dir ./reproduced
```

Add `--execute` to train that run on CUDA. Its outputs go into `./reproduced/run_887/`. Other example IDs are 1750 for ImageNet-100, 1895 for a learning-rate pilot, 1929 for network-wide α and 2034 for the rotation control. The recorded rotation control uses `augmentation=none` with both flips disabled. The matched reference uses `organsmnist_no_flip` with rotation enabled.

Use the software versions recorded in `../../environment/reference_environment.txt` and install the repository requirements. The images come from the dataset providers. The trainer uses torchvision CIFAR-100, the 64-pixel DermaMNIST and OrganSMNIST archives, and the train and validation partitions of `clane9/imagenet-100` at revision `0519dc2f402a3a18c6e57f7913db059215eee25b`. Supply any provider access required in the local environment.

The learning-rate pilots require the extracted CIFAR-100 `torchvision/cifar-100-python/train` and `meta` files under the data directory. The validation adapter verifies those files and the recorded stratified split of 45,000 training and 5,000 validation images. The holdout seed is 2026. The official test file is never opened. The selection endpoint is validation macro-F1 at epoch 100. The adapter verifies the release engine files before reusing the epoch loop. Its release hashes and the original source hashes are listed in `validation_source_hashes.json`. A fresh reproduction records its own software identity.
