# Dataset sources and redistribution

No image data is stored in this repository. The preparation script obtains each
dataset through its original public distribution channel and writes it to the
ignored local `data/` directory.

| Dataset | Interface used by the code | Study resolution | Redistribution note |
|---|---|---:|---|
| CIFAR-100 | `torchvision.datasets.CIFAR100` | 32 × 32 | Not bundled; users must follow the original CIFAR terms |
| DermaMNIST | MedMNIST v2 / `dermamnist_64.npz` | 64 × 64 | Source metadata reports CC BY-NC 4.0; not bundled |
| OrganSMNIST | MedMNIST v2 / `organsmnist_64.npz` | 64 × 64 | Source metadata reports CC BY 4.0; not bundled |

MedMNIST v2 is described by Yang et al., *Scientific Data* 10, 41 (2023),
<https://doi.org/10.1038/s41597-022-01721-8>.

The exact class counts, label hashes, source checksums, and split fingerprints
used in this study are recorded in `results/dataset_manifests/`. Online
augmentation does not create or store additional independent samples; the
number of stored training examples therefore remains unchanged.

