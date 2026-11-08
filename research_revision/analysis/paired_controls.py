from itertools import product
import json
import math
from statistics import fmean, stdev

from study_data import MODELS, SEEDS, arguments, load_runs


RECIPES = {
    ("cifar100", "network_alpha", "LGTS-f4-Alpha", "network", "standard"): "network",
    ("cifar100", "components_baselines", "LGTS-f4-Fixed", "site", "standard"): "fixed",
    ("cifar100", "components_baselines", "LGTS-f4-Alpha", "site", "standard"): "site",
    ("cifar100", "alpha_sharing", "LGTS-f4-Alpha", "stage", "standard"): "stage",
    ("organsmnist", "organsmnist_no_rotation", "LGTS-f4-Alpha", "site", "none"): "rotation_off",
    ("organsmnist", "organsmnist_no_flip", "LGTS-f4-Alpha", "site", "organsmnist_no_flip"): "rotation_on",
}
MATCH = ("dataset", "model", "seed", "training_seed", "split_seed", "epochs", "batch_size",
         "image_size", "normalization", "evaluation_split", "selection_policy", "optimizer",
         "scheduler", "learning_rate_initial", "weight_decay", "pretrained", "tf32", "no_amp",
         "dataset_fingerprint_sha256", "code_sha256", "activation_scope", "num_workers",
         "prefetch_factor", "lgts_alpha_min", "lgts_alpha_max", "lgts_beta_min", "lgts_beta_max")


def groups(directory):
    found = {}
    for run, outcome in load_runs(directory):
        recipe = RECIPES.get(tuple(run[k] for k in ("dataset", "panel", "activation", "activation_sharing", "augmentation")))
        if recipe is None:
            continue
        key = recipe, run["model"], int(run["seed"])
        if (key in found or run["epochs"] != "100" or run["protocol"] != "full_train"
                or run["evaluation_split"] != "test" or outcome["epochs_run"] not in ("", "100")):
            raise ValueError(f"Duplicate or incompatible control: {key}")
        found[key] = run, outcome
    if found.keys() != set(product(RECIPES.values(), MODELS, SEEDS)):
        raise ValueError("The six control arms must each have three models and the same five seeds")
    return found


def contrasts(found, candidate, references):
    result = []
    for model in sorted(MODELS):
        for comparator in references:
            pairs = []
            for seed in SEEDS:
                run_a, a = found[candidate, model, seed]
                run_b, b = found[comparator, model, seed]
                for key in MATCH:
                    if run_a[key] != run_b[key]:
                        raise ValueError(f"Unmatched {key}: {model}, {seed}, {comparator}")
                nparams = int(a["trainable_parameters"]) - int(b["trainable_parameters"])
                if (comparator == "fixed" and nparams != 1) or (comparator == "rotation_on" and nparams != 0):
                    raise ValueError(f"Unexpected parameter counts: {model}, {seed}, {comparator}")
                pairs.append((a, b))
            for metric in ("test_f1_macro", "test_accuracy"):
                observed = [float(a[metric]) for a, b in pairs]
                reference = [float(b[metric]) for a, b in pairs]
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in observed + reference):
                    raise ValueError(f"Invalid {metric} for {model}")
                differences = [a - b for a, b in zip(observed, reference)]
                means = sorted(sum(draw) / 5 for draw in product(differences, repeat=5))
                def percentile(q):
                    position = q * (len(means) - 1)
                    i = int(position)
                    return means[i] + (position - i) * (means[i + 1] - means[i])
                result.append(dict(model=model, comparator=comparator, metric=metric, seed_ids=list(SEEDS),
                                   paired_differences=differences, candidate_mean=fmean(observed),
                                   reference_mean=fmean(reference), candidate_sd=stdev(observed),
                                   reference_sd=stdev(reference), delta_mean=fmean(differences),
                                   delta_sd=stdev(differences), delta_ci95_lower=percentile(.025),
                                   delta_ci95_upper=percentile(.975)))
    return result


def main():
    args = arguments("Paired CIFAR-100 scale-sharing and OrganSMNIST rotation comparisons.")
    arms = groups(args.results_dir)
    panels = {"shared_alpha_paired": contrasts(arms, "network", ("fixed", "site", "stage")),
              "sagittal_rotation_paired": contrasts(arms, "rotation_off", ("rotation_on",))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, results in panels.items():
        (args.output_dir / (name + ".json")).write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
