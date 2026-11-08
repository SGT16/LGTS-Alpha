import csv
import json
from itertools import product

import numpy as np
import scipy
from scipy.optimize import isotonic_regression
from scipy.stats import t

from study_data import ACTIVATIONS, DATASETS, MODELS, SEEDS, arguments, load_runs


def observed_scores(directory):
    scores = {}
    for run, outcome in load_runs(directory):
        if run["panel"] != "q2_confirmatory":
            continue
        activation = run["activation"]
        if activation.startswith("Native"):
            activation = "Native"
        key = run["dataset"], run["model"], activation, int(run["seed"])
        value = float(outcome["test_f1_macro"])
        if (key in scores or run["evaluation_split"] != "test" or outcome["epochs_run"] != "100"
                or run["protocol"] != "full_train" or not 0 <= value <= 1):
            raise ValueError(f"Invalid original-panel result: {key}")
        scores[key] = value
    if scores.keys() != set(product(DATASETS, MODELS, ACTIVATIONS, SEEDS)):
        raise ValueError("Expected the original 225 unique observations")
    contexts = list(product(DATASETS, MODELS))
    values = np.array([[[scores[d, m, a, seed] for a in ACTIVATIONS] for d, m in contexts] for seed in SEEDS])
    return values, contexts


def fit(values):
    seeds, contexts, acts = values.shape
    residual = values - values.mean(axis=0)
    pair_mean = residual.mean(axis=2, keepdims=True)
    seed_mean = pair_mean.mean(axis=1, keepdims=True)
    df = np.array([(seeds - 1) * contexts * (acts - 1), (seeds - 1) * (contexts - 1), seeds - 1])
    ss = np.array([np.sum((residual - pair_mean) ** 2),
                   acts * np.sum((pair_mean - seed_mean) ** 2),
                   acts * contexts * np.sum(seed_mean ** 2)])
    eigenvalues = isotonic_regression(ss / df, weights=df).x
    error, pair, seed = (eigenvalues[0], (eigenvalues[1] - eigenvalues[0]) / acts,
                         (eigenvalues[2] - eigenvalues[1]) / (acts * contexts))
    return df, ss, eigenvalues, (error, pair, seed)


def gls_check(values, variances):
    seeds, contexts, acts = values.shape
    error, pair, seed = variances
    pair_index = np.repeat(np.arange(seeds * contexts), acts)
    seed_index = np.repeat(np.arange(seeds), contexts * acts)
    covariance = (error * np.eye(values.size)
                  + pair * np.equal.outer(pair_index, pair_index)
                  + seed * np.equal.outer(seed_index, seed_index))
    design = np.tile(np.eye(contexts * acts), (seeds, 1))
    precision_design = np.linalg.solve(covariance, design)
    beta_cov = np.linalg.inv(precision_design.T @ design)
    beta = beta_cov @ precision_design.T @ values.ravel()
    np.testing.assert_allclose(beta, values.mean(axis=0).ravel(), rtol=1e-10, atol=1e-12)
    for i in range(1, acts):
        contrast = np.zeros((contexts, acts))
        contrast[:, 0], contrast[:, i] = 1 / contexts, -1 / contexts
        variance = contrast.ravel() @ beta_cov @ contrast.ravel()
        np.testing.assert_allclose(variance, 2 * error / (seeds * contexts), rtol=1e-10)


def intervals(method, comparator, difference, se, df, family):
    rows = []
    for adjustment, alpha in (("pointwise", .05), ("bonferroni_4_contrasts", .05 / family)):
        width = float(t.ppf(1 - alpha / 2, df)) * se
        low, high = difference - width, difference + width
        row = dict(method=method, activation=ACTIVATIONS[0], comparator=comparator,
                   macro_f1_difference=difference, standard_error=se, degrees_of_freedom=int(df),
                   interval=adjustment, lower=low, upper=high)
        for margin in (.005, .010, .020):
            row[f"lower_above_minus_{margin:.3f}"] = bool(low > -margin)
        rows.append(row)
    return rows


def main():
    args = arguments("Balanced mixed model for the original 225-run panel.")
    values, contexts = observed_scores(args.results_dir)
    seeds, ncontexts, acts = values.shape
    df, ss, eigenvalues, variances = fit(values)
    gls_check(values, variances)
    model_se = float(np.sqrt(2 * ss[0] / df[0] / (seeds * ncontexts)))
    rows = []
    for i, comparator in enumerate(ACTIVATIONS[1:], 1):
        seed_differences = (values[:, :, 0] - values[:, :, i]).mean(axis=1)
        estimate = float(seed_differences.mean())
        rows.extend(intervals("mixed_model_fixed_nine_contexts", comparator, estimate, model_se, df[0], acts - 1))
        cluster_se = float(seed_differences.std(ddof=1) / np.sqrt(seeds))
        rows.extend(intervals("paired_whole_seed_sensitivity", comparator, estimate, cluster_se, seeds - 1, acts - 1))
    evidence = {
        "classification": "Post-review supplementary analysis of the original locked panel",
        "outcome": "Final test macro-F1, 100 epochs, original confirmatory preprocessing and augmentations",
        "observations": int(values.size), "outcome_cells": ncontexts * acts, "seeds": list(SEEDS),
        "contexts": contexts, "activations": list(ACTIVATIONS),
        "fixed_effects": "Separate mean for every activation x fixed dataset x fixed backbone combination",
        "random_effects": "Independent Gaussian intercepts for global seed and for seed x context, plus iid Gaussian errors",
        "target": "LGTS-Alpha minus comparator, equal weight for every one of these nine fixed contexts",
        "variance_estimator": "Balanced-design REML, with nonnegative additive variances and pooled adjacent error strata",
        "variance_components": dict(zip(("residual", "seed_by_context_intercept", "global_seed_intercept"), map(float, variances))),
        "error_strata": [dict(stratum=name, df=int(degrees), ss=float(total), mean_square=float(total / degrees),
                              fitted_eigenvalue=float(value)) for name, degrees, total, value in
                         zip(("within_seed_context", "seed_by_context", "between_seed"), df, ss, eigenvalues)],
        "model_contrast_interval": "Two-sided t interval from the within-seed-context residual mean square. "
                                   "The numerator cancels both intercepts. Under this Gaussian model the numerator "
                                   "is independent of the error stratum (144 df).",
        "whole_seed_sensitivity": "Average paired activation differences equally over the nine contexts within each seed. "
                                  "Student t over five seed-level averages (4 df). Allows other within-seed dependencies "
                                  "and context-specific variances, assumes independent, approximately normal seed averages.",
        "interpretation": ["Strict lower bound > -margin, using two-sided 95% pointwise intervals; "
                           "the four-comparison Bonferroni intervals have simultaneous coverage of at least 95%.",
                           "Post-review interval criteria supplement the locked confirmatory results.",
                           "The fixed contexts do not represent draws from a population of datasets or backbones.",
                           "The model assumes common residual variance across all activation-context cells. "
                           "There are only five global seeds. A zero REML variance estimate alone does not establish independence."],
        "fixed_cell_means": [dict(dataset=d, model=m, activation=a, mean_macro_f1=float(values[:, j, i].mean()))
                             for j, (d, m) in enumerate(contexts) for i, a in enumerate(ACTIVATIONS)],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mixed_model.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    with (args.output_dir / "mixed_model_contrasts.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{values.size} validated scores, {len(rows)} contrast intervals")


if __name__ == "__main__":
    main()
