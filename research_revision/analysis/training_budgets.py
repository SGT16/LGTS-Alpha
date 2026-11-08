from collections import defaultdict
from itertools import product
import math
from statistics import fmean, stdev

from study_data import ACTIVATIONS, SEEDS, arguments, load_runs, rows, write_csv


def main():
    args = arguments("CIFAR-100/EfficientNetV2-S training loss and final test scores at 100 and 200 epochs.")
    panels = {"q2_confirmatory": (100, ACTIVATIONS), "components_baselines": (100, ("LGTS-f4-Alpha",)),
              "longer_budget": (200, ACTIVATIONS)}
    expected = {(panel, a, s) for panel, (_, acts) in panels.items() for a, s in product(acts, SEEDS)}
    cases = {}
    for run, score in load_runs(args.results_dir):
        key = run["panel"], run["activation"], int(run["seed"])
        if key not in expected or (run["dataset"], run["model"]) != ("cifar100", "EfficientNetV2S"):
            continue
        if key in cases or run["evaluation_split"] != "test" or int(run["epochs"]) != panels[key[0]][0]:
            raise ValueError(f"Duplicate condition or wrong test budget: {key}")
        cases[key] = run["run_id"], score
    if cases.keys() != expected:
        raise ValueError("Expected five complete seed pairs per activation and training recipe")
    histories = {run_id: {} for run_id, _ in cases.values()}
    for row in rows(args.results_dir / "training_history.csv"):
        if row["run_id"] in histories:
            group = histories[row["run_id"]]
            ep = int(row["epoch"])
            loss = float(row["train_loss"])
            if ep in group or not math.isfinite(loss):
                raise ValueError(f"Repeated epoch or invalid training loss for run {row['run_id']}")
            group[ep] = loss
    scores = []
    curves = defaultdict(list)
    for (panel, act, seed), (run_id, score) in sorted(cases.items()):
        budget = panels[panel][0]
        history = histories[run_id]
        if set(history) != set(range(1, budget + 1)):
            raise ValueError(f"Incomplete or extended history for {run_id}")
        scores.append(dict(panel=panel, activation=act, epochs=budget, seed=seed,
                           test_macro_f1=float(score["test_f1_macro"]), test_accuracy=float(score["test_accuracy"]),
                           mean_loss_penultimate_10=fmean(history[e] for e in range(budget-19, budget-9)),
                           mean_loss_final_10=fmean(history[e] for e in range(budget-9, budget+1))))
        for epoch, loss in history.items():
            curves[panel, act, epoch].append(loss)
    trajectory = [dict(panel=p, activation=a, epoch=e, n=len(y), mean_train_loss=fmean(y), sd_train_loss=stdev(y))
                  for (p, a, e), y in sorted(curves.items())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "budget_seed_results.csv", scores)
    write_csv(args.output_dir / "training_loss_by_epoch.csv", trajectory)


if __name__ == "__main__":
    main()
