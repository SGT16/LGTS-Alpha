"""Verify the portable revision tables and reproduce key manuscript values on CPU."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import friedmanchisquare, t


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_csv(name):
    with (RESULTS / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def assert_close(a, b, tol=1e-8):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=tol):
        raise ValueError(f"Numeric mismatch: {a} vs {b}")


def main():
    checks = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in checks:
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    settings = {r["run_id"]:r for r in read_csv("run_settings")}
    seed_rows = {r["run_id"]:r for r in read_csv("seed_results")}
    if set(settings) != set(seed_rows) or len(settings) != 1335:
        raise ValueError("Seed-to-setting link or scope changed")
    panels = Counter(r["panel"] for r in settings.values())
    if panels != json.loads((ROOT / "protocols/panel_counts.json").read_text()):
        raise ValueError("Panel count mismatch")
    groups = defaultdict(dict)
    for i, r in settings.items():
        if r["panel"] == "q2_confirmatory":
            name = "Native" if r["activation"].startswith("Native") else r["activation"]
            key = (r["dataset"],r["model"],r["seed"])
            if key in groups[name]:
                raise ValueError("Duplicate main-panel key")
            groups[name][key] = float(seed_rows[i]["test_f1_macro"])
    names = ["LGTS-f4-Alpha","LGTS-f2-Slope","ACON-C","GELU","Native"]
    cases = sorted(set((d,m) for d,m,s in groups[names[0]]))
    seeds = sorted(set(s for d,m,s in groups[names[0]]))
    cube = np.array([[[groups[n][(d,m,s)] for n in names] for d,m in cases] for s in seeds])
    if cube.shape != (5,9,5):
        raise ValueError("Main-panel factorial design changed")
    q,p = friedmanchisquare(*cube.mean(axis=0).T)
    assert_close(q,11.2888888888889)
    whole_seed = (cube[:,:,0] - cube[:,:,2]).mean(axis=1)
    observed = whole_seed.mean()
    half = t.ppf(.975,4) * whole_seed.std(ddof=1) / math.sqrt(5)
    model_values = read_csv("mixed_model_contrasts")
    matching = next(r for r in model_values if r["method"] == "paired_whole_seed_sensitivity" and r["comparator"] == "ACON-C" and r["interval"] == "pointwise")
    assert_close(observed - half,float(matching["lower"]))
    assert_close(observed + half,float(matching["upper"]))
    variance = cube - cube.mean(axis=0,keepdims=True)
    residual = variance - variance.mean(axis=2,keepdims=True)
    residual_df = 4*9*4
    width = t.ppf(.975,residual_df) * math.sqrt(2*float((residual**2).sum())/residual_df/45)
    matching = next(r for r in model_values if r["method"] == "mixed_model_fixed_nine_contexts" and r["comparator"] == "ACON-C" and r["interval"] == "pointwise")
    assert_close(observed-width,float(matching["lower"]))
    assert_close(observed+width,float(matching["upper"]))
    epoch = read_csv("real_epochs")
    if len(epoch) != 144:
        raise ValueError("Expected 72 original and 72 matched-update arms")
    matched = json.loads((RESULTS/"matched_update_contexts.json").read_text())
    if matched["completed"] != 72 or matched["excluded"] or len(matched["contexts"]) != 12:
        raise ValueError("Matched-update scope mismatch")
    four = json.loads((RESULTS/"prelu_fp16_slope_replays.json").read_text())
    if len(four["units"]) != 4 or four["accepted"] or any(r["slope_gradient_finite"] or not r["input_gradient_finite"] or r["completed_blocks"] != 10 for r in four["units"]):
        raise ValueError("Incorrect diagnostic replay evidence")
    print(f"{len(checks)} file hashes verified; {len(settings)} final runs; 4 diagnostic replays.")
    print(f"Original Q = {q:.5f}, p = {p:.6f}; Alpha minus ACON-C = {observed:.6f}.")
    print(f"Whole-seed CI [{observed-half:.6f}, {observed+half:.6f}]; model CI [{observed-width:.6f}, {observed+width:.6f}].")


if __name__ == "__main__":
    main()
