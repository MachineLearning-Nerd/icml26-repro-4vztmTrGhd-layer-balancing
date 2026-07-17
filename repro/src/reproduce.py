#!/usr/bin/env python3
"""Layer-Balancing (4vztmTrGhd) — exact-instance CPU repro.

Imports the authors' closed-form (theoretical_test_loss) + GD rollout
(compute_curves_for_one_lrmax / train_step) and verifies:
  C1 argmin identity: on the diagonal lr1+lr2=lr_max, step-1 test loss is
     minimized at UNEQUAL LRs (lr1 != lr2), step-2 at EQUAL LRs (lr1=lr2).
  C2 exact closed-form: theoretical_test_loss matches the actual GD-rollout
     empirical test loss to high precision (orthonormal init/X => exact dynamics).
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, "..", "..", "upstream"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(out):
    out = os.path.abspath(out)
    os.chdir(UP)
    theory = load(os.path.join(UP, "2-NN-Orthogonal-theory.py"), "theory")
    exp = load(os.path.join(UP, "2-NN-Orthogonal-experiment.py"), "exp")
    device = torch.device("cpu")
    t0 = time.perf_counter()

    d = h = n = 1000
    test_size = 4000
    lr_max = 40000
    steps_list = [1, 2]
    seeds = [2020, 2021, 2022, 2023, 2024]
    xs, emp = exp.compute_curves_for_one_lrmax(
        lr_max, d=d, h=h, n=n, test_size=test_size, device=device,
        seeds=seeds, steps_list=steps_list)

    def theory_curve(steps):
        return np.array([theory.theoretical_test_loss(float(xs[k]), float(lr_max - xs[k]), h, steps)
                         for k in range(len(xs))])

    diffs = {}
    rel_diffs = {}
    for s in steps_list:
        t = theory_curve(s)
        e = emp[s]
        diffs[s] = float(np.max(np.abs(t - e)))
        rel_diffs[s] = float(np.max(np.abs(t - e) / np.maximum(np.abs(t), 1e-9)))
    max_diff = max(diffs.values())
    max_rel = max(rel_diffs.values())

    mid = len(xs) // 2
    argmin1 = int(np.argmin(theory_curve(1)))
    argmin2 = int(np.argmin(theory_curve(2)))
    c1 = bool(argmin1 != mid and abs(argmin2 - mid) <= 1)   # step1 unequal, step2 equal
    c2 = bool(max_rel < 1e-3)                                # closed-form matches GD rollout (relative; loss~h)

    summary = {
        "claim_1": "verified" if c1 else "not-verified",
        "claim_2": "verified" if c2 else "not-verified",
        "max_abs_diff_theory_vs_GD": max_diff, "max_RELATIVE_diff": max_rel,
        "diff_by_step": diffs, "rel_diff_by_step": rel_diffs,
        "argmin_step1_idx": argmin1, "argmin_step2_idx": argmin2, "mid_idx": mid,
        "argmin_step1_lr1_over_lr2": float(xs[argmin1]) / float(lr_max - xs[argmin1]),
        "argmin_step2_lr1_over_lr2": float(xs[argmin2]) / float(lr_max - xs[argmin2]),
        "config": {"d": d, "h": h, "n": n, "test_size": test_size, "lr_max": lr_max, "seeds": seeds},
        "runtime_seconds": time.perf_counter() - t0,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs")
    run(ap.parse_args().output)
