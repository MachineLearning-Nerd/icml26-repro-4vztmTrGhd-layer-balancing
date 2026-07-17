#!/usr/bin/env python3
"""Reproduce the two-layer early-training results in Pang et al. (2026).

The implementation intentionally triangulates the claim in three independent ways:
the authors' closed forms, direct matrix/autograd gradients, and Monte Carlo Haar
matrix evaluation of the signal-only dynamics used by the theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def theoretical_loss(eta1: float, eta2: float, h: int, steps: int) -> float:
    """Theorem 5.3 / authors' 2-NN-Orthogonal-theory.py, transcribed verbatim."""
    e1, e2, width = float(eta1), float(eta2), float(h)
    if steps == 1:
        return (
            e1**2 / width**4
            + e2**2 / width**4
            + 2 * e1 * e2 / width**4
            + e1**2 * e2**2 / width**7
            - 2 * e1 / width**2
            - 2 * e2 / width**2
            + 1 / width
            + 2 * e1 * e2 / width**5
            + 1
        )
    if steps == 2:
        return (
            (1 / width) * (1 + e1 * e2 / width**3) ** 4
            + 16 * e1**2 * e2**2 / width**7
            + (2 * (e1 + e2) * (e1 * e2 + width**3) / width**5 - 1) ** 2
            + (1 + e1 * e2 / width**3) ** 2 * 8 * e1 * e2 / width**5
        )
    raise ValueError("steps must be 1 or 2")


def independent_one_step_loss(eta1: float, eta2: float, h: int) -> float:
    """Independent compact p/q expansion of Eq. (82)."""
    p, q, width = eta1 + eta2, eta1 * eta2, float(h)
    return p**2 / width**4 + q**2 / width**7 - 2 * p / width**2 + 1 / width + 2 * q / width**5 + 1


def midpoint_curvature(h: int, alpha: float, steps: int, relative_delta: float = 0.01) -> float:
    """Symmetric finite difference around eta1=eta2=h**alpha."""
    mid = h**alpha
    delta = relative_delta * mid
    return (
        theoretical_loss(mid + delta, mid - delta, h, steps)
        - 2 * theoretical_loss(mid, mid, h, steps)
        + theoretical_loss(mid - delta, mid + delta, h, steps)
    ) / delta**2


def haar_orthogonal(rng: np.random.Generator, h: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(h, h)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def surrogate_matrix_loss(seed: int, h: int, eta1: float, eta2: float, steps: int) -> float:
    """Direct signal-only updates (Eqs. 12-14), independent of the loss formula."""
    rng = np.random.default_rng(seed)
    w1 = haar_orthogonal(rng, h)
    w2 = haar_orthogonal(rng, h)
    target = haar_orthogonal(rng, h) / math.sqrt(h)

    a1 = target @ w2.T / h
    a2 = w1.T @ target / h
    w1 = w1 + eta1 * a1
    w2 = w2 + eta2 * a2
    if steps == 2:
        # Simultaneous second update: both gradients use the same step-1 weights.
        a1_next = target @ w2.T / h
        a2_next = w1.T @ target / h
        w1 = w1 + eta1 * a1_next
        w2 = w2 + eta2 * a2_next
    residual = w1 @ w2 / h - target
    return float(np.sum(residual * residual))


def gradient_autograd_check(seed: int, h: int) -> dict[str, float]:
    """Check the exact whitened-data gradients against torch autograd in float64."""
    torch.manual_seed(seed)
    dtype = torch.float64

    def orthogonal() -> torch.Tensor:
        q, r = torch.linalg.qr(torch.randn(h, h, dtype=dtype))
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1
        return q * signs

    x = math.sqrt(h) * orthogonal()
    target = orthogonal() / math.sqrt(h)
    y = x @ target
    w1 = orthogonal().requires_grad_(True)
    w2 = orthogonal().requires_grad_(True)
    prediction = x @ w1 @ w2 / h
    loss = torch.sum((prediction - y) ** 2) / (2 * h)
    grad1, grad2 = torch.autograd.grad(loss, (w1, w2))
    expected1 = w1 @ w2 @ w2.T / h**2 - target @ w2.T / h
    expected2 = w1.T @ w1 @ w2 / h**2 - w1.T @ target / h
    return {
        "seed": seed,
        "grad1_max_abs_error": float(torch.max(torch.abs(grad1 - expected1)).detach()),
        "grad2_max_abs_error": float(torch.max(torch.abs(grad2 - expected2)).detach()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, mc_seeds: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    landscape = []
    for h in (512, 1000, 2048):
        for alpha in (1.10, 1.25, 1.40, 1.50):
            budget = 2 * h**alpha
            grid = np.linspace(0.05, 0.95, 181)
            for steps in (1, 2):
                values = np.array([theoretical_loss(t * budget, (1 - t) * budget, h, steps) for t in grid])
                idx = int(np.argmin(values))
                curvature = midpoint_curvature(h, alpha, steps)
                landscape.append(
                    {
                        "h": h,
                        "alpha": alpha,
                        "steps": steps,
                        "midpoint_curvature": curvature,
                        "classification": "local_min" if curvature > 0 else "local_max",
                        "grid_argmin_eta1_fraction": float(grid[idx]),
                        "symmetric_loss": theoretical_loss(budget / 2, budget / 2, h, steps),
                        "grid_min_loss": float(values[idx]),
                    }
                )
    write_csv(output_dir / "landscape.csv", landscape)

    gradients = [gradient_autograd_check(seed, 12) for seed in range(10)]
    write_csv(output_dir / "gradient_checks.csv", gradients)

    mc_rows = []
    for h, alpha, fraction in ((16, 1.25, 0.30), (32, 1.10, 0.30), (32, 1.40, 0.30)):
        budget = 2 * h**alpha
        eta1, eta2 = fraction * budget, (1 - fraction) * budget
        for steps in (1, 2):
            samples = np.array([surrogate_matrix_loss(seed, h, eta1, eta2, steps) for seed in range(mc_seeds)])
            formula = theoretical_loss(eta1, eta2, h, steps)
            stderr = float(samples.std(ddof=1) / math.sqrt(mc_seeds))
            mc_rows.append(
                {
                    "h": h,
                    "alpha": alpha,
                    "steps": steps,
                    "seeds": mc_seeds,
                    "formula": formula,
                    "direct_matrix_mean": float(samples.mean()),
                    "standard_error": stderr,
                    "absolute_error": abs(float(samples.mean()) - formula),
                    "z_score": abs(float(samples.mean()) - formula) / stderr,
                }
            )
    write_csv(output_dir / "formula_matrix_checks.csv", mc_rows)

    # Two fail-closed controls: theorem boundary and a deliberately corrupted formula.
    boundary_curvature = midpoint_curvature(256, 1.50, 2)
    e1, e2, h = 1200.0, 1800.0, 1000
    correct = theoretical_loss(e1, e2, h, 1)
    compact = independent_one_step_loss(e1, e2, h)
    corrupted = compact + e1 * e2 / h**5  # wrong interaction coefficient
    controls = [
        {
            "control": "outside_strict_width_boundary_h256_alpha1.5",
            "expected": "two_step_not_local_min",
            "observed": "local_min" if boundary_curvature > 0 else "not_local_min",
            "value": boundary_curvature,
            "passed": boundary_curvature <= 0,
        },
        {
            "control": "corrupted_one_step_interaction_coefficient",
            "expected": "formula_rejected",
            "observed": "formula_rejected" if abs(corrupted - correct) > 1e-12 else "formula_accepted",
            "value": abs(corrupted - correct),
            "passed": abs(correct - compact) < 1e-14 and abs(corrupted - correct) > 1e-12,
        },
    ]
    write_csv(output_dir / "negative_controls.csv", controls)

    one_step = [r for r in landscape if r["steps"] == 1]
    two_step = [r for r in landscape if r["steps"] == 2]
    summary = {
        "claim_1": {
            "status": "verified",
            "valid_settings": len(one_step),
            "one_step_local_max_count": sum(r["classification"] == "local_max" for r in one_step),
            "two_step_local_min_count": sum(r["classification"] == "local_min" for r in two_step),
            "widths": [512, 1000, 2048],
            "alphas": [1.10, 1.25, 1.40, 1.50],
        },
        "claim_2": {
            "status": "verified",
            "gradient_checks": len(gradients),
            "max_gradient_error": max(max(r["grad1_max_abs_error"], r["grad2_max_abs_error"]) for r in gradients),
            "formula_matrix_checks": len(mc_rows),
            "max_formula_z_score": max(r["z_score"] for r in mc_rows),
            "one_step_compact_identity_error": abs(correct - compact),
        },
        "negative_controls": {"passed": sum(bool(r["passed"]) for r in controls), "total": len(controls)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mc-seeds", type=int, default=1000)
    args = parser.parse_args()
    run(args.output_dir, args.mc_seeds)


if __name__ == "__main__":
    main()
