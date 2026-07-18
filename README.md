# Repro — Balancing Learning Rates Across Layers (ICML 2026)

Reproduction of *Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and
Optimal Scaling in Linear Neural Networks* ([arXiv:2606.00340](https://arxiv.org/abs/2606.00340),
OpenReview `4vztmTrGhd`) for the ICML 2026 Agent Reproduction Challenge.

## Claims

1. **Asymmetry then balance — verified for both two and three layers.** The
   two-layer result holds across 12 theorem-valid settings (`h` in 512, 1000,
   2048; `alpha` in 1.10, 1.25, 1.40, 1.50). The three-layer result passes ten
   independent approaches, including 30/30 Figure-2 formula curves, symbolic and
   arbitrary-precision curvature, and direct clean/noisy matrix GD at the paper's
   `h=d=n=1000` scale over five seeds. Equal learning rates are a local maximum
   after one step and a global minimum after two steps on both full-scale direct
   grids.
2. **Closed-form gradients and loss characterization — verified with the paper's
   stated approximation scope.** Exact whitened-data gradients match PyTorch
   autograd in float64, the one-step loss matches an independent compact polynomial
   expansion, and direct Haar-matrix signal-only updates agree statistically with
   the expected-loss formulas. At the authors' full `h=1000` scale, the formula
   tracks exact GD within 0.145% after one step and 1.633% after two steps. The
   latter is below the theorem's `O(h^-1/2)` scale (3.16% before its constant), but
   confirms that the formula is not a literal bit-exact identity for full GD.

The boundary control `h=256, alpha=1.5` correctly rejects the two-step conclusion,
showing that the strict width condition is necessary. A deliberately corrupted
interaction coefficient is also rejected.

The three-layer source audit also found a small but exact discrepancy: the pinned
authors' plotting script omits the paper's `2 eta1 eta2 / h^3` one-step term. The
residual after accounting for this term is `4.89e-16`, the two-step expressions
agree to `2.21e-16` relative error, and both variants retain the claimed landscape
shape on all 30 Figure-2 curves.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy torch pytest
source .venv/bin/activate
python repro/src/run_layer_balancing.py --output-dir outputs --mc-seeds 1000
python repro/src/run_three_layer_attempts.py --output-dir outputs
python -m pytest repro/tests -q
```

Official code is pinned at commit `97997b4307187836b18ce0269bb7b4275f16db9e`.
