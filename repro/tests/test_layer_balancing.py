import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "run_layer_balancing.py"
SPEC = importlib.util.spec_from_file_location("layer_balancing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("h", [512, 1000, 2048])
@pytest.mark.parametrize("alpha", [1.10, 1.25, 1.40, 1.50])
def test_asymmetry_then_balance_in_stated_regime(h, alpha):
    assert MODULE.midpoint_curvature(h, alpha, 1) < 0
    assert MODULE.midpoint_curvature(h, alpha, 2) > 0


@pytest.mark.parametrize("eta1,eta2,h", [(10, 20, 32), (1200, 1800, 1000), (30000, 30000, 1000)])
def test_independent_one_step_expansion(eta1, eta2, h):
    assert MODULE.theoretical_loss(eta1, eta2, h, 1) == pytest.approx(
        MODULE.independent_one_step_loss(eta1, eta2, h), abs=1e-14
    )


def test_exact_gradient_formula_matches_autograd():
    for seed in range(3):
        result = MODULE.gradient_autograd_check(seed, 10)
        assert result["grad1_max_abs_error"] < 1e-14
        assert result["grad2_max_abs_error"] < 1e-14


def test_boundary_control_rejects_overgeneralization():
    assert MODULE.midpoint_curvature(256, 1.50, 2) < 0


@pytest.mark.parametrize("h", [512, 1000, 2048])
@pytest.mark.parametrize("alpha", [0.25, 0.40, 0.50, 0.55])
def test_three_layer_asymmetry_then_balance_in_stated_regime(h, alpha):
    assert MODULE.three_layer_midpoint_curvature(h, alpha, 1) < 0
    assert MODULE.three_layer_midpoint_curvature(h, alpha, 2) > 0


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("eta1,eta2,h", [(3.0, 27.0, 1000), (19.0, 61.0, 1000), (12.5, 19.5, 512)])
def test_three_layer_independent_polynomial_identity(eta1, eta2, h, steps):
    assert MODULE.three_layer_theoretical_loss(eta1, eta2, h, steps) == pytest.approx(
        MODULE.independent_three_layer_loss(eta1, eta2, h, steps), abs=1e-14
    )


def test_three_layer_formula_audits_pinned_source_omission():
    official = MODULE.load_official_three_layer_theory()
    for h in (512, 1000, 2048):
        for eta1, eta2 in ((3.0, 27.0), (19.0, 61.0), (40.0, 40.0)):
            # The pinned plotting source omits the paper's +2*eta1*eta2/h^3
            # one-step term.  Audit the exact discrepancy rather than hiding it.
            paper_one = MODULE.three_layer_theoretical_loss(eta1, eta2, h, 1)
            source_one = official.theoretical_test_loss(eta1, eta2, h, 1)
            assert paper_one - source_one == pytest.approx(2 * eta1 * eta2 / h**3, abs=1e-14)
            # Its two-step expression is identical to Theorem 5.5.
            assert MODULE.three_layer_theoretical_loss(eta1, eta2, h, 2) == pytest.approx(
                official.theoretical_test_loss(eta1, eta2, h, 2), abs=1e-14
            )


def test_three_layer_autograd_hessian_has_claimed_signs():
    for h in (512, 1000, 2048):
        assert MODULE.three_layer_autograd_curvature(h, 0.5, 1) < 0
        assert MODULE.three_layer_autograd_curvature(h, 0.5, 2) > 0
