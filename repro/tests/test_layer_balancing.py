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

