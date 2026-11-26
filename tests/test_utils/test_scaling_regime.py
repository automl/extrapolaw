from __future__ import annotations

import numpy as np

from scalefm.utils.power_law import scaling_regime_start


def _plateau_then_law(num_plateau: int, num_law: int) -> tuple[np.ndarray, np.ndarray]:
    """A Pareto front on 50 * flops^-0.1, but stuck just above its first point for
    the cheapest runs.
    """
    flops = np.logspace(10, 16, num_plateau + num_law)
    law = 50.0 * flops**-0.1
    plateau = law[num_plateau] * (1.04 - 0.01 * np.arange(num_plateau))
    return flops, np.r_[plateau, law[num_plateau:]]


def test_a_pure_power_law_has_no_plateau() -> None:
    flops = np.logspace(10, 16, 12)
    assert scaling_regime_start(flops, 50.0 * flops**-0.1) == 0


def test_the_plateau_is_found() -> None:
    flops, losses = _plateau_then_law(4, 8)
    assert scaling_regime_start(flops, losses) == 4


def test_too_few_points_have_no_plateau() -> None:
    flops, losses = _plateau_then_law(2, 2)
    assert scaling_regime_start(flops, losses) == 0


def test_a_noisy_power_law_has_no_plateau() -> None:
    rng = np.random.default_rng(0)
    flops = np.logspace(10, 16, 15)
    losses = 50.0 * flops**-0.1 * 10 ** rng.normal(0, 0.01, len(flops))
    assert scaling_regime_start(flops, losses) == 0
