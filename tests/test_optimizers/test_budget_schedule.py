from __future__ import annotations

import logging
import time

import pytest
import torch
from botorch.models import SingleTaskGP

from extrapolaw.exceptions import BudgetExhaustedError
from extrapolaw.optimizers.acquisition.information_theoretic import (
    CostWeightedModifiedJES,
)
from extrapolaw.optimizers.algorithms import bayesian_optimization
from extrapolaw.optimizers.models.gp import LogCost
from extrapolaw.optimizers.optimizer import SampledConfig
from extrapolaw.space import HPOFloat, HPOInteger, SearchSpace
from extrapolaw.space.encoding import ConfigEncoder
from extrapolaw.state import BudgetInfo, Trial


def _space(lower: int = 1, upper: int = 1000) -> SearchSpace:
    return SearchSpace(
        {
            "n_params": HPOInteger(lower, upper, log=True),
            "n_data": HPOInteger(lower, upper, log=True),
            "x": HPOFloat(0.0, 1.0),
        }
    )


def _cost(n_params: float, n_data: float, **_: float) -> float:
    return n_params * n_data


def _objective(config: dict) -> float:
    return 1 + 10 / _cost(**config) ** 0.3 + (config["x"] - 0.3) ** 2


def _trial(sampled: SampledConfig) -> Trial:
    trial = Trial.new(
        trial_id=sampled.id,
        config=sampled.config,
        location="",
        previous_trial=None,
        previous_trial_location=None,
        time_sampled=time.time(),
        worker_id="0",
    )
    trial.report = trial.set_complete(
        report_as="success",
        time_end=time.time(),
        objective_to_minimize=_objective(sampled.config),
        cost=_cost(**sampled.config),
        learning_curve=None,
        err=None,
        tb=None,
        extra=None,
        evaluation_duration=0.0,
    )
    return trial


def _optimizer(schedule: str, *, max_cost: float | None = 1e6, **kwargs):
    optimizer = bayesian_optimization(
        kwargs.pop("space", _space()),
        initial_design_size=4,
        num_restarts=2,
        n_initial_start_points=32,
        budget_schedule=schedule,
        **kwargs,
    )
    optimizer.set_cost_estimator(_cost, max_cost_per_config=max_cost)
    return optimizer


def test_space_expansion_levels_and_caps() -> None:
    # Budget of 8 max cost configs: L = 3 levels, the initial design gets 1/8 of it
    optimizer = _optimizer("space_expansion")
    total = 8e6
    assert [optimizer._expansion_level(s, total) for s in (0, 1e6, 3e6, 7e6, 1e7)] == [
        0,
        1,
        2,
        3,
        3,
    ]

    def cap(spent: float, **kwargs) -> float:
        return optimizer._schedule_cap(spent, total, **kwargs)

    assert cap(0, is_initial_design=True) == 1e6 / 4
    assert cap(5e5, is_initial_design=False) == 5e5
    # Level 1 and 2 allow 2e6 and 4e6, clamped at `max_cost_per_config`
    assert cap(1e6, is_initial_design=False) == 1e6
    assert cap(3e6, is_initial_design=False) == 1e6

    optimizer.max_cost_per_config = None
    optimizer.target_flops = 1e6
    assert cap(1e6, is_initial_design=False) == 2e6
    assert cap(3e6, is_initial_design=False) == 4e6
    assert cap(0, is_initial_design=True, expand=1) == 2e6


def test_caps_are_clamped_at_the_budget_left() -> None:
    optimizer = _optimizer("space_expansion")
    assert optimizer._schedule_cap(7.5e6, 8e6, is_initial_design=False) == 5e5


def _spent(*configs: tuple[int, int]) -> dict[str, Trial]:
    sampled = [
        SampledConfig(id=str(i), config={"n_params": p, "n_data": d, "x": 0.5})
        for i, (p, d) in enumerate(configs)
    ]
    return {s.id: _trial(s) for s in sampled}


def test_too_little_budget_left_stops_the_schedule() -> None:
    optimizer = _optimizer("space_expansion")
    # 1e5 left, less than the 8e6 / 8 / 4 = 2.5e5 of an initial design config
    trials = _spent(*[(1000, 1000)] * 7, (900, 1000))
    with pytest.raises(BudgetExhaustedError, match="cheapest config"):
        optimizer(trials, BudgetInfo(worker_cost_to_spend=8e6))


def test_no_feasible_config_in_the_budget_left_stops_the_schedule() -> None:
    # 111 left, more than the 3200 / 8 / 4 = 100 of an initial design config, but all
    # configs cost at least 121
    optimizer = _optimizer("space_expansion", max_cost=400, space=_space(11, 100))
    trials = _spent((20, 100), (33, 33))
    with pytest.raises(BudgetExhaustedError, match="No feasible config"):
        optimizer(trials, BudgetInfo(worker_cost_to_spend=3200))


def test_left_budget_contraction_caps() -> None:
    optimizer = _optimizer("left_budget_contraction", max_cost=None)
    assert optimizer._schedule_cap(0, 8e6, is_initial_design=True) == 8e6 / 40
    assert optimizer._schedule_cap(3e6, 8e6, is_initial_design=False) == 5e6


@pytest.mark.parametrize("schedule", ["space_expansion", "left_budget_contraction"])
def test_sampled_configs_stay_under_the_schedule_cap(schedule: str) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(schedule)
    budget_info = BudgetInfo(worker_cost_to_spend=8e6)
    trials: dict[str, Trial] = {}
    for _ in range(6):
        spent = sum(_cost(**t.config) for t in trials.values())
        cap = optimizer._schedule_cap(
            spent, 8e6, is_initial_design=len(trials) < optimizer.n_initial_design
        )
        sampled = optimizer(trials, budget_info)
        assert _cost(**sampled.config) <= cap
        trials[sampled.id] = _trial(sampled)


def test_infeasible_level_moves_to_the_next() -> None:
    torch.manual_seed(0)
    # The initial design configs may cost 400 / 4 = 100, but all cost at least 121
    space = _space(lower=11, upper=100)
    optimizer = _optimizer("space_expansion", max_cost=400, space=space)
    sampled = optimizer({}, BudgetInfo(worker_cost_to_spend=3200))
    # Level 1 allows 3200 / 4 = 800, clamped at 400
    assert 121 <= _cost(**sampled.config) <= 400


def test_schedule_needs_a_budget() -> None:
    optimizer = _optimizer("space_expansion")
    with pytest.raises(ValueError, match="cost budget"):
        optimizer({}, BudgetInfo())


def test_modified_jes_under_space_expansion_logs_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(
        "space_expansion", acquisition_func_type="IT-SCAN-JES", target_flops=1e6
    )
    # L = 2 levels, the initial design gets 2.1e6 / 4 = 5.25e5 of the budget
    budget_info = BudgetInfo(worker_cost_to_spend=2.1e6)
    trials: dict[str, Trial] = {}
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            sampled = optimizer(trials, budget_info)
            trials[sampled.id] = _trial(sampled)
    # After the initial design, no feasible config is within 1.5x of the target 1e6
    assert "target FLOPs" in caplog.text


def test_modified_jes_divides_by_the_normalized_exact_cost() -> None:
    torch.manual_seed(0)
    encoder = ConfigEncoder.from_parameters(_space().searchables)
    x = torch.rand(8, 3, dtype=torch.float64)
    y = torch.rand(8, 1, dtype=torch.float64)
    gp = SingleTaskGP(x, y)
    gp.posterior(x)  # Fantasizing needs the caches of a prediction
    optima = {"optimal_inputs": x[:2], "optimal_outputs": y[:2] - 1}
    log_cost = LogCost(encoder, _cost)
    weighted = CostWeightedModifiedJES(
        gp, **optima, log_cost=log_cost, c_min_log=0.0, c_max_log=6.0
    )
    unweighted = CostWeightedModifiedJES(
        gp, **optima, log_cost=None, c_min_log=0.0, c_max_log=6.0
    )

    X = torch.rand(5, 1, 3, dtype=torch.float64)
    c_norm = (log_cost.evaluate(X[:, 0]) / 6.0).clamp(min=1e-4)
    with torch.no_grad():
        assert torch.allclose(weighted(X), unweighted(X) / c_norm)
