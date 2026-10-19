from __future__ import annotations

import torch

from extrapolaw.optimizers.models.gp import PowerLawMean, make_default_single_obj_gp
from extrapolaw.space import ConfigEncoder, HPOFloat
from extrapolaw.utils.power_law import fit_power_law_mean_params


def _cost(n_params: float, n_data: float) -> float:
    return 6.0 * n_params * n_data


def test_power_law_mean_matches_the_neps_checkpoints() -> None:
    encoder = ConfigEncoder.from_parameters(
        {
            "n_params": HPOFloat(1e6, 1e9, log=True),
            "n_data": HPOFloat(1e8, 1e11, log=True),
        }
    )
    x = torch.rand(5, 2, dtype=torch.float64)
    y = torch.rand(5, 1, dtype=torch.float64)
    gp = make_default_single_obj_gp(
        x,
        y,
        encoder,
        cost_estimator=_cost,
        min_cost=1e15,
        mean_slope=-0.1,
        mean_bias=2.0,
    )
    assert isinstance(gp.mean_module, PowerLawMean)
    # GP checkpoints written with NePS store the power law under these names
    keys = set(gp.state_dict())
    assert {
        "mean_module.fixed_slope",
        "mean_module.fixed_bias",
        "mean_module.reference_flops",
    } <= keys

    costs = torch.tensor([_cost(**c) for c in encoder.decode(x)], dtype=torch.float64)
    # The fitted law in the standardized outcome space of the GP
    loss = 10 ** (-0.1 * torch.log10(costs) + 2.0)
    expected = (loss - y.mean()) / y.std()
    assert torch.allclose(gp.mean_module(x), expected)


def test_power_law_mean_params_only_fit_the_pareto_front() -> None:
    def _loss(n_params: float, n_data: float) -> float:
        return 10.0 * _cost(n_params, n_data) ** -0.05

    front = [{"n_params": 10.0**k, "n_data": 1e9} for k in range(6, 10)]
    front_losses = [_loss(**c) for c in front]
    # Worse than a cheaper configuration on the front, so dominated
    dominated = [{"n_params": 10.0**k, "n_data": 2e9} for k in range(6, 10)]
    dominated_losses = [2 * front_losses[0]] * len(dominated)

    on_front = fit_power_law_mean_params(front, front_losses, _cost)
    with_dominated = fit_power_law_mean_params(
        front + dominated, front_losses + dominated_losses, _cost
    )
    assert with_dominated.fit is not None
    assert with_dominated.fit.num_points == len(front)
    assert with_dominated.slope == on_front.slope
    assert with_dominated.intercept == on_front.intercept
    assert abs(on_front.slope + 0.05) < 1e-9
