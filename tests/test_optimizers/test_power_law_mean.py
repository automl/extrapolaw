from __future__ import annotations

import torch

from autoscan.optimizers.models.gp import PowerLawMean, make_default_single_obj_gp
from autoscan.space import ConfigEncoder, HPOFloat


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
    gp = make_default_single_obj_gp(
        x,
        torch.rand(5, 1, dtype=torch.float64),
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
    expected = -0.1 * torch.log(costs / 1e15) + 2.0
    assert torch.allclose(gp.mean_module(x), expected)
