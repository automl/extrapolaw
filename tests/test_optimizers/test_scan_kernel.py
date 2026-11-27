from __future__ import annotations

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel

from scalefm.optimizers import bayesian_optimization as bo_module
from scalefm.optimizers.models.gp import ScanKernel, make_single_obj_gp
from scalefm.space import HPOCategorical, HPOFloat
from scalefm.space.encoding import ConfigEncoder
from scalefm.state import BudgetInfo, Trial
from tests.test_optimizers.test_budget_schedule import _optimizer, _trial

ENCODER = ConfigEncoder.from_parameters(
    {"lr": HPOFloat(1e-4, 1e-1, log=True), "bs": HPOFloat(16, 1024, log=True)}
)


def _valley(x: torch.Tensor) -> torch.Tensor:
    """A narrow valley along `log(lr) - log(bs) = const`, the linear scaling rule."""
    u = (x[:, 0] - x[:, 1]) / 2**0.5
    return 1 - torch.exp(-((u / 0.12) ** 2)) + 0.3 * (x[:, 0] + x[:, 1])


def _fit(kernel: str, seed: int = 0) -> tuple[SingleTaskGP, float]:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(30, 2, generator=g, dtype=torch.float64)
    torch.manual_seed(seed)
    gp = make_single_obj_gp(
        x, _valley(x).unsqueeze(-1), ENCODER, prior_mean="zero", kernel=kernel
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    x_test = torch.rand(1000, 2, generator=g, dtype=torch.float64)
    with torch.no_grad():
        mean = gp.posterior(x_test).mean.squeeze(-1)
    return gp, ((mean - _valley(x_test)) ** 2).mean().sqrt().item()


def test_scan_kernel_learns_the_coupled_direction() -> None:
    scan_gp, scan_rmse = _fit("scan")
    _, rbf_rmse = _fit("rbf")

    assert isinstance(scan_gp.covar_module, ScanKernel)
    metric = scan_gp.covar_module.metric().detach()
    # Moving `lr` and `bs` together is cheap: a strongly negative off-diagonal
    assert metric[0, 1] < -0.5 * metric.diagonal().min()
    assert scan_rmse < 0.5 * rbf_rmse


def test_scan_kernel_raises_without_numericals() -> None:
    encoder = ConfigEncoder.from_parameters({"c": HPOCategorical(["a", "b"])})
    x = encoder.encode([{"c": "a"}, {"c": "b"}])
    with pytest.raises(ValueError, match="numerical"):
        make_single_obj_gp(
            x,
            torch.rand(2, 1, dtype=torch.float64),
            encoder,
            prior_mean="zero",
            kernel="scan",
        )


def test_unknown_kernel_raises() -> None:
    x = torch.rand(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Unknown kernel"):
        make_single_obj_gp(
            x,
            torch.rand(3, 1, dtype=torch.float64),
            ENCODER,
            prior_mean="zero",
            kernel="matern",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kernel", "kernel_type"), [("rbf", RBFKernel), ("scan", ScanKernel)]
)
def test_bayesian_optimization_uses_the_kernel(
    kernel: str, kernel_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    gps: list[SingleTaskGP] = []
    make_gp = bo_module.make_single_obj_gp

    def spy(*args, **kwargs) -> SingleTaskGP:
        gps.append(make_gp(*args, **kwargs))
        return gps[-1]

    monkeypatch.setattr(bo_module, "make_single_obj_gp", spy)
    optimizer = _optimizer("none", kernel=kernel)
    assert optimizer.kernel == kernel
    trials: dict[str, Trial] = {}
    for _ in range(5):
        sampled = optimizer(trials, BudgetInfo())
        trials[sampled.id] = _trial(sampled)

    assert gps
    assert all(type(gp.covar_module) is kernel_type for gp in gps)
