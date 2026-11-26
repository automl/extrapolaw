"""Log-log power-law fits used to derive the GP mean prior from observed trials."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class LogLogPowerLawFit:
    """A log-log power-law fit.

    `log10(metric - offset) = slope * log10(flops) + intercept`, where `offset` is the
    irreducible part of the metric, which the power law decays to, 0 for a pure power
    law. The quality stats are those of the linear fit in log-log space,
    given the offset.
    """

    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    intercept_std_err: float
    num_points: int
    offset: float = 0.0


def fit_power_law(
    flops: np.ndarray, y_metric: np.ndarray, *, log_scale: bool = True
) -> LogLogPowerLawFit | None:
    """Fit log10(metric) = slope * log10(flops) + intercept with quality stats."""
    fit_x = np.asarray(flops, dtype=float)
    fit_y = np.asarray(y_metric, dtype=float)
    mask = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_x > 0)
    if log_scale:
        # Only the log needs a positive metric; a linear fit keeps zeros/negatives
        # (e.g. ordinal HP indices starting at 0).
        mask &= fit_y > 0
    fit_x = fit_x[mask]
    fit_y = fit_y[mask]

    if len(fit_x) == 0:
        return None

    if log_scale:
        fit_x = np.log10(fit_x)
        fit_y = np.log10(fit_y)

    if len(np.unique(fit_x)) < 2:
        # A single FLOPs value can't determine a slope: fall back to a flat fit
        # that predicts the observed (mean) metric everywhere.
        return LogLogPowerLawFit(
            slope=0.0,
            intercept=float(np.mean(fit_y)),
            r_squared=0.0,
            slope_std_err=float("nan"),
            intercept_std_err=float("nan"),
            num_points=len(fit_x),
        )

    result: Any = linregress(fit_x, fit_y)

    return LogLogPowerLawFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
        slope_std_err=float(result.stderr),
        intercept_std_err=float(result.intercept_stderr),
        num_points=len(fit_x),
    )


#: Number of offsets, as fractions of the smallest metric, tried before refining the
#: best one in `fit_power_law_with_offset`.
OFFSET_GRID_SIZE = 200


def fit_power_law_with_offset(
    flops: np.ndarray, y_metric: np.ndarray
) -> LogLogPowerLawFit | None:
    """Fit `log10(metric - offset) = slope * log10(flops) + intercept`.

    The offset, the irreducible metric, lies in `[0, min(metric))`: for each offset the
    slope and intercept are a linear fit in log-log space, and the offset is the one
    whose power law fits log10(metric) best. With fewer than three distinct FLOPs values
    it cannot be told apart from the slope, so this is a pure power law then.
    """
    fit_x = np.asarray(flops, dtype=float)
    fit_y = np.asarray(y_metric, dtype=float)
    mask = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_x > 0) & (fit_y > 0)
    fit_x, fit_y = fit_x[mask], fit_y[mask]
    if len(np.unique(fit_x)) < 3:
        return fit_power_law(fit_x, fit_y)

    log_x = np.log10(fit_x)
    log_y = np.log10(fit_y)
    y_min = float(np.min(fit_y))

    def error(offset: float) -> float:
        result: Any = linregress(log_x, np.log10(fit_y - offset))
        predicted = offset + np.power(10.0, result.intercept + result.slope * log_x)
        return float(np.sum((np.log10(predicted) - log_y) ** 2))

    # From 0 up to just below the smallest metric, denser towards it, where the fit
    # changes fastest
    offsets = y_min * (1.0 - np.geomspace(1.0, 1e-4, OFFSET_GRID_SIZE))
    errors = np.array([error(o) for o in offsets])
    best = int(np.nanargmin(errors))
    offset, best_error = float(offsets[best]), float(errors[best])

    lower = offsets[max(best - 1, 0)]
    upper = offsets[min(best + 1, len(offsets) - 1)]
    if upper > lower:
        refined: Any = minimize_scalar(error, bounds=(lower, upper), method="bounded")
        if np.isfinite(refined.fun) and refined.fun < best_error:
            offset = float(refined.x)

    fit = fit_power_law(fit_x, fit_y - offset)
    if fit is None:
        return None
    fit.offset = offset
    return fit


#: Fewest Pareto points `scaling_regime_start` treats as a low-compute plateau.
MIN_PLATEAU_POINTS = 2
#: Fewest Pareto points `scaling_regime_start` keeps for the power law.
MIN_SCALING_POINTS = 3


def _line_sse(x: np.ndarray, y: np.ndarray) -> float:
    """Squared error of the least-squares line through (x, y)."""
    result: Any = linregress(x, y)
    return float(np.sum((y - (result.intercept + result.slope * x)) ** 2))


def scaling_regime_start(flops: Any, metric: Any) -> int:
    """Index of the first Pareto point in the power-law regime.

    Cheap runs often sit on a plateau, e.g. a downstream loss stuck near chance, before
    the metric starts to scale. That plateau would flatten a power law fit through the
    whole front. So the front, sorted by flops, is fit in log-log space both as one line
    and as a plateau (constant) followed by a line, at each possible break. The model
    with the lowest BIC wins, so a break must explain enough to pay for its two extra
    parameters (plateau level and break). Returns 0 when there is no plateau.
    """
    x = np.asarray(flops, dtype=float)
    y = np.asarray(metric, dtype=float)
    n = len(x)
    if (
        n < MIN_PLATEAU_POINTS + MIN_SCALING_POINTS
        or len(y) != n
        or not np.all(np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
        or np.any(np.diff(x) < 0)
    ):
        return 0
    x, y = np.log10(x), np.log10(y)
    # Floor the error, so an exact fit does not make the BIC -inf
    eps = 1e-12 * max(float(np.var(y)), 1e-12)

    def bic(sse: float, num_params: int) -> float:
        return float(n * np.log(sse / n + eps) + num_params * np.log(n))

    best_start, best_bic = 0, bic(_line_sse(x, y), 2)
    for k in range(MIN_PLATEAU_POINTS, n - MIN_SCALING_POINTS + 1):
        if len(np.unique(x[k:])) < 2:
            continue
        plateau_sse = float(np.sum((y[:k] - np.mean(y[:k])) ** 2))
        candidate = bic(plateau_sse + _line_sse(x[k:], y[k:]), 4)
        if candidate < best_bic:
            best_start, best_bic = k, candidate
    return best_start


@dataclass
class PowerLawMeanParams:
    """Power-law GP mean prior (slope/intercept), plus the FLOPs of all valid trials.

    The fit only uses the Pareto optimal ones, see `fit.num_points`.
    """

    slope: float
    intercept: float
    min_flops: int
    flops: list[float] = field(default_factory=list)
    fit: LogLogPowerLawFit | None = None


def fit_power_law_mean_params(
    configs: Sequence[Mapping[str, Any]],
    metrics: Sequence[float],
    cost_estimator: Callable[..., float],
    default_slope: float = -0.5,
    default_intercept: float = 5.0,
) -> PowerLawMeanParams:
    """Derive the (slope, intercept, min_flops) power-law prior.

    The power law is fit on the (flops, metric) Pareto front of the configurations.
    """
    flops_list: list[float] = []
    metrics_list: list[float] = []
    for config, metric in zip(configs, metrics, strict=False):
        try:
            flop_val = cost_estimator(**config)
        except Exception:
            logger.warning(
                "Failed to estimate FLOPs for config %s: %s",
                config,
                metric,
                exc_info=True,
            )
            continue
        if flop_val > 0 and np.isfinite(flop_val) and np.isfinite(metric):
            flops_list.append(float(flop_val))
            metrics_list.append(float(metric))

    # Imported here, as `scalefm.prediction` imports this module
    from scalefm.prediction.pareto import non_dominated_mask

    # The power law is the compute optimal frontier, so only the Pareto optimal
    # trials are fit, dominated ones would bias it towards worse losses, and only
    # past a low-compute plateau, if any
    front = non_dominated_mask(flops_list, metrics_list)
    front_flops = np.array(flops_list)[front]
    front_metrics = np.array(metrics_list)[front]
    order = np.argsort(front_flops, kind="stable")
    front_flops, front_metrics = front_flops[order], front_metrics[order]
    start = scaling_regime_start(front_flops, front_metrics)
    front_flops, front_metrics = front_flops[start:], front_metrics[start:]

    slope, intercept = default_slope, default_intercept
    fit_result = None
    if len(front_flops) >= 2 and len(np.unique(front_flops)) >= 2:
        fit_result = fit_power_law(
            flops=front_flops,
            y_metric=front_metrics,
            log_scale=True,
        )
        if fit_result is not None:
            slope, intercept = fit_result.slope, fit_result.intercept

    min_flops = int(min(flops_list)) if flops_list else int(1e16)
    return PowerLawMeanParams(
        slope=slope,
        intercept=intercept,
        min_flops=min_flops,
        flops=flops_list,
        fit=fit_result,
    )
