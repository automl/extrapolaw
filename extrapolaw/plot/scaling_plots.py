"""Scaling plots, with the CSV of their points, saved to `summary/scaling/`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from extrapolaw.optimizers.optimizer import Artifact, ArtifactType
from extrapolaw.utils.pareto import pareto_efficient
from extrapolaw.utils.power_law import fit_power_law

if TYPE_CHECKING:
    from pathlib import Path

    from extrapolaw.state.trial import Trial

logger = logging.getLogger(__name__)

PARETO_COLOR = "#D55E00"
DOMINATED_COLOR = "#B0B0B0"
COLUMNS = ["trial_id", "cost", "loss", "on_frontier"]

DOWNSTREAM_KEY = "predictions"
"""Key of the downstream metrics in the `info_dict`:
`{"predictions": {"<name>": {"mean": value}}}`."""


def _is_higher_better(metric: str) -> bool:
    """Accuracy-like metrics, which are flipped to `1 - value`."""
    return "_acc" in metric or "recall" in metric


def _downstream(extra: dict) -> dict[str, float]:
    """The downstream metrics of a report, as `downstream/<name>` columns."""
    metrics = {}
    for metric, reported in (extra.get(DOWNSTREAM_KEY) or {}).items():
        if not isinstance(reported, dict):
            continue
        value = reported.get("mean")
        if isinstance(value, int | float) and not isinstance(value, bool):
            flipped = 1.0 - value if _is_higher_better(metric) else value
            metrics[f"downstream/{metric}"] = float(flipped)
    return metrics


def runs_table(trials: Sequence[Trial]) -> pd.DataFrame:
    """One row per successful trial with a positive cost: its cost, loss, whether it is
    on the Pareto frontier, the numbers of its config and `info_dict`, and its
    downstream metrics.
    """
    rows: list[dict] = []
    for trial in trials:
        report = trial.report
        if report is None or report.reported_as != "success":
            continue
        if report.cost is None or not report.cost > 0:
            continue
        extra = dict(report.extra or {})
        row: dict = {}
        for values in (extra, trial.config):
            for name, value in values.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    row[name] = value
        row.update(_downstream(extra))
        row["trial_id"] = trial.id
        row["cost"] = float(report.cost)
        row["loss"] = float(report.objective_to_minimize)  # type: ignore[arg-type]
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["on_frontier"] = pareto_efficient(table[["cost", "loss"]].to_numpy(float))
    return table


def plot_loss_over_cost(table: pd.DataFrame, loss: str = "loss") -> list[Artifact]:
    """Every run's loss over its cost, the Pareto frontier and a power law through it.

    Args:
        table: The runs, from `runs_table`.
        loss: The column to plot: `"loss"`, or a downstream metric, which then gets a
            frontier of its own.
    """
    table = table.dropna(subset=[loss])
    on_frontier = pareto_efficient(table[["cost", loss]].to_numpy(float))
    table = table.assign(on_frontier=on_frontier)
    front = table[table["on_frontier"]].sort_values("cost")
    others = table[~table["on_frontier"]]

    metric = loss.removeprefix("downstream/")
    label = "Loss" if loss == "loss" else metric
    if _is_higher_better(metric):
        label = f"1 - {metric}"

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.scatter(
        others["cost"],
        others[loss],
        s=10,
        color=DOMINATED_COLOR,
        label="Dominated runs",
    )
    fit = fit_power_law(front["cost"].to_numpy(), front[loss].to_numpy())
    if fit is not None:
        ax.plot(
            front["cost"],
            10**fit.intercept * front["cost"] ** fit.slope,
            color="black",
            linewidth=2,
            label=f"Power-law fit (slope {fit.slope:.3f})",
        )
    ax.scatter(
        front["cost"],
        front[loss],
        s=36,
        color=PARETO_COLOR,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label="Pareto frontier",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Cost",
        ylabel=label,
        title=f"{label} over cost",
    )
    ax.grid(visible=True, alpha=0.3, linestyle="--")
    ax.legend()

    if loss == "loss":
        name = "scaling/loss_over_cost"
    else:
        name = f"scaling/downstream/{metric.replace('.', '_')}_over_cost"
    return [
        Artifact(name, fig, ArtifactType.FIGURE),
        Artifact(
            name, table[["trial_id", "cost", loss, "on_frontier"]], ArtifactType.CSV
        ),
    ]


def plot_parameter_over_cost(table: pd.DataFrame, parameter: str) -> list[Artifact]:
    """One parameter of every run over its cost, coloured by loss, with the frontier
    runs circled and a power law through them.
    """
    front = table[table["on_frontier"]].sort_values("cost")

    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot()
    points = ax.scatter(table["cost"], table[parameter], c=table["loss"], s=50, alpha=0.7)
    ax.scatter(
        front["cost"],
        front[parameter],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Pareto frontier",
    )
    fit = fit_power_law(front["cost"].to_numpy(), front[parameter].to_numpy())
    if fit is not None:
        ax.plot(
            front["cost"],
            10**fit.intercept * front["cost"] ** fit.slope,
            "r--",
            linewidth=2.5,
            label=f"Fit (slope {fit.slope:.3f})",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Cost",
        ylabel=parameter,
        title=f"{parameter} over cost",
    )
    ax.grid(visible=True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.colorbar(points, ax=ax, label="Loss")

    name = f"scaling/parameters/{parameter.replace('.', '_')}_over_cost"
    columns = ["trial_id", "cost", parameter, "loss", "on_frontier"]
    return [
        Artifact(name, fig, ArtifactType.FIGURE),
        Artifact(name, table[columns], ArtifactType.CSV),
    ]


def pareto_optimal_points(table: pd.DataFrame, trials: Sequence[Trial]) -> Artifact:
    """The runs on the Pareto frontier, by cost, with their cost, loss and config."""
    configs = {trial.id: dict(trial.config) for trial in trials}
    front = table[table["on_frontier"]].sort_values("cost")
    points = [
        {
            "trial_id": row.trial_id,
            "cost": row.cost,
            "loss": row.loss,
            "config": configs[str(row.trial_id)],
        }
        for row in front.itertuples()
    ]
    return Artifact("scaling/pareto_optimal_points", points, ArtifactType.JSON)


def plot_scaling(trials: Sequence[Trial]) -> list[Artifact]:
    """All scaling plots, if any successful run reports a cost."""
    table = runs_table(trials)
    if table.empty:
        return []

    artifacts = plot_loss_over_cost(table)
    artifacts.append(pareto_optimal_points(table, trials))

    downstream = [c for c in table.columns if c.startswith("downstream/")]
    for metric in downstream:
        artifacts += plot_loss_over_cost(table, metric)

    for parameter in table.columns.drop([*COLUMNS, *downstream]):
        values = table[parameter]
        if (
            values.isna().any()
            or values.nunique() < 2
            or np.allclose(values, table["cost"])
        ):
            continue
        artifacts += plot_parameter_over_cost(table, parameter)
    return artifacts


def log_to_tensorboard(artifacts: Sequence[Artifact], log_dir: Path, step: int) -> None:
    """Log the scaling figures to TensorBoard, one step per number of evaluated trials."""
    figures = [
        a
        for a in artifacts
        if a.name.startswith("scaling/") and a.artifact_type == ArtifactType.FIGURE
    ]
    if not figures:
        return
    try:
        from torch.utils.tensorboard.writer import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        for figure in figures:
            writer.add_figure(figure.name, figure.content, step, close=False)
        writer.close()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to log the scaling plots to TensorBoard: {e}")
