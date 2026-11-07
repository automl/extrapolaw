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


def _is_higher_better(metric: str) -> bool:
    """Accuracy-like metrics, which are flipped to `1 - value`."""
    return "_acc" in metric or "recall" in metric


def _downstream(performances: dict[str, float]) -> dict[str, float]:
    """The downstream performances of a report, as `downstream/<name>` columns."""
    metrics = {}
    for metric, value in performances.items():
        flipped = 1.0 - value if _is_higher_better(metric) else value
        metrics[f"downstream/{metric}"] = flipped
    return metrics


def _costs(table: pd.DataFrame) -> list[str]:
    """`cost`, and `inference_cost` if the runs report it."""
    return ["cost", "inference_cost"] if "inference_cost" in table.columns else ["cost"]


def _with_frontier(table: pd.DataFrame, cost: str, loss: str) -> pd.DataFrame:
    """The runs that report `cost` and `loss`, with whether each is on their frontier."""
    table = table.dropna(subset=[cost, loss])
    return table.assign(on_frontier=pareto_efficient(table[[cost, loss]].to_numpy(float)))


def runs_table(trials: Sequence[Trial]) -> pd.DataFrame:
    """One row per successful trial with a positive cost: its cost, loss, whether it is
    on the Pareto frontier, the numbers of its config and `info_dict`, its inference
    cost and its downstream performances.
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
        row.update(_downstream(report.downstream_performances))
        if report.inference_cost is not None:
            row["inference_cost"] = report.inference_cost
        row["trial_id"] = trial.id
        row["cost"] = float(report.cost)
        row["loss"] = float(report.objective_to_minimize)  # type: ignore[arg-type]
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["on_frontier"] = pareto_efficient(table[["cost", "loss"]].to_numpy(float))
    return table


def plot_loss_over_cost(
    table: pd.DataFrame, loss: str = "loss", cost: str = "cost"
) -> list[Artifact]:
    """Every run's loss over its cost, the Pareto frontier and a power law through it.

    Args:
        table: The runs, from `runs_table`.
        loss: The column to plot: "loss", or a downstream metric, which then gets a
            frontier of its own.
        cost: The cost to plot over: "cost", or "inference_cost", which then gets
            a frontier of its own.
    """
    table = _with_frontier(table, cost, loss)
    front = table[table["on_frontier"]].sort_values("cost")
    others = table[~table["on_frontier"]]

    metric = loss.removeprefix("downstream/")
    label = "Loss" if loss == "loss" else metric
    if _is_higher_better(metric):
        label = f"1 - {metric}"
    cost_label = cost.replace("_", " ")

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.scatter(
        others[cost],
        others[loss],
        s=10,
        color=DOMINATED_COLOR,
        label="Dominated runs",
    )
    fit = fit_power_law(front[cost].to_numpy(), front[loss].to_numpy())
    if fit is not None:
        ax.plot(
            front[cost],
            10**fit.intercept * front[cost] ** fit.slope,
            color="black",
            linewidth=2,
            label=f"Power-law fit (slope {fit.slope:.3f})",
        )
    ax.scatter(
        front[cost],
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
        xlabel=cost_label.capitalize(),
        ylabel=label,
        title=f"{label} over {cost_label}",
    )
    ax.grid(visible=True, alpha=0.3, linestyle="--")
    ax.legend()

    if loss == "loss":
        name = f"scaling/loss_over_{cost}"
    else:
        name = f"scaling/downstream/{metric.replace('.', '_')}_over_{cost}"
    return [
        Artifact(name, fig, ArtifactType.FIGURE),
        Artifact(name, table[["trial_id", cost, loss, "on_frontier"]], ArtifactType.CSV),
    ]


def plot_parameter_over_cost(
    table: pd.DataFrame, parameter: str, cost: str = "cost"
) -> list[Artifact]:
    """One parameter of every run over its cost, coloured by loss, with the runs on the
    frontier of loss over this cost circled and a power law through them.
    """
    table = _with_frontier(table, cost, "loss")
    front = table[table["on_frontier"]].sort_values(cost)
    cost_label = cost.replace("_", " ")

    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot()
    points = ax.scatter(table[cost], table[parameter], c=table["loss"], s=50, alpha=0.7)
    ax.scatter(
        front[cost],
        front[parameter],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Pareto frontier",
    )
    fit = fit_power_law(front[cost].to_numpy(), front[parameter].to_numpy())
    if fit is not None:
        ax.plot(
            front[cost],
            10**fit.intercept * front[cost] ** fit.slope,
            "r--",
            linewidth=2.5,
            label=f"Fit (slope {fit.slope:.3f})",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=cost_label.capitalize(),
        ylabel=parameter,
        title=f"{parameter} over {cost_label}",
    )
    ax.grid(visible=True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.colorbar(points, ax=ax, label="Loss")

    name = f"scaling/parameters/{parameter.replace('.', '_')}_over_{cost}"
    columns = ["trial_id", cost, parameter, "loss", "on_frontier"]
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
    """All scaling plots, over the cost and, if reported, the inference cost, if any
    successful run reports a cost.
    """
    table = runs_table(trials)
    if table.empty:
        return []

    downstream = [c for c in table.columns if c.startswith("downstream/")]
    costs = _costs(table)
    parameters = []
    for parameter in table.columns.drop([*COLUMNS, *downstream, *costs[1:]]):
        values = table[parameter]
        if (
            values.isna().any()
            or values.nunique() < 2
            or np.allclose(values, table["cost"])
        ):
            continue
        parameters.append(parameter)

    artifacts = [pareto_optimal_points(table, trials)]
    for cost in costs:
        artifacts += plot_loss_over_cost(table, cost=cost)
        for metric in downstream:
            artifacts += plot_loss_over_cost(table, metric, cost)
        for parameter in parameters:
            artifacts += plot_parameter_over_cost(table, parameter, cost)
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
