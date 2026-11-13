"""Scaling plots, with the CSV of their points, saved to `summary/scaling/`."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import portalocker as pl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from extrapolaw.optimizers.optimizer import Artifact, ArtifactType
from extrapolaw.state.filebased import FileLocker
from extrapolaw.utils.pareto import pareto_efficient
from extrapolaw.utils.power_law import fit_power_law

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from extrapolaw.state.trial import Trial

logger = logging.getLogger(__name__)

PARETO_COLOR = "#D55E00"
DOMINATED_COLOR = "#B0B0B0"
INK = "#2B2B2B"
SECONDARY_INK = "#52514E"
AXIS_COLOR = "#C3C2B7"
GRID_COLOR = "#E6E5DF"
EXTRAPOLATION_COLOR = "#F3F2EE"
TARGET_COLORS = ["#86B6EF", "#6DA7EC", "#5598E7", "#3987E5", "#2A78D6", "#256ABF"]
"""Blues from light to dark, one per target budget of the prediction plot."""
COLUMNS = ["trial_id", "cost", "loss", "on_frontier"]
TENSORBOARD_LOCK_TIMEOUT = 5.0
"""Seconds to wait for another worker logging to TensorBoard, before trying again at
the next update."""


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
    table: pd.DataFrame,
    loss: str = "loss",
    cost: str = "cost",
    fit_line: tuple[Sequence[float], Sequence[float], str] | None = None,
) -> list[Artifact]:
    """Every run's loss over its cost, the Pareto frontier and a power law through it.

    Args:
        table: The runs, from `runs_table`.
        loss: The column to plot: "loss", or a downstream metric, which then gets a
            frontier of its own.
        cost: The cost to plot over: "cost", or "inference_cost", which then gets
            a frontier of its own.
        fit_line: The x and y values and the label of a fitted line to draw instead
            of the power law, e.g. from `fitted_losses`.
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
    if fit_line is not None:
        xs, ys, fit_label = fit_line
        ax.plot(xs, ys, color="black", linewidth=2, label=fit_label)
    elif fit is not None:
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


def fitted_losses(
    trials: Sequence[Trial],
    predictor: str,
    flops: Sequence[float],
    data_parameter_name: str | None = None,
    params_estimator: Callable[..., float] | None = None,
) -> list[float] | None:
    """The best loss the scaling law `predictor` predicts at each of `flops`, fitted on
    `trials`, or None if it cannot be fitted on them.
    """
    from extrapolaw.prediction import (
        Target,
        fit_and_predict,
        get_predictor_type,
        rows_from_trials,
    )

    rows, _curves, configs = rows_from_trials(
        trials,
        params_estimator=params_estimator,
        data_parameter_name=data_parameter_name,
    )
    try:
        predictions = fit_and_predict(
            get_predictor_type(predictor),
            [float(f) for f in flops],
            targets=[Target.LOSS],
            rows=rows,
            configs=configs,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not fit '{predictor}' on the runs: {e}")
        return None

    losses = [p.loss for p in predictions]
    for loss in losses:
        if loss is None or not np.isfinite(loss):
            return None
    return [float(loss) for loss in losses]  # type: ignore[arg-type]


def plot_scaling(
    trials: Sequence[Trial],
    predictor: str = "power_law",
    data_parameter_name: str | None = None,
) -> list[Artifact]:
    """All scaling plots, over the cost and, if reported, the inference cost, if any
    successful run reports a cost.

    The loss over the cost is fitted with `predictor`, the scaling law of the run's
    optimizer, falling back to a plain power law if it cannot be fitted.
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

    xs = np.geomspace(table["cost"].min(), table["cost"].max(), 50)
    ys = fitted_losses(trials, predictor, xs, data_parameter_name)
    fit_line = None
    if ys is not None:
        fit_line = (list(xs), ys, f"{predictor} fit")

    artifacts = [pareto_optimal_points(table, trials)]
    for cost in costs:
        if cost == "cost":
            artifacts += plot_loss_over_cost(table, cost=cost, fit_line=fit_line)
        else:
            artifacts += plot_loss_over_cost(table, cost=cost)
        for metric in downstream:
            artifacts += plot_loss_over_cost(table, metric, cost)
        for parameter in parameters:
            artifacts += plot_parameter_over_cost(table, parameter, cost)
    return artifacts


def _power_of_ten(value: float) -> str:
    """`value` as `10^13` or `3x10^13`, in matplotlib's math text."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = round(value / 10**exponent, 1)
    if mantissa == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\times10^{{{exponent}}}$"


def _plain_log_labels(ax: Axes) -> None:
    """Label the log y axis with plain numbers, e.g. 2, 5, 10, instead of 2x10^0."""
    low, high = ax.get_ylim()
    shown = list(range(1, 10))
    if high / low > 10:
        shown = [1, 2, 5]
    elif high / low > 3:
        shown = [1, 2, 3, 5]

    def label(value: float, _position: int) -> str:
        mantissa = round(value / 10 ** np.floor(np.log10(value)))
        return f"{value:g}" if mantissa in shown else ""

    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(FuncFormatter(label))


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Light axes and grid, so the data stands out."""
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=SECONDARY_INK)
    ax.set_ylabel(ylabel, color=SECONDARY_INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_COLOR)
    ax.tick_params(which="both", color=AXIS_COLOR, labelcolor=SECONDARY_INK)
    ax.grid(visible=True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)


def _target_colors(n_targets: int) -> list[str]:
    """A blue per target budget, darker for larger budgets."""
    if n_targets == 1:
        return [TARGET_COLORS[-2]]
    steps = np.linspace(0, len(TARGET_COLORS) - 1, n_targets).round()
    return [TARGET_COLORS[int(step)] for step in steps]


def _prediction_history(
    finished: Sequence[Trial],
    predictor: str,
    target_flops: Sequence[float],
    data_parameter_name: str | None = None,
    params_estimator: Callable[..., float] | None = None,
) -> pd.DataFrame:
    """The prediction at `target_flops` after each finished run, at most 20 refits."""
    history = []
    for n in sorted({int(n) for n in np.linspace(3, len(finished), 20)}):
        losses = fitted_losses(
            finished[:n], predictor, target_flops, data_parameter_name, params_estimator
        )
        if losses is None:
            continue
        spent = runs_table(finished[:n])["cost"].sum()
        for flops, loss in zip(target_flops, losses, strict=True):
            history.append(
                {"n_runs": n, "cost_spent": spent, "target_flops": flops, "loss": loss}
            )
    return pd.DataFrame(history)


def plot_prediction(
    trials: Sequence[Trial],
    predictor: str,
    target_flops: Sequence[float],
    data_parameter_name: str | None = None,
    params_estimator: Callable[..., float] | None = None,
) -> list[Artifact]:
    """The prediction of `predictor` at `target_flops`, in two panels.

    Left: the runs, their Pareto frontier and the fit, extended (dashed) to the target
    budgets with the predicted losses. Right: the predicted loss at each target
    budget, refitted after each finished run, to see whether it has settled.
    """
    finished = sorted(trials, key=lambda t: t.metadata.time_end or float("inf"))
    table = runs_table(finished)
    if table.empty:
        return []
    predicted = fitted_losses(
        finished, predictor, target_flops, data_parameter_name, params_estimator
    )
    if predicted is None:
        return []
    table = _with_frontier(table, "cost", "loss")
    front = table[table["on_frontier"]]
    largest_run = table["cost"].max()
    colors = _target_colors(len(target_flops))

    fig = Figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(
        table["cost"],
        table["loss"],
        s=16,
        color=DOMINATED_COLOR,
        linewidths=0,
        label="Runs",
    )
    ax.scatter(
        front["cost"],
        front["loss"],
        s=44,
        color=PARETO_COLOR,
        edgecolors="white",
        linewidths=1,
        zorder=3,
        label="Pareto frontier",
    )
    xs = np.geomspace(table["cost"].min(), max(target_flops), 60)
    ys = fitted_losses(finished, predictor, xs, data_parameter_name, params_estimator)
    if ys is not None:
        fit = np.asarray(ys)
        inside = xs <= largest_run
        last = inside.sum() - 1
        ax.plot(xs[inside], fit[inside], color=INK, linewidth=2, label=f"{predictor} fit")
        ax.plot(xs[last:], fit[last:], color=INK, linewidth=2, linestyle=(0, (4, 3)))
    ax.scatter(
        target_flops,
        predicted,
        s=90,
        c=colors,
        edgecolors="white",
        linewidths=1.5,
        zorder=4,
        label="Prediction",
    )
    for flops, loss in zip(target_flops, predicted, strict=True):
        ax.annotate(
            f"{loss:.3g}",
            (flops, loss),
            textcoords="offset points",
            xytext=(8, 6),
            color=INK,
            fontweight="bold",
        )
    ax.set(xscale="log", yscale="log")
    xmin, xmax = ax.get_xlim()
    ax.axvspan(
        largest_run,
        xmax,
        color=EXTRAPOLATION_COLOR,
        zorder=0,
        label="Extrapolation region",
    )
    ax.set_xlim(xmin, xmax)
    _plain_log_labels(ax)
    _style(ax, f"Predicted loss with {predictor}", "Training cost", "Loss")

    history = _prediction_history(
        finished, predictor, target_flops, data_parameter_name, params_estimator
    )
    ax = fig.add_subplot(1, 2, 2)
    for flops, color in zip(target_flops, colors, strict=True):
        if history.empty:
            break
        rows = history[history["target_flops"] == flops]
        ax.plot(
            rows["cost_spent"],
            rows["loss"],
            color=color,
            linewidth=2,
            drawstyle="steps-post",
            marker="o",
            markersize=6,
            markeredgecolor="white",
            label=f"Target {_power_of_ten(flops)}",
        )
    ax.set(xscale="log", yscale="log")
    _plain_log_labels(ax)
    _style(ax, "Prediction convergence", "Cumulative training cost", "Predicted loss")
    fig.tight_layout(w_pad=3)

    predictions = pd.DataFrame({"target_flops": target_flops, "loss": predicted})
    return [
        Artifact("scaling/prediction", fig, ArtifactType.FIGURE),
        Artifact("scaling/prediction", predictions, ArtifactType.CSV),
        Artifact("scaling/prediction_history", history, ArtifactType.CSV),
    ]


def log_to_tensorboard(trials: Sequence[Trial], log_dir: Path) -> None:
    """Log the scaling data to TensorBoard, one step per run, adding only new runs.

    A run's step is the order it was logged in, which is about the order the runs
    finished. At every step:

    - `scaling/incumbent/loss`: the best loss so far
    - `scaling/frontier/...`: the runs on each Pareto frontier so far, as
      `(trial id, value, cost)`

    The ids of the logged runs are kept, in order, in `log_dir/logged_trial_ids.txt`,
    so that later updates, other workers and `extrapolaw.analyze()` add only the new
    runs, at the next steps.
    It never raises: if logging fails, or another worker holds the lock for longer
    than `TENSORBOARD_LOCK_TIMEOUT`, it warns and the runs are logged at the next
    update.
    """
    finished = sorted(trials, key=lambda t: t.metadata.time_end or float("inf"))
    try:
        table = runs_table(finished)
        if table.empty:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        locker = FileLocker(
            log_dir / ".tensorboard.lock", poll=0.1, timeout=TENSORBOARD_LOCK_TIMEOUT
        )
        with locker.lock():
            _log_new_runs(table, log_dir)
    except pl.exceptions.LockException:
        logger.warning(
            "TensorBoard is busy with another worker, the runs are logged to it at"
            " the next update."
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not log the scaling data to TensorBoard, retrying at the next"
            f" update: {e}"
        )


def _log_new_runs(table: pd.DataFrame, log_dir: Path) -> None:
    """Log the runs of `table` that are not logged yet."""
    logged_file = log_dir / "logged_trial_ids.txt"
    logged = logged_file.read_text().split() if logged_file.exists() else []
    is_new = ~table["trial_id"].isin(logged)
    if not is_new.any():
        return

    from torch.utils.tensorboard.writer import SummaryWriter

    first_step = len(logged) + 1
    with tempfile.TemporaryDirectory() as staging:
        writer = SummaryWriter(log_dir=staging)
        downstream = [c for c in table.columns if c.startswith("downstream/")]
        new = table[is_new]
        for i, trial_id in enumerate(new["trial_id"]):
            logged.append(trial_id)
            step = len(logged)
            seen = pd.concat([table[~is_new], new.iloc[: i + 1]])
            writer.add_scalar("scaling/incumbent/loss", seen["loss"].min(), step)
            for cost in _costs(table):
                for loss in ["loss", *downstream]:
                    tag = f"scaling/frontier/{loss}_over_{cost}"
                    writer.add_text(tag, _frontier_text(seen, cost, loss), step)
        writer.close()
        [event_file] = Path(staging).iterdir()
        shutil.move(event_file, log_dir / f"events.out.tfevents.{first_step:010d}")
    logged_file.write_text("\n".join(logged))


def _frontier_text(table: pd.DataFrame, cost: str, loss: str) -> str:
    """The runs on the frontier of `loss` over `cost`, as `(trial id, value, cost)`."""
    front = _with_frontier(table, cost, loss)
    front = front[front["on_frontier"]].sort_values(cost)
    values = front[loss]
    if _is_higher_better(loss.removeprefix("downstream/")):
        values = 1.0 - values
    points = [
        f"({trial_id}, {value:.4g}, {run_cost:.4g})"
        for trial_id, value, run_cost in zip(
            front["trial_id"], values, front[cost], strict=True
        )
    ]
    return ", ".join(points)
