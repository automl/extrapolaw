"""Tests for the scaling plots in `summary/scaling/`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import extrapolaw
from extrapolaw.space.extrapolaw_spaces.parameters import Float, Integer, PipelineSpace


class _Space(PipelineSpace):
    width = Integer(8, 512, log=True, is_compute_scaling=True)
    lr = Float(1e-4, 1e-1, log=True)


def _scaling(width: int, lr: float) -> dict:
    n_params = width**2
    return {
        "objective_to_minimize": 1.0 + 10.0 / n_params**0.3 + abs(np.log10(lr) + 2.5),
        "cost": 6.0 * n_params * 1e6,
        "info_dict": {
            "n_param": n_params,
            "dropout": 0.0 if width < 64 else 0.1,
            "predictions": {"task_acc": {"mean": 0.9 - 1.0 / width}},
        },
    }


def _no_cost(width: int, lr: float) -> dict:
    return {"objective_to_minimize": lr}


def _run(root: Path, evaluate_pipeline: object) -> None:
    extrapolaw.acquire(
        evaluate_pipeline=evaluate_pipeline,  # type: ignore[arg-type]
        pipeline_space=_Space(),
        root_directory=root,
        optimizer="random_search",
        worker_evaluations_to_spend=6,
    )


def test_scaling_plots_are_written(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _scaling)
    assert not (root / "summary" / "scaling").exists()

    extrapolaw.analyze(root)
    scaling = root / "summary" / "scaling"
    for name in (
        "loss_over_cost",
        "parameters/width_over_cost",
        "parameters/lr_over_cost",
        "parameters/n_param_over_cost",
        "downstream/task_acc_over_cost",
    ):
        assert (scaling / f"{name}.png").exists()
        assert len(pd.read_csv(scaling / f"{name}.csv")) == 6

    downstream = pd.read_csv(scaling / "downstream" / "task_acc_over_cost.csv")
    assert (downstream["downstream/task_acc"] < 0.5).all()

    table = pd.read_csv(scaling / "loss_over_cost.csv")
    assert table.loc[table["loss"].idxmin(), "on_frontier"]

    points = json.loads((scaling / "pareto_optimal_points.json").read_text())
    assert len(points) == table["on_frontier"].sum()
    assert set(points[0]["config"]) == {"width", "lr"}

    assert list((root / "summary_tb").glob("events.out.tfevents.*"))


def test_no_scaling_plots_without_cost(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _no_cost)
    extrapolaw.analyze(root)

    assert (root / "summary" / "incumbent_trajectory.png").exists()
    assert not (root / "summary" / "scaling").exists()
