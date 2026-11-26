from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pytest_cases import fixture

import scalefm
from scalefm import Float, PipelineSpace, analyze, save_pipeline_results
from scalefm.optimizers import OptimizerInfo
from scalefm.optimizers.algorithms import random_search
from scalefm.runtime import DefaultWorker
from scalefm.state import (
    DefaultReportValues,
    OnErrorPossibilities,
    OptimizationState,
    ScaleFMState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
)


@fixture
def scalefm_state(tmp_path: Path) -> ScaleFMState:
    return ScaleFMState.create_or_load(
        path=tmp_path / "scalefm_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_async_happy_path_changes_state(scalefm_state: ScaleFMState) -> None:
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            cost_if_not_provided=10
        ),  # <- it is ignored
        worker_evaluations_to_spend=2,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
        is_async=True,
    )

    def never_called(**_):
        raise AssertionError("An async worker only samples.")

    DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=never_called,
        settings=settings,
    ).run()

    trials = scalefm_state.lock_and_read_trials()
    assert len(trials) == 2
    for trial in trials.values():
        assert trial.metadata.state == Trial.State.PENDING
        assert trial.report is None
    first, second = sorted(trials, key=lambda i: trials[i].metadata.time_sampled)

    def save(pipeline_id: str, user_result: dict) -> None:
        save_pipeline_results(
            pipeline_id=pipeline_id,
            user_result=user_result,
            root_directory=Path(scalefm_state.path),
        )

    save(first, {"objective_to_minimize": 0.3, "cost": 1.2})
    trials = scalefm_state.lock_and_read_trials()
    assert trials[first].metadata.state == Trial.State.SUCCESS
    assert trials[first].report.objective_to_minimize == 0.3
    assert trials[first].report.cost == 1.2

    # second trial is not reported yet
    assert trials[second].metadata.state == Trial.State.PENDING

    # Without a cost, the result is refused and the trial is still pending
    with pytest.raises(ValueError, match="no 'cost'"):
        save(second, {"objective_to_minimize": 10})
    trials = scalefm_state.lock_and_read_trials()
    assert trials[second].metadata.state == Trial.State.PENDING

    save(second, {"objective_to_minimize": 10, "cost": 2.0})
    trials = scalefm_state.lock_and_read_trials()
    assert trials[second].metadata.state == Trial.State.SUCCESS
    assert trials[second].report.objective_to_minimize == 10
    assert trials[second].report.cost == 2.0


def test_async_save_updates_the_whole_summary(scalefm_state: ScaleFMState) -> None:
    """The detached path writes the same summary artifacts as the worker loop,
    not only the CSVs.
    """
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
        is_async=True,
    )

    DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=lambda **_: None,
        settings=settings,
    ).run()

    summary_dir = Path(scalefm_state.path) / "summary"

    (pipeline_id,) = scalefm_state.lock_and_read_trials()
    save_pipeline_results(
        pipeline_id=pipeline_id,
        user_result={"objective_to_minimize": 0.3, "cost": 1.2},
        root_directory=Path(scalefm_state.path),
    )

    full = (summary_dir / "full.csv").read_text()
    assert "objective_to_minimize" in full
    assert "0.3" in full


def test_returning_none_without_async_raises(
    scalefm_state: ScaleFMState,
) -> None:
    """Returning None is only allowed when the worker runs in async mode."""
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: None,
        settings=settings,
    )
    with pytest.raises(ValueError, match="returned None"):
        worker.run()


def test_analyze_rebuilds_the_summary(scalefm_state: ScaleFMState) -> None:
    """`analyze` rebuilds the whole summary folder, plots included, from disk."""
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=3,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    DefaultWorker.new(
        state=scalefm_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: {"objective_to_minimize": 0.5, "cost": 1.0},
        settings=settings,
    ).run()

    summary_dir = Path(scalefm_state.path) / "summary"
    shutil.rmtree(summary_dir)

    analyze(scalefm_state.path)

    assert "objective_to_minimize" in (summary_dir / "full.csv").read_text()
    assert list(summary_dir.rglob("*.png"))


def test_analyze_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        analyze(missing)
    assert not missing.exists()


def test_async_acquire_refuses_an_evaluate_pipeline(tmp_path: Path) -> None:
    """With `is_async=True`, `acquire` only samples, so it takes no function."""
    with pytest.raises(ValueError, match="must be None"):
        scalefm.acquire(
            evaluate_pipeline=lambda **_: None,
            pipeline_space=ASpace(),
            root_directory=tmp_path / "root",
            is_async=True,
            worker_evaluations_to_spend=1,
        )
    assert not (tmp_path / "root").exists()


def test_sync_acquire_requires_an_evaluate_pipeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is required"):
        scalefm.acquire(
            pipeline_space=ASpace(),
            root_directory=tmp_path / "root",
            worker_evaluations_to_spend=1,
        )


def test_async_acquire_samples_configs_a_job_reports_later(tmp_path: Path) -> None:
    """`acquire(is_async=True)` leaves its configs pending in `config.yaml`, and
    `save_pipeline_results` completes them.
    """
    root = tmp_path / "root"
    scalefm.acquire(
        pipeline_space=ASpace(),
        root_directory=root,
        is_async=True,
        worker_evaluations_to_spend=2,
    )

    config_dirs = sorted((root / "configs").glob("config_*"))
    assert len(config_dirs) == 2
    assert all((d / "config.yaml").exists() for d in config_dirs)
    assert not any((d / "report.yaml").exists() for d in config_dirs)

    pipeline_id = config_dirs[0].name.removeprefix("config_")
    save_pipeline_results(
        user_result={"objective_to_minimize": 0.3, "cost": 1.0},
        pipeline_id=pipeline_id,
        root_directory=root,
    )
    states = {
        trial.id: trial.metadata.state
        for trial in ScaleFMState.create_or_load(root, load_only=True)
        .lock_and_read_trials()
        .values()
    }
    assert states[pipeline_id] == Trial.State.SUCCESS
    assert list(states.values()).count(Trial.State.PENDING) == 1


def test_fetch_pending_trials_and_mark_them_evaluating(tmp_path: Path) -> None:
    """Fetched trials stay pending until marked, marking takes them out of the
    pending ones, and the pending trials count toward the budgets.
    """
    root = tmp_path / "root"
    scalefm.acquire(
        pipeline_space=ASpace(),
        root_directory=root,
        is_async=True,
        worker_evaluations_to_spend=3,
    )

    def states() -> dict[str, Trial.State]:
        trials = ScaleFMState.create_or_load(root, load_only=True).lock_and_read_trials()
        return {t.id: t.metadata.state for t in trials.values()}

    assert list(states().values()) == [Trial.State.PENDING] * 3

    fetched = scalefm.fetch_pending_trials(root, 2)
    assert len(fetched) == 2
    assert all(set(config) == {"a"} for _, config in fetched)
    # Not marked yet, so fetched again
    assert scalefm.fetch_pending_trials(root, 2) == fetched

    pipeline_id = fetched[0][0]
    scalefm.mark_trial_evaluating(root, pipeline_id, worker_id="w1")
    trial = ScaleFMState.create_or_load(root, load_only=True).lock_and_get_trial_by_id(
        pipeline_id
    )
    assert trial.metadata.state == Trial.State.EVALUATING
    assert trial.metadata.evaluating_worker_id == "w1"
    with pytest.raises(scalefm.exceptions.ScaleFMError, match="not pending"):
        scalefm.mark_trial_evaluating(root, pipeline_id, worker_id="w2")

    remaining = [i for i, _ in scalefm.fetch_pending_trials(root, 5)]
    assert len(remaining) == 2
    assert pipeline_id not in remaining

    save_pipeline_results(
        user_result={"objective_to_minimize": 0.3, "cost": 1.0},
        pipeline_id=pipeline_id,
        root_directory=root,
    )
    assert states()[pipeline_id] == Trial.State.SUCCESS

    # The pending trials count toward the total budget: only one more is sampled
    scalefm.acquire(
        root_directory=root,
        is_async=True,
        worker_evaluations_to_spend=2,
        total_evaluations_to_spend=4,
    )
    assert len(states()) == 4


def test_acquire_defaults_to_random_search(tmp_path: Path) -> None:
    root = tmp_path / "root"
    scalefm.acquire(
        evaluate_pipeline=lambda a: {"objective_to_minimize": a, "cost": 1.0},
        pipeline_space=ASpace(),
        root_directory=root,
        worker_evaluations_to_spend=1,
    )
    assert scalefm.load_optimizer_info(root)["name"] == "random_search"
