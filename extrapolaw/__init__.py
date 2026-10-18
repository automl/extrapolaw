"""Extrapolaw: A framework for Neural Architecture Search and Hyperparameter Optimization.
This module provides a unified interface for defining search spaces, running optimizers,
and visualizing results. It includes various optimizers, search space definitions,
and plotting utilities, making it easy to experiment with different configurations
and algorithms.
"""

import logging

from extrapolaw.api import (
    analyze,
    create_config,
    import_trials,
    load_config,
    load_optimizer_info,
    load_pipeline_space,
    predict,
    run,
    save_pipeline_results,
)
from extrapolaw.optimizers import algorithms
from extrapolaw.optimizers.ask_and_tell import AskAndTell
from extrapolaw.optimizers.optimizer import SampledConfig
from extrapolaw.plot.plot import plot
from extrapolaw.plot.tensorboard_eval import tblogger
from extrapolaw.space import SearchSpace
from extrapolaw.space.extrapolaw_spaces.parameters import (
    ByName,
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    FloatFidelity,
    Integer,
    IntegerFidelity,
    Operation,
    PipelineSpace,
    Resample,
)
from extrapolaw.state import BudgetInfo, Trial
from extrapolaw.state.pipeline_eval import UserResultDict
from extrapolaw.status.status import status
from extrapolaw.utils import convert_operation_to_callable
from extrapolaw.utils.files import load_and_merge_yamls

# As a library, Extrapolaw does not configure logging: no handlers, no levels, no format.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AskAndTell",
    "BudgetInfo",
    "ByName",
    "Categorical",
    "ConfidenceLevel",
    "Fidelity",
    "Float",
    "FloatFidelity",
    "Integer",
    "IntegerFidelity",
    "Operation",
    "PipelineSpace",
    "Resample",
    "SampledConfig",
    "SearchSpace",
    "Trial",
    "UserResultDict",
    "algorithms",
    "analyze",
    "convert_operation_to_callable",
    "create_config",
    "import_trials",
    "load_and_merge_yamls",
    "load_config",
    "load_optimizer_info",
    "load_pipeline_space",
    "plot",
    "predict",
    "run",
    "save_pipeline_results",
    "status",
    "tblogger",
]
