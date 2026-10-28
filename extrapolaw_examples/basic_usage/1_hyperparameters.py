"""
This example demonstrates how to use Extrapolaw to optimize hyperparameters
of a pipeline. The pipeline is a simple function that takes in
five hyperparameters and returns their sum.
Extrapolaw uses the default optimizer to minimize this objective function.
"""

import logging
import numpy as np
import extrapolaw


def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    return {
        "objective_to_minimize": objective_to_minimize,
        "cost": categorical,
    }


class HPOSpace(extrapolaw.PipelineSpace):
    float1 = extrapolaw.Float(lower=0, upper=1)
    float2 = extrapolaw.Float(lower=-10, upper=10)
    categorical = extrapolaw.Categorical(choices=(0, 1))
    integer1 = extrapolaw.Integer(lower=0, upper=1)
    integer2 = extrapolaw.Integer(lower=2, upper=1024, log=True, log_base=2)


logging.basicConfig(level=logging.INFO)
extrapolaw.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/hyperparameters_example",
    worker_evaluations_to_spend=5,
    overwrite_root_directory=True,
)
