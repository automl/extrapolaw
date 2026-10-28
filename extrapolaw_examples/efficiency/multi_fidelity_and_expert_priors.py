import logging

import numpy as np
import extrapolaw

# This example demonstrates Extrapolaw uses both fidelity and expert priors to
# optimize hyperparameters of a pipeline.


def evaluate_pipeline(float1, float2, integer1, fidelity):
    objective_to_minimize = -float(np.sum([float1, float2, integer1])) / fidelity
    return objective_to_minimize


class HPOSpace(extrapolaw.PipelineSpace):
    float1 = extrapolaw.Float(
        lower=1,
        upper=1000,
        log=False,
        prior=600,
        prior_confidence="medium",
    )
    float2 = extrapolaw.Float(
        lower=-10,
        upper=10,
        prior=0,
        prior_confidence="medium",
    )
    integer1 = extrapolaw.Integer(
        lower=0,
        upper=50,
        prior=35,
        prior_confidence="low",
    )
    fidelity = extrapolaw.IntegerFidelity(lower=1, upper=10)


logging.basicConfig(level=logging.INFO)
extrapolaw.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/multifidelity_priors",
    worker_fidelities_to_spend=25,  # For an alternate stopping method see multi_fidelity.py
)
