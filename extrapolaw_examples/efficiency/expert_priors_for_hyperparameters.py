import logging
import time

import extrapolaw


def evaluate_pipeline(some_float, some_integer, some_cat):
    start = time.time()
    if some_cat != "a":
        y = some_float + some_integer
    else:
        y = -some_float - some_integer
    end = time.time()
    return {
        "objective_to_minimize": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


# extrapolaw uses the default values and a confidence in this default value to construct a prior
# that speeds up the search
class HPOSpace(extrapolaw.PipelineSpace):
    some_float = extrapolaw.Float(
        lower=1,
        upper=1000,
        log=True,
        prior=900,
        prior_confidence="medium",
    )
    some_integer = extrapolaw.Integer(
        lower=0,
        upper=50,
        prior=35,
        prior_confidence="low",
    )
    some_cat = extrapolaw.Categorical(
        choices=("a", "b", "c"),
        prior=0,
        prior_confidence="high",
    )


logging.basicConfig(level=logging.INFO)
extrapolaw.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/user_priors_example",
    worker_evaluations_to_spend=15,
)
