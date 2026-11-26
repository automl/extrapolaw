
This content tree contains the core components and utilities of ScaleFM, designed to simplify and enhance the process of running scaling studies. Below is an overview of the files and their purposes:

- [`scalefm.api`](scalefm/api.md): The main functions of a scaling study:
    - `scalefm.acquire()` **samples configurations** under a cost budget and **trains them** with your `evaluate_pipeline`, or only samples them with `is_async=True`;
    - `scalefm.save_pipeline_results()` **reports the result** of a configuration you trained yourself, e.g. in a Slurm job;
    - `scalefm.predict()` **fits the scaling law** and predicts at larger costs;
- `scalefm.prediction`: The predictors behind `scalefm.predict()`: the [power law](scalefm/prediction/power_law.md), [Chinchilla](scalefm/prediction/chinchilla3.md) and the [GP posterior](scalefm/prediction/gp.md).
- `scalefm.optimizers`:
    - [`scalefm.algorithms`](scalefm/optimizers/algorithms.md): Contains the optimization algorithms, such as random search, grid search, Bayesian optimization, CARBS and Chinchilla, for sampling new trials.
    - [`scalefm.AskAndTell`](scalefm/optimizers/ask_and_tell.md): An alternative to `scalefm.acquire` that allows full control of the evaluation loop. This is useful when you don’t want to use ScaleFM’s runtime but still want to benefit from its optimizers and state management.
- [`scalefm.runtime`](scalefm/runtime.md): Implements the `Worker`, offering functions to create a worker, sample new trials, and evaluate them.
- [`scalefm.state`](scalefm/state/scalefm_state.md): Manages the state of workers, trials, and optimizers, ensuring reproducibility and continuity.
- [`scalefm.status`](scalefm/status/status.md): Provides functions to retrieve the status of a run and export it to CSV files for analysis, and the [`summary`](scalefm/status/summary.md) folder with the scaling plots.
- [`scalefm.plot`](scalefm/plot/scaling_plots.md): The scaling plots of a run, and [`tblogger`](scalefm/plot/tensorboard_eval.md) to log your own metrics to TensorBoard.
