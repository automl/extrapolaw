"""How to generate a summary (scalefm.status) of a run.
Before running this example analysis, run the hyperparameters example with:
    python -m scalefm_examples.basic_usage.hyperparameters
"""

import scalefm

# 1. At all times, ScaleFM maintains several files in the root directory that are human
# read-able and can be useful

# 2. Printing a summary and reading in results.
full, summary = scalefm.status("results/hyperparameters_example", print_summary=True)
config_id = "1"

print("\n", full.head(), "\n")
print(full.loc[config_id])
