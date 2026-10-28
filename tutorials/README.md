# Extrapolaw Tutorials

Interactive tutorials for learning hyperparameter optimization and neural architecture search with Extrapolaw.

## Tutorials

| Tutorial | Description | Run |
|----------|-------------|-----|
| **1. Getting Started with HPO** | Basic HPO workflow, synthetic functions, and deep learning tasks | [Local](1_getting_started_hpo.py) · [Colab](https://colab.research.google.com/github/automl/extrapolaw/blob/master/tutorials/1_getting_started_hpo.ipynb) |
| **2. Defining Search Spaces** | Parameter types, fidelity parameters, and PipelineSpace classes | [Local](2_search_spaces.py) · [Colab](https://colab.research.google.com/github/automl/extrapolaw/blob/master/tutorials/2_search_spaces.ipynb) |
| **3. Efficient Optimization** | Multi-fidelity optimization, expert priors, and advanced strategies | [Local](3_efficiency_techniques.py) · [Colab](https://colab.research.google.com/github/automl/extrapolaw/blob/master/tutorials/3_efficiency_techniques.ipynb) |
| **4. Multi-Objective Optimization** | Multi-objective optimization with PriMO, including per-objective expert priors | [Local](4_multi_objective.py) · [Colab](https://colab.research.google.com/github/automl/extrapolaw/blob/master/tutorials/4_multi_objective.ipynb) |

## Quick Start

### Run in Google Colab (Recommended)
Click the Colab link in the table above. No setup required.

### Run Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run a tutorial:
```bash
python 1_getting_started_hpo.py
```

Or use Jupyter:
```bash
pip install jupytext
jupytext --to notebook 1_getting_started_hpo.py
jupyter notebook 1_getting_started_hpo.ipynb
```

## Resources

- [Documentation](https://automl.github.io/extrapolaw/latest/)
- [GitHub Repository](https://github.com/automl/extrapolaw)
- [API Reference](https://automl.github.io/extrapolaw/latest/api/extrapolaw/api/)
- [More Examples](https://github.com/automl/extrapolaw/tree/master/extrapolaw_examples)
