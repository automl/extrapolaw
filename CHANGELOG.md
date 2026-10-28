# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `extrapolaw.acquire(cost_estimator=..., max_cost_per_config=...)` for scaling studies. The cost estimator is a function or, for finite spaces, a JSON table. It fills in the `cost` of evaluations that do not report one and is handed to optimizers inheriting from `ScalingOptimizer`.
- `carbs` optimizer, wrapping [CARBS](https://github.com/imbue-ai/carbs).
- `chinchilla` optimizer for [Chinchilla](https://arxiv.org/abs/2203.15556) guided scaling studies.

### Changed
- **BREAKING:** Renamed `extrapolaw.run()` to `extrapolaw.acquire()`.
- **BREAKING:** Forked from NePS 0.17.0 and renamed the package `neps` → `extrapolaw` (PyPI package, import name, CLI, `EXTRAPOLAW_*` env vars, optimizer names, default result dirs, examples in `extrapolaw_examples/`).
- `bayesian_optimization` passes the space's constants to `constraints_func`, like `random_search` does.

Extrapolaw was forked from [NePS](https://github.com/automl/neps) 0.17.0. The earlier history is in [CHANGELOG_NEPS.md](CHANGELOG_NEPS.md).

[unreleased]: https://github.com/automl/extrapolaw/compare/v0.17.0...HEAD
