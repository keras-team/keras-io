# KerasTuner Oracles

The `Oracle` class is the base class for all the search algorithms in KerasTuner.
An `Oracle` object receives evaluation results for a model (from a `Tuner` class)
and generates new hyperparameter values.

The built-in `Oracle` classes are
`RandomSearchOracle`, `BayesianOptimizationOracle`, and `HyperbandOracle`.

You can also write your own tuning algorithm by subclassing the `Oracle` class.

{{toc}}
