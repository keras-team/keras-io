# Keras Tuner Oracles

The `Oracle` class is the base class for all the tuning algorithms in Keras
Tuner.  A tuning algorithm implemented by subclassing the `Oracle` class needs
to receive evaluation results of models and generate new hyperparameter values
for the `Tuner` to create new models.

The built-in tuning algorithms subclassing the `Oracle` class includes
`RandomSearchOracle`, `BayesianOptimizationOracle`, and `HyperbandOracle`.

You can also write your own tuning algorithm by sublcassing the `Oracle` class.

{{toc}}
