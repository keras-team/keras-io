# The Tuner classes in KerasTuner

The base `Tuner` class is the class that manages the hyperparameter search process,
including model creation, training, and evaluation.  For each trial, a `Tuner` receives new
hyperparameter values from an `Oracle` instance.  After calling `model.fit(...)`, it
sends the evaluation results back to the `Oracle` instance and it retrieves the next set
of hyperparameters to try.

There are a few built-in `Tuner` subclasses available for widely-used tuning
algorithms: `RandomSearch`, `BayesianOptimization` and `Hyperband`.

You can also subclass the `Tuner` class to customize your tuning process.
In particular, you can [override the `run_trial` function](/guides/keras_tuner/custom_tuner/#overriding-runtrial)
to customize model building and training.

{{toc}}
