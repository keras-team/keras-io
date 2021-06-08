# The Tuner classes in Keras Tuner

The base `Tuner` class is the class the manage the entire search process,
including model creation, fit and evaluation.  In each trial, it receives new
hyperparameter values from the `Oracle` instance.  After `model.fit(...)`, it
sends the evaluation results to the `Oracle` instance,

There are some built-in `Tuner` subclasses for the most widely used tuning
algorithms, including `RandomSearch`, `BayesianOptimization` and `Hyperband`.

You can subclass the `Tuner` subclasses to customize your tuning process.
For example, you can [override the `run_trial`
function](/guides/keras-tuner/custom_tuner/#overriding-runtrial) to customize
your model building and training process.

{{toc}}
