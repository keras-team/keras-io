"""
Title: Getting started with KerasTuner
Authors: Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin
Date created: 2019/05/31
Last modified: 2021/06/07
Description: The basics of using KerasTuner to tune model hyperparameters.
"""

"""
## Setup
"""

"""shell
pip install keras-tuner -q
"""

"""
## Introduction

Here's how to perform hyperparameter tuning for a single-layer dense neural
network using random search.
First, we need to prepare the dataset -- let's use MNIST dataset as an example.
"""

from tensorflow import keras
import numpy as np

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Prepare a model-building function

Then, we define a model-building function. It takes an argument `hp` from
which you can sample hyperparameters, such as
`hp.Int('units', min_value=32, max_value=512, step=32)`
(an integer from a certain range).

This function returns a compiled model.
"""

from tensorflow.keras import layers
from keras_tuner import RandomSearch


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


"""
## Start the search

Next, let's instantiate a tuner. You should specify the model-building function, the
name of the objective to optimize (whether to minimize or maximize is
automatically inferred for built-in metrics), the total number of trials
(`max_trials`) to test, and the number of models that should be built and fit
for each trial (`executions_per_trial`).

We use the `overwrite` argument to control whether to overwrite the previous
results in the same directory or resume the previous search instead.  Here we
set `overwrite=True` to start a new search and ignore any previous results.

Available tuners are `RandomSearch`, `BayesianOptimization` and `Hyperband`.

**Note:** the purpose of having multiple executions per trial is to reduce
results variance and therefore be able to more accurately assess the
performance of a model. If you want to get results faster, you could set
`executions_per_trial=1` (single round of training for each model
configuration).
"""

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

"""
You can print a summary of the search space:
"""

tuner.search_space_summary()

"""
Then, start the search for the best hyperparameter configuration.
The call to `search` has the same signature as `model.fit()`.
"""

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

"""
Here's what happens in `search`: models are built iteratively by calling the
model-building function, which populates the hyperparameter space (search
space) tracked by the `hp` object. The tuner progressively explores the space,
recording metrics for each configuration.
"""

"""
## Query the results

When search is over, you can retrieve the best model(s):
"""

models = tuner.get_best_models(num_models=2)

"""
Or print a summary of the results:
"""

tuner.results_summary()

"""
You will also find detailed logs, checkpoints, etc, in the folder `my_dir/helloworld`, i.e. `directory/project_name`.
"""

"""
## The search space may contain conditional hyperparameters

Below, we have a `for` loop creating a tunable number of layers,
which themselves involve a tunable `units` parameter.

This can be pushed to any level of parameter interdependency, including recursion.

Note that all parameter names should be unique (here, in the loop over `i`,
we name the inner parameters `'units_' + str(i)`).
"""


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers", 2, 20)):
        model.add(
            layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


"""
## You can use a HyperModel subclass instead of a model-building function

This makes it easy to share and reuse hypermodels.

A `HyperModel` subclass only needs to implement a `build(self, hp)` method.
"""

from keras_tuner import HyperModel


class MyHyperModel(HyperModel):
    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(self.classes, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


hypermodel = MyHyperModel(classes=10)

tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

"""
## Use a custom metric as the tuning objective

You may implement your own metric and use it as the hyperparameter search objective.

Here, we use mean squared error (MSE) as an example.

First, we implement the MSE metric by extending the `keras.metrics.Metric`
class. Remember to give a name to your metric using the `name` argument of
`super().__init__()`, which will be used later.

Note: MSE is actully a build-in metric, which can be imported with
`keras.metrics.MeanSquaredError`. This is just an example to show how to use a custom metric
as the hyperparameter search objective.
For more information about implementing custom metrics, please
see [this tutorial](https://keras.io/api/metrics/#creating-custom-metrics). If
you would like a metric with a different function signature than
`update_state(y_true, y_pred, sample_weight)`, you can override the
`train_step()` method of your model following
[this tutorial](https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel).
"""

import tensorflow as tf

class MeanSquaredError(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="mean_squared_error", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.int32, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.math.squared_difference(y_pred, y_true)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
            count = tf.multiply(count, sample_weight)
        self.sum.assign_add(tf.reduce_sum(values))
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide(self.sum, tf.cast(self.count, tf.float32))

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)


"""
Second, we pass a `MeanSquaredError` instance to `model.compile()` in `build_model()`.
"""


def build_model(hp):
    model = keras.Sequential(
        [
            layers.Dense(10, activation="relu"),
            layers.Dense(hp.Int("units", 5, 50), activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="mae", metrics=[MeanSquaredError()])  # Use mse as a metric
    return model


"""
Finally, we use `Objective(name='val_mean_squared_error', direction='min')`
as the `objective` of the `Tuner`. The `name` should be in the form of
`'val_' + metric_name`, referring the metric on the validation data. We
specified the metric name as `'mean_squared_error'` in the initializer.
Therefore, the name for the objective is `'val_mean_squared_error'`.  The
`direction` should be either `'min'` (the lower the better) or `'max'` (the
higher the better).

"""

from keras_tuner import Objective

tuner = RandomSearch(
    build_model,
    objective=Objective("val_mean_squared_error", direction="min"),
    max_trials=3,
    overwrite=True,
)

# Run the search as a quick test.
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    epochs=2,
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1))
)

"""
## Pre-made tunable applications: HyperResNet and HyperXception

These are ready-to-use hypermodels for computer vision.

They come pre-compiled with `loss="categorical_crossentropy"` and `metrics=["accuracy"]`.
"""

from keras_tuner.applications import HyperResNet

hypermodel = HyperResNet(input_shape=(28, 28, 1), classes=10)

tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_val[:100], y_val[:100])
)

"""
## You can easily restrict the search space to just a few parameters

If you have an existing hypermodel, and you want to search over only a few parameters
(such as the learning rate), you can do so by passing a `hyperparameters` argument
to the tuner constructor, as well as `tune_new_entries=False` to specify that parameters
that you didn't list in `hyperparameters` should not be tuned. For these parameters, the default
value gets used.
"""

from keras_tuner import HyperParameters
from keras_tuner.applications import HyperXception

hypermodel = HyperXception(input_shape=(28, 28, 1), classes=10)

hp = HyperParameters()

# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

tuner = RandomSearch(
    hypermodel,
    hyperparameters=hp,
    # `tune_new_entries=False` prevents unlisted parameters from being tuned
    tune_new_entries=False,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_val[:100], y_val[:100])
)

"""
## About parameter default values

Whenever you register a hyperparameter inside a model-building function or the `build` method of a hypermodel,
you can specify a default value:

```python
hp.Int("units", min_value=32, max_value=512, step=32, default=128)
```

If you don't, hyperparameters always have a default default (for `Int`, it is equal to `min_value`).

## Fixing values in a hypermodel

What if you want to do the reverse -- tune all available parameters in a hypermodel, **except** one (the learning rate)?

Pass a `hyperparameters` argument with a `Fixed` entry (or any number of `Fixed` entries), and specify `tune_new_entries=True`.
"""

hypermodel = HyperXception(input_shape=(28, 28, 1), classes=10)

hp = HyperParameters()
hp.Fixed("learning_rate", value=1e-4)

tuner = RandomSearch(
    hypermodel,
    hyperparameters=hp,
    tune_new_entries=True,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_val[:100], y_val[:100])
)

"""
## Overriding compilation arguments

If you have a hypermodel for which you want to change the existing optimizer,
loss, or metrics, you can do so by passing these arguments
to the tuner constructor:
"""

hypermodel = HyperXception(input_shape=(28, 28, 1), classes=10)

tuner = RandomSearch(
    hypermodel,
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
    objective="val_loss",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_val[:100], y_val[:100])
)
