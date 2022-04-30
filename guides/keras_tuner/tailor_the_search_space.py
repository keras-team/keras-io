"""
Title: Tailor the search space
Authors: Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin
Date created: 2019/05/31
Last modified: 2021/10/27
Description: Tune a subset of the hyperparameters without changing the hypermodel.
"""

"""shell
pip install keras-tuner -q
"""

"""
In this guide, we will show how to tailor the search space without changing the
`HyperModel` code directly.  For example, you can only tune some of the
hyperparameters and keep the rest fixed, or you can override the compile
arguments, like `optimizer`, `loss`, and `metrics`.

## The default value of a hyperparameter

Before we tailor the search space, it is important to know that every
hyperparameter has a default value.  This default value is used as the
hyperparameter value when not tuning it during our tailoring the search space.

Whenever you register a hyperparameter, you can use the `default` argument to
specify a default value:

```python
hp.Int("units", min_value=32, max_value=128, step=32, default=64)
```

If you don't, hyperparameters always have a default default (for `Int`, it is
equal to `min_value`).

In the following model-building function, we specified the default value for
the `units` hyperparameter as 64.
"""

from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import numpy as np


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=128, step=32, default=64)
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


"""
We will reuse this search space in the rest of the tutorial by overriding the
hyperparameters without defining a new search space.

## Search a few and fix the rest

If you have an existing hypermodel, and you want to search over only a few
hyperparameters, and keep the rest fixed, you don't have to change the code in
the model-building function or the `HyperModel`.  You can pass a
`HyperParameters` to the `hyperparameters` argument to the tuner constructor
with all the hyperparameters you want to tune.  Specify
`tune_new_entries=False` to prevent it from tuning other hyperparameters, the
default value of which would be used.

In the following example, we only tune the `learning_rate` hyperparameter, and
changed its type and value ranges.
"""

hp = keras_tuner.HyperParameters()

# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    # Prevents unlisted parameters from being tuned
    tune_new_entries=False,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="search_a_few",
)

# Generate random data
x_train = np.random.rand(100, 28, 28, 1)
y_train = np.random.randint(0, 10, (100, 1))
x_val = np.random.rand(20, 28, 28, 1)
y_val = np.random.randint(0, 10, (20, 1))

# Run the search
tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

"""
If you summarize the search space, you will see only one hyperparameter.
"""

tuner.search_space_summary()

"""
## Fix a few and tune the rest

In the example above we showed how to tune only a few hyperparameters and keep
the rest fixed.  You can also do the reverse: only fix a few hyperparameters
and tune all the rest.

In the following example, we fixed the value of the `learning_rate`
hyperparameter.  Pass a `hyperparameters` argument with a `Fixed` entry (or any
number of `Fixed` entries).  Also remember to specify `tune_new_entries=True`,
which allows us to tune the rest of the hyperparameters.
"""

hp = keras_tuner.HyperParameters()
hp.Fixed("learning_rate", value=1e-4)

tuner = keras_tuner.RandomSearch(
    build_model,
    hyperparameters=hp,
    tune_new_entries=True,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="fix_a_few",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

"""
If you summarize the search space, you will see the `learning_rate` is marked
as fixed, and the rest of the hyperparameters are being tuned.
"""

tuner.search_space_summary()

"""
## Overriding compilation arguments

If you have a hypermodel for which you want to change the existing optimizer,
loss, or metrics, you can do so by passing these arguments to the tuner
constructor:
"""

tuner = keras_tuner.RandomSearch(
    build_model,
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["sparse_categorical_crossentropy",],
    objective="val_loss",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="override_compile",
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

"""
If you get the best model, you can see the loss function has changed to MSE.
"""

tuner.get_best_models()[0].loss

"""
## Tailor the search space of pre-build HyperModels

You can also use these techniques with the pre-build models in KerasTuner, like
`HyperResNet` or `HyperXception`.  However, to see what hyperparameters are in
these pre-build `HyperModel`s, you will have to read the source code.

In the following example, we only tune the `learning_rate` of `HyperXception`
and fixed all the rest of the hyperparameters. Because the default loss of
`HyperXception` is `categorical_crossentropy`, which expect the labels to be
one-hot encoded, which doesn't match our raw integer label data, we need to
change it by overriding the `loss` in the compile args to
`sparse_categorical_crossentropy`.
"""

hypermodel = keras_tuner.applications.HyperXception(input_shape=(28, 28, 1), classes=10)

hp = keras_tuner.HyperParameters()

# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

tuner = keras_tuner.RandomSearch(
    hypermodel,
    hyperparameters=hp,
    # Prevents unlisted parameters from being tuned
    tune_new_entries=False,
    # Override the loss.
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# Run the search
tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
tuner.search_space_summary()
