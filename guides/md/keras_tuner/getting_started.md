# Getting started with KerasTuner

**Authors:** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin<br>
**Date created:** 2019/05/31<br>
**Last modified:** 2021/06/07<br>
**Description:** The basics of using KerasTuner to tune model hyperparameters.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/getting_started.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/getting_started.py)



---
## Setup


```python
!pip install keras-tuner -q
```

---
## Introduction

Here's how to perform hyperparameter tuning for a single-layer dense neural
network using random search.
First, we need to prepare the dataset -- let's use MNIST dataset as an example.


```python
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
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step

```
</div>
---
## Prepare a model-building function

Then, we define a model-building function. It takes an argument `hp` from
which you can sample hyperparameters, such as
`hp.Int('units', min_value=32, max_value=512, step=32)`
(an integer from a certain range).

This function returns a compiled model.


```python
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

```

---
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


```python
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
```

You can print a summary of the search space:


```python
tuner.search_space_summary()
```

<div class="k-default-codeblock">
```
Search space summary
Default search space size: 2
units (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': None}
learning_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}

```
</div>
Then, start the search for the best hyperparameter configuration.
The call to `search` has the same signature as `model.fit()`.


```python
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 18s]
val_accuracy: 0.9421000182628632
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.9730499982833862
Total elapsed time: 00h 00m 48s
INFO:tensorflow:Oracle triggered exit

```
</div>
Here's what happens in `search`: models are built iteratively by calling the
model-building function, which populates the hyperparameter space (search
space) tracked by the `hp` object. The tuner progressively explores the space,
recording metrics for each configuration.

---
## Query the results

When search is over, you can retrieve the best model(s):


```python
models = tuner.get_best_models(num_models=2)
```

Or print a summary of the results:


```python
tuner.results_summary()
```

<div class="k-default-codeblock">
```
Results summary
Results in my_dir/helloworld
Showing 10 best trials
Objective(name='val_accuracy', direction='max')
Trial summary
Hyperparameters:
units: 480
learning_rate: 0.001
Score: 0.9730499982833862
Trial summary
Hyperparameters:
units: 160
learning_rate: 0.001
Score: 0.9692499935626984
Trial summary
Hyperparameters:
units: 320
learning_rate: 0.0001
Score: 0.9421000182628632

```
</div>
You will also find detailed logs, checkpoints, etc, in the folder `my_dir/helloworld`, i.e. `directory/project_name`.

---
## The search space may contain conditional hyperparameters

Below, we have a `for` loop creating a tunable number of layers,
which themselves involve a tunable `units` parameter.

This can be pushed to any level of parameter interdependency, including recursion.

Note that all parameter names should be unique (here, in the loop over `i`,
we name the inner parameters `'units_' + str(i)`).


```python

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

```

---
## You can use a HyperModel subclass instead of a model-building function

This makes it easy to share and reuse hypermodels.

A `HyperModel` subclass only needs to implement a `build(self, hp)` method.


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 10s]
val_accuracy: 0.9571999907493591
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.9573000073432922
Total elapsed time: 00h 00m 28s
INFO:tensorflow:Oracle triggered exit

```
</div>
---
## KerasTuner includes pre-made tunable applications: HyperResNet and HyperXception

These are ready-to-use hypermodels for computer vision.

They come pre-compiled with `loss="categorical_crossentropy"` and `metrics=["accuracy"]`.


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 34s]
val_accuracy: 0.10000000149011612
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.10000000149011612
Total elapsed time: 00h 01m 45s
INFO:tensorflow:Oracle triggered exit

```
</div>
---
## You can easily restrict the search space to just a few parameters

If you have an existing hypermodel, and you want to search over only a few parameters
(such as the learning rate), you can do so by passing a `hyperparameters` argument
to the tuner constructor, as well as `tune_new_entries=False` to specify that parameters
that you didn't list in `hyperparameters` should not be tuned. For these parameters, the default
value gets used.


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 02s]
val_accuracy: 0.10999999940395355
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.10999999940395355
Total elapsed time: 00h 00m 08s
INFO:tensorflow:Oracle triggered exit

```
</div>
---
## About parameter default values

Whenever you register a hyperparameter inside a model-building function or the `build` method of a hypermodel,
you can specify a default value:

```python
hp.Int("units", min_value=32, max_value=512, step=32, default=128)
```

If you don't, hyperparameters always have a default default (for `Int`, it is equal to `min_value`).

---
## Fixing values in a hypermodel

What if you want to do the reverse -- tune all available parameters in a hypermodel, **except** one (the learning rate)?

Pass a `hyperparameters` argument with a `Fixed` entry (or any number of `Fixed` entries), and specify `tune_new_entries=True`.


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 07s]
val_accuracy: 0.10000000149011612
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.11999999731779099
Total elapsed time: 00h 00m 17s
INFO:tensorflow:Oracle triggered exit

```
</div>
---
## Overriding compilation arguments

If you have a hypermodel for which you want to change the existing optimizer,
loss, or metrics, you can do so by passing these arguments
to the tuner constructor:


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 07s]
val_loss: 0.09853753447532654
```
</div>
    
<div class="k-default-codeblock">
```
Best val_loss So Far: 0.08992436528205872
Total elapsed time: 00h 00m 23s
INFO:tensorflow:Oracle triggered exit

```
</div>
