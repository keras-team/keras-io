# Tailor the search space

**Authors:** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin<br>
**Date created:** 2019/05/31<br>
**Last modified:** 2021/10/27<br>
**Description:** Tune a subset of the hyperparameters without changing the hypermodel.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/tailor_the_search_space.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/tailor_the_search_space.py)




```python
!pip install keras-tuner -q
```

In this guide, we will show how to tailor the search space without changing the
`HyperModel` code directly.  For example, you can only tune some of the
hyperparameters and keep the rest fixed, or you can override the compile
arguments, like `optimizer`, `loss`, and `metrics`.

---
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


```python
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

```

<div class="k-default-codeblock">
```
2022-04-28 04:09:21.924650: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-04-28 04:09:21.924705: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

```
</div>
We will reuse this search space in the rest of the tutorial by overriding the
hyperparameters without defining a new search space.

---
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


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 00s]
val_accuracy: 0.0
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.10000000149011612
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit

```
</div>
If you summarize the search space, you will see only one hyperparameter.


```python
tuner.search_space_summary()
```

<div class="k-default-codeblock">
```
Search space summary
Default search space size: 1
learning_rate (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}

```
</div>
---
## Fix a few and tune the rest

In the example above we showed how to tune only a few hyperparameters and keep
the rest fixed.  You can also do the reverse: only fix a few hyperparameters
and tune all the rest.

In the following example, we fixed the value of the `learning_rate`
hyperparameter.  Pass a `hyperparameters` argument with a `Fixed` entry (or any
number of `Fixed` entries).  Also remember to specify `tune_new_entries=True`,
which allows us to tune the rest of the hyperparameters.


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 00s]
val_accuracy: 0.0
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.05000000074505806
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit

```
</div>
If you summarize the search space, you will see the `learning_rate` is marked
as fixed, and the rest of the hyperparameters are being tuned.


```python
tuner.search_space_summary()
```

<div class="k-default-codeblock">
```
Search space summary
Default search space size: 3
learning_rate (Fixed)
{'conditions': [], 'value': 0.0001}
units (Int)
{'default': 64, 'conditions': [], 'min_value': 32, 'max_value': 128, 'step': 32, 'sampling': None}
dropout (Boolean)
{'default': False, 'conditions': []}

```
</div>
---
## Overriding compilation arguments

If you have a hypermodel for which you want to change the existing optimizer,
loss, or metrics, you can do so by passing these arguments to the tuner
constructor:


```python
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
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 00s]
val_loss: 24.548675537109375
```
</div>
    
<div class="k-default-codeblock">
```
Best val_loss So Far: 22.171010971069336
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit

```
</div>
If you get the best model, you can see the loss function has changed to MSE.


```python
tuner.get_best_models()[0].loss
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).layer_with_weights-0.kernel
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).layer_with_weights-0.bias
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'm' for (root).layer_with_weights-0.kernel
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'm' for (root).layer_with_weights-0.bias
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'v' for (root).layer_with_weights-0.kernel
WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'v' for (root).layer_with_weights-0.bias

'mse'

```
</div>
---
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


```python
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
```

<div class="k-default-codeblock">
```
Trial 2 Complete [00h 00m 05s]
val_accuracy: 0.05000000074505806
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.15000000596046448
Total elapsed time: 00h 00m 11s
INFO:tensorflow:Oracle triggered exit
Search space summary
Default search space size: 1
learning_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}

```
</div>