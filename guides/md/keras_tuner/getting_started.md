# Getting started with KerasTuner

**Authors:** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin<br>
**Date created:** 2019/05/31<br>
**Last modified:** 2021/10/27<br>
**Description:** The basics of using KerasTuner to tune model hyperparameters.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/getting_started.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/getting_started.py)




```python
!pip install keras-tuner -q
```

---
## Introduction

KerasTuner is a general-purpose hyperparameter tuning library. It has strong
integration with Keras workflows, but it isn't limited to them: you could use
it to tune scikit-learn models, or anything else. In this tutorial, you will
see how to tune model architecture, training process, and data preprocessing
steps with KerasTuner. Let's start from a simple example.

---
## Tune the model architecture

The first thing we need to do is writing a function, which returns a compiled
Keras model. It takes an argument `hp` for defining the hyperparameters while
building the model.

### Define the search space

In the following code example, we define a Keras model with two `Dense` layers.
We want to tune the number of units in the first `Dense` layer. We just define
an integer hyperparameter with `hp.Int('units', min_value=32, max_value=512, step=32)`,
whose range is from 32 to 512 inclusive. When sampling from it, the minimum
step for walking through the interval is 32.


```python
import keras
from keras import layers


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

```

You can quickly test if the model builds successfully.


```python
import keras_tuner

build_model(keras_tuner.HyperParameters())
```




<div class="k-default-codeblock">
```
<Sequential name=sequential, built=False>

```
</div>
There are many other types of hyperparameters as well. We can define multiple
hyperparameters in the function. In the following code, we tune whether to
use a `Dropout` layer with `hp.Boolean()`, tune which activation function to
use with `hp.Choice()`, tune the learning rate of the optimizer with
`hp.Float()`.


```python

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Tune number of units.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # Tune the activation function to use.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```




<div class="k-default-codeblock">
```
<Sequential name=sequential_1, built=False>

```
</div>
As shown below, the hyperparameters are actual values. In fact, they are just
functions returning actual values. For example, `hp.Int()` returns an `int`
value. Therefore, you can put them into variables, for loops, or if
conditions.


```python
hp = keras_tuner.HyperParameters()
print(hp.Int("units", min_value=32, max_value=512, step=32))
```

<div class="k-default-codeblock">
```
32

```
</div>
You can also define the hyperparameters in advance and keep your Keras code in
a separate function.


```python

def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


build_model(keras_tuner.HyperParameters())
```




<div class="k-default-codeblock">
```
<Sequential name=sequential_2, built=False>

```
</div>
Each of the hyperparameters is uniquely identified by its name (the first
argument). To tune the number of units in different `Dense` layers separately
as different hyperparameters, we give them different names as `f"units_{i}"`.

Notably, this is also an example of creating conditional hyperparameters.
There are many hyperparameters specifying the number of units in the `Dense`
layers. The number of such hyperparameters is decided by the number of layers,
which is also a hyperparameter. Therefore, the total number of hyperparameters
used may be different from trial to trial. Some hyperparameter is only used
when a certain condition is satisfied. For example, `units_3` is only used
when `num_layers` is larger than 3. With KerasTuner, you can easily define
such hyperparameters dynamically while creating the model.


```python

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```




<div class="k-default-codeblock">
```
<Sequential name=sequential_3, built=False>

```
</div>
### Start the search

After defining the search space, we need to select a tuner class to run the
search. You may choose from `RandomSearch`, `BayesianOptimization` and
`Hyperband`, which correspond to different tuning algorithms. Here we use
`RandomSearch` as an example.

To initialize the tuner, we need to specify several arguments in the initializer.

* `hypermodel`. The model-building function, which is `build_model` in our case.
* `objective`. The name of the objective to optimize (whether to minimize or
maximize is automatically inferred for built-in metrics). We will introduce how
to use custom metrics later in this tutorial.
* `max_trials`. The total number of trials to run during the search.
* `executions_per_trial`. The number of models that should be built and fit for
each trial. Different trials have different hyperparameter values. The
executions within the same trial have the same hyperparameter values. The
purpose of having multiple executions per trial is to reduce results variance
and therefore be able to more accurately assess the performance of a model. If
you want to get results faster, you could set `executions_per_trial=1` (single
round of training for each model configuration).
* `overwrite`. Control whether to overwrite the previous results in the same
directory or resume the previous search instead. Here we set `overwrite=True`
to start a new search and ignore any previous results.
* `directory`. A path to a directory for storing the search results.
* `project_name`. The name of the sub-directory in the `directory`.


```python
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
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
Default search space size: 5
num_layers (Int)
{'default': None, 'conditions': [], 'min_value': 1, 'max_value': 3, 'step': 1, 'sampling': 'linear'}
units_0 (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': 'linear'}
activation (Choice)
{'default': 'relu', 'conditions': [], 'values': ['relu', 'tanh'], 'ordered': False}
dropout (Boolean)
{'default': False, 'conditions': []}
lr (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}

```
</div>
Before starting the search, let's prepare the MNIST dataset.


```python
import keras
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

Then, start the search for the best hyperparameter configuration.
All the arguments passed to `search` is passed to `model.fit()` in each
execution. Remember to pass `validation_data` to evaluate the model.


```python
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 19s]
val_accuracy: 0.9665500223636627
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.9665500223636627
Total elapsed time: 00h 00m 40s

```
</div>
During the `search`, the model-building function is called with different
hyperparameter values in different trial. In each trial, the tuner would
generate a new set of hyperparameter values to build the model. The model is
then fit and evaluated. The metrics are recorded. The tuner progressively
explores the space and finally finds a good set of hyperparameter values.

### Query the results

When search is over, you can retrieve the best model(s). The model is saved at
its best performing epoch evaluated on the `validation_data`.


```python
# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
```

<div class="k-default-codeblock">
```
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:388: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 18 variables. 
  trackable.load_own_variables(weights_store.get(inner_path))
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:388: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 10 variables. 
  trackable.load_own_variables(weights_store.get(inner_path))

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">784</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">416</span>)                 │    <span style="color: #00af00; text-decoration-color: #00af00">326,560</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)                 │    <span style="color: #00af00; text-decoration-color: #00af00">213,504</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │     <span style="color: #00af00; text-decoration-color: #00af00">16,416</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                  │        <span style="color: #00af00; text-decoration-color: #00af00">330</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">556,810</span> (2.12 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">556,810</span> (2.12 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



You can also print a summary of the search results.


```python
tuner.results_summary()
```

<div class="k-default-codeblock">
```
Results summary
Results in my_dir/helloworld
Showing 10 best trials
Objective(name="val_accuracy", direction="max")
```
</div>
    
<div class="k-default-codeblock">
```
Trial 2 summary
Hyperparameters:
num_layers: 3
units_0: 416
activation: relu
dropout: True
lr: 0.0001324166048504802
units_1: 512
units_2: 32
Score: 0.9665500223636627
```
</div>
    
<div class="k-default-codeblock">
```
Trial 0 summary
Hyperparameters:
num_layers: 1
units_0: 128
activation: tanh
dropout: False
lr: 0.001425162921397599
Score: 0.9623999893665314
```
</div>
    
<div class="k-default-codeblock">
```
Trial 1 summary
Hyperparameters:
num_layers: 2
units_0: 512
activation: tanh
dropout: True
lr: 0.0010584293918512798
units_1: 32
Score: 0.9606499969959259

```
</div>
You will find detailed logs, checkpoints, etc, in the folder
`my_dir/helloworld`, i.e. `directory/project_name`.

You can also visualize the tuning results using TensorBoard and HParams plugin.
For more information, please following
[this link](https://keras.io/guides/keras_tuner/visualize_tuning/).

### Retrain the model

If you want to train the model with the entire dataset, you may retrieve the
best hyperparameters and retrain the model by yourself.


```python
# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
# Fit with the entire dataset.
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)
```

    
    1/1875 [37m━━━━━━━━━━━━━━━━━━━━  17:57 575ms/step - accuracy: 0.1250 - loss: 2.3113

<div class="k-default-codeblock">
```

```
</div>
   29/1875 [37m━━━━━━━━━━━━━━━━━━━━  3s 2ms/step - accuracy: 0.1753 - loss: 2.2296     

<div class="k-default-codeblock">
```

```
</div>
   63/1875 [37m━━━━━━━━━━━━━━━━━━━━  3s 2ms/step - accuracy: 0.2626 - loss: 2.1206

<div class="k-default-codeblock">
```

```
</div>
   96/1875 ━[37m━━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.3252 - loss: 2.0103

<div class="k-default-codeblock">
```

```
</div>
  130/1875 ━[37m━━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.3745 - loss: 1.9041

<div class="k-default-codeblock">
```

```
</div>
  164/1875 ━[37m━━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.4139 - loss: 1.8094

<div class="k-default-codeblock">
```

```
</div>
  199/1875 ━━[37m━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.4470 - loss: 1.7246

<div class="k-default-codeblock">
```

```
</div>
  235/1875 ━━[37m━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.4752 - loss: 1.6493

<div class="k-default-codeblock">
```

```
</div>
  270/1875 ━━[37m━━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.4982 - loss: 1.5857

<div class="k-default-codeblock">
```

```
</div>
  305/1875 ━━━[37m━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.5182 - loss: 1.5293

<div class="k-default-codeblock">
```

```
</div>
  339/1875 ━━━[37m━━━━━━━━━━━━━━━━━  2s 2ms/step - accuracy: 0.5354 - loss: 1.4800

<div class="k-default-codeblock">
```

```
</div>
  374/1875 ━━━[37m━━━━━━━━━━━━━━━━━  2s 1ms/step - accuracy: 0.5513 - loss: 1.4340

<div class="k-default-codeblock">
```

```
</div>
  409/1875 ━━━━[37m━━━━━━━━━━━━━━━━  2s 1ms/step - accuracy: 0.5656 - loss: 1.3924

<div class="k-default-codeblock">
```

```
</div>
  444/1875 ━━━━[37m━━━━━━━━━━━━━━━━  2s 1ms/step - accuracy: 0.5785 - loss: 1.3545

<div class="k-default-codeblock">
```

```
</div>
  478/1875 ━━━━━[37m━━━━━━━━━━━━━━━  2s 1ms/step - accuracy: 0.5899 - loss: 1.3208

<div class="k-default-codeblock">
```

```
</div>
  513/1875 ━━━━━[37m━━━━━━━━━━━━━━━  2s 1ms/step - accuracy: 0.6006 - loss: 1.2887

<div class="k-default-codeblock">
```

```
</div>
  548/1875 ━━━━━[37m━━━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6104 - loss: 1.2592

<div class="k-default-codeblock">
```

```
</div>
  583/1875 ━━━━━━[37m━━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6195 - loss: 1.2318

<div class="k-default-codeblock">
```

```
</div>
  618/1875 ━━━━━━[37m━━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6279 - loss: 1.2063

<div class="k-default-codeblock">
```

```
</div>
  653/1875 ━━━━━━[37m━━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6358 - loss: 1.1823

<div class="k-default-codeblock">
```

```
</div>
  688/1875 ━━━━━━━[37m━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6431 - loss: 1.1598

<div class="k-default-codeblock">
```

```
</div>
  723/1875 ━━━━━━━[37m━━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6500 - loss: 1.1387

<div class="k-default-codeblock">
```

```
</div>
  758/1875 ━━━━━━━━[37m━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6564 - loss: 1.1189

<div class="k-default-codeblock">
```

```
</div>
  793/1875 ━━━━━━━━[37m━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6625 - loss: 1.1002

<div class="k-default-codeblock">
```

```
</div>
  828/1875 ━━━━━━━━[37m━━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6682 - loss: 1.0826

<div class="k-default-codeblock">
```

```
</div>
  863/1875 ━━━━━━━━━[37m━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6736 - loss: 1.0658

<div class="k-default-codeblock">
```

```
</div>
  899/1875 ━━━━━━━━━[37m━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6788 - loss: 1.0495

<div class="k-default-codeblock">
```

```
</div>
  935/1875 ━━━━━━━━━[37m━━━━━━━━━━━  1s 1ms/step - accuracy: 0.6838 - loss: 1.0339

<div class="k-default-codeblock">
```

```
</div>
  970/1875 ━━━━━━━━━━[37m━━━━━━━━━━  1s 1ms/step - accuracy: 0.6885 - loss: 1.0195

<div class="k-default-codeblock">
```

```
</div>
 1005/1875 ━━━━━━━━━━[37m━━━━━━━━━━  1s 1ms/step - accuracy: 0.6929 - loss: 1.0058

<div class="k-default-codeblock">
```

```
</div>
 1041/1875 ━━━━━━━━━━━[37m━━━━━━━━━  1s 1ms/step - accuracy: 0.6972 - loss: 0.9923

<div class="k-default-codeblock">
```

```
</div>
 1076/1875 ━━━━━━━━━━━[37m━━━━━━━━━  1s 1ms/step - accuracy: 0.7012 - loss: 0.9798

<div class="k-default-codeblock">
```

```
</div>
 1111/1875 ━━━━━━━━━━━[37m━━━━━━━━━  1s 1ms/step - accuracy: 0.7051 - loss: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 1146/1875 ━━━━━━━━━━━━[37m━━━━━━━━  1s 1ms/step - accuracy: 0.7088 - loss: 0.9561

<div class="k-default-codeblock">
```

```
</div>
 1182/1875 ━━━━━━━━━━━━[37m━━━━━━━━  1s 1ms/step - accuracy: 0.7124 - loss: 0.9446

<div class="k-default-codeblock">
```

```
</div>
 1218/1875 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.7159 - loss: 0.9336

<div class="k-default-codeblock">
```

```
</div>
 1254/1875 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.7193 - loss: 0.9230

<div class="k-default-codeblock">
```

```
</div>
 1289/1875 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.7225 - loss: 0.9131

<div class="k-default-codeblock">
```

```
</div>
 1324/1875 ━━━━━━━━━━━━━━[37m━━━━━━  0s 1ms/step - accuracy: 0.7255 - loss: 0.9035

<div class="k-default-codeblock">
```

```
</div>
 1359/1875 ━━━━━━━━━━━━━━[37m━━━━━━  0s 1ms/step - accuracy: 0.7284 - loss: 0.8943

<div class="k-default-codeblock">
```

```
</div>
 1394/1875 ━━━━━━━━━━━━━━[37m━━━━━━  0s 1ms/step - accuracy: 0.7313 - loss: 0.8853

<div class="k-default-codeblock">
```

```
</div>
 1429/1875 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.7341 - loss: 0.8767

<div class="k-default-codeblock">
```

```
</div>
 1465/1875 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.7368 - loss: 0.8680

<div class="k-default-codeblock">
```

```
</div>
 1500/1875 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.7394 - loss: 0.8599

<div class="k-default-codeblock">
```

```
</div>
 1535/1875 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.7419 - loss: 0.8520

<div class="k-default-codeblock">
```

```
</div>
 1570/1875 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.7443 - loss: 0.8444

<div class="k-default-codeblock">
```

```
</div>
 1605/1875 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.7467 - loss: 0.8370

<div class="k-default-codeblock">
```

```
</div>
 1639/1875 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.7489 - loss: 0.8299

<div class="k-default-codeblock">
```

```
</div>
 1674/1875 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.7511 - loss: 0.8229

<div class="k-default-codeblock">
```

```
</div>
 1707/1875 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.7532 - loss: 0.8164

<div class="k-default-codeblock">
```

```
</div>
 1741/1875 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.7552 - loss: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1774/1875 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.7572 - loss: 0.8038

<div class="k-default-codeblock">
```

```
</div>
 1809/1875 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.7592 - loss: 0.7975

<div class="k-default-codeblock">
```

```
</div>
 1843/1875 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.7611 - loss: 0.7915

<div class="k-default-codeblock">
```

```
</div>
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.7629 - loss: 0.7858





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f31883d9e10>

```
</div>
---
## Tune model training

To tune the model building process, we need to subclass the `HyperModel` class,
which also makes it easy to share and reuse hypermodels.

We need to override `HyperModel.build()` and `HyperModel.fit()` to tune the
model building and training process respectively. A `HyperModel.build()`
method is the same as the model-building function, which creates a Keras model
using the hyperparameters and returns it.

In `HyperModel.fit()`, you can access the model returned by
`HyperModel.build()`,`hp` and all the arguments passed to `search()`. You need
to train the model and return the training history.

In the following code, we will tune the `shuffle` argument in `model.fit()`.

It is generally not needed to tune the number of epochs because a built-in
callback is passed to `model.fit()` to save the model at its best epoch
evaluated by the `validation_data`.

> **Note**: The `**kwargs` should always be passed to `model.fit()` because it
contains the callbacks for model saving and tensorboard plugins.


```python

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
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
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )

```

Again, we can do a quick check to see if the code works correctly.


```python
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

    
 1/4 ━━━━━[37m━━━━━━━━━━━━━━━  0s 279ms/step - accuracy: 0.0000e+00 - loss: 12.2230

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step - accuracy: 0.0679 - loss: 11.9568    

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 109ms/step - accuracy: 0.0763 - loss: 11.8941





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f318865c100>

```
</div>
---
## Tune data preprocessing

To tune data preprocessing, we just add an additional step in
`HyperModel.fit()`, where we can access the dataset from the arguments. In the
following code, we tune whether to normalize the data before training the
model. This time we explicitly put `x` and `y` in the function signature
because we need to use them.


```python

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
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
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        return model.fit(
            x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

    
 1/4 ━━━━━[37m━━━━━━━━━━━━━━━  0s 276ms/step - accuracy: 0.1250 - loss: 12.0090

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.0994 - loss: 12.1242 

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 95ms/step - accuracy: 0.0955 - loss: 12.1594





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f31ba836200>

```
</div>
If a hyperparameter is used both in `build()` and `fit()`, you can define it in
`build()` and use `hp.get(hp_name)` to retrieve it in `fit()`. We use the
image size as an example. It is both used as the input shape in `build()`, and
used by data prerprocessing step to crop the images in `fit()`.


```python

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        image_size = hp.Int("image_size", 10, 28)
        inputs = keras.Input(shape=(image_size, image_size))
        outputs = layers.Flatten()(inputs)
        outputs = layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )(outputs)
        outputs = layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, validation_data=None, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        image_size = hp.get("image_size")
        cropped_x = x[:, :image_size, :image_size, :]
        if validation_data:
            x_val, y_val = validation_data
            cropped_x_val = x_val[:, :image_size, :image_size, :]
            validation_data = (cropped_x_val, y_val)
        return model.fit(
            cropped_x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            validation_data=validation_data,
            **kwargs,
        )


tuner = keras_tuner.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 04s]
val_accuracy: 0.9567000269889832
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.9685999751091003
Total elapsed time: 00h 00m 13s

```
</div>
### Retrain the model

Using `HyperModel` also allows you to retrain the best model by yourself.


```python
hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
hypermodel.fit(best_hp, model, x_all, y_all, epochs=1)
```

    
    1/1875 [37m━━━━━━━━━━━━━━━━━━━━  9:00 289ms/step - accuracy: 0.0000e+00 - loss: 2.4352

<div class="k-default-codeblock">
```

```
</div>
   52/1875 [37m━━━━━━━━━━━━━━━━━━━━  1s 996us/step - accuracy: 0.6035 - loss: 1.3521      

<div class="k-default-codeblock">
```

```
</div>
  110/1875 ━[37m━━━━━━━━━━━━━━━━━━━  1s 925us/step - accuracy: 0.7037 - loss: 1.0231

<div class="k-default-codeblock">
```

```
</div>
  171/1875 ━[37m━━━━━━━━━━━━━━━━━━━  1s 890us/step - accuracy: 0.7522 - loss: 0.8572

<div class="k-default-codeblock">
```

```
</div>
  231/1875 ━━[37m━━━━━━━━━━━━━━━━━━  1s 877us/step - accuracy: 0.7804 - loss: 0.7590

<div class="k-default-codeblock">
```

```
</div>
  291/1875 ━━━[37m━━━━━━━━━━━━━━━━━  1s 870us/step - accuracy: 0.7993 - loss: 0.6932

<div class="k-default-codeblock">
```

```
</div>
  350/1875 ━━━[37m━━━━━━━━━━━━━━━━━  1s 867us/step - accuracy: 0.8127 - loss: 0.6467

<div class="k-default-codeblock">
```

```
</div>
  413/1875 ━━━━[37m━━━━━━━━━━━━━━━━  1s 856us/step - accuracy: 0.8238 - loss: 0.6079

<div class="k-default-codeblock">
```

```
</div>
  476/1875 ━━━━━[37m━━━━━━━━━━━━━━━  1s 848us/step - accuracy: 0.8326 - loss: 0.5774

<div class="k-default-codeblock">
```

```
</div>
  535/1875 ━━━━━[37m━━━━━━━━━━━━━━━  1s 849us/step - accuracy: 0.8394 - loss: 0.5536

<div class="k-default-codeblock">
```

```
</div>
  600/1875 ━━━━━━[37m━━━━━━━━━━━━━━  1s 841us/step - accuracy: 0.8458 - loss: 0.5309

<div class="k-default-codeblock">
```

```
</div>
  661/1875 ━━━━━━━[37m━━━━━━━━━━━━━  1s 840us/step - accuracy: 0.8511 - loss: 0.5123

<div class="k-default-codeblock">
```

```
</div>
  723/1875 ━━━━━━━[37m━━━━━━━━━━━━━  0s 837us/step - accuracy: 0.8559 - loss: 0.4955

<div class="k-default-codeblock">
```

```
</div>
  783/1875 ━━━━━━━━[37m━━━━━━━━━━━━  0s 838us/step - accuracy: 0.8600 - loss: 0.4811

<div class="k-default-codeblock">
```

```
</div>
  847/1875 ━━━━━━━━━[37m━━━━━━━━━━━  0s 834us/step - accuracy: 0.8640 - loss: 0.4671

<div class="k-default-codeblock">
```

```
</div>
  912/1875 ━━━━━━━━━[37m━━━━━━━━━━━  0s 830us/step - accuracy: 0.8677 - loss: 0.4544

<div class="k-default-codeblock">
```

```
</div>
  976/1875 ━━━━━━━━━━[37m━━━━━━━━━━  0s 827us/step - accuracy: 0.8709 - loss: 0.4429

<div class="k-default-codeblock">
```

```
</div>
 1040/1875 ━━━━━━━━━━━[37m━━━━━━━━━  0s 825us/step - accuracy: 0.8738 - loss: 0.4325

<div class="k-default-codeblock">
```

```
</div>
 1104/1875 ━━━━━━━━━━━[37m━━━━━━━━━  0s 822us/step - accuracy: 0.8766 - loss: 0.4229

<div class="k-default-codeblock">
```

```
</div>
 1168/1875 ━━━━━━━━━━━━[37m━━━━━━━━  0s 821us/step - accuracy: 0.8791 - loss: 0.4140

<div class="k-default-codeblock">
```

```
</div>
 1233/1875 ━━━━━━━━━━━━━[37m━━━━━━━  0s 818us/step - accuracy: 0.8815 - loss: 0.4056

<div class="k-default-codeblock">
```

```
</div>
 1296/1875 ━━━━━━━━━━━━━[37m━━━━━━━  0s 817us/step - accuracy: 0.8837 - loss: 0.3980

<div class="k-default-codeblock">
```

```
</div>
 1361/1875 ━━━━━━━━━━━━━━[37m━━━━━━  0s 815us/step - accuracy: 0.8858 - loss: 0.3907

<div class="k-default-codeblock">
```

```
</div>
 1424/1875 ━━━━━━━━━━━━━━━[37m━━━━━  0s 814us/step - accuracy: 0.8877 - loss: 0.3840

<div class="k-default-codeblock">
```

```
</div>
 1488/1875 ━━━━━━━━━━━━━━━[37m━━━━━  0s 813us/step - accuracy: 0.8895 - loss: 0.3776

<div class="k-default-codeblock">
```

```
</div>
 1550/1875 ━━━━━━━━━━━━━━━━[37m━━━━  0s 813us/step - accuracy: 0.8912 - loss: 0.3718

<div class="k-default-codeblock">
```

```
</div>
 1613/1875 ━━━━━━━━━━━━━━━━━[37m━━━  0s 813us/step - accuracy: 0.8928 - loss: 0.3662

<div class="k-default-codeblock">
```

```
</div>
 1678/1875 ━━━━━━━━━━━━━━━━━[37m━━━  0s 811us/step - accuracy: 0.8944 - loss: 0.3607

<div class="k-default-codeblock">
```

```
</div>
 1744/1875 ━━━━━━━━━━━━━━━━━━[37m━━  0s 809us/step - accuracy: 0.8959 - loss: 0.3555

<div class="k-default-codeblock">
```

```
</div>
 1810/1875 ━━━━━━━━━━━━━━━━━━━[37m━  0s 808us/step - accuracy: 0.8973 - loss: 0.3504

<div class="k-default-codeblock">
```

```
</div>
 1874/1875 ━━━━━━━━━━━━━━━━━━━[37m━  0s 807us/step - accuracy: 0.8987 - loss: 0.3457

<div class="k-default-codeblock">
```

```
</div>
 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 808us/step - accuracy: 0.8987 - loss: 0.3456





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f31884b3070>

```
</div>
---
## Specify the tuning objective

In all previous examples, we all just used validation accuracy
(`"val_accuracy"`) as the tuning objective to select the best model. Actually,
you can use any metric as the objective. The most commonly used metric is
`"val_loss"`, which is the validation loss.

### Built-in metric as the objective

There are many other built-in metrics in Keras you can use as the objective.
Here is [a list of the built-in metrics](https://keras.io/api/metrics/).

To use a built-in metric as the objective, you need to follow these steps:

* Compile the model with the the built-in metric. For example, you want to use
`MeanAbsoluteError()`. You need to compile the model with
`metrics=[MeanAbsoluteError()]`. You may also use its name string instead:
`metrics=["mean_absolute_error"]`. The name string of the metric is always
the snake case of the class name.

* Identify the objective name string. The name string of the objective is
always in the format of `f"val_{metric_name_string}"`. For example, the
objective name string of mean squared error evaluated on the validation data
should be `"val_mean_absolute_error"`.

* Wrap it into `keras_tuner.Objective`. We usually need to wrap the objective
into a `keras_tuner.Objective` object to specify the direction to optimize the
objective. For example, we want to minimize the mean squared error, we can use
`keras_tuner.Objective("val_mean_absolute_error", "min")`. The direction should
be either `"min"` or `"max"`.

* Pass the wrapped objective to the tuner.

You can see the following barebone code example.


```python

def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # Objective is one of the metrics.
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # The objective name and direction.
    # Name is the f"val_{snake_case_metric_class_name}".
    objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 01s]
val_mean_absolute_error: 0.39589792490005493
```
</div>
    
<div class="k-default-codeblock">
```
Best val_mean_absolute_error So Far: 0.34321871399879456
Total elapsed time: 00h 00m 03s
Results summary
Results in my_dir/built_in_metrics
Showing 10 best trials
Objective(name="val_mean_absolute_error", direction="min")
```
</div>
    
<div class="k-default-codeblock">
```
Trial 1 summary
Hyperparameters:
units: 32
Score: 0.34321871399879456
```
</div>
    
<div class="k-default-codeblock">
```
Trial 2 summary
Hyperparameters:
units: 128
Score: 0.39589792490005493
```
</div>
    
<div class="k-default-codeblock">
```
Trial 0 summary
Hyperparameters:
units: 96
Score: 0.5005304217338562

```
</div>
### Custom metric as the objective

You may implement your own metric and use it as the hyperparameter search
objective. Here, we use mean squared error (MSE) as an example. First, we
implement the MSE metric by subclassing `keras.metrics.Metric`. Remember to
give a name to your metric using the `name` argument of `super().__init__()`,
which will be used later. Note: MSE is actually a build-in metric, which can be
imported with `keras.metrics.MeanSquaredError`. This is just an example to show
how to use a custom metric as the hyperparameter search objective.

For more information about implementing custom metrics, please see [this
tutorial](https://keras.io/api/metrics/#creating-custom-metrics). If you would
like a metric with a different function signature than `update_state(y_true,
y_pred, sample_weight)`, you can override the `train_step()` method of your
model following [this
tutorial](https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel).


```python
from keras import ops


class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype="int32", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.square(y_true - y_pred)
        count = ops.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            values *= sample_weight
            count *= sample_weight
        self.sum.assign_add(ops.sum(values))
        self.count.assign_add(count)

    def result(self):
        return self.sum / ops.cast(self.count, "float32")

    def reset_state(self):
        self.sum.assign(0)
        self.count.assign(0)

```

Run the search with the custom objective.


```python

def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # Put custom metric into the metrics.
        metrics=[CustomMetric()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # Specify the name and direction of the objective.
    objective=keras_tuner.Objective("val_custom_metric", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 01s]
val_custom_metric: 0.2830956280231476
```
</div>
    
<div class="k-default-codeblock">
```
Best val_custom_metric So Far: 0.2529197633266449
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_metrics
Showing 10 best trials
Objective(name="val_custom_metric", direction="min")
```
</div>
    
<div class="k-default-codeblock">
```
Trial 0 summary
Hyperparameters:
units: 32
Score: 0.2529197633266449
```
</div>
    
<div class="k-default-codeblock">
```
Trial 2 summary
Hyperparameters:
units: 128
Score: 0.2830956280231476
```
</div>
    
<div class="k-default-codeblock">
```
Trial 1 summary
Hyperparameters:
units: 96
Score: 0.4656866192817688

```
</div>
If your custom objective is hard to put into a custom metric, you can also
evaluate the model by yourself in `HyperModel.fit()` and return the objective
value. The objective value would be minimized by default. In this case, you
don't need to specify the `objective` when initializing the tuner. However, in
this case, the metric value will not be tracked in the Keras logs by only
KerasTuner logs. Therefore, these values would not be displayed by any
TensorBoard view using the Keras metrics.


```python

class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # Return a single float to minimize.
        return np.mean(np.abs(y_pred - y_val))


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # No objective to specify.
    # Objective is the return value of `HyperModel.fit()`.
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 01s]
default_objective: 0.6571611521766413
```
</div>
    
<div class="k-default-codeblock">
```
Best default_objective So Far: 0.40719249752993525
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_eval
Showing 10 best trials
Objective(name="default_objective", direction="min")
```
</div>
    
<div class="k-default-codeblock">
```
Trial 1 summary
Hyperparameters:
units: 128
Score: 0.40719249752993525
```
</div>
    
<div class="k-default-codeblock">
```
Trial 0 summary
Hyperparameters:
units: 96
Score: 0.4992297225533352
```
</div>
    
<div class="k-default-codeblock">
```
Trial 2 summary
Hyperparameters:
units: 32
Score: 0.6571611521766413

```
</div>
If you have multiple metrics to track in KerasTuner, but only use one of them
as the objective, you can return a dictionary, whose keys are the metric names
and the values are the metrics values, for example, return `{"metric_a": 1.0,
"metric_b", 2.0}`. Use one of the keys as the objective name, for example,
`keras_tuner.Objective("metric_a", "min")`.


```python

class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # Return a dictionary of metrics for KerasTuner to track.
        return {
            "metric_a": -np.mean(np.abs(y_pred - y_val)),
            "metric_b": np.mean(np.square(y_pred - y_val)),
        }


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # Objective is one of the keys.
    # Maximize the negative MAE, equivalent to minimize MAE.
    objective=keras_tuner.Objective("metric_a", "max"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval_dict",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 01s]
metric_a: -0.39470441501524833
```
</div>
    
<div class="k-default-codeblock">
```
Best metric_a So Far: -0.3836997988261662
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_eval_dict
Showing 10 best trials
Objective(name="metric_a", direction="max")
```
</div>
    
<div class="k-default-codeblock">
```
Trial 1 summary
Hyperparameters:
units: 64
Score: -0.3836997988261662
```
</div>
    
<div class="k-default-codeblock">
```
Trial 2 summary
Hyperparameters:
units: 32
Score: -0.39470441501524833
```
</div>
    
<div class="k-default-codeblock">
```
Trial 0 summary
Hyperparameters:
units: 96
Score: -0.46081380465766364

```
</div>
---
## Tune end-to-end workflows

In some cases, it is hard to align your code into build and fit functions. You
can also keep your end-to-end workflow in one place by overriding
`Tuner.run_trial()`, which gives you full control of a trial. You can see it
as a black-box optimizer for anything.

### Tune any function

For example, you can find a value of `x`, which minimizes `f(x)=x*x+1`. In the
following code, we just define `x` as a hyperparameter, and return `f(x)` as
the objective value. The `hypermodel` and `objective` argument for initializing
the tuner can be omitted.


```python

class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        x = hp.Float("x", min_value=-1.0, max_value=1.0)
        # Return the objective value to minimize.
        return x * x + 1


tuner = MyTuner(
    # No hypermodel or objective specified.
    max_trials=20,
    overwrite=True,
    directory="my_dir",
    project_name="tune_anything",
)

# No need to pass anything to search()
# unless you use them in run_trial().
tuner.search()
print(tuner.get_best_hyperparameters()[0].get("x"))
```

<div class="k-default-codeblock">
```
Trial 20 Complete [00h 00m 00s]
default_objective: 1.6547719581194267
```
</div>
    
<div class="k-default-codeblock">
```
Best default_objective So Far: 1.0013236767905302
Total elapsed time: 00h 00m 00s
0.03638236922645777

```
</div>
### Keep Keras code separate

You can keep all your Keras code unchanged and use KerasTuner to tune it. It
is useful if you cannot modify the Keras code for some reason.

It also gives you more flexibility. You don't have to separate the model
building and training code apart. However, this workflow would not help you
save the model or connect with the TensorBoard plugins.

To save the model, you can use `trial.trial_id`, which is a string to uniquely
identify a trial, to construct different paths to save the models from
different trials.


```python
import os


def keras_code(units, optimizer, saving_path):
    # Build model
    model = keras.Sequential(
        [
            layers.Dense(units=units, activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    # Prepare data
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    x_val = np.random.rand(20, 10)
    y_val = np.random.rand(20, 1)

    # Train & eval model
    model.fit(x_train, y_train)

    # Save model
    model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    y_pred = model.predict(x_val)
    return np.mean(np.abs(y_pred - y_val))


class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(
            units=hp.Int("units", 32, 128, 32),
            optimizer=hp.Choice("optimizer", ["adam", "adadelta"]),
            saving_path=os.path.join("/tmp", f"{trial.trial_id}.keras"),
        )


tuner = MyTuner(
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="keep_code_separate",
)
tuner.search()
# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
keras_code(**best_hp.values, saving_path="/tmp/best_model.keras")
```

<div class="k-default-codeblock">
```
Trial 3 Complete [00h 00m 00s]
default_objective: 0.18014027375230962
```
</div>
    
<div class="k-default-codeblock">
```
Best default_objective So Far: 0.18014027375230962
Total elapsed time: 00h 00m 03s

```
</div>
    
 1/4 ━━━━━[37m━━━━━━━━━━━━━━━  0s 172ms/step - loss: 0.5030

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 0.5288 

<div class="k-default-codeblock">
```

```
</div>
 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.5367


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step





<div class="k-default-codeblock">
```
0.5918120126201316

```
</div>
---
## KerasTuner includes pre-made tunable applications: HyperResNet and HyperXception

These are ready-to-use hypermodels for computer vision.

They come pre-compiled with `loss="categorical_crossentropy"` and
`metrics=["accuracy"]`.


```python
from keras_tuner.applications import HyperResNet

hypermodel = HyperResNet(input_shape=(28, 28, 1), classes=10)

tuner = keras_tuner.RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_hypermodel",
)
```
