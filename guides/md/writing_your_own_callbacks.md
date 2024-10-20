# Writing your own callbacks

**Authors:** Rick Chao, Francois Chollet<br>
**Date created:** 2019/03/20<br>
**Last modified:** 2023/06/25<br>
**Description:** Complete guide to writing new Keras callbacks.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_your_own_callbacks.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/writing_your_own_callbacks.py)



---
## Introduction

A callback is a powerful tool to customize the behavior of a Keras model during
training, evaluation, or inference. Examples include `keras.callbacks.TensorBoard`
to visualize training progress and results with TensorBoard, or
`keras.callbacks.ModelCheckpoint` to periodically save your model during training.

In this guide, you will learn what a Keras callback is, what it can do, and how you can
build your own. We provide a few demos of simple callback applications to get you
started.

---
## Setup


```python
import numpy as np
import keras
```

---
## Keras callbacks overview

All callbacks subclass the `keras.callbacks.Callback` class, and
override a set of methods called at various stages of training, testing, and
predicting. Callbacks are useful to get a view on internal states and statistics of
the model during training.

You can pass a list of callbacks (as the keyword argument `callbacks`) to the following
model methods:

- `keras.Model.fit()`
- `keras.Model.evaluate()`
- `keras.Model.predict()`

---
## An overview of callback methods

### Global methods

#### `on_(train|test|predict)_begin(self, logs=None)`

Called at the beginning of `fit`/`evaluate`/`predict`.

#### `on_(train|test|predict)_end(self, logs=None)`

Called at the end of `fit`/`evaluate`/`predict`.

### Batch-level methods for training/testing/predicting

#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)`

Called right before processing a batch during training/testing/predicting.

#### `on_(train|test|predict)_batch_end(self, batch, logs=None)`

Called at the end of training/testing/predicting a batch. Within this method, `logs` is
a dict containing the metrics results.

### Epoch-level methods (training only)

#### `on_epoch_begin(self, epoch, logs=None)`

Called at the beginning of an epoch during training.

#### `on_epoch_end(self, epoch, logs=None)`

Called at the end of an epoch during training.

---
## A basic example

Let's take a look at a concrete example. To get started, let's import tensorflow and
define a simple Sequential Keras model:


```python

# Define the Keras model to add callbacks to
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

```

Then, load the MNIST data for training and testing from Keras datasets API:


```python
# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Limit the data to 1000 samples
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
```

Now, define a simple custom callback that logs:

- When `fit`/`evaluate`/`predict` starts & ends
- When each epoch starts & ends
- When each training batch starts & ends
- When each evaluation (test) batch starts & ends
- When each inference (prediction) batch starts & ends


```python

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

```

Let's try it out:


```python
model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=1,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()],
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
)

res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])
```

<div class="k-default-codeblock">
```
Starting training; got log keys: []
Start epoch 0 of training; got log keys: []
...Training: start of batch 0; got log keys: []
...Training: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 1; got log keys: []
...Training: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 2; got log keys: []
...Training: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 3; got log keys: []
...Training: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
End epoch 0 of training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Stop training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 4; got log keys: []
...Evaluating: end of batch 4; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 5; got log keys: []
...Evaluating: end of batch 5; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 6; got log keys: []
...Evaluating: end of batch 6; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 7; got log keys: []
...Evaluating: end of batch 7; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
Start predicting; got log keys: []
...Predicting: start of batch 0; got log keys: []
...Predicting: end of batch 0; got log keys: ['outputs']
 1/8 ━━[37m━━━━━━━━━━━━━━━━━━  0s 13ms/step...Predicting: start of batch 1; got log keys: []
...Predicting: end of batch 1; got log keys: ['outputs']
...Predicting: start of batch 2; got log keys: []
...Predicting: end of batch 2; got log keys: ['outputs']
...Predicting: start of batch 3; got log keys: []
...Predicting: end of batch 3; got log keys: ['outputs']
...Predicting: start of batch 4; got log keys: []
...Predicting: end of batch 4; got log keys: ['outputs']
...Predicting: start of batch 5; got log keys: []
...Predicting: end of batch 5; got log keys: ['outputs']
...Predicting: start of batch 6; got log keys: []
...Predicting: end of batch 6; got log keys: ['outputs']
...Predicting: start of batch 7; got log keys: []
...Predicting: end of batch 7; got log keys: ['outputs']
Stop predicting; got log keys: []
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

```
</div>
### Usage of `logs` dict

The `logs` dict contains the loss value, and all the metrics at the end of a batch or
epoch. Example includes the loss and mean absolute error.


```python

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)

res = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)
```

<div class="k-default-codeblock">
```
Up to batch 0, the average loss is   29.25.
Up to batch 1, the average loss is  485.36.
Up to batch 2, the average loss is  330.94.
Up to batch 3, the average loss is  250.62.
Up to batch 4, the average loss is  202.20.
Up to batch 5, the average loss is  169.51.
Up to batch 6, the average loss is  145.98.
Up to batch 7, the average loss is  128.48.
The average loss for epoch 0 is  128.48 and mean absolute error is    6.01.
Up to batch 0, the average loss is    5.10.
Up to batch 1, the average loss is    4.80.
Up to batch 2, the average loss is    4.96.
Up to batch 3, the average loss is    4.96.
Up to batch 4, the average loss is    4.82.
Up to batch 5, the average loss is    4.69.
Up to batch 6, the average loss is    4.51.
Up to batch 7, the average loss is    4.53.
The average loss for epoch 1 is    4.53 and mean absolute error is    1.72.
Up to batch 0, the average loss is    5.08.
Up to batch 1, the average loss is    4.66.
Up to batch 2, the average loss is    4.64.
Up to batch 3, the average loss is    4.72.
Up to batch 4, the average loss is    4.82.
Up to batch 5, the average loss is    4.83.
Up to batch 6, the average loss is    4.77.
Up to batch 7, the average loss is    4.72.

```
</div>
---
## Usage of `self.model` attribute

In addition to receiving log information when one of their methods is called,
callbacks have access to the model associated with the current round of
training/evaluation/inference: `self.model`.

Here are a few of the things you can do with `self.model` in a callback:

- Set `self.model.stop_training = True` to immediately interrupt training.
- Mutate hyperparameters of the optimizer (available as `self.model.optimizer`),
such as `self.model.optimizer.learning_rate`.
- Save the model at period intervals.
- Record the output of `model.predict()` on a few test samples at the end of each
epoch, to use as a sanity check during training.
- Extract visualizations of intermediate features at the end of each epoch, to monitor
what the model is learning over time.
- etc.

Let's see this in action in a couple of examples.

---
## Examples of Keras callback applications

### Early stopping at minimum loss

This first example shows the creation of a `Callback` that stops training when the
minimum of loss has been reached, by setting the attribute `self.model.stop_training`
(boolean). Optionally, you can provide an argument `patience` to specify how many
epochs we should wait before stopping after having reached a local minimum.

`keras.callbacks.EarlyStopping` provides a more complete and general implementation.


```python

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()],
)
```

<div class="k-default-codeblock">
```
Up to batch 0, the average loss is   25.57.
Up to batch 1, the average loss is  471.66.
Up to batch 2, the average loss is  322.55.
Up to batch 3, the average loss is  243.88.
Up to batch 4, the average loss is  196.53.
Up to batch 5, the average loss is  165.02.
Up to batch 6, the average loss is  142.34.
Up to batch 7, the average loss is  125.17.
Up to batch 8, the average loss is  111.83.
Up to batch 9, the average loss is  101.35.
Up to batch 10, the average loss is   92.60.
Up to batch 11, the average loss is   85.16.
Up to batch 12, the average loss is   79.02.
Up to batch 13, the average loss is   73.71.
Up to batch 14, the average loss is   69.23.
Up to batch 15, the average loss is   65.26.
The average loss for epoch 0 is   65.26 and mean absolute error is    3.89.
Up to batch 0, the average loss is    3.92.
Up to batch 1, the average loss is    4.34.
Up to batch 2, the average loss is    5.39.
Up to batch 3, the average loss is    6.58.
Up to batch 4, the average loss is   10.55.
Up to batch 5, the average loss is   19.29.
Up to batch 6, the average loss is   31.58.
Up to batch 7, the average loss is   38.20.
Up to batch 8, the average loss is   41.96.
Up to batch 9, the average loss is   41.30.
Up to batch 10, the average loss is   39.31.
Up to batch 11, the average loss is   37.09.
Up to batch 12, the average loss is   35.08.
Up to batch 13, the average loss is   33.27.
Up to batch 14, the average loss is   31.54.
Up to batch 15, the average loss is   30.00.
The average loss for epoch 1 is   30.00 and mean absolute error is    4.23.
Up to batch 0, the average loss is    5.70.
Up to batch 1, the average loss is    6.90.
Up to batch 2, the average loss is    7.74.
Up to batch 3, the average loss is    8.85.
Up to batch 4, the average loss is   12.53.
Up to batch 5, the average loss is   21.55.
Up to batch 6, the average loss is   35.70.
Up to batch 7, the average loss is   44.16.
Up to batch 8, the average loss is   44.82.
Up to batch 9, the average loss is   43.07.
Up to batch 10, the average loss is   40.51.
Up to batch 11, the average loss is   38.44.
Up to batch 12, the average loss is   36.69.
Up to batch 13, the average loss is   34.77.
Up to batch 14, the average loss is   32.97.
Up to batch 15, the average loss is   31.32.
The average loss for epoch 2 is   31.32 and mean absolute error is    4.39.
Restoring model weights from the end of the best epoch.
Epoch 3: early stopping

<keras.src.callbacks.history.History at 0x1187b7430>

```
</div>
### Learning rate scheduling

In this example, we show how a custom Callback can be used to dynamically change the
learning rate of the optimizer during the course of training.

See `callbacks.LearningRateScheduler` for a more general implementations.


```python

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        self.model.optimizer.learning_rate = scheduled_lr
        print(f"\nEpoch {epoch}: Learning rate is {float(np.array(scheduled_lr))}.")


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=15,
    verbose=0,
    callbacks=[
        LossAndErrorPrintingCallback(),
        CustomLearningRateScheduler(lr_schedule),
    ],
)
```

    
<div class="k-default-codeblock">
```
Epoch 0: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   27.90.
Up to batch 1, the average loss is  439.49.
Up to batch 2, the average loss is  302.08.
Up to batch 3, the average loss is  228.83.
Up to batch 4, the average loss is  184.97.
Up to batch 5, the average loss is  155.25.
Up to batch 6, the average loss is  134.03.
Up to batch 7, the average loss is  118.29.
Up to batch 8, the average loss is  105.65.
Up to batch 9, the average loss is   95.53.
Up to batch 10, the average loss is   87.25.
Up to batch 11, the average loss is   80.33.
Up to batch 12, the average loss is   74.48.
Up to batch 13, the average loss is   69.46.
Up to batch 14, the average loss is   65.05.
Up to batch 15, the average loss is   61.31.
The average loss for epoch 0 is   61.31 and mean absolute error is    3.85.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 1: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   57.96.
Up to batch 1, the average loss is   55.11.
Up to batch 2, the average loss is   52.81.
Up to batch 3, the average loss is   51.06.
Up to batch 4, the average loss is   50.58.
Up to batch 5, the average loss is   51.49.
Up to batch 6, the average loss is   53.24.
Up to batch 7, the average loss is   54.20.
Up to batch 8, the average loss is   54.39.
Up to batch 9, the average loss is   54.31.
Up to batch 10, the average loss is   53.83.
Up to batch 11, the average loss is   52.93.
Up to batch 12, the average loss is   51.73.
Up to batch 13, the average loss is   50.34.
Up to batch 14, the average loss is   48.94.
Up to batch 15, the average loss is   47.65.
The average loss for epoch 1 is   47.65 and mean absolute error is    4.30.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 2: Learning rate is 0.10000000149011612.
Up to batch 0, the average loss is   46.38.
Up to batch 1, the average loss is   45.16.
Up to batch 2, the average loss is   44.03.
Up to batch 3, the average loss is   43.11.
Up to batch 4, the average loss is   42.52.
Up to batch 5, the average loss is   42.32.
Up to batch 6, the average loss is   43.06.
Up to batch 7, the average loss is   44.58.
Up to batch 8, the average loss is   45.33.
Up to batch 9, the average loss is   45.15.
Up to batch 10, the average loss is   44.59.
Up to batch 11, the average loss is   43.88.
Up to batch 12, the average loss is   43.17.
Up to batch 13, the average loss is   42.40.
Up to batch 14, the average loss is   41.74.
Up to batch 15, the average loss is   41.19.
The average loss for epoch 2 is   41.19 and mean absolute error is    4.27.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 3: Learning rate is 0.05.
Up to batch 0, the average loss is   40.85.
Up to batch 1, the average loss is   40.11.
Up to batch 2, the average loss is   39.38.
Up to batch 3, the average loss is   38.69.
Up to batch 4, the average loss is   38.01.
Up to batch 5, the average loss is   37.38.
Up to batch 6, the average loss is   36.77.
Up to batch 7, the average loss is   36.18.
Up to batch 8, the average loss is   35.61.
Up to batch 9, the average loss is   35.08.
Up to batch 10, the average loss is   34.54.
Up to batch 11, the average loss is   34.04.
Up to batch 12, the average loss is   33.56.
Up to batch 13, the average loss is   33.08.
Up to batch 14, the average loss is   32.64.
Up to batch 15, the average loss is   32.25.
The average loss for epoch 3 is   32.25 and mean absolute error is    3.64.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 4: Learning rate is 0.05000000074505806.
Up to batch 0, the average loss is   31.83.
Up to batch 1, the average loss is   31.42.
Up to batch 2, the average loss is   31.05.
Up to batch 3, the average loss is   30.72.
Up to batch 4, the average loss is   30.49.
Up to batch 5, the average loss is   30.37.
Up to batch 6, the average loss is   30.15.
Up to batch 7, the average loss is   29.94.
Up to batch 8, the average loss is   29.75.
Up to batch 9, the average loss is   29.56.
Up to batch 10, the average loss is   29.27.
Up to batch 11, the average loss is   28.96.
Up to batch 12, the average loss is   28.67.
Up to batch 13, the average loss is   28.39.
Up to batch 14, the average loss is   28.11.
Up to batch 15, the average loss is   27.80.
The average loss for epoch 4 is   27.80 and mean absolute error is    3.43.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 5: Learning rate is 0.05000000074505806.
Up to batch 0, the average loss is   27.51.
Up to batch 1, the average loss is   27.25.
Up to batch 2, the average loss is   27.05.
Up to batch 3, the average loss is   26.88.
Up to batch 4, the average loss is   26.76.
Up to batch 5, the average loss is   26.60.
Up to batch 6, the average loss is   26.44.
Up to batch 7, the average loss is   26.25.
Up to batch 8, the average loss is   26.08.
Up to batch 9, the average loss is   25.89.
Up to batch 10, the average loss is   25.71.
Up to batch 11, the average loss is   25.48.
Up to batch 12, the average loss is   25.26.
Up to batch 13, the average loss is   25.03.
Up to batch 14, the average loss is   24.81.
Up to batch 15, the average loss is   24.58.
The average loss for epoch 5 is   24.58 and mean absolute error is    3.25.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 6: Learning rate is 0.01.
Up to batch 0, the average loss is   24.36.
Up to batch 1, the average loss is   24.14.
Up to batch 2, the average loss is   23.93.
Up to batch 3, the average loss is   23.71.
Up to batch 4, the average loss is   23.52.
Up to batch 5, the average loss is   23.32.
Up to batch 6, the average loss is   23.12.
Up to batch 7, the average loss is   22.93.
Up to batch 8, the average loss is   22.74.
Up to batch 9, the average loss is   22.55.
Up to batch 10, the average loss is   22.37.
Up to batch 11, the average loss is   22.19.
Up to batch 12, the average loss is   22.01.
Up to batch 13, the average loss is   21.83.
Up to batch 14, the average loss is   21.67.
Up to batch 15, the average loss is   21.50.
The average loss for epoch 6 is   21.50 and mean absolute error is    2.98.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 7: Learning rate is 0.009999999776482582.
Up to batch 0, the average loss is   21.33.
Up to batch 1, the average loss is   21.17.
Up to batch 2, the average loss is   21.01.
Up to batch 3, the average loss is   20.85.
Up to batch 4, the average loss is   20.71.
Up to batch 5, the average loss is   20.57.
Up to batch 6, the average loss is   20.41.
Up to batch 7, the average loss is   20.27.
Up to batch 8, the average loss is   20.13.
Up to batch 9, the average loss is   19.98.
Up to batch 10, the average loss is   19.83.
Up to batch 11, the average loss is   19.69.
Up to batch 12, the average loss is   19.57.
Up to batch 13, the average loss is   19.44.
Up to batch 14, the average loss is   19.32.
Up to batch 15, the average loss is   19.19.
The average loss for epoch 7 is   19.19 and mean absolute error is    2.77.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 8: Learning rate is 0.009999999776482582.
Up to batch 0, the average loss is   19.07.
Up to batch 1, the average loss is   18.95.
Up to batch 2, the average loss is   18.83.
Up to batch 3, the average loss is   18.70.
Up to batch 4, the average loss is   18.58.
Up to batch 5, the average loss is   18.46.
Up to batch 6, the average loss is   18.35.
Up to batch 7, the average loss is   18.24.
Up to batch 8, the average loss is   18.12.
Up to batch 9, the average loss is   18.01.
Up to batch 10, the average loss is   17.90.
Up to batch 11, the average loss is   17.79.
Up to batch 12, the average loss is   17.68.
Up to batch 13, the average loss is   17.58.
Up to batch 14, the average loss is   17.48.
Up to batch 15, the average loss is   17.38.
The average loss for epoch 8 is   17.38 and mean absolute error is    2.61.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 9: Learning rate is 0.005.
Up to batch 0, the average loss is   17.28.
Up to batch 1, the average loss is   17.18.
Up to batch 2, the average loss is   17.08.
Up to batch 3, the average loss is   16.99.
Up to batch 4, the average loss is   16.90.
Up to batch 5, the average loss is   16.80.
Up to batch 6, the average loss is   16.71.
Up to batch 7, the average loss is   16.62.
Up to batch 8, the average loss is   16.53.
Up to batch 9, the average loss is   16.44.
Up to batch 10, the average loss is   16.35.
Up to batch 11, the average loss is   16.26.
Up to batch 12, the average loss is   16.17.
Up to batch 13, the average loss is   16.09.
Up to batch 14, the average loss is   16.00.
Up to batch 15, the average loss is   15.92.
The average loss for epoch 9 is   15.92 and mean absolute error is    2.48.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 10: Learning rate is 0.004999999888241291.
Up to batch 0, the average loss is   15.84.
Up to batch 1, the average loss is   15.76.
Up to batch 2, the average loss is   15.68.
Up to batch 3, the average loss is   15.61.
Up to batch 4, the average loss is   15.53.
Up to batch 5, the average loss is   15.45.
Up to batch 6, the average loss is   15.37.
Up to batch 7, the average loss is   15.29.
Up to batch 8, the average loss is   15.23.
Up to batch 9, the average loss is   15.15.
Up to batch 10, the average loss is   15.08.
Up to batch 11, the average loss is   15.00.
Up to batch 12, the average loss is   14.93.
Up to batch 13, the average loss is   14.86.
Up to batch 14, the average loss is   14.79.
Up to batch 15, the average loss is   14.72.
The average loss for epoch 10 is   14.72 and mean absolute error is    2.37.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 11: Learning rate is 0.004999999888241291.
Up to batch 0, the average loss is   14.65.
Up to batch 1, the average loss is   14.58.
Up to batch 2, the average loss is   14.52.
Up to batch 3, the average loss is   14.45.
Up to batch 4, the average loss is   14.39.
Up to batch 5, the average loss is   14.33.
Up to batch 6, the average loss is   14.26.
Up to batch 7, the average loss is   14.20.
Up to batch 8, the average loss is   14.14.
Up to batch 9, the average loss is   14.08.
Up to batch 10, the average loss is   14.02.
Up to batch 11, the average loss is   13.96.
Up to batch 12, the average loss is   13.90.
Up to batch 13, the average loss is   13.84.
Up to batch 14, the average loss is   13.78.
Up to batch 15, the average loss is   13.72.
The average loss for epoch 11 is   13.72 and mean absolute error is    2.27.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 12: Learning rate is 0.001.
Up to batch 0, the average loss is   13.67.
Up to batch 1, the average loss is   13.60.
Up to batch 2, the average loss is   13.55.
Up to batch 3, the average loss is   13.49.
Up to batch 4, the average loss is   13.44.
Up to batch 5, the average loss is   13.38.
Up to batch 6, the average loss is   13.33.
Up to batch 7, the average loss is   13.28.
Up to batch 8, the average loss is   13.22.
Up to batch 9, the average loss is   13.17.
Up to batch 10, the average loss is   13.12.
Up to batch 11, the average loss is   13.07.
Up to batch 12, the average loss is   13.02.
Up to batch 13, the average loss is   12.97.
Up to batch 14, the average loss is   12.92.
Up to batch 15, the average loss is   12.87.
The average loss for epoch 12 is   12.87 and mean absolute error is    2.19.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 13: Learning rate is 0.0010000000474974513.
Up to batch 0, the average loss is   12.82.
Up to batch 1, the average loss is   12.77.
Up to batch 2, the average loss is   12.72.
Up to batch 3, the average loss is   12.68.
Up to batch 4, the average loss is   12.63.
Up to batch 5, the average loss is   12.58.
Up to batch 6, the average loss is   12.53.
Up to batch 7, the average loss is   12.49.
Up to batch 8, the average loss is   12.45.
Up to batch 9, the average loss is   12.40.
Up to batch 10, the average loss is   12.35.
Up to batch 11, the average loss is   12.30.
Up to batch 12, the average loss is   12.26.
Up to batch 13, the average loss is   12.22.
Up to batch 14, the average loss is   12.17.
Up to batch 15, the average loss is   12.13.
The average loss for epoch 13 is   12.13 and mean absolute error is    2.12.
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 14: Learning rate is 0.0010000000474974513.
Up to batch 0, the average loss is   12.09.
Up to batch 1, the average loss is   12.05.
Up to batch 2, the average loss is   12.01.
Up to batch 3, the average loss is   11.97.
Up to batch 4, the average loss is   11.92.
Up to batch 5, the average loss is   11.88.
Up to batch 6, the average loss is   11.84.
Up to batch 7, the average loss is   11.80.
Up to batch 8, the average loss is   11.76.
Up to batch 9, the average loss is   11.72.
Up to batch 10, the average loss is   11.68.
Up to batch 11, the average loss is   11.64.
Up to batch 12, the average loss is   11.60.
Up to batch 13, the average loss is   11.57.
Up to batch 14, the average loss is   11.54.
Up to batch 15, the average loss is   11.50.
The average loss for epoch 14 is   11.50 and mean absolute error is    2.06.

<keras.src.callbacks.history.History at 0x168619c60>

```
</div>
### Built-in Keras callbacks

Be sure to check out the existing Keras callbacks by
reading the [API docs](https://keras.io/api/callbacks/).
Applications include logging to CSV, saving
the model, visualizing metrics in TensorBoard, and a lot more!
