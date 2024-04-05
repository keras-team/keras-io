# Trainer pattern

**Author:** [nkovela1](https://nkovela1.github.io/)<br>
**Date created:** 2022/09/19<br>
**Last modified:** 2022/09/26<br>
**Description:** Guide on how to share a custom training step across multiple Keras models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/trainer_pattern.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/trainer_pattern.py)



---
## Introduction

This example shows how to create a custom training step using the "Trainer pattern",
which can then be shared across multiple Keras models. This pattern overrides the
`train_step()` method of the `keras.Model` class, allowing for training loops
beyond plain supervised learning.

The Trainer pattern can also easily be adapted to more complex models with larger
custom training steps, such as
[this end-to-end GAN model](https://keras.io/guides/custom_train_step_in_tensorflow/#wrapping-up-an-endtoend-gan-example),
by putting the custom training step in the Trainer class definition.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

# Load MNIST dataset and standardize the data
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

```

---
## Define the Trainer class

A custom training and evaluation step can be created by overriding
the `train_step()` and `test_step()` method of a `Model` subclass:


```python

class MyTrainer(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Create loss and metrics here.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy()

    @property
    def metrics(self):
        # List metrics here.
        return [self.accuracy_metric]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # Compute loss value
            loss = self.loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        for metric in self.metrics:
            metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        # Inference step
        y_pred = self.model(x, training=False)

        # Update metrics
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        # Equivalent to `call()` of the wrapped keras.Model
        x = self.model(x)
        return x

```

---
## Define multiple models to share the custom training step

Let's define two different models that can share our Trainer class and its custom `train_step()`:


```python
# A model defined using Sequential API
model_a = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

# A model defined using Functional API
func_input = keras.Input(shape=(28, 28, 1))
x = keras.layers.Flatten(input_shape=(28, 28))(func_input)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dropout(0.4)(x)
func_output = keras.layers.Dense(10, activation="softmax")(x)

model_b = keras.Model(func_input, func_output)
```

---
## Create Trainer class objects from the models


```python
trainer_1 = MyTrainer(model_a)
trainer_2 = MyTrainer(model_b)
```

---
## Compile and fit the models to the MNIST dataset


```python
trainer_1.compile(optimizer=keras.optimizers.SGD())
trainer_1.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test)
)

trainer_2.compile(optimizer=keras.optimizers.Adam())
trainer_2.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test)
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
...
Epoch 4/5
 938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - sparse_categorical_accuracy: 0.9770 - val_sparse_categorical_accuracy: 0.9770
Epoch 5/5
 938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - sparse_categorical_accuracy: 0.9805 - val_sparse_categorical_accuracy: 0.9789

<keras.src.callbacks.history.History at 0x7efe405fe560>

```
</div>