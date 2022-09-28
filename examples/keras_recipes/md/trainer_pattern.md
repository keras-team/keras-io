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
[this end-to-end GAN model](https://keras.io/guides/customizing_what_happens_in_fit/#wrapping-up-an-endtoend-gan-example),
by putting the custom training step in the Trainer class definition.

---
## Setup


```python
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset and standardize the data
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

```

---
## Define the custom training step

A custom training step can be created by overriding the `train_step()` method of a Model subclass:


```python

class MyTrainer(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # Compute loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics configured in `compile()`
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
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
trainer_1.compile(
    keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
trainer_1.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test)
)

trainer_2.compile(
    keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
trainer_2.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test)
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
938/938 [==============================] - 1s 1ms/step - loss: 0.8928 - accuracy: 0.7692 - val_loss: 0.4520 - val_accuracy: 0.8834
Epoch 2/5
938/938 [==============================] - 1s 1ms/step - loss: 0.4449 - accuracy: 0.8767 - val_loss: 0.3450 - val_accuracy: 0.9066
Epoch 3/5
938/938 [==============================] - 1s 1ms/step - loss: 0.3741 - accuracy: 0.8949 - val_loss: 0.3030 - val_accuracy: 0.9163
Epoch 4/5
938/938 [==============================] - 1s 1ms/step - loss: 0.3340 - accuracy: 0.9060 - val_loss: 0.2773 - val_accuracy: 0.9231
Epoch 5/5
938/938 [==============================] - 1s 1ms/step - loss: 0.3059 - accuracy: 0.9136 - val_loss: 0.2573 - val_accuracy: 0.9284
Epoch 1/5
938/938 [==============================] - 2s 2ms/step - loss: 0.2739 - accuracy: 0.9190 - val_loss: 0.1256 - val_accuracy: 0.9620
Epoch 2/5
938/938 [==============================] - 2s 2ms/step - loss: 0.1265 - accuracy: 0.9624 - val_loss: 0.0976 - val_accuracy: 0.9692
Epoch 3/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0923 - accuracy: 0.9726 - val_loss: 0.0785 - val_accuracy: 0.9758
Epoch 4/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0752 - accuracy: 0.9753 - val_loss: 0.0709 - val_accuracy: 0.9786
Epoch 5/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0636 - accuracy: 0.9795 - val_loss: 0.0713 - val_accuracy: 0.9785

<keras.callbacks.History at 0x1460c99a0>

```
</div>