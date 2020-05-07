# Simple custom layer example: Antirectifier

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2016/01/06<br>
**Last modified:** 2020/04/20<br>
**Description:** Demonstration of custom layer creation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/antirectifier.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/antirectifier.py)



---
## Introduction

This example shows how to create custom layers, using the Antirectifier layer
 (originally proposed as a Keras example script in January 2016), an alternative
to ReLU. Instead of zeroing-out the negative part of the input, it splits the negative
 and positive parts and returns the concatenation of the absolute value
of both. This avoids loss of information, at the cost of an increase in dimensionality.
 To fix the dimensionality increase, we linearly combine the
features back to a space of the original size.


---
## Setup



```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

```

---
## The Antirectifier layer



```python

class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Antirectifier, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, self.kernel)
        return mixed

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Antirectifier, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


```

---
## Let's test-drive it on MNIST



```python
# Training parameters
batch_size = 128
num_classes = 10
epochs = 20

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(256),
        Antirectifier(),
        layers.Dense(256),
        Antirectifier(),
        layers.Dropout(0.5),
        layers.Dense(10),
    ]
)

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)

# Test the model
model.evaluate(x_test, y_test)

```

<div class="k-default-codeblock">
```
60000 train samples
10000 test samples
Epoch 1/20
399/399 [==============================] - 2s 5ms/step - loss: 0.3793 - sparse_categorical_accuracy: 0.8861 - val_loss: 0.4210 - val_sparse_categorical_accuracy: 0.8872
Epoch 2/20
399/399 [==============================] - 2s 4ms/step - loss: 0.1782 - sparse_categorical_accuracy: 0.9494 - val_loss: 0.1416 - val_sparse_categorical_accuracy: 0.9604
Epoch 3/20
399/399 [==============================] - 2s 4ms/step - loss: 0.1383 - sparse_categorical_accuracy: 0.9626 - val_loss: 0.2097 - val_sparse_categorical_accuracy: 0.9489
Epoch 4/20
399/399 [==============================] - 2s 5ms/step - loss: 0.1176 - sparse_categorical_accuracy: 0.9685 - val_loss: 0.1733 - val_sparse_categorical_accuracy: 0.9617
Epoch 5/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0994 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1337 - val_sparse_categorical_accuracy: 0.9723
Epoch 6/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0933 - sparse_categorical_accuracy: 0.9759 - val_loss: 0.1502 - val_sparse_categorical_accuracy: 0.9669
Epoch 7/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0766 - sparse_categorical_accuracy: 0.9794 - val_loss: 0.1338 - val_sparse_categorical_accuracy: 0.9750
Epoch 8/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1565 - val_sparse_categorical_accuracy: 0.9720
Epoch 9/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1402 - val_sparse_categorical_accuracy: 0.9757
Epoch 10/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0633 - sparse_categorical_accuracy: 0.9836 - val_loss: 0.1739 - val_sparse_categorical_accuracy: 0.9719
Epoch 11/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9841 - val_loss: 0.1246 - val_sparse_categorical_accuracy: 0.9778
Epoch 12/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0596 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1640 - val_sparse_categorical_accuracy: 0.9721
Epoch 13/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9873 - val_loss: 0.1610 - val_sparse_categorical_accuracy: 0.9762
Epoch 14/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0588 - sparse_categorical_accuracy: 0.9874 - val_loss: 0.2567 - val_sparse_categorical_accuracy: 0.9612
Epoch 15/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.1624 - val_sparse_categorical_accuracy: 0.9757
Epoch 16/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9890 - val_loss: 0.2040 - val_sparse_categorical_accuracy: 0.9740
Epoch 17/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0537 - sparse_categorical_accuracy: 0.9890 - val_loss: 0.2011 - val_sparse_categorical_accuracy: 0.9780
Epoch 18/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9893 - val_loss: 0.1991 - val_sparse_categorical_accuracy: 0.9786
Epoch 19/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0532 - sparse_categorical_accuracy: 0.9895 - val_loss: 0.2442 - val_sparse_categorical_accuracy: 0.9709
Epoch 20/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0562 - sparse_categorical_accuracy: 0.9899 - val_loss: 0.2186 - val_sparse_categorical_accuracy: 0.9796
313/313 [==============================] - 0s 670us/step - loss: 0.2146 - sparse_categorical_accuracy: 0.9767

[0.21462036669254303, 0.9767000079154968]

```
</div>