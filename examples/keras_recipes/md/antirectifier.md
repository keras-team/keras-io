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
399/399 [==============================] - 2s 5ms/step - loss: 0.3827 - sparse_categorical_accuracy: 0.8882 - val_loss: 0.1407 - val_sparse_categorical_accuracy: 0.9587
Epoch 2/20
399/399 [==============================] - 2s 5ms/step - loss: 0.1771 - sparse_categorical_accuracy: 0.9513 - val_loss: 0.1337 - val_sparse_categorical_accuracy: 0.9674
Epoch 3/20
399/399 [==============================] - 2s 5ms/step - loss: 0.1400 - sparse_categorical_accuracy: 0.9620 - val_loss: 0.1225 - val_sparse_categorical_accuracy: 0.9709
Epoch 4/20
399/399 [==============================] - 2s 5ms/step - loss: 0.1099 - sparse_categorical_accuracy: 0.9707 - val_loss: 0.1465 - val_sparse_categorical_accuracy: 0.9636
Epoch 5/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0996 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.1703 - val_sparse_categorical_accuracy: 0.9626
Epoch 6/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0860 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1354 - val_sparse_categorical_accuracy: 0.9712
Epoch 7/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0833 - sparse_categorical_accuracy: 0.9791 - val_loss: 0.2018 - val_sparse_categorical_accuracy: 0.9574
Epoch 8/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9814 - val_loss: 0.1527 - val_sparse_categorical_accuracy: 0.9723
Epoch 9/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0710 - sparse_categorical_accuracy: 0.9827 - val_loss: 0.1613 - val_sparse_categorical_accuracy: 0.9694
Epoch 10/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0633 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1463 - val_sparse_categorical_accuracy: 0.9758
Epoch 11/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9856 - val_loss: 0.1390 - val_sparse_categorical_accuracy: 0.9769
Epoch 12/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0561 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1761 - val_sparse_categorical_accuracy: 0.9740
Epoch 13/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9873 - val_loss: 0.1598 - val_sparse_categorical_accuracy: 0.9769
Epoch 14/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0527 - sparse_categorical_accuracy: 0.9879 - val_loss: 0.1565 - val_sparse_categorical_accuracy: 0.9802
Epoch 15/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.1970 - val_sparse_categorical_accuracy: 0.9758
Epoch 16/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9888 - val_loss: 0.1937 - val_sparse_categorical_accuracy: 0.9757
Epoch 17/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9898 - val_loss: 0.1777 - val_sparse_categorical_accuracy: 0.9797
Epoch 18/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0568 - sparse_categorical_accuracy: 0.9894 - val_loss: 0.1831 - val_sparse_categorical_accuracy: 0.9791
Epoch 19/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9900 - val_loss: 0.1812 - val_sparse_categorical_accuracy: 0.9782
Epoch 20/20
399/399 [==============================] - 2s 5ms/step - loss: 0.0503 - sparse_categorical_accuracy: 0.9902 - val_loss: 0.2098 - val_sparse_categorical_accuracy: 0.9776
313/313 [==============================] - 0s 731us/step - loss: 0.2002 - sparse_categorical_accuracy: 0.9776

[0.20024622976779938, 0.9775999784469604]

```
</div>