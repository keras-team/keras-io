# Simple custom layer example: Antirectifier

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2016/01/06<br>
**Last modified:** 2023/11/20<br>
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
import keras
from keras import layers
from keras import ops
```

---
## The Antirectifier layer

To implement a custom layer:

- Create the state variables via `add_weight()` in `__init__` or `build()`.
Similarly, you can also create sublayers.
- Implement the `call()` method, taking the layer's input tensor(s) and
return the output tensor(s).
- Optionally, you can also enable serialization by implementing `get_config()`,
which returns a configuration dictionary.

See also the guide
[Making new layers and models via subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/).


```python

class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
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
        inputs -= ops.mean(inputs, axis=-1, keepdims=True)
        pos = ops.relu(inputs)
        neg = ops.relu(-inputs)
        concatenated = ops.concatenate([pos, neg], axis=-1)
        mixed = ops.matmul(concatenated, self.kernel)
        return mixed

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super().get_config()
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
 399/399 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - loss: 0.6226 - sparse_categorical_accuracy: 0.8146 - val_loss: 0.4256 - val_sparse_categorical_accuracy: 0.8808
Epoch 2/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.1887 - sparse_categorical_accuracy: 0.9455 - val_loss: 0.1556 - val_sparse_categorical_accuracy: 0.9588
Epoch 3/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.1406 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.1531 - val_sparse_categorical_accuracy: 0.9611
Epoch 4/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.1084 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.1178 - val_sparse_categorical_accuracy: 0.9731
Epoch 5/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0995 - sparse_categorical_accuracy: 0.9738 - val_loss: 0.2207 - val_sparse_categorical_accuracy: 0.9526
Epoch 6/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0831 - sparse_categorical_accuracy: 0.9769 - val_loss: 0.2092 - val_sparse_categorical_accuracy: 0.9533
Epoch 7/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9807 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9749
Epoch 8/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0653 - sparse_categorical_accuracy: 0.9827 - val_loss: 0.1000 - val_sparse_categorical_accuracy: 0.9791
Epoch 9/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0590 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1320 - val_sparse_categorical_accuracy: 0.9750
Epoch 10/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0587 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1439 - val_sparse_categorical_accuracy: 0.9747
Epoch 11/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0622 - sparse_categorical_accuracy: 0.9853 - val_loss: 0.1473 - val_sparse_categorical_accuracy: 0.9753
Epoch 12/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0554 - sparse_categorical_accuracy: 0.9869 - val_loss: 0.1529 - val_sparse_categorical_accuracy: 0.9757
Epoch 13/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0507 - sparse_categorical_accuracy: 0.9884 - val_loss: 0.1452 - val_sparse_categorical_accuracy: 0.9783
Epoch 14/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0468 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.1435 - val_sparse_categorical_accuracy: 0.9796
Epoch 15/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0478 - sparse_categorical_accuracy: 0.9892 - val_loss: 0.1580 - val_sparse_categorical_accuracy: 0.9770
Epoch 16/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9888 - val_loss: 0.1957 - val_sparse_categorical_accuracy: 0.9753
Epoch 17/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0478 - sparse_categorical_accuracy: 0.9896 - val_loss: 0.1865 - val_sparse_categorical_accuracy: 0.9779
Epoch 18/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0478 - sparse_categorical_accuracy: 0.9893 - val_loss: 0.2107 - val_sparse_categorical_accuracy: 0.9747
Epoch 19/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9894 - val_loss: 0.2306 - val_sparse_categorical_accuracy: 0.9734
Epoch 20/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9910 - val_loss: 0.2201 - val_sparse_categorical_accuracy: 0.9731
 313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 802us/step - loss: 0.2086 - sparse_categorical_accuracy: 0.9710

[0.19070196151733398, 0.9740999937057495]

```
</div>