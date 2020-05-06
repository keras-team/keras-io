"""
Title: Guide to the Functional API
Author: fchollet
Date created: 2020/04/04
Last modified: 2020/04/04
Description: Complete guide to the functional API.
"""

"""
## Setup
"""

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

"""
## Introduction

The Keras *functional API* is a way to create models that is more flexible than the `tf.keras.Sequential` API. The functional API can handle models with non-linear topology, models with shared layers, and models with multiple inputs or outputs.

The main idea that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build *graphs of layers*.

Consider the following model:

```
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
```

This is a basic graph with three layers. To build this model using the functional API, start by creating an input node:
"""

inputs = keras.Input(shape=(784,))

"""
The shape of the data is set as a 784-dimensional vector. The batch size is always omitted since only the shape of each sample is specified.

If, for example, you have an image input with a shape of `(32, 32, 3)`, you would use:
"""

# Just for demonstration purposes.
img_inputs = keras.Input(shape=(32, 32, 3))

"""
The `inputs` that is returned contains information about the shape and `dtype` of the input data that you feed to your model:
"""

inputs.shape

inputs.dtype

"""
You create a new node in the graph of layers by calling a layer on this `inputs` object:
"""

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

"""
The "layer call" action is like drawing an arrow from "inputs" to this layer you created.
You're "passing" the inputs to the `dense` layer, and out you get `x`.

Let's add a few more layers to the graph of layers:
"""

x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

"""
At this point, you can create a `Model` by specifying its inputs and outputs in the graph of layers:
"""

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

"""
Let's check out what the model summary looks like:
"""

model.summary()

"""
You can also plot the model as a graph:
"""

keras.utils.plot_model(model, "my_first_model.png")

"""
And, optionally, display the input and output shapes of each layer in the plotted graph:
"""

keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

"""
This figure and the code are almost identical. In the code version, the connection arrows are replaced by the call operation.

A "graph of layers" is an intuitive mental image for a deep learning model, and the functional API is a way to create models that closely mirror this.
"""

"""
## Training, evaluation, and inference

Training, evaluation, and inference work exactly in the same way for models built using the functional API as for `Sequential` models.

Here, load the MNIST image data, reshape it into vectors, fit the model on the data (while monitoring performance on a validation split), then evaluate the model on the test data:
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

"""
For further reading, see the [train and evaluate](./train_and_evaluate.ipynb) guide.
"""
