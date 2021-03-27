"""
Title: Keras debugging tips
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/16
Last modified: 2020/05/16
Description: Four simple tips to help you debug your Keras code.
"""

"""
## Introduction

It's generally possible to do almost anything in Keras *without writing code* per se:
whether you're implementing a new type of GAN or the latest convnet architecture for
image segmentation, you can usually stick to calling built-in methods. Because all
built-in methods do extensive input validation checks, you will have little to no
debugging to do. A Functional API model made entirely of built-in layers will work on
first try -- if you can compile it, it will run.

However, sometimes, you will need to dive deeper and write your own code. Here are some
common examples:

- Creating a new `Layer` subclass.
- Creating a custom `Metric` subclass.
- Implementing a custom `train_step` on a `Model`.

This document provides a few simple tips to help you navigate debugging in these
situations.

"""

"""
## Tip 1: test each part before you test the whole

If you've created any object that has a chance of not working as expected, don't just
drop it in your end-to-end process and watch sparks fly. Rather, test your custom object
in isolation first. This may seem obvious -- but you'd be surprised how often people
don't start with this.

- If you write a custom layer, don't call `fit()` on your entire model just yet. Call
your layer on some test data first.
- If you write a custom metric, start by printing its output for some reference inputs.

Here's a simple example. Let's write a custom layer a bug in it:

"""

import tensorflow as tf
from tensorflow.keras import layers


class MyAntirectifier(layers.Layer):
    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer="he_normal",
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        # Take the positive part of the input
        pos = tf.nn.relu(inputs)
        # Take the negative part of the input
        neg = tf.nn.relu(-inputs)
        # Concatenate the positive and negative parts
        concatenated = tf.concat([pos, neg], axis=0)
        # Project the concatenation down to the same dimensionality as the input
        return tf.matmul(concatenated, self.kernel)


"""
Now, rather than using it in a end-to-end model directly, let's try to call the layer on
some test data:

```python
x = tf.random.normal(shape=(2, 5))
y = MyAntirectifier()(x)
```

We get the following error:

```
...
      1 x = tf.random.normal(shape=(2, 5))
----> 2 y = MyAntirectifier()(x)
...
     17         neg = tf.nn.relu(-inputs)
     18         concatenated = tf.concat([pos, neg], axis=0)
---> 19         return tf.matmul(concatenated, self.kernel)
...
InvalidArgumentError: Matrix size-incompatible: In[0]: [4,5], In[1]: [10,5] [Op:MatMul]
```

Looks like our input tensor in the `matmul` op may have an incorrect shape.
Let's add a print statement to check the actual shapes:

"""


class MyAntirectifier(layers.Layer):
    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer="he_normal",
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        print("pos.shape:", pos.shape)
        print("neg.shape:", neg.shape)
        concatenated = tf.concat([pos, neg], axis=0)
        print("concatenated.shape:", concatenated.shape)
        print("kernel.shape:", self.kernel.shape)
        return tf.matmul(concatenated, self.kernel)


"""
We get the following:

```
pos.shape: (2, 5)
neg.shape: (2, 5)
concatenated.shape: (4, 5)
kernel.shape: (10, 5)
```

Turns out we had the wrong axis for the `concat` op! We should be concatenating `neg` and
`pos` alongside the feature axis 1, not the batch axis 0. Here's the correct version:
"""


class MyAntirectifier(layers.Layer):
    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer="he_normal",
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        print("pos.shape:", pos.shape)
        print("neg.shape:", neg.shape)
        concatenated = tf.concat([pos, neg], axis=1)
        print("concatenated.shape:", concatenated.shape)
        print("kernel.shape:", self.kernel.shape)
        return tf.matmul(concatenated, self.kernel)


"""
Now our code works fine:
"""

x = tf.random.normal(shape=(2, 5))
y = MyAntirectifier()(x)

"""
## Tip 2: use `model.summary()` and `plot_model()` to check layer output shapes

If you're working with complex network topologies, you're going to need a way
to visualize how your layers are connected and how they transform the data that passes
through them.

Here's an example. Consider this model with three inputs and two outputs (lifted from the
[Functional API
guide](https://keras.io/guides/functional_api/#manipulate-complex-graph-topologies)):

"""

from tensorflow import keras

num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)

"""
Calling `summary()` can help you check the output shape of each layer:
"""

model.summary()

"""
You can also visualize the entire network topology alongside output shapes using
`plot_model`:
"""

keras.utils.plot_model(model, show_shapes=True)

"""
With this plot, any connectivity-level error becomes immediately obvious.
"""

"""
## Tip 3: to debug what happens during `fit()`, use `run_eagerly=True`

The `fit()` method is fast: it runs a well-optimized, fully-compiled computation graph.
That's great for performance, but it also means that the code you're executing isn't the
Python code you've written. This can be problematic when debugging. As you may recall,
Python is slow -- so we use it as a staging language, not as an execution language.

Thankfully, there's an easy way to run your code in "debug mode", fully eagerly:
pass `run_eagerly=True` to `compile()`. Your call to `fit()` will now get executed line
by line, without any optimization. It's slower, but it makes it possible to print the
value of intermediate tensors, or to use a Python debugger. Great for debugging.

Here's a basic example: let's write a really simple model with a custom `train_step`. Our
model just implements gradient descent, but instead of first-order gradients, it uses a
combination of first-order and second-order gradients. Pretty trivial so far.

Can you spot what we're doing wrong?
"""


class MyModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self(inputs, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(targets, preds)
            # Compute first-order gradients
            dl_dw = tape1.gradient(loss, trainable_vars)
        # Compute second-order gradients
        d2l_dw2 = tape2.gradient(dl_dw, trainable_vars)

        # Combine first-order and second-order gradients
        grads = [0.5 * w1 + 0.5 * w2 for (w1, w2) in zip(d2l_dw2, dl_dw)]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets, preds)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


"""
Let's train a one-layer model on MNIST with this custom training loop.

We pick, somewhat at random, a batch size of 1024 and a learning rate of 0.1. The general
idea being to use larger batches and a larger learning rate than usual, since our
"improved" gradients should lead us to quicker convergence.
"""

import numpy as np

# Construct an instance of MyModel
def get_model():
    inputs = keras.Input(shape=(784,))
    intermediate = layers.Dense(256, activation="relu")(inputs)
    outputs = layers.Dense(10, activation="softmax")(intermediate)
    model = MyModel(inputs, outputs)
    return model


# Prepare data
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)) / 255

model = get_model()
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=3, batch_size=1024, validation_split=0.1)

"""
Oh no, it doesn't converge! Something is not working as planned.

Time for some step-by-step printing of what's going on with our gradients.

We add various `print` statements in the `train_step` method, and we make sure to pass
`run_eagerly=True` to `compile()` to run our code step-by-step, eagerly.
"""


class MyModel(keras.Model):
    def train_step(self, data):
        print()
        print("----Start of step: %d" % (self.step_counter,))
        self.step_counter += 1

        inputs, targets = data
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self(inputs, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(targets, preds)
            # Compute first-order gradients
            dl_dw = tape1.gradient(loss, trainable_vars)
        # Compute second-order gradients
        d2l_dw2 = tape2.gradient(dl_dw, trainable_vars)

        print("Max of dl_dw[0]: %.4f" % tf.reduce_max(dl_dw[0]))
        print("Min of dl_dw[0]: %.4f" % tf.reduce_min(dl_dw[0]))
        print("Mean of dl_dw[0]: %.4f" % tf.reduce_mean(dl_dw[0]))
        print("-")
        print("Max of d2l_dw2[0]: %.4f" % tf.reduce_max(d2l_dw2[0]))
        print("Min of d2l_dw2[0]: %.4f" % tf.reduce_min(d2l_dw2[0]))
        print("Mean of d2l_dw2[0]: %.4f" % tf.reduce_mean(d2l_dw2[0]))

        # Combine first-order and second-order gradients
        grads = [0.5 * w1 + 0.5 * w2 for (w1, w2) in zip(d2l_dw2, dl_dw)]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets, preds)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


model = get_model()
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True,
)
model.step_counter = 0
# We pass epochs=1 and steps_per_epoch=10 to only run 10 steps of training.
model.fit(x_train, y_train, epochs=1, batch_size=1024, verbose=0, steps_per_epoch=10)

"""
What did we learn?

- The first order and second order gradients can have values that differ by orders of
magnitudes.
- Sometimes, they may not even have the same sign.
- Their values can vary greatly at each step.

This leads us to an obvious idea: let's normalize the gradients before combining them.
"""


class MyModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self(inputs, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(targets, preds)
            # Compute first-order gradients
            dl_dw = tape1.gradient(loss, trainable_vars)
        # Compute second-order gradients
        d2l_dw2 = tape2.gradient(dl_dw, trainable_vars)

        dl_dw = [tf.math.l2_normalize(w) for w in dl_dw]
        d2l_dw2 = [tf.math.l2_normalize(w) for w in d2l_dw2]

        # Combine first-order and second-order gradients
        grads = [0.5 * w1 + 0.5 * w2 for (w1, w2) in zip(d2l_dw2, dl_dw)]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets, preds)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


model = get_model()
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=5, batch_size=1024, validation_split=0.1)

"""
Now, training converges! It doesn't work well at all, but at least the model learns
something.

After spending a few minutes tuning parameters, we get to the following configuration
that works somewhat well (achieves 97% validation accuracy and seems reasonably robust to
overfitting):

- Use `0.2 * w1 + 0.8 * w2` for combining gradients.
- Use a learning rate that decays linearly over time.

I'm not going to say that the idea works -- this isn't at all how you're supposed to do
second-order optimization (pointers: see the Newton & Gauss-Newton methods, quasi-Newton
methods, and BFGS). But hopefully this demonstration gave you an idea of how you can
debug your way out of uncomfortable training situations.

Remember: use `run_eagerly=True` for debugging what happens in `fit()`. And when your code
is finally working as expected, make sure to remove this flag in order to get the best
runtime performance!

Here's our final training run:
"""


class MyModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self(inputs, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(targets, preds)
            # Compute first-order gradients
            dl_dw = tape1.gradient(loss, trainable_vars)
        # Compute second-order gradients
        d2l_dw2 = tape2.gradient(dl_dw, trainable_vars)

        dl_dw = [tf.math.l2_normalize(w) for w in dl_dw]
        d2l_dw2 = [tf.math.l2_normalize(w) for w in d2l_dw2]

        # Combine first-order and second-order gradients
        grads = [0.2 * w1 + 0.8 * w2 for (w1, w2) in zip(d2l_dw2, dl_dw)]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets, preds)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


model = get_model()
lr = learning_rate = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.1, decay_steps=25, decay_rate=0.1
)
model.compile(
    optimizer=keras.optimizers.SGD(lr),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=50, batch_size=2048, validation_split=0.1)

"""
## Tip 4: if your code is slow, run the TensorFlow profiler

One last tip -- if your code seems slower than it should be, you're going to want to plot
how much time is spent on each computation step. Look for any bottleneck that might be
causing less than 100% device utilization.

To learn more about TensorFlow profiling, see
[this extensive guide](https://www.tensorflow.org/guide/profiler).

You can quickly profile a Keras model via the TensorBoard callback:

```python
# Profile from batches 10 to 15
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             profile_batch=(10, 15))
# Train the model and use the TensorBoard Keras callback to collect
# performance profiling data
model.fit(dataset,
          epochs=1,
          callbacks=[tb_callback])
```

Then navigate to the TensorBoard app and check the "profile" tab.

"""
