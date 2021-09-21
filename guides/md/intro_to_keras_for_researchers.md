# Introduction to Keras for Researchers

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/01<br>
**Last modified:** 2020/10/02<br>
**Description:** Everything you need to know to use Keras & TensorFlow for deep learning research.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/intro_to_keras_for_researchers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/intro_to_keras_for_researchers.py)



---
## Setup


```python
import tensorflow as tf
from tensorflow import keras
```

---
## Introduction

Are you a machine learning researcher? Do you publish at NeurIPS and push the
state-of-the-art in CV and NLP? This guide will serve as your first introduction to core
Keras & TensorFlow API concepts.

In this guide, you will learn about:

- Tensors, variables, and gradients in TensorFlow
- Creating layers by subclassing the `Layer` class
- Writing low-level training loops
- Tracking losses created by layers via the `add_loss()` method
- Tracking metrics in a low-level training loop
- Speeding up execution with a compiled `tf.function`
- Executing layers in training or inference mode
- The Keras Functional API

You will also see the Keras API in action in two end-to-end research examples:
a Variational Autoencoder, and a Hypernetwork.

---
## Tensors

TensorFlow is an infrastructure layer for differentiable programming.
At its heart, it's a framework for manipulating N-dimensional arrays (tensors),
much like NumPy.

However, there are three key differences between NumPy and TensorFlow:

- TensorFlow can leverage hardware accelerators such as GPUs and TPUs.
- TensorFlow can automatically compute the gradient of arbitrary differentiable tensor expressions.
- TensorFlow computation can be distributed to large numbers of devices on a single machine, and large number of
machines (potentially with multiple devices each).

Let's take a look at the object that is at the core of TensorFlow: the Tensor.

Here's a constant tensor:


```python
x = tf.constant([[5, 2], [1, 3]])
print(x)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[5 2]
 [1 3]], shape=(2, 2), dtype=int32)

```
</div>
You can get its value as a NumPy array by calling `.numpy()`:


```python
x.numpy()
```




<div class="k-default-codeblock">
```
array([[5, 2],
       [1, 3]], dtype=int32)

```
</div>
Much like a NumPy array, it features the attributes `dtype` and `shape`:


```python
print("dtype:", x.dtype)
print("shape:", x.shape)
```

<div class="k-default-codeblock">
```
dtype: <dtype: 'int32'>
shape: (2, 2)

```
</div>
A common way to create constant tensors is via `tf.ones` and `tf.zeros` (just like `np.ones` and `np.zeros`):


```python
print(tf.ones(shape=(2, 1)))
print(tf.zeros(shape=(2, 1)))
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[1.]
 [1.]], shape=(2, 1), dtype=float32)
tf.Tensor(
[[0.]
 [0.]], shape=(2, 1), dtype=float32)

```
</div>
You can also create random constant tensors:


```python
x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)

x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")

```

---
## Variables

Variables are special tensors used to store mutable state (such as the weights of a neural network).
You create a `Variable` using some initial value:


```python
initial_value = tf.random.normal(shape=(2, 2))
a = tf.Variable(initial_value)
print(a)

```

<div class="k-default-codeblock">
```
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[-1.7639292,  0.4263797],
       [-0.3954156, -0.6072024]], dtype=float32)>

```
</div>
You update the value of a `Variable` by using the methods `.assign(value)`, `.assign_add(increment)`, or `.assign_sub(decrement)`:


```python
new_value = tf.random.normal(shape=(2, 2))
a.assign(new_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j]

added_value = tf.random.normal(shape=(2, 2))
a.assign_add(added_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j] + added_value[i, j]
```

---
## Doing math in TensorFlow

If you've used NumPy, doing math in TensorFlow will look very familiar.
The main difference is that your TensorFlow code can run on GPU and TPU.


```python
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

c = a + b
d = tf.square(c)
e = tf.exp(d)
```

---
## Gradients

Here's another big difference with NumPy: you can automatically retrieve the gradient of any differentiable expression.

Just open a `GradientTape`, start "watching" a tensor via `tape.watch()`,
and compose a differentiable expression using this tensor as input:


```python
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    tape.watch(a)  # Start recording the history of operations applied to `a`
    c = tf.sqrt(tf.square(a) + tf.square(b))  # Do some math using `a`
    # What's the gradient of `c` with respect to `a`?
    dc_da = tape.gradient(c, a)
    print(dc_da)

```

<div class="k-default-codeblock">
```
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)

```
</div>
By default, variables are watched automatically, so you don't need to manually `watch` them:


```python
a = tf.Variable(a)

with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)

```
</div>
Note that you can compute higher-order derivatives by nesting tapes:


```python
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print(d2c_da2)

```

<div class="k-default-codeblock">
```
tf.Tensor(
[[1.2510717e-03 4.4079739e-01]
 [2.1326542e-04 3.7843192e-01]], shape=(2, 2), dtype=float32)

```
</div>
---
## Keras layers

While TensorFlow is an **infrastructure layer for differentiable programming**,
dealing with tensors, variables, and gradients,
Keras is a **user interface for deep learning**, dealing with
layers, models, optimizers, loss functions, metrics, and more.

Keras serves as the high-level API for TensorFlow:
Keras is what makes TensorFlow simple and productive.

The `Layer` class is the fundamental abstraction in Keras.
A `Layer` encapsulates a state (weights) and some computation
(defined in the call method).

A simple layer looks like this:


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

```

You would use a `Layer` instance much like a Python function:


```python
# Instantiate our layer.
linear_layer = Linear(units=4, input_dim=2)

# The layer can be treated as a function.
# Here we call it on some data.
y = linear_layer(tf.ones((2, 2)))
assert y.shape == (2, 4)
```

The weight variables (created in `__init__`) are automatically
tracked under the `weights` property:


```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

You have many built-in layers available, from `Dense` to `Conv2D` to `LSTM` to
fancier ones like `Conv3DTranspose` or `ConvLSTM2D`. Be smart about reusing
built-in functionality.

---
## Layer weight creation

The `self.add_weight()` method gives you a shortcut for creating weights:


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Instantiate our lazy layer.
linear_layer = Linear(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
```

---
## Layer gradients

You can automatically retrieve the gradients of the weights of a layer by
calling it inside a `GradientTape`. Using these gradients, you can update the
weights of the layer, either manually, or using an optimizer object. Of course,
you can modify the gradients before using them, if you need to.


```python
# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# Instantiate our linear layer (defined above) with 10 units.
linear_layer = Linear(10)

# Instantiate a logistic loss function that expects integer targets.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# Iterate over the batches of the dataset.
for step, (x, y) in enumerate(dataset):

    # Open a GradientTape.
    with tf.GradientTape() as tape:

        # Forward pass.
        logits = linear_layer(x)

        # Loss value for this batch.
        loss = loss_fn(y, logits)

    # Get gradients of the loss wrt the weights.
    gradients = tape.gradient(loss, linear_layer.trainable_weights)

    # Update the weights of our linear layer.
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))

    # Logging.
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 2.4605865478515625
Step: 100 Loss: 2.3112568855285645
Step: 200 Loss: 2.1920084953308105
Step: 300 Loss: 2.1255125999450684
Step: 400 Loss: 2.020744562149048
Step: 500 Loss: 2.060229539871216
Step: 600 Loss: 1.9214580059051514
Step: 700 Loss: 1.7613574266433716
Step: 800 Loss: 1.6828575134277344
Step: 900 Loss: 1.6320191621780396

```
</div>
---
## Trainable and non-trainable weights

Weights created by layers can be either trainable or non-trainable. They're
exposed in `trainable_weights` and `non_trainable_weights` respectively.
Here's a layer with a non-trainable weight:


```python

class ComputeSum(keras.layers.Layer):
    """Returns the sum of the inputs."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy())  # [2. 2.]

y = my_sum(x)
print(y.numpy())  # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

<div class="k-default-codeblock">
```
[2. 2.]
[4. 4.]

```
</div>
---
## Layers that own layers

Layers can be recursively nested to create bigger computation blocks.
Each layer will track the weights of its sublayers
(both trainable and non-trainable).


```python
# Let's reuse the Linear class
# with a `build` method that we defined above.


class MLP(keras.layers.Layer):
    """Simple stack of Linear layers."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLP()

# The first call to the `mlp` object will create the weights.
y = mlp(tf.ones(shape=(3, 64)))

# Weights are recursively tracked.
assert len(mlp.weights) == 6
```

Note that our manually-created MLP above is equivalent to the following
built-in option:


```python
mlp = keras.Sequential(
    [
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10),
    ]
)
```

---
## Tracking losses created by layers

Layers can create losses during the forward pass via the `add_loss()` method.
This is especially useful for regularization losses.
The losses created by sublayers are recursively tracked by the parent layers.

Here's a layer that creates an activity regularization loss:


```python

class ActivityRegularization(keras.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

```

Any model incorporating this layer will track this regularization loss:


```python
# Let's use the loss layer in a MLP block.


class SparseMLP(keras.layers.Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self):
        super(SparseMLP, self).__init__()
        self.linear_1 = Linear(32)
        self.regularization = ActivityRegularization(1e-2)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.regularization(x)
        return self.linear_3(x)


mlp = SparseMLP()
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # List containing one float32 scalar
```

<div class="k-default-codeblock">
```
[<tf.Tensor: shape=(), dtype=float32, numpy=0.21796302>]

```
</div>
These losses are cleared by the top-level layer at the start of each forward
pass -- they don't accumulate. `layer.losses` always contains only the losses
created during the last forward pass. You would typically use these losses by
summing them before computing your gradients when writing a training loop.


```python
# Losses correspond to the *last* forward pass.
mlp = SparseMLP()
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # No accumulation.

# Let's demonstrate how to use these losses in a training loop.

# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# A new MLP.
mlp = SparseMLP()

# Loss and optimizer.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:

        # Forward pass.
        logits = mlp(x)

        # External loss value for this batch.
        loss = loss_fn(y, logits)

        # Add the losses created during the forward pass.
        loss += sum(mlp.losses)

        # Get gradients of the loss wrt the weights.
        gradients = tape.gradient(loss, mlp.trainable_weights)

    # Update the weights of our linear layer.
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # Logging.
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 6.307978630065918
Step: 100 Loss: 2.5283541679382324
Step: 200 Loss: 2.4068050384521484
Step: 300 Loss: 2.3749840259552
Step: 400 Loss: 2.34563946723938
Step: 500 Loss: 2.3380157947540283
Step: 600 Loss: 2.3201656341552734
Step: 700 Loss: 2.3250539302825928
Step: 800 Loss: 2.344613790512085
Step: 900 Loss: 2.3183579444885254

```
</div>
---
## Keeping track of training metrics

Keras offers a broad range of built-in metrics, like `tf.keras.metrics.AUC`
or `tf.keras.metrics.PrecisionAtRecall`. It's also easy to create your
own metrics in a few lines of code.

To use a metric in a custom training loop, you would:

- Instantiate the metric object, e.g. `metric = tf.keras.metrics.AUC()`
- Call its `metric.udpate_state(targets, predictions)` method for each batch of data
- Query its result via `metric.result()`
- Reset the metric's state at the end of an epoch or at the start of an evaluation via
`metric.reset_state()`

Here's a simple example:


```python
# Instantiate a metric object
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Prepare our layer, loss, and optimizer.
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(2):
    # Iterate over the batches of a dataset.
    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x)
            # Compute the loss value for this batch.
            loss_value = loss_fn(y, logits)

        # Update the state of the `accuracy` metric.
        accuracy.update_state(y, logits)

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Logging the current accuracy value so far.
        if step % 200 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("Total running accuracy so far: %.3f" % accuracy.result())

    # Reset the metric's state at the end of an epoch
    accuracy.reset_state()
```

<div class="k-default-codeblock">
```
Epoch: 0 Step: 0
Total running accuracy so far: 0.141
Epoch: 0 Step: 200
Total running accuracy so far: 0.751
Epoch: 0 Step: 400
Total running accuracy so far: 0.827
Epoch: 0 Step: 600
Total running accuracy so far: 0.859
Epoch: 0 Step: 800
Total running accuracy so far: 0.876
Epoch: 1 Step: 0
Total running accuracy so far: 0.938
Epoch: 1 Step: 200
Total running accuracy so far: 0.944
Epoch: 1 Step: 400
Total running accuracy so far: 0.944
Epoch: 1 Step: 600
Total running accuracy so far: 0.945
Epoch: 1 Step: 800
Total running accuracy so far: 0.945

```
</div>
In addition to this, similarly to the `self.add_loss()` method, you have access
to an `self.add_metric()` method on layers. It tracks the average of
whatever quantity you pass to it. You can reset the value of these metrics
by calling `layer.reset_metrics()` on any layer or model.

You can also define your own metrics by subclassing `keras.metrics.Metric`.
You need to override the three functions called above:

- Override `update_state()` to update the statistic values.
- Override `result()` to return the metric value.
- Override `reset_state()` to reset the metric to its initial state.

Here is an example where we implement the F1-score metric
(with support for sample weighting).


```python

class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", dtype="float32", threshold=0.5, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = 0.5
        self.true_positives = self.add_weight(
            name="tp", dtype=dtype, initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", dtype=dtype, initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", dtype=dtype, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater_equal(y_pred, self.threshold)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.cast(y_true & y_pred, self.dtype)
        false_positives = tf.cast(~y_true & y_pred, self.dtype)
        false_negatives = tf.cast(y_true & ~y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_positives *= sample_weight
            false_positives *= sample_weight
            false_negatives *= sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return precision * recall * 2.0 / (precision + recall)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

```

Let's test-drive it:


```python
m = F1Score()
m.update_state([0, 1, 0, 0], [0.3, 0.5, 0.8, 0.9])
print("Intermediate result:", float(m.result()))

m.update_state([1, 1, 1, 1], [0.1, 0.7, 0.6, 0.0])
print("Final result:", float(m.result()))

```

<div class="k-default-codeblock">
```
Intermediate result: 0.5
Final result: 0.6000000238418579

```
</div>
---
## Compiled functions

Running eagerly is great for debugging, but you will get better performance by
compiling your computation into static graphs. Static graphs are a researcher's
best friends. You can compile any function by wrapping it in a `tf.function`
decorator.


```python
# Prepare our layer, loss, and optimizer.
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Create a training step function.


@tf.function  # Make it fast.
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 2.291861057281494
Step: 100 Loss: 0.5378965735435486
Step: 200 Loss: 0.48008084297180176
Step: 300 Loss: 0.3359006941318512
Step: 400 Loss: 0.28147661685943604
Step: 500 Loss: 0.31419697403907776
Step: 600 Loss: 0.2735794484615326
Step: 700 Loss: 0.3001103401184082
Step: 800 Loss: 0.18827161192893982
Step: 900 Loss: 0.15798673033714294

```
</div>
---
## Training mode & inference mode

Some layers, in particular the `BatchNormalization` layer and the `Dropout`
layer, have different behaviors during training and inference. For such layers,
it is standard practice to expose a `training` (boolean) argument in the `call`
method.

By exposing this argument in `call`, you enable the built-in training and
evaluation loops (e.g. fit) to correctly use the layer in training and
inference modes.


```python

class Dropout(keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class MLPWithDropout(keras.layers.Layer):
    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)


mlp = MLPWithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)
y_test = mlp(tf.ones((2, 2)), training=False)
```

---
## The Functional API for model-building

To build deep learning models, you don't have to use object-oriented programming all the
time. All layers we've seen so far can also be composed functionally, like this (we call
it the "Functional API"):


```python
# We use an `Input` object to describe the shape and dtype of the inputs.
# This is the deep learning equivalent of *declaring a type*.
# The shape argument is per-sample; it does not include the batch size.
# The functional API focused on defining per-sample transformations.
# The model we create will automatically batch the per-sample transformations,
# so that it can be called on batches of data.
inputs = tf.keras.Input(shape=(16,), dtype="float32")

# We call layers on these "type" objects
# and they return updated types (new shapes/dtypes).
x = Linear(32)(inputs)  # We are reusing the Linear layer we defined earlier.
x = Dropout(0.5)(x)  # We are reusing the Dropout layer we defined earlier.
outputs = Linear(10)(x)

# A functional `Model` can be defined by specifying inputs and outputs.
# A model is itself a layer like any other.
model = tf.keras.Model(inputs, outputs)

# A functional model already has weights, before being called on any data.
# That's because we defined its input shape in advance (in `Input`).
assert len(model.weights) == 4

# Let's call our model on some data, for fun.
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)

# You can pass a `training` argument in `__call__`
# (it will get passed down to the Dropout layer).
y = model(tf.ones((2, 16)), training=True)
```

The Functional API tends to be more concise than subclassing, and provides a few other
advantages (generally the same advantages that functional, typed languages provide over
untyped OO development). However, it can only be used to define DAGs of layers --
recursive networks should be defined as Layer subclasses instead.

Learn more about the Functional API [here](/guides/functional_api/).

In your research workflows, you may often find yourself mix-and-matching OO models and
Functional models.

Note that the `Model` class also features built-in training & evaluation loops:
`fit()`, `predict()` and `evaluate()` (configured via the `compile()` method).
These built-in functions give you access to the
following built-in training infrastructure features:

* [Callbacks](/api/callbacks/). You can leverage built-in
callbacks for early-stopping, model checkpointing,
and monitoring training with TensorBoard. You can also
[implement custom callbacks](/guides/writing_your_own_callbacks/) if needed.
* [Distributed training](https://keras.io/guides/distributed_training/). You
can easily scale up your training to multiple GPUs, TPU, or even multiple machines
with the `tf.distribute` API -- with no changes to your code.
* [Step fusing](https://keras.io/api/models/model_training_apis/#compile-method).
With the `steps_per_execution` argument in `Model.compile()`, you can process
multiple batches in a single `tf.function` call, which greatly improves
device utilization on TPUs.

We won't go into the details, but we provide a simple code example
below. It leverages the built-in training infrastructure to implement the MNIST
example above.


```python
inputs = tf.keras.Input(shape=(784,), dtype="float32")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)

# Specify the loss, optimizer, and metrics with `compile()`.
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model with the dataset for 2 epochs.
model.fit(dataset, epochs=2)
model.predict(dataset)
model.evaluate(dataset)
```

<div class="k-default-codeblock">
```
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3958 - sparse_categorical_accuracy: 0.8872
Epoch 2/2
938/938 [==============================] - 1s 1ms/step - loss: 0.1916 - sparse_categorical_accuracy: 0.9447
938/938 [==============================] - 1s 798us/step - loss: 0.1729 - sparse_categorical_accuracy: 0.9485

[0.1728748232126236, 0.9484500288963318]

```
</div>
You can always subclass the `Model` class (it works exactly like subclassing
`Layer`) if you want to leverage built-in training loops for your OO models.
Just override the `Model.train_step()` to
customize what happens in `fit()` while retaining support
for the built-in infrastructure features outlined above -- callbacks,
zero-code distribution support, and step fusing support.
You may also override `test_step()` to customize what happens in `evaluate()`,
and override `predict_step()` to customize what happens in `predict()`. For more
information, please refer to
[this guide](https://keras.io/guides/customizing_what_happens_in_fit/).


```python

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch.
        return [self.loss_tracker, self.accuracy]


inputs = tf.keras.Input(shape=(784,), dtype="float32")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)
model = CustomModel(inputs, outputs)
model.compile()
model.fit(dataset, epochs=2)
```

<div class="k-default-codeblock">
```
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3737 - accuracy: 0.8340
Epoch 2/2
938/938 [==============================] - 1s 946us/step - loss: 0.1934 - accuracy: 0.9405

<keras.callbacks.History at 0x15dfae110>

```
</div>
---
## End-to-end experiment example 1: variational autoencoders.

Here are some of the things you've learned so far:

- A `Layer` encapsulates a state (created in `__init__` or `build`) and some computation
(defined in `call`).
- Layers can be recursively nested to create new, bigger computation blocks.
- You can easily write highly hackable training loops by opening a
`GradientTape`, calling your model inside the tape's scope, then retrieving
gradients and applying them via an optimizer.
- You can speed up your training loops using the `@tf.function` decorator.
- Layers can create and track losses (typically regularization losses) via
`self.add_loss()`.

Let's put all of these things together into an end-to-end example: we're going to
implement a Variational AutoEncoder (VAE). We'll train it on MNIST digits.

Our VAE will be a subclass of `Layer`, built as a nested composition of layers that
subclass `Layer`. It will feature a regularization loss (KL divergence).

Below is our model definition.

First, we have an `Encoder` class, which uses a `Sampling` layer to map a MNIST digit to
a latent-space triplet `(z_mean, z_log_var, z)`.


```python
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

```

Next, we have a `Decoder` class, which maps the probabilistic latent space coordinates
back to a MNIST digit.


```python

class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

```

Finally, our `VariationalAutoEncoder` composes together an encoder and a decoder, and
creates a KL divergence regularization loss via `add_loss()`.


```python

class VariationalAutoEncoder(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

```

Now, let's write a training loop. Our training step is decorated with a `@tf.function` to
compile into a super fast graph function.


```python
# Our model.
vae = VariationalAutoEncoder(original_dim=784, intermediate_dim=64, latent_dim=32)

# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Prepare a dataset.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.shuffle(buffer_size=1024).batch(32)


@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        reconstructed = vae(x)  # Compute input reconstruction.
        # Compute loss.
        loss = loss_fn(x, reconstructed)
        loss += sum(vae.losses)  # Add KLD term.
    # Update the weights of the VAE.
    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return loss


losses = []  # Keep track of the losses over time.
for step, x in enumerate(dataset):
    loss = training_step(x)
    # Logging.
    losses.append(float(loss))
    if step % 100 == 0:
        print("Step:", step, "Loss:", sum(losses) / len(losses))

    # Stop after 1000 steps.
    # Training the model to convergence is left
    # as an exercise to the reader.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 0.3246927559375763
Step: 100 Loss: 0.12636583357459247
Step: 200 Loss: 0.099717023916802
Step: 300 Loss: 0.0896754782535507
Step: 400 Loss: 0.08474012454065896
Step: 500 Loss: 0.08153954131933981
Step: 600 Loss: 0.07914437327577349
Step: 700 Loss: 0.07779341802723738
Step: 800 Loss: 0.07658644887466406
Step: 900 Loss: 0.07564477964855325
Step: 1000 Loss: 0.07468595038671474

```
</div>
As you can see, building and training this type of model in Keras
is quick and painless.

Now, you may find that the code above is somewhat verbose: we handle every little detail
on our own, by hand. This gives the most flexibility, but it's also a bit of work.

Let's take a look at what the Functional API version of
our VAE looks like:


```python
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)
```

Much more concise, right?

By the way, Keras also features built-in training & evaluation loops on its `Model` class
(`fit()` and `evaluate()`). Check it out:


```python
# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Prepare a dataset.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.map(lambda x: (x, x))  # Use x_train as both inputs & targets
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# Configure the model for training.
vae.compile(optimizer, loss=loss_fn)

# Actually training the model.
vae.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
1875/1875 [==============================] - 3s 1ms/step - loss: 0.0713

<keras.callbacks.History at 0x15e150f10>

```
</div>
The use of the Functional API and `fit` reduces our example from 65 lines to 25 lines
(including model definition & training). The Keras philosophy is to offer you
productivity-boosting features like
these, while simultaneously empowering you to write everything yourself to gain absolute
control over every little detail. Like we did in the low-level training loop two
paragraphs earlier.

---
## End-to-end experiment example 2: hypernetworks.

Let's take a look at another kind of research experiment: hypernetworks.

The idea is to use a small deep neural network (the hypernetwork) to generate
the weights for a larger network (the main network).

Let's implement a really trivial hypernetwork: we'll use a small 2-layer network  to
generate the weights of a larger 3-layer network.


```python
import numpy as np

input_dim = 784
classes = 10

# This is the main network we'll actually use to predict labels.
main_network = keras.Sequential(
    [keras.layers.Dense(64, activation=tf.nn.relu), keras.layers.Dense(classes),]
)

# It doesn't need to create its own weights, so let's mark its layers
# as already built. That way, calling `main_network` won't create new variables.
for layer in main_network.layers:
    layer.built = True

# This is the number of weight coefficients to generate. Each layer in the
# main network requires output_dim * input_dim + output_dim coefficients.
num_weights_to_generate = (classes * 64 + classes) + (64 * input_dim + 64)

# This is the hypernetwork that generates the weights of the `main_network` above.
hypernetwork = keras.Sequential(
    [
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(num_weights_to_generate, activation=tf.nn.sigmoid),
    ]
)
```

This is our training loop. For each batch of data:

- We use `hypernetwork` to generate an array of weight coefficients, `weights_pred`
- We reshape these coefficients into kernel & bias tensors for the `main_network`
- We run the forward pass of the `main_network` to compute the actual MNIST predictions
- We run backprop through the weights of the `hypernetwork` to minimize the
final classification loss


```python
# Loss and optimizer.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)

# We'll use a batch size of 1 for this experiment.
dataset = dataset.shuffle(buffer_size=1024).batch(1)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Predict weights for the outer model.
        weights_pred = hypernetwork(x)

        # Reshape them to the expected shapes for w and b for the outer model.
        # Layer 0 kernel.
        start_index = 0
        w0_shape = (input_dim, 64)
        w0_coeffs = weights_pred[:, start_index : start_index + np.prod(w0_shape)]
        w0 = tf.reshape(w0_coeffs, w0_shape)
        start_index += np.prod(w0_shape)
        # Layer 0 bias.
        b0_shape = (64,)
        b0_coeffs = weights_pred[:, start_index : start_index + np.prod(b0_shape)]
        b0 = tf.reshape(b0_coeffs, b0_shape)
        start_index += np.prod(b0_shape)
        # Layer 1 kernel.
        w1_shape = (64, classes)
        w1_coeffs = weights_pred[:, start_index : start_index + np.prod(w1_shape)]
        w1 = tf.reshape(w1_coeffs, w1_shape)
        start_index += np.prod(w1_shape)
        # Layer 1 bias.
        b1_shape = (classes,)
        b1_coeffs = weights_pred[:, start_index : start_index + np.prod(b1_shape)]
        b1 = tf.reshape(b1_coeffs, b1_shape)
        start_index += np.prod(b1_shape)

        # Set the weight predictions as the weight variables on the outer model.
        main_network.layers[0].kernel = w0
        main_network.layers[0].bias = b0
        main_network.layers[1].kernel = w1
        main_network.layers[1].bias = b1

        # Inference on the outer model.
        preds = main_network(x)
        loss = loss_fn(y, preds)

    # Train only inner model.
    grads = tape.gradient(loss, hypernetwork.trainable_weights)
    optimizer.apply_gradients(zip(grads, hypernetwork.trainable_weights))
    return loss


losses = []  # Keep track of the losses over time.
for step, (x, y) in enumerate(dataset):
    loss = train_step(x, y)

    # Logging.
    losses.append(float(loss))
    if step % 100 == 0:
        print("Step:", step, "Loss:", sum(losses) / len(losses))

    # Stop after 1000 steps.
    # Training the model to convergence is left
    # as an exercise to the reader.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 1.3274627923965454
Step: 100 Loss: 2.5709669510326765
Step: 200 Loss: 2.2051062234700542
Step: 300 Loss: 2.0191424489686534
Step: 400 Loss: 1.8865989956417193
Step: 500 Loss: 1.7706833476604333
Step: 600 Loss: 1.6479115988951523
Step: 700 Loss: 1.603230944064981
Step: 800 Loss: 1.533307248778922
Step: 900 Loss: 1.513232192888781
Step: 1000 Loss: 1.4671869220568465

```
</div>
Implementing arbitrary research ideas with Keras is straightforward and highly
productive. Imagine trying out 25 ideas per day (20 minutes per experiment on average)!

Keras has been designed to go from idea to results as fast as possible, because we
believe this is
the key to doing great research.

We hope you enjoyed this quick introduction. Let us know what you build with Keras!
