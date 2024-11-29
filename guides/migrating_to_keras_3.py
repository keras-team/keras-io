"""
Title: Migrating Keras 2 code to multi-backend Keras 3
Author: [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2023/10/23
Last modified: 2023/10/30
Description: Instructions & troubleshooting for migrating your Keras 2 code to multi-backend Keras 3.
Accelerator: None
"""

"""
This guide will help you migrate TensorFlow-only Keras 2 code to multi-backend Keras
3 code. The overhead for the migration is minimal. Once you have migrated,
you can run Keras workflows on top of either JAX, TensorFlow, or PyTorch.

This guide has two parts:

1. Migrating your legacy Keras 2 code to Keras 3, running on top of the TensorFlow backend.
This is generally very easy, though there are minor issues to be mindful of, that we will go over
in detail.
2. Further migrating your Keras 3 + TensorFlow code to multi-backend Keras 3, so that it can run on
JAX and PyTorch.

Let's get started.
"""

"""
## Setup

First, lets install `keras-nightly`.

This example uses the TensorFlow backend (`os.environ["KERAS_BACKEND"] = "tensorflow"`).
After you've migrated your code, you can change the `"tensorflow"` string to `"jax"` or `"torch"`
and click "Restart runtime" in Colab, and your code will run on the JAX or PyTorch backend.
"""

"""shell
pip install -q keras-nightly
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np

"""
## Going from Keras 2 to Keras 3 with the TensorFlow backend

First, replace your imports:

1. Replace `from tensorflow import keras` to `import keras`
2. Replace `from tensorflow.keras import xyz` (e.g. `from tensorflow.keras import layers`)
to `from keras import xyz` (e.g. `from keras import layers`)
3. Replace `tf.keras.*` to `keras.*`

Next, start running your tests. Most of the time, your code will execute on Keras 3 just fine.
All issues you might encounter are detailed below, with their fixes.
"""

"""
### `jit_compile` is set to `True` by default on GPU.

The default value of the `jit_compile` argument to the `Model` constructor has been set to
`True` on GPU in Keras 3. This means that models will be compiled with Just-In-Time (JIT)
compilation by default on GPU.

JIT compilation can improve the performance of some models. However, it may not work with
all TensorFlow operations. If you are using a custom model or layer and you see an
XLA-related error, you may need to set the `jit_compile` argument to `False`. Here is a list
of [known issues](https://www.tensorflow.org/xla/known_issues) encountered when
using XLA with TensorFlow. In addition to these issues, there are some
ops that are not supported by XLA.

The error message you could encounter would be as follows:

```
Detected unsupported operations when trying to compile graph
__inference_one_step_on_data_125[] on XLA_GPU_JIT
```

For example, the following snippet of code will reproduce the above error:

```python
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])
subclass_model.compile(optimizer="sgd", loss="mse")
subclass_model.predict(x_train)
```
"""

"""
**How to fix it:** set `jit_compile=False` in `model.compile(..., jit_compile=False)`,
or set the `jit_compile` attribute to `False`, like this:
"""


class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # tf.strings ops aren't support by XLA
        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])
subclass_model.jit_compile = False
subclass_model.predict(x_train)

"""
### Saving a model in the TF SavedModel format

Saving to the TF SavedModel format via `model.save()` is no longer supported in Keras 3.

The error message you could encounter would be as follows:

```
>>> model.save("mymodel")
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension
for the native Keras format (recommended) or a `.h5` extension. Use
`model.export(filepath)` if you want to export a SavedModel for use with
TFLite/TFServing/etc. Received: filepath=saved_model.
```

The following snippet of code will reproduce the above error:

```python
sequential_model = keras.Sequential([
    keras.layers.Dense(2)
])
sequential_model.save("saved_model")
```
"""

"""
**How to fix it:** use `model.export(filepath)` instead of `model.save(filepath)`
"""

sequential_model = keras.Sequential([keras.layers.Dense(2)])
sequential_model(np.random.rand(3, 5))
sequential_model.export("saved_model")

"""
### Loading a TF SavedModel

Loading a TF SavedModel file via `keras.models.load_model()` is no longer supported
If you try to use `keras.models.load_model()` with a TF SavedModel, you will get the following error:

```python
ValueError: File format not supported: filepath=saved_model. Keras 3 only supports V3
`.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy
SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a
TensorFlow SavedModel as an inference-only layer in Keras 3, use
`keras.layers.TFSMLayer(saved_model, call_endpoint='serving_default')` (note that your
`call_endpoint` might have a different name).
```

The following snippet of code will reproduce the above error:

```python
keras.models.load_model("saved_model")
```
"""

"""
**How to fix it:** Use `keras.layers.TFSMLayer(filepath, call_endpoint="serving_default")` to reload a TF
SavedModel as a Keras layer. This is not limited to SavedModels that originate from Keras -- it will work
with any SavedModel, e.g. TF-Hub models.
"""

keras.layers.TFSMLayer("saved_model", call_endpoint="serving_default")

"""
### Using deeply nested inputs in Functional Models

`Model()` can no longer be passed deeply nested inputs/outputs (nested more than 1 level
deep, e.g. lists of lists of tensors).

You would encounter errors as follows:

```
ValueError: When providing `inputs` as a dict, all values in the dict must be
KerasTensors. Received: inputs={'foo': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=foo>, 'bar': {'baz': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=bar>}} including invalid value {'baz': <KerasTensor shape=(None, 1),
dtype=float32, sparse=None, name=bar>} of type <class 'dict'>
```

The following snippet of code will reproduce the above error:

```python
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": keras.Input(shape=(1,), name="bar"),
    },
}
outputs = inputs["foo"] + inputs["bar"]["baz"]
keras.Model(inputs, outputs)
```

"""

"""
**How to fix it:** replace nested input with either dicts, lists, and tuples
of input tensors.
"""

inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": keras.Input(shape=(1,), name="bar"),
}
outputs = inputs["foo"] + inputs["bar"]
keras.Model(inputs, outputs)

"""
### TF autograph

In Keras 2, TF autograph is enabled by default on the `call()` method of custom
layers. In Keras 3, it is not. This means you may have to use cond ops if you're using
control flow, or alternatively you can decorate your `call()` method with `@tf.function`.

You would encounter an error as follows:
```
OperatorNotAllowedInGraphError: Exception encountered when calling MyCustomLayer.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the
following resolutions to the problem: If you are running in Graph mode, use Eager
execution mode or decorate this function with @tf.function. If you are using AutoGraph,
you can try decorating this function with @tf.function. If that does not work, then you
may be using an unsupported feature or your source code may not be visible to AutoGraph.
Here is a [link for more information](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/ref
erence/limitations.md#access-to-source-code).
```

The following snippet of code will reproduce the above error:

```python
class MyCustomLayer(keras.layers.Layer):

  def call(self, inputs):
    if tf.random.uniform(()) > 0.5:
      return inputs * 2
    else:
      return inputs / 2


layer = MyCustomLayer()
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
```
"""

"""
**How to fix it:** decorate your `call()` method with `@tf.function`
"""


class MyCustomLayer(keras.layers.Layer):
    @tf.function()
    def call(self, inputs):
        if tf.random.uniform(()) > 0.5:
            return inputs * 2
        else:
            return inputs / 2


layer = MyCustomLayer()
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)

"""
### Calling TF ops with a `KerasTensor`

Using a TF op on a Keras tensor during functional model construction is disallowed: "A
KerasTensor cannot be used as input to a TensorFlow function".

The error you would encounter would be as follows:

```
ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor
is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional
models or Keras Functions. You can only use it as input to a Keras layer or a Keras
operation (from the namespaces `keras.layers` and `keras.operations`).
```

The following snippet of code will reproduce the error:

```python
input = keras.layers.Input([2, 2, 1])
tf.squeeze(input)
```
"""

"""
**How to fix it:** use an equivalent op from `keras.ops`.
"""

input = keras.layers.Input([2, 2, 1])
keras.ops.squeeze(input)

"""
### Multi-output model `evaluate()`

The `evaluate()` method of a multi-output model no longer returns individual output
losses separately. Instead, you should utilize the `metrics` argument in the `compile()`
method to keep track of these losses.


When dealing with multiple named outputs, such as output_a and output_b, the legacy
`tf.keras` would include <output_a>_loss, <output_b>_loss, and similar entries in
metrics. However, in keras 3.0, these entries are not automatically added to metrics.
They must be explicitly provided in the metrics list for each individual output.

The following snippet of code will reproduce the above behavior:

```python
from keras import layers
# A functional model with multiple outputs
inputs = layers.Input(shape=(10,))
x1 = layers.Dense(5, activation='relu')(inputs)
x2 = layers.Dense(5, activation='relu')(x1)
output_1 = layers.Dense(5, activation='softmax', name="output_1")(x1)
output_2 = layers.Dense(5, activation='softmax', name="output_2")(x2)
model = keras.Model(inputs=inputs, outputs=[output_1, output_2])
model.compile(optimizer='adam', loss='categorical_crossentropy')
# dummy data
x_test = np.random.uniform(size=[10, 10])
y_test = np.random.uniform(size=[10, 5])

model.evaluate(x_test, y_test)
```
"""

from keras import layers

# A functional model with multiple outputs
inputs = layers.Input(shape=(10,))
x1 = layers.Dense(5, activation="relu")(inputs)
x2 = layers.Dense(5, activation="relu")(x1)
output_1 = layers.Dense(5, activation="softmax", name="output_1")(x1)
output_2 = layers.Dense(5, activation="softmax", name="output_2")(x2)
# dummy data
x_test = np.random.uniform(size=[10, 10])
y_test = np.random.uniform(size=[10, 5])
multi_output_model = keras.Model(inputs=inputs, outputs=[output_1, output_2])
multi_output_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_crossentropy", "categorical_crossentropy"],
)
multi_output_model.evaluate(x_test, y_test)


"""
### TensorFlow variables tracking

Setting a `tf.Variable` as an attribute of a Keras 3 layer or model will not automatically
track the variable, unlike in Keras 2. The following snippet of code will show that the `tf.Variables`
are not being tracked.

```python
class MyCustomLayer(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = tf.Variable(initial_value=tf.zeros([input_dim, self.units]))
        self.b = tf.Variable(initial_value=tf.zeros([self.units,]))

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b


layer = MyCustomLayer(3)
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
# The model does not have any trainable variables
for layer in model.layers:
    print(layer.trainable_variables)
```

You will see the following warning:

```
UserWarning: The model does not have any trainable weights.
  warnings.warn("The model does not have any trainable weights.")
```

**How to fix it:** use `self.add_weight()` method or opt for a `keras.Variable` instead. If you
are currently using `tf.variable`, you can switch to `keras.Variable`.
"""


class MyCustomLayer(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=[input_dim, self.units],
            initializer="zeros",
        )
        self.b = self.add_weight(
            shape=[
                self.units,
            ],
            initializer="zeros",
        )

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b


layer = MyCustomLayer(3)
data = np.random.uniform(size=[3, 3])
model = keras.models.Sequential([layer])
model.compile(optimizer="adam", loss="mse")
model.predict(data)
# Verify that the variables are now being tracked
for layer in model.layers:
    print(layer.trainable_variables)

"""
### `None` entries in nested `call()` arguments

`None` entries are not allowed as part of nested (e.g. list/tuples) tensor
arguments in `Layer.call()`, nor as part of `call()`'s nested return values.

If the `None` in the argument is intentional and serves a specific purpose,
ensure that the argument is optional and structure it as a separate parameter.
For example, consider defining the `call` method with optional argument.

The following snippet of code will reproduce the error.

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        foo = inputs["foo"]
        baz = inputs["bar"]["baz"]
        if baz is not None:
            return foo + baz
        return foo

layer = CustomLayer()
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": None,
    },
}
layer(inputs)
```
"""

"""
**How to fix it:**

**Solution 1:** Replace `None` with a value, like this:
"""


class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        foo = inputs["foo"]
        baz = inputs["bar"]["baz"]
        return foo + baz


layer = CustomLayer()
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": keras.Input(shape=(1,), name="bar"),
    },
}
layer(inputs)


"""
**Solution 2:** Define the call method with an optional argument.
Here is an example of this fix:
"""


class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, foo, baz=None):
        if baz is not None:
            return foo + baz
        return foo


layer = CustomLayer()
foo = keras.Input(shape=(1,), name="foo")
baz = None
layer(foo, baz=baz)

"""
### State-building issues

Keras 3 is significantly stricter than Keras 2 about when state (e.g. numerical weight variables)
can be created. Keras 3 wants all state to be created before the model can be trained. This is a requirement
for using JAX (whereas TensorFlow was very lenient about state creation timing).

Keras layers should create their state either in their constructor (`__init__()` method) or in their `build()` method.
They should avoid creating state in `call()`.

If you ignore this recommendation and create state in `call()`
anyway (e.g. by calling a previously unbuilt layer), then Keras will attempt to build the layer automatically
by calling the `call()` method on symbolic inputs before training.
However, this attempt at automatic state creation may fail in certain cases.
This will cause an error that looks like like this:

```
Layer 'frame_position_embedding' looks like it has unbuilt state,
but Keras is not able to trace the layer `call()` in order to build it automatically.
Possible causes:
1. The `call()` method of your layer may be crashing.
Try to `__call__()` the layer eagerly on some test input first to see if it works.
E.g. `x = np.random.random((3, 4)); y = layer(x)`
2. If the `call()` method is correct, then you may need to implement
the `def build(self, input_shape)` method on your layer.
It should create all variables used by the layer
(e.g. by calling `layer.build()` on all its children layers).
```

You could reproduce this error with the following layer, when used with the JAX backend:

```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
```

**How to fix it:** Do exactly what the error message asks. First, try to run the layer eagerly
to see if the `call()` method is in fact correct (note: if it was working in Keras 2, then it is correct
and does not need to be changed). If it is indeed correct, then you should implement a `build(self, input_shape)`
method that creates all of the layer's state, including the state of sublayers. Here's the fix as applied for the layer above
(note the `build()` method):

```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
```
"""


"""
### Removed features

A small number of legacy features with very low usage were removed from Keras 3 as a cleanup measure:

* `keras.layers.ThresholdedReLU` is removed. Instead, you can simply use the `ReLU` layer
with the argument `threshold`.
* Symbolic `Layer.add_loss()`: Symbolic `add_loss()` is removed (you can still use
`add_loss()` inside the `call()` method of a layer/model).
* Locally connected layers (`LocallyConnected1D`, `LocallyConnected2D`
are removed due to very low usage. To
use locally connected layers, copy the layer implementation into your own codebase.
* `keras.layers.experimental.RandomFourierFeatures` is removed due to very low usage.
To use it, copy the layer implementation into your own codebase.
* Removed layer attributes: Layer attributes `metrics`, `dynamic` are removed. `metrics` is still
available on the `Model` class.
* The `constants` and `time_major` arguments in RNN layers are removed.
The `constants` argument was a remnant of Theano and had very low usage. The `time_major`
argument also had very low usage.
* `reset_metrics` argument: The `reset_metrics` argument is removed from `model.*_on_batch()`
methods. This argument had very low usage.
* The `keras.constraints.RadialConstraint` object is removed. This object had very low usage.
"""

"""
## Transitioning to backend-agnostic Keras 3

Keras 3 code with the TensorFlow backend will work with native TensorFlow APIs.
However, if you want your code to be backend-agnostic, you will need to:

- Replace all of the `tf.*` API calls with their equivalent Keras APIs.
- Convert your custom `train_step`/`test_step` methods to a multi-framework
implementation.
- Make sure you're using stateless `keras.random` ops correctly in your layers.

Let's go over each point in detail.

### Switching to Keras ops

In many cases, this is the only thing you need to do to start being able to run
your custom layers and metrics with JAX and PyTorch:
replace any `tf.*`, `tf.math*`, `tf.linalg.*`, etc. with `keras.ops.*`. Most TF ops
should be consistent with Keras 3. If the names different, they will be
highlighted in this guide.

#### NumPy ops

Keras implements the NumPy API as part of `keras.ops`.

The table below only lists a small subset of TensorFlow and Keras ops; ops not listed
are usually named the same in both frameworks (e.g. `reshape`, `matmul`, `cast`, etc.)

| TensorFlow                                 | Keras 3.0                                 |
|--------------------------------------------|-------------------------------------------|
| `tf.abs`                                   | `keras.ops.absolute`                      |
| `tf.reduce_all`                            | `keras.ops.all`                           |
| `tf.reduce_max`                            | `keras.ops.amax`                          |
| `tf.reduce_min`                            | `keras.ops.amin`                          |
| `tf.reduce_any`                            | `keras.ops.any`                           |
| `tf.concat`                                | `keras.ops.concatenate`                   |
| `tf.range`                                 | `keras.ops.arange`                        |
| `tf.acos`                                  | `keras.ops.arccos`                        |
| `tf.asin`                                  | `keras.ops.arcsin`                        |
| `tf.asinh`                                 | `keras.ops.arcsinh`                       |
| `tf.atan`                                  | `keras.ops.arctan`                        |
| `tf.atan2`                                 | `keras.ops.arctan2`                       |
| `tf.atanh`                                 | `keras.ops.arctanh`                       |
| `tf.convert_to_tensor`                     | `keras.ops.convert_to_tensor`             |
| `tf.reduce_mean`                           | `keras.ops.mean`                          |
| `tf.clip_by_value`                         | `keras.ops.clip`                          |
| `tf.math.conj`                             | `keras.ops.conjugate`                     |
| `tf.linalg.diag_part`                      | `keras.ops.diagonal`                      |
| `tf.reverse`                               | `keras.ops.flip`                          |
| `tf.gather`                                | `keras.ops.take`                          |
| `tf.math.is_finite`                        | `keras.ops.isfinite`                      |
| `tf.math.is_inf`                           | `keras.ops.isinf`                         |
| `tf.math.is_nan`                           | `keras.ops.isnan`                         |
| `tf.reduce_max`                            | `keras.ops.max`                           |
| `tf.reduce_mean`                           | `keras.ops.mean`                          |
| `tf.reduce_min`                            | `keras.ops.min`                           |
| `tf.rank`                                  | `keras.ops.ndim`                          |
| `tf.math.pow`                              | `keras.ops.power`                         |
| `tf.reduce_prod`                           | `keras.ops.prod`                          |
| `tf.math.reduce_std`                       | `keras.ops.std`                           |
| `tf.reduce_sum`                            | `keras.ops.sum`                           |
| `tf.gather`                                | `keras.ops.take`                          |
| `tf.gather_nd`                             | `keras.ops.take_along_axis`               |
| `tf.math.reduce_variance`                  | `keras.ops.var`                           |


#### Others ops

| TensorFlow                                         | Keras 3.0                                                         |
|----------------------------------------------------|-------------------------------------------------------------------|
| `tf.nn.sigmoid_cross_entropy_with_logits`          | `keras.ops.binary_crossentropy` (mind the `from_logits` argument) |
| `tf.nn.sparse_softmax_cross_entropy_with_logits`   | `keras.ops.sparse_categorical_crossentropy` (mind the `from_logits` argument)|
| `tf.nn.sparse_softmax_cross_entropy_with_logits`   | `keras.ops.categorical_crossentropy(target, output, from_logits=False, axis=-1)`|
| `tf.nn.conv1d`, `tf.nn.conv2d`, `tf.nn.conv3d`, `tf.nn.convolution` | `keras.ops.conv`                                 |
| `tf.nn.conv_transpose`, `tf.nn.conv1d_transpose`, `tf.nn.conv2d_transpose`, `tf.nn.conv3d_transpose` | `keras.ops.conv_transpose` |
| `tf.nn.depthwise_conv2d`                           | `keras.ops.depthwise_conv`                                        |
| `tf.nn.separable_conv2d`                           | `keras.ops.separable_conv`                                        |
| `tf.nn.batch_normalization`                        |  No direct equivalent; use `keras.layers.BatchNormalization`      |
| `tf.nn.dropout`                                    | `keras.random.dropout`                                            |
| `tf.nn.embedding_lookup`                           | `keras.ops.take`                                                  |
| `tf.nn.l2_normalize`                               | `keras.utils.normalize` (not an op)                               |
| `x.numpy`                                          | `keras.ops.convert_to_numpy`                                      |
| `tf.scatter_nd_update`                             | `keras.ops.scatter_update`                                        |
| `tf.tensor_scatter_nd_update`                      | `keras.ops.slice_update`                                          |
| `tf.signal.fft2d`                                  | `keras.ops.fft2`                                                  |
| `tf.signal.inverse_stft`                           | `keras.ops.istft`                                                 |
| `tf.image.crop_to_bounding_box`                    | `keras.ops.image.crop_images`                                     |
| `tf.image.pad_to_bounding_box`                     | `keras.ops.image.pad_images`                                      |

"""

"""
### Custom `train_step()` methods

Your models may include a custom `train_step()` or `test_step()` method, which rely
on TensorFlow-only APIs -- for instance, your `train_step()` method may leverage TensorFlow's `tf.GradientTape`.
To convert such models to run on JAX or PyTorch, you will have a write a different `train_step()` implementation
for each backend you want to support.

In some cases, you might be able to simply override the `Model.compute_loss()` method and make it fully backend-agnostic,
instead of overriding `train_step()`. Here's an example of a layer with a custom `compute_loss()` method which works
across JAX, TensorFlow, and PyTorch:
"""


class MyModel(keras.Model):
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        loss = keras.ops.sum(keras.losses.mean_squared_error(y, y_pred, sample_weight))
        return loss


"""
If you need to modify the optimization mechanism itself, beyond the loss computation,
then you will need to override `train_step()`, and implement one `train_step` method per backend, like below.

See the following guides for details on how each backend should be handled:

- [Customizing what happens in `fit()` with JAX](https://keras.io/guides/custom_train_step_in_jax/)
- [Customizing what happens in `fit()` with TensorFlow](https://keras.io/guides/custom_train_step_in_tensorflow/)
- [Customizing what happens in `fit()` with PyTorch](https://keras.io/guides/custom_train_step_in_torch/)
"""


class MyModel(keras.Model):
    def train_step(self, *args, **kwargs):
        if keras.backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _jax_train_step(self, state, data):
        pass  # See guide: keras.io/guides/custom_train_step_in_jax/

    def _tensorflow_train_step(self, data):
        pass  # See guide: keras.io/guides/custom_train_step_in_tensorflow/

    def _torch_train_step(self, data):
        pass  # See guide: keras.io/guides/custom_train_step_in_torch/


"""
### RNG-using layers

Keras 3 has a new `keras.random` namespace, containing:

- `keras.random.normal`
- `keras.random.uniform`
- `keras.random.shuffle`
- etc.

These operations are **stateless**, which means that if you pass a `seed`
argument, they will return the same result every time. Like this:
"""

print(keras.random.normal(shape=(), seed=123))
print(keras.random.normal(shape=(), seed=123))

"""
Crucially, this differs from the behavior of stateful `tf.random` ops:
"""

print(tf.random.normal(shape=(), seed=123))
print(tf.random.normal(shape=(), seed=123))

"""
When you write a RNG-using layer, such as a custom dropout layer, you are
going to want to use a different seed value at layer call. However, you cannot
just increment a Python integer and pass it, because while this would work fine
when executed eagerly, it would not work as expected when using compilation
(which is available with JAX, TensorFlow, and PyTorch). When compiling the layer,
the first Python integer seed value seen by the layer would be hardcoded into the
compiled graph.

To address this, you should pass as the `seed` argument an instance of a
stateful `keras.random.SeedGenerator` object, like this:
"""

seed_generator = keras.random.SeedGenerator(1337)
print(keras.random.normal(shape=(), seed=seed_generator))
print(keras.random.normal(shape=(), seed=seed_generator))


"""
So when writing a RNG using layer, you would use the following pattern:
"""


class RandomNoiseLayer(keras.layers.Layer):
    def __init__(self, noise_rate, **kwargs):
        super().__init__(**kwargs)
        self.noise_rate = noise_rate
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        noise = keras.random.uniform(
            minval=0, maxval=self.noise_rate, seed=self.seed_generator
        )
        return inputs + noise


"""
Such a layer is safe to use in any setting -- in eager execution or in a compiled model. Each
layer call will be using a different seed value, as expected.
"""
