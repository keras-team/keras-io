"""
Title: Migration guide : Tensorflow to Keras 3.0
Author: [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2023/10/23
Last modified: 2023/10/30
Description: Instructions on migrating your TensorFlow code to Keras 3.0
"""

"""
This guide will help you migrate code from TensorFlow 2.x to keras 3.0. The overhead for
the migration is minimal. But once you have migrated you can run Keras workflows on top
of arbitrary frameworks — starting with TensorFlow, JAX, and PyTorch.

Keras 3.0 is also a drop-in replacement for `tf.keras`, with near-full backwards
compatibility with `tf.keras` code when using the TensorFlow backend. In the vast
majority of cases you can just start importing it via `import keras` in place of `from
tensorflow import keras` and your existing code will run with no issue — and generally
with slightly improved performance, thanks to XLA compilation.

Commonly encountered issues and frequently asked questions can be located in
the following links.

[Known
Issues](https://keras.io/keras_core/announcement/#:~:text=Enjoy%20the%20library!-,Known%20issues,-Keras%20Core%20is)

[FAQs](https://keras.io/keras_core/announcement/#:~:text=Frequently%20asked%20questions)
"""

"""
## Setup

First, lets install keras-nightly.

We're going to be using the Tensorflow backend here -- but you can edit the string below
to "tensorflow" or "torch" and hit "Restart runtime", once you have migrated your code
and your code will run just the same!
"""

"""shell
! pip install keras-nightly
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np

"""
## Switching Tensorflow code to Keras 3.0 - Tensorflow backend

Follow these instructions to migrate your existing TensorFlow code to Keras 3.0 and run
it with the TensorFlow backend:

1.   Update imports : replace `from tensorflow import keras` to `import keras`
2.   Update code : replace `tf.keras.*` to `keras.*`



"""

"""
## Migration incompatabilities : `tf.keras` to `keras 3.0`
Keras 3 is a significant milestone in the evolution of the Keras API. It features a
number of cleanups and modernizations that have resulted in a few breaking changes
compared to Keras 2. All APIs that were removed were dropped due to extremely low usage.

The following list provides a comprehensive overview of the breaking changes in Keras 3.
While the majority of these changes are unlikely to affect most users, a small number of
them may require code changes.
"""

"""
### `jit_compile` is set to `True` by default
The default value of the `jit_compile` argument to the Model constructor has been set to
`True` in Keras 3. This means that models will be compiled with Just-In-Time (JIT)
compilation by default.

JIT compilation can improve the performance of some models. However, it may not work with
all TensorFlow operations. If you are using a custom model or layer and you see an
XLA-related error, you may need to set the jit_compile argument to False. Here is a list
of known issues encountered when using xla with tensorflow backend -
https://www.tensorflow.org/xla/known_issues. In addition to these issues, there are some
ops that are not supported by XLA.

The error message you could encounter would be as follows:


```Detected unsupported operations when trying to compile graph
__inference_one_step_on_data_125[] on XLA_CPU_JIT```

The following snippet of code will reproduce the above error:

```
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])

y_train = np.array([["1", "2", "3"], ["4", "5", "6"]])
subclass_model.compile(optimizer="sgd", loss="mse")
subclass_model.fit(x_train, x_train, epochs=1)
```
"""

"""
Here is how you fix it:

set `jit_compile=False` in `model.compile(..., jit_compile=False)`
"""


class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        string_input = tf.strings.as_string(inputs)
        return tf.strings.to_number(string_input)


subclass_model = MyModel()
x_train = np.array([[1, 2, 3], [4, 5, 6]])

y_train = np.array([["1", "2", "3"], ["4", "5", "6"]])
subclass_model.compile(optimizer="sgd", loss="mse", jit_compile=False)
subclass_model.fit(x_train, x_train, epochs=1)

"""
### Saving model in TF SavedModel format
Saving to TF SavedModel format via `model.save()` is no longer supported.


The error message you could encounter would be as follows:

```
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension
for the native Keras format (recommended) or a `.h5` extension. Use
`tf.saved_model.save()` if you want to export a SavedModel for use with
TFLite/TFServing/etc. Received: filepath=saved_model.
```

The following snippet of code will reproduce the above error:

```
sequential_model = keras.Sequential([
    keras.layers.Dense(2)
])
sequential_model.save("saved_model")
```
"""

"""
Here is how you fix it:

use `tf.saved_model.save` instead of `model.save`
"""

sequential_model = keras.Sequential([keras.layers.Dense(2)])
tf.saved_model.save(sequential_model, "saved_model")

"""
### Load a TF SavedModel
Loading a TF SavedModel file via keras.models.load_model() is no longer supported

if you try to use `keras.models.load_model` you would get the following error
```
ValueError: File format not supported: filepath=saved_model. Keras 3 only supports V3
`.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy
SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a
TensorFlow SavedModel as an inference-only layer in Keras 3, use
`keras.layers.TFSMLayer(saved_model, call_endpoint='serving_default')` (note that your
`call_endpoint` might have a different name).
```

The following snippet of code will reproduce the above error:

```
keras.models.load_model("saved_model")
```
"""

"""
Here is how you fix it:

Use `keras.layers.TFSMLayer(filepath, call_endpoint="serving_default")` to reload any TF
SavedModel as a Keras layer
"""

keras.layers.TFSMLayer("saved_model", call_endpoint="serving_default")

"""
### Nested inputs to Model()
Model() can no longer be passed deeply nested inputs/outputs (nested more than 1 level
deep, e.g. lists of lists of tensors)

you would encounter errors as follows:

```
ValueError: When providing `inputs` as a dict, all values in the dict must be
KerasTensors. Received: inputs={'foo': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=foo>, 'bar': {'baz': <KerasTensor shape=(None, 1), dtype=float32,
sparse=None, name=bar>}} including invalid value {'baz': <KerasTensor shape=(None, 1),
dtype=float32, sparse=None, name=bar>} of type <class 'dict'>
```

The following snippet of code will reproduce the above error:

```
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
Here is how you fix it:

"""

inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "baz": keras.Input(shape=(1,), name="bar"),
}
outputs = inputs["foo"] + inputs["baz"]
keras.Model(inputs, outputs)

"""
### TF autograph
In old `tf.keras`, TF autograph is enabled by default on the `call()` method of custom
layers. In Keras 3, it is not. This means you may have to use cond ops if you're using
control flow, or alternatively you can decorate your `call()` method with `@tf.function`.

You would encounted an error as follows
```
OperatorNotAllowedInGraphError: Exception encountered when calling MyCustomLayer.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the
following resolutions to the problem: If you are running in Graph mode, use Eager
execution mode or decorate this function with @tf.function. If you are using AutoGraph,
you can try decorating this function with @tf.function. If that does not work, then you
may be using an unsupported feature or your source code may not be visible to AutoGraph.
See
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/ref
erence/limitations.md#access-to-source-code for more information.
```

The following snippet of code will reproduce the above error:
```
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
model.fit(data, data)
```
"""

"""
Here is how you fix it:

decorate your `call()` method with `@tf.function`
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
model.fit(data, data)

"""
### TF op on a Keras tensor
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

```
input = keras.layers.Input([2, 2, 1])
tf.squeeze(input)
```
"""

"""
Here is how you fix it:

use an equivalent op from `keras.ops`.
"""

input = keras.layers.Input([2, 2, 1])
keras.ops.squeeze(input)

"""
### Multi output model
Multioutput model's `evaluate()` method does not return individual output losses anymore
-> use the metrics argument in compile to track them

When having multiple named outputs (for example named output_a and output_b, old tf.keras
adds <output_a>_loss, <output_b>_loss and so on to metrics. keras_ 3.0 doesn't add them
to metrics and needs to be done them to the output metrics by explicitly providing them
in metrics list of individual outputs.

The following snippet of code will reproduce the above behavior:

```
from keras.layers import Input, Dense, Flatten, Softmax
# A functional model with multiple outputs
inputs = Input(shape=(10,))
x1 = Dense(5, activation='relu')(inputs)
x2 = Dense(5, activation='relu')(x1)
output_1 = Dense(5, activation='softmax', name="output_1")(x1)
output_2 = Dense(5, activation='softmax', name="output_2")(x2)
model = keras.Model(inputs=inputs, outputs=[output_1, output_2])
model.compile(optimizer='adam', loss='categorical_crossentropy')
# dummy data
x_test = np.random.uniform(size=[10, 10])
y_test = np.random.uniform(size=[10, 5])

model.evaluate(x_test, y_test)
```


"""

from keras.layers import Input, Dense, Flatten, Softmax

# A functional model with multiple outputs
inputs = Input(shape=(10,))
x1 = Dense(5, activation="relu")(inputs)
x2 = Dense(5, activation="relu")(x1)
output_1 = Dense(5, activation="softmax", name="output_1")(x1)
output_2 = Dense(5, activation="softmax", name="output_2")(x2)
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
### Variables tracking

The following snippet of code will show that the tf.Variables are not being tracked.

```
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
model.fit(data, data)
# The model does not have any trainable variables
for layer in model.layers:
    print(layer.trainable_variables)
```

`UserWarning: The model does not have any trainable weights.
  warnings.warn("The model does not have any trainable weights.")`
"""

"""
Here is how you fix it:

use `self.add_weight()` method or use a `keras.Variable` instead.

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
model.fit(data, data)
# Verify that the variables are now being tracked
for layer in model.layers:
    print(layer.trainable_variables)

"""
### None entries are not allowed as part of nested (e.g. list/tuples) tensor arguments in
Layer.call(), nor as part of call() return values.

The following snippet of code will reproduce the error.

```
class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
      return  inputs["foo"]

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
Here is how you will fix it:

Replace `None` with a value or Keras Tensor
"""


class CustomLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        output = inputs["foo"]
        return None


layer = CustomLayer()
inputs = {
    "foo": keras.Input(shape=(1,), name="foo"),
    "bar": {
        "baz": keras.Input(shape=(1,), name="bar"),
    },
}
layer(inputs)

"""
### Removed features


1. Symbolic `add_loss()`: Symbolic `add_loss()` is removed (you can still use
`add_loss()` inside the `call()` method of a layer/model).
2. Locally-connected layers: Locally-connected layers are removed due to low usage. To
use locally-connected layers, copy the layer implementation into your own codebase.
3. Kernelized layers: Kernelized layers are removed due to low usage. To use kernelized
layers, copy the layer implementation into your own codebase.
4. Removed layer attributes: Layer attributes `metrics`, `dynamic` are removed
5. RNN layer args: The `constants` and `time_major` arguments in RNN layers are removed.
The `constants` argument was a remnant of Theano and had very low usage. The `time_major`
argument was also infrequently used.
6. reset_metrics argument: The reset_metrics argument is removed from `model. *_on_batch`
methods. This argument had very low usage.
7. RadialConstraint: The RadialConstraint constraint object is removed. This object had
very low usage.
"""

"""
## Switching Tensorflow code to backend agnostic keras 3.0

Keras 3.0 code with the TensorFlow backend will work with native TensorFlow APIs.
However, if you want your code to be backend-agnostic, you will need to replace all of
the `tf.*` API calls with their equivalent Keras APIs.

Follow these instructions to migrate your existing TensorFlow code to Keras 3.0 and run
it with any backend of your choice :

1. Update imports : replace from tensorflow import keras to import keras
2. Update code : replace `tf.keras.*` to `keras.*`. 99% of the tf.keras.* API is
consistent with Keras 3.0. Any differences have been called out in this guide. If an API
is not specifically called out in this guide, that means that the API call is consistent
with tf.keras. If you notice that the same API name results in an error and if it has not
been called out in this document the implementation in keras 3.0 was likely dropped due
to extremely low usage.
3. Replace any `tf.*`, `tf.math*`, `tf.linalg.*`, etc with `keras.ops.*`. Most of the ops
should be consistent with Keras 3.0. If the names are slightly different, they will be
highlighted in this guide. If the same name results in an error and you do not find a
Keras 3.0 equivalent op in this guide, it is likely that the implementation of the op in
keras 3.0 was likely dropped due to extremely low usage.
4. If you are able to replace all the tf ops with keras, you can remove the tensorflow
import and run your code with a backend of your choice.
"""

"""
### Renamed API calls

\begin{array}{|c|c|}
\hline
\textbf{`tf.keras`} & \textbf{`keras 3.0`} \\
\hline
\text{tf.keras.layers.ThresholdedReLU} & \text{keras.layers.ReLU} \\ \hline
\text{tf.keras.constraints.RadialConstraint} & \text{No equivalent} \\ \hline
\text{tf.keras.mixed_precision.Policy} & \text{keras.mixed_precision.DTypePolicy} \\
\hline
\text{tf.keras.mixed_precision.set_global_policy} &
\text{keras.mixed_precision.set_dtype_policy} \\ \hline
\text{tf.keras.mixed_precision.LossScaleOptimizer} &
\text{keras.mixed_precision.LossScaleOptimizer}\\
\text{keras.optimizers.LossScaleOptimizer}  \\ \hline
\text{tf.keras.utils.to_ordinal} & \text{No equivalent} \\ \hline
\text{tf.keras.backend.rnn} & \text{keras.layers.rnn} \\ \hline
\text{tf.linalg.normalize} \\
\text{tf.math.l2_normalize} & \text{keras.utils.normalize} \\ \hline
\hline
\end{array}

"""

"""
### Renamed ops

Replace any `tf.*`, `tf.math*`, `tf.linalg.*`, etc with `keras.ops.*`. Most of the ops
should be consistent with Keras 3.0. If the names are slightly different, they will be
highlighted in this guide. If the same name results in an error and you do not find a
Keras 3.0 equivalent op in this guide, it is likely that the implementation of the op in
keras 3.0 was likely dropped due to extremely low usage.

### Numpy ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.abs} & \text{keras.ops.absolute} \\ \hline
\text{tf.reduce\_all} & \text{keras.ops.all} \\ \hline
\text{tf.reduce\_max} & \text{keras.ops.amax} \\ \hline
\text{tf.reduce\_min} & \text{keras.ops.amin} \\ \hline
\text{tf.reduce\_any} & \text{keras.ops.any} \\ \hline
\text{tf.concat} & \text{keras.ops.append, keras.ops.concatenate,} \\
 & \text{keras.ops.hstack, keras.ops.vstack} \\ \hline
\text{tf.range} & \text{keras.ops.arange} \\ \hline
\text{tf.acos} & \text{keras.ops.arccos} \\ \hline
\text{tf.asin} & \text{keras.ops.arcsin} \\ \hline
\text{tf.asinh} & \text{keras.ops.arcsinh} \\ \hline
\text{tf.atan} & \text{keras.ops.arctan} \\ \hline
\text{tf.atan2} & \text{keras.ops.arctan2} \\ \hline
\text{tf.atanh} & \text{keras.ops.arctanh} \\ \hline
\text{tf.convert_to_tensor} & \text{keras.ops.array} \\ \hline
\text{tf.reduce_mean} & \text{keras.ops.average} \\ \hline
\text{tf.clip_by_value} & \text{keras.ops.clip} \\ \hline
\text{tf.math.conj} & \text{keras.ops.conjugate, keras.ops.conj} \\ \hline
\text{tf.identity} & \text{keras.ops.copy} \\ \hline
\text{tf.linalg.diag_part} & \text{keras.ops.diagonal} \\ \hline
\text{tf.tensordot} & \text{keras.ops.dot} \\ \hline
\text{tf.constant} & \text{keras.ops.empty} \\ \hline
\text{tf.reverse} & \text{keras.ops.flip} \\ \hline
\text{tf.fill} & \text{keras.ops.full} \\ \hline
\text{tf.ones_like} & \text{keras.ops.full_like} \\ \hline
\text{tf.gather} & \text{keras.ops.take} \\ \hline
\text{tf.eye} & \text{keras.ops.identity} \\ \hline
\text{tf.math.is_close} & \text{keras.ops.isclose} \\ \hline
\text{tf.math.is_finite} & \text{keras.ops.isfinite} \\ \hline
\text{tf.math.is_inf} & \text{keras.ops.isinf} \\ \hline
\text{tf.math.is_nan} & \text{keras.ops.isnan} \\ \hline
\text{tf.math.log} & \text{keras.ops.log10} \\ \hline
\text{tf.math.log} & \text{keras.ops.log2} \\ \hline
\text{tf.reduce_logsumexp} & \text{keras.ops.logaddexp} \\ \hline
\text{tf.reduce_max} & \text{keras.ops.max} \\ \hline
\text{tf.reduce_mean} & \text{keras.ops.mean} \\ \hline
\text{tf.reduce_min} & \text{keras.ops.min} \\ \hline
\text{tf.transpose} & \text{keras.ops.moveaxis} \\ \hline
\text{tf.rank} & \text{keras.ops.ndim} \\ \hline
\text{tf.linalg.matmul} & \text{keras.ops.outer} \\ \hline
\text{tf.math.pow} & \text{keras.ops.power} \\ \hline
\text{tf.reduce_prod} & \text{keras.ops.prod} \\ \hline
\text{tf.reshape}, \text{tf.keras.layers.Flatten} & \text{keras.ops.ravel} \\ \hline
\text{tf.math.reduce_std} & \text{keras.ops.std} \\ \hline
\text{tf.reduce_sum} & \text{keras.ops.sum} \\ \hline
\text{tf.transpose} & \text{keras.ops.swapaxes} \\ \hline
\text{tf.gather} & \text{keras.ops.take} \\ \hline
\text{tf.gather_nd} & \text{keras.ops.take_along_axis} \\ \hline
\text{tf.linalg.matmul} & \text{keras.ops.tensordot} \\ \hline
\text{tf.math.divide} & \text{keras.ops.true_divide} \\ \hline
\text{tf.math.reduce_variance} & \text{keras.ops.var} \\ \hline
\text{tf.linalg.tensordot} & \text{keras.ops.vdot} \\ \hline
\text{tf.where} & \text{keras.ops.where, keras.ops.nonzero} \\
\hline
\end{array}


#### NN ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.nn.sigmoid_cross_entropy_with_logits} &
\text{keras.ops.binary_crossentropy(target, output, from_logits=False)} \\ \hline
\text{tf.nn.sparse_softmax_cross_entropy_with_logits} &
\text{keras.ops.categorical_crossentropy(target, output, from_logits=False, axis=-1)} \\
\hline
\text{tf.nn.conv1d} \\ \text{tf.nn.conv2d} \\
\text{tf.nn.conv3d} \\ \text{tf.nn.convolution} \\
& \text{keras.ops.conv(inputs, kernel, strides=1, padding="valid",} \\
& \text{data_format=None, dilation_rate=1)} \\ \hline
\text{tf.nn.conv_transpose} \\ \text{tf.nn.conv1d_transpose} \\
\text{tf.nn.conv2d_transpose} \\ \text{tf.nn.conv3d_transpose} \\
& \text{keras.ops.conv_transpose(inputs, kernel, strides, padding="valid",} \\ \hline
& \text{output_padding=None, data_format=None, dilation_rate=1)} \\
\text{tf.nn.depthwise_conv2d} & \text{keras.ops.depthwise_conv(inputs, kernel,
strides=1,} \\
& \text{padding="valid", data_format=None, dilation_rate=1)} \\ \hline
\text{tf.nn.separable_conv2d} & \text{keras.ops.separable_conv(inputs, depthwise_kernel,}
\\
& \text{pointwise_kernel, strides=1, padding="valid", data_format=None,} \\
& \text{dilation_rate=1)} \\ \hline
\text{tf.nn.sparse_softmax_cross_entropy_with_logits} &
\text{keras.ops.sparse_categorical_crossentropy(target, output,} \\
& \text{from_logits=False, axis=-1)} \\ \hline
\text{tf.nn.batch_normalization} & \text{keras.layers.BatchNormalization} \\ \hline
\text{tf.nn.dropout} & \text{keras.layers.Dropout} \\ \hline
\text{tf.nn.embedding_lookup} & \text{tf.nn.embedding_lookup_sparse} \\ \hline
\text{tf.nn.l2_normalize} & \text{keras.utils.normalize} \\
\hline
\end{array}


#### Core ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{x.numpy} & \text{keras.ops.convert_to_numpy} \\ \hline
\text{NA} & \text{keras.ops.extract_sequences(x, sequence_length, sequence_stride)} \\
\hline
\text{tf.scatter_nd_update} & \text{keras.ops.scatter_update} \\ \hline
\text{tf.tensor_scatter_nd_update} & \text{keras.ops.slice_update} \\
\hline
\end{array}


#### Image ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.keras.preprocessing.image.apply_affine_transform} &
\text{keras.ops.image.affine_transform} \\
\hline
\end{array}

#### FFT ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.signal.fft2d} & \text{keras.ops.fft2} \\ \hline
\text{tf.signal.inverse_stft} & \text{keras.ops.istft} \\ \hline
\hline
\end{array}

#### Random ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.random.normal} & \text{keras.random.normal} \\ \hline
\text{tf.random.categorical} & \text{keras.random.categorical} \\ \hline
\text{tf.random.uniform} & \text{keras.random.uniform} \\ \hline
\text{tf.random.uniform} & \text{keras.random.randint} \\ \hline
\text{tf.random.truncated_normal} & \text{keras.random.truncated_normal} \\ \hline
\text{tf.nn.dropout} & \text{keras.layers.Dropout} \\ \hline
\text{tf.random.shuffle} & \text{keras.utils.shuffle} \\ \hline
\text{Rest of tf.random.* ops} & \text{Not yet supported} \\
\hline
\end{array}

#### MISC ops

\begin{array}{|c|c|}
\hline
\textbf{TensorFlow} & \textbf{Keras 3.0} \\
\hline
\text{tf.lookup.*} & \text{Not yet supported} \\ \hline
\text{tf.quantization.*} & \text{Not yet supported} \\ \hline
\text{tf.ragged.*} & \text{Not yet supported} \\ \hline
\text{tf.sparse.*} & \text{All of the ops supported above also support sparse inputs for
TensorFlow backend} \\
\hline
\end{array}
"""

"""
## Additional developer guides

This wraps up the migration guide overview. We hope that this guide has been helpful in
successfully transitioning your code from TensorFlow to Keras 3.0. Explore a variety of
developer guides to help you begin, craft custom training loops, and establish
distributed training setups. Take a look!

* [Getting started with Keras
Core](https://keras.io/keras_core/guides/getting_started_with_keras_core/)
* [The Functional API](https://keras.io/keras_core/guides/functional_api/)
* [The Sequential model](https://keras.io/keras_core/guides/sequential_model/)
* [Making new layers & models via
subclassing](https://keras.io/keras_core/guides/making_new_layers_and_models_via_subclassing/)
subclassing](https://keras.io/keras_core/guides/making_new_layers_and_models_via_subclassing/)
* [Training & evaluation with the built-in
methods](https://keras.io/keras_core/guides/training_with_built_in_methods/)
* [Writing your own
callbacks](https://keras.io/keras_core/guides/writing_your_own_callbacks/)
* [Transfer learning](https://keras.io/keras_core/guides/transfer_learning/)
* [Understanding masking &
padding](https://keras.io/keras_core/guides/understanding_masking_and_padding/)
* [Customizing what happens in fit() with
TensorFlow](https://keras.io/keras_core/guides/custom_train_step_in_tensorflow/)
* [Customizing what happens in fit() with
JAX](https://keras.io/keras_core/guides/custom_train_step_in_jax/)
* [Customizing what happens in fit() with
PyTorch](https://keras.io/keras_core/guides/custom_train_step_in_torch/)
* [Writing a custom training loop with
TensorFlow](https://keras.io/keras_core/guides/writing_a_custom_training_loop_in_tensorflow/)
TensorFlow](https://keras.io/keras_core/guides/writing_a_custom_training_loop_in_tensorflow/)
* [Writing a custom training loop with
JAX](https://keras.io/keras_core/guides/writing_a_custom_training_loop_in_jax/)
* [Writing a custom training loop with
PyTorch](https://keras.io/keras_core/guides/writing_a_custom_training_loop_in_torch/)
* [Distributed training with
TensorFlow](https://keras.io/keras_core/guides/distributed_training_with_tensorflow/)
* [Distributed training with
JAX](https://keras.io/keras_core/guides/distributed_training_with_jax/)
* [Distributed training with
PyTorch](https://keras.io/keras_core/guides/distributed_training_with_torch/)

"""
