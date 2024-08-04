# Distributed training with Keras 3

**Author:** [Qianli Zhu](https://github.com/qlzh727)<br>
**Date created:** 2023/11/07<br>
**Last modified:** 2023/11/07<br>
**Description:** Complete guide to the distribution API for multi-backend Keras.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distribution.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/distribution.py)



---
## Introduction

The Keras distribution API is a new interface designed to facilitate
distributed deep learning across a variety of backends like JAX, TensorFlow and
PyTorch. This powerful API introduces a suite of tools enabling data and model
parallelism, allowing for efficient scaling of deep learning models on multiple
accelerators and hosts. Whether leveraging the power of GPUs or TPUs, the API
provides a streamlined approach to initializing distributed environments,
defining device meshes, and orchestrating the layout of tensors across
computational resources. Through classes like `DataParallel` and
`ModelParallel`, it abstracts the complexity involved in parallel computation,
making it easier for developers to accelerate their machine learning
workflows.

---
## How it works

The Keras distribution API provides a global programming model that allows
developers to compose applications that operate on tensors in a global context
(as if working with a single device) while
automatically managing distribution across many devices. The API leverages the
underlying framework (e.g. JAX) to distribute the program and tensors according to the
sharding directives through a procedure called single program, multiple data
(SPMD) expansion.

By decoupling the application from sharding directives, the API enables running
the same application on a single device, multiple devices, or even multiple
clients, while preserving its global semantics.

---
## Setup


```python
import os

# The distribution API is only implemented for the JAX backend for now.
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import jax
import numpy as np
from tensorflow import data as tf_data  # For dataset input.
```

---
## `DeviceMesh` and `TensorLayout`

The `keras.distribution.DeviceMesh` class in Keras distribution API represents a cluster of
computational devices configured for distributed computation. It aligns with
similar concepts in [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh) and
[`tf.dtensor.Mesh`](https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh),
where it's used to map the physical devices to a logical mesh structure.

The `TensorLayout` class then specifies how tensors are distributed across the
`DeviceMesh`, detailing the sharding of tensors along specified axes that
correspond to the names of the axes in the `DeviceMesh`.

You can find more detailed concept explainers in the
[TensorFlow DTensor guide](https://www.tensorflow.org/guide/dtensor_overview#dtensors_model_of_distributed_tensors).


```python
# Retrieve the local available gpu devices.
devices = jax.devices("gpu")  # Assume it has 8 local GPUs.

# Define a 2x4 device mesh with data and model parallel axes
mesh = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)

# A 2D layout, which describes how a tensor is distributed across the
# mesh. The layout can be visualized as a 2D grid with "model" as rows and
# "data" as columns, and it is a [4, 2] grid when it mapped to the physical
# devices on the mesh.
layout_2d = keras.distribution.TensorLayout(axes=("model", "data"), device_mesh=mesh)

# A 4D layout which could be used for data parallel of a image input.
replicated_layout_4d = keras.distribution.TensorLayout(
    axes=("data", None, None, None), device_mesh=mesh
)
```

---
## Distribution

The `Distribution` class in Keras serves as a foundational abstract class designed
for developing custom distribution strategies. It encapsulates the core logic
needed to distribute a model's variables, input data, and intermediate
computations across a device mesh. As an end user, you won't have to interact
directly with this class, but its subclasses like `DataParallel` or
`ModelParallel`.

---
## DataParallel

The `DataParallel` class in the Keras distribution API is designed for the
data parallelism strategy in distributed training, where the model weights are
replicated across all devices in the `DeviceMesh`, and each device processes a
portion of the input data.

Here is a sample usage of this class.


```python
# Create DataParallel with list of devices.
# As a shortcut, the devices can be skipped,
# and Keras will detect all local available devices.
# E.g. data_parallel = DataParallel()
data_parallel = keras.distribution.DataParallel(devices=devices)

# Or you can choose to create DataParallel with a 1D `DeviceMesh`.
mesh_1d = keras.distribution.DeviceMesh(
    shape=(8,), axis_names=["data"], devices=devices
)
data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)

inputs = np.random.normal(size=(128, 28, 28, 1))
labels = np.random.normal(size=(128, 10))
dataset = tf_data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

# Set the global distribution.
keras.distribution.set_distribution(data_parallel)

# Note that all the model weights from here on are replicated to
# all the devices of the `DeviceMesh`. This includes the RNG
# state, optimizer states, metrics, etc. The dataset fed into `model.fit` or
# `model.evaluate` will be split evenly on the batch dimension, and sent to
# all the devices. You don't have to do any manual aggregration of losses,
# since all the computation happens in a global context.
inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(units=10, activation="softmax")(y)
model = keras.Model(inputs=inputs, outputs=y)

model.compile(loss="mse")
model.fit(dataset, epochs=3)
model.evaluate(dataset)
```

<div class="k-default-codeblock">
```
Epoch 1/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 8s 30ms/step - loss: 1.0116
Epoch 2/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.9237
Epoch 3/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.8736
 8/8 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - loss: 0.8349

0.842325747013092

```
</div>
---
## `ModelParallel` and `LayoutMap`

`ModelParallel` will be mostly useful when model weights are too large to fit
on a single accelerator. This setting allows you to spit your model weights or
activation tensors across all the devices on the `DeviceMesh`, and enable the
horizontal scaling for the large models.

Unlike the `DataParallel` model where all weights are fully replicated,
the weights layout under `ModelParallel` usually need some customization for
best performances. We introduce `LayoutMap` to let you specify the
`TensorLayout` for any weights and intermediate tensors from global perspective.

`LayoutMap` is a dict-like object that maps a string to `TensorLayout`
instances. It behaves differently from a normal Python dict in that the string
key is treated as a regex when retrieving the value. The class allows you to
define the naming schema of `TensorLayout` and then retrieve the corresponding
`TensorLayout` instance. Typically, the key used to query
is the `variable.path` attribute,  which is the identifier of the variable.
As a shortcut, a tuple or list of axis
names is also allowed when inserting a value, and it will be converted to
`TensorLayout`.

The `LayoutMap` can also optionally contain a `DeviceMesh` to populate the
`TensorLayout.device_mesh` if it is not set. When retrieving a layout with a
key, and if there isn't an exact match, all existing keys in the layout map will
be treated as regex and matched against the input key again. If there are
multiple matches, a `ValueError` is raised. If no matches are found, `None` is
returned.


```python
mesh_2d = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)
layout_map = keras.distribution.LayoutMap(mesh_2d)
# The rule below means that for any weights that match with d1/kernel, it
# will be sharded with model dimensions (4 devices), same for the d1/bias.
# All other weights will be fully replicated.
layout_map["d1/kernel"] = (None, "model")
layout_map["d1/bias"] = ("model",)

# You can also set the layout for the layer output like
layout_map["d2/output"] = ("data", None)

model_parallel = keras.distribution.ModelParallel(layout_map, batch_dim_name="data")

keras.distribution.set_distribution(model_parallel)

inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, use_bias=False, activation="relu", name="d1")(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(units=10, activation="softmax", name="d2")(y)
model = keras.Model(inputs=inputs, outputs=y)

# The data will be sharded across the "data" dimension of the method, which
# has 2 devices.
model.compile(loss="mse")
model.fit(dataset, epochs=3)
model.evaluate(dataset)
```

<div class="k-default-codeblock">
```
Epoch 1/3

/opt/conda/envs/keras-jax/lib/python3.10/site-packages/jax/_src/interpreters/mlir.py:761: UserWarning: Some donated buffers were not usable: ShapedArray(float32[784,50]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"

 8/8 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - loss: 1.0266
Epoch 2/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.9181
Epoch 3/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.8725
 8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.8381  

0.8502610325813293

```
</div>
It is also easy to change the mesh structure to tune the computation between
more data parallel or model parallel. You can do this by adjusting the shape of
the mesh. And no changes are needed for any other code.


```python
full_data_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(8, 1), axis_names=["data", "model"], devices=devices
)
more_data_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(4, 2), axis_names=["data", "model"], devices=devices
)
more_model_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)
full_model_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(1, 8), axis_names=["data", "model"], devices=devices
)
```

### Further reading

1. [JAX Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
2. [JAX sharding module](https://jax.readthedocs.io/en/latest/jax.sharding.html)
3. [TensorFlow Distributed training with DTensors](https://www.tensorflow.org/tutorials/distribute/dtensor_ml_tutorial)
4. [TensorFlow DTensor concepts](https://www.tensorflow.org/guide/dtensor_overview)
5. [Using DTensors with tf.keras](https://www.tensorflow.org/tutorials/distribute/dtensor_keras_tutorial)
