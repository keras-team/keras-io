# Distributed training with Keras

**Authors:** [Qianli Zhu](https://github.com/qlzh727), [Suhana](https://github.com/buildwithsuhana)  
**Date created:** 2023/11/07  
**Last modified:** 2026/08/13  
**Description:** Complete guide to the distribution API for multi-backend Keras (JAX & PyTorch).

---
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distribution.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/distribution.py)

---
## Introduction

The Keras distribution API is a unified interface designed to facilitate
distributed deep learning across backends. Currently, it supports **JAX** and **PyTorch**,
providing a streamlined approach to scaling models on multiple accelerators and hosts.
(Note: TensorFlow support is currently in development).

This API introduces a suite of tools enabling data and model parallelism through classes
like `DataParallel` and `ModelParallel`. It abstracts the complexity involved in
parallel computation, allowing you to write backend-agnostic distributed training code
that leverages the native strengths of each backend—such as XLA's SPMD in JAX and
DDP/DTensor in PyTorch.


---
## Architectural Differences: JAX vs PyTorch

Before diving into the API, it is essential to understand that JAX and PyTorch have
fundamentally different distributed architectures. This difference shapes every
design decision in Keras distribution.

### JAX: Single-Process, Compiler-Driven
JAX uses a single-process, multi-device model. One Python process controls all devices.
The XLA compiler analyzes the full computation graph, partitions it across devices,
and inserts the necessary communication collectives (all-reduce, all-gather, etc.)
automatically.

```
$ python train.py          # ONE process, ONE interpreter, ONE memory space

┌─────────────────────────────────────────────────────┐
│  Python Process (PID 1234)                          │
│                                                     │
│  jax.jit(train_step)                                │
│       │                                             │
│       ▼                                             │
│  XLA Compiler ──► partitions graph across devices   │
│       │                                             │
│       ├──► GPU 0  (via XLA runtime)                 │
│       ├──► GPU 1  (via XLA runtime)                 │
│       ├──► GPU 2  (via XLA runtime)                 │
│       └──► GPU 3  (via XLA runtime)                 │
└─────────────────────────────────────────────────────┘
```

For multi-host, JAX launches one process per host (not per device).
`keras.distribution.initialize()` connects them, and XLA still handles
device-level partitioning within each host.

### PyTorch: Multi-Process, Runtime-Driven
PyTorch uses a one-process-per-device model. Each GPU is owned by a separate OS process
with its own Python interpreter and memory space. Processes coordinate via explicit
communication backends (NCCL for GPU, gloo for CPU).

```
$ torchrun --nproc_per_node=4 train.py    # spawns 4 SEPARATE processes

┌──────────────────┐  ┌──────────────────┐
│ Process 0 (RANK=0)│  │ Process 1 (RANK=1)│
│ Python interp #0  │  │ Python interp #1  │
│ GPU 0             │  │ GPU 1             │
│ own memory space  │  │ own memory space  │
└────────┬─────────┘  └────────┬─────────┘
         │    NCCL collectives   │
         ├───────────────────────┤
         │                       │
┌────────┴─────────┐  ┌────────┴─────────┐
│ Process 2 (RANK=2)│  │ Process 3 (RANK=3)│
│ Python interp #2  │  │ Python interp #2  │
│ GPU 2             │  │ GPU 3             │
│ own memory space  │  │ own memory space  │
└──────────────────┘  └──────────────────┘
```

There is no central compiler partitioning the graph. Instead, Keras leverages:
1. **DistributedDataParallel (DDP)**: For `DataParallel`. DDP hooks into autograd
to synchronize gradients at runtime.
2. **DTensor (torch.distributed.tensor)**: For `ModelParallel`. DTensor uses
dispatch mechanisms to insert collectives per-operator.

### Summary of Key Differences

| Aspect | JAX | PyTorch |
| --- | --- | --- |
| **Process model** | 1 process, N devices | N devices = N processes |
| **Parallelism engine**| XLA compiler (graph-level) | Runtime hooks (operator-level) |
| **Communication** | Inserted by compiler | Explicit (DDP hooks, DTensor dispatch) |
| **Launch** | `python train.py` | `torchrun --nproc_per_node=N train.py` |
| **Process init** | `keras.distribution.initialize()` | `init_process_group()` per process |
| **Multi-host** | gRPC (handled by XLA) | NCCL/gloo (explicit MASTER_ADDR) |


---
## Setup

You can choose your backend by setting the `KERAS_BACKEND` environment variable.
We will use JAX for this guide as it provides a clean SPMD (Single Program, Multiple Data)
model that is easy to demonstrate.

```python
import os

# Set backend to "jax" or "torch". Default to "jax" for this guide.
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import numpy as np

# Initialize the distribution system.
# We wrap this in a try-except to handle environments where distributed
# initialization might not be configured (like single-process CPU).
try:
    keras.distribution.initialize()
    print("Keras distribution initialized.")
except Exception as e:
    print(f"Keras distribution initialization skipped or failed: {e}")
```

```
Keras distribution initialization skipped or failed: coordinator_address should be defined.
```

---
## `DeviceMesh` and `TensorLayout`

The `keras.distribution.DeviceMesh` represents a cluster of computational devices
configured for distributed computation. It maps physical devices to a logical mesh
structure.

The `TensorLayout` specifies how tensors (weights or data) are distributed across
the `DeviceMesh`, sharding them along specified axes that correspond to the names
of the axes in the `DeviceMesh`.

```python
# Retrieve global devices
devices = keras.distribution.list_devices()
print(f"Global devices: {devices}")

# Define a device mesh.
# For demonstration, we use a shape that fits the available devices.
# In a real 8-GPU setup, you would use shape=(2, 4) or similar.
if len(devices) >= 8:
    mesh_shape = (2, 4)
else:
    mesh_shape = (len(devices), 1)

mesh = keras.distribution.DeviceMesh(
    shape=mesh_shape, axis_names=["data", "model"], devices=devices
)
print(f"DeviceMesh created with shape: {mesh_shape}")

# A 2D layout specifying sharding along "model" and "data" axes.
# Axes not mentioned are replicated.
layout_2d = keras.distribution.TensorLayout(axes=("model", "data"), device_mesh=mesh)
print(f"TensorLayout created: {layout_2d}")
```

```
Global devices: ['cpu:0']
DeviceMesh created with shape: (1, 1)
TensorLayout created: <TensorLayout axes=('model', 'data'), device_mesh=<DeviceMesh shape=(1, 1), axis_names=['data', 'model']>>
```

---
## DataParallel

`DataParallel` is designed for the data parallelism strategy, where the model weights
are replicated across all devices, and each device processes a portion of the
input data.

On the **PyTorch** backend, Keras leverages native DDP, which is heavily optimized
for the replicate-weights/shard-data pattern using techniques like gradient bucketing
and compute/communication overlap.

On the **JAX** backend, it leverages XLA's SPMD expansion to replicate weights and
shard inputs.

```python
# Simple setup: detects all available local devices automatically
data_parallel = keras.distribution.DataParallel()

# Set the global distribution
keras.distribution.set_distribution(data_parallel)
print(f"Active distribution: {keras.distribution.distribution()}")

# Build your model normally. Under DataParallel, Keras ensures variables
# are replicated.
inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, activation="relu")(y)
y = layers.Dropout(0.4)(y)
outputs = layers.Dense(units=10, activation="softmax")(y)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="mse", optimizer="adam")
model.summary()

# Dataset fed into model.fit will be split evenly on the batch dimension.
# Keras handles the DistributedSampler injection automatically for PyTorch.
# model.fit(dataset, epochs=3)
```

```
Active distribution: <DataParallel device_mesh=<DeviceMesh shape=(1,), axis_names=['batch']>>
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 200)            │       157,000 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 200)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         2,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 159,010 (621.13 KB)
 Trainable params: 159,010 (621.13 KB)
 Non-trainable params: 0 (0.00 B)
```

---
## ModelParallel and `LayoutMap`

`ModelParallel` is useful when model weights are too large to fit on a single
accelerator. It allows you to shard weights or activation tensors across
devices on the `DeviceMesh`.

Keras uses a `LayoutMap` to specify the `TensorLayout` for any weights and
intermediate tensors from a global perspective. `LayoutMap` maps string keys
(variable paths) to `TensorLayout` instances using regex matching.

On the **PyTorch** backend, this is implemented using **DTensor**, which
transparently handles sharded storage and computation.

```python
layout_map = keras.distribution.LayoutMap(mesh)

# Shard weights matching "d1/kernel" along the "model" axis.
layout_map["d1/kernel"] = (None, "model")
layout_map["d1/bias"] = ("model",)

# You can also shard layer outputs
layout_map["d2/output"] = ("data", None)

model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map, batch_dim_name="data"
)

keras.distribution.set_distribution(model_parallel)
print(f"Active distribution: {keras.distribution.distribution()}")

# When the model is built, variables check the active distribution.
# If ModelParallel is active, variables are initialized as sharded entities
# (DTensors on PyTorch, sharded arrays on JAX).
model = keras.Sequential(
    [
        layers.Input(shape=(784,)),
        layers.Dense(200, activation="relu", name="d1"),
        layers.Dense(10, activation="softmax", name="d2"),
    ]
)
model.summary()
```

```
Active distribution: <ModelParallel device_mesh=<DeviceMesh shape=(1, 1), axis_names=['data', 'model']>>
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ d1 (Dense)                      │ (None, 200)            │       157,000 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ d2 (Dense)                      │ (None, 10)             │         2,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 159,010 (621.13 KB)
 Trainable params: 159,010 (621.13 KB)
 Non-trainable params: 0 (0.00 B)
```

---
## Internal Implementation Details

For those interested in how Keras bridges the gap between JAX and PyTorch:

### Variable Lifecycle (Model Parallelism)
1. **Empty Slot**: Variable is born with an empty `_layout`.
2. **Fetching Blueprint**: Before initialization, it queries the global
distribution. If `ModelParallel` is active, it receives a backend-specific
sharding spec.
3. **Safe Assignment**: The variable applies the blueprint. On PyTorch, it
becomes a `DTensor` wrapped in an `nn.Parameter`. On JAX, it becomes a
sharded array.

### TorchTrainer Architecture
The `TorchTrainer` dynamically adapts based on the distribution strategy:
- **DataParallel**: Keras wraps the model in PyTorch's native `DistributedDataParallel`.
Forward passes go through the DDP wrapper to trigger autograd hooks for gradient
synchronization.
- **ModelParallel**: The trainer wraps incoming data batches as `DTensors`. Once
inputs and variables are both `DTensors`, PyTorch handles collectives (all-reduce,
reduce-scatter) automatically via operator dispatch.

### Evaluation and Metric Aggregation
In distributed settings, each rank computes metrics only on its local data slice.
Keras performs an **End-of-Epoch Metric Sync** using `all_reduce(SUM)` on the
raw accumulators (total and count) before computing final results. This ensures
the global accuracy and loss are correctly reported.

### Cross-Framework Data Loading
Keras 3 allows using `tf.data.Dataset` or `PyDataset` with a PyTorch backend.
The data adapter layer injects a `DistributedSampler` when `DataParallel`
is active to ensure each rank receives a non-overlapping slice of data.


---
## Challenges and Limitations

### Launch Mechanism
JAX can perform N-device parallelism from a single `python train.py` call.
PyTorch requires N separate processes. You must use `torchrun` to launch:

```bash
torchrun --nproc_per_node=N train.py
```

### Checkpointing
Keras uses its own HDF5-based format. Under `ModelParallel`, converting a
sharded variable to NumPy (for saving) implicitly triggers an all-gather, so
the full tensor is materialized on rank 0 for writing. Loading handles scattering
automatically.

### Future Work: Fully Sharded Data Parallel (FSDP)
FSDP acts like Data Parallelism but dynamically shards weights, gradients, and
optimizer states. In Keras, this will be achieved by using `ModelParallel` to shard
both weights and optimizer states across the entire mesh.


---
### Further reading

1. [JAX Distributed arrays](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
2. [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
3. [TensorFlow Distributed Training](https://www.tensorflow.org/guide/distributed_training)
