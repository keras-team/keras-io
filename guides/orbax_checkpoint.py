"""
Title: Orbax Checkpointing in Keras
Author: [Amit Srivastava](https://github.com/amitsrivastava78)
Date created: 2025/11/20
Last modified: 2026/03/04
Description: Save and load Orbax checkpoints with distributed resharding.
Accelerator: None
"""

"""
## Introduction

[Orbax](https://orbax.readthedocs.io/) is the recommended checkpointing library
for the JAX ecosystem. It provides high-level functionality for checkpoint
management, composable serialization, and multi-host coordination.

Starting with Keras 3.14, the built-in `keras.callbacks.OrbaxCheckpoint` callback
makes it easy to:

- Save and restore model checkpoints (weights, optimizer state, metrics) during
  training.
- Resume training seamlessly from the latest checkpoint.
- Use `save_best_only` monitoring (just like `ModelCheckpoint`).
- Run in **multi-host distributed** environments with automatic coordination.
- **Reshard** checkpoints when loading under a different distribution layout
  than the one used at save time.
"""

"""
## Setup

Install the Orbax checkpointing library:
"""

"""shell
pip install -q -U orbax-checkpoint
"""

"""
Set the Keras backend to JAX, configure virtual devices for the distributed
demo, and import the required libraries.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import shutil

import jax
import keras
import numpy as np

# Simulate 4 CPU devices for the distributed demo.
# Remove this line if using real multi-device hardware.
jax.config.update("jax_num_cpu_devices", 4)

"""
## Basic Usage

`OrbaxCheckpoint` works like `ModelCheckpoint` — pass it as a callback to
`model.fit()`. No boilerplate classes or wrappers are needed.
"""

"""
### Define a model and dataset
"""


def get_model():
    inputs = keras.Input(shape=(32,))
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    outputs = keras.layers.Dense(1, name="dense_2")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


model = get_model()
x_train = np.random.random((256, 32))
y_train = np.random.random((256, 1))

"""
### Save a checkpoint every epoch
"""

checkpoint_dir = "/tmp/orbax_ckpt_basic"
shutil.rmtree(checkpoint_dir, ignore_errors=True)

callback = keras.callbacks.OrbaxCheckpoint(
    directory=checkpoint_dir,
    max_to_keep=3,
)

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=3,
    verbose=1,
    validation_split=0.2,
    callbacks=[callback],
)

"""
The checkpoint directory now contains a step-directory for each
saved epoch.
"""

"""shell
ls /tmp/orbax_ckpt_basic
"""

"""
## Loading a model from an Orbax checkpoint

Use `keras.saving.load_model()` to reload a full model (config + weights +
optimizer state) from an Orbax checkpoint directory.
"""

loaded_model = keras.saving.load_model(checkpoint_dir)
loaded_model.summary()

"""
## Loading weights only

If you already have a model instance and just want to load the weights, use
`load_weights()`:
"""

fresh_model = get_model()
fresh_model.load_weights(checkpoint_dir)

# Verify both loaded_model and fresh_model match the original.
for m in [loaded_model, fresh_model]:
    for orig, restored in zip(model.weights, m.weights):
        np.testing.assert_allclose(orig.numpy(), restored.numpy(), atol=1e-6)
print("Weights match!")

"""
## Resuming training

When you pass `OrbaxCheckpoint` to a new `fit()` call, training resumes from
the correct step number. Pass `initial_epoch` to continue the epoch
count from where you left off.
"""

resumed_model = keras.saving.load_model(checkpoint_dir)

# Create a new callback pointing to the same directory.
resume_callback = keras.callbacks.OrbaxCheckpoint(
    directory=checkpoint_dir,
    max_to_keep=5,
)

# Continue training — epochs 3 and 4 (picking up from epoch 3).
resumed_model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    initial_epoch=3,
    verbose=1,
    validation_split=0.2,
    callbacks=[resume_callback],
)

"""
## Save best only

Monitor a metric and keep only the best checkpoint:
"""

best_dir = "/tmp/orbax_ckpt_best"
shutil.rmtree(best_dir, ignore_errors=True)

best_callback = keras.callbacks.OrbaxCheckpoint(
    directory=best_dir,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    max_to_keep=1,
)

model_best = get_model()
model_best.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_split=0.2,
    callbacks=[best_callback],
)

"""
## Batch-level checkpointing

Save every N batches instead of every epoch by setting `save_freq` to an
integer:
"""

batch_dir = "/tmp/orbax_ckpt_batch"
shutil.rmtree(batch_dir, ignore_errors=True)

batch_callback = keras.callbacks.OrbaxCheckpoint(
    directory=batch_dir,
    save_freq=4,  # Save every 4 batches.
    max_to_keep=3,
)

model_batch = get_model()
model_batch.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=2,
    verbose=1,
    callbacks=[batch_callback],
)

"""
## Distributed training with model parallelism

`OrbaxCheckpoint` works seamlessly with the Keras Distribution API. When you
set a distribution, model variables are automatically sharded according to
your `LayoutMap` and checkpoints capture the distributed state.

> **Note**: The example below uses virtual devices to simulate a multi-device
> environment. In production, use your actual accelerators (GPUs/TPUs).
"""

devices = jax.devices()
print(f"Available devices: {len(devices)}")

"""
### Define a layout map and distribution
"""

mesh = keras.distribution.DeviceMesh(
    shape=(2, 2),
    axis_names=["data", "model"],
    devices=devices,
)

layout_map = keras.distribution.LayoutMap(mesh)
layout_map["dense_1/kernel"] = (None, "model")

distribution = keras.distribution.ModelParallel(
    layout_map=layout_map, batch_dim_name="data"
)
keras.distribution.set_distribution(distribution)

"""
### Train with distributed checkpointing
"""

dist_dir = "/tmp/orbax_ckpt_dist"
shutil.rmtree(dist_dir, ignore_errors=True)
dist_model = get_model()

dist_callback = keras.callbacks.OrbaxCheckpoint(
    directory=dist_dir,
    max_to_keep=2,
)

dist_model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=3,
    verbose=1,
    validation_split=0.2,
    callbacks=[dist_callback],
)

"""
### Verify sharding

After training, you can inspect the sharding of saved variables:
"""

loaded_dist = keras.saving.load_model(dist_dir)
for v in loaded_dist.trainable_variables:
    print(f"{v.path}: shape={v.shape}, sharding={v.value.sharding}")

"""
## Cross-layout resharding

One of the most powerful features of Orbax checkpointing in Keras is the
ability to load a checkpoint saved with one sharding layout and restore it
under a **different** layout. Keras automatically provides the target
shardings to Orbax so arrays are resharded on load.

For example, a checkpoint saved with `dense_1/kernel` sharded on the
`"model"` axis can be loaded with that same kernel sharded on the `"data"`
axis instead:
"""

# Clear the current distribution first.
keras.distribution.set_distribution(None)

# Define a new layout that shards dense_1/kernel on a different axis.
new_layout_map = keras.distribution.LayoutMap(mesh)
new_layout_map["dense_1/kernel"] = ("data", None)

new_distribution = keras.distribution.ModelParallel(
    layout_map=new_layout_map, batch_dim_name="data"
)
keras.distribution.set_distribution(new_distribution)

# Load the checkpoint saved with the original layout.
resharded_model = keras.saving.load_model(dist_dir)

print("\nResharded variable shardings:")
for v in resharded_model.trainable_variables:
    print(f"  {v.path}: sharding={v.value.sharding}")

"""
The `dense_1/kernel` variable was originally sharded as `(None, "model")`
but is now sharded as `("data", None)`. Orbax handles the data movement
automatically during loading.
"""

"""
## Callback parameters reference

| Parameter | Default | Description |
|---|---|---|
| `directory` | *(required)* | Path to checkpoint directory. |
| `monitor` | `"val_loss"` | Metric to monitor for `save_best_only`. |
| `mode` | `"auto"` | `"min"`, `"max"`, or `"auto"`. |
| `save_best_only` | `False` | Only save when monitored metric improves. |
| `save_freq` | `"epoch"` | `"epoch"` or an integer (every N batches). |
| `max_to_keep` | `1` | Max recent checkpoints to retain. |
| `save_on_background` | `True` | Save asynchronously to avoid blocking. |
| `save_weights_only` | `False` | Save only weights (no model config/assets). |
| `initial_value_threshold` | `None` | Initial "best" value for the monitor. |
| `verbose` | `0` | Verbosity (0 = silent, 1 = messages). |
"""

"""
## Summary

- **`keras.callbacks.OrbaxCheckpoint`** is the built-in callback for Orbax
  checkpointing — no wrapper classes needed.
- Use **`keras.saving.load_model()`** or **`model.load_weights()`** to
  restore from an Orbax checkpoint directory.
- Works with **distributed training** (Keras Distribution API) and supports
  **cross-layout resharding** when loading under a different `LayoutMap`.
- Supports **multi-host** JAX environments with automatic coordination.
- Mirrors the `ModelCheckpoint` API: `save_best_only`, `monitor`, `mode`,
  `save_freq`, `max_to_keep`.
"""

# Clean up the distribution for any subsequent cells.
keras.distribution.set_distribution(None)
