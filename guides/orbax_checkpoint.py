"""
Title: Orbax Checkpointing in Keras
Author: [Samaneh Saadat](https://github.com/SamanehSaadat/)
Date created: 2025/08/20
Last modified: 2025/08/20
Description: A guide on how to save Orbax checkpoints during model training with the Jax backend.
Accelerator: GPU
"""

"""
## Introduction
Orbax is the default checkpointing library recommended for JAX ecosystem
users. It is a high-level checkpointing library which provides functionality
for both checkpoint management and composable and extensible serialization.
This guide explains how to do Orbax checkpointing when training a model in
the Jax backend.

The default `.keras` format doesn't support multi-host checkpointing so if
you are using Keras distribution API for multi-host training, you need to
use Orbax checkpointing.
"""

"""
## Setup
Let's start by installing Orbax checkpointing library:
"""

"""shell
pip install -q -u orbax-checkpoint
"""

"""
We need to set the Keras backend to Jax as this guide is intended for the
Jax backend. Then we import Keras and other libraries needed including the
Orbax checkpointing library.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import orbax.checkpoint as ocp

"""
## Orbax Callback
We need to create two main utilities to manage Orbax checkpointing in Keras:
1. `KerasOrbaxCheckpointManager`: A wrapper around
`orbax.checkpoint.CheckpointManager` for Keras models.
`KerasOrbaxCheckpointManager` uses `Model`'s `get_state_tree` and
`set_state_tree` APIs to save and restore the model variables.
2. `OrbaxCheckpointCallback`: A Keras callback that uses
`KerasOrbaxCheckpointManager` to automatically save and restore model states
during training.

Orbax checkpointing in Keras is as simple as copying these utilities to your
own codebase and passing `OrbaxCheckpointCallback` to the `fit`.
"""


class KerasOrbaxCheckpointManager(ocp.CheckpointManager):
    """A wrapper over Orbax CheckpointManager for Keras with the Jax
    backend."""

    def __init__(
        self,
        model,
        checkpoint_dir,
        max_to_keep=5,
        steps_per_epoch=1,
        **kwargs,
    ):
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, enable_async_checkpointing=False, **kwargs
        )
        self._model = model
        self._steps_per_epoch = steps_per_epoch
        self._checkpoint_dir = checkpoint_dir
        super().__init__(checkpoint_dir, options=options)

    def _get_state(self):
        """Gets the model state and metrics"""
        model_state = self._model.get_state_tree()
        state = {}
        metrics = None
        for k, v in model_state.items():
            if k == "metrics_variables":
                metrics = v
            else:
                state[k] = v
        return state, metrics

    def save_state(self, epoch):
        """Saves the model to the checkpoint directory.

        Args:
          epoch: The epoch number at which the state is saved.
        """
        state, metrics_value = self._get_state()
        self.save(
            epoch * self._steps_per_epoch,
            args=ocp.args.StandardSave(item=state),
            metrics=metrics_value,
        )

    def restore_state(self, step=None):
        """Restores the model from the checkpoint directory.

        Args:
          step: The step number to restore the state from. Default=None
            restores the latest step.
        """
        step = step or self.latest_step()
        if step is None:
            return
        # Restore the model state only, not metrics.
        state, _ = self._get_state()
        restored_state = self.restore(step, args=ocp.args.StandardRestore(item=state))
        self._model.set_state_tree(restored_state)


class OrbaxCheckpointCallback(keras.callbacks.Callback):
    """A callback for checkpointing and restoring state using Orbax."""

    def __init__(
        self,
        model,
        checkpoint_dir,
        max_to_keep=5,
        steps_per_epoch=1,
        **kwargs,
    ):
        if keras.config.backend() != "jax":
            raise ValueError(
                "`OrbaxCheckpointCallback` is only supported on a "
                "`jax` backend. Provided backend is %s." % keras.config.backend()
            )
        self._checkpoint_manager = KerasOrbaxCheckpointManager(
            model, checkpoint_dir, max_to_keep, steps_per_epoch, **kwargs
        )

    def on_train_begin(self, logs=None):
        if not self.model.built or not self.model.optimizer.built:
            raise ValueError(
                "To use `OrbaxCheckpointCallback`, your model and "
                "optimizer must be built before you call `fit()`."
            )
        latest_epoch = self._checkpoint_manager.latest_step()
        if latest_epoch is not None:
            print("Load Orbax checkpoint on_train_begin.")
            self._checkpoint_manager.restore_state(step=latest_epoch)

    def on_epoch_end(self, epoch, logs=None):
        print("Save Orbax checkpoint on_epoch_end.")
        self._checkpoint_manager.save_state(epoch)


"""
## An Orbax checkpointing example
Let's look at how we can use `OrbaxCheckpointCallback` to save Orbax
checkpoints during the training. To get started, let's define a simple model
and a toy training dataset.
"""


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1, name="dense")(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model


model = get_model()

x_train = np.random.random((128, 32))
y_train = np.random.random((128, 1))

"""
Then, we create an Orbax checkpointing callback and pass it to the
`callbacks` argument in the `fit` function.
"""

orbax_callback = OrbaxCheckpointCallback(
    model,
    checkpoint_dir="/tmp/ckpt",
    max_to_keep=1,
    steps_per_epoch=1,
)
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=3,
    verbose=0,
    validation_split=0.2,
    callbacks=[orbax_callback],
)

"""
Now if you look at the Orbax checkpoint directory, you can see all the files
saved as part of Orbax checkpointing.
"""
