"""
Title: Customizing Saving and Serialization
Author: Neel Kovelamudi
Date created: 2023/03/15
Last modified: 2023/03/15
Description: A more advanced guide on customizing saving for your layers and models.
Accelerator: None
"""

"""
## Introduction

This guide covers advanced methods that can be customized in Keras saving. For most
users, the methods outlined in the primary
[Serialize, save, and export guide](https://keras.io/guides/serialization_and_saving)
are sufficient.
"""

"""
### APIs
We will cover the following APIs:

- `save_assets()` and `load_assets()`
- `save_own_variables()` and `load_own_variables()`
- `get_build_config()` and `build_from_config()`
- `get_compile_config()` and `compile_from_config()`

When restoring a model, these get executed in the following order:

- `build_from_config()`
- `compile_from_config()`
- `load_own_variables()`
- `load_assets()`

"""

"""
## Setup
"""

import os
import numpy as np
import keras

"""
## State saving customization

These methods determine how the state of your model's layers is saved when calling
`model.save()`. You can override them to take full control of the state saving process.
"""

"""
### `save_own_variables()` and `load_own_variables()`

These methods save and load the state variables of the layer when `model.save()` and
`keras.models.load_model()` are called, respectively. By default, the state variables
saved and loaded are the weights of the layer (both trainable and non-trainable). Here is
the default implementation of `save_own_variables()`:

```python
def save_own_variables(self, store):
    all_vars = self._trainable_weights + self._non_trainable_weights
    for i, v in enumerate(all_vars):
        store[f"{i}"] = v.numpy()
```

The store used by these methods is a dictionary that can be populated with the layer
variables. Let's take a look at an example customizing this.

**Example:**
"""


@keras.utils.register_keras_serializable(package="my_custom_package")
class LayerWithCustomVariable(keras.layers.Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.my_variable = keras.Variable(
            np.random.random((units,)), name="my_variable", dtype="float32"
        )

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # Stores the value of the variable upon saving
        store["variables"] = self.my_variable.numpy()

    def load_own_variables(self, store):
        # Assigns the value of the variable upon loading
        self.my_variable.assign(store["variables"])
        # Load the remaining weights
        for i, v in enumerate(self.weights):
            v.assign(store[f"{i}"])
        # Note: You must specify how all variables (including layer weights)
        # are loaded in `load_own_variables.`

    def call(self, inputs):
        dense_out = super().call(inputs)
        return dense_out + self.my_variable


model = keras.Sequential([LayerWithCustomVariable(1)])

ref_input = np.random.random((8, 10))
ref_output = np.random.random((8, 10))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(ref_input, ref_output)

model.save("custom_vars_model.keras")
restored_model = keras.models.load_model("custom_vars_model.keras")

np.testing.assert_allclose(
    model.layers[0].my_variable.numpy(),
    restored_model.layers[0].my_variable.numpy(),
)

"""
### `save_assets()` and `load_assets()`

These methods can be added to your model class definition to store and load any
additional information that your model needs.

For example, NLP domain layers such as TextVectorization layers and IndexLookup layers
may need to store their associated vocabulary (or lookup table) in a text file upon
saving.

Let's take at the basics of this workflow with a simple file `assets.txt`.

**Example:**
"""


@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomAssets(keras.layers.Dense):
    def __init__(self, vocab=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab

    def save_assets(self, inner_path):
        # Writes the vocab (sentence) to text file at save time.
        with open(os.path.join(inner_path, "vocabulary.txt"), "w") as f:
            f.write(self.vocab)

    def load_assets(self, inner_path):
        # Reads the vocab (sentence) from text file at load time.
        with open(os.path.join(inner_path, "vocabulary.txt"), "r") as f:
            text = f.read()
        self.vocab = text.replace("<unk>", "little")


model = keras.Sequential(
    [LayerWithCustomAssets(vocab="Mary had a <unk> lamb.", units=5)]
)

x = np.random.random((10, 10))
y = model(x)

model.save("custom_assets_model.keras")
restored_model = keras.models.load_model("custom_assets_model.keras")

np.testing.assert_string_equal(
    restored_model.layers[0].vocab, "Mary had a little lamb."
)

"""
## `build` and `compile` saving customization

### `get_build_config()` and `build_from_config()`

These methods work together to save the layer's built states and restore them upon
loading.

By default, this only includes a build config dictionary with the layer's input shape,
but overriding these methods can be used to include further Variables and Lookup Tables
that can be useful to restore for your built model.

**Example:**
"""


@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomBuild(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        return dict(units=self.units, **super().get_config())

    def build(self, input_shape, layer_init):
        # Note the overriding of `build()` to add an extra argument.
        # Therefore, we will need to manually call build with `layer_init` argument
        # before the first execution of `call()`.
        super().build(input_shape)
        self._input_shape = input_shape
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=layer_init,
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=layer_init,
            trainable=True,
        )
        self.layer_init = layer_init

    def get_build_config(self):
        build_config = {
            "layer_init": self.layer_init,
            "input_shape": self._input_shape,
        }  # Stores our initializer for `build()`
        return build_config

    def build_from_config(self, config):
        # Calls `build()` with the parameters at loading time
        self.build(config["input_shape"], config["layer_init"])


custom_layer = LayerWithCustomBuild(units=16)
custom_layer.build(input_shape=(8,), layer_init="random_normal")

model = keras.Sequential(
    [
        custom_layer,
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

x = np.random.random((16, 8))
y = model(x)

model.save("custom_build_model.keras")
restored_model = keras.models.load_model("custom_build_model.keras")

np.testing.assert_equal(restored_model.layers[0].layer_init, "random_normal")
np.testing.assert_equal(restored_model.built, True)

"""
### `get_compile_config()` and `compile_from_config()`

These methods work together to save the information with which the model was compiled
(optimizers, losses, etc.) and restore and re-compile the model with this information.

Overriding these methods can be useful for compiling the restored model with custom
optimizers, custom losses, etc., as these will need to be deserialized prior to calling
`model.compile` in `compile_from_config()`.

Let's take a look at an example of this.

**Example:**
"""


@keras.saving.register_keras_serializable(package="my_custom_package")
def small_square_sum_loss(y_true, y_pred):
    loss = keras.ops.square(y_pred - y_true)
    loss = loss / 10.0
    loss = keras.ops.sum(loss, axis=1)
    return loss


@keras.saving.register_keras_serializable(package="my_custom_package")
def mean_pred(y_true, y_pred):
    return keras.ops.mean(y_pred)


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomCompile(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(8, activation="relu")
        self.dense2 = keras.layers.Dense(4, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def compile(self, optimizer, loss_fn, metrics):
        super().compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model_optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metrics = metrics

    def get_compile_config(self):
        # These parameters will be serialized at saving time.
        return {
            "model_optimizer": self.model_optimizer,
            "loss_fn": self.loss_fn,
            "metric": self.loss_metrics,
        }

    def compile_from_config(self, config):
        # Deserializes the compile parameters (important, since many are custom)
        optimizer = keras.utils.deserialize_keras_object(config["model_optimizer"])
        loss_fn = keras.utils.deserialize_keras_object(config["loss_fn"])
        metrics = keras.utils.deserialize_keras_object(config["metric"])

        # Calls compile with the deserialized parameters
        self.compile(optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)


model = ModelWithCustomCompile()
model.compile(
    optimizer="SGD", loss_fn=small_square_sum_loss, metrics=["accuracy", mean_pred]
)

x = np.random.random((4, 8))
y = np.random.random((4,))

model.fit(x, y)

model.save("custom_compile_model.keras")
restored_model = keras.models.load_model("custom_compile_model.keras")

np.testing.assert_equal(model.model_optimizer, restored_model.model_optimizer)
np.testing.assert_equal(model.loss_fn, restored_model.loss_fn)
np.testing.assert_equal(model.loss_metrics, restored_model.loss_metrics)

"""
## Conclusion

Using the methods learned in this tutorial allows for a wide variety of use cases,
allowing the saving and loading of complex models with exotic assets and state
elements. To recap:

- `save_own_variables` and `load_own_variables` determine how your states are saved
and loaded.
- `save_assets` and `load_assets` can be added to store and load any additional
information your model needs.
- `get_build_config` and `build_from_config` save and restore the model's built
states.
- `get_compile_config` and `compile_from_config` save and restore the model's
compiled states.
"""
