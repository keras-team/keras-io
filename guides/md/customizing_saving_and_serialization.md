# Customizing Saving and Serialization

**Author:** Neel Kovelamudi<br>
**Date created:** 2023/03/15<br>
**Last modified:** 2023/03/15<br>
**Description:** A more advanced guide on customizing saving for your layers and models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/customizing_saving_and_serialization.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/customizing_saving_and_serialization.py)



---
## Introduction

This guide covers advanced methods that can be customized in Keras saving. For most
users, the methods outlined in the primary
[Serialize, save, and export guide](https://keras.io/guides/serialization_and_saving)
are sufficient.

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

---
## Setup


```python
import os
import numpy as np
import tensorflow as tf
import keras
```

---
## State saving customization

These methods determine how the state of your model's layers is saved when calling
`model.save()`. You can override them to take full control of the state saving process.


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


```python

@keras.utils.register_keras_serializable(package="my_custom_package")
class LayerWithCustomVariables(keras.layers.Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.stored_variables = tf.Variable(
            np.random.random((10,)), name="special_arr", dtype=tf.float32
        )

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # Stores the value of the `tf.Variable` upon saving
        store["variables"] = self.stored_variables.numpy()

    def load_own_variables(self, store):
        # Assigns the value of the `tf.Variable` upon loading
        self.stored_variables.assign(store["variables"])
        # Load the remaining weights
        for i, v in enumerate(self.weights):
            v.assign(store[f"{i}"])
        # Note: You must specify how all variables (including layer weights)
        # are loaded in `load_own_variables.`

    def call(self, inputs):
        return super().call(inputs) * self.stored_variables


model = keras.Sequential([LayerWithCustomVariables(1)])

ref_input = np.random.random((8, 10))
ref_output = np.random.random((8,))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(ref_input, ref_output)

model.save("custom_vars_model.keras")
restored_model = keras.models.load_model("custom_vars_model.keras")

np.testing.assert_allclose(
    model.layers[0].stored_variables.numpy(),
    restored_model.layers[0].stored_variables.numpy(),
)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 418ms/step - loss: 0.7132

```
</div>
### `save_assets()` and `load_assets()`

These methods can be added to your model class definition to store and load any
additional information that your model needs.

For example, NLP domain layers such as TextVectorization layers and IndexLookup layers
may need to store their associated vocabulary (or lookup table) in a text file upon
saving.

Let's take at the basics of this workflow with a simple file `assets.txt`.

**Example:**


```python

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
```

---
## `build` and `compile` saving customization

### `get_build_config()` and `build_from_config()`

These methods work together to save the layer's built states and restore them upon
loading.

By default, this only includes a build config dictionary with the layer's input shape,
but overriding these methods can be used to include further Variables and Lookup Tables
that can be useful to restore for your built model.

**Example:**


```python

@keras.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomBuild(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return dict(units=self.units, **super().get_config())

    def build(self, input_shape, layer_init):
        # Note the customization in overriding `build()` adds an extra argument.
        # Therefore, we will need to manually call build with `layer_init` argument
        # before the first execution of `call()`.
        super().build(input_shape)
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
        build_config = super().get_build_config()  # only gives `input_shape`
        build_config.update(
            {"layer_init": self.layer_init}  # Stores our initializer for `build()`
        )
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
```

### `get_compile_config()` and `compile_from_config()`

These methods work together to save the information with which the model was compiled
(optimizers, losses, etc.) and restore and re-compile the model with this information.

Overriding these methods can be useful for compiling the restored model with custom
optimizers, custom losses, etc., as these will need to be deserialized prior to calling
`model.compile` in `compile_from_config()`.

Let's take a look at an example of this.

**Example:**


```python

@keras.saving.register_keras_serializable(package="my_custom_package")
def small_square_sum_loss(y_true, y_pred):
    loss = tf.math.squared_difference(y_pred, y_true)
    loss = loss / 10.0
    loss = tf.reduce_sum(loss, axis=1)
    return loss


@keras.saving.register_keras_serializable(package="my_custom_package")
def mean_pred(y_true, y_pred):
    return tf.reduce_mean(y_pred)


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomCompile(keras.Model):
    def __init__(self):
        super().__init__()
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
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 386ms/step - loss: 0.0704 - accuracy: 0.0000e+00 - mean_pred: 0.2500

WARNING:absl:Skipping variable loading for optimizer 'SGD', because it has 1 variables whereas the saved optimizer has 5 variables. 

```
</div>
---
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
