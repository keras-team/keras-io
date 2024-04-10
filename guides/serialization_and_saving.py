"""
Title: Save, serialize, and export models
Authors: Neel Kovelamudi, Francois Chollet
Date created: 2023/06/14
Last modified: 2023/06/30
Description: Complete guide to saving, serializing, and exporting models.
Accelerator: None
"""

"""
## Introduction

A Keras model consists of multiple components:

- The architecture, or configuration, which specifies what layers the model
contain, and how they're connected.
- A set of weights values (the "state of the model").
- An optimizer (defined by compiling the model).
- A set of losses and metrics (defined by compiling the model).

The Keras API saves all of these pieces together in a unified format,
marked by the `.keras` extension. This is a zip archive consisting of the
following:

- A JSON-based configuration file (config.json): Records of model, layer, and
other trackables' configuration.
- A H5-based state file, such as `model.weights.h5` (for the whole model),
with directory keys for layers and their weights.
- A metadata file in JSON, storing things such as the current Keras version.

Let's take a look at how this works.
"""

"""
## How to save and load a model

If you only have 10 seconds to read this guide, here's what you need to know.

**Saving a Keras model:**

```python
model = ...  # Get model (Sequential, Functional Model, or Model subclass)
model.save('path/to/location.keras')  # The file needs to end with the .keras extension
```

**Loading the model back:**

```python
model = keras.models.load_model('path/to/location.keras')
```

Now, let's look at the details.
"""

"""
## Setup
"""

import numpy as np
import keras
from keras import ops

"""
## Saving

This section is about saving an entire model to a single file. The file will include:

- The model's architecture/config
- The model's weight values (which were learned during training)
- The model's compilation information (if `compile()` was called)
- The optimizer and its state, if any (this enables you to restart training
where you left)

#### APIs

You can save a model with `model.save()` or `keras.models.save_model()` (which is equivalent).
You can load it back with `keras.models.load_model()`.

The only supported format in Keras 3 is the "Keras v3" format,
which uses the `.keras` extension.

**Example:**
"""


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("my_model.keras")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

"""
### Custom objects

This section covers the basic workflows for handling custom layers, functions, and
models in Keras saving and reloading.

When saving a model that includes custom objects, such as a subclassed Layer,
you **must** define a `get_config()` method on the object class.
If the arguments passed to the constructor (`__init__()` method) of the custom object
aren't Python objects (anything other than base types like ints, strings,
etc.), then you **must** also explicitly deserialize these arguments in the `from_config()`
class method.

Like this:

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)
```

Please see the [Defining the config methods section](#config_methods) for more
details and examples.

The saved `.keras` file is lightweight and does not store the Python code for custom
objects. Therefore, to reload the model, `load_model` requires access to the definition
of any custom objects used through one of the following methods:

1. Registering custom objects **(preferred)**,
2. Passing custom objects directly when loading, or
3. Using a custom object scope

Below are examples of each workflow:

#### Registering custom objects (**preferred**)

This is the preferred method, as custom object registration greatly simplifies saving and
loading code. Adding the `@keras.saving.register_keras_serializable` decorator to the
class definition of a custom object registers the object globally in a master list,
allowing Keras to recognize the object when loading the model.

Let's create a custom model involving both a custom layer and a custom activation
function to demonstrate this.

**Example:**
"""

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()


# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to `Custom` and the name defaults to
# the class name.
@keras.saving.register_keras_serializable(package="MyLayers")
class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def custom_fn(x):
    return x**2


# Create the model.
def get_model():
    inputs = keras.Input(shape=(4,))
    mid = CustomLayer(0.5)(inputs)
    outputs = keras.layers.Dense(1, activation=custom_fn)(mid)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    return model


# Train the model.
def train_model(model):
    input = np.random.random((4, 4))
    target = np.random.random((4, 1))
    model.fit(input, target)
    return model


test_input = np.random.random((4, 4))
test_target = np.random.random((4, 1))

model = get_model()
model = train_model(model)
model.save("custom_model.keras")

# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model("custom_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

"""
#### Passing custom objects to `load_model()`
"""

model = get_model()
model = train_model(model)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("custom_model.keras")

# Upon loading, pass a dict containing the custom objects used in the
# `custom_objects` argument of `keras.models.load_model()`.
reconstructed_model = keras.models.load_model(
    "custom_model.keras",
    custom_objects={"CustomLayer": CustomLayer, "custom_fn": custom_fn},
)

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)


"""
#### Using a custom object scope

Any code within the custom object scope will be able to recognize the custom objects
passed to the scope argument. Therefore, loading the model within the scope will allow
the loading of our custom objects.

**Example:**
"""

model = get_model()
model = train_model(model)
model.save("custom_model.keras")

# Pass the custom objects dictionary to a custom object scope and place
# the `keras.models.load_model()` call within the scope.
custom_objects = {"CustomLayer": CustomLayer, "custom_fn": custom_fn}

with keras.saving.custom_object_scope(custom_objects):
    reconstructed_model = keras.models.load_model("custom_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

"""
### Model serialization

This section is about saving only the model's configuration, without its state.
The model's configuration (or architecture) specifies what layers the model
contains, and how these layers are connected. If you have the configuration of a model,
then the model can be created with a freshly initialized state (no weights or compilation
information).

#### APIs

The following serialization APIs are available:

- `keras.models.clone_model(model)`: make a (randomly initialized) copy of a model.
- `get_config()` and `cls.from_config()`: retrieve the configuration of a layer or model, and recreate
a model instance from its config, respectively.
- `keras.models.model_to_json()` and `keras.models.model_from_json()`: similar, but as JSON strings.
- `keras.saving.serialize_keras_object()`: retrieve the configuration any arbitrary Keras object.
- `keras.saving.deserialize_keras_object()`: recreate an object instance from its configuration.

#### In-memory model cloning

You can do in-memory cloning of a model via `keras.models.clone_model()`.
This is equivalent to getting the config then recreating the model from its config
(so it does not preserve compilation information or layer weights values).

**Example:**
"""

new_model = keras.models.clone_model(model)

"""
#### `get_config()` and `from_config()`

Calling `model.get_config()` or `layer.get_config()` will return a Python dict containing
the configuration of the model or layer, respectively. You should define `get_config()`
to contain arguments needed for the `__init__()` method of the model or layer. At loading time,
the `from_config(config)` method will then call `__init__()` with these arguments to
reconstruct the model or layer.


**Layer example:**
"""

layer = keras.layers.Dense(3, activation="relu")
layer_config = layer.get_config()
print(layer_config)

"""
Now let's reconstruct the layer using the `from_config()` method:
"""

new_layer = keras.layers.Dense.from_config(layer_config)

"""
**Sequential model example:**
"""

model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
config = model.get_config()
new_model = keras.Sequential.from_config(config)

"""
**Functional model example:**
"""

inputs = keras.Input((32,))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config)

"""
#### `to_json()` and `keras.models.model_from_json()`

This is similar to `get_config` / `from_config`, except it turns the model
into a JSON string, which can then be loaded without the original model class.
It is also specific to models, it isn't meant for layers.

**Example:**
"""

model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
json_config = model.to_json()
new_model = keras.models.model_from_json(json_config)


"""
#### Arbitrary object serialization and deserialization

The `keras.saving.serialize_keras_object()` and `keras.saving.deserialize_keras_object()`
APIs are general-purpose APIs that can be used to serialize or deserialize any Keras
object and any custom object. It is at the foundation of saving model architecture and is
behind all `serialize()`/`deserialize()` calls in keras.

**Example**:
"""

my_reg = keras.regularizers.L1(0.005)
config = keras.saving.serialize_keras_object(my_reg)
print(config)

"""
Note the serialization format containing all the necessary information for proper
reconstruction:

- `module` containing the name of the Keras module or other identifying module the object
comes from
- `class_name` containing the name of the object's class.
- `config` with all the information needed to reconstruct the object
- `registered_name` for custom objects. See [here](#custom_object_serialization).

Now we can reconstruct the regularizer.
"""

new_reg = keras.saving.deserialize_keras_object(config)

"""
### Model weights saving

You can choose to only save & load a model's weights. This can be useful if:

- You only need the model for inference: in this case you won't need to
restart training, so you don't need the compilation information or optimizer state.
- You are doing transfer learning: in this case you will be training a new model
reusing the state of a prior model, so you don't need the compilation
information of the prior model.

#### APIs for in-memory weight transfer

Weights can be copied between different objects by using `get_weights()`
and `set_weights()`:

* `keras.layers.Layer.get_weights()`: Returns a list of NumPy arrays of weight values.
* `keras.layers.Layer.set_weights(weights)`: Sets the model weights to the values
provided (as NumPy arrays).

Examples:

***Transferring weights from one layer to another, in memory***
"""


def create_layer():
    layer = keras.layers.Dense(64, activation="relu", name="dense_2")
    layer.build((None, 784))
    return layer


layer_1 = create_layer()
layer_2 = create_layer()

# Copy weights from layer 1 to layer 2
layer_2.set_weights(layer_1.get_weights())

"""
***Transferring weights from one model to another model with a compatible architecture, in memory***
"""

# Create a simple functional model
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation="relu", name="dense_1")
        self.dense_2 = keras.layers.Dense(64, activation="relu", name="dense_2")
        self.dense_3 = keras.layers.Dense(output_dim, name="predictions")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


subclassed_model = SubclassedModel(10)
# Call the subclassed model once to create the weights.
subclassed_model(np.ones((1, 784)))

# Copy weights from functional_model to subclassed_model.
subclassed_model.set_weights(functional_model.get_weights())

assert len(functional_model.weights) == len(subclassed_model.weights)
for a, b in zip(functional_model.weights, subclassed_model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

"""
***The case of stateless layers***

Because stateless layers do not change the order or number of weights,
models can have compatible architectures even if there are extra/missing
stateless layers.
"""

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)

# Add a dropout layer, which does not contain any weights.
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model_with_dropout = keras.Model(
    inputs=inputs, outputs=outputs, name="3_layer_mlp"
)

functional_model_with_dropout.set_weights(functional_model.get_weights())

"""
#### APIs for saving weights to disk & loading them back

Weights can be saved to disk by calling `model.save_weights(filepath)`.
The filename should end in `.weights.h5`.

**Example:**
"""

# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)
sequential_model.save_weights("my_model.weights.h5")
sequential_model.load_weights("my_model.weights.h5")

"""
Note that changing `layer.trainable` may result in a different
`layer.weights` ordering when the model contains nested layers.
"""


class NestedDenseLayer(keras.layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.dense_1 = keras.layers.Dense(units, name="dense_1")
        self.dense_2 = keras.layers.Dense(units, name="dense_2")

    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))


nested_model = keras.Sequential([keras.Input((784,)), NestedDenseLayer(10, "nested")])
variable_names = [v.name for v in nested_model.weights]
print("variables: {}".format(variable_names))

print("\nChanging trainable status of one of the nested layers...")
nested_model.get_layer("nested").dense_1.trainable = False

variable_names_2 = [v.name for v in nested_model.weights]
print("\nvariables: {}".format(variable_names_2))
print("variable ordering changed:", variable_names != variable_names_2)

"""
##### **Transfer learning example**

When loading pretrained weights from a weights file, it is recommended to load
the weights into the original checkpointed model, and then extract
the desired weights/layers into a new model.

**Example:**
"""


def create_functional_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(10, name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


functional_model = create_functional_model()
functional_model.save_weights("pretrained.weights.h5")

# In a separate program:
pretrained_model = create_functional_model()
pretrained_model.load_weights("pretrained.weights.h5")

# Create a new model by extracting layers from the original model:
extracted_layers = pretrained_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(5, name="dense_3"))
model = keras.Sequential(extracted_layers)
model.summary()

"""
### Appendix: Handling custom objects

<a name="config_methods"></a>
#### Defining the config methods

Specifications:

* `get_config()` should return a JSON-serializable dictionary in order to be
compatible with the Keras architecture- and model-saving APIs.
* `from_config(config)` (a `classmethod`) should return a new layer or model
object that is created from the config.
The default implementation returns `cls(**config)`.

**NOTE**:  If all your constructor arguments are already serializable, e.g. strings and
ints, or non-custom Keras objects, overriding `from_config` is not necessary. However,
for more complex objects such as layers or models passed to `__init__`, deserialization
must be handled explicitly either in `__init__` itself or overriding the `from_config()`
method.

**Example:**
"""


@keras.saving.register_keras_serializable(package="MyLayers", name="KernelMult")
class MyDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        *,
        kernel_regularizer=None,
        kernel_initializer=None,
        nested_model=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_units = units
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.nested_model = nested_model

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "units": self.hidden_units,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_initializer": self.kernel_initializer,
                "nested_model": self.nested_model,
            }
        )
        return config

    def build(self, input_shape):
        input_units = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_units, self.hidden_units),
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.kernel)


layer = MyDense(units=16, kernel_regularizer="l1", kernel_initializer="ones")
layer3 = MyDense(units=64, nested_model=layer)

config = keras.layers.serialize(layer3)

print(config)

new_layer = keras.layers.deserialize(config)

print(new_layer)

"""
Note that overriding `from_config` is unnecessary above for `MyDense` because
`hidden_units`, `kernel_initializer`, and `kernel_regularizer` are ints, strings, and a
built-in Keras object, respectively. This means that the default `from_config`
implementation of `cls(**config)` will work as intended.

For more complex objects, such as layers and models passed to `__init__`, for
example, you must explicitly deserialize these objects. Let's take a look at an example
of a model where a `from_config` override is necessary.

**Example:**
<a name="registration_example"></a>
"""


@keras.saving.register_keras_serializable(package="ComplexModels")
class CustomModel(keras.layers.Layer):
    def __init__(self, first_layer, second_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.first_layer = first_layer
        if second_layer is not None:
            self.second_layer = second_layer
        else:
            self.second_layer = keras.layers.Dense(8)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "first_layer": self.first_layer,
                "second_layer": self.second_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["first_layer"] = keras.layers.deserialize(config["first_layer"])
        config["second_layer"] = keras.layers.deserialize(config["second_layer"])
        return cls(**config)

    def call(self, inputs):
        return self.first_layer(self.second_layer(inputs))


# Let's make our first layer the custom layer from the previous example (MyDense)
inputs = keras.Input((32,))
outputs = CustomModel(first_layer=layer)(inputs)
model = keras.Model(inputs, outputs)

config = model.get_config()
new_model = keras.Model.from_config(config)

"""
<a name="custom_object_serialization"></a>
#### How custom objects are serialized

The serialization format has a special key for custom objects registered via
`@keras.saving.register_keras_serializable`. This `registered_name` key allows for easy
retrieval at loading/deserialization time while also allowing users to add custom naming.

Let's take a look at the config from serializing the custom layer `MyDense` we defined
above.

**Example**:
"""

layer = MyDense(
    units=16,
    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    kernel_initializer="ones",
)
config = keras.layers.serialize(layer)
print(config)

"""
As shown, the `registered_name` key contains the lookup information for the Keras master
list, including the package `MyLayers` and the custom name `KernelMult` that we gave in
the `@keras.saving.register_keras_serializable` decorator. Take a look again at the custom
class definition/registration [here](#registration_example).

Note that the `class_name` key contains the original name of the class, allowing for
proper re-initialization in `from_config`.

Additionally, note that the `module` key is `None` since this is a custom object.
"""
