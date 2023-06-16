"""
Title: Save, serialize, and export
Authors: Neel Kovelamudi, Francois Chollet
Date created: 2023/06/14
Last modified: 2023/06/14
Description: Complete guide to saving, serializing, and exporting models.
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

- a JSON-based configuration file (config.json): Records of model, layer, and
other trackables' configuration.
- a H5-based state file, such as `model.weights.h5` (for the whole model),
with directory keys for layers and subcomponents and their weights.
- a Metadata file in JSON, storing things such as your Keras version.

Let's take a look at how this works.
"""

"""
## How to save and load a model

If you only have 10 seconds to read this guide, here's what you need to know.

**Saving a Keras model:**

```python
model = ...  # Get model (Sequential, Functional Model, or Model subclass)
model.save('path/to/location.keras')  # Add .keras extension at end of file path
```

**Loading the model back:**

```python
from tensorflow import keras
model = keras.models.load_model('path/to/location.keras')
```

Now, let's look at the details.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
import keras

"""
## Saving

"""

"""

You can save an entire model to a single artifact. It will include:

- The model's architecture/config
- The model's weight values (which were learned during training)
- The model's compilation information (if `compile()` was called)
- The optimizer and its state, if any (this enables you to restart training
where you left)

#### APIs

- `model.save()` or `keras.models.save_model()`
- `keras.models.load_model()`

The recommended format is default with the `.keras` extension. There are,
however, two legacy formats that are available: the **TensorFlow
SavedModel format** and the older Keras **H5 format**.<br>
You can switch to the SavedModel format by:

- Passing `save_format='tf'` to `save()`
- Passing a filename without an extension

You can switch to the H5 format by:
- Passing `save_format='h5'` to `save()`
- Passing a filename that ends in `.h5`

#### The `.keras` format **(recommended)**

The new Keras v3 saving format, marked by the `.keras` extension, is a more
simple, efficient format consisting of the model's configuration (its
architecture and specifications) and the model's states (weights, variables,
etc.), together ensuring exact model restoration at loading time.
"""

"""
### Simple saving and reloading in Python

We will walk through a simple example saving and reloading a model with
the `.keras` format.

**Example:**
"""


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
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
"""

"""
This section covers the basic workflows for handling basic custom layers, functions, and
models in Keras saving.

For custom objects, you **must** define a `get_config()` method. If arguments passed to
`__init__()` are Python objects (anything other than base types like ints, strings,
etc.), you **must** also explicitly deserialize these arguments in the `from_config`
method. Please see the [Defining the config methods section](#config_methods) for more
details and examples.

The saved `.keras` file is lightweight and does not store the Python code for custom
objects. Therefore, to reload the model, `load_model` requires access to the definition
of any custom objects used through one of the following methods:

1. Registering custom objects **(preferred)**,
2. Passing custom objects directly when loading, or
3. Using a custom object scope
"""

"""
Below are examples of each workflow:
"""

"""
#### Registering custom objects (**preferred**)
"""

"""
This is the preferred method, as custom object registration greatly simplifies saving and
loading code. Adding the `@keras.utils.register_keras_serializable` decorator to the
class definition of a custom object registers the object globally in a master list,
allowing Keras to recognize the object when loading the model.
"""

"""
Let's create a custom model involving both a custom layer and a custom activation
function to demonstrate this.

**Example:**
"""

# Clear all previously registered custom objects
keras.utils.get_custom_objects().clear()

# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to `Custom` and the name defaults to
# the class name.
@keras.utils.register_keras_serializable(package="MyLayers")
class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.utils.register_keras_serializable(package="MyFunctions", name="SquareFn")
def custom_fn(x):
    return x**2


# Create the model.
def get_model():
    inputs = keras.Input(shape=(4, 4))
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
model.save("my_reg_custom_model.keras")

# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model("my_reg_custom_model.keras")

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
model.save("my_custom_model.keras")

# Upon loading, pass a dict containing the custom objects used in the
# `custom_objects` argument of `keras.models.load_model()`.
reconstructed_model = keras.models.load_model(
    "my_custom_model.keras",
    custom_objects={"CustomLayer": CustomLayer, "custom_fn": custom_fn},
)

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)


"""
#### Using a custom object scope
"""

"""
Any code within the custom object scope will be able to recognize the custom objects
passed to the scope argument. Therefore, loading the model within the scope will allow
the loading of our custom objects.

**Example:**
"""

model = get_model()
model = train_model(model)
model.save("my_scoped_custom_model.keras")

# Pass the custom objects dictionary to a custom object scope and place
# the `keras.models.load_model()` call within the scope.
custom_objects = {"CustomLayer": CustomLayer, "custom_fn": custom_fn}

with keras.utils.custom_object_scope(custom_objects):
    reconstructed_model = keras.models.load_model("my_scoped_custom_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

"""
### Model architecture methods

The model's configuration (or architecture) specifies what layers the model
contains, and how these layers are connected. If you have the configuration of a model,
then the model can be created with a freshly initialized state (no weights or compilation
information).
"""

"""
#### APIs

- `keras.models.clone_model`
- `get_config()` and `from_config()`
- `keras.models.model_to_json()` and `keras.models.model_from_json()`
"""

"""
#### In-memory model cloning

You can also do in-memory cloning of a model via `keras.models.clone_model()`.
This is equivalent to getting the config then recreating the model from its config
(so it does not preserve compilation information or layer weights values).

**Example:**
"""

new_model = keras.models.clone_model(model)

"""
#### `get_config()` and `from_config()`

Calling `model.get_config()` or `layer.get_config()` will return a Python dict containing
the configuration of the model or layer, respectively. You should define `get_config()`
to contain arguments needed for the `__init__()` of the model or layer. At loading time,
the `from_config(config)` method will then call `__init__()` with these arguments to
reconstruct the model or layer.



**Layer example:**
"""

layer = keras.layers.Dense(3, activation="relu")
layer_config = layer.get_config()
layer_config

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
### Model weights methods

You can choose to only save & load a model's weights. This can be useful if:

- You only need the model for inference: in this case you won't need to
restart training, so you don't need the compilation information or optimizer state.
- You are doing transfer learning: in this case you will be training a new model
reusing the state of a prior model, so you don't need the compilation
information of the prior model.
"""

"""
#### APIs for in-memory weight transfer

Weights can be copied between different objects by using `get_weights`
and `set_weights`:

* `keras.layers.Layer.get_weights()`: Returns a list of numpy arrays.
* `keras.layers.Layer.set_weights()`: Sets the model weights to the values
in the `weights` argument.

Examples below.


***Transfering weights from one layer to another, in memory***
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
***Transfering weights from one model to another model with a
compatible architecture, in memory***
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
subclassed_model(tf.ones((1, 784)))

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

Weights can be saved to disk by calling `model.save_weights`
in the following formats:

* TensorFlow Checkpoint
* HDF5

The default format for `model.save_weights` is TensorFlow checkpoint.
There are two ways to specify the save format:

1. `save_format` argument: Set the value to `save_format="tf"` or `save_format="h5"`.
2. `path` argument: If the path ends with `.h5` or `.hdf5`,
then the HDF5 format is used. Other suffixes will result in a TensorFlow
checkpoint unless `save_format` is set.

There is also an option of retrieving weights as in-memory numpy arrays.
Each API has its pros and cons which are detailed below.
"""

"""
#### **TF Checkpoint format**

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
sequential_model.save_weights("ckpt")
load_status = sequential_model.load_weights("ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
load_status.assert_consumed()

"""
##### **Format details**

The TensorFlow Checkpoint format saves and restores the weights using
object attribute names. For instance, consider the `keras.layers.Dense` layer.
The layer contains two weights: `dense.kernel` and `dense.bias`.
When the layer is saved to the `tf` format, the resulting checkpoint contains the keys
`"kernel"` and `"bias"` and their corresponding weight values.
For more information see
["Loading mechanics" in the TF Checkpoint
guide](https://www.tensorflow.org/guide/checkpoint#loading_mechanics).

Note that attribute/graph edge is named after **the name used in parent object,
not the name of the variable**. Consider the `CustomLayer` in the example below.
The variable `CustomLayer.var` is saved with `"var"` as part of key, not `"var_a"`.
"""


class CustomLayer(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name="var_a")


layer = CustomLayer(5)
layer_ckpt = tf.train.Checkpoint(layer=layer).save("custom_layer")

ckpt_reader = tf.train.load_checkpoint(layer_ckpt)

ckpt_reader.get_variable_to_dtype_map()

"""
##### **Transfer learning example**

Essentially, as long as two models have the same architecture,
they are able to share the same checkpoint.

**Example:**
"""

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

# Extract a portion of the functional model defined in the Setup section.
# The following lines produce a new model that excludes the final output
# layer of the functional model.
pretrained = keras.Model(
    functional_model.inputs, functional_model.layers[-1].input, name="pretrained_model"
)
# Randomly assign "trained" weights.
for w in pretrained.weights:
    w.assign(tf.random.normal(w.shape))
pretrained.save_weights("pretrained_ckpt")
pretrained.summary()

"""
Now let's create a new functional model with a different output dimension and load the
pretrained weights into that model:
"""

# Assume this is a separate program where only 'pretrained_ckpt' exists.
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(5, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="new_model")

# Load the weights from pretrained_ckpt into model.
model.load_weights("pretrained_ckpt")

# Check that all of the pretrained weights have been loaded.
for a, b in zip(pretrained.weights, model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

print("\n", "-" * 50)
model.summary()


"""
The same works with a Sequential model:
"""

# Recreate the pretrained model, and load the saved weights.
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
pretrained_model = keras.Model(inputs=inputs, outputs=x, name="pretrained")

# Sequential example:
model = keras.Sequential([pretrained_model, keras.layers.Dense(5, name="predictions")])
model.summary()

pretrained_model.load_weights("pretrained_ckpt")

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error,
# but will *not* work as expected. If you inspect the weights, you'll see that
# none of the weights will have loaded. `pretrained_model.load_weights()` is the
# correct method to call.

"""
It is generally recommended to stick to the same API for building models. If you
switch between Sequential and Functional, or Functional and subclassed,
etc., then always rebuild the pre-trained model and load the pre-trained
weights to that model.
"""

"""
The next question is, how can weights be saved and loaded to different models
if the model architectures are quite different?
The solution is to use `tf.train.Checkpoint` to save and restore the exact
layers/variables.

**Example:**
"""

# Create a subclassed model that essentially uses functional_model's first
# and last layers.
# First, save the weights of functional_model's first and last dense layers.
first_dense = functional_model.layers[1]
last_dense = functional_model.layers[-1]
ckpt_path = tf.train.Checkpoint(
    dense=first_dense, kernel=last_dense.kernel, bias=last_dense.bias
).save("ckpt")

# Define the subclassed model.
class ContrivedModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.first_dense = keras.layers.Dense(64)
        self.kernel = self.add_weight("kernel", shape=(64, 10))
        self.bias = self.add_weight("bias", shape=(10,))

    def call(self, inputs):
        x = self.first_dense(inputs)
        return tf.matmul(x, self.kernel) + self.bias


model = ContrivedModel()
# Call model on inputs to create the variables of the dense layer.
_ = model(tf.ones((1, 784)))

# Create a Checkpoint with the same structure as before, and load the weights.
tf.train.Checkpoint(
    dense=model.first_dense, kernel=model.kernel, bias=model.bias
).restore(ckpt_path).assert_consumed()

"""
#### **HDF5 format**

The HDF5 format contains weights grouped by layer names.
The weights are lists ordered by concatenating the list of trainable weights
to the list of non-trainable weights (same as `layer.weights`).
Thus, a model can use a hdf5 checkpoint if it has the same layers and trainable
statuses as saved in the checkpoint.

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
sequential_model.save_weights("weights.h5")
sequential_model.load_weights("weights.h5")

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

When loading pretrained weights from HDF5, it is recommended to load
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
functional_model.save_weights("pretrained_weights.h5")

# In a separate program:
pretrained_model = create_functional_model()
pretrained_model.load_weights("pretrained_weights.h5")

# Create a new model by extracting layers from the original model:
extracted_layers = pretrained_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(5, name="dense_3"))
model = keras.Sequential(extracted_layers)
model.summary()

"""
## Serialization
"""

"""
All layers, models, and other objects can be serialized or deserialized directly.

#### APIs

- `keras.layers.serialize()` and `keras.layers.deserialize`
- `keras.utils.serialize_keras_object()` and `keras.utils.deserialize_keras_object`

Each of the following modules in `keras` contain their own `serialize()` and
`deserialize()` APIs:
- `layers`
- `activations`
- `constraints`
- `initializers`
- `losses`
- `metrics`
- `optimizers`
- `regularizers`
"""

"""
### Simple serialization and deserialization
"""

"""
The `keras.utils.serialize_keras_object()` and `keras.utils.deserialize_keras_object()`
APIs are general-purpose APIs that can be used to serialize or deserialize any Keras
object and any custom object. It is at the foundation of saving model architecture and is
behind all `serialize()`/`deserialize()` calls in keras.

**Example**:
"""

my_reg = keras.regularizers.L1(0.005)
config = keras.utils.serialize_keras_object(my_reg)
config

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

new_reg = keras.utils.deserialize_keras_object(config)

"""
Let's now take a look at the `serialize()`/`deserialize()` calls specific to a module,
such as `keras.layers`.
Note that these calls are interchangeable with the `serialize_keras_object` and
`deserialize_keras_object` APIs we just discussed.

The `keras.layers.serialize()` and `keras.layers.deserialize()` APIs can be used to
serialize or deserialize all layers, including custom ones.

**Example**:
"""

layer = keras.layers.Dense(16)
config = keras.layers.serialize(layer)
new_layer = keras.layers.deserialize(config)

"""
**Example (with custom layer)**:
"""


class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


layer = CustomLayer(0.5)
config = keras.layers.serialize(layer)

new_layer = keras.layers.deserialize(
    config, custom_objects={"CustomLayer": CustomLayer}
)

"""
### Handling custom objects
"""

"""
<a name="config_methods"></a>
#### Defining the config methods

Specifications:

* `get_config` should return a JSON-serializable dictionary in order to be
compatible with the Keras architecture- and model-saving APIs.
* `from_config(config)` (`classmethod`) should return a new layer or model
object that is created from the config.
The default implementation returns `cls(**config)`.

**NOTE**:  If all your constructor arguments are already serializable, e.g. strings and
ints, or non-custom Keras objects, overriding `from_config` is not necessary. However,
for more complex objects such as layers or models passed to `__init__`, deserialization
must be handled explicitly either in `__init__` itself or overriding the `from_config()`
method.


**Example:**
"""


@keras.utils.register_keras_serializable(package="MyLayers", name="KernelMult")
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
        unused_batch_size, input_units = input_shape
        self.kernel = self.add_weight(
            "kernel",
            [input_units, self.hidden_units],
            dtype=tf.float32,
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


layer = MyDense(units=16, kernel_regularizer="l1", kernel_initializer="ones")
layer3 = MyDense(units=64, nested_model=layer)

config = keras.layers.serialize(layer3)

config

new_layer = keras.layers.deserialize(config)

new_layer

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


@keras.utils.register_keras_serializable(package="ComplexModels")
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
        # Note that you can also use `keras.utils.deserialize_keras_object` here
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
`@keras.utils.register_keras_serializable`. This `registered_name` key allows for easy
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
config

"""
As shown, the `registered_name` key contains the lookup information for the Keras master
list, including the package `MyLayers` and the custom name `KernelMult` that we gave in
the `@keras.utils.register_keras_serializable` decorator. Take a look again at the custom
class definition/registration [here](#registration_example).

Note that the `class_name` key contains the original name of the class, allowing for
proper re-initialization in `from_config`.

Additionally, note that the `module` key is `None` since this is a custom object.
"""

"""
## Exporting
"""

"""
The new export library allows you to create a lightweight version of your model for
inferencing that contains the model's forward pass only (the `call()` method). This TF
SavedModel artifact can then be served via TF-Serving, and all original code of the model
(including custom layers) are no longer necessary to reload the artifact--it is entirely
standalone!

#### APIs

- `model.export()`, which exports the model to a lightweight SavedModel artifact for
inference
- `artifact.serve()`, which calls the exported artifact's forward pass

Lower level API for customization:
- `keras.export.ExportArchive`, which can be used to customize the serving endpoints.
This is used internally by `model.export()`.


"""

"""
### Simple exporting with .export()
"""

"""
Let's go through a simple example of `model.export()` using a Functional model.

**Example:**


"""

x = keras.Input(shape=(16,))
mid = keras.layers.Dense(8, activation="relu")(x)
norm = keras.layers.BatchNormalization()(mid)
y = keras.layers.Dense(1, activation="sigmoid")(norm)
model = keras.Model(x, y)

input_data = tf.random.normal((8, 16))
output_data = model(input_data)  # **NOTE**: Make sure your model is built!

# Export the model as a SavedModel artifact in a filepath.
model.export("exported_model")

# Reload the SavedModel artifact
reloaded_artifact = tf.saved_model.load("exported_model")

# Use the `.serve()` endpoint to call the forward pass on the input data
new_output_data = reloaded_artifact.serve(input_data)

np.testing.assert_allclose(output_data.numpy(), new_output_data.numpy(), atol=1e-6)

"""
### Customizing export artifacts with ExportArchive

The `ExportArchive` object allows you to customize exporting the model and add additional
endpoints for serving. Here are its associated APIs:

- `track()` to register the layer(s) or model(s) to be used,
- `add_endpoint()` method to register a new serving endpoint.
- `write_out()` method to save the artifact.
- `add_variable_collection` method to register a set of variables to be retrieved after
reloading.

By default, `model.export("path/to/location")` does the following:
```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],  # `input_signature`
changes depending on model.
)
export_archive.write_out("path/to/location")
```

Let's look at an example customizing this for a MultiHeadAttention layer.

**Example:**
"""

from keras.src.export import export_lib

layer = keras.layers.MultiHeadAttention(2, 2)
x1 = tf.random.normal((3, 2, 2))
x2 = tf.random.normal((3, 2, 2))
ref_output = layer(x1, x2).numpy()  # **NOTE**: Make sure layer is built!

export_archive = export_lib.ExportArchive()  # Instantiate ExportArchive object
export_archive.track(layer)  # Register the layer to be used
export_archive.add_endpoint(  # New endpoint `call` corresponding to `model.call`
    "call",
    layer.call,
    input_signature=[  # input signature corresponding to 2 inputs
        tf.TensorSpec(
            shape=(None, 2, 2),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(None, 2, 2),
            dtype=tf.float32,
        ),
    ],
)

# Register the layer weights as a set of variables to be retrieved
export_archive.add_variable_collection("my_vars", layer.weights)
np.testing.assert_equal(len(export_archive.my_vars), 8)
# weights corresponding to 2 inputs, each of which are 2*2

# Save the artifact
export_archive.write_out("exported_mha_layer")

# Reload the artifact
revived_layer = tf.saved_model.load("exported_mha_layer")
np.testing.assert_allclose(
    ref_output,
    revived_layer.call(query=x1, value=x2).numpy(),
    atol=1e-6,
)
np.testing.assert_equal(len(revived_layer.my_vars), 8)
