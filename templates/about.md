# About Keras

Keras is a deep learning API written in Python and capable of running on top of either [JAX](https://jax.readthedocs.io/),
[TensorFlow](https://github.com/tensorflow/tensorflow),
or [PyTorch](https://pytorch.org/).

Keras is:

- **Simple** -- but not simplistic. Keras reduces developer *cognitive load* to free you to focus on the parts of the problem that really matter.
- **Flexible** -- Keras adopts the principle of *progressive disclosure of complexity*: simple workflows should be quick and easy,
while arbitrarily advanced workflows should be *possible* via a clear path that builds upon what you've already learned.
- **Powerful** -- Keras provides industry-strength performance and scalability: it is used by organizations and companies including NASA, YouTube, or Waymo.

As a cross-framework API, Keras can be used to develop modular components that are compatible with any framework -- JAX, TensorFlow, or PyTorch.

---

## First contact with Keras

The core data structures of Keras are __layers__ and __models__.
The simplest type of model is the [`Sequential` model](/guides/sequential_model/), a linear stack of layers.
For more complex architectures, you should use the [Keras functional API](/guides/functional_api/),
which allows to build arbitrary graphs of layers, or [write models entirely from scratch via subclasssing](/guides/making_new_layers_and_models_via_subclassing/).

Here is the `Sequential` model:

```python
import keras

model = keras.Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras import layers

model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. The Keras philosophy is to keep simple things simple,
while allowing the user to be fully in control when they need to
(the ultimate control being the easy extensibility of the source code via subclassing).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Evaluate your test loss and metrics in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

What you just saw is the most elementary way to use Keras.

However, Keras is also a highly-flexible framework suitable to iterate on state-of-the-art research ideas.
Keras follows the principle of **progressive disclosure of complexity**: it makes it easy to get started,
yet it makes it possible to handle arbitrarily advanced use cases,
only requiring incremental learning at each step.

In much the same way that you were able to train and evaluate a simple neural network above in a few lines,
you can use Keras to quickly develop new training procedures or state-of-the-art model architectures.

Here's an example of a custom Keras layer -- which can be used in low-level
workflows in JAX, TensorFlow, or PyTorch, interchangeably:

```python
import keras
from keras import ops

class TokenAndPositionEmbedding(keras.Layer):
    def __init__(self, max_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = self.add_weight(
            shape=(vocab_size, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.position_embed = self.add_weight(
            shape=(max_length, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )

    def call(self, token_ids):
        # Embed positions
        length = token_ids.shape[-1]
        positions = ops.arange(0, length, dtype="int32")
        positions_vectors = ops.take(self.position_embed, positions, axis=0)
        # Embed tokens
        token_ids = ops.cast(token_ids, dtype="int32")
        token_vectors = ops.take(self.token_embed, token_ids, axis=0)
        # Sum both
        embed = token_vectors + positions_vectors
        # Normalize embeddings
        power_sum = ops.sum(ops.square(embed), axis=-1, keepdims=True)
        return embed / ops.sqrt(ops.maximum(power_sum, 1e-7))
```

For more in-depth tutorials about Keras, you can check out:

- [Introduction to Keras for engineers](/getting_started/intro_to_keras_for_engineers/)
- [Introduction to Keras for researchers](/getting_started/intro_to_keras_for_researchers/)
- [Developer guides](/guides/)

---

## Installation

You can install Keras from PyPI via `pip install keras`.
You can check your local Keras version number via `import keras; print(keras.__version__)`.

To use Keras, you will also need to install a backend framework -- either JAX, TensorFlow, or PyTorch.

To make sure you're able to run Keras on GPU, use the following backend-specific requirements files:

- [requirements-jax-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt)
- [requirements-tensorflow-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt)
- [requirements-torch-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt)

These install all CUDA-enabled dependencies via pip. They expect a NVIDIA driver to be preinstalled.
We recommend a clean python environment for each backend to avoid CUDA version mismatches.
As an example, here is how to create a JAX GPU environment with [Conda](https://docs.conda.io/en/latest/):

```
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
pip install keras
```

Note that it may not always be possible to use the GPU with multiple backends in the same environment due to conflicting
dependency requirements between backends.
The above requirements files only enable GPU usage for one target backends while keeping the other two backends CPU-only.
We recommend using [Conda](https://docs.conda.io/en/latest/) to maintain three separate environments `keras-jax`, `keras-tensorflow`, `keras-torch`.

If you want to attempt to create a "universal environment" where any backend can use the GPU, we recommend following
[the dependency versions used by Colab](https://colab.sandbox.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu)
(which seeks to solve this exact problem).

---

## Configuring your backend

You can export the environment variable `KERAS_BACKEND`
or you can edit your local config file at `~/.keras/keras.json`` to configure your backend.
Available backend options are: `"jax"`, `"tensorflow"`, `"torch"`. Example:

```
export KERAS_BACKEND="jax"
```

In Colab, you can do:

```
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```

**Note:** The backend must be configured before importing Keras, and the backend cannot be changed after the package has been imported.

---

## TensorFlow + Keras 2 backwards compatibility

From TensorFlow 2.0 to TensorFlow 2.15 (included), doing `pip install tensorflow` will also
install the corresponding version of Keras 2 -- for instance, `pip install tensorflow==2.14.0` will
install `keras==2.14.0`. That version of Keras is then available via both `import keras` and `from tensorflow import keras`
(the `tf.keras` namespace).

Starting with TensorFlow 2.16, doing `pip install tensorflow` will install Keras 3. When you have TensorFlow >= 2.16
and Keras 3, then by default `from tensorflow import keras` (`tf.keras`) will be Keras 3.

Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf-keras`/`tf_keras`
(note that `-` and `_` are equivalent in PyPI package names).
To use it, you can install it via `pip install tf_keras` then import it via `import tf_keras as keras`.

Should you want `tf.keras` to stay on Keras 2 after upgrading to TensorFlow 2.16+, you can configure your TensorFlow installation
so that `tf.keras` points to `tf_keras`. To achieve this:

1. Make sure to install `tf_keras`. Note that TensorFlow does not install by default.
2. Export the environment variable `TF_USE_LEGACY_KERAS=1`.

There are several ways to export the environment variable:

1. You can simply run the shell command `export TF_USE_LEGACY_KERAS=1` before launching the Python interpreter.
2. You can add `export TF_USE_LEGACY_KERAS=1` to your `.bashrc` file. That way the variable will still be exported when you restart your shell.
3. You can start your Python script with:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

These lines would need to be before any `import tensorflow` statement.

---

## Compatibility matrix

### JAX compatibility

The following Keras + JAX versions are compatible with each other:

- `jax==0.4.20` & `keras==3.0.0`

### TensorFlow compatibility

The following Keras + TensorFlow versions are compatible with each other:

To use Keras 2:

- `tensorflow==2.13.0` & `keras==2.13.0`
- `tensorflow==2.14.0` & `keras==2.14.0`
- `tensorflow==2.15.0` & `keras==2.15.0`

To use Keras 3:

- `tensorflow==2.15.0` & `keras==3.0.0`
- `tensorflow==2.16.0` & `keras==3.0.0`

### PyTorch compatibility

The following Keras + PyTorch versions are compatible with each other:

- `torch==2.1.0` & `keras==3.0.0`

---

## Support

You can ask questions and join the development discussion on the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/keras-team/keras/issues).
Make sure to read [our guidelines](https://github.com/keras-team/keras-io/blob/master/templates/contributing.md) first.

---

## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive dreamers with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling - who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).
