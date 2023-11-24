# Getting started with Keras

## Learning resources

Are you a machine learning engineer looking for a Keras introduction one-pager?
Read our guide [Introduction to Keras for engineers](/getting_started/intro_to_keras_for_engineers/).

Want to learn more about Keras 3 and its capabilities? See the [Keras 3 launch announcement](/keras_3/).

Are you looking for detailed guides covering in-depth usage of different parts of the Keras API?
Read our [Keras developer guides](/guides/).

Are you looking for tutorials showing Keras in action across a wide range of use cases?
See the [Keras code examples](/examples/): over 150 well-explained notebooks demonstrating Keras best practices
in computer vision, natural language processing, and generative AI.

---

## Installing Keras 3
 
You can install Keras from PyPI via:

```
pip install keras
```

You can check your local Keras version number via:

```python
import keras
print(keras.__version__)
```

To use Keras 3, you will also need to install a backend framework -- either JAX, TensorFlow, or PyTorch:

- [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Installing TensorFlow](https://www.tensorflow.org/install)
- [Installing PyTorch](https://pytorch.org/get-started/locally/)

If you install TensorFlow, critically, **you should reinstall Keras 3 afterwards**.
This is a temporary step while TensorFlow is pinned to Keras 2, and will no longer be necessary after TensorFlow 2.16.
The cause is that `tensorflow==2.15` will overwrite your Keras installation with `keras==2.15`.


### Installing KerasCV and KerasNLP

KerasCV and KerasNLP can be installed via pip:

```
pip install keras-cv
pip install keras-nlp
```

Critically, **you should reinstall Keras 3 after installing KerasNLP**.
This is a temporary step while TensorFlow is pinned to Keras 2, and will no longer be necessary after TensorFlow 2.16.
The cause is that `keras-nlp` depends on `tensorflow-text`, which will install `tensorflow==2.15`, which will
overwrite your Keras installation with `keras==2.15`.


### GPU dependencies

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
or you can edit your local config file at `~/.keras/keras.json` to configure your backend.
Available backend options are: `"jax"`, `"tensorflow"`, `"torch"`. Example:

```
export KERAS_BACKEND="jax"
```

In Colab, you can do:

```python
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

Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf_keras`
(or equivalently `tf-keras` -- note that `-` and `_` are equivalent in PyPI package names).
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
