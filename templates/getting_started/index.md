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
pip install --upgrade keras
```

You can check your local Keras version number via:

```python
import keras
print(keras.__version__)
```

To use Keras 3, you will also need to install a backend framework, such as JAX, TensorFlow, or PyTorch:

- [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Installing TensorFlow](https://www.tensorflow.org/install)
- [Installing PyTorch](https://pytorch.org/get-started/locally/)

Still on Keras 2? See [Using Keras 2 (legacy)](/guides/migrating_to_keras_3/#using-keras-2-legacy) in the migration guide.

### Installing KerasHub

KerasHub can be installed via pip:

```
pip install --upgrade keras-hub
pip install --upgrade keras
```

---

## Configuring your backend

You can export the environment variable `KERAS_BACKEND`
or you can edit your local config file at `~/.keras/keras.json` to configure your backend.
Keras supports multiple pluggable backends; current options include `"jax"`, `"torch"`, and `"tensorflow"`. Example:

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


### GPU dependencies

#### Colab or Kaggle
If you are running on Colab or Kaggle, the GPU should already be configured, with the correct CUDA version. 
Installing a newer version of CUDA on Colab or Kaggle is typically not possible. Even though pip installers exist,
they rely on a pre-installed NVIDIA driver and there is no way to update the driver on Colab or Kaggle.

#### Universal GPU environment

If you want to attempt to create a "universal environment" where any backend can use the GPU, we recommend following
[the dependency versions used by Colab](https://colab.research.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu)
(which seeks to solve this exact problem). You can install the CUDA driver [from here](https://developer.nvidia.com/cuda-downloads),
then pip install backends by following their respective CUDA installation instructions:
[Installing JAX](https://jax.readthedocs.io/en/latest/installation.html),
[Installing TensorFlow](https://www.tensorflow.org/install),
[Installing PyTorch](https://pytorch.org/get-started/locally/)

#### Most stable GPU environment

This setup is recommended  if you are a Keras contributor and are running Keras tests. It installs all backends but only
gives GPU access to one backend at a time, avoiding potentially conflicting dependency requirements between backends.
You can use the following backend-specific requirements files:

- [requirements-jax-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt)
- [requirements-tensorflow-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt)
- [requirements-torch-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt)

These install all CUDA-enabled dependencies via pip. They expect a NVIDIA driver to be preinstalled.
We recommend a clean python environment for each backend to avoid CUDA version mismatches.
As an example, here is how to create a JAX GPU environment with [Conda](https://docs.conda.io/en/latest/):

```
conda create -y -n keras-jax python=3.11
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
pip install --upgrade keras
```
---

## Compatibility matrix

Keras 3 requires Python 3.11 or newer. For the exact backend versions used in Keras CI, see the
[Keras requirements files](https://github.com/keras-team/keras/blob/master/requirements.txt).

### JAX compatibility

- `keras==3.16.0` is tested with the latest `jax` release
([requirements-jax-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt))

### TensorFlow compatibility

The following Keras + TensorFlow versions are compatible with each other:

To use Keras 2 (legacy):

- `tensorflow~=2.13.0` & `keras~=2.13.0`
- `tensorflow~=2.14.0` & `keras~=2.14.0`
- `tensorflow~=2.15.0` & `keras~=2.15.0`

To use Keras 3:

- `keras==3.16.0` is tested with `tensorflow==2.20.0`
([requirements-tensorflow-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt))

### PyTorch compatibility

- `keras==3.16.0` is tested with the latest `torch` release
([requirements-torch-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt))
