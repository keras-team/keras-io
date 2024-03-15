# TensorFlow 2.16 with Keras

With the release of TensorFlow 2.16, `tf.keras` switched to [Keras 3](https://github.com/keras-team/keras), which is 95% compatible with TensorFlow 2, but drops support for TensorFlow 1. The detailed list of incompatibilities is documented [here](https://github.com/keras-team/keras/issues/18467). Most users can start using Keras 3 with very little changes to their code and get the benefits of backend optionality to run in JAX, PyTorch or TensorFlow and use them with any input data loaders - `tf.data` or PyTorch or Numpy data pipeline. For more details, checkout examples and guides at [keras.io](keras.io) 

In some cases, you may need to keep using Keras 2 (TensorFlow backend), for a variety of reasons - 

1. A dependent package may not be updated for Keras 3 or is pinned to use Keras 2 via `tf_keras`.
2. Codebases that are not actively developed may not want to switch to Keras 3.
3. Codebases that use TensorFlow 1, TensorFlow Estimator or other APIs that are not supported in Keras 3.

In these cases, use the following guidelines to stay on Keras 2 (via [tf-keras](https://github.com/keras-team/tf-keras)) with TensorFlow 2.16. Please note that Keras 2 will **continue to be maintained** and released with every TensorFlow release, for the foreseeable future. 


## Using TensorFlow with Keras 3

To use TensorFlow, JAX or PyTorch with Keras 3, checkout the getting started page in [keras.io](keras.io).  Starting with TensorFlow 2.16, Keras 3 is available directly via `import keras `(recommended) or as `from tensorflow import keras`. By default, Keras 3 uses the TensorFlow backend. Users can set the environment variable `KERAS_BACKEND` to `jax` or `torch` or `tensorflow` before importing Keras to use a specific backend.


## Using TensorFlow with Keras 2

### TensorFlow<= 2.14

Continue to access `keras` via `tf.keras`. No change to existing codebase and usages are required.

```python
pip install tensorflow~=2.14.0 # This will also install keras~=2.14
from tensorflow import keras  # That's Keras 2.14
```

### TensorFlow==2.15

Continue to access `keras` via `tf.keras`. No change to existing codebase and usages are required when Keras version is also 2.15 (default with TensorFlow 2.15). But if the user has upgraded to Keras 3, follow steps outlined for TensorFlow>=2.16.  

TensorFlow 2.15 introduced an environment variable [TF_USE_LEGACY_KERAS](https://github.com/tensorflow/tensorflow/blob/r2.15/tensorflow/python/util/lazy_loader.py#L96) which when set to "1" or "true" or "True", redirects usagges of `tf.keras` to `tf_keras==2.15` and not `keras==2.15`. This would have no effect if the user uses `tf.keras`, but would matter if the user uses `import keras` directly.  In other words, `from tensorflow import keras`, will either use `keras 2.15` or `tf_keras 2.15` depending on whether env. variable `TF_USE_LEGACY_KERAS` is set. 


#### Using TensorFlow 2.15 with Keras 2.15 (Default)
```python
pip install tensorflow~=2.15  # This will also install keras~=2.15

from tensorflow import keras  # That's your locally installed Keras, which is now 2.15
```

#### Using TensorFlow 2.15 with Keras 3
```python
pip install tensorflow~=2.15  # This will also install keras~=2.15
pip install keras –-upgrade  # Now your keras version will be 3

from tensorflow import keras  # That's your locally installed Keras, which is now Keras 3
import keras  # Alternatively, import Keras directly using Keras 3 with JAX or PyTorch backends
```

#### Using Tensorflow 2.15 with tf-keras 2.15 (Uncommon)
```python
pip install tensorflow~=2.15  # This will also install keras~=2.15
pip install tf_keras~=2.15

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # switches tf.keras to use tf_keras
from tensorflow import keras  # That's now tf_keras, which is the API-equivalent to Keras 2
```

It's recommended that users use `tf.keras` or `from tensorflow import keras` wherever possible and avoid usage of `import keras` with TensorFlow 2.15 so that `tf.keras` and `keras` consistently refers to either `keras 2.15` or `tf_keras 2.15` and they are identical.

### TensorFlow>=2.16

With TensorFlow 2.16+, `tf.keras` usage defaults to Keras 3. Also, `import keras` will import Keras 3. To continue to use Keras 2, users of the package will need to set the environment variable in their codebases `TF_USE_LEGACY_KERAS` to “1 or True or true”,  Setting this environment variable will switch usage of `tf.keras` to `tf_keras`.  Library authors who want to keep working with Keras 2 shouldn't rely on this environment variable being set in users environment and would need to update their codebases by changing usages of `tf.keras` to `tf_keras` for public API and use `tf_keras.src` when referring to private source files. 

#### Using TensorFlow 2.16 with Keras 3 (Default)
```python
pip install tensorflow~=2.16.0  # This will also install keras 3

from tensorflow import keras  # That's Keras 3!
import keras  # Import Keras directly using Keras 3 with JAX or PyTorch backends
```

#### Using TensorFlow 2.15 with Keras 2
```python
pip install tensorflow~=2.16.0  # This will also install keras 3
pip install tf_keras  # Installs tf_keras compatible with TensorFlow 2.16

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # switches tf.keras to use tf_keras
from tensorflow import keras  # That's now tf_keras, which is the API-equivalent to Keras 2
```

Please note that this environment variable must be set before importing keras in the python runtime environment for it to take effect. So it's recommended that users `import os; os.environ["TF_USE_LEGACY_KERAS"]=1` at the top of the main python program if they intend to use Keras 2 with TensorFlow 2.16 or set in the users terminal runtime environment. Users cannot switch between Keras 2 and Keras 3 within the same runtime.
