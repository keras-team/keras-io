# Keras Core: Keras for TensorFlow, JAX, and PyTorch

**Keras Core** is a preview release of Keras 3.0. Keras 3.0 will be generally available in Fall 2023
and will replace `tf.keras`.

Keras Core makes it possible to run Keras workflows on top of TensorFlow, JAX, and PyTorch.
It also enables you to seamlessly integrate Keras components (like layers, models, or metrics)
as part of low-level TensorFlow, JAX, and PyTorch workflows.

- Learn more in the [Keras Core announcement](/keras_core/announcement/).
- Get started with the [Keras Core introduction Colab](/keras_core/guides/getting_started_with_keras_core/).
- Contribute to the project on GitHub at [github.com/keras-team/keras-core](https://github.com/keras-team/keras-core).

<a class="github-button" href="https://github.com/keras-team/keras-core" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-core on GitHub">Star</a>

---

## Key benefits

- **Always get the best performance for your models.** In our benchmarks,
we found that JAX typically delivers the best training and inference performance
on GPU, TPU, and CPU -- but results vary from model to model, as non-XLA
TensorFlow is occasionally faster on GPU. The ability to dynamically select
the backend that will deliver the best performance for your model
*without having to change anything to your code* means you're always guaranteed
to train and serve with the highest achievable efficiency.
- **Maximize available ecosystem surface for your models.** Any Keras Core
model can be instantiated as a PyTorch `Module`, can be exported as a TensorFlow
`SavedModel`, or can be instantiated as a stateless JAX function. That means
that you can use your Keras Core models with PyTorch ecosystem packages,
with the full range of TensorFlow deployment & production tools, and with
JAX large-scale TPU training infrastructure. Write one `model.py` using
Keras Core APIs, and get access to everything the ML world has to offer.
- **Maximize distribution for your open-source model releases.** Want to
release a pretrained model? Want as many people as possible
to be able to use it? If you implement it in pure TensorFlow or PyTorch,
it will be usable by roughly half of the market.
If you implement it in Keras Core, it is instantly usable by anyone regardless
of their framework of choice (even if they're not Keras users).
Twice the impact at no added development cost.
- **Use data pipelines from any source.** The Keras Core
`fit()`/`evaluate()`/`predict()` routines are compatible with `tf.data.Dataset` objects,
with PyTorch `DataLoader` objects, with NumPy arrays, Pandas dataframes --
regardless of the backend you're using. You can train a Keras Core + TensorFlow
model on a PyTorch `DataLoader` or train a Keras Core + PyTorch model on a
`tf.data.Dataset`.

---

## Installation

You can install Keras Core via `pip`:

```
pip install keras-core
```

You can then import it in Python:

```python
import keras_core as keras
```

Note that Keras Core requires `tensorflow` to be installed, as it uses the `tf.nest` Python datastructure
preprocessing utility. In the future, this dependency will be removed, and you will only need
to install the specific backend frameworks you intend to use.

**Note on cuDNN:** If you intend to use Keras Core on GPU with multiple frameworks in the same environment,
be mindful that the latest versions of the three backend frameworks tend to each require a different cuDNN
version. Hence you will need to find a combination of framework versions that all work with the same
cuDNN version (meanwhile, Keras Core is expected to work with both the latest and prior version of
each backend framework, which gives you flexibility).

You can find the right combination of backend framework versions and cuDNN version by
looking at [what's installed by default on Colab](https://colab.research.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu?usp=sharing),
since Colab faces the same version compatibility issue.

---

## Configuring your backend

To configure which backend the `keras-core` package should use, can export the environment variable `KERAS_BACKEND`
or you can edit your local config file at `~/.keras/keras.json` (it gets automatically created when you import `keras_core`).
Available backend options are: `"tensorflow"`, `"jax"`, `"torch"`.

Example:

```
$ export KERAS_BACKEND="jax"
$ python train.py
```

Or alternatively:

```
$ KERAS_BACKEND=jax python train.py
```

In Colab, you can use:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
```

Note that the backend must be configured before importing `keras_core`,
and the backend cannot be changed after the package has been imported.

If you always want to use the same backend, just edit `~/.keras/keras.json` to 
specify your default backend.

---

## Using KerasCV and KerasNLP with Keras Core

As of version `0.6.0`, KerasCV and KerasNLP support multiple backends with Keras 
Core out of the box. There are two ways to configure these libraries to run with 
multi-backend support. Using KerasCV as an example:

1. Via the `KERAS_BACKEND` environment variable. If set, then KerasCV will be 
using Keras Core with the backend specified (e.g., `KERAS_BACKEND=jax`).
2. Via the `.keras/keras.json` and `.keras/keras_cv.json` config files (which 
are automatically created the first time you import KerasCV):
   - Set your backend of choice in `.keras/keras.json`; e.g., `"backend": "jax"`. 
   - Set `"multi_backend": True` in `.keras/keras_cv.json`.

Once that configuration step is done, you can just import KerasCV and start 
using it on top of your backend of choice:

```python
import keras_cv
import keras_core as keras

filepath = keras.utils.get_file(origin="https://i.imgur.com/gCNcJJI.jpg")
image = np.array(keras.utils.load_img(filepath))
image_resized = ops.image.resize(image, (640, 640))[None, ...]

model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc",
    bounding_box_format="xywh",
)
predictions = model.predict(image_resized)
```

KerasNLP works the same way once configured with `.keras/keras_nlp.json`. For 
example:

```python
import keras_nlp

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
gpt2_lm.generate("My trip to Yosemite was", max_length=200)
```

Until Keras Core is officially released as Keras 3.0, KerasCV and KerasNLP will 
use `tf.keras` as the default backend. To restore this default behavior, simply 
`unset KERAS_BACKEND` and ensure that  `"multi_backend": False` or is unset in 
`.keras/keras_cv.json` or `.keras/keras_nlp.json`. You will need to restart the 
Python runtime for changes to take effect.