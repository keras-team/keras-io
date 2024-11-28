# KerasHub

<a class="github-button" href="https://github.com/keras-team/keras-hub" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-hub on GitHub">Star</a>

**KerasHub** is a pretrained modeling library that aims to be simple, flexible,
and fast. The library provides [Keras 3](https://keras.io/keras_3/)
implementations of popular model architectures, paired with a collection of
pretrained checkpoints available on [Kaggle Models](https://kaggle.com/models/).
Models can be used for both training and inference, on any of the TensorFlow,
Jax, and Torch backends.

KerasHub is an extension of the core Keras API; KerasHub components are provided
as `keras.layers.Layer` and `keras.Model` implementations. If you are familiar
with Keras, congratulations! You already understand most of KerasHub.

---
## Quick links

* [Getting started with KerasHub](/keras_hub/getting_started/)
* [Developer guides](/keras_hub/guides)
* [API documentation](/keras_hub/api/)
* [KerasHub on GitHub](https://github.com/keras-team/keras-hub)
* [KerasHub models on Kaggle](https://www.kaggle.com/organizations/keras/models)
* [Pretrained model list](/keras_hub/presets/)

---
## Installation

To install the latest KerasHub release with Keras 3, simply run:

```
pip install --upgrade keras-hub
```

To install the latest nightly changes for both KerasHub and Keras, you can use
our nightly package.

```
pip install --upgrade keras-hub-nightly
```

Currently, installing KerasHub will always pull in TensorFlow for use of the
`tf.data` API for preprocessing. When pre-processing with `tf.data`, training
can still happen on any backend.

Visit the [core Keras getting started page](https://keras.io/getting_started/)
for more information on installing Keras 3, accelerator support, and
compatibility with different frameworks.

---
## Quickstart

Choose a backend:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"!
```

Import KerasHub and other libraries:

```python
import keras
import keras_hub
import numpy as np
import tensorflow_datasets as tfds
```

Load a resnet model and use it to predict a label for an image:

```python
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet",
    activation="softmax",
)
url = "https://upload.wikimedia.org/wikipedia/commons/a/aa/California_quail.jpg"
path = keras.utils.get_file(origin=url)
image = keras.utils.load_img(path)
preds = classifier.predict(np.array([image]))
print(keras_hub.utils.decode_imagenet_predictions(preds))
```

Load a Bert model and fine-tune it on IMDb movie reviews:

```python
classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    activation="softmax",
    num_classes=2,
)
imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
classifier.fit(imdb_train, validation_data=imdb_test)
preds = classifier.predict(["What an amazing movie!", "A total waste of time."])
print(preds)
```

---
## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

---
## Disclaimer

KerasHub provides access to pre-trained models via the `keras_hub.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind.

---
## Citing KerasHub

If KerasHub helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerashub2024,
  title={KerasHub},
  author={Watson, Matthew, and  Chollet, Fran\c{c}ois and Sreepathihalli,
  Divyashree, and Saadat, Samaneh and Sampath, Ramesh, and Rasskin, Gabriel and
  and Zhu, Scott and Singh, Varun and Wood, Luke and Tan, Zhenyu and Stenbit,
  Ian and Qian, Chen, and Bischof, Jonathan and others},
  year={2024},
  howpublished={\url{https://github.com/keras-team/keras-hub}},
}
```
