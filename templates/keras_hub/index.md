# KerasHub

<a class="github-button" href="https://github.com/keras-team/keras-hub" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-hub on GitHub">Star</a>

**KerasHub** is a pretrained modeling library that aims to be simple, flexible,
and fast. The library provides [Keras 3](https://keras.io/keras_3/)
implementations of popular model archtictures, paired with a collection of
pretrained checkpoints available on [Kaggle Models](https://kaggle.com/models/).
Models can be use for both training and inference, on any of the TensorFlow,
Jax, and Torch backends.

KerasHub is an extension of the core Keras API; KerasHub components are provide
as [`Layers`](/api/layers/) and [`Models`](/api/models/). If you are familiar
with Keras, congratulations! You already understand most of KerasHub.

See our [Getting Started guide](/guides/keras_hub/getting_started)
to start learning our API. We welcome
[contributions](https://github.com/keras-team/keras-hub/issues/1835).

---
## Quick links

* [KerasHub API reference](/api/keras_hub/)
* [KerasHub on GitHub](https://github.com/keras-team/keras-hub)
* [KerasHub models on Kaggle](https://www.kaggle.com/organizations/keras/models)
* [List of available pretrained models](/api/keras_hub/models/)

## Guides
* [Classification with KerasHub](/guides/keras_hub/classification_with_keras_hub/)
* [Segment Anything in KerasHub](/guides/keras_hub/segment_anything_in_keras_hub/)
* [Stable Diffusion 3 in KerasHub](/guides/keras_hub/stable_diffusion_3_in_keras_hub/)

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

Note that currently, installing KerasHub will always pull in TensorFlow for use
of the `tf.data` API for preprocessing. Even when pre-processing with `tf.data`,
training can still happen on any backend.

Read [Getting started with Keras](https://keras.io/getting_started/) for more
information on installing Keras 3 and compatibility with different frameworks.

**Note:** We recommend using KerasHub with TensorFlow 2.16 or later, as TF 2.16
packages Keras 3 by default.

---
## Quickstart

Below is a quick example using ResNet to predict an image, and BERT to train a
classifier:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"!

import keras
import keras_hub
import numpy as np
import tensorflow_datasets as tfds

# Load a ResNet model.
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet",
    activation="softmax",
)
# Predict a label for a single image.
image_url = "https://upload.wikimedia.org/wikipedia/commons/a/aa/California_quail.jpg"
image_path = keras.utils.get_file(origin=image_url)
image = keras.utils.load_img(image_path)
batch = np.array([image])
preds = classifier.predict(batch)
print(keras_hub.utils.decode_imagenet_predictions(preds))

# Load a BERT model.
classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    activation="softmax",
    num_classes=2,
)

# Fine-tune on IMDb movie reviews.
imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
preds = classifier.predict(
    ["What an amazing movie!", "A total waste of my time."]
)
print(preds)
```

---
## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

## Disclaimer

KerasHub provides access to pre-trained models via the `keras_hub.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind.

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
