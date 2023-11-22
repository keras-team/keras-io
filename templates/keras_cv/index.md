# KerasCV

<a class="github-button" href="https://github.com/keras-team/keras-cv" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-cv on GitHub">Star</a>

KerasCV is a library of modular computer vision components that work natively
with TensorFlow, JAX, or PyTorch. Built on Keras 3, these models, layers, 
metrics, callbacks, etc., can be trained and serialized in any framework and 
re-used in another without costly migrations.

KerasCV can be understood as a horizontal extension of the Keras API: the 
components are new first-party Keras objects that are too specialized to be 
added to core Keras. They receive the same level of polish and backwards 
compatibility guarantees as the core Keras API, and they are maintained by the 
Keras team.

Our APIs assist in common computer vision tasks such as data augmentation, 
classification, object detection, segmentation, image generation, and more.
Applied computer vision engineers can leverage KerasCV to quickly assemble 
production-grade, state-of-the-art training and inference pipelines for all of 
these common tasks.


<img style="width: 440px; max-width: 90%;" src="/img/keras-cv-augmentations.gif">

## Quick Links
- [List of available models and presets](https://keras.io/api/keras_cv/models/)
- [Developer Guides](https://keras.io/guides/keras_cv/)
- [Contributing Guide](https://github.com/keras-team/keras-cv/blob/master/.github/CONTRIBUTING.md)
- [API Design Guidelines](https://github.com/keras-team/keras-cv/blob/master/.github/API_DESIGN.md)

## Installation

KerasCV supports both Keras 2 and Keras 3. We recommend Keras 3 for all new
users, as it enables using KerasCV models and layers with JAX, TensorFlow and
PyTorch.

### Keras 2 Installation

To install the latest KerasCV release with Keras 2, simply run:

```
pip install --upgrade keras-cv tensorflow
```

### Keras 3 Installation

There are currently two ways to install Keras 3 with KerasCV. To install the
stable versions of KerasCV and Keras 3, you should install Keras 3 **after**
installing KerasCV. This is a temporary step while TensorFlow is pinned to
Keras 2, and will no longer be necessary after TensorFlow 2.16.

```
pip install --upgrade keras-cv tensorflow
pip install keras>=3
```

To install the latest changes nightly for KerasCV and Keras, you can use our
nightly package.

```
pip install --upgrade keras-cv-nightly tf-nightly
```

**Note:** Keras 3 will not function with TensorFlow 2.14 or earlier.

See [Getting started with Keras](/getting_started/) for more information on
installing Keras generally and compatibility with different frameworks.

## Quickstart

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
import keras

# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ],
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs)
    return outputs['images'], outputs['labels']

train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# Create a model using a pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Train your model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
```

## Disclaimer

KerasCV provides access to pre-trained models via the `keras_cv.models` API.
These pre-trained models are provided on an "as is" basis, without warranties or
conditions of any kind. The following underlying models are provided by third
parties, and are subject to separate licenses: StableDiffusion, Vision
Transfomer

## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Bischof, Jonathan and Zhu, Scott and Chollet, Fran\c{c}ois and Sreepathihalli, Divyashree and Sampath, Ramesh and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
