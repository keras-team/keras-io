# KerasCV

<a class="github-button" href="https://github.com/keras-team/keras-cv" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-cv on GitHub">Star</a>

KerasCV is a library of modular computer vision oriented Keras components.
These components include models, layers, metrics, losses, callbacks, and utility
functions.

KerasCV's primary goal is to provide a coherent, elegant, and pleasant API to
train state of the art computer vision models. Users should be able to train
state of the art models using only `Keras`, `KerasCV`, and TensorFlow core (i.e.
`tf.data`) components.

KerasCV can be understood as a horizontal extension of the Keras API: the
components are new first-party Keras objects (layers, metrics, etc.) that are
too specialized to be added to core Keras. They receive the same level of polish
and backwards compatibility guarantees as the core Keras API, and they are
maintained by the Keras team.

Our APIs assist in common computer vision tasks such as data-augmentation,
classification, object detection, image generation, and more. Applied computer
vision engineers can leverage KerasCV to quickly assemble production-grade,
state-of-the-art training and inference pipelines for all of these common tasks.

In addition to API consistency, KerasCV components aim to be mixed-precision
compatible, QAT compatible, XLA compilable, and TPU compatible. We also aim to
provide generic model optimization tools for deployment on devices such as
onboard GPUs, mobile, and edge chips.

<img style="width: 440px; max-width: 90%;" src="/img/keras-cv-augmentations.gif">

## Quick Links
- [List of available models and presets](https://keras.io/api/keras_cv/models/)
- [Developer Guides](https://keras.io/guides/keras_cv/)
- [Contributing Guide](https://github.com/keras-team/keras-cv/blob/master/.github/CONTRIBUTING.md)
- [Call for Contributions](https://github.com/keras-team/keras-cv/issues?q=is%3Aopen+is%3Aissue+label%3Acontribution-welcome)
- [API Design Guidelines](https://github.com/keras-team/keras-cv/blob/master/.github/API_DESIGN.md)

## Installation

To install the latest official release:

```
pip install keras-cv tensorflow --upgrade
```

To install the latest unreleased changes to the library, we recommend using
pip to install directly from the master branch on github:

```
pip install git+https://github.com/keras-team/keras-cv.git tensorflow --upgrade
```

## Quickstart

```python
import tensorflow as tf
from tensorflow import keras
import keras_cv
import tensorflow_datasets as tfds

# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ]
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = augmenter(inputs) if augment else inputs
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
  author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Bischof, Jonathan and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```