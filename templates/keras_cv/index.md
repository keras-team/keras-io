# KerasCV

<a class="github-button" href="https://github.com/keras-team/keras-cv" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-cv on GitHub">Star</a>

KerasCV is a toolbox of modular building blocks (layers, metrics, losses, data augmentation) that computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

<img style="width: 440px; max-width: 90%;" src="/img/keras-cv-augmentations.gif">

---
## Quick links

* [KerasCV developer guides](/guides/keras_cv/)
* [KerasCV API reference](/api/keras_cv/)
* [KerasCV on GitHub](https://github.com/keras-team/keras-cv)

---
## Installation

KerasCV requires **Python 3.7+** and **TensorFlow 2.9+**.

Install the latest release:

```
pip install keras-cv --upgrade
```

You can also check out other versions in our
[GitHub repository](https://github.com/keras-team/keras-cv/releases).

## Quick Introduction

Create a preprocessing pipeline:

```python
import keras_cv
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

augmenter = keras.Sequential(
  layers=[
      keras_cv.layers.RandomFlip(),
      keras_cv.layers.RandAugment(value_range=(0, 255)),
      keras_cv.layers.CutMix(),
      keras_cv.layers.MixUp()
    ]
)

def augment_data(images, labels):
  labels = tf.one_hot(labels, 3)
  inputs = {"images": images, "labels": labels}
  outputs = augmenter(inputs)
  return outputs['images'], outputs['labels']
```

Augment a `tf.data.Dataset`:

```python
dataset = tfds.load('rock_paper_scissors', as_supervised=True, split='train')
dataset = dataset.batch(64)
dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
```

Create a model:

```python
densenet = keras_cv.models.DenseNet121(
  include_rescaling=True,
  include_top=True,
  classes=3
)
densenet.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
```

Train your model:

```python
densenet.fit(dataset)
```

---
## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Ian, Stenbit and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
