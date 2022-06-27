# KerasCV

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
from tensorflow import keras

preprocessing_model = keras.Sequential([
    keras_cv.layers.RandAugment(value_range=(0, 255))
    keras_cv.layers.CutMix(),
    keras_cv.layers.MixUp()
], name="preprocessing_model")
```

Augment a `tf.data.Dataset`:

```python
dataset = dataset.map(lambda images, labels: {"images": images, "labels": labels})
dataset = dataset.map(preprocessing_model)
dataset = dataset.map(lambda inputs: (inputs["images"], inputs["labels"]))
```

Create a model:

```python
densenet = keras_cv.models.DenseNet121(
  include_rescaling=True,
  include_top=True,
  num_classes=102
)
densenet.compile(optimizer='adam', metrics=['accuracy'])
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
  author={Wood, Luke and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
