# Classification with KerasHub

**Author:** [Gowtham Paimagam](https://github.com/gowthamkpr), [lukewood](https://lukewood.xyz)<br>
**Date created:** 09/24/2024<br>
**Last modified:** 10/04/2024<br>
**Description:** Use KerasHub to train powerful image classifiers.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/classification_with_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/classification_with_keras_hub.py)



Classification is the process of predicting a categorical label for a given
input image.
While classification is a relatively straightforward computer vision task,
modern approaches still are built of several complex components.
Luckily, Keras provides APIs to construct commonly used components.

This guide demonstrates Keras' modular approach to solving image
classification problems at three levels of complexity:

- Inference with a pretrained classifier
- Fine-tuning a pretrained backbone
- Training a image classifier from scratch

KerasHub uses Keras 3 to work with any of TensorFlow, PyTorch or Jax. In the
guide below, we will use the `jax` backend. This guide runs in
TensorFlow or PyTorch backends with zero changes, simply update the
`KERAS_BACKEND` below.

We use Professor Keras, the official Keras mascot, as a
visual reference for the complexity of the material:

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_evolution.png)


```python
!!pip install -q git+https://github.com/keras-team/keras-hub.git
!!pip install -q --upgrade keras  # Upgrade to Keras 3.
```




```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import json
import math
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import losses
from keras import ops
from keras import optimizers
from keras.optimizers import schedules
from keras import metrics
import keras_hub

# Import tensorflow for `tf.data` and its preprocessing functions
import tensorflow as tf
import tensorflow_datasets as tfds

```
<div class="k-default-codeblock">
```
['',
 '\x1b[1m[\x1b[0m\x1b[34;49mnotice\x1b[0m\x1b[1;39;49m]\x1b[0m\x1b[39;49m A new release of pip is available: \x1b[0m\x1b[31;49m23.0.1\x1b[0m\x1b[39;49m -> \x1b[0m\x1b[32;49m24.2\x1b[0m',
 '\x1b[1m[\x1b[0m\x1b[34;49mnotice\x1b[0m\x1b[1;39;49m]\x1b[0m\x1b[39;49m To update, run: \x1b[0m\x1b[32;49mpip install --upgrade pip\x1b[0m']

```
</div>
---
## Inference with a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png)

Let's get started with the simplest KerasHub API: a pretrained classifier.
In this example, we will construct a classifier that was
pretrained on the ImageNet dataset.
We'll use this model to solve the age old "Cat or Dog" problem.

The highest level module in KerasHub is a *task*. A *task* is a `keras.Model`
consisting of a (generally pretrained) backbone model and task-specific layers.
Here's an example using `keras_hub.models.ImageClassifier` with an
ResNet Backbone.

ResNet is a great starting model when constructing an image
classification pipeline.
This architecture manages to achieve high accuracy, while using a
compact parameter count.
If a ResNet is not powerful enough for the task you are hoping to
solve, be sure to check out [KerasHub's other available
Backbones](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/models)!


```python
classifier = keras_hub.models.ImageClassifier.from_preset("resnet_v2_50_imagenet")
```

You may notice a small deviation from the old `keras.applications` API; where
you would construct the class with `Resnet50V2(weights="imagenet")`.
While the old API was great for classification, it did not scale effectively to
other use cases that required complex architectures, like object detection and
semantic segmentation.

We first create a utility function for plotting images throughout this tutorial:


```python

def plot_image_gallery(images, titles=None, n_cols=3, figsize=(6, 12)):
    n_images = len(images)
    images = np.asarray(images) / 255.0
    images = np.minimum(np.maximum(images, 0.0), 1.0)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()  # Flatten in case the axes is a 2D array

    for i, ax in enumerate(axes):
        if i < n_images:
            # Plot the image
            ax.imshow(images[i])
            ax.axis("off")  # Remove axis
            if titles and len(titles) > i:
                ax.set_title(titles[i], fontsize=12)
        else:
            # Turn off the axis for any empty subplot
            ax.axis("off")

    plt.show()
    plt.close()

```

Now that our classifier is built, let's apply it to this cute cat picture!


```python
filepath = keras.utils.get_file(
    origin="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/5hR96puA_VA.jpg/1024px-5hR96puA_VA.jpg"
)
image = keras.utils.load_img(filepath)
image = np.array(image)
plot_image_gallery(np.array([image]), n_cols=1, figsize=(3, 3))
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_9_0.png)
    


Next, let's get some predictions from our classifier:


```python
predictions = classifier.predict(np.expand_dims(image, axis=0))
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step


Predictions come in the form of softmax-ed category rankings.
We can find the index of the top classes using a simple argsort function:


```python
top_classes = predictions[0].argsort(axis=-1)

```

In order to decode the class mappings, we can construct a mapping from
category indices to ImageNet class names.
For convenience, I've stored the ImageNet class mapping in a GitHub gist.
Let's download and load it now.


```python
classes = keras.utils.get_file(
    origin="https://gist.githubusercontent.com/LukeWood/62eebcd5c5c4a4d0e0b7845780f76d55/raw/fde63e5e4c09e2fa0a3436680f436bdcb8325aac/ImagenetClassnames.json"
)
with open(classes, "rb") as f:
    classes = json.load(f)
```

Now we can simply look up the class names via index:


```python
top_two = [classes[str(i)] for i in top_classes[-2:]]
print("Top two classes are:", top_two)
```

<div class="k-default-codeblock">
```
Top two classes are: ['bath towel', 'Persian cat']

```
</div>
Great!  Both of these appear to be correct!
However, one of the classes is "Bath towel".
We're trying to classify Cats VS Dogs.
We don't care about the towel!

Ideally, we'd have a classifier that only performs computation to determine if
an image is a cat or a dog, and has all of its resources dedicated to this task.
This can be solved by fine tuning our own classifier.

---
## Fine tuning a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png)

When labeled images specific to our task are available, fine-tuning a custom
classifier can improve performance.
If we want to train a Cats vs Dogs Classifier, using explicitly labeled Cat vs
Dog data should perform better than the generic classifier!
For many tasks, no relevant pretrained model
will be available (e.g., categorizing images specific to your application).

First, let's get started by loading some data:


```python
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()

data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
train_dataset = data["train"]

num_classes = dataset_info.features["label"].num_classes

resizing = keras.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)


def preprocess_inputs(image, label):
    image = tf.cast(image, tf.float32)
    # Staticly resize images as we only iterate the dataset once.
    return resizing(image), tf.one_hot(label, num_classes)


# Shuffle the dataset to increase diversity of batches.
# 10*BATCH_SIZE follows the assumption that bigger machines can handle bigger
# shuffle buffers.
train_dataset = train_dataset.shuffle(
    10 * BATCH_SIZE, reshuffle_each_iteration=True
).map(preprocess_inputs, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

images = next(iter(train_dataset.take(1)))[0]
plot_image_gallery(images)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_19_0.png)
    


Meow!

Next let's construct our model.
The use of imagenet in the preset name indicates that the backbone was
pretrained on the ImageNet dataset.
Pretrained backbones extract more information from our labeled examples by
leveraging patterns extracted from potentially much larger datasets.

Next lets put together our classifier:


```python
model = keras_hub.models.ImageClassifier.from_preset(
    "resnet_v2_50_imagenet", num_classes=2
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)
```

Here our classifier is just a simple `keras.Sequential`.
All that is left to do is call `model.fit()`:


```python
model.fit(train_dataset)

```

    
   1/727 [37m━━━━━━━━━━━━━━━━━━━━  4:30:35 22s/step - accuracy: 0.4375 - loss: 3.8293
   2/727 [37m━━━━━━━━━━━━━━━━━━━━  2:48 233ms/step - accuracy: 0.4766 - loss: 3.8293

    
   3/727 [37m━━━━━━━━━━━━━━━━━━━━  2:53 239ms/step - accuracy: 0.4913 - loss: 3.8293 

<div class="k-default-codeblock">
```

```
</div>
   4/727 [37m━━━━━━━━━━━━━━━━━━━━  2:50 236ms/step - accuracy: 0.4974 - loss: 4.0268

<div class="k-default-codeblock">
```

```
</div>
   5/727 [37m━━━━━━━━━━━━━━━━━━━━  2:50 237ms/step - accuracy: 0.5029 - loss: 4.1289

<div class="k-default-codeblock">
```

```
</div>
   6/727 [37m━━━━━━━━━━━━━━━━━━━━  2:50 237ms/step - accuracy: 0.5033 - loss: 4.0894

<div class="k-default-codeblock">
```

```
</div>
   7/727 [37m━━━━━━━━━━━━━━━━━━━━  2:49 236ms/step - accuracy: 0.4977 - loss: 3.9940

<div class="k-default-codeblock">
```

```
</div>
   8/727 [37m━━━━━━━━━━━━━━━━━━━━  2:49 236ms/step - accuracy: 0.4917 - loss: 3.8792

<div class="k-default-codeblock">
```

```
</div>
   9/727 [37m━━━━━━━━━━━━━━━━━━━━  2:48 235ms/step - accuracy: 0.4872 - loss: 3.7604

<div class="k-default-codeblock">
```

```
</div>
  10/727 [37m━━━━━━━━━━━━━━━━━━━━  2:48 235ms/step - accuracy: 0.4828 - loss: 3.6441

<div class="k-default-codeblock">
```

```
</div>
  11/727 [37m━━━━━━━━━━━━━━━━━━━━  2:48 235ms/step - accuracy: 0.4787 - loss: 3.5329

<div class="k-default-codeblock">
```

```
</div>
  12/727 [37m━━━━━━━━━━━━━━━━━━━━  2:48 235ms/step - accuracy: 0.4751 - loss: 3.4280

<div class="k-default-codeblock">
```

```
</div>
  13/727 [37m━━━━━━━━━━━━━━━━━━━━  2:47 235ms/step - accuracy: 0.4716 - loss: 3.3296

<div class="k-default-codeblock">
```

```
</div>
  14/727 [37m━━━━━━━━━━━━━━━━━━━━  2:47 235ms/step - accuracy: 0.4684 - loss: 3.2376

<div class="k-default-codeblock">
```

```
</div>
  15/727 [37m━━━━━━━━━━━━━━━━━━━━  2:47 235ms/step - accuracy: 0.4649 - loss: 3.1517

<div class="k-default-codeblock">
```

```
</div>
  16/727 [37m━━━━━━━━━━━━━━━━━━━━  2:47 235ms/step - accuracy: 0.4615 - loss: 3.0714

<div class="k-default-codeblock">
```

```
</div>
  17/727 [37m━━━━━━━━━━━━━━━━━━━━  2:46 235ms/step - accuracy: 0.4579 - loss: 2.9964

<div class="k-default-codeblock">
```

```
</div>
  18/727 [37m━━━━━━━━━━━━━━━━━━━━  2:46 235ms/step - accuracy: 0.4546 - loss: 2.9261

<div class="k-default-codeblock">
```

```
</div>
  19/727 [37m━━━━━━━━━━━━━━━━━━━━  2:46 235ms/step - accuracy: 0.4514 - loss: 2.8603

<div class="k-default-codeblock">
```

```
</div>
  20/727 [37m━━━━━━━━━━━━━━━━━━━━  2:46 235ms/step - accuracy: 0.4484 - loss: 2.7985

<div class="k-default-codeblock">
```

```
</div>
  21/727 [37m━━━━━━━━━━━━━━━━━━━━  2:46 235ms/step - accuracy: 0.4455 - loss: 2.7404

<div class="k-default-codeblock">
```

```
</div>
  22/727 [37m━━━━━━━━━━━━━━━━━━━━  2:45 235ms/step - accuracy: 0.4427 - loss: 2.6856

<div class="k-default-codeblock">
```

```
</div>
  23/727 [37m━━━━━━━━━━━━━━━━━━━━  2:45 235ms/step - accuracy: 0.4401 - loss: 2.6340

<div class="k-default-codeblock">
```

```
</div>
  24/727 [37m━━━━━━━━━━━━━━━━━━━━  2:45 235ms/step - accuracy: 0.4374 - loss: 2.5851

<div class="k-default-codeblock">
```

```
</div>
  25/727 [37m━━━━━━━━━━━━━━━━━━━━  2:44 235ms/step - accuracy: 0.4349 - loss: 2.5389

<div class="k-default-codeblock">
```

```
</div>
  26/727 [37m━━━━━━━━━━━━━━━━━━━━  2:44 235ms/step - accuracy: 0.4323 - loss: 2.4951

<div class="k-default-codeblock">
```

```
</div>
  27/727 [37m━━━━━━━━━━━━━━━━━━━━  2:44 235ms/step - accuracy: 0.4298 - loss: 2.4534

<div class="k-default-codeblock">
```

```
</div>
  28/727 [37m━━━━━━━━━━━━━━━━━━━━  2:44 235ms/step - accuracy: 0.4274 - loss: 2.4138

<div class="k-default-codeblock">
```

```
</div>
  29/727 [37m━━━━━━━━━━━━━━━━━━━━  2:44 235ms/step - accuracy: 0.4250 - loss: 2.3761

<div class="k-default-codeblock">
```

```
</div>
  30/727 [37m━━━━━━━━━━━━━━━━━━━━  2:43 235ms/step - accuracy: 0.4228 - loss: 2.3402

<div class="k-default-codeblock">
```

```
</div>
  31/727 [37m━━━━━━━━━━━━━━━━━━━━  2:43 235ms/step - accuracy: 0.4206 - loss: 2.3059

<div class="k-default-codeblock">
```

```
</div>
  32/727 [37m━━━━━━━━━━━━━━━━━━━━  2:43 235ms/step - accuracy: 0.4185 - loss: 2.2731

<div class="k-default-codeblock">
```

```
</div>
  33/727 [37m━━━━━━━━━━━━━━━━━━━━  2:43 235ms/step - accuracy: 0.4166 - loss: 2.2418

<div class="k-default-codeblock">
```

```
</div>
  34/727 [37m━━━━━━━━━━━━━━━━━━━━  2:42 235ms/step - accuracy: 0.4147 - loss: 2.2117

<div class="k-default-codeblock">
```

```
</div>
  35/727 [37m━━━━━━━━━━━━━━━━━━━━  2:42 235ms/step - accuracy: 0.4128 - loss: 2.1829

<div class="k-default-codeblock">
```

```
</div>
  36/727 [37m━━━━━━━━━━━━━━━━━━━━  2:42 235ms/step - accuracy: 0.4108 - loss: 2.1553

<div class="k-default-codeblock">
```

```
</div>
  37/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:42 235ms/step - accuracy: 0.4088 - loss: 2.1287

<div class="k-default-codeblock">
```

```
</div>
  38/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:41 235ms/step - accuracy: 0.4068 - loss: 2.1031

<div class="k-default-codeblock">
```

```
</div>
  39/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:41 235ms/step - accuracy: 0.4048 - loss: 2.0785

<div class="k-default-codeblock">
```

```
</div>
  40/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:41 235ms/step - accuracy: 0.4029 - loss: 2.0548

<div class="k-default-codeblock">
```

```
</div>
  41/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:41 235ms/step - accuracy: 0.4010 - loss: 2.0319

<div class="k-default-codeblock">
```

```
</div>
  42/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:41 235ms/step - accuracy: 0.3991 - loss: 2.0098

<div class="k-default-codeblock">
```

```
</div>
  43/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:40 235ms/step - accuracy: 0.3971 - loss: 1.9885

<div class="k-default-codeblock">
```

```
</div>
  44/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:40 235ms/step - accuracy: 0.3952 - loss: 1.9679

<div class="k-default-codeblock">
```

```
</div>
  45/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:40 235ms/step - accuracy: 0.3933 - loss: 1.9480

<div class="k-default-codeblock">
```

```
</div>
  46/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:40 235ms/step - accuracy: 0.3915 - loss: 1.9287

<div class="k-default-codeblock">
```

```
</div>
  47/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:40 235ms/step - accuracy: 0.3897 - loss: 1.9101

<div class="k-default-codeblock">
```

```
</div>
  48/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:39 235ms/step - accuracy: 0.3879 - loss: 1.8920

<div class="k-default-codeblock">
```

```
</div>
  49/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:39 235ms/step - accuracy: 0.3862 - loss: 1.8744

<div class="k-default-codeblock">
```

```
</div>
  50/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:39 235ms/step - accuracy: 0.3845 - loss: 1.8574

<div class="k-default-codeblock">
```

```
</div>
  51/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:39 235ms/step - accuracy: 0.3828 - loss: 1.8409

<div class="k-default-codeblock">
```

```
</div>
  52/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:38 235ms/step - accuracy: 0.3811 - loss: 1.8249

<div class="k-default-codeblock">
```

```
</div>
  53/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:38 235ms/step - accuracy: 0.3795 - loss: 1.8093

<div class="k-default-codeblock">
```

```
</div>
  54/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:38 235ms/step - accuracy: 0.3778 - loss: 1.7942

<div class="k-default-codeblock">
```

```
</div>
  55/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:38 235ms/step - accuracy: 0.3761 - loss: 1.7795

<div class="k-default-codeblock">
```

```
</div>
  56/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:38 235ms/step - accuracy: 0.3745 - loss: 1.7652

<div class="k-default-codeblock">
```

```
</div>
  57/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:37 236ms/step - accuracy: 0.3728 - loss: 1.7512

<div class="k-default-codeblock">
```

```
</div>
  58/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:37 236ms/step - accuracy: 0.3712 - loss: 1.7377

<div class="k-default-codeblock">
```

```
</div>
  59/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:37 236ms/step - accuracy: 0.3696 - loss: 1.7245

<div class="k-default-codeblock">
```

```
</div>
  60/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:37 236ms/step - accuracy: 0.3680 - loss: 1.7116

<div class="k-default-codeblock">
```

```
</div>
  61/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:36 236ms/step - accuracy: 0.3664 - loss: 1.6990

<div class="k-default-codeblock">
```

```
</div>
  62/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:36 236ms/step - accuracy: 0.3648 - loss: 1.6867

<div class="k-default-codeblock">
```

```
</div>
  63/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:36 236ms/step - accuracy: 0.3632 - loss: 1.6747

<div class="k-default-codeblock">
```

```
</div>
  64/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:36 236ms/step - accuracy: 0.3616 - loss: 1.6630

<div class="k-default-codeblock">
```

```
</div>
  65/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:35 236ms/step - accuracy: 0.3601 - loss: 1.6516

<div class="k-default-codeblock">
```

```
</div>
  66/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:35 236ms/step - accuracy: 0.3585 - loss: 1.6405

<div class="k-default-codeblock">
```

```
</div>
  67/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:35 236ms/step - accuracy: 0.3570 - loss: 1.6296

<div class="k-default-codeblock">
```

```
</div>
  68/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:35 236ms/step - accuracy: 0.3555 - loss: 1.6189

<div class="k-default-codeblock">
```

```
</div>
  69/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:35 236ms/step - accuracy: 0.3541 - loss: 1.6085

<div class="k-default-codeblock">
```

```
</div>
  70/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:34 236ms/step - accuracy: 0.3527 - loss: 1.5983

<div class="k-default-codeblock">
```

```
</div>
  71/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:34 236ms/step - accuracy: 0.3513 - loss: 1.5884

<div class="k-default-codeblock">
```

```
</div>
  72/727 ━[37m━━━━━━━━━━━━━━━━━━━  2:34 236ms/step - accuracy: 0.3499 - loss: 1.5787

<div class="k-default-codeblock">
```

```
</div>
  73/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:34 236ms/step - accuracy: 0.3486 - loss: 1.5691

<div class="k-default-codeblock">
```

```
</div>
  74/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:33 236ms/step - accuracy: 0.3472 - loss: 1.5598

<div class="k-default-codeblock">
```

```
</div>
  75/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:33 236ms/step - accuracy: 0.3458 - loss: 1.5506

<div class="k-default-codeblock">
```

```
</div>
  76/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:33 236ms/step - accuracy: 0.3445 - loss: 1.5416

<div class="k-default-codeblock">
```

```
</div>
  77/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:33 236ms/step - accuracy: 0.3432 - loss: 1.5328

<div class="k-default-codeblock">
```

```
</div>
  78/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:33 236ms/step - accuracy: 0.3419 - loss: 1.5242

<div class="k-default-codeblock">
```

```
</div>
  79/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:32 236ms/step - accuracy: 0.3406 - loss: 1.5157

<div class="k-default-codeblock">
```

```
</div>
  80/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:32 236ms/step - accuracy: 0.3393 - loss: 1.5075

<div class="k-default-codeblock">
```

```
</div>
  81/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:32 236ms/step - accuracy: 0.3380 - loss: 1.4993

<div class="k-default-codeblock">
```

```
</div>
  82/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:32 236ms/step - accuracy: 0.3368 - loss: 1.4913

<div class="k-default-codeblock">
```

```
</div>
  83/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:31 236ms/step - accuracy: 0.3355 - loss: 1.4835

<div class="k-default-codeblock">
```

```
</div>
  84/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:31 236ms/step - accuracy: 0.3343 - loss: 1.4758

<div class="k-default-codeblock">
```

```
</div>
  85/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:31 236ms/step - accuracy: 0.3331 - loss: 1.4682

<div class="k-default-codeblock">
```

```
</div>
  86/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:31 236ms/step - accuracy: 0.3320 - loss: 1.4608

<div class="k-default-codeblock">
```

```
</div>
  87/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:30 236ms/step - accuracy: 0.3308 - loss: 1.4535

<div class="k-default-codeblock">
```

```
</div>
  88/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:30 236ms/step - accuracy: 0.3297 - loss: 1.4463

<div class="k-default-codeblock">
```

```
</div>
  89/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:30 236ms/step - accuracy: 0.3285 - loss: 1.4393

<div class="k-default-codeblock">
```

```
</div>
  90/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:30 236ms/step - accuracy: 0.3274 - loss: 1.4323

<div class="k-default-codeblock">
```

```
</div>
  91/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:30 236ms/step - accuracy: 0.3263 - loss: 1.4255

<div class="k-default-codeblock">
```

```
</div>
  92/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:29 236ms/step - accuracy: 0.3252 - loss: 1.4188

<div class="k-default-codeblock">
```

```
</div>
  93/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:29 236ms/step - accuracy: 0.3241 - loss: 1.4122

<div class="k-default-codeblock">
```

```
</div>
  94/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:29 236ms/step - accuracy: 0.3231 - loss: 1.4057

<div class="k-default-codeblock">
```

```
</div>
  95/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:29 236ms/step - accuracy: 0.3220 - loss: 1.3993

<div class="k-default-codeblock">
```

```
</div>
  96/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 236ms/step - accuracy: 0.3210 - loss: 1.3930

<div class="k-default-codeblock">
```

```
</div>
  97/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 236ms/step - accuracy: 0.3200 - loss: 1.3868

<div class="k-default-codeblock">
```

```
</div>
  98/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 236ms/step - accuracy: 0.3190 - loss: 1.3807

<div class="k-default-codeblock">
```

```
</div>
  99/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 236ms/step - accuracy: 0.3180 - loss: 1.3747

<div class="k-default-codeblock">
```

```
</div>
 100/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 236ms/step - accuracy: 0.3170 - loss: 1.3688

<div class="k-default-codeblock">
```

```
</div>
 101/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:27 236ms/step - accuracy: 0.3160 - loss: 1.3630

<div class="k-default-codeblock">
```

```
</div>
 102/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:27 236ms/step - accuracy: 0.3151 - loss: 1.3572

<div class="k-default-codeblock">
```

```
</div>
 103/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:27 236ms/step - accuracy: 0.3141 - loss: 1.3515

<div class="k-default-codeblock">
```

```
</div>
 104/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:27 236ms/step - accuracy: 0.3132 - loss: 1.3459

<div class="k-default-codeblock">
```

```
</div>
 105/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:26 236ms/step - accuracy: 0.3122 - loss: 1.3404

<div class="k-default-codeblock">
```

```
</div>
 106/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:26 236ms/step - accuracy: 0.3113 - loss: 1.3350

<div class="k-default-codeblock">
```

```
</div>
 107/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:26 236ms/step - accuracy: 0.3104 - loss: 1.3296

<div class="k-default-codeblock">
```

```
</div>
 108/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:26 236ms/step - accuracy: 0.3094 - loss: 1.3243

<div class="k-default-codeblock">
```

```
</div>
 109/727 ━━[37m━━━━━━━━━━━━━━━━━━  2:26 236ms/step - accuracy: 0.3085 - loss: 1.3191

<div class="k-default-codeblock">
```

```
</div>
 110/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:25 236ms/step - accuracy: 0.3076 - loss: 1.3139

<div class="k-default-codeblock">
```

```
</div>
 111/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:25 236ms/step - accuracy: 0.3068 - loss: 1.3088

<div class="k-default-codeblock">
```

```
</div>
 112/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:25 236ms/step - accuracy: 0.3059 - loss: 1.3037

<div class="k-default-codeblock">
```

```
</div>
 113/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:25 236ms/step - accuracy: 0.3050 - loss: 1.2987

<div class="k-default-codeblock">
```

```
</div>
 114/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:24 236ms/step - accuracy: 0.3041 - loss: 1.2938

<div class="k-default-codeblock">
```

```
</div>
 115/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:24 236ms/step - accuracy: 0.3033 - loss: 1.2889

<div class="k-default-codeblock">
```

```
</div>
 116/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:24 236ms/step - accuracy: 0.3024 - loss: 1.2841

<div class="k-default-codeblock">
```

```
</div>
 117/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:24 236ms/step - accuracy: 0.3015 - loss: 1.2793

<div class="k-default-codeblock">
```

```
</div>
 118/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 236ms/step - accuracy: 0.3007 - loss: 1.2746

<div class="k-default-codeblock">
```

```
</div>
 119/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 236ms/step - accuracy: 0.2998 - loss: 1.2699

<div class="k-default-codeblock">
```

```
</div>
 120/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 236ms/step - accuracy: 0.2990 - loss: 1.2653

<div class="k-default-codeblock">
```

```
</div>
 121/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 236ms/step - accuracy: 0.2982 - loss: 1.2607

<div class="k-default-codeblock">
```

```
</div>
 122/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 236ms/step - accuracy: 0.2974 - loss: 1.2562

<div class="k-default-codeblock">
```

```
</div>
 123/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:22 236ms/step - accuracy: 0.2966 - loss: 1.2517

<div class="k-default-codeblock">
```

```
</div>
 124/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:22 236ms/step - accuracy: 0.2958 - loss: 1.2473

<div class="k-default-codeblock">
```

```
</div>
 125/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:22 236ms/step - accuracy: 0.2950 - loss: 1.2429

<div class="k-default-codeblock">
```

```
</div>
 126/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:22 236ms/step - accuracy: 0.2942 - loss: 1.2385

<div class="k-default-codeblock">
```

```
</div>
 127/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:21 236ms/step - accuracy: 0.2934 - loss: 1.2343

<div class="k-default-codeblock">
```

```
</div>
 128/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:21 237ms/step - accuracy: 0.2926 - loss: 1.2300

<div class="k-default-codeblock">
```

```
</div>
 129/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:21 237ms/step - accuracy: 0.2919 - loss: 1.2258

<div class="k-default-codeblock">
```

```
</div>
 130/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:21 237ms/step - accuracy: 0.2911 - loss: 1.2217

<div class="k-default-codeblock">
```

```
</div>
 131/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:20 237ms/step - accuracy: 0.2903 - loss: 1.2176

<div class="k-default-codeblock">
```

```
</div>
 132/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:20 237ms/step - accuracy: 0.2896 - loss: 1.2135

<div class="k-default-codeblock">
```

```
</div>
 133/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:20 237ms/step - accuracy: 0.2888 - loss: 1.2095

<div class="k-default-codeblock">
```

```
</div>
 134/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:20 237ms/step - accuracy: 0.2881 - loss: 1.2055

<div class="k-default-codeblock">
```

```
</div>
 135/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:20 237ms/step - accuracy: 0.2873 - loss: 1.2016

<div class="k-default-codeblock">
```

```
</div>
 136/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:19 237ms/step - accuracy: 0.2866 - loss: 1.1977

<div class="k-default-codeblock">
```

```
</div>
 137/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:19 237ms/step - accuracy: 0.2858 - loss: 1.1939

<div class="k-default-codeblock">
```

```
</div>
 138/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:19 237ms/step - accuracy: 0.2851 - loss: 1.1901

<div class="k-default-codeblock">
```

```
</div>
 139/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:19 237ms/step - accuracy: 0.2844 - loss: 1.1864

<div class="k-default-codeblock">
```

```
</div>
 140/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 237ms/step - accuracy: 0.2837 - loss: 1.1827

<div class="k-default-codeblock">
```

```
</div>
 141/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 237ms/step - accuracy: 0.2830 - loss: 1.1791

<div class="k-default-codeblock">
```

```
</div>
 142/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 237ms/step - accuracy: 0.2822 - loss: 1.1755

<div class="k-default-codeblock">
```

```
</div>
 143/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 237ms/step - accuracy: 0.2815 - loss: 1.1719

<div class="k-default-codeblock">
```

```
</div>
 144/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 237ms/step - accuracy: 0.2809 - loss: 1.1683

<div class="k-default-codeblock">
```

```
</div>
 145/727 ━━━[37m━━━━━━━━━━━━━━━━━  2:17 237ms/step - accuracy: 0.2802 - loss: 1.1648

<div class="k-default-codeblock">
```

```
</div>
 146/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:17 237ms/step - accuracy: 0.2795 - loss: 1.1614

<div class="k-default-codeblock">
```

```
</div>
 147/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:17 237ms/step - accuracy: 0.2788 - loss: 1.1579

<div class="k-default-codeblock">
```

```
</div>
 148/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:17 237ms/step - accuracy: 0.2782 - loss: 1.1545

<div class="k-default-codeblock">
```

```
</div>
 149/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:16 237ms/step - accuracy: 0.2775 - loss: 1.1511

<div class="k-default-codeblock">
```

```
</div>
 150/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:16 237ms/step - accuracy: 0.2769 - loss: 1.1478

<div class="k-default-codeblock">
```

```
</div>
 151/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:16 237ms/step - accuracy: 0.2762 - loss: 1.1444

<div class="k-default-codeblock">
```

```
</div>
 152/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:16 237ms/step - accuracy: 0.2756 - loss: 1.1411

<div class="k-default-codeblock">
```

```
</div>
 153/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:15 237ms/step - accuracy: 0.2749 - loss: 1.1379

<div class="k-default-codeblock">
```

```
</div>
 154/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:15 237ms/step - accuracy: 0.2743 - loss: 1.1346

<div class="k-default-codeblock">
```

```
</div>
 155/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:15 237ms/step - accuracy: 0.2737 - loss: 1.1314

<div class="k-default-codeblock">
```

```
</div>
 156/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:15 237ms/step - accuracy: 0.2731 - loss: 1.1282

<div class="k-default-codeblock">
```

```
</div>
 157/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:15 237ms/step - accuracy: 0.2724 - loss: 1.1250

<div class="k-default-codeblock">
```

```
</div>
 158/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:14 237ms/step - accuracy: 0.2718 - loss: 1.1219

<div class="k-default-codeblock">
```

```
</div>
 159/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:14 237ms/step - accuracy: 0.2712 - loss: 1.1187

<div class="k-default-codeblock">
```

```
</div>
 160/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:14 237ms/step - accuracy: 0.2706 - loss: 1.1157

<div class="k-default-codeblock">
```

```
</div>
 161/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:14 237ms/step - accuracy: 0.2700 - loss: 1.1126

<div class="k-default-codeblock">
```

```
</div>
 162/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:13 237ms/step - accuracy: 0.2694 - loss: 1.1096

<div class="k-default-codeblock">
```

```
</div>
 163/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:13 237ms/step - accuracy: 0.2688 - loss: 1.1067

<div class="k-default-codeblock">
```

```
</div>
 164/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:13 237ms/step - accuracy: 0.2682 - loss: 1.1037

<div class="k-default-codeblock">
```

```
</div>
 165/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:13 237ms/step - accuracy: 0.2676 - loss: 1.1008

<div class="k-default-codeblock">
```

```
</div>
 166/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:12 237ms/step - accuracy: 0.2670 - loss: 1.0979

<div class="k-default-codeblock">
```

```
</div>
 167/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:12 237ms/step - accuracy: 0.2664 - loss: 1.0951

<div class="k-default-codeblock">
```

```
</div>
 168/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:12 237ms/step - accuracy: 0.2659 - loss: 1.0923

<div class="k-default-codeblock">
```

```
</div>
 169/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:12 237ms/step - accuracy: 0.2653 - loss: 1.0895

<div class="k-default-codeblock">
```

```
</div>
 170/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:11 237ms/step - accuracy: 0.2647 - loss: 1.0868

<div class="k-default-codeblock">
```

```
</div>
 171/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:11 237ms/step - accuracy: 0.2642 - loss: 1.0841

<div class="k-default-codeblock">
```

```
</div>
 172/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:11 237ms/step - accuracy: 0.2636 - loss: 1.0814

<div class="k-default-codeblock">
```

```
</div>
 173/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:11 237ms/step - accuracy: 0.2631 - loss: 1.0788

<div class="k-default-codeblock">
```

```
</div>
 174/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:11 237ms/step - accuracy: 0.2626 - loss: 1.0762

<div class="k-default-codeblock">
```

```
</div>
 175/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:10 237ms/step - accuracy: 0.2620 - loss: 1.0736

<div class="k-default-codeblock">
```

```
</div>
 176/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:10 237ms/step - accuracy: 0.2615 - loss: 1.0710

<div class="k-default-codeblock">
```

```
</div>
 177/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:10 237ms/step - accuracy: 0.2610 - loss: 1.0684

<div class="k-default-codeblock">
```

```
</div>
 178/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:10 237ms/step - accuracy: 0.2605 - loss: 1.0659

<div class="k-default-codeblock">
```

```
</div>
 179/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:09 237ms/step - accuracy: 0.2600 - loss: 1.0634

<div class="k-default-codeblock">
```

```
</div>
 180/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:09 237ms/step - accuracy: 0.2595 - loss: 1.0609

<div class="k-default-codeblock">
```

```
</div>
 181/727 ━━━━[37m━━━━━━━━━━━━━━━━  2:09 237ms/step - accuracy: 0.2590 - loss: 1.0584

<div class="k-default-codeblock">
```

```
</div>
 182/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:09 237ms/step - accuracy: 0.2585 - loss: 1.0560

<div class="k-default-codeblock">
```

```
</div>
 183/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:08 237ms/step - accuracy: 0.2581 - loss: 1.0535

<div class="k-default-codeblock">
```

```
</div>
 184/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:08 237ms/step - accuracy: 0.2576 - loss: 1.0511

<div class="k-default-codeblock">
```

```
</div>
 185/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:08 237ms/step - accuracy: 0.2571 - loss: 1.0487

<div class="k-default-codeblock">
```

```
</div>
 186/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:08 237ms/step - accuracy: 0.2567 - loss: 1.0463

<div class="k-default-codeblock">
```

```
</div>
 187/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:08 237ms/step - accuracy: 0.2562 - loss: 1.0440

<div class="k-default-codeblock">
```

```
</div>
 188/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:07 237ms/step - accuracy: 0.2558 - loss: 1.0416

<div class="k-default-codeblock">
```

```
</div>
 189/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:07 237ms/step - accuracy: 0.2553 - loss: 1.0393

<div class="k-default-codeblock">
```

```
</div>
 190/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:07 237ms/step - accuracy: 0.2549 - loss: 1.0370

<div class="k-default-codeblock">
```

```
</div>
 191/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:07 237ms/step - accuracy: 0.2544 - loss: 1.0347

<div class="k-default-codeblock">
```

```
</div>
 192/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:06 237ms/step - accuracy: 0.2540 - loss: 1.0324

<div class="k-default-codeblock">
```

```
</div>
 193/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:06 237ms/step - accuracy: 0.2535 - loss: 1.0301

<div class="k-default-codeblock">
```

```
</div>
 194/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:06 237ms/step - accuracy: 0.2531 - loss: 1.0279

<div class="k-default-codeblock">
```

```
</div>
 195/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:06 237ms/step - accuracy: 0.2526 - loss: 1.0256

<div class="k-default-codeblock">
```

```
</div>
 196/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:05 237ms/step - accuracy: 0.2522 - loss: 1.0234

<div class="k-default-codeblock">
```

```
</div>
 197/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:05 237ms/step - accuracy: 0.2518 - loss: 1.0212

<div class="k-default-codeblock">
```

```
</div>
 198/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:05 237ms/step - accuracy: 0.2513 - loss: 1.0190

<div class="k-default-codeblock">
```

```
</div>
 199/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:05 237ms/step - accuracy: 0.2509 - loss: 1.0168

<div class="k-default-codeblock">
```

```
</div>
 200/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:05 237ms/step - accuracy: 0.2505 - loss: 1.0146

<div class="k-default-codeblock">
```

```
</div>
 201/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:04 237ms/step - accuracy: 0.2501 - loss: 1.0125

<div class="k-default-codeblock">
```

```
</div>
 202/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:04 237ms/step - accuracy: 0.2496 - loss: 1.0103

<div class="k-default-codeblock">
```

```
</div>
 203/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:04 237ms/step - accuracy: 0.2492 - loss: 1.0082

<div class="k-default-codeblock">
```

```
</div>
 204/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:04 237ms/step - accuracy: 0.2488 - loss: 1.0061

<div class="k-default-codeblock">
```

```
</div>
 205/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:03 237ms/step - accuracy: 0.2484 - loss: 1.0040

<div class="k-default-codeblock">
```
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9


```
</div>
 206/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:03 237ms/step - accuracy: 0.2480 - loss: 1.0019

<div class="k-default-codeblock">
```

```
</div>
 207/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:03 237ms/step - accuracy: 0.2476 - loss: 0.9999

<div class="k-default-codeblock">
```

```
</div>
 208/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:03 237ms/step - accuracy: 0.2471 - loss: 0.9978

<div class="k-default-codeblock">
```

```
</div>
 209/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:02 237ms/step - accuracy: 0.2467 - loss: 0.9958

<div class="k-default-codeblock">
```

```
</div>
 210/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:02 237ms/step - accuracy: 0.2463 - loss: 0.9938

<div class="k-default-codeblock">
```

```
</div>
 211/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:02 237ms/step - accuracy: 0.2459 - loss: 0.9918

<div class="k-default-codeblock">
```

```
</div>
 212/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:02 237ms/step - accuracy: 0.2455 - loss: 0.9898

<div class="k-default-codeblock">
```

```
</div>
 213/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:01 237ms/step - accuracy: 0.2451 - loss: 0.9878

<div class="k-default-codeblock">
```

```
</div>
 214/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:01 237ms/step - accuracy: 0.2447 - loss: 0.9859

<div class="k-default-codeblock">
```

```
</div>
 215/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:01 237ms/step - accuracy: 0.2443 - loss: 0.9840

<div class="k-default-codeblock">
```

```
</div>
 216/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:01 237ms/step - accuracy: 0.2439 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 217/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:01 237ms/step - accuracy: 0.2435 - loss: 0.9802

<div class="k-default-codeblock">
```

```
</div>
 218/727 ━━━━━[37m━━━━━━━━━━━━━━━  2:00 237ms/step - accuracy: 0.2431 - loss: 0.9784

<div class="k-default-codeblock">
```

```
</div>
 219/727 ━━━━━━[37m━━━━━━━━━━━━━━  2:00 237ms/step - accuracy: 0.2427 - loss: 0.9750

<div class="k-default-codeblock">
```

```
</div>
 220/727 ━━━━━━[37m━━━━━━━━━━━━━━  2:00 237ms/step - accuracy: 0.2423 - loss: 0.9750

<div class="k-default-codeblock">
```

```
</div>
 221/727 ━━━━━━[37m━━━━━━━━━━━━━━  2:00 237ms/step - accuracy: 0.2419 - loss: 0.9733

<div class="k-default-codeblock">
```

```
</div>
 222/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:59 237ms/step - accuracy: 0.2415 - loss: 0.9718

<div class="k-default-codeblock">
```

```
</div>
 223/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:59 237ms/step - accuracy: 0.2412 - loss: 0.9702

<div class="k-default-codeblock">
```

```
</div>
 224/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:59 237ms/step - accuracy: 0.2408 - loss: 0.9687

<div class="k-default-codeblock">
```

```
</div>
 225/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:59 237ms/step - accuracy: 0.2404 - loss: 0.9673

<div class="k-default-codeblock">
```

```
</div>
 226/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:58 237ms/step - accuracy: 0.2400 - loss: 0.9659

<div class="k-default-codeblock">
```

```
</div>
 227/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:58 237ms/step - accuracy: 0.2396 - loss: 0.9646

<div class="k-default-codeblock">
```

```
</div>
 228/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:58 237ms/step - accuracy: 0.2392 - loss: 0.9634

<div class="k-default-codeblock">
```

```
</div>
 229/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:58 237ms/step - accuracy: 0.2388 - loss: 0.9623

<div class="k-default-codeblock">
```

```
</div>
 230/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:58 237ms/step - accuracy: 0.2385 - loss: 0.9612

<div class="k-default-codeblock">
```

```
</div>
 231/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:57 237ms/step - accuracy: 0.2381 - loss: 0.9602

<div class="k-default-codeblock">
```

```
</div>
 232/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:57 237ms/step - accuracy: 0.2377 - loss: 0.9593

<div class="k-default-codeblock">
```

```
</div>
 233/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:57 237ms/step - accuracy: 0.2374 - loss: 0.9585

<div class="k-default-codeblock">
```

```
</div>
 234/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:57 237ms/step - accuracy: 0.2370 - loss: 0.9579

<div class="k-default-codeblock">
```

```
</div>
 235/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:56 237ms/step - accuracy: 0.2366 - loss: 0.9574

<div class="k-default-codeblock">
```

```
</div>
 236/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:56 238ms/step - accuracy: 0.2363 - loss: 0.9572

<div class="k-default-codeblock">
```

```
</div>
 237/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:56 238ms/step - accuracy: 0.2359 - loss: 0.9570

<div class="k-default-codeblock">
```

```
</div>
 238/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:56 238ms/step - accuracy: 0.2356 - loss: 0.9571

<div class="k-default-codeblock">
```

```
</div>
 239/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:55 238ms/step - accuracy: 0.2352 - loss: 0.9572

<div class="k-default-codeblock">
```

```
</div>
 240/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:55 238ms/step - accuracy: 0.2349 - loss: 0.9575

<div class="k-default-codeblock">
```

```
</div>
 241/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:55 238ms/step - accuracy: 0.2346 - loss: 0.9580

<div class="k-default-codeblock">
```
Warning: unknown JFIF revision number 0.00


```
</div>
 242/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:55 238ms/step - accuracy: 0.2342 - loss: 0.9586

<div class="k-default-codeblock">
```

```
</div>
 243/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:54 238ms/step - accuracy: 0.2339 - loss: 0.9593

<div class="k-default-codeblock">
```

```
</div>
 244/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:54 238ms/step - accuracy: 0.2336 - loss: 0.9600

<div class="k-default-codeblock">
```

```
</div>
 245/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:54 238ms/step - accuracy: 0.2333 - loss: 0.9609

<div class="k-default-codeblock">
```

```
</div>
 246/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:54 238ms/step - accuracy: 0.2331 - loss: 0.9618

<div class="k-default-codeblock">
```

```
</div>
 247/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:54 238ms/step - accuracy: 0.2328 - loss: 0.9629

<div class="k-default-codeblock">
```

```
</div>
 248/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:53 238ms/step - accuracy: 0.2325 - loss: 0.9641

<div class="k-default-codeblock">
```

```
</div>
 249/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:53 238ms/step - accuracy: 0.2323 - loss: 0.9654

<div class="k-default-codeblock">
```

```
</div>
 250/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:53 238ms/step - accuracy: 0.2320 - loss: 0.9668

<div class="k-default-codeblock">
```

```
</div>
 251/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:53 238ms/step - accuracy: 0.2318 - loss: 0.9683

<div class="k-default-codeblock">
```

```
</div>
 252/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:52 238ms/step - accuracy: 0.2315 - loss: 0.9699

<div class="k-default-codeblock">
```

```
</div>
 253/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:52 238ms/step - accuracy: 0.2313 - loss: 0.9715

<div class="k-default-codeblock">
```
Corrupt JPEG data: 396 extraneous bytes before marker 0xd9


```
</div>
 254/727 ━━━━━━[37m━━━━━━━━━━━━━━  1:52 238ms/step - accuracy: 0.2311 - loss: 0.9733

<div class="k-default-codeblock">
```

```
</div>
 255/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:52 238ms/step - accuracy: 0.2308 - loss: 0.9752

<div class="k-default-codeblock">
```

```
</div>
 256/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:51 238ms/step - accuracy: 0.2306 - loss: 0.9771

<div class="k-default-codeblock">
```

```
</div>
 257/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:51 238ms/step - accuracy: 0.2304 - loss: 0.9791

<div class="k-default-codeblock">
```

```
</div>
 258/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:51 238ms/step - accuracy: 0.2302 - loss: 0.9812

<div class="k-default-codeblock">
```

```
</div>
 259/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:51 238ms/step - accuracy: 0.2300 - loss: 0.9833

<div class="k-default-codeblock">
```

```
</div>
 260/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:50 238ms/step - accuracy: 0.2299 - loss: 0.9855

<div class="k-default-codeblock">
```

```
</div>
 261/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:50 238ms/step - accuracy: 0.2297 - loss: 0.9878

<div class="k-default-codeblock">
```

```
</div>
 262/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:50 238ms/step - accuracy: 0.2295 - loss: 0.9902

<div class="k-default-codeblock">
```

```
</div>
 263/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:50 238ms/step - accuracy: 0.2294 - loss: 0.9926

<div class="k-default-codeblock">
```

```
</div>
 264/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:50 238ms/step - accuracy: 0.2292 - loss: 0.9951

<div class="k-default-codeblock">
```

```
</div>
 265/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:49 238ms/step - accuracy: 0.2290 - loss: 0.9977

<div class="k-default-codeblock">
```

```
</div>
 266/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:49 238ms/step - accuracy: 0.2289 - loss: 1.0004

<div class="k-default-codeblock">
```

```
</div>
 267/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:49 238ms/step - accuracy: 0.2288 - loss: 1.0032

<div class="k-default-codeblock">
```

```
</div>
 268/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:49 238ms/step - accuracy: 0.2286 - loss: 1.0060

<div class="k-default-codeblock">
```

```
</div>
 269/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:48 238ms/step - accuracy: 0.2285 - loss: 1.0089

<div class="k-default-codeblock">
```

```
</div>
 270/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:48 238ms/step - accuracy: 0.2284 - loss: 1.0119

<div class="k-default-codeblock">
```

```
</div>
 271/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:48 238ms/step - accuracy: 0.2282 - loss: 1.0149

<div class="k-default-codeblock">
```

```
</div>
 272/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:48 238ms/step - accuracy: 0.2281 - loss: 1.0180

<div class="k-default-codeblock">
```

```
</div>
 273/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:47 237ms/step - accuracy: 0.2280 - loss: 1.0211

<div class="k-default-codeblock">
```

```
</div>
 274/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:47 237ms/step - accuracy: 0.2279 - loss: 1.0243

<div class="k-default-codeblock">
```

```
</div>
 275/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:47 237ms/step - accuracy: 0.2278 - loss: 1.0275

<div class="k-default-codeblock">
```

```
</div>
 276/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:47 237ms/step - accuracy: 0.2277 - loss: 1.0308

<div class="k-default-codeblock">
```

```
</div>
 277/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:46 237ms/step - accuracy: 0.2276 - loss: 1.0341

<div class="k-default-codeblock">
```

```
</div>
 278/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:46 237ms/step - accuracy: 0.2275 - loss: 1.0375

<div class="k-default-codeblock">
```

```
</div>
 279/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:46 237ms/step - accuracy: 0.2275 - loss: 1.0410

<div class="k-default-codeblock">
```

```
</div>
 280/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:46 237ms/step - accuracy: 0.2274 - loss: 1.0445

<div class="k-default-codeblock">
```

```
</div>
 281/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:45 237ms/step - accuracy: 0.2273 - loss: 1.0480

<div class="k-default-codeblock">
```

```
</div>
 282/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:45 237ms/step - accuracy: 0.2272 - loss: 1.0516

<div class="k-default-codeblock">
```

```
</div>
 283/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:45 237ms/step - accuracy: 0.2272 - loss: 1.0553

<div class="k-default-codeblock">
```

```
</div>
 284/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:45 237ms/step - accuracy: 0.2271 - loss: 1.0590

<div class="k-default-codeblock">
```

```
</div>
 285/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:44 237ms/step - accuracy: 0.2270 - loss: 1.0627

<div class="k-default-codeblock">
```

```
</div>
 286/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:44 237ms/step - accuracy: 0.2270 - loss: 1.0665

<div class="k-default-codeblock">
```

```
</div>
 287/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:44 237ms/step - accuracy: 0.2269 - loss: 1.0704

<div class="k-default-codeblock">
```

```
</div>
 288/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:44 237ms/step - accuracy: 0.2269 - loss: 1.0742

<div class="k-default-codeblock">
```

```
</div>
 289/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:43 237ms/step - accuracy: 0.2269 - loss: 1.0782

<div class="k-default-codeblock">
```

```
</div>
 290/727 ━━━━━━━[37m━━━━━━━━━━━━━  1:43 237ms/step - accuracy: 0.2268 - loss: 1.0821

<div class="k-default-codeblock">
```

```
</div>
 291/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:43 237ms/step - accuracy: 0.2268 - loss: 1.0861

<div class="k-default-codeblock">
```

```
</div>
 292/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:43 237ms/step - accuracy: 0.2268 - loss: 1.0901

<div class="k-default-codeblock">
```

```
</div>
 293/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:42 237ms/step - accuracy: 0.2267 - loss: 1.0942

<div class="k-default-codeblock">
```

```
</div>
 294/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:42 237ms/step - accuracy: 0.2267 - loss: 1.0982

<div class="k-default-codeblock">
```

```
</div>
 295/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:42 237ms/step - accuracy: 0.2267 - loss: 1.1024

<div class="k-default-codeblock">
```

```
</div>
 296/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:42 237ms/step - accuracy: 0.2267 - loss: 1.1065

<div class="k-default-codeblock">
```

```
</div>
 297/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:41 237ms/step - accuracy: 0.2267 - loss: 1.1107

<div class="k-default-codeblock">
```

```
</div>
 298/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:41 237ms/step - accuracy: 0.2267 - loss: 1.1150

<div class="k-default-codeblock">
```
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9


```
</div>
 299/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:41 237ms/step - accuracy: 0.2267 - loss: 1.1192

<div class="k-default-codeblock">
```

```
</div>
 300/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:41 237ms/step - accuracy: 0.2267 - loss: 1.1235

<div class="k-default-codeblock">
```

```
</div>
 301/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:40 237ms/step - accuracy: 0.2267 - loss: 1.1278

<div class="k-default-codeblock">
```

```
</div>
 302/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:40 237ms/step - accuracy: 0.2267 - loss: 1.1322

<div class="k-default-codeblock">
```

```
</div>
 303/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:40 237ms/step - accuracy: 0.2267 - loss: 1.1366

<div class="k-default-codeblock">
```

```
</div>
 304/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:40 237ms/step - accuracy: 0.2268 - loss: 1.1410

<div class="k-default-codeblock">
```

```
</div>
 305/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:39 237ms/step - accuracy: 0.2268 - loss: 1.1455

<div class="k-default-codeblock">
```

```
</div>
 306/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:39 237ms/step - accuracy: 0.2268 - loss: 1.1500

<div class="k-default-codeblock">
```

```
</div>
 307/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:39 237ms/step - accuracy: 0.2268 - loss: 1.1545

<div class="k-default-codeblock">
```

```
</div>
 308/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:39 237ms/step - accuracy: 0.2268 - loss: 1.1591

<div class="k-default-codeblock">
```

```
</div>
 309/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:38 237ms/step - accuracy: 0.2269 - loss: 1.1637

<div class="k-default-codeblock">
```

```
</div>
 310/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:38 237ms/step - accuracy: 0.2269 - loss: 1.1683

<div class="k-default-codeblock">
```

```
</div>
 311/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:38 236ms/step - accuracy: 0.2269 - loss: 1.1730

<div class="k-default-codeblock">
```

```
</div>
 312/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:38 236ms/step - accuracy: 0.2270 - loss: 1.1776

<div class="k-default-codeblock">
```

```
</div>
 313/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:37 236ms/step - accuracy: 0.2270 - loss: 1.1823

<div class="k-default-codeblock">
```

```
</div>
 314/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:37 236ms/step - accuracy: 0.2271 - loss: 1.1869

<div class="k-default-codeblock">
```

```
</div>
 315/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:37 236ms/step - accuracy: 0.2271 - loss: 1.1916

<div class="k-default-codeblock">
```

```
</div>
 316/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:37 236ms/step - accuracy: 0.2272 - loss: 1.1963

<div class="k-default-codeblock">
```

```
</div>
 317/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:36 236ms/step - accuracy: 0.2272 - loss: 1.2011

<div class="k-default-codeblock">
```

```
</div>
 318/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:36 236ms/step - accuracy: 0.2273 - loss: 1.2058

<div class="k-default-codeblock">
```

```
</div>
 319/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:36 236ms/step - accuracy: 0.2273 - loss: 1.2105

<div class="k-default-codeblock">
```

```
</div>
 320/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:36 236ms/step - accuracy: 0.2274 - loss: 1.2153

<div class="k-default-codeblock">
```

```
</div>
 321/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:35 236ms/step - accuracy: 0.2274 - loss: 1.2201

<div class="k-default-codeblock">
```

```
</div>
 322/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:35 236ms/step - accuracy: 0.2275 - loss: 1.2249

<div class="k-default-codeblock">
```

```
</div>
 323/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:35 236ms/step - accuracy: 0.2276 - loss: 1.2298

<div class="k-default-codeblock">
```

```
</div>
 324/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:35 236ms/step - accuracy: 0.2276 - loss: 1.2347

<div class="k-default-codeblock">
```

```
</div>
 325/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:34 236ms/step - accuracy: 0.2277 - loss: 1.2396

<div class="k-default-codeblock">
```

```
</div>
 326/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:34 236ms/step - accuracy: 0.2278 - loss: 1.2445

<div class="k-default-codeblock">
```

```
</div>
 327/727 ━━━━━━━━[37m━━━━━━━━━━━━  1:34 236ms/step - accuracy: 0.2278 - loss: 1.2495

<div class="k-default-codeblock">
```

```
</div>
 328/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:34 236ms/step - accuracy: 0.2279 - loss: 1.2544

<div class="k-default-codeblock">
```

```
</div>
 329/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:33 236ms/step - accuracy: 0.2280 - loss: 1.2594

<div class="k-default-codeblock">
```

```
</div>
 330/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:33 236ms/step - accuracy: 0.2281 - loss: 1.2644

<div class="k-default-codeblock">
```

```
</div>
 331/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:33 236ms/step - accuracy: 0.2281 - loss: 1.2694

<div class="k-default-codeblock">
```

```
</div>
 332/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:33 236ms/step - accuracy: 0.2282 - loss: 1.2744

<div class="k-default-codeblock">
```

```
</div>
 333/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:32 236ms/step - accuracy: 0.2283 - loss: 1.2794

<div class="k-default-codeblock">
```

```
</div>
 334/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:32 236ms/step - accuracy: 0.2284 - loss: 1.2844

<div class="k-default-codeblock">
```

```
</div>
 335/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:32 236ms/step - accuracy: 0.2285 - loss: 1.2894

<div class="k-default-codeblock">
```

```
</div>
 336/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:32 236ms/step - accuracy: 0.2286 - loss: 1.2945

<div class="k-default-codeblock">
```

```
</div>
 337/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:31 236ms/step - accuracy: 0.2287 - loss: 1.2995

<div class="k-default-codeblock">
```

```
</div>
 338/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:31 236ms/step - accuracy: 0.2288 - loss: 1.3046

<div class="k-default-codeblock">
```

```
</div>
 339/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:31 236ms/step - accuracy: 0.2289 - loss: 1.3096

<div class="k-default-codeblock">
```

```
</div>
 340/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:31 236ms/step - accuracy: 0.2290 - loss: 1.3147

<div class="k-default-codeblock">
```

```
</div>
 341/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:30 236ms/step - accuracy: 0.2291 - loss: 1.3198

<div class="k-default-codeblock">
```

```
</div>
 342/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:30 236ms/step - accuracy: 0.2292 - loss: 1.3249

<div class="k-default-codeblock">
```

```
</div>
 343/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:30 235ms/step - accuracy: 0.2293 - loss: 1.3300

<div class="k-default-codeblock">
```

```
</div>
 344/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:30 235ms/step - accuracy: 0.2294 - loss: 1.3351

<div class="k-default-codeblock">
```

```
</div>
 345/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:29 235ms/step - accuracy: 0.2295 - loss: 1.3403

<div class="k-default-codeblock">
```

```
</div>
 346/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:29 235ms/step - accuracy: 0.2296 - loss: 1.3455

<div class="k-default-codeblock">
```
Corrupt JPEG data: 252 extraneous bytes before marker 0xd9


```
</div>
 347/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:29 235ms/step - accuracy: 0.2297 - loss: 1.3506

<div class="k-default-codeblock">
```

```
</div>
 348/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:29 235ms/step - accuracy: 0.2298 - loss: 1.3558

<div class="k-default-codeblock">
```
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9


```
</div>
 349/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:28 235ms/step - accuracy: 0.2299 - loss: 1.3610

<div class="k-default-codeblock">
```

```
</div>
 350/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:28 235ms/step - accuracy: 0.2301 - loss: 1.3662

<div class="k-default-codeblock">
```

```
</div>
 351/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:28 235ms/step - accuracy: 0.2302 - loss: 1.3714

<div class="k-default-codeblock">
```

```
</div>
 352/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:28 235ms/step - accuracy: 0.2303 - loss: 1.3766

<div class="k-default-codeblock">
```

```
</div>
 353/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:27 235ms/step - accuracy: 0.2304 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 354/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:27 235ms/step - accuracy: 0.2305 - loss: 1.3870

<div class="k-default-codeblock">
```

```
</div>
 355/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:27 235ms/step - accuracy: 0.2307 - loss: 1.3923

<div class="k-default-codeblock">
```

```
</div>
 356/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:27 235ms/step - accuracy: 0.2308 - loss: 1.3975

<div class="k-default-codeblock">
```

```
</div>
 357/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:26 235ms/step - accuracy: 0.2309 - loss: 1.4028

<div class="k-default-codeblock">
```

```
</div>
 358/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:26 235ms/step - accuracy: 0.2310 - loss: 1.4080

<div class="k-default-codeblock">
```

```
</div>
 359/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:26 235ms/step - accuracy: 0.2311 - loss: 1.4133

<div class="k-default-codeblock">
```

```
</div>
 360/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:26 235ms/step - accuracy: 0.2313 - loss: 1.4185

<div class="k-default-codeblock">
```

```
</div>
 361/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:25 235ms/step - accuracy: 0.2314 - loss: 1.4237

<div class="k-default-codeblock">
```

```
</div>
 362/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:25 235ms/step - accuracy: 0.2315 - loss: 1.4290

<div class="k-default-codeblock">
```

```
</div>
 363/727 ━━━━━━━━━[37m━━━━━━━━━━━  1:25 235ms/step - accuracy: 0.2316 - loss: 1.4342

<div class="k-default-codeblock">
```
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9


```
</div>
 364/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:25 235ms/step - accuracy: 0.2318 - loss: 1.4395

<div class="k-default-codeblock">
```

```
</div>
 365/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:24 235ms/step - accuracy: 0.2319 - loss: 1.4447

<div class="k-default-codeblock">
```

```
</div>
 366/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:24 235ms/step - accuracy: 0.2320 - loss: 1.4500

<div class="k-default-codeblock">
```

```
</div>
 367/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:24 235ms/step - accuracy: 0.2322 - loss: 1.4553

<div class="k-default-codeblock">
```

```
</div>
 368/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:24 235ms/step - accuracy: 0.2323 - loss: 1.4606

<div class="k-default-codeblock">
```

```
</div>
 369/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:24 235ms/step - accuracy: 0.2324 - loss: 1.4659

<div class="k-default-codeblock">
```

```
</div>
 370/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:23 235ms/step - accuracy: 0.2326 - loss: 1.4712

<div class="k-default-codeblock">
```

```
</div>
 371/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:23 235ms/step - accuracy: 0.2327 - loss: 1.4765

<div class="k-default-codeblock">
```

```
</div>
 372/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:23 235ms/step - accuracy: 0.2328 - loss: 1.4818

<div class="k-default-codeblock">
```

```
</div>
 373/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:23 235ms/step - accuracy: 0.2330 - loss: 1.4871

<div class="k-default-codeblock">
```

```
</div>
 374/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:22 234ms/step - accuracy: 0.2331 - loss: 1.4924

<div class="k-default-codeblock">
```

```
</div>
 375/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:22 234ms/step - accuracy: 0.2333 - loss: 1.4977

<div class="k-default-codeblock">
```

```
</div>
 376/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:22 234ms/step - accuracy: 0.2334 - loss: 1.5030

<div class="k-default-codeblock">
```

```
</div>
 377/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:22 234ms/step - accuracy: 0.2335 - loss: 1.5084

<div class="k-default-codeblock">
```

```
</div>
 378/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:21 234ms/step - accuracy: 0.2337 - loss: 1.5137

<div class="k-default-codeblock">
```

```
</div>
 379/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:21 234ms/step - accuracy: 0.2338 - loss: 1.5191

<div class="k-default-codeblock">
```

```
</div>
 380/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:21 234ms/step - accuracy: 0.2340 - loss: 1.5243

<div class="k-default-codeblock">
```

```
</div>
 381/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:21 234ms/step - accuracy: 0.2341 - loss: 1.5295

<div class="k-default-codeblock">
```

```
</div>
 382/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:20 234ms/step - accuracy: 0.2342 - loss: 1.5347

<div class="k-default-codeblock">
```

```
</div>
 383/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:20 234ms/step - accuracy: 0.2344 - loss: 1.5398

<div class="k-default-codeblock">
```

```
</div>
 384/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:20 234ms/step - accuracy: 0.2345 - loss: 1.5449

<div class="k-default-codeblock">
```

```
</div>
 385/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:20 235ms/step - accuracy: 0.2346 - loss: 1.5499

<div class="k-default-codeblock">
```

```
</div>
 386/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:19 235ms/step - accuracy: 0.2348 - loss: 1.5549

<div class="k-default-codeblock">
```

```
</div>
 387/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:19 235ms/step - accuracy: 0.2349 - loss: 1.5598

<div class="k-default-codeblock">
```

```
</div>
 388/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:19 235ms/step - accuracy: 0.2350 - loss: 1.5647

<div class="k-default-codeblock">
```

```
</div>
 389/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:19 235ms/step - accuracy: 0.2352 - loss: 1.5695

<div class="k-default-codeblock">
```

```
</div>
 390/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:19 235ms/step - accuracy: 0.2353 - loss: 1.5743

<div class="k-default-codeblock">
```

```
</div>
 391/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:18 235ms/step - accuracy: 0.2354 - loss: 1.5791

<div class="k-default-codeblock">
```

```
</div>
 392/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:18 235ms/step - accuracy: 0.2355 - loss: 1.5838

<div class="k-default-codeblock">
```

```
</div>
 393/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:18 235ms/step - accuracy: 0.2357 - loss: 1.5885

<div class="k-default-codeblock">
```

```
</div>
 394/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:18 235ms/step - accuracy: 0.2358 - loss: 1.5931

<div class="k-default-codeblock">
```

```
</div>
 395/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:17 235ms/step - accuracy: 0.2359 - loss: 1.5977

<div class="k-default-codeblock">
```

```
</div>
 396/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:17 235ms/step - accuracy: 0.2360 - loss: 1.6022

<div class="k-default-codeblock">
```

```
</div>
 397/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:17 235ms/step - accuracy: 0.2361 - loss: 1.6067

<div class="k-default-codeblock">
```

```
</div>
 398/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:17 235ms/step - accuracy: 0.2363 - loss: 1.6111

<div class="k-default-codeblock">
```

```
</div>
 399/727 ━━━━━━━━━━[37m━━━━━━━━━━  1:16 235ms/step - accuracy: 0.2364 - loss: 1.6156

<div class="k-default-codeblock">
```

```
</div>
 400/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:16 235ms/step - accuracy: 0.2365 - loss: 1.6199

<div class="k-default-codeblock">
```

```
</div>
 401/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:16 235ms/step - accuracy: 0.2366 - loss: 1.6243

<div class="k-default-codeblock">
```

```
</div>
 402/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:16 235ms/step - accuracy: 0.2367 - loss: 1.6286

<div class="k-default-codeblock">
```

```
</div>
 403/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:16 235ms/step - accuracy: 0.2368 - loss: 1.6328

<div class="k-default-codeblock">
```

```
</div>
 404/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:15 235ms/step - accuracy: 0.2369 - loss: 1.6370

<div class="k-default-codeblock">
```

```
</div>
 405/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:15 235ms/step - accuracy: 0.2370 - loss: 1.6412

<div class="k-default-codeblock">
```

```
</div>
 406/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:15 235ms/step - accuracy: 0.2371 - loss: 1.6453

<div class="k-default-codeblock">
```

```
</div>
 407/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:15 235ms/step - accuracy: 0.2372 - loss: 1.6494

<div class="k-default-codeblock">
```

```
</div>
 408/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:14 235ms/step - accuracy: 0.2373 - loss: 1.6535

<div class="k-default-codeblock">
```

```
</div>
 409/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:14 235ms/step - accuracy: 0.2374 - loss: 1.6575

<div class="k-default-codeblock">
```

```
</div>
 410/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:14 235ms/step - accuracy: 0.2375 - loss: 1.6615

<div class="k-default-codeblock">
```

```
</div>
 411/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:14 235ms/step - accuracy: 0.2376 - loss: 1.6654

<div class="k-default-codeblock">
```

```
</div>
 412/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:13 235ms/step - accuracy: 0.2377 - loss: 1.6693

<div class="k-default-codeblock">
```

```
</div>
 413/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:13 235ms/step - accuracy: 0.2378 - loss: 1.6732

<div class="k-default-codeblock">
```

```
</div>
 414/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:13 235ms/step - accuracy: 0.2379 - loss: 1.6770

<div class="k-default-codeblock">
```

```
</div>
 415/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:13 235ms/step - accuracy: 0.2380 - loss: 1.6808

<div class="k-default-codeblock">
```

```
</div>
 416/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:13 235ms/step - accuracy: 0.2381 - loss: 1.6846

<div class="k-default-codeblock">
```

```
</div>
 417/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:12 235ms/step - accuracy: 0.2381 - loss: 1.6883

<div class="k-default-codeblock">
```

```
</div>
 418/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:12 235ms/step - accuracy: 0.2382 - loss: 1.6920

<div class="k-default-codeblock">
```

```
</div>
 419/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:12 235ms/step - accuracy: 0.2383 - loss: 1.6957

<div class="k-default-codeblock">
```

```
</div>
 420/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:12 235ms/step - accuracy: 0.2384 - loss: 1.6993

<div class="k-default-codeblock">
```

```
</div>
 421/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:11 235ms/step - accuracy: 0.2385 - loss: 1.7029

<div class="k-default-codeblock">
```

```
</div>
 422/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:11 235ms/step - accuracy: 0.2385 - loss: 1.7065

<div class="k-default-codeblock">
```

```
</div>
 423/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:11 235ms/step - accuracy: 0.2386 - loss: 1.7100

<div class="k-default-codeblock">
```

```
</div>
 424/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:11 235ms/step - accuracy: 0.2387 - loss: 1.7135

<div class="k-default-codeblock">
```

```
</div>
 425/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:10 235ms/step - accuracy: 0.2388 - loss: 1.7169

<div class="k-default-codeblock">
```

```
</div>
 426/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:10 235ms/step - accuracy: 0.2388 - loss: 1.7204

<div class="k-default-codeblock">
```

```
</div>
 427/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:10 235ms/step - accuracy: 0.2389 - loss: 1.7238

<div class="k-default-codeblock">
```

```
</div>
 428/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:10 235ms/step - accuracy: 0.2390 - loss: 1.7271

<div class="k-default-codeblock">
```

```
</div>
 429/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:10 235ms/step - accuracy: 0.2391 - loss: 1.7305

<div class="k-default-codeblock">
```

```
</div>
 430/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:09 235ms/step - accuracy: 0.2391 - loss: 1.7338

<div class="k-default-codeblock">
```

```
</div>
 431/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:09 235ms/step - accuracy: 0.2392 - loss: 1.7370

<div class="k-default-codeblock">
```

```
</div>
 432/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:09 235ms/step - accuracy: 0.2393 - loss: 1.7403

<div class="k-default-codeblock">
```

```
</div>
 433/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:09 235ms/step - accuracy: 0.2393 - loss: 1.7435

<div class="k-default-codeblock">
```

```
</div>
 434/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:08 235ms/step - accuracy: 0.2394 - loss: 1.7467

<div class="k-default-codeblock">
```

```
</div>
 435/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:08 235ms/step - accuracy: 0.2394 - loss: 1.7498

<div class="k-default-codeblock">
```

```
</div>
 436/727 ━━━━━━━━━━━[37m━━━━━━━━━  1:08 235ms/step - accuracy: 0.2395 - loss: 1.7529

<div class="k-default-codeblock">
```

```
</div>
 437/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:08 235ms/step - accuracy: 0.2396 - loss: 1.7560

<div class="k-default-codeblock">
```

```
</div>
 438/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:07 235ms/step - accuracy: 0.2396 - loss: 1.7591

<div class="k-default-codeblock">
```

```
</div>
 439/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:07 235ms/step - accuracy: 0.2397 - loss: 1.7621

<div class="k-default-codeblock">
```

```
</div>
 440/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:07 235ms/step - accuracy: 0.2397 - loss: 1.7651

<div class="k-default-codeblock">
```

```
</div>
 441/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:07 235ms/step - accuracy: 0.2398 - loss: 1.7681

<div class="k-default-codeblock">
```

```
</div>
 442/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:07 235ms/step - accuracy: 0.2398 - loss: 1.7711

<div class="k-default-codeblock">
```

```
</div>
 443/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:06 235ms/step - accuracy: 0.2399 - loss: 1.7740

<div class="k-default-codeblock">
```

```
</div>
 444/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:06 235ms/step - accuracy: 0.2399 - loss: 1.7769

<div class="k-default-codeblock">
```

```
</div>
 445/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:06 235ms/step - accuracy: 0.2400 - loss: 1.7797

<div class="k-default-codeblock">
```

```
</div>
 446/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:06 235ms/step - accuracy: 0.2400 - loss: 1.7826

<div class="k-default-codeblock">
```

```
</div>
 447/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:05 235ms/step - accuracy: 0.2401 - loss: 1.7854

<div class="k-default-codeblock">
```

```
</div>
 448/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:05 235ms/step - accuracy: 0.2401 - loss: 1.7882

<div class="k-default-codeblock">
```

```
</div>
 449/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:05 235ms/step - accuracy: 0.2401 - loss: 1.7909

<div class="k-default-codeblock">
```

```
</div>
 450/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:05 235ms/step - accuracy: 0.2402 - loss: 1.7937

<div class="k-default-codeblock">
```

```
</div>
 451/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:04 235ms/step - accuracy: 0.2402 - loss: 1.7964

<div class="k-default-codeblock">
```

```
</div>
 452/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:04 235ms/step - accuracy: 0.2403 - loss: 1.7991

<div class="k-default-codeblock">
```

```
</div>
 453/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:04 235ms/step - accuracy: 0.2403 - loss: 1.8017

<div class="k-default-codeblock">
```

```
</div>
 454/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:04 235ms/step - accuracy: 0.2403 - loss: 1.8044

<div class="k-default-codeblock">
```

```
</div>
 455/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:03 235ms/step - accuracy: 0.2404 - loss: 1.8070

<div class="k-default-codeblock">
```

```
</div>
 456/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:03 235ms/step - accuracy: 0.2404 - loss: 1.8096

<div class="k-default-codeblock">
```

```
</div>
 457/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:03 235ms/step - accuracy: 0.2405 - loss: 1.8121

<div class="k-default-codeblock">
```

```
</div>
 458/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:03 235ms/step - accuracy: 0.2405 - loss: 1.8147

<div class="k-default-codeblock">
```

```
</div>
 459/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:03 235ms/step - accuracy: 0.2405 - loss: 1.8172

<div class="k-default-codeblock">
```

```
</div>
 460/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:02 235ms/step - accuracy: 0.2406 - loss: 1.8197

<div class="k-default-codeblock">
```

```
</div>
 461/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:02 235ms/step - accuracy: 0.2406 - loss: 1.8222

<div class="k-default-codeblock">
```

```
</div>
 462/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:02 235ms/step - accuracy: 0.2406 - loss: 1.8246

<div class="k-default-codeblock">
```

```
</div>
 463/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:02 235ms/step - accuracy: 0.2407 - loss: 1.8271

<div class="k-default-codeblock">
```

```
</div>
 464/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:01 235ms/step - accuracy: 0.2407 - loss: 1.8295

<div class="k-default-codeblock">
```

```
</div>
 465/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:01 235ms/step - accuracy: 0.2407 - loss: 1.8319

<div class="k-default-codeblock">
```

```
</div>
 466/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:01 235ms/step - accuracy: 0.2407 - loss: 1.8342

<div class="k-default-codeblock">
```

```
</div>
 467/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:01 235ms/step - accuracy: 0.2408 - loss: 1.8366

<div class="k-default-codeblock">
```

```
</div>
 468/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:00 235ms/step - accuracy: 0.2408 - loss: 1.8389

<div class="k-default-codeblock">
```

```
</div>
 469/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:00 235ms/step - accuracy: 0.2408 - loss: 1.8412

<div class="k-default-codeblock">
```

```
</div>
 470/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:00 235ms/step - accuracy: 0.2408 - loss: 1.8435

<div class="k-default-codeblock">
```

```
</div>
 471/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:00 235ms/step - accuracy: 0.2409 - loss: 1.8457

<div class="k-default-codeblock">
```

```
</div>
 472/727 ━━━━━━━━━━━━[37m━━━━━━━━  1:00 235ms/step - accuracy: 0.2409 - loss: 1.8480

<div class="k-default-codeblock">
```

```
</div>
 473/727 ━━━━━━━━━━━━━[37m━━━━━━━  59s 235ms/step - accuracy: 0.2409 - loss: 1.8502 

<div class="k-default-codeblock">
```

```
</div>
 474/727 ━━━━━━━━━━━━━[37m━━━━━━━  59s 235ms/step - accuracy: 0.2409 - loss: 1.8524

<div class="k-default-codeblock">
```

```
</div>
 475/727 ━━━━━━━━━━━━━[37m━━━━━━━  59s 235ms/step - accuracy: 0.2409 - loss: 1.8546

<div class="k-default-codeblock">
```

```
</div>
 476/727 ━━━━━━━━━━━━━[37m━━━━━━━  59s 235ms/step - accuracy: 0.2410 - loss: 1.8567

<div class="k-default-codeblock">
```

```
</div>
 477/727 ━━━━━━━━━━━━━[37m━━━━━━━  58s 235ms/step - accuracy: 0.2410 - loss: 1.8588

<div class="k-default-codeblock">
```

```
</div>
 478/727 ━━━━━━━━━━━━━[37m━━━━━━━  58s 235ms/step - accuracy: 0.2410 - loss: 1.8610

<div class="k-default-codeblock">
```

```
</div>
 479/727 ━━━━━━━━━━━━━[37m━━━━━━━  58s 235ms/step - accuracy: 0.2410 - loss: 1.8631

<div class="k-default-codeblock">
```

```
</div>
 480/727 ━━━━━━━━━━━━━[37m━━━━━━━  58s 235ms/step - accuracy: 0.2410 - loss: 1.8651

<div class="k-default-codeblock">
```

```
</div>
 481/727 ━━━━━━━━━━━━━[37m━━━━━━━  57s 235ms/step - accuracy: 0.2410 - loss: 1.8672

<div class="k-default-codeblock">
```

```
</div>
 482/727 ━━━━━━━━━━━━━[37m━━━━━━━  57s 235ms/step - accuracy: 0.2410 - loss: 1.8692

<div class="k-default-codeblock">
```

```
</div>
 483/727 ━━━━━━━━━━━━━[37m━━━━━━━  57s 235ms/step - accuracy: 0.2411 - loss: 1.8713

<div class="k-default-codeblock">
```

```
</div>
 484/727 ━━━━━━━━━━━━━[37m━━━━━━━  57s 235ms/step - accuracy: 0.2411 - loss: 1.8733

<div class="k-default-codeblock">
```

```
</div>
 485/727 ━━━━━━━━━━━━━[37m━━━━━━━  56s 235ms/step - accuracy: 0.2411 - loss: 1.8753

<div class="k-default-codeblock">
```

```
</div>
 486/727 ━━━━━━━━━━━━━[37m━━━━━━━  56s 235ms/step - accuracy: 0.2411 - loss: 1.8772

<div class="k-default-codeblock">
```

```
</div>
 487/727 ━━━━━━━━━━━━━[37m━━━━━━━  56s 235ms/step - accuracy: 0.2411 - loss: 1.8792

<div class="k-default-codeblock">
```

```
</div>
 488/727 ━━━━━━━━━━━━━[37m━━━━━━━  56s 235ms/step - accuracy: 0.2411 - loss: 1.8811

<div class="k-default-codeblock">
```

```
</div>
 489/727 ━━━━━━━━━━━━━[37m━━━━━━━  56s 235ms/step - accuracy: 0.2411 - loss: 1.8830

<div class="k-default-codeblock">
```

```
</div>
 490/727 ━━━━━━━━━━━━━[37m━━━━━━━  55s 235ms/step - accuracy: 0.2412 - loss: 1.8849

<div class="k-default-codeblock">
```

```
</div>
 491/727 ━━━━━━━━━━━━━[37m━━━━━━━  55s 235ms/step - accuracy: 0.2412 - loss: 1.8868

<div class="k-default-codeblock">
```

```
</div>
 492/727 ━━━━━━━━━━━━━[37m━━━━━━━  55s 235ms/step - accuracy: 0.2412 - loss: 1.8886

<div class="k-default-codeblock">
```

```
</div>
 493/727 ━━━━━━━━━━━━━[37m━━━━━━━  55s 235ms/step - accuracy: 0.2412 - loss: 1.8905

<div class="k-default-codeblock">
```

```
</div>
 494/727 ━━━━━━━━━━━━━[37m━━━━━━━  54s 235ms/step - accuracy: 0.2412 - loss: 1.8923

<div class="k-default-codeblock">
```

```
</div>
 495/727 ━━━━━━━━━━━━━[37m━━━━━━━  54s 236ms/step - accuracy: 0.2412 - loss: 1.8941

<div class="k-default-codeblock">
```

```
</div>
 496/727 ━━━━━━━━━━━━━[37m━━━━━━━  54s 236ms/step - accuracy: 0.2412 - loss: 1.8959

<div class="k-default-codeblock">
```

```
</div>
 497/727 ━━━━━━━━━━━━━[37m━━━━━━━  54s 236ms/step - accuracy: 0.2412 - loss: 1.8977

<div class="k-default-codeblock">
```

```
</div>
 498/727 ━━━━━━━━━━━━━[37m━━━━━━━  53s 236ms/step - accuracy: 0.2412 - loss: 1.8994

<div class="k-default-codeblock">
```

```
</div>
 499/727 ━━━━━━━━━━━━━[37m━━━━━━━  53s 236ms/step - accuracy: 0.2412 - loss: 1.9011

<div class="k-default-codeblock">
```

```
</div>
 500/727 ━━━━━━━━━━━━━[37m━━━━━━━  53s 236ms/step - accuracy: 0.2412 - loss: 1.9029

<div class="k-default-codeblock">
```

```
</div>
 501/727 ━━━━━━━━━━━━━[37m━━━━━━━  53s 236ms/step - accuracy: 0.2412 - loss: 1.9046

<div class="k-default-codeblock">
```

```
</div>
 502/727 ━━━━━━━━━━━━━[37m━━━━━━━  52s 236ms/step - accuracy: 0.2413 - loss: 1.9062

<div class="k-default-codeblock">
```

```
</div>
 503/727 ━━━━━━━━━━━━━[37m━━━━━━━  52s 236ms/step - accuracy: 0.2413 - loss: 1.9079

<div class="k-default-codeblock">
```

```
</div>
 504/727 ━━━━━━━━━━━━━[37m━━━━━━━  52s 236ms/step - accuracy: 0.2413 - loss: 1.9096

<div class="k-default-codeblock">
```

```
</div>
 505/727 ━━━━━━━━━━━━━[37m━━━━━━━  52s 236ms/step - accuracy: 0.2413 - loss: 1.9112

<div class="k-default-codeblock">
```

```
</div>
 506/727 ━━━━━━━━━━━━━[37m━━━━━━━  52s 236ms/step - accuracy: 0.2413 - loss: 1.9128

<div class="k-default-codeblock">
```

```
</div>
 507/727 ━━━━━━━━━━━━━[37m━━━━━━━  51s 236ms/step - accuracy: 0.2413 - loss: 1.9144

<div class="k-default-codeblock">
```

```
</div>
 508/727 ━━━━━━━━━━━━━[37m━━━━━━━  51s 236ms/step - accuracy: 0.2413 - loss: 1.9160

<div class="k-default-codeblock">
```

```
</div>
 509/727 ━━━━━━━━━━━━━━[37m━━━━━━  51s 236ms/step - accuracy: 0.2413 - loss: 1.9176

<div class="k-default-codeblock">
```

```
</div>
 510/727 ━━━━━━━━━━━━━━[37m━━━━━━  51s 236ms/step - accuracy: 0.2413 - loss: 1.9191

<div class="k-default-codeblock">
```

```
</div>
 511/727 ━━━━━━━━━━━━━━[37m━━━━━━  50s 236ms/step - accuracy: 0.2413 - loss: 1.9206

<div class="k-default-codeblock">
```

```
</div>
 512/727 ━━━━━━━━━━━━━━[37m━━━━━━  50s 236ms/step - accuracy: 0.2413 - loss: 1.9222

<div class="k-default-codeblock">
```

```
</div>
 513/727 ━━━━━━━━━━━━━━[37m━━━━━━  50s 236ms/step - accuracy: 0.2413 - loss: 1.9237

<div class="k-default-codeblock">
```

```
</div>
 514/727 ━━━━━━━━━━━━━━[37m━━━━━━  50s 236ms/step - accuracy: 0.2413 - loss: 1.9252

<div class="k-default-codeblock">
```

```
</div>
 515/727 ━━━━━━━━━━━━━━[37m━━━━━━  49s 236ms/step - accuracy: 0.2413 - loss: 1.9266

<div class="k-default-codeblock">
```

```
</div>
 516/727 ━━━━━━━━━━━━━━[37m━━━━━━  49s 236ms/step - accuracy: 0.2412 - loss: 1.9281

<div class="k-default-codeblock">
```

```
</div>
 517/727 ━━━━━━━━━━━━━━[37m━━━━━━  49s 236ms/step - accuracy: 0.2412 - loss: 1.9295

<div class="k-default-codeblock">
```

```
</div>
 518/727 ━━━━━━━━━━━━━━[37m━━━━━━  49s 236ms/step - accuracy: 0.2412 - loss: 1.9310

<div class="k-default-codeblock">
```

```
</div>
 519/727 ━━━━━━━━━━━━━━[37m━━━━━━  49s 236ms/step - accuracy: 0.2412 - loss: 1.9324

<div class="k-default-codeblock">
```

```
</div>
 520/727 ━━━━━━━━━━━━━━[37m━━━━━━  48s 236ms/step - accuracy: 0.2412 - loss: 1.9338

<div class="k-default-codeblock">
```

```
</div>
 521/727 ━━━━━━━━━━━━━━[37m━━━━━━  48s 236ms/step - accuracy: 0.2412 - loss: 1.9352

<div class="k-default-codeblock">
```

```
</div>
 522/727 ━━━━━━━━━━━━━━[37m━━━━━━  48s 236ms/step - accuracy: 0.2412 - loss: 1.9366

<div class="k-default-codeblock">
```
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9


```
</div>
 523/727 ━━━━━━━━━━━━━━[37m━━━━━━  48s 236ms/step - accuracy: 0.2412 - loss: 1.9379

<div class="k-default-codeblock">
```

```
</div>
 524/727 ━━━━━━━━━━━━━━[37m━━━━━━  47s 236ms/step - accuracy: 0.2412 - loss: 1.9393

<div class="k-default-codeblock">
```

```
</div>
 525/727 ━━━━━━━━━━━━━━[37m━━━━━━  47s 236ms/step - accuracy: 0.2412 - loss: 1.9406

<div class="k-default-codeblock">
```

```
</div>
 526/727 ━━━━━━━━━━━━━━[37m━━━━━━  47s 236ms/step - accuracy: 0.2412 - loss: 1.9419

<div class="k-default-codeblock">
```

```
</div>
 527/727 ━━━━━━━━━━━━━━[37m━━━━━━  47s 236ms/step - accuracy: 0.2412 - loss: 1.9432

<div class="k-default-codeblock">
```

```
</div>
 528/727 ━━━━━━━━━━━━━━[37m━━━━━━  46s 236ms/step - accuracy: 0.2411 - loss: 1.9445

<div class="k-default-codeblock">
```

```
</div>
 529/727 ━━━━━━━━━━━━━━[37m━━━━━━  46s 236ms/step - accuracy: 0.2411 - loss: 1.9458

<div class="k-default-codeblock">
```

```
</div>
 530/727 ━━━━━━━━━━━━━━[37m━━━━━━  46s 236ms/step - accuracy: 0.2411 - loss: 1.9471

<div class="k-default-codeblock">
```

```
</div>
 531/727 ━━━━━━━━━━━━━━[37m━━━━━━  46s 236ms/step - accuracy: 0.2411 - loss: 1.9483

<div class="k-default-codeblock">
```

```
</div>
 532/727 ━━━━━━━━━━━━━━[37m━━━━━━  45s 236ms/step - accuracy: 0.2411 - loss: 1.9496

<div class="k-default-codeblock">
```

```
</div>
 533/727 ━━━━━━━━━━━━━━[37m━━━━━━  45s 236ms/step - accuracy: 0.2411 - loss: 1.9508

<div class="k-default-codeblock">
```

```
</div>
 534/727 ━━━━━━━━━━━━━━[37m━━━━━━  45s 236ms/step - accuracy: 0.2411 - loss: 1.9521

<div class="k-default-codeblock">
```

```
</div>
 535/727 ━━━━━━━━━━━━━━[37m━━━━━━  45s 236ms/step - accuracy: 0.2411 - loss: 1.9533

<div class="k-default-codeblock">
```

```
</div>
 536/727 ━━━━━━━━━━━━━━[37m━━━━━━  45s 236ms/step - accuracy: 0.2410 - loss: 1.9545

<div class="k-default-codeblock">
```

```
</div>
 537/727 ━━━━━━━━━━━━━━[37m━━━━━━  44s 236ms/step - accuracy: 0.2410 - loss: 1.9557

<div class="k-default-codeblock">
```

```
</div>
 538/727 ━━━━━━━━━━━━━━[37m━━━━━━  44s 236ms/step - accuracy: 0.2410 - loss: 1.9569

<div class="k-default-codeblock">
```

```
</div>
 539/727 ━━━━━━━━━━━━━━[37m━━━━━━  44s 236ms/step - accuracy: 0.2410 - loss: 1.9580

<div class="k-default-codeblock">
```

```
</div>
 540/727 ━━━━━━━━━━━━━━[37m━━━━━━  44s 236ms/step - accuracy: 0.2410 - loss: 1.9592

<div class="k-default-codeblock">
```

```
</div>
 541/727 ━━━━━━━━━━━━━━[37m━━━━━━  43s 236ms/step - accuracy: 0.2409 - loss: 1.9603

<div class="k-default-codeblock">
```

```
</div>
 542/727 ━━━━━━━━━━━━━━[37m━━━━━━  43s 236ms/step - accuracy: 0.2409 - loss: 1.9615

<div class="k-default-codeblock">
```

```
</div>
 543/727 ━━━━━━━━━━━━━━[37m━━━━━━  43s 236ms/step - accuracy: 0.2409 - loss: 1.9626

<div class="k-default-codeblock">
```

```
</div>
 544/727 ━━━━━━━━━━━━━━[37m━━━━━━  43s 236ms/step - accuracy: 0.2409 - loss: 1.9637

<div class="k-default-codeblock">
```

```
</div>
 545/727 ━━━━━━━━━━━━━━[37m━━━━━━  42s 236ms/step - accuracy: 0.2409 - loss: 1.9648

<div class="k-default-codeblock">
```

```
</div>
 546/727 ━━━━━━━━━━━━━━━[37m━━━━━  42s 236ms/step - accuracy: 0.2408 - loss: 1.9659

<div class="k-default-codeblock">
```

```
</div>
 547/727 ━━━━━━━━━━━━━━━[37m━━━━━  42s 236ms/step - accuracy: 0.2408 - loss: 1.9670

<div class="k-default-codeblock">
```

```
</div>
 548/727 ━━━━━━━━━━━━━━━[37m━━━━━  42s 236ms/step - accuracy: 0.2408 - loss: 1.9681

<div class="k-default-codeblock">
```

```
</div>
 549/727 ━━━━━━━━━━━━━━━[37m━━━━━  41s 236ms/step - accuracy: 0.2408 - loss: 1.9691

<div class="k-default-codeblock">
```

```
</div>
 550/727 ━━━━━━━━━━━━━━━[37m━━━━━  41s 236ms/step - accuracy: 0.2408 - loss: 1.9702

<div class="k-default-codeblock">
```

```
</div>
 551/727 ━━━━━━━━━━━━━━━[37m━━━━━  41s 236ms/step - accuracy: 0.2407 - loss: 1.9712

<div class="k-default-codeblock">
```

```
</div>
 552/727 ━━━━━━━━━━━━━━━[37m━━━━━  41s 236ms/step - accuracy: 0.2407 - loss: 1.9722

<div class="k-default-codeblock">
```

```
</div>
 553/727 ━━━━━━━━━━━━━━━[37m━━━━━  41s 236ms/step - accuracy: 0.2407 - loss: 1.9733

<div class="k-default-codeblock">
```

```
</div>
 554/727 ━━━━━━━━━━━━━━━[37m━━━━━  40s 236ms/step - accuracy: 0.2407 - loss: 1.9743

<div class="k-default-codeblock">
```

```
</div>
 555/727 ━━━━━━━━━━━━━━━[37m━━━━━  40s 236ms/step - accuracy: 0.2406 - loss: 1.9753

<div class="k-default-codeblock">
```

```
</div>
 556/727 ━━━━━━━━━━━━━━━[37m━━━━━  40s 236ms/step - accuracy: 0.2406 - loss: 1.9762

<div class="k-default-codeblock">
```

```
</div>
 557/727 ━━━━━━━━━━━━━━━[37m━━━━━  40s 236ms/step - accuracy: 0.2406 - loss: 1.9772

<div class="k-default-codeblock">
```

```
</div>
 558/727 ━━━━━━━━━━━━━━━[37m━━━━━  39s 236ms/step - accuracy: 0.2406 - loss: 1.9782

<div class="k-default-codeblock">
```

```
</div>
 559/727 ━━━━━━━━━━━━━━━[37m━━━━━  39s 236ms/step - accuracy: 0.2406 - loss: 1.9791

<div class="k-default-codeblock">
```

```
</div>
 560/727 ━━━━━━━━━━━━━━━[37m━━━━━  39s 236ms/step - accuracy: 0.2405 - loss: 1.9801

<div class="k-default-codeblock">
```

```
</div>
 561/727 ━━━━━━━━━━━━━━━[37m━━━━━  39s 236ms/step - accuracy: 0.2405 - loss: 1.9810

<div class="k-default-codeblock">
```

```
</div>
 562/727 ━━━━━━━━━━━━━━━[37m━━━━━  38s 236ms/step - accuracy: 0.2405 - loss: 1.9819

<div class="k-default-codeblock">
```

```
</div>
 563/727 ━━━━━━━━━━━━━━━[37m━━━━━  38s 236ms/step - accuracy: 0.2405 - loss: 1.9828

<div class="k-default-codeblock">
```

```
</div>
 564/727 ━━━━━━━━━━━━━━━[37m━━━━━  38s 236ms/step - accuracy: 0.2404 - loss: 1.9837

<div class="k-default-codeblock">
```

```
</div>
 565/727 ━━━━━━━━━━━━━━━[37m━━━━━  38s 236ms/step - accuracy: 0.2404 - loss: 1.9846

<div class="k-default-codeblock">
```

```
</div>
 566/727 ━━━━━━━━━━━━━━━[37m━━━━━  37s 236ms/step - accuracy: 0.2404 - loss: 1.9855

<div class="k-default-codeblock">
```

```
</div>
 567/727 ━━━━━━━━━━━━━━━[37m━━━━━  37s 236ms/step - accuracy: 0.2404 - loss: 1.9864

<div class="k-default-codeblock">
```

```
</div>
 568/727 ━━━━━━━━━━━━━━━[37m━━━━━  37s 236ms/step - accuracy: 0.2403 - loss: 1.9872

<div class="k-default-codeblock">
```

```
</div>
 569/727 ━━━━━━━━━━━━━━━[37m━━━━━  37s 236ms/step - accuracy: 0.2403 - loss: 1.9881

<div class="k-default-codeblock">
```

```
</div>
 570/727 ━━━━━━━━━━━━━━━[37m━━━━━  37s 236ms/step - accuracy: 0.2403 - loss: 1.9889

<div class="k-default-codeblock">
```

```
</div>
 571/727 ━━━━━━━━━━━━━━━[37m━━━━━  36s 236ms/step - accuracy: 0.2402 - loss: 1.9898

<div class="k-default-codeblock">
```

```
</div>
 572/727 ━━━━━━━━━━━━━━━[37m━━━━━  36s 236ms/step - accuracy: 0.2402 - loss: 1.9906

<div class="k-default-codeblock">
```

```
</div>
 573/727 ━━━━━━━━━━━━━━━[37m━━━━━  36s 236ms/step - accuracy: 0.2402 - loss: 1.9914

<div class="k-default-codeblock">
```

```
</div>
 574/727 ━━━━━━━━━━━━━━━[37m━━━━━  36s 236ms/step - accuracy: 0.2402 - loss: 1.9922

<div class="k-default-codeblock">
```

```
</div>
 575/727 ━━━━━━━━━━━━━━━[37m━━━━━  35s 236ms/step - accuracy: 0.2401 - loss: 1.9930

<div class="k-default-codeblock">
```

```
</div>
 576/727 ━━━━━━━━━━━━━━━[37m━━━━━  35s 236ms/step - accuracy: 0.2401 - loss: 1.9938

<div class="k-default-codeblock">
```

```
</div>
 577/727 ━━━━━━━━━━━━━━━[37m━━━━━  35s 236ms/step - accuracy: 0.2401 - loss: 1.9945

<div class="k-default-codeblock">
```

```
</div>
 578/727 ━━━━━━━━━━━━━━━[37m━━━━━  35s 236ms/step - accuracy: 0.2400 - loss: 1.9953

<div class="k-default-codeblock">
```

```
</div>
 579/727 ━━━━━━━━━━━━━━━[37m━━━━━  34s 236ms/step - accuracy: 0.2400 - loss: 1.9960

<div class="k-default-codeblock">
```

```
</div>
 580/727 ━━━━━━━━━━━━━━━[37m━━━━━  34s 236ms/step - accuracy: 0.2400 - loss: 1.9968

<div class="k-default-codeblock">
```

```
</div>
 581/727 ━━━━━━━━━━━━━━━[37m━━━━━  34s 236ms/step - accuracy: 0.2400 - loss: 1.9975

<div class="k-default-codeblock">
```

```
</div>
 582/727 ━━━━━━━━━━━━━━━━[37m━━━━  34s 236ms/step - accuracy: 0.2399 - loss: 1.9982

<div class="k-default-codeblock">
```

```
</div>
 583/727 ━━━━━━━━━━━━━━━━[37m━━━━  33s 236ms/step - accuracy: 0.2399 - loss: 1.9990

<div class="k-default-codeblock">
```

```
</div>
 584/727 ━━━━━━━━━━━━━━━━[37m━━━━  33s 236ms/step - accuracy: 0.2399 - loss: 1.9997

<div class="k-default-codeblock">
```

```
</div>
 585/727 ━━━━━━━━━━━━━━━━[37m━━━━  33s 236ms/step - accuracy: 0.2398 - loss: 2.0004

<div class="k-default-codeblock">
```

```
</div>
 586/727 ━━━━━━━━━━━━━━━━[37m━━━━  33s 236ms/step - accuracy: 0.2398 - loss: 2.0011

<div class="k-default-codeblock">
```

```
</div>
 587/727 ━━━━━━━━━━━━━━━━[37m━━━━  33s 236ms/step - accuracy: 0.2398 - loss: 2.0017

<div class="k-default-codeblock">
```

```
</div>
 588/727 ━━━━━━━━━━━━━━━━[37m━━━━  32s 236ms/step - accuracy: 0.2397 - loss: 2.0024

<div class="k-default-codeblock">
```

```
</div>
 589/727 ━━━━━━━━━━━━━━━━[37m━━━━  32s 236ms/step - accuracy: 0.2397 - loss: 2.0031

<div class="k-default-codeblock">
```

```
</div>
 590/727 ━━━━━━━━━━━━━━━━[37m━━━━  32s 236ms/step - accuracy: 0.2397 - loss: 2.0037

<div class="k-default-codeblock">
```

```
</div>
 591/727 ━━━━━━━━━━━━━━━━[37m━━━━  32s 236ms/step - accuracy: 0.2396 - loss: 2.0044

<div class="k-default-codeblock">
```

```
</div>
 592/727 ━━━━━━━━━━━━━━━━[37m━━━━  31s 236ms/step - accuracy: 0.2396 - loss: 2.0050

<div class="k-default-codeblock">
```

```
</div>
 593/727 ━━━━━━━━━━━━━━━━[37m━━━━  31s 236ms/step - accuracy: 0.2396 - loss: 2.0056

<div class="k-default-codeblock">
```

```
</div>
 594/727 ━━━━━━━━━━━━━━━━[37m━━━━  31s 236ms/step - accuracy: 0.2395 - loss: 2.0063

<div class="k-default-codeblock">
```

```
</div>
 595/727 ━━━━━━━━━━━━━━━━[37m━━━━  31s 236ms/step - accuracy: 0.2395 - loss: 2.0069

<div class="k-default-codeblock">
```

```
</div>
 596/727 ━━━━━━━━━━━━━━━━[37m━━━━  30s 236ms/step - accuracy: 0.2395 - loss: 2.0075

<div class="k-default-codeblock">
```

```
</div>
 597/727 ━━━━━━━━━━━━━━━━[37m━━━━  30s 236ms/step - accuracy: 0.2394 - loss: 2.0081

<div class="k-default-codeblock">
```

```
</div>
 598/727 ━━━━━━━━━━━━━━━━[37m━━━━  30s 236ms/step - accuracy: 0.2394 - loss: 2.0087

<div class="k-default-codeblock">
```

```
</div>
 599/727 ━━━━━━━━━━━━━━━━[37m━━━━  30s 236ms/step - accuracy: 0.2394 - loss: 2.0093

<div class="k-default-codeblock">
```

```
</div>
 600/727 ━━━━━━━━━━━━━━━━[37m━━━━  29s 236ms/step - accuracy: 0.2393 - loss: 2.0098

<div class="k-default-codeblock">
```

```
</div>
 601/727 ━━━━━━━━━━━━━━━━[37m━━━━  29s 236ms/step - accuracy: 0.2393 - loss: 2.0104

<div class="k-default-codeblock">
```

```
</div>
 602/727 ━━━━━━━━━━━━━━━━[37m━━━━  29s 236ms/step - accuracy: 0.2392 - loss: 2.0110

<div class="k-default-codeblock">
```

```
</div>
 603/727 ━━━━━━━━━━━━━━━━[37m━━━━  29s 236ms/step - accuracy: 0.2392 - loss: 2.0115

<div class="k-default-codeblock">
```

```
</div>
 604/727 ━━━━━━━━━━━━━━━━[37m━━━━  29s 236ms/step - accuracy: 0.2392 - loss: 2.0121

<div class="k-default-codeblock">
```

```
</div>
 605/727 ━━━━━━━━━━━━━━━━[37m━━━━  28s 236ms/step - accuracy: 0.2391 - loss: 2.0126

<div class="k-default-codeblock">
```

```
</div>
 606/727 ━━━━━━━━━━━━━━━━[37m━━━━  28s 236ms/step - accuracy: 0.2391 - loss: 2.0131

<div class="k-default-codeblock">
```

```
</div>
 607/727 ━━━━━━━━━━━━━━━━[37m━━━━  28s 236ms/step - accuracy: 0.2391 - loss: 2.0137

<div class="k-default-codeblock">
```

```
</div>
 608/727 ━━━━━━━━━━━━━━━━[37m━━━━  28s 236ms/step - accuracy: 0.2390 - loss: 2.0142

<div class="k-default-codeblock">
```

```
</div>
 609/727 ━━━━━━━━━━━━━━━━[37m━━━━  27s 236ms/step - accuracy: 0.2390 - loss: 2.0147

<div class="k-default-codeblock">
```

```
</div>
 610/727 ━━━━━━━━━━━━━━━━[37m━━━━  27s 236ms/step - accuracy: 0.2389 - loss: 2.0152

<div class="k-default-codeblock">
```

```
</div>
 611/727 ━━━━━━━━━━━━━━━━[37m━━━━  27s 236ms/step - accuracy: 0.2389 - loss: 2.0157

<div class="k-default-codeblock">
```

```
</div>
 612/727 ━━━━━━━━━━━━━━━━[37m━━━━  27s 236ms/step - accuracy: 0.2389 - loss: 2.0162

<div class="k-default-codeblock">
```

```
</div>
 613/727 ━━━━━━━━━━━━━━━━[37m━━━━  26s 236ms/step - accuracy: 0.2388 - loss: 2.0166

<div class="k-default-codeblock">
```

```
</div>
 614/727 ━━━━━━━━━━━━━━━━[37m━━━━  26s 236ms/step - accuracy: 0.2388 - loss: 2.0171

<div class="k-default-codeblock">
```

```
</div>
 615/727 ━━━━━━━━━━━━━━━━[37m━━━━  26s 236ms/step - accuracy: 0.2387 - loss: 2.0176

<div class="k-default-codeblock">
```

```
</div>
 616/727 ━━━━━━━━━━━━━━━━[37m━━━━  26s 236ms/step - accuracy: 0.2387 - loss: 2.0181

<div class="k-default-codeblock">
```

```
</div>
 617/727 ━━━━━━━━━━━━━━━━[37m━━━━  25s 236ms/step - accuracy: 0.2387 - loss: 2.0185

<div class="k-default-codeblock">
```

```
</div>
 618/727 ━━━━━━━━━━━━━━━━━[37m━━━  25s 236ms/step - accuracy: 0.2386 - loss: 2.0190

<div class="k-default-codeblock">
```

```
</div>
 619/727 ━━━━━━━━━━━━━━━━━[37m━━━  25s 236ms/step - accuracy: 0.2386 - loss: 2.0194

<div class="k-default-codeblock">
```

```
</div>
 620/727 ━━━━━━━━━━━━━━━━━[37m━━━  25s 236ms/step - accuracy: 0.2385 - loss: 2.0199

<div class="k-default-codeblock">
```

```
</div>
 621/727 ━━━━━━━━━━━━━━━━━[37m━━━  25s 236ms/step - accuracy: 0.2385 - loss: 2.0203

<div class="k-default-codeblock">
```

```
</div>
 622/727 ━━━━━━━━━━━━━━━━━[37m━━━  24s 236ms/step - accuracy: 0.2385 - loss: 2.0207

<div class="k-default-codeblock">
```

```
</div>
 623/727 ━━━━━━━━━━━━━━━━━[37m━━━  24s 236ms/step - accuracy: 0.2384 - loss: 2.0212

<div class="k-default-codeblock">
```

```
</div>
 624/727 ━━━━━━━━━━━━━━━━━[37m━━━  24s 236ms/step - accuracy: 0.2384 - loss: 2.0216

<div class="k-default-codeblock">
```
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9


```
</div>
 625/727 ━━━━━━━━━━━━━━━━━[37m━━━  24s 236ms/step - accuracy: 0.2383 - loss: 2.0220

<div class="k-default-codeblock">
```

```
</div>
 626/727 ━━━━━━━━━━━━━━━━━[37m━━━  23s 236ms/step - accuracy: 0.2383 - loss: 2.0224

<div class="k-default-codeblock">
```

```
</div>
 627/727 ━━━━━━━━━━━━━━━━━[37m━━━  23s 236ms/step - accuracy: 0.2382 - loss: 2.0228

<div class="k-default-codeblock">
```

```
</div>
 628/727 ━━━━━━━━━━━━━━━━━[37m━━━  23s 236ms/step - accuracy: 0.2382 - loss: 2.0232

<div class="k-default-codeblock">
```

```
</div>
 629/727 ━━━━━━━━━━━━━━━━━[37m━━━  23s 236ms/step - accuracy: 0.2382 - loss: 2.0236

<div class="k-default-codeblock">
```

```
</div>
 630/727 ━━━━━━━━━━━━━━━━━[37m━━━  22s 236ms/step - accuracy: 0.2381 - loss: 2.0240

<div class="k-default-codeblock">
```

```
</div>
 631/727 ━━━━━━━━━━━━━━━━━[37m━━━  22s 236ms/step - accuracy: 0.2381 - loss: 2.0243

<div class="k-default-codeblock">
```

```
</div>
 632/727 ━━━━━━━━━━━━━━━━━[37m━━━  22s 236ms/step - accuracy: 0.2380 - loss: 2.0247

<div class="k-default-codeblock">
```

```
</div>
 633/727 ━━━━━━━━━━━━━━━━━[37m━━━  22s 236ms/step - accuracy: 0.2380 - loss: 2.0251

<div class="k-default-codeblock">
```

```
</div>
 634/727 ━━━━━━━━━━━━━━━━━[37m━━━  21s 236ms/step - accuracy: 0.2379 - loss: 2.0254

<div class="k-default-codeblock">
```

```
</div>
 635/727 ━━━━━━━━━━━━━━━━━[37m━━━  21s 236ms/step - accuracy: 0.2379 - loss: 2.0258

<div class="k-default-codeblock">
```

```
</div>
 636/727 ━━━━━━━━━━━━━━━━━[37m━━━  21s 236ms/step - accuracy: 0.2379 - loss: 2.0261

<div class="k-default-codeblock">
```

```
</div>
 637/727 ━━━━━━━━━━━━━━━━━[37m━━━  21s 236ms/step - accuracy: 0.2378 - loss: 2.0265

<div class="k-default-codeblock">
```

```
</div>
 638/727 ━━━━━━━━━━━━━━━━━[37m━━━  21s 236ms/step - accuracy: 0.2378 - loss: 2.0268

<div class="k-default-codeblock">
```

```
</div>
 639/727 ━━━━━━━━━━━━━━━━━[37m━━━  20s 236ms/step - accuracy: 0.2377 - loss: 2.0272

<div class="k-default-codeblock">
```

```
</div>
 640/727 ━━━━━━━━━━━━━━━━━[37m━━━  20s 236ms/step - accuracy: 0.2377 - loss: 2.0275

<div class="k-default-codeblock">
```

```
</div>
 641/727 ━━━━━━━━━━━━━━━━━[37m━━━  20s 236ms/step - accuracy: 0.2376 - loss: 2.0278

<div class="k-default-codeblock">
```
Corrupt JPEG data: 128 extraneous bytes before marker 0xd9


```
</div>
 642/727 ━━━━━━━━━━━━━━━━━[37m━━━  20s 236ms/step - accuracy: 0.2376 - loss: 2.0281

<div class="k-default-codeblock">
```

```
</div>
 643/727 ━━━━━━━━━━━━━━━━━[37m━━━  19s 236ms/step - accuracy: 0.2376 - loss: 2.0284

<div class="k-default-codeblock">
```

```
</div>
 644/727 ━━━━━━━━━━━━━━━━━[37m━━━  19s 236ms/step - accuracy: 0.2375 - loss: 2.0287

<div class="k-default-codeblock">
```

```
</div>
 645/727 ━━━━━━━━━━━━━━━━━[37m━━━  19s 236ms/step - accuracy: 0.2375 - loss: 2.0290

<div class="k-default-codeblock">
```

```
</div>
 646/727 ━━━━━━━━━━━━━━━━━[37m━━━  19s 236ms/step - accuracy: 0.2374 - loss: 2.0293

<div class="k-default-codeblock">
```

```
</div>
 647/727 ━━━━━━━━━━━━━━━━━[37m━━━  18s 236ms/step - accuracy: 0.2374 - loss: 2.0296

<div class="k-default-codeblock">
```

```
</div>
 648/727 ━━━━━━━━━━━━━━━━━[37m━━━  18s 236ms/step - accuracy: 0.2373 - loss: 2.0299

<div class="k-default-codeblock">
```

```
</div>
 649/727 ━━━━━━━━━━━━━━━━━[37m━━━  18s 236ms/step - accuracy: 0.2373 - loss: 2.0301

<div class="k-default-codeblock">
```

```
</div>
 650/727 ━━━━━━━━━━━━━━━━━[37m━━━  18s 236ms/step - accuracy: 0.2372 - loss: 2.0304

<div class="k-default-codeblock">
```

```
</div>
 651/727 ━━━━━━━━━━━━━━━━━[37m━━━  17s 236ms/step - accuracy: 0.2372 - loss: 2.0307

<div class="k-default-codeblock">
```

```
</div>
 652/727 ━━━━━━━━━━━━━━━━━[37m━━━  17s 236ms/step - accuracy: 0.2371 - loss: 2.0309

<div class="k-default-codeblock">
```

```
</div>
 653/727 ━━━━━━━━━━━━━━━━━[37m━━━  17s 236ms/step - accuracy: 0.2371 - loss: 2.0312

<div class="k-default-codeblock">
```

```
</div>
 654/727 ━━━━━━━━━━━━━━━━━[37m━━━  17s 236ms/step - accuracy: 0.2371 - loss: 2.0314

<div class="k-default-codeblock">
```

```
</div>
 655/727 ━━━━━━━━━━━━━━━━━━[37m━━  17s 236ms/step - accuracy: 0.2370 - loss: 2.0317

<div class="k-default-codeblock">
```
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9


```
</div>
 656/727 ━━━━━━━━━━━━━━━━━━[37m━━  16s 236ms/step - accuracy: 0.2370 - loss: 2.0319

<div class="k-default-codeblock">
```

```
</div>
 657/727 ━━━━━━━━━━━━━━━━━━[37m━━  16s 236ms/step - accuracy: 0.2369 - loss: 2.0321

<div class="k-default-codeblock">
```

```
</div>
 658/727 ━━━━━━━━━━━━━━━━━━[37m━━  16s 236ms/step - accuracy: 0.2369 - loss: 2.0324

<div class="k-default-codeblock">
```

```
</div>
 659/727 ━━━━━━━━━━━━━━━━━━[37m━━  16s 236ms/step - accuracy: 0.2368 - loss: 2.0326

<div class="k-default-codeblock">
```

```
</div>
 660/727 ━━━━━━━━━━━━━━━━━━[37m━━  15s 236ms/step - accuracy: 0.2368 - loss: 2.0328

<div class="k-default-codeblock">
```

```
</div>
 661/727 ━━━━━━━━━━━━━━━━━━[37m━━  15s 236ms/step - accuracy: 0.2367 - loss: 2.0330

<div class="k-default-codeblock">
```

```
</div>
 662/727 ━━━━━━━━━━━━━━━━━━[37m━━  15s 236ms/step - accuracy: 0.2367 - loss: 2.0332

<div class="k-default-codeblock">
```

```
</div>
 663/727 ━━━━━━━━━━━━━━━━━━[37m━━  15s 236ms/step - accuracy: 0.2366 - loss: 2.0334

<div class="k-default-codeblock">
```

```
</div>
 664/727 ━━━━━━━━━━━━━━━━━━[37m━━  14s 236ms/step - accuracy: 0.2366 - loss: 2.0336

<div class="k-default-codeblock">
```

```
</div>
 665/727 ━━━━━━━━━━━━━━━━━━[37m━━  14s 236ms/step - accuracy: 0.2365 - loss: 2.0338

<div class="k-default-codeblock">
```

```
</div>
 666/727 ━━━━━━━━━━━━━━━━━━[37m━━  14s 236ms/step - accuracy: 0.2365 - loss: 2.0340

<div class="k-default-codeblock">
```

```
</div>
 667/727 ━━━━━━━━━━━━━━━━━━[37m━━  14s 236ms/step - accuracy: 0.2364 - loss: 2.0341

<div class="k-default-codeblock">
```

```
</div>
 668/727 ━━━━━━━━━━━━━━━━━━[37m━━  13s 236ms/step - accuracy: 0.2364 - loss: 2.0343

<div class="k-default-codeblock">
```

```
</div>
 669/727 ━━━━━━━━━━━━━━━━━━[37m━━  13s 236ms/step - accuracy: 0.2363 - loss: 2.0345

<div class="k-default-codeblock">
```

```
</div>
 670/727 ━━━━━━━━━━━━━━━━━━[37m━━  13s 236ms/step - accuracy: 0.2363 - loss: 2.0346

<div class="k-default-codeblock">
```

```
</div>
 671/727 ━━━━━━━━━━━━━━━━━━[37m━━  13s 236ms/step - accuracy: 0.2362 - loss: 2.0348

<div class="k-default-codeblock">
```

```
</div>
 672/727 ━━━━━━━━━━━━━━━━━━[37m━━  13s 236ms/step - accuracy: 0.2362 - loss: 2.0350

<div class="k-default-codeblock">
```

```
</div>
 673/727 ━━━━━━━━━━━━━━━━━━[37m━━  12s 236ms/step - accuracy: 0.2361 - loss: 2.0351

<div class="k-default-codeblock">
```

```
</div>
 674/727 ━━━━━━━━━━━━━━━━━━[37m━━  12s 236ms/step - accuracy: 0.2361 - loss: 2.0352

<div class="k-default-codeblock">
```

```
</div>
 675/727 ━━━━━━━━━━━━━━━━━━[37m━━  12s 236ms/step - accuracy: 0.2360 - loss: 2.0354

<div class="k-default-codeblock">
```

```
</div>
 676/727 ━━━━━━━━━━━━━━━━━━[37m━━  12s 236ms/step - accuracy: 0.2360 - loss: 2.0355

<div class="k-default-codeblock">
```

```
</div>
 677/727 ━━━━━━━━━━━━━━━━━━[37m━━  11s 236ms/step - accuracy: 0.2359 - loss: 2.0357

<div class="k-default-codeblock">
```

```
</div>
 678/727 ━━━━━━━━━━━━━━━━━━[37m━━  11s 236ms/step - accuracy: 0.2359 - loss: 2.0358

<div class="k-default-codeblock">
```

```
</div>
 679/727 ━━━━━━━━━━━━━━━━━━[37m━━  11s 236ms/step - accuracy: 0.2358 - loss: 2.0359

<div class="k-default-codeblock">
```

```
</div>
 680/727 ━━━━━━━━━━━━━━━━━━[37m━━  11s 236ms/step - accuracy: 0.2358 - loss: 2.0360

<div class="k-default-codeblock">
```

```
</div>
 681/727 ━━━━━━━━━━━━━━━━━━[37m━━  10s 237ms/step - accuracy: 0.2357 - loss: 2.0361

<div class="k-default-codeblock">
```

```
</div>
 682/727 ━━━━━━━━━━━━━━━━━━[37m━━  10s 237ms/step - accuracy: 0.2357 - loss: 2.0363

<div class="k-default-codeblock">
```

```
</div>
 683/727 ━━━━━━━━━━━━━━━━━━[37m━━  10s 237ms/step - accuracy: 0.2356 - loss: 2.0364

<div class="k-default-codeblock">
```

```
</div>
 684/727 ━━━━━━━━━━━━━━━━━━[37m━━  10s 237ms/step - accuracy: 0.2356 - loss: 2.0365

<div class="k-default-codeblock">
```

```
</div>
 685/727 ━━━━━━━━━━━━━━━━━━[37m━━  9s 237ms/step - accuracy: 0.2355 - loss: 2.0366 

<div class="k-default-codeblock">
```

```
</div>
 686/727 ━━━━━━━━━━━━━━━━━━[37m━━  9s 237ms/step - accuracy: 0.2355 - loss: 2.0367

<div class="k-default-codeblock">
```

```
</div>
 687/727 ━━━━━━━━━━━━━━━━━━[37m━━  9s 237ms/step - accuracy: 0.2354 - loss: 2.0367

<div class="k-default-codeblock">
```

```
</div>
 688/727 ━━━━━━━━━━━━━━━━━━[37m━━  9s 237ms/step - accuracy: 0.2354 - loss: 2.0368

<div class="k-default-codeblock">
```

```
</div>
 689/727 ━━━━━━━━━━━━━━━━━━[37m━━  8s 237ms/step - accuracy: 0.2353 - loss: 2.0369

<div class="k-default-codeblock">
```

```
</div>
 690/727 ━━━━━━━━━━━━━━━━━━[37m━━  8s 237ms/step - accuracy: 0.2353 - loss: 2.0370

<div class="k-default-codeblock">
```

```
</div>
 691/727 ━━━━━━━━━━━━━━━━━━━[37m━  8s 237ms/step - accuracy: 0.2352 - loss: 2.0371

<div class="k-default-codeblock">
```
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9


```
</div>
 692/727 ━━━━━━━━━━━━━━━━━━━[37m━  8s 237ms/step - accuracy: 0.2352 - loss: 2.0371

<div class="k-default-codeblock">
```

```
</div>
 693/727 ━━━━━━━━━━━━━━━━━━━[37m━  8s 237ms/step - accuracy: 0.2351 - loss: 2.0372

<div class="k-default-codeblock">
```

```
</div>
 694/727 ━━━━━━━━━━━━━━━━━━━[37m━  7s 237ms/step - accuracy: 0.2351 - loss: 2.0373

<div class="k-default-codeblock">
```

```
</div>
 695/727 ━━━━━━━━━━━━━━━━━━━[37m━  7s 237ms/step - accuracy: 0.2350 - loss: 2.0373

<div class="k-default-codeblock">
```

```
</div>
 696/727 ━━━━━━━━━━━━━━━━━━━[37m━  7s 237ms/step - accuracy: 0.2350 - loss: 2.0374

<div class="k-default-codeblock">
```

```
</div>
 697/727 ━━━━━━━━━━━━━━━━━━━[37m━  7s 237ms/step - accuracy: 0.2349 - loss: 2.0375

<div class="k-default-codeblock">
```

```
</div>
 698/727 ━━━━━━━━━━━━━━━━━━━[37m━  6s 237ms/step - accuracy: 0.2349 - loss: 2.0375

<div class="k-default-codeblock">
```

```
</div>
 699/727 ━━━━━━━━━━━━━━━━━━━[37m━  6s 237ms/step - accuracy: 0.2348 - loss: 2.0376

<div class="k-default-codeblock">
```

```
</div>
 700/727 ━━━━━━━━━━━━━━━━━━━[37m━  6s 237ms/step - accuracy: 0.2348 - loss: 2.0376

<div class="k-default-codeblock">
```
Corrupt JPEG data: 228 extraneous bytes before marker 0xd9


```
</div>
 701/727 ━━━━━━━━━━━━━━━━━━━[37m━  6s 237ms/step - accuracy: 0.2347 - loss: 2.0377

<div class="k-default-codeblock">
```

```
</div>
 702/727 ━━━━━━━━━━━━━━━━━━━[37m━  5s 237ms/step - accuracy: 0.2346 - loss: 2.0377

<div class="k-default-codeblock">
```

```
</div>
 703/727 ━━━━━━━━━━━━━━━━━━━[37m━  5s 237ms/step - accuracy: 0.2346 - loss: 2.0378

<div class="k-default-codeblock">
```

```
</div>
 704/727 ━━━━━━━━━━━━━━━━━━━[37m━  5s 237ms/step - accuracy: 0.2345 - loss: 2.0378

<div class="k-default-codeblock">
```

```
</div>
 705/727 ━━━━━━━━━━━━━━━━━━━[37m━  5s 237ms/step - accuracy: 0.2345 - loss: 2.0378

<div class="k-default-codeblock">
```

```
</div>
 706/727 ━━━━━━━━━━━━━━━━━━━[37m━  4s 237ms/step - accuracy: 0.2344 - loss: 2.0379

<div class="k-default-codeblock">
```

```
</div>
 707/727 ━━━━━━━━━━━━━━━━━━━[37m━  4s 237ms/step - accuracy: 0.2344 - loss: 2.0379

<div class="k-default-codeblock">
```

```
</div>
 708/727 ━━━━━━━━━━━━━━━━━━━[37m━  4s 237ms/step - accuracy: 0.2343 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 709/727 ━━━━━━━━━━━━━━━━━━━[37m━  4s 237ms/step - accuracy: 0.2343 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 710/727 ━━━━━━━━━━━━━━━━━━━[37m━  4s 237ms/step - accuracy: 0.2342 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 711/727 ━━━━━━━━━━━━━━━━━━━[37m━  3s 237ms/step - accuracy: 0.2342 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 712/727 ━━━━━━━━━━━━━━━━━━━[37m━  3s 237ms/step - accuracy: 0.2341 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 713/727 ━━━━━━━━━━━━━━━━━━━[37m━  3s 237ms/step - accuracy: 0.2340 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 714/727 ━━━━━━━━━━━━━━━━━━━[37m━  3s 237ms/step - accuracy: 0.2340 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 715/727 ━━━━━━━━━━━━━━━━━━━[37m━  2s 237ms/step - accuracy: 0.2339 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 716/727 ━━━━━━━━━━━━━━━━━━━[37m━  2s 237ms/step - accuracy: 0.2339 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 717/727 ━━━━━━━━━━━━━━━━━━━[37m━  2s 237ms/step - accuracy: 0.2338 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 718/727 ━━━━━━━━━━━━━━━━━━━[37m━  2s 237ms/step - accuracy: 0.2338 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 719/727 ━━━━━━━━━━━━━━━━━━━[37m━  1s 237ms/step - accuracy: 0.2337 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 720/727 ━━━━━━━━━━━━━━━━━━━[37m━  1s 237ms/step - accuracy: 0.2337 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 721/727 ━━━━━━━━━━━━━━━━━━━[37m━  1s 237ms/step - accuracy: 0.2336 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 722/727 ━━━━━━━━━━━━━━━━━━━[37m━  1s 237ms/step - accuracy: 0.2336 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 723/727 ━━━━━━━━━━━━━━━━━━━[37m━  0s 237ms/step - accuracy: 0.2335 - loss: 2.0381

<div class="k-default-codeblock">
```

```
</div>
 724/727 ━━━━━━━━━━━━━━━━━━━[37m━  0s 237ms/step - accuracy: 0.2334 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 725/727 ━━━━━━━━━━━━━━━━━━━[37m━  0s 237ms/step - accuracy: 0.2334 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 726/727 ━━━━━━━━━━━━━━━━━━━[37m━  0s 237ms/step - accuracy: 0.2333 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 727/727 ━━━━━━━━━━━━━━━━━━━━ 0s 266ms/step - accuracy: 0.2333 - loss: 2.0380

<div class="k-default-codeblock">
```

```
</div>
 727/727 ━━━━━━━━━━━━━━━━━━━━ 216s 266ms/step - accuracy: 0.2332 - loss: 2.0379





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f9008228d90>

```
</div>
Let's look at how our model performs after the fine tuning:


```python
predictions = model.predict(np.expand_dims(image, axis=0))

classes = {0: "cat", 1: "dog"}
print("Top class is:", classes[predictions[0].argmax()])
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step


<div class="k-default-codeblock">
```
Top class is: dog

```
</div>
Awesome - looks like the model correctly classified the image.

---
## Train a Classifier from Scratch

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png)

Now that we've gotten our hands dirty with classification, let's take on one
last task: training a classification model from scratch!
A standard benchmark for image classification is the ImageNet dataset, however
due to licensing constraints we will use the CalTech 101 image classification
dataset in this tutorial.
While we use the simpler CalTech 101 dataset in this guide, the same training
template may be used on ImageNet to achieve near state-of-the-art scores.

Let's start out by tackling data loading:


```python
BATCH_SIZE = 32
NUM_CLASSES = 101
IMAGE_SIZE = (224, 224)

# Change epochs to 100~ to fully train.
EPOCHS = 1


def package_inputs(image, label):
    return {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}


train_ds, eval_ds = tfds.load(
    "caltech101", split=["train", "test"], as_supervised="true"
)
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(BATCH_SIZE * 16)
augmenters = []
```

The CalTech101 dataset has different sizes for every image, so we resize images before
batching them using the
`batch()` API.


```python
resize = keras.layers.Resizing(*IMAGE_SIZE, crop_to_aspect_ratio=True)
train_ds = train_ds.map(resize)
eval_ds = eval_ds.map(resize)

train_ds = train_ds.batch(BATCH_SIZE)
eval_ds = eval_ds.batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
image_batch = batch["images"]
label_batch = batch["labels"]

plot_image_gallery(
    image_batch,
)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_30_0.png)
    


### Data Augmentation

In our previous finetuning example, we performed a static resizing operation and
did not utilize any image augmentation.
This is because a single pass over the training set was sufficient to achieve
decent results.
When training to solve a more difficult task, you'll want to include data
augmentation in your data pipeline.

Data augmentation is a technique to make your model robust to changes in input
data such as lighting, cropping, and orientation.
Keras includes some of the most useful augmentations in the `keras.layers`
API.
Creating an optimal pipeline of augmentations is an art, but in this section of
the guide we'll offer some tips on best practices for classification.

One caveat to be aware of with image data augmentation is that you must be careful
to not shift your augmented data distribution too far from the original data
distribution.
The goal is to prevent overfitting and increase generalization,
but samples that lie completely out of the data distribution simply add noise to
the training process.

The first augmentation we'll use is `RandomFlip`.
This augmentation behaves more or less how you'd expect: it either flips the
image or not.
While this augmentation is useful in CalTech101 and ImageNet, it should be noted
that it should not be used on tasks where the data distribution is not vertical
mirror invariant.
An example of a dataset where this occurs is MNIST hand written digits.
Flipping a `6` over the
vertical axis will make the digit appear more like a `7` than a `6`, but the
label will still show a `6`.


```python
random_flip = keras.layers.RandomFlip()
augmenters += [random_flip]

image_batch = random_flip(image_batch)
plot_image_gallery(image_batch)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_32_0.png)
    


Half of the images have been flipped!

The next augmentation we'll use is `RandomCrop`.
This operation selects a random subset of the image.
By using this augmentation, we force our classifier to become spatially invariant.

Let's add a `RandomCrop` to our set of augmentations:


```python
crop = keras.layers.RandomCrop(
    int(IMAGE_SIZE[0] * 0.9),
    int(IMAGE_SIZE[1] * 0.9),
)

augmenters += [crop]

image_batch = crop(image_batch)
plot_image_gallery(
    image_batch,
)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_34_0.png)
    


We can also rotate images by a random angle using Keras' `RandomRotation` layer. Let's
apply a rotation by a randomly selected angle in the interval -45°...45°:


```python
rotate = keras.layers.RandomRotation((-45 / 360, 45 / 360))

augmenters += [rotate]

image_batch = rotate(image_batch)
plot_image_gallery(image_batch)

resize = keras.layers.Resizing(*IMAGE_SIZE, crop_to_aspect_ratio=True)
augmenters += [resize]

image_batch = resize(image_batch)
plot_image_gallery(image_batch)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_36_0.png)
    



    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_36_1.png)
    


Now let's apply our final augmenter to the training data:


```python

def create_augmenter_fn(augmenters):
    def augmenter_fn(inputs):
        for augmenter in augmenters:
            inputs["images"] = augmenter(inputs["images"])
        return inputs

    return augmenter_fn


augmenter_fn = create_augmenter_fn(augmenters)
train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(iter(train_ds.take(1)))["images"]
plot_image_gallery(
    image_batch,
)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_38_0.png)
    


We also need to resize our evaluation set to get dense batches of the image size
expected by our model. We directly use the deterministic `keras.layers.Resizing` in
this case to avoid adding noise to our evaluation metric due to applying random
augmentations.


```python
inference_resizing = keras.layers.Resizing(*IMAGE_SIZE, crop_to_aspect_ratio=True)


def do_resize(inputs):
    inputs["images"] = inference_resizing(inputs["images"])
    return inputs


eval_ds = eval_ds.map(do_resize, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(iter(eval_ds.take(1)))["images"]
plot_image_gallery(
    image_batch,
)
```


    
![png](/home/admin/keras-io/guides/img/classification_with_keras_hub/classification_with_keras_hub_40_0.png)
    


Finally, lets unpackage our datasets and prepare to pass them to `model.fit()`,
which accepts a tuple of `(images, labels)`.


```python

def unpackage_dict(inputs):
    return inputs["images"], inputs["labels"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
```

Data augmentation is by far the hardest piece of training a modern
classifier.
Congratulations on making it this far!

### Optimizer Tuning

To achieve optimal performance, we need to use a learning rate schedule instead
of a single learning rate. While we won't go into detail on the Cosine decay
with warmup schedule used here, [you can read more about it
here](https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b).


```python

def lr_warmup_cosine_decay(
    global_step,
    warmup_steps,
    hold=0,
    total_steps=0,
    start_lr=0.0,
    target_lr=1e-2,
):
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + ops.cos(
                math.pi
                * ops.convert_to_tensor(
                    global_step - warmup_steps - hold, dtype="float32"
                )
                / ops.convert_to_tensor(
                    total_steps - warmup_steps - hold, dtype="float32"
                )
            )
        )
    )

    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = ops.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = ops.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, hold, start_lr=0.0, target_lr=1e-2):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return ops.where(step > self.total_steps, 0.0, lr)

```

![WarmUpCosineDecay schedule](https://i.imgur.com/YCr5pII.png)

The schedule looks a as we expect.

Next let's construct this optimizer:


```python
total_images = 9000
total_steps = (total_images // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)
hold_steps = int(0.45 * total_steps)
schedule = WarmUpCosineDecay(
    start_lr=0.05,
    target_lr=1e-2,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    hold=hold_steps,
)
optimizer = optimizers.SGD(
    weight_decay=5e-4,
    learning_rate=schedule,
    momentum=0.9,
)
```

At long last, we can now build our model and call `fit()`!
Here, we directly instantiate our `ResNetBackbone`, specifying all architectural
parameters, which gives us full control to tweak the architecture.


```python
backbone = keras_hub.models.ResNetBackbone(
    input_conv_filters=[64],
    input_conv_kernel_sizes=[7],
    stackwise_num_filters=[64, 64, 64],
    stackwise_num_blocks=[2, 2, 2],
    stackwise_num_strides=[1, 2, 2],
    block_type="basic_block",
)
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(101, activation="softmax"),
    ]
)
```

We employ label smoothing to prevent the model from overfitting to artifacts of
our augmentation process.


```python
loss = losses.CategoricalCrossentropy(label_smoothing=0.1)
```

Let's compile our model:


```python
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[
        metrics.CategoricalAccuracy(),
        metrics.TopKCategoricalAccuracy(k=5),
    ],
)
```

and finally call fit().


```python
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=eval_ds,
)
```

    
  1/96 [37m━━━━━━━━━━━━━━━━━━━━  11:17 7s/step - categorical_accuracy: 0.0312 - loss: 12.0736 - top_k_categorical_accuracy: 0.0938

    
  2/96 [37m━━━━━━━━━━━━━━━━━━━━  7s 78ms/step - categorical_accuracy: 0.0234 - loss: 12.4812 - top_k_categorical_accuracy: 0.0703 

<div class="k-default-codeblock">
```

```
</div>
  3/96 [37m━━━━━━━━━━━━━━━━━━━━  7s 75ms/step - categorical_accuracy: 0.0226 - loss: 12.5447 - top_k_categorical_accuracy: 0.0642

<div class="k-default-codeblock">
```

```
</div>
  5/96 ━[37m━━━━━━━━━━━━━━━━━━━  5s 61ms/step - categorical_accuracy: 0.0192 - loss: 12.5319 - top_k_categorical_accuracy: 0.0567

<div class="k-default-codeblock">
```

```
</div>
  6/96 ━[37m━━━━━━━━━━━━━━━━━━━  6s 78ms/step - categorical_accuracy: 0.0177 - loss: 12.5094 - top_k_categorical_accuracy: 0.0533

<div class="k-default-codeblock">
```

```
</div>
  7/96 ━[37m━━━━━━━━━━━━━━━━━━━  7s 80ms/step - categorical_accuracy: 0.0165 - loss: 12.4605 - top_k_categorical_accuracy: 0.0508

<div class="k-default-codeblock">
```

```
</div>
  8/96 ━[37m━━━━━━━━━━━━━━━━━━━  7s 89ms/step - categorical_accuracy: 0.0154 - loss: 12.4164 - top_k_categorical_accuracy: 0.0488

<div class="k-default-codeblock">
```

```
</div>
  9/96 ━[37m━━━━━━━━━━━━━━━━━━━  7s 89ms/step - categorical_accuracy: 0.0144 - loss: 12.3574 - top_k_categorical_accuracy: 0.0469

<div class="k-default-codeblock">
```

```
</div>
 10/96 ━━[37m━━━━━━━━━━━━━━━━━━  8s 96ms/step - categorical_accuracy: 0.0136 - loss: 12.2943 - top_k_categorical_accuracy: 0.0453

<div class="k-default-codeblock">
```

```
</div>
 11/96 ━━[37m━━━━━━━━━━━━━━━━━━  8s 96ms/step - categorical_accuracy: 0.0129 - loss: 12.2254 - top_k_categorical_accuracy: 0.0440

<div class="k-default-codeblock">
```

```
</div>
 12/96 ━━[37m━━━━━━━━━━━━━━━━━━  8s 100ms/step - categorical_accuracy: 0.0123 - loss: 12.1517 - top_k_categorical_accuracy: 0.0432

<div class="k-default-codeblock">
```

```
</div>
 13/96 ━━[37m━━━━━━━━━━━━━━━━━━  8s 102ms/step - categorical_accuracy: 0.0119 - loss: 12.0706 - top_k_categorical_accuracy: 0.0428

<div class="k-default-codeblock">
```

```
</div>
 14/96 ━━[37m━━━━━━━━━━━━━━━━━━  8s 102ms/step - categorical_accuracy: 0.0117 - loss: 11.9796 - top_k_categorical_accuracy: 0.0426

<div class="k-default-codeblock">
```

```
</div>
 15/96 ━━━[37m━━━━━━━━━━━━━━━━━  8s 104ms/step - categorical_accuracy: 0.0114 - loss: 11.8843 - top_k_categorical_accuracy: 0.0427

<div class="k-default-codeblock">
```

```
</div>
 16/96 ━━━[37m━━━━━━━━━━━━━━━━━  8s 104ms/step - categorical_accuracy: 0.0112 - loss: 11.7882 - top_k_categorical_accuracy: 0.0428

<div class="k-default-codeblock">
```

```
</div>
 17/96 ━━━[37m━━━━━━━━━━━━━━━━━  8s 106ms/step - categorical_accuracy: 0.0110 - loss: 11.6938 - top_k_categorical_accuracy: 0.0430

<div class="k-default-codeblock">
```

```
</div>
 18/96 ━━━[37m━━━━━━━━━━━━━━━━━  8s 106ms/step - categorical_accuracy: 0.0108 - loss: 11.5967 - top_k_categorical_accuracy: 0.0431

<div class="k-default-codeblock">
```

```
</div>
 19/96 ━━━[37m━━━━━━━━━━━━━━━━━  8s 107ms/step - categorical_accuracy: 0.0105 - loss: 11.4991 - top_k_categorical_accuracy: 0.0431

<div class="k-default-codeblock">
```

```
</div>
 20/96 ━━━━[37m━━━━━━━━━━━━━━━━  8s 107ms/step - categorical_accuracy: 0.0103 - loss: 11.4042 - top_k_categorical_accuracy: 0.0431

<div class="k-default-codeblock">
```

```
</div>
 21/96 ━━━━[37m━━━━━━━━━━━━━━━━  8s 108ms/step - categorical_accuracy: 0.0102 - loss: 11.3113 - top_k_categorical_accuracy: 0.0430

<div class="k-default-codeblock">
```

```
</div>
 22/96 ━━━━[37m━━━━━━━━━━━━━━━━  8s 108ms/step - categorical_accuracy: 0.0100 - loss: 11.2201 - top_k_categorical_accuracy: 0.0429

<div class="k-default-codeblock">
```

```
</div>
 23/96 ━━━━[37m━━━━━━━━━━━━━━━━  7s 109ms/step - categorical_accuracy: 0.0099 - loss: 11.1308 - top_k_categorical_accuracy: 0.0428

<div class="k-default-codeblock">
```

```
</div>
 24/96 ━━━━━[37m━━━━━━━━━━━━━━━  7s 109ms/step - categorical_accuracy: 0.0098 - loss: 11.0438 - top_k_categorical_accuracy: 0.0426

<div class="k-default-codeblock">
```

```
</div>
 25/96 ━━━━━[37m━━━━━━━━━━━━━━━  7s 110ms/step - categorical_accuracy: 0.0096 - loss: 10.9591 - top_k_categorical_accuracy: 0.0424

<div class="k-default-codeblock">
```

```
</div>
 26/96 ━━━━━[37m━━━━━━━━━━━━━━━  7s 110ms/step - categorical_accuracy: 0.0095 - loss: 10.8774 - top_k_categorical_accuracy: 0.0422

<div class="k-default-codeblock">
```

```
</div>
 27/96 ━━━━━[37m━━━━━━━━━━━━━━━  7s 110ms/step - categorical_accuracy: 0.0095 - loss: 10.7983 - top_k_categorical_accuracy: 0.0422

<div class="k-default-codeblock">
```

```
</div>
 28/96 ━━━━━[37m━━━━━━━━━━━━━━━  7s 111ms/step - categorical_accuracy: 0.0094 - loss: 10.7217 - top_k_categorical_accuracy: 0.0420

<div class="k-default-codeblock">
```

```
</div>
 29/96 ━━━━━━[37m━━━━━━━━━━━━━━  7s 111ms/step - categorical_accuracy: 0.0094 - loss: 10.6470 - top_k_categorical_accuracy: 0.0419

<div class="k-default-codeblock">
```

```
</div>
 30/96 ━━━━━━[37m━━━━━━━━━━━━━━  7s 111ms/step - categorical_accuracy: 0.0093 - loss: 10.5740 - top_k_categorical_accuracy: 0.0419

<div class="k-default-codeblock">
```

```
</div>
 31/96 ━━━━━━[37m━━━━━━━━━━━━━━  7s 112ms/step - categorical_accuracy: 0.0092 - loss: 10.5030 - top_k_categorical_accuracy: 0.0418

<div class="k-default-codeblock">
```

```
</div>
 32/96 ━━━━━━[37m━━━━━━━━━━━━━━  7s 111ms/step - categorical_accuracy: 0.0092 - loss: 10.4349 - top_k_categorical_accuracy: 0.0417

<div class="k-default-codeblock">
```

```
</div>
 33/96 ━━━━━━[37m━━━━━━━━━━━━━━  7s 112ms/step - categorical_accuracy: 0.0091 - loss: 10.3685 - top_k_categorical_accuracy: 0.0417

<div class="k-default-codeblock">
```

```
</div>
 34/96 ━━━━━━━[37m━━━━━━━━━━━━━  6s 112ms/step - categorical_accuracy: 0.0091 - loss: 10.3043 - top_k_categorical_accuracy: 0.0417

<div class="k-default-codeblock">
```

```
</div>
 35/96 ━━━━━━━[37m━━━━━━━━━━━━━  6s 113ms/step - categorical_accuracy: 0.0091 - loss: 10.2418 - top_k_categorical_accuracy: 0.0416

<div class="k-default-codeblock">
```

```
</div>
 36/96 ━━━━━━━[37m━━━━━━━━━━━━━  6s 113ms/step - categorical_accuracy: 0.0090 - loss: 10.1803 - top_k_categorical_accuracy: 0.0416

<div class="k-default-codeblock">
```

```
</div>
 37/96 ━━━━━━━[37m━━━━━━━━━━━━━  6s 113ms/step - categorical_accuracy: 0.0090 - loss: 10.1200 - top_k_categorical_accuracy: 0.0415

<div class="k-default-codeblock">
```

```
</div>
 38/96 ━━━━━━━[37m━━━━━━━━━━━━━  6s 113ms/step - categorical_accuracy: 0.0090 - loss: 10.0610 - top_k_categorical_accuracy: 0.0414

<div class="k-default-codeblock">
```

```
</div>
 39/96 ━━━━━━━━[37m━━━━━━━━━━━━  6s 114ms/step - categorical_accuracy: 0.0089 - loss: 10.0037 - top_k_categorical_accuracy: 0.0413

<div class="k-default-codeblock">
```

```
</div>
 40/96 ━━━━━━━━[37m━━━━━━━━━━━━  6s 113ms/step - categorical_accuracy: 0.0089 - loss: 9.9472 - top_k_categorical_accuracy: 0.0413 

<div class="k-default-codeblock">
```

```
</div>
 41/96 ━━━━━━━━[37m━━━━━━━━━━━━  6s 114ms/step - categorical_accuracy: 0.0088 - loss: 9.8921 - top_k_categorical_accuracy: 0.0412

<div class="k-default-codeblock">
```

```
</div>
 42/96 ━━━━━━━━[37m━━━━━━━━━━━━  6s 114ms/step - categorical_accuracy: 0.0088 - loss: 9.8384 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 43/96 ━━━━━━━━[37m━━━━━━━━━━━━  6s 114ms/step - categorical_accuracy: 0.0087 - loss: 9.7862 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 44/96 ━━━━━━━━━[37m━━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0087 - loss: 9.7348 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 45/96 ━━━━━━━━━[37m━━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.6845 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 46/96 ━━━━━━━━━[37m━━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.6354 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 47/96 ━━━━━━━━━[37m━━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.5873 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 48/96 ━━━━━━━━━━[37m━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.5405 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 49/96 ━━━━━━━━━━[37m━━━━━━━━━━  5s 115ms/step - categorical_accuracy: 0.0085 - loss: 9.4946 - top_k_categorical_accuracy: 0.0411

<div class="k-default-codeblock">
```

```
</div>
 50/96 ━━━━━━━━━━[37m━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.4499 - top_k_categorical_accuracy: 0.0412

<div class="k-default-codeblock">
```

```
</div>
 51/96 ━━━━━━━━━━[37m━━━━━━━━━━  5s 115ms/step - categorical_accuracy: 0.0086 - loss: 9.4065 - top_k_categorical_accuracy: 0.0412

<div class="k-default-codeblock">
```

```
</div>
 52/96 ━━━━━━━━━━[37m━━━━━━━━━━  5s 114ms/step - categorical_accuracy: 0.0086 - loss: 9.3643 - top_k_categorical_accuracy: 0.0413

<div class="k-default-codeblock">
```

```
</div>
 53/96 ━━━━━━━━━━━[37m━━━━━━━━━  4s 115ms/step - categorical_accuracy: 0.0086 - loss: 9.3230 - top_k_categorical_accuracy: 0.0414

<div class="k-default-codeblock">
```

```
</div>
 54/96 ━━━━━━━━━━━[37m━━━━━━━━━  4s 115ms/step - categorical_accuracy: 0.0086 - loss: 9.2827 - top_k_categorical_accuracy: 0.0415

<div class="k-default-codeblock">
```

```
</div>
 55/96 ━━━━━━━━━━━[37m━━━━━━━━━  4s 116ms/step - categorical_accuracy: 0.0087 - loss: 9.2431 - top_k_categorical_accuracy: 0.0416

<div class="k-default-codeblock">
```

```
</div>
 56/96 ━━━━━━━━━━━[37m━━━━━━━━━  4s 115ms/step - categorical_accuracy: 0.0087 - loss: 9.2046 - top_k_categorical_accuracy: 0.0417

<div class="k-default-codeblock">
```

```
</div>
 57/96 ━━━━━━━━━━━[37m━━━━━━━━━  4s 116ms/step - categorical_accuracy: 0.0087 - loss: 9.1671 - top_k_categorical_accuracy: 0.0418

<div class="k-default-codeblock">
```

```
</div>
 58/96 ━━━━━━━━━━━━[37m━━━━━━━━  4s 115ms/step - categorical_accuracy: 0.0087 - loss: 9.1304 - top_k_categorical_accuracy: 0.0418

<div class="k-default-codeblock">
```

```
</div>
 59/96 ━━━━━━━━━━━━[37m━━━━━━━━  4s 116ms/step - categorical_accuracy: 0.0087 - loss: 9.0942 - top_k_categorical_accuracy: 0.0419

<div class="k-default-codeblock">
```

```
</div>
 60/96 ━━━━━━━━━━━━[37m━━━━━━━━  4s 115ms/step - categorical_accuracy: 0.0087 - loss: 9.0588 - top_k_categorical_accuracy: 0.0420

<div class="k-default-codeblock">
```

```
</div>
 61/96 ━━━━━━━━━━━━[37m━━━━━━━━  4s 116ms/step - categorical_accuracy: 0.0087 - loss: 9.0241 - top_k_categorical_accuracy: 0.0421

<div class="k-default-codeblock">
```

```
</div>
 62/96 ━━━━━━━━━━━━[37m━━━━━━━━  3s 115ms/step - categorical_accuracy: 0.0088 - loss: 8.9903 - top_k_categorical_accuracy: 0.0422

<div class="k-default-codeblock">
```

```
</div>
 63/96 ━━━━━━━━━━━━━[37m━━━━━━━  3s 116ms/step - categorical_accuracy: 0.0088 - loss: 8.9569 - top_k_categorical_accuracy: 0.0423

<div class="k-default-codeblock">
```

```
</div>
 64/96 ━━━━━━━━━━━━━[37m━━━━━━━  3s 116ms/step - categorical_accuracy: 0.0088 - loss: 8.9245 - top_k_categorical_accuracy: 0.0424

<div class="k-default-codeblock">
```

```
</div>
 65/96 ━━━━━━━━━━━━━[37m━━━━━━━  3s 117ms/step - categorical_accuracy: 0.0088 - loss: 8.8929 - top_k_categorical_accuracy: 0.0424

<div class="k-default-codeblock">
```

```
</div>
 66/96 ━━━━━━━━━━━━━[37m━━━━━━━  3s 116ms/step - categorical_accuracy: 0.0088 - loss: 8.8621 - top_k_categorical_accuracy: 0.0425

<div class="k-default-codeblock">
```

```
</div>
 67/96 ━━━━━━━━━━━━━[37m━━━━━━━  3s 117ms/step - categorical_accuracy: 0.0088 - loss: 8.8319 - top_k_categorical_accuracy: 0.0426

<div class="k-default-codeblock">
```

```
</div>
 68/96 ━━━━━━━━━━━━━━[37m━━━━━━  3s 116ms/step - categorical_accuracy: 0.0088 - loss: 8.8023 - top_k_categorical_accuracy: 0.0427

<div class="k-default-codeblock">
```

```
</div>
 69/96 ━━━━━━━━━━━━━━[37m━━━━━━  3s 117ms/step - categorical_accuracy: 0.0089 - loss: 8.7733 - top_k_categorical_accuracy: 0.0428

<div class="k-default-codeblock">
```

```
</div>
 70/96 ━━━━━━━━━━━━━━[37m━━━━━━  3s 116ms/step - categorical_accuracy: 0.0089 - loss: 8.7447 - top_k_categorical_accuracy: 0.0429

<div class="k-default-codeblock">
```

```
</div>
 71/96 ━━━━━━━━━━━━━━[37m━━━━━━  2s 117ms/step - categorical_accuracy: 0.0089 - loss: 8.7165 - top_k_categorical_accuracy: 0.0430

<div class="k-default-codeblock">
```

```
</div>
 73/96 ━━━━━━━━━━━━━━━[37m━━━━━  2s 117ms/step - categorical_accuracy: 0.0090 - loss: 8.6615 - top_k_categorical_accuracy: 0.0432

<div class="k-default-codeblock">
```

```
</div>
 75/96 ━━━━━━━━━━━━━━━[37m━━━━━  2s 117ms/step - categorical_accuracy: 0.0090 - loss: 8.6083 - top_k_categorical_accuracy: 0.0434

<div class="k-default-codeblock">
```

```
</div>
 77/96 ━━━━━━━━━━━━━━━━[37m━━━━  2s 118ms/step - categorical_accuracy: 0.0091 - loss: 8.5566 - top_k_categorical_accuracy: 0.0437

<div class="k-default-codeblock">
```

```
</div>
 78/96 ━━━━━━━━━━━━━━━━[37m━━━━  2s 117ms/step - categorical_accuracy: 0.0091 - loss: 8.5314 - top_k_categorical_accuracy: 0.0437

<div class="k-default-codeblock">
```

```
</div>
 79/96 ━━━━━━━━━━━━━━━━[37m━━━━  1s 117ms/step - categorical_accuracy: 0.0091 - loss: 8.5068 - top_k_categorical_accuracy: 0.0438

<div class="k-default-codeblock">
```

```
</div>
 80/96 ━━━━━━━━━━━━━━━━[37m━━━━  1s 117ms/step - categorical_accuracy: 0.0092 - loss: 8.4825 - top_k_categorical_accuracy: 0.0439

<div class="k-default-codeblock">
```

```
</div>
 81/96 ━━━━━━━━━━━━━━━━[37m━━━━  1s 118ms/step - categorical_accuracy: 0.0092 - loss: 8.4586 - top_k_categorical_accuracy: 0.0441

<div class="k-default-codeblock">
```

```
</div>
 83/96 ━━━━━━━━━━━━━━━━━[37m━━━  1s 118ms/step - categorical_accuracy: 0.0093 - loss: 8.4117 - top_k_categorical_accuracy: 0.0444

<div class="k-default-codeblock">
```

```
</div>
 84/96 ━━━━━━━━━━━━━━━━━[37m━━━  1s 117ms/step - categorical_accuracy: 0.0093 - loss: 8.3887 - top_k_categorical_accuracy: 0.0444

<div class="k-default-codeblock">
```

```
</div>
 85/96 ━━━━━━━━━━━━━━━━━[37m━━━  1s 118ms/step - categorical_accuracy: 0.0093 - loss: 8.3661 - top_k_categorical_accuracy: 0.0445

<div class="k-default-codeblock">
```

```
</div>
 86/96 ━━━━━━━━━━━━━━━━━[37m━━━  1s 117ms/step - categorical_accuracy: 0.0094 - loss: 8.3440 - top_k_categorical_accuracy: 0.0446

<div class="k-default-codeblock">
```

```
</div>
 87/96 ━━━━━━━━━━━━━━━━━━[37m━━  1s 118ms/step - categorical_accuracy: 0.0094 - loss: 8.3223 - top_k_categorical_accuracy: 0.0447

<div class="k-default-codeblock">
```

```
</div>
 89/96 ━━━━━━━━━━━━━━━━━━[37m━━  0s 118ms/step - categorical_accuracy: 0.0095 - loss: 8.2596 - top_k_categorical_accuracy: 0.0450

<div class="k-default-codeblock">
```

```
</div>
 90/96 ━━━━━━━━━━━━━━━━━━[37m━━  0s 117ms/step - categorical_accuracy: 0.0095 - loss: 8.2596 - top_k_categorical_accuracy: 0.0450

<div class="k-default-codeblock">
```

```
</div>
 91/96 ━━━━━━━━━━━━━━━━━━[37m━━  0s 118ms/step - categorical_accuracy: 0.0095 - loss: 8.2393 - top_k_categorical_accuracy: 0.0451

<div class="k-default-codeblock">
```

```
</div>
 92/96 ━━━━━━━━━━━━━━━━━━━[37m━  0s 117ms/step - categorical_accuracy: 0.0096 - loss: 8.2193 - top_k_categorical_accuracy: 0.0451

<div class="k-default-codeblock">
```

```
</div>
 93/96 ━━━━━━━━━━━━━━━━━━━[37m━  0s 118ms/step - categorical_accuracy: 0.0096 - loss: 8.1997 - top_k_categorical_accuracy: 0.0452

<div class="k-default-codeblock">
```

```
</div>
 95/96 ━━━━━━━━━━━━━━━━━━━[37m━  0s 120ms/step - categorical_accuracy: 0.0096 - loss: 8.1615 - top_k_categorical_accuracy: 0.0453

<div class="k-default-codeblock">
```

```
</div>
 96/96 ━━━━━━━━━━━━━━━━━━━━ 0s 188ms/step - categorical_accuracy: 0.0097 - loss: 8.1428 - top_k_categorical_accuracy: 0.0454

<div class="k-default-codeblock">
```

```
</div>
 96/96 ━━━━━━━━━━━━━━━━━━━━ 32s 264ms/step - categorical_accuracy: 0.0097 - loss: 8.1245 - top_k_categorical_accuracy: 0.0454 - val_categorical_accuracy: 0.0748 - val_loss: 5.5626 - val_top_k_categorical_accuracy: 0.1351





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f8ce03c49a0>

```
</div>
Congratulations!  You now know how to train a powerful image classifier from
scratch using KerasHub.
Depending on the availability of labeled data for your application, training
from scratch may or may not be more powerful than using transfer learning in
addition to the data augmentations discussed above. For smaller datasets,
pretrained models generally produce high accuracy and faster convergence.

---
## Conclusions

While image classification is perhaps the simplest problem in computer vision,
the modern landscape has numerous complex components.
Luckily, KerasHub offers robust, production-grade APIs to make assembling most
of these components possible in one line of code.
Through the use of KerasHub's `ImageClassifier` API, pretrained weights, and
Keras' data augmentations you can assemble everything you need to train a
powerful classifier in a few hundred lines of code!

As a follow up exercise, try fine tuning a KerasHub classifier on your own dataset!
