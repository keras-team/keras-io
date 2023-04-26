# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Title: Classification with KerasCV
Author: [lukewood](https://lukewood.xyz)
Date created: 03/28/2023
Last modified: 03/28/2023
Description: Use KerasCV to train a state of the art image classifier.
"""

"""
This guide demonstrates KerasCV's modular approach to solving image
classification problems at two levels of complexity:

- Inference with a pretrained classifier
- Fine-tuning a pretrained backbone

We use Professor Keras, the official Keras mascot, as a
visual reference for the complexity of the material:

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_evolution.png)

Please note that due to classification being a pretty simple use case,
this guide only covers beginner and intermediate workflows.
Advanced and expert workflows may be found in [other KerasCV guides](https://keras.io/guides/keras_cv/)!
"""

"""shell
!pip install -q --upgrade git+https://github.com/keras-team/keras-cv.git tensorflow
"""

import json
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np

"""
## Inference with a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png)

Let's get started with the simples KerasCV API: a pretrained classifier.
In this example, we will build a Dogs vs Cats classifier using a model that was
pretrained on the ImageNet dataset.

The highest level module in KerasCV is a *task*. A *task* is a `keras.Model`
consisting of a (generally pretrained) backbone model and task-specific layers.
Here's an example using `keras_cv.models.ImageClassifier` with a EfficientNetV2S
Backbone.
"""

classifier = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_s_imagenet_classifier",
)

"""
You may notice a small deviation from the old `keras.applications` API; where
you would construct the class with `EfficientNetV2S(weights="imagenet")`.
While the old API was great for classification, it did not scale effectively to
other use cases that required complex architectures, like object deteciton and
semantic segmentation.

Now that we have a classifier build, lets take our model for a spin!
Let's run inference on a picture of  a cute cat:
"""
filepath = tf.keras.utils.get_file(origin="https://i.imgur.com/9i63gLN.jpg")
image = keras.utils.load_img(filepath)
image = np.array(image)
keras_cv.visualization.plot_image_gallery(
    [image],
    rows=1,
    cols=1,
    value_range=(0, 255),
    show=True,
)

"""
Lets also fetch the class mapping for ImageNet.  I have this class mapping
hosted in a GitHub gist.
"""
import json
class_mapping = keras.utils.get_file(origin="https://gist.githubusercontent.com/LukeWood/368e2e89bb0e36bd34ff7043e0247289/raw/0615d1e88a93d4e971bf2dea0cfc52f30a12dd99/imagenet%2520mapping")
class_mapping = json.load(open(class_mapping, 'r'))\\

"""
Let's get some predictions from our classifier:
"""

predictions = classifier.predict([image])
"""
Predictions come in the form of softmax-ed category rankings.
We can find the index of the top classes using a simple argsort function:
"""
top_classes = predictions[0].argsort(axis=-1)

"""
In order to decode the class mappings, we can construct a mapping from
category indices to ImageNet class names.
For conveneince, I've stored the ImageNet class mapping in a GitHub gist.
Let's download and load it now.
"""
classes = keras.utils.get_file(
    origin="https://gist.githubusercontent.com/LukeWood/62eebcd5c5c4a4d0e0b7845780f76d55/raw/fde63e5e4c09e2fa0a3436680f436bdcb8325aac/ImagenetClassnames.json"
)
with open(classes, 'rb') as f:
    classes = json.load(f)
"""
Now we can simply look up the class names via index:
"""
top_two = [classes[i] for i in top_classes[-2:]]
print("Top two classes are:", top_two)

"""
Great!  Both of these appear to be correct!
But what if you don't care about the
velvet blanket?
Perhaps instead, you only want to know if a cat is in the image or not.
This can be solved using fine tuning your own classifier.

# Fine tuning a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png)

When labeled images specific to our task are available, fine-tuning a custom
classifier can improve performance. If we want to train a Cats vs Dogs
Classifier, using explicitly labeled Cat vs Dog data should perform better than
the generic classifier data! And for many tasks, no relevant pretrained model
will be available (e.g., categorizing images specific to your application).

The biggest difficulty when fine-tuning a KerasCV model is loading and augmenting
your data.  Luckily, we've handled the second half for you, so all you'll have
to do is load your own data.

First, let's setup our data pipeline:
"""

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()

data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
train_dataset = data['train']

IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes

random_crop = keras_cv.layers.Resizing(224, 224, crop_to_aspect_ratio=True)

def package_dict(image, label):
    image = tf.cast(image, tf.float32)
    image = random_crop(image)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


train_dataset = train_dataset.shuffle(10 * BATCH_SIZE).map(package_dict, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

images = next(iter(train_dataset.take(1)))['images']
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))

"""
Next, lets assemble a `keras_cv` augmentation pipeline.
In this guide, we use the standard pipeline
[CutMix, MixUp, and RandAugment](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
augmentation pipeline.  More information on the behavior of these augmentations
may be found in their
[corresponding Keras.io guide](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/).
"""

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
        keras_cv.layers.MixUp()
    ]
)

train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

images = next(iter(train_dataset.take(1)))['images']
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))

"""
Next let's construct our model:
"""

backbone = keras_cv.models.DenseNet121(
    include_rescaling=True,
    include_top=False,
    num_classes=2,
    pooling='max',
    weights="imagenet/classification"
)
model = keras.Sequential(
    [backbone, keras.layers.Dense(2, activation='softmax')]
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['accuracy'],
)

"""
All that is left to do is construct a standard Keras `model.fit()` loop!
"""

def unpackage_data(inputs):
  return inputs['images'], inputs['labels']

train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

model.fit(train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE))

"""
Let's look at how our model performs after the fine tuning!
"""

predictions = model.predict([image])

classes = {
    0: 'cat',
    1: 'dog'
}
print("Top class is:", classes[predictions[0].argmax()])

"""
Awesome!  Looks like the model correctly classified the image.
"""

"""
## Conclusions

KerasCV makes image classification easy.
Making use of the KerasCV `ImageClassifier` API, pretrained weights, and the
KerasCV data augmentations allows you to train a powerful classifier in `<50`
lines of code.

As a follow up exercise, give the following a try:

- Fine tune a KerasCV classifier on your own dataset
- Learn more about [KerasCV's data augmentations](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
- Check out how we train our models on [ImageNet](https://github.com/keras-team/keras-cv/blob/master/examples/training/classification/imagenet/basic_training.py)
"""
