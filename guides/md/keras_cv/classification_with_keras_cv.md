# Classification with KerasCV

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 03/28/2023<br>
**Last modified:** 03/28/2023<br>
**Description:** Use KerasCV to train powerful image classifiers.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/classification_with_keras_cv.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/classification_with_keras_cv.py)



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


```python
!!pip install -q --upgrade git+https://github.com/keras-team/keras-cv.git tensorflow
```




```python
import json
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np
```
<div class="k-default-codeblock">
```
[]

```
</div>
---
## Inference with a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png)

Let's get started with the simples KerasCV API: a pretrained classifier.
In this example, we will build a Dogs vs Cats classifier using a model that was
pretrained on the ImageNet dataset.

The highest level module in KerasCV is a *task*. A *task* is a `keras.Model`
consisting of a (generally pretrained) backbone model and task-specific layers.
Here's an example using `keras_cv.models.ImageClassifier` with a EfficientNetV2S
Backbone.


```python
classifier = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2-s_imagenet_classifier"
)
```

You may notice a small deviation from the old `keras.applications` API; where
you would construct the class with `EfficientNetV2S(weights="imagenet")`.
While the old API was great for classification, it did not scale effectively to
other use cases that required complex architectures, like object deteciton and
semantic segmentation.

Now that we have a classifier build, lets take our model for a spin!
Let's run inference on a picture of  a cute cat:


```python
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
```


    
![png](/img/guides/classification_with_keras_cv/classification_with_keras_cv_7_0.png)
    


Lets also fetch the class mapping for ImageNet.  I have this class mapping
hosted in a GitHub gist.


```python
import json

class_mapping = keras.utils.get_file(
    origin="https://gist.githubusercontent.com/LukeWood/368e2e89bb0e36bd34ff7043e0247289/raw/0615d1e88a93d4e971bf2dea0cfc52f30a12dd99/imagenet%2520mapping"
)
class_mapping = json.load(open(class_mapping, "r"))
```

Let's get some predictions from our classifier:


```python
predictions = classifier.predict(np.expand_dims(image, axis=0))
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 5s 5s/step

```
</div>
Predictions come in the form of softmax-ed category rankings.
We can find the index of the top classes using a simple argsort function:


```python
top_classes = predictions[0].argsort(axis=-1)
```

In order to decode the class mappings, we can construct a mapping from
category indices to ImageNet class names.
For conveneince, I've stored the ImageNet class mapping in a GitHub gist.
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
Top two classes are: ['fur coat', 'Egyptian cat']

```
</div>
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


```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()

data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
train_dataset = data["train"]

IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes

random_crop = keras_cv.layers.Resizing(224, 224, crop_to_aspect_ratio=True)


def package_dict(image, label):
    image = tf.cast(image, tf.float32)
    image = random_crop(image)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


train_dataset = train_dataset.shuffle(10 * BATCH_SIZE).map(
    package_dict, num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.batch(BATCH_SIZE)

images = next(iter(train_dataset.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))
```


    
![png](/img/guides/classification_with_keras_cv/classification_with_keras_cv_19_0.png)
    


Next, lets assemble a `keras_cv` augmentation pipeline.
In this guide, we use the standard pipeline
[CutMix, MixUp, and RandAugment](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
augmentation pipeline.  More information on the behavior of these augmentations
may be found in their
[corresponding Keras.io guide](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/).


```python
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
        keras_cv.layers.MixUp(),
    ]
)

train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

images = next(iter(train_dataset.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor. Received: inputs={'images': <tf.Tensor 'args_0:0' shape=(None, 224, 224, 3) dtype=float32>, 'labels': <tf.Tensor 'args_1:0' shape=(None, 2) dtype=float32>}. Consider rewriting this model with the Functional API.

WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor. Received: inputs={'images': <tf.Tensor 'args_0:0' shape=(None, 224, 224, 3) dtype=float32>, 'labels': <tf.Tensor 'args_1:0' shape=(None, 2) dtype=float32>}. Consider rewriting this model with the Functional API.

```
</div>
    
![png](/img/guides/classification_with_keras_cv/classification_with_keras_cv_21_2.png)
    


Next let's construct our model:


```python
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2-s_imagenet",
)
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(2, activation="softmax"),
    ]
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)
```

All that is left to do is construct a standard Keras `model.fit()` loop!


```python

def unpackage_data(inputs):
    return inputs["images"], inputs["labels"]


train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

model.fit(train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE))
```

<div class="k-default-codeblock">
```
200/727 [=======>......................] - ETA: 1:11 - loss: 2.6703 - accuracy: 0.6687

Corrupt JPEG data: 99 extraneous bytes before marker 0xd9

237/727 [========>.....................] - ETA: 1:06 - loss: 2.3729 - accuracy: 0.6791

Warning: unknown JFIF revision number 0.00

249/727 [=========>....................] - ETA: 1:04 - loss: 2.2892 - accuracy: 0.6831

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

293/727 [===========>..................] - ETA: 58s - loss: 2.0561 - accuracy: 0.6918

Corrupt JPEG data: 162 extraneous bytes before marker 0xd9

341/727 [=============>................] - ETA: 52s - loss: 1.8625 - accuracy: 0.6990

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

357/727 [=============>................] - ETA: 50s - loss: 1.8078 - accuracy: 0.7008

Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

517/727 [====================>.........] - ETA: 28s - loss: 1.4442 - accuracy: 0.7140

Corrupt JPEG data: 214 extraneous bytes before marker 0xd9

620/727 [========================>.....] - ETA: 14s - loss: 1.2989 - accuracy: 0.7242

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

636/727 [=========================>....] - ETA: 12s - loss: 1.2799 - accuracy: 0.7255

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

651/727 [=========================>....] - ETA: 10s - loss: 1.2626 - accuracy: 0.7269

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

687/727 [===========================>..] - ETA: 5s - loss: 1.2253 - accuracy: 0.7298

Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9

695/727 [===========================>..] - ETA: 4s - loss: 1.2171 - accuracy: 0.7306

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

727/727 [==============================] - 144s 137ms/step - loss: 1.1887 - accuracy: 0.7321

<keras.callbacks.History at 0x7f84942384c0>

```
</div>
Let's look at how our model performs after the fine tuning!


```python
predictions = model.predict(np.expand_dims(image, axis=0))

classes = {0: "cat", 1: "dog"}
print("Top class is:", classes[predictions[0].argmax()])
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 3s 3s/step
Top class is: cat

```
</div>
Awesome!  Looks like the model correctly classified the image.

---
## Conclusions

KerasCV makes image classification easy.
Making use of the KerasCV `ImageClassifier` API, pretrained weights, and the
KerasCV data augmentations allows you to train a powerful classifier in `<50`
lines of code.

As a follow up exercise, give the following a try:

- Fine tune a KerasCV classifier on your own dataset
- Learn more about [KerasCV's data augmentations](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
- Check out how we train our models on [ImageNet](https://github.com/keras-team/keras-cv/blob/master/examples/training/classification/imagenet/basic_training.py)
