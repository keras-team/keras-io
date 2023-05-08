"""
Title: Classification with KerasCV
Author: [lukewood](https://lukewood.xyz)
Date created: 03/28/2023
Last modified: 03/28/2023
Description: Use KerasCV to train powerful image classifiers.
Accelerator: GPU
"""

"""
Classification is the process of predicting a categorical label for a given
input image.
While classification is a relatively straightforward computer vision task,
modern approaches still are built of several complex components.
Luckily, KerasCV provides APIs to construct commonly used components.

This guide demonstrates KerasCV's modular approach to solving image
classification problems at three levels of complexity:

- Inference with a pretrained classifier
- Fine-tuning a pretrained backbone
- Training a image classifier from scratch

We use Professor Keras, the official Keras mascot, as a
visual reference for the complexity of the material:

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_evolution.png)
"""


"""shell
!pip install -q --upgrade git+https://github.com/keras-team/keras-cv.git tensorflow
"""

import json
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import losses
import numpy as np
from keras import metrics


"""
## Inference with a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png)

Let's get started with the simples KerasCV API: a pretrained classifier.
In this example, we will construct a classifier that was
pretrained on the ImageNet dataset.
We'll use this model to solve the age old "Cat or Dog" problem.

The highest level module in KerasCV is a *task*. A *task* is a `keras.Model`
consisting of a (generally pretrained) backbone model and task-specific layers.
Here's an example using `keras_cv.models.ImageClassifier` with an
EfficientNetV2B0 Backbone.

EfficientNetV2B0 is a great starting model when constructing an image
classification pipeline.
This architecture manages to achieve high accuracy, while using a
parameter count of `7_200_312`.
If an EfficientNetV2B0 is not powerful enough for the task you are hoping to
solve, be sure to check out [KerasCV's other available Backbones](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/backbones)!
"""
classifier = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2-b0_imagenet_classifier"
)

"""
You may notice a small deviation from the old `keras.applications` API; where
you would construct the class with `EfficientNetV2B0(weights="imagenet")`.
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
Next, let's get some predictions from our classifier:
"""

predictions = classifier.predict(np.expand_dims(image, axis=0))
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
with open(classes, "rb") as f:
    classes = json.load(f)
"""
Now we can simply look up the class names via index:
"""
top_two = [classes[str(i)] for i in top_classes[-2:]]
print("Top two classes are:", top_two)

"""
Great!  Both of these appear to be correct!
However, the top class here is "Velvet".
We're trying to classify Cats VS Dogs. We don't care about the
velvet blanket!
This can be solved by fine tuning our own classifier.

# Fine tuning a pretrained classifier

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png)

When labeled images specific to our task are available, fine-tuning a custom
classifier can improve performance.
If we want to train a Cats vs Dogs Classifier, using explicitly labeled Cat vs
Dog data should perform better than the generic classifier!
For many tasks, no relevant pretrained model
will be available (e.g., categorizing images specific to your application).

First, let's get started by loading some data:
"""

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()

data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
train_dataset = data["train"]

IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes

resizing = keras_cv.layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True)


def preprocess_inputs(image, label):
    image = tf.cast(image, tf.float32)
    image = resizing(image)
    return resizing(image), tf.one_hot(label, num_classes)


train_dataset = train_dataset.shuffle(10 * BATCH_SIZE).map(
    preprocess_inputs, num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.batch(BATCH_SIZE)

images = next(iter(train_dataset.take(1)))[0]
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))


"""
Next let's construct our model:
"""

backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2-s_imagenet",
)
"""
The use of imagenet in the preset name indicates that the backbone was
pretrained on the ImageNet dataset.
This is called transfer learning, and results in a more efficient fitting of the
model to our new dataset.
Transfer learning models fit in fewer epochs, and often achieve better final
results.

Next lets put together our classifier:
"""

model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2, activation="softmax"),
    ]
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

"""
Here our classifier is just a simple `keras.Sequential`.
All that is left to do is call `model.fit()`!
"""

model.fit(train_dataset)


"""
Let's look at how our model performs after the fine tuning!
"""

predictions = model.predict(np.expand_dims(image, axis=0))

classes = {0: "cat", 1: "dog"}
print("Top class is:", classes[predictions[0].argmax()])

"""
Awesome!  Looks like the model correctly classified the image.
"""

"""
# Train a Classifier from Scratch

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png)

Now that we've gotten our hands dirty with classification, let's take on one
last task: training a classification model from scratch!
A standard benchmark for image classification is the ImageNet dataset, however
due to licensing constraints we will use the CalTech 101 image classification
dataset in this tutorial.
While we use the simpler CalTech 101 dataset in this guide, the same training
template may be used on ImageNet to achieve state of the art scores.

Let's start out by tackling data loading:
"""
NUM_CLASSES = 101
# Change to 100~ to fully train.
EPOCHS = 1

def package_inputs(image, label):
    return {
        "images": image,
        "labels": tf.one_hot(label, NUM_CLASSES)
    }

train_ds, eval_ds = tfds.load("caltech101", split=["train", "test"], as_supervised="true")
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.ragged_batch(BATCH_SIZE)
eval_ds = eval_ds.ragged_batch(BATCH_SIZE)

image_batch = next(train_ds)["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
"""

## Data Augmentation

In our previous finetuning exmaple, we performed a static resizing operation and
did not include any image augmentation.
This is because a single pass over the training set was sufficient to achieve
decent results.
When training to solve a more difficult task, you'll want to include data
augmentation in your data pipeline.

Data augmentation is a way as cheaply producing more training examples for your
model to learn from.
Data augmentation is a vital technique when attempting to traing powerful image
classifier, but the modern augmetnation landscape is extremely complex.
There are numerous powerful augmentations available, and there's no one set of
augmentations that is optimal for all tasks.
Despite this, we have prepared a set of augmentations that while not optimal for
all tests, tend to perform pretty well.
One caveat to be aware of with image data augmentation is that you must be careful
to not shift your augmented data distribution too far from the original data
distribution.
The goal is to introduce noise to prevent overfitting and increase generalization,
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
"""
random_flip = keras_cv.layers.RandomFlip()
augmenters = [
    random_flip
]

image_batch = random_flip(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
"""
The next augmentation we'll use is `RandomCropAndResize`.
This operation selects a random subset of the image, then resizes it to the provided target size.
By using this augmentation, we force our classifier to become spatially invariant.
Additionally, this layer accepts a `aspect_ratio_factor` which can be used to distort the aspect ratio of the image.
While this can improve model performance, it should be used with caution.
It is very easy for an aspect ratio distortion to shift a sample too far from the original training set's data distribution.
Remember - the goal of data augmentation is to produce more training samples that align with the data distribution of your training set!

`RandomCropAndResize` also can handle `tf.RaggedTensor` inputs.  In the
CalTech101 image dataset images come in a wide variety of sizes.
As such they cannot easily be batched together into a dense training batch.
Luckily, `RandomCropAndResize` handles the Ragged -> Dense conversion process
for you!

Let's add a `RandomCropAndResize` to our set of augmentations:
"""
crop_and_resize = keras_cv.layers.RandomCropAndResize(
    target_size=IMAGE_SIZE,
    area_factor=(0.8, 1.0),
    aspect_ratio_factor=(0.9, 1.1),
)
augmenters+=[
    crop_and_resize
]

image_batch = crop_and_resize(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
"""
Great!  We are now working with a batch of dense images.
Next up, lets include some spatial and color-based jitter to our training set.
This will allow us to produce a classifier that is robust to lighting flickers,
shadows, and more.

There are limitless ways to augment an image by altering color and spatial
features, but perhaps
[the most battle tested technique is `RandAugment`](https://arxiv.org/abs/1909.13719).
`RandAugment` is actually a set of 10 different augmentations:
`AutoContrast`, `Equalize`, `Solarize`, `RandomColorJitter`, `RandomContrast`,
`RandomBrightness`, `ShearX`, `ShearY`, `TranslateX` and `TranslateY`.
At inference time, `num_augmentations` augmenters are sampled for each image,
and random magnitude factors are sampled for each.
These augmentations are then applied sequentially.

KerasCV makes tuning these parameters easy using the `augmentations_per_image`
and `magnitude` parameters!
Let's take it for a spin:
"""
rand_augment = keras_cv.layers.RandAugment(
    augmentations_per_image=3,
    magnitude=0.3,
    value_range=(0, 255),
)
augmenters+=[
    rand_augment
]

image_batch = rand_augment(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

"""
Looks great; but we're not done yet!
What if an image is missing one critical feature of a class?  For example, what
if a leaf is blocking the view of a Cat's ear, but our classifier learned to
classify cats simply by observing their ears?

One easy approach to tackling this is to use `RandomCutout`, which randomly
strips out a sub-section of the image:
"""
random_cutout = keras_cv.layers.RandomCutout()
keras_cv.visualization.plot_image_gallery(
    random_cutout(image_batch),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

"""
While this tackles the problem reasonably well, it can cause the classifier to
develop responses to borders between features and black pixel areas caused by
the cutout.

[`CutMix`](https://arxiv.org/abs/1905.04899) solves the same issue; but by using
a more complex (and effective!) technique.
Instead of replacing the cut-out areas with black pixels, `CutMix` replaces
these regions with regions of other images sampled from within your training
set!
Following this replacement, the image's classification label is updated to be a
blend of the original image's class label, as well as the image that was
overlaid into the cutout section's label.

What does this look like in practice?  Let's check it out:
"""
cut_mix = keras_cv.layers.CutMix()
keras_cv.visualization.plot_image_gallery(
    cut_mix(image_batch),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
"""
Let's hold off from adding it to our augmenter for a minute - more on that
soon!

Next, let's look into `MixUp()```.
Unfortunately, while `MixUp()` has been empirically shown to *substantially*
improve both the robustness and the generalization of the trained model,
it is not well-understood why such improvement occurs.
A little alchemy never hurt anyone!

![](https://i.imgur.com/d4ZZYvW.png)

`MixUp()` works by sampling two images from a batch, then proceeding to
literally blend together their pixel intensities as well as their classification
labels.

Let's see it in action:
"""
mix_up = keras_cv.layers.MixUp()
keras_cv.visualization.plot_image_gallery(
    mix_up(image_batch),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
"""
Instead of applying `CutMix()` and `MixUp()` to every image, we instead pick
one or the other to apply to each batch.
This can be expressed using `keras_cv.layers.RandomChoice()`
"""

cut_mix_or_mix_up = keras_cv.layers.RandomChoice([cut_mix, mix_up])
augmenters += [cut_mix_or_mix_up]

"""
Applying it to your training pipeline is easy:
"""

augmenter = keras.Sequential([augmenters])
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(train_ds)["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

"""
We also need to resize our evaluation set, but luckily that's trivial:
"""
inference_resizing = keras_cv.layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(eval_ds)["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

"""
Finally, lets unpackage our datasets and prepare to pass them to the `model.fit()`
call, which accepts a tuple of `(images, labels)`.
"""

def unpackage_dict(inputs):
    return inputs["images"], inputs["labels"]

train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

"""
Cool!  Data augmentation is by far the hardest piece of training a classifier
in the modern era.
Congratulations on making it this far!

## Optimizer Tuning

To achieve optimal performance, we must implement a Warm up Cosinde decay
learning rate schedule.
While we won't go into detail on this schedule, [you can read more about it
here](https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b).
"""


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
            + tf.cos(
                tf.constant(math.pi)
                * tf.cast(global_step - warmup_steps - hold, tf.float32)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )

    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)
    target_lr = tf.cast(target_lr, tf.float32)

    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(
        global_step < warmup_steps, warmup_lr, learning_rate
    )
    return learning_rate


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, warmup_steps, total_steps, hold, start_lr=0.0, target_lr=1e-2
    ):
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

        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")

"""
Next let's construct this optimizer:
"""
total_steps = (NUM_IMAGES // BATCH_SIZE) * EPOCHS
warmup_steps = int(FLAGS.warmup_steps_percentage * total_steps)
hold_steps = int(FLAGS.warmup_hold_steps_percentage * total_steps)
schedule = WarmUpCosineDecay(
    start_lr=0.0,
    target_lr=INITIAL_LEARNING_RATE,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    hold=hold_steps,
)
optimizer = optimizers.SGD(
    weight_decay=FLAGS.weight_decay,
    learning_rate=schedule,
    momentum=0.9,
    use_ema=FLAGS.use_ema,
)
"""
At long last, we can now build our model and call `fit()`!
"""

backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2-b0",
)
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2, activation="softmax"),
    ]
)


model.fit(train_dataset)

"""
When using `MixUp()` and `CutMix()`, using `label_smoothing` in your loss is
extremely important.
"""
loss = losses.CategoricalCrossentropy(label_smoothing=0.1)
"""
Let's compile our model:
"""
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[
        metrics.CategoricalAccuracy(),
        metrics.TopKCategoricalAccuracy(k=5),
    ],
)
"""
and finally call fit().
"""
model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=model_callbacks,
    validation_data=test_ds,
)
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
