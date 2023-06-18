"""
Title: [KerasCV] Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet), updated by [Aritra Roy
Gosthipaty](https://twitter.com/ariG23498) and [Margaret
Maynard-Reid](https://twitter.com/margaretmz)
Reviewer: [Martin Gorner](https://twitter.com/martin_gorner)
Date created: 2019/03/20
Last modified: 2023/06/19
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
Accelerator: GPU
"""

"""
This tutorial uses a U-Net like architecture for image segmentation. Data processing and
augmentations are implemented with [KerasCV](https://keras.io/keras_cv/).

U-Net was introduced in the paper, [U-Net: Convolutional Networks for Biomedical Image
Segmentation](https://arxiv.org/abs/1505.04597/). Although U-Net is a model for image
segmentation, it's also used in generative models such as Pix2Pix and diffusion models.
So it's important to have a solid understanding of its architecture.
"""

"""
## Setup and Imports

First let's set up install and imports of the dependencies.
"""

import keras
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds

import random
import numpy as np
from matplotlib import pyplot as plt

"""
## Configuration

Please feel free to tweak the configurations yourself and note how the model training
changes. This is an excellent exercise to get more understanding of the training
pipeline.
"""

# Image Config
height = 160
width = 160
num_classes = 3

# Augmentation Config
rotation_factor = (-0.2, 0.2)

# Training Config
batch_size = 128
epochs = 50
initial_learning_rate = 1e-4
max_learning_rate = 5e-4
warmup_epoch_percentage = 0.15
AUTOTUNE = tf.data.AUTOTUNE

"""
## Utility Functions

The `unpackage_inputs` is a utility function that is later used to reformat the input
dictionary into a tuple of images and segmentation masks.
"""


def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks


"""
## Download the data

We download [the Oxford-IIT Pet
dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) with TensorFlow
Datasets (TFDS) with one line of code. Combine the training and test data, and then split
the combined data into 80% training dataset and 20% test dataset (used later on for both
validation and testing).
"""

tfds.disable_progress_bar()
orig_train_ds, orig_val_ds = tfds.load(
    name="oxford_iiit_pet",
    split=["train+test[:80%]", "test[80%:]"],
)

"""
## Preprocess the data

Here we processes the images and segmentation masks in the inputs **dictionary**, mapping
the image pixel intensities from `[0, 255]` to the range `[0.0, 1.0]` and adjusting
segmentation masks indices from 1-based to 0-based.

Also note the renaming of the keys of the dictionary. The processed datasets are
formatted suitably for KerasCV layers, which expect inputs in a specific dictionary
format.
"""

key_rename_fn = lambda inputs: {
    "images": tf.cast(inputs["image"], dtype=tf.float32) / 255.0,
    "segmentation_masks": inputs["segmentation_mask"] - 1,
}

train_ds = orig_train_ds.map(key_rename_fn, num_parallel_calls=AUTOTUNE)
val_ds = orig_val_ds.map(key_rename_fn, num_parallel_calls=AUTOTUNE)

"""
Let's visualized a few images and their segmentation masks from the training data, with
the `keras_cv.visualization.plot_segmentation_mask_gallery` API.


"""

plot_train_ds = train_ds.map(unpackage_inputs).ragged_batch(4)
images, segmentation_masks = next(iter(plot_train_ds.take(1)))

keras_cv.visualization.plot_segmentation_mask_gallery(
    images,
    value_range=(0, 1),
    num_classes=3,
    y_true=segmentation_masks,
    y_pred=None,
    scale=4,
    rows=2,
    cols=2,
)

"""
## Data Augmentation
We resize both the images and masks to the width/height as specified. Then use KerasCV's
`RandomFlip`, `RandomRotation` and `RandAugment` to apply image augmentation of random
flip, random rotation and RandAugment to the train dataset. Here is [a tutorial with more
details on RandAugment](https://keras.io/examples/vision/randaugment/).

We apply only resizing operation to the validation dataset.
"""

resize_fn = keras_cv.layers.Resizing(
    height,
    width,
)

augment_fn = keras.Sequential(
    [
        resize_fn,
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandomRotation(
            factor=rotation_factor,
            segmentation_classes=num_classes,
        ),
        keras_cv.layers.RandAugment(
            value_range=(0, 1),
            geometric=False,
        ),
    ]
)

"""
Create training and validation datasets.
"""

augmented_train_ds = (
    train_ds.cache()
    .shuffle(batch_size * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
resized_val_ds = (
    val_ds.cache()
    .map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

"""
## Visualize the data
Now let's again visualize a few of the images and their segmentation masks with the
`keras_cv.visualization.plot_segmentation_mask_gallery` API. Note the effects from the
data augmentation.
"""

images, segmentation_masks = next(iter(augmented_train_ds.take(1)))

keras_cv.visualization.plot_segmentation_mask_gallery(
    images,
    value_range=(0, 1),
    num_classes=3,
    y_true=segmentation_masks,
    y_pred=None,
    scale=4,
    rows=2,
    cols=2,
)

"""
## Model architecture
The U-Net consists of an encoder for downsammpling and a decoder for upsampling with skip
connections.

The model architecture shapes like the letter U hence the name U-Net.

![unet.png](https://i.imgur.com/PgGRty2.png)
"""

"""
We create a function `get_model` to define a U-Net like architecture.
"""


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(
        x
    )

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Taking a batch of test inputs to measure model's progress.
test_images, test_masks = next(iter(resized_val_ds))

"""
Let us define a custom learning rate scheduler for the U-Net model we created, which uses
a warmup phase followed by cosine decay, in order to help improve the training of the
model later.

We start with a small learning rate (warmup phase), gradually increase it to avoid large
initial updates, and then slowly decrease it following a cosine schedule after the warmup
phase, allowing the learning rate to explore different regions of the weight space and
aiding in achieving a good solution.
"""


class WarmUpCosine(
    keras.optimizers.schedules.learning_rate_schedule.LearningRateSchedule
):
    """A LearningRateSchedule that uses a warmup cosine decay schedule."""

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        """
        Args:
            lr_start: The initial learning rate
            lr_max: The maximum learning rate to which lr should increase to in
                the warmup steps
            warmup_steps: The number of steps for which the model warms up
            total_steps: The total number of steps for the model training
        """
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        # Check whether the total number of steps is larger than the warmup
        # steps. If not, then throw a value error.
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )

        # `cos_annealed_lr` is a graph that increases to 1 from the initial
        # step to the warmup step. After that this graph decays to -1 at the
        # final step mark.
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )

        # Shift the mean of the `cos_annealed_lr` graph to 1. Now the grpah goes
        # from 0 to 2. Normalize the graph with 0.5 so that now it goes from 0
        # to 1. With the normalized graph we scale it with `lr_max` such that
        # it goes from 0 to `lr_max`
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        # Check whether warmup_steps is more than 0.
        if self.warmup_steps > 0:
            # Check whether lr_max is larger that lr_start. If not, throw a value
            # error.
            if self.lr_max < self.lr_start:
                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f"equal to lr_max {self.lr_max}."
                )

            # Calculate the slope with which the learning rate should increase
            # in the warumup schedule. The formula for slope is m = ((b-a)/steps)
            slope = (self.lr_max - self.lr_start) / self.warmup_steps

            # With the formula for a straight line (y = mx+c) build the warmup
            # schedule
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            # When the current step is lesser that warmup steps, get the line
            # graph. When the current step is greater than the warmup steps, get
            # the scaled cos graph.
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )

        # When the current step is more that the total steps, return 0 else return
        # the calculated graph.
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


# Get the total number of steps for training.
steps_per_epoch = augmented_train_ds.cardinality().numpy()
total_steps = int(steps_per_epoch * epochs)

# Calculate the number of steps for warmup.
warmup_steps = int(total_steps * warmup_epoch_percentage)

# Initialize the warmupcosine schedule.
scheduled_lrs = WarmUpCosine(
    lr_start=initial_learning_rate,
    lr_max=max_learning_rate,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)

lrs = [scheduled_lrs(step) for step in range(total_steps)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.show()

"""
We subclass `Callback` to monitor the model training progress: training and validation
loss, and visually inspect the images, predicted masks and ground truth masks.
"""


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None):
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            pred_masks = self.model.predict(test_images)
            pred_masks = tf.math.argmax(pred_masks, axis=-1)
            pred_masks = pred_masks[..., tf.newaxis]

            # Randomly select an image from the test batch
            random_index = random.randint(0, batch_size - 1)
            random_image = test_images[random_index]
            random_pred_mask = pred_masks[random_index]
            random_true_mask = test_masks[random_index]

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            ax[0].imshow(random_image)
            ax[0].set_title(f"Image: {epoch:03d}")

            ax[1].imshow(random_true_mask)
            ax[1].set_title(f"Ground Truth Mask: {epoch:03d}")

            ax[2].imshow(random_pred_mask)
            ax[2].set_title(
                f"Predicted Mask: {epoch:03d}",
            )

            plt.show()
            plt.close()


callbacks = [DisplayCallback(5)]

"""
## Train the model

Now let's create the model, compile and train it for 50 epochs by calling `model.fit()`.
"""

# Build model
model = get_model(img_size=(height, width), num_classes=num_classes)

model.compile(
    optimizer=keras.optimizers.Adam(scheduled_lrs),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model, doing validation at the end of each epoch.
history = model.fit(
    augmented_train_ds,
    epochs=epochs,
    validation_data=resized_val_ds,
    callbacks=callbacks,
)

"""
The learning curves of training / validation loss and training / validation accuracy
indicate that the modelis generalize well wihtout much overfitting.
"""

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.show()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.figure()
plt.plot(acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy Value")
plt.legend()
plt.show()

"""
## Prediction with trained model
Now that the model training of U-Net has completed, let's test it by making predications
on a few sample images.
"""

pred_masks = model.predict(test_images)
pred_masks = tf.math.argmax(pred_masks, axis=-1)[..., None]

keras_cv.visualization.plot_segmentation_mask_gallery(
    test_images,
    value_range=(0, 1),
    num_classes=3,
    y_true=test_masks,
    y_pred=pred_masks,
    scale=4,
    rows=2,
    cols=2,
)
