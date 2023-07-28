"""
Title: [KerasCV] Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet), updated by [Aritra Roy Gosthipaty](https://twitter.com/ariG23498) and [Margaret Maynard-Reid](https://twitter.com/margaretmz)
Date created: 2019/03/20
Last modified: 2023/06/19
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
Accelerator: GPU
"""

"""
This tutorial uses a U-Net like architecture for image segmentation. Data processing and
augmentations are implemented with [KerasCV](https://keras.io/keras_cv/).

U-Net was introduced in the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597/).
Although U-Net is a model for image
segmentation, it's also used in generative models such as Pix2Pix and diffusion models.
So it's important to have a solid understanding of its architecture.
"""

"""
## Setup and Imports

First let's set up install and imports of the dependencies.

To run this tutorial, you will need to install keras-cv with the following command:
`pip install keras-cv`
"""

import random

import keras
import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

"""
## Configuration

Please feel free to tweak the configurations yourself and note how the model training
changes. This is an excellent exercise to get a better understanding of the training
pipeline.
"""

# Image Config
HEIGHT = 160
WIDTH = 160
NUM_CLASSES = 3

# Augmentation Config
ROTATION_FACTOR = (-0.2, 0.2)

# Training Config
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
AUTOTUNE = tf.data.AUTOTUNE

"""
## Download the data

We download
[the Oxford-IIT Pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
with TensorFlow
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

rescale_images_and_correct_masks = lambda inputs: {
    "images": tf.cast(inputs["image"], dtype=tf.float32) / 255.0,
    "segmentation_masks": inputs["segmentation_mask"] - 1,
}

train_ds = orig_train_ds.map(
    rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE
)
val_ds = orig_val_ds.map(rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE)

"""
## Utility Function

The `unpackage_inputs` is a utility function that is used to unpack the inputs from the
dictionary format to a tuple of `(images, segmentation_masks)`. This will be used later
on for visualizing the images and segmentation masks and also the model predictions.
"""


def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks


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
flip, random rotation and RandAugment to the train dataset. Here is
[a tutorial with more details on RandAugment](https://keras.io/examples/vision/randaugment/).

We only apply the resizing operation to the validation dataset
"""

resize_fn = keras_cv.layers.Resizing(
    HEIGHT,
    WIDTH,
)

augment_fn = keras.Sequential(
    [
        resize_fn,
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandomRotation(
            factor=ROTATION_FACTOR,
            segmentation_classes=NUM_CLASSES,
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
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
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

The U-Net consists of an encoder for downsampling and a decoder for upsampling with skip
connections.

The model architecture shapes like the letter U hence the name U-Net.

![unet.png](https://i.imgur.com/PgGRty2.png)

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
            random_index = random.randint(0, BATCH_SIZE - 1)
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
model = get_model(img_size=(HEIGHT, WIDTH), num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model, doing validation at the end of each epoch.
history = model.fit(
    augmented_train_ds,
    epochs=EPOCHS,
    validation_data=resized_val_ds,
    callbacks=callbacks,
)

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

"""
## Acknowledgements

We would like to thank [Martin Gorner](https://twitter.com/martin_gorner) for his thorough review.
Google Cloud credits were provided for this project.
"""
