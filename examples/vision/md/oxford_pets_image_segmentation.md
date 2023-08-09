# [KerasCV] Image segmentation with a U-Net-like architecture

**Author:** [fchollet](https://twitter.com/fchollet), updated by [Aritra Roy Gosthipaty](https://twitter.com/ariG23498) and [Margaret Maynard-Reid](https://twitter.com/margaretmz)<br>
**Date created:** 2019/03/20<br>
**Last modified:** 2023/06/19<br>
**Description:** Image segmentation model trained from scratch on the Oxford Pets dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/oxford_pets_image_segmentation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/oxford_pets_image_segmentation.py)



This tutorial uses a U-Net like architecture for image segmentation. Data processing and
augmentations are implemented with [KerasCV](https://keras.io/keras_cv/).

U-Net was introduced in the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597/).
Although U-Net is a model for image
segmentation, it's also used in generative models such as Pix2Pix and diffusion models.
So it's important to have a solid understanding of its architecture.

---
## Setup and Imports

First let's set up install and imports of the dependencies.

To run this tutorial, you will need to install keras-cv with the following command:
`pip install keras-cv`


```python
import random

import keras
import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
```

---
## Configuration

Please feel free to tweak the configurations yourself and note how the model training
changes. This is an excellent exercise to get a better understanding of the training
pipeline.


```python
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
```

---
## Download the data

We download
[the Oxford-IIT Pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
with TensorFlow
Datasets (TFDS) with one line of code. Combine the training and test data, and then split
the combined data into 80% training dataset and 20% test dataset (used later on for both
validation and testing).


```python
tfds.disable_progress_bar()
orig_train_ds, orig_val_ds = tfds.load(
    name="oxford_iiit_pet",
    split=["train+test[:80%]", "test[80%:]"],
)
```

<div class="k-default-codeblock">
```
 Downloading and preparing dataset 773.52 MiB (download: 773.52 MiB, generated: 774.69 MiB, total: 1.51 GiB) to /home/arig23498/tensorflow_datasets/oxford_iiit_pet/3.2.0...
 Dataset oxford_iiit_pet downloaded and prepared to /home/arig23498/tensorflow_datasets/oxford_iiit_pet/3.2.0. Subsequent calls will reuse this data.

```
</div>
---
## Preprocess the data

Here we processes the images and segmentation masks in the inputs **dictionary**, mapping
the image pixel intensities from `[0, 255]` to the range `[0.0, 1.0]` and adjusting
segmentation masks indices from 1-based to 0-based.

Also note the renaming of the keys of the dictionary. The processed datasets are
formatted suitably for KerasCV layers, which expect inputs in a specific dictionary
format.


```python
rescale_images_and_correct_masks = lambda inputs: {
    "images": tf.cast(inputs["image"], dtype=tf.float32) / 255.0,
    "segmentation_masks": inputs["segmentation_mask"] - 1,
}

train_ds = orig_train_ds.map(
    rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE
)
val_ds = orig_val_ds.map(rescale_images_and_correct_masks, num_parallel_calls=AUTOTUNE)
```

---
## Utility Function

The `unpackage_inputs` is a utility function that is used to unpack the inputs from the
dictionary format to a tuple of `(images, segmentation_masks)`. This will be used later
on for visualizing the images and segmentation masks and also the model predictions.


```python

def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks

```

Let's visualized a few images and their segmentation masks from the training data, with
the `keras_cv.visualization.plot_segmentation_mask_gallery` API.


```python
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
```


    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_13_0.png)
    


---
## Data Augmentation

We resize both the images and masks to the width/height as specified. Then use KerasCV's
`RandomFlip`, `RandomRotation` and `RandAugment` to apply image augmentation of random
flip, random rotation and RandAugment to the train dataset. Here is
[a tutorial with more details on RandAugment](https://keras.io/examples/vision/randaugment/).

We only apply the resizing operation to the validation dataset


```python
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
```

Create training and validation datasets.


```python
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
```

---
## Visualize the data

Now let's again visualize a few of the images and their segmentation masks with the
`keras_cv.visualization.plot_segmentation_mask_gallery` API. Note the effects from the
data augmentation.


```python
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
```


    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_19_0.png)
    


---
## Model architecture

The U-Net consists of an encoder for downsampling and a decoder for upsampling with skip
connections.

The model architecture shapes like the letter U hence the name U-Net.

![unet.png](https://i.imgur.com/PgGRty2.png)

We create a function `get_model` to define a U-Net like architecture.


```python

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

```

We subclass `Callback` to monitor the model training progress: training and validation
loss, and visually inspect the images, predicted masks and ground truth masks.


```python

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
```

---
## Train the model

Now let's create the model, compile and train it for 50 epochs by calling `model.fit()`.


```python
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
```

<div class="k-default-codeblock">
```
Epoch 1/50
 6/52 [==>...........................] - ETA: 13s - loss: 2.7063 - accuracy: 0.5196

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 1s 14ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_3.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 55s 575ms/step - loss: 1.4094 - accuracy: 0.5980 - val_loss: 1.3672 - val_accuracy: 0.5740
Epoch 2/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.9728 - accuracy: 0.6325

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 500ms/step - loss: 0.9081 - accuracy: 0.6519 - val_loss: 2.3679 - val_accuracy: 0.5740
Epoch 3/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.8528 - accuracy: 0.6631

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.8293 - accuracy: 0.6710 - val_loss: 2.9935 - val_accuracy: 0.5740
Epoch 4/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.8152 - accuracy: 0.6737

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 500ms/step - loss: 0.8014 - accuracy: 0.6799 - val_loss: 3.2562 - val_accuracy: 0.5740
Epoch 5/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.8009 - accuracy: 0.6765

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 496ms/step - loss: 0.7764 - accuracy: 0.6869 - val_loss: 3.7120 - val_accuracy: 0.5740
Epoch 6/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.7636 - accuracy: 0.6922

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 17ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_15.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 504ms/step - loss: 0.7605 - accuracy: 0.6930 - val_loss: 3.7244 - val_accuracy: 0.5740
Epoch 7/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.7541 - accuracy: 0.6944

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.7483 - accuracy: 0.6975 - val_loss: 3.6724 - val_accuracy: 0.5740
Epoch 8/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.7458 - accuracy: 0.6980

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 490ms/step - loss: 0.7342 - accuracy: 0.7015 - val_loss: 3.2679 - val_accuracy: 0.5740
Epoch 9/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.7312 - accuracy: 0.7025

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 499ms/step - loss: 0.7197 - accuracy: 0.7078 - val_loss: 2.3689 - val_accuracy: 0.5740
Epoch 10/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.7299 - accuracy: 0.7030

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 488ms/step - loss: 0.7126 - accuracy: 0.7104 - val_loss: 1.5338 - val_accuracy: 0.5748
Epoch 11/50
 6/52 [==>...........................] - ETA: 21s - loss: 0.7153 - accuracy: 0.7079

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 18ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_27.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 504ms/step - loss: 0.7040 - accuracy: 0.7137 - val_loss: 1.2267 - val_accuracy: 0.5882
Epoch 12/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.7059 - accuracy: 0.7126

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.6987 - accuracy: 0.7160 - val_loss: 0.7751 - val_accuracy: 0.6842
Epoch 13/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6892 - accuracy: 0.7192

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 497ms/step - loss: 0.6884 - accuracy: 0.7200 - val_loss: 0.6641 - val_accuracy: 0.7282
Epoch 14/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6877 - accuracy: 0.7212

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.6858 - accuracy: 0.7216 - val_loss: 0.6056 - val_accuracy: 0.7551
Epoch 15/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6857 - accuracy: 0.7218

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 489ms/step - loss: 0.6761 - accuracy: 0.7254 - val_loss: 0.6095 - val_accuracy: 0.7533
Epoch 16/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.6823 - accuracy: 0.7216

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 18ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_39.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 27s 507ms/step - loss: 0.6758 - accuracy: 0.7258 - val_loss: 0.5887 - val_accuracy: 0.7607
Epoch 17/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6768 - accuracy: 0.7244

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 491ms/step - loss: 0.6648 - accuracy: 0.7303 - val_loss: 0.5810 - val_accuracy: 0.7653
Epoch 18/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6642 - accuracy: 0.7310

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 495ms/step - loss: 0.6604 - accuracy: 0.7318 - val_loss: 0.5755 - val_accuracy: 0.7686
Epoch 19/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.6670 - accuracy: 0.7296

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.6569 - accuracy: 0.7332 - val_loss: 0.5770 - val_accuracy: 0.7681
Epoch 20/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6626 - accuracy: 0.7312

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 495ms/step - loss: 0.6561 - accuracy: 0.7334 - val_loss: 0.5612 - val_accuracy: 0.7741
Epoch 21/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6550 - accuracy: 0.7348

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 17ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_51.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 504ms/step - loss: 0.6462 - accuracy: 0.7382 - val_loss: 0.5584 - val_accuracy: 0.7745
Epoch 22/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6421 - accuracy: 0.7389

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 495ms/step - loss: 0.6409 - accuracy: 0.7400 - val_loss: 0.5635 - val_accuracy: 0.7738
Epoch 23/50
 7/52 [===>..........................] - ETA: 21s - loss: 0.6484 - accuracy: 0.7353

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 501ms/step - loss: 0.6376 - accuracy: 0.7414 - val_loss: 0.5513 - val_accuracy: 0.7786
Epoch 24/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6369 - accuracy: 0.7399

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 489ms/step - loss: 0.6357 - accuracy: 0.7422 - val_loss: 0.5448 - val_accuracy: 0.7805
Epoch 25/50
 6/52 [==>...........................] - ETA: 24s - loss: 0.6409 - accuracy: 0.7400

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 501ms/step - loss: 0.6320 - accuracy: 0.7442 - val_loss: 0.5632 - val_accuracy: 0.7743
Epoch 26/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6304 - accuracy: 0.7448

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 17ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_63.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 503ms/step - loss: 0.6276 - accuracy: 0.7458 - val_loss: 0.5377 - val_accuracy: 0.7839
Epoch 27/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6179 - accuracy: 0.7504

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.6208 - accuracy: 0.7485 - val_loss: 0.5313 - val_accuracy: 0.7873
Epoch 28/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6193 - accuracy: 0.7482

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.6210 - accuracy: 0.7485 - val_loss: 0.5246 - val_accuracy: 0.7905
Epoch 29/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6299 - accuracy: 0.7444

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 496ms/step - loss: 0.6133 - accuracy: 0.7521 - val_loss: 0.5208 - val_accuracy: 0.7916
Epoch 30/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6043 - accuracy: 0.7545

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 494ms/step - loss: 0.6077 - accuracy: 0.7538 - val_loss: 0.5251 - val_accuracy: 0.7902
Epoch 31/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6129 - accuracy: 0.7510

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 18ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_75.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 505ms/step - loss: 0.6122 - accuracy: 0.7523 - val_loss: 0.5182 - val_accuracy: 0.7947
Epoch 32/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6201 - accuracy: 0.7491

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.6088 - accuracy: 0.7540 - val_loss: 0.5094 - val_accuracy: 0.7968
Epoch 33/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5996 - accuracy: 0.7568

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 490ms/step - loss: 0.6025 - accuracy: 0.7565 - val_loss: 0.5079 - val_accuracy: 0.7975
Epoch 34/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.6076 - accuracy: 0.7536

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.6003 - accuracy: 0.7575 - val_loss: 0.5242 - val_accuracy: 0.7871
Epoch 35/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6066 - accuracy: 0.7543

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 491ms/step - loss: 0.5981 - accuracy: 0.7581 - val_loss: 0.5052 - val_accuracy: 0.7982
Epoch 36/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5967 - accuracy: 0.7585

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 17ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_87.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 502ms/step - loss: 0.5918 - accuracy: 0.7604 - val_loss: 0.4997 - val_accuracy: 0.7999
Epoch 37/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5972 - accuracy: 0.7586

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.5991 - accuracy: 0.7583 - val_loss: 0.5364 - val_accuracy: 0.7831
Epoch 38/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.6019 - accuracy: 0.7566

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 494ms/step - loss: 0.5918 - accuracy: 0.7612 - val_loss: 0.4941 - val_accuracy: 0.8041
Epoch 39/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5880 - accuracy: 0.7624

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.5889 - accuracy: 0.7627 - val_loss: 0.4931 - val_accuracy: 0.8054
Epoch 40/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5847 - accuracy: 0.7629

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.5801 - accuracy: 0.7660 - val_loss: 0.4968 - val_accuracy: 0.8026
Epoch 41/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5877 - accuracy: 0.7630

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 18ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_99.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 502ms/step - loss: 0.5794 - accuracy: 0.7662 - val_loss: 0.4861 - val_accuracy: 0.8065
Epoch 42/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5876 - accuracy: 0.7627

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 491ms/step - loss: 0.5790 - accuracy: 0.7667 - val_loss: 0.4890 - val_accuracy: 0.8047
Epoch 43/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5750 - accuracy: 0.7666

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 496ms/step - loss: 0.5745 - accuracy: 0.7683 - val_loss: 0.4936 - val_accuracy: 0.8024
Epoch 44/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5752 - accuracy: 0.7677

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 495ms/step - loss: 0.5757 - accuracy: 0.7677 - val_loss: 0.4835 - val_accuracy: 0.8080
Epoch 45/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5801 - accuracy: 0.7665

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 493ms/step - loss: 0.5689 - accuracy: 0.7707 - val_loss: 0.4786 - val_accuracy: 0.8108
Epoch 46/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5644 - accuracy: 0.7707

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

4/4 [==============================] - 0s 18ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_25_111.png)
    


<div class="k-default-codeblock">
```
52/52 [==============================] - 26s 503ms/step - loss: 0.5683 - accuracy: 0.7711 - val_loss: 0.4850 - val_accuracy: 0.8057
Epoch 47/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5829 - accuracy: 0.7647

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 494ms/step - loss: 0.5745 - accuracy: 0.7680 - val_loss: 0.5031 - val_accuracy: 0.7967
Epoch 48/50
 6/52 [==>...........................] - ETA: 22s - loss: 0.5782 - accuracy: 0.7670

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 492ms/step - loss: 0.5698 - accuracy: 0.7710 - val_loss: 0.4775 - val_accuracy: 0.8084
Epoch 49/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.5704 - accuracy: 0.7694

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 496ms/step - loss: 0.5604 - accuracy: 0.7740 - val_loss: 0.4723 - val_accuracy: 0.8116
Epoch 50/50
 6/52 [==>...........................] - ETA: 23s - loss: 0.5633 - accuracy: 0.7737

Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment

52/52 [==============================] - 26s 490ms/step - loss: 0.5608 - accuracy: 0.7743 - val_loss: 0.4619 - val_accuracy: 0.8163

```
</div>
---
## Prediction with trained model
Now that the model training of U-Net has completed, let's test it by making predications
on a few sample images.


```python
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
```

<div class="k-default-codeblock">
```
4/4 [==============================] - 0s 17ms/step

```
</div>
    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_27_1.png)
    


---
## Acknowledgements

We would like to thank [Martin Gorner](https://twitter.com/martin_gorner) for his thorough review.
Google Cloud credits were provided for this project.
