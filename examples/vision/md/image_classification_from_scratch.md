# Image classification from scratch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/27<br>
**Last modified:** 2023/11/09<br>
**Description:** Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py)



---
## Introduction

This example shows how to do image classification from scratch, starting from JPEG
image files on disk, without leveraging pre-trained weights or a pre-made Keras
Application model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary
classification dataset.

We use the `image_dataset_from_directory` utility to generate the datasets, and
we use Keras image preprocessing layers for image standardization and data augmentation.

---
## Setup


```python
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
```

---
## Load the data: the Cats vs Dogs dataset

### Raw data download

First, let's download the 786M ZIP archive of the raw data:


```python
!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

```python
!unzip -q kagglecatsanddogs_5340.zip
!ls
```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  786M  100  786M    0     0  73.1M      0  0:00:10  0:00:10 --:--:-- 68.0M

 CDLA-Permissive-2.0.pdf		   kagglecatsanddogs_5340.zip
 PetImages				  'readme[1].txt'
 image_classification_from_scratch.ipynb

```
</div>
Now we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each
subfolder contains image files for each category.


```python
!ls PetImages
```

<div class="k-default-codeblock">
```
Cat  Dog

```
</div>
### Filter out corrupted images

When working with lots of real-world image data, corrupted images are a common
occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
in their header.


```python
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")
```

<div class="k-default-codeblock">
```
Deleted 1590 images.

```
</div>
---
## Generate a `Dataset`


```python
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
```

<div class="k-default-codeblock">
```
Found 23410 files belonging to 2 classes.
Using 18728 files for training.
Using 4682 files for validation.

```
</div>
---
## Visualize the data

Here are the first 9 images in the training dataset.


```python

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```

<div class="k-default-codeblock">
```
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_14_1.png)
    


---
## Using image data augmentation

When you don't have a large image dataset, it's a good practice to artificially
introduce sample diversity by applying random yet realistic transformations to the
training images, such as random horizontal flipping or small random rotations. This
helps expose the model to different aspects of the training data while slowing down
overfitting.


```python
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

```

Let's visualize what the augmented samples look like, by applying `data_augmentation`
repeatedly to the first few images in the dataset:


```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")

```

<div class="k-default-codeblock">
```
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_18_1.png)
    


---
## Standardizing the data

Our image are already in a standard size (180x180), as they are being yielded as
contiguous `float32` batches by our dataset. However, their RGB channel values are in
the `[0, 255]` range. This is not ideal for a neural network;
in general you should seek to make your input values small. Here, we will
standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
our model.

---
## Two options to preprocess the data

There are two ways you could be using the `data_augmentation` preprocessor:

**Option 1: Make it part of the model**, like this:

```python
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model
```

With this option, your data augmentation will happen *on device*, synchronously
with the rest of the model execution, meaning that it will benefit from GPU
acceleration.

Note that data augmentation is inactive at test time, so the input samples will only be
augmented during `fit()`, not when calling `evaluate()` or `predict()`.

If you're training on GPU, this may be a good option.

**Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of
augmented images, like this:

```python
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
```

With this option, your data augmentation will happen **on CPU**, asynchronously, and will
be buffered before going into the model.

If you're training on CPU, this is the better option, since it makes data augmentation
asynchronous and non-blocking.

In our case, we'll go with the second option. If you're not sure
which one to pick, this second option (asynchronous preprocessing) is always a solid choice.

---
## Configure the dataset for performance

Let's apply data augmentation to our training dataset,
and let's make sure to use buffered prefetching so we can yield data from disk without
having I/O becoming blocking:


```python
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
```

---
## Build a model

We'll build a small version of the Xception network. We haven't particularly tried to
optimize the architecture; if you want to do a systematic search for the best model
configuration, consider using
[KerasTuner](https://github.com/keras-team/keras-tuner).

Note that:

- We start the model with the `data_augmentation` preprocessor, followed by a
 `Rescaling` layer.
- We include a `Dropout` layer before the final classification layer.


```python

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
```

<div class="k-default-codeblock">
```
You must install graphviz (see instructions at https://graphviz.gitlab.io/download/) for `plot_model` to work.

```
</div>
---
## Train the model


```python
epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
```

<div class="k-default-codeblock">
```
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

Epoch 1/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 351ms/step - acc: 0.6089 - loss: 0.6581

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 352ms/step - acc: 0.6117 - loss: 0.6551

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.6190 - loss: 0.6472

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.6197 - loss: 0.6465

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 350ms/step - acc: 0.6207 - loss: 0.6453

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 349ms/step - acc: 0.6214 - loss: 0.6446

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 616ms/step - acc: 0.6330 - loss: 0.6313

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 222s 810ms/step - acc: 0.6332 - loss: 0.6310 - val_acc: 0.4958 - val_loss: 0.6989
Epoch 2/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 354ms/step - acc: 0.7420 - loss: 0.5040

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 353ms/step - acc: 0.7421 - loss: 0.5034

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 351ms/step - acc: 0.7433 - loss: 0.5012

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 351ms/step - acc: 0.7434 - loss: 0.5010

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 351ms/step - acc: 0.7436 - loss: 0.5006

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 352ms/step - acc: 0.7438 - loss: 0.5004

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.7473 - loss: 0.4947

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 578ms/step - acc: 0.7473 - loss: 0.4945 - val_acc: 0.4958 - val_loss: 0.7227
Epoch 3/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 348ms/step - acc: 0.7909 - loss: 0.4189

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 348ms/step - acc: 0.7915 - loss: 0.4181

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 348ms/step - acc: 0.7937 - loss: 0.4152

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 347ms/step - acc: 0.7940 - loss: 0.4149

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 348ms/step - acc: 0.7944 - loss: 0.4143

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 349ms/step - acc: 0.7946 - loss: 0.4140

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.7996 - loss: 0.4077

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 579ms/step - acc: 0.7998 - loss: 0.4076 - val_acc: 0.4958 - val_loss: 0.7792
Epoch 4/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 353ms/step - acc: 0.8436 - loss: 0.3483

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 355ms/step - acc: 0.8440 - loss: 0.3473

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 353ms/step - acc: 0.8446 - loss: 0.3454

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 354ms/step - acc: 0.8446 - loss: 0.3452

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 353ms/step - acc: 0.8447 - loss: 0.3449

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 353ms/step - acc: 0.8448 - loss: 0.3448

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 347ms/step - acc: 0.8465 - loss: 0.3409

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 87s 580ms/step - acc: 0.8466 - loss: 0.3408 - val_acc: 0.4968 - val_loss: 0.8377
Epoch 5/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  25s 346ms/step - acc: 0.8710 - loss: 0.2936

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 345ms/step - acc: 0.8713 - loss: 0.2925

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 345ms/step - acc: 0.8720 - loss: 0.2907

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 346ms/step - acc: 0.8720 - loss: 0.2905

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 346ms/step - acc: 0.8721 - loss: 0.2903

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 345ms/step - acc: 0.8721 - loss: 0.2902

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 342ms/step - acc: 0.8727 - loss: 0.2875

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 370ms/step - acc: 0.8727 - loss: 0.2874 - val_acc: 0.5834 - val_loss: 0.6349
Epoch 6/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  27s 358ms/step - acc: 0.8824 - loss: 0.2601

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 355ms/step - acc: 0.8829 - loss: 0.2593

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.8838 - loss: 0.2575

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 102/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 352ms/step - acc: 0.8839 - loss: 0.2573

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 353ms/step - acc: 0.8840 - loss: 0.2571

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 352ms/step - acc: 0.8841 - loss: 0.2570

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.8853 - loss: 0.2542

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 375ms/step - acc: 0.8853 - loss: 0.2541 - val_acc: 0.8800 - val_loss: 0.2971
Epoch 7/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 357ms/step - acc: 0.9054 - loss: 0.2296

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 355ms/step - acc: 0.9053 - loss: 0.2290

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9049 - loss: 0.2280

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 354ms/step - acc: 0.9049 - loss: 0.2279

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 353ms/step - acc: 0.9049 - loss: 0.2277

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 353ms/step - acc: 0.9049 - loss: 0.2276

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 347ms/step - acc: 0.9047 - loss: 0.2260

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 376ms/step - acc: 0.9047 - loss: 0.2260 - val_acc: 0.8144 - val_loss: 0.3371
Epoch 8/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 355ms/step - acc: 0.9088 - loss: 0.2129

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 353ms/step - acc: 0.9091 - loss: 0.2119

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9093 - loss: 0.2104

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9093 - loss: 0.2103

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 351ms/step - acc: 0.9093 - loss: 0.2101

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 351ms/step - acc: 0.9094 - loss: 0.2100

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.9101 - loss: 0.2075

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 375ms/step - acc: 0.9101 - loss: 0.2074 - val_acc: 0.9037 - val_loss: 0.2515
Epoch 9/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 354ms/step - acc: 0.9132 - loss: 0.1941

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 355ms/step - acc: 0.9135 - loss: 0.1937

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9136 - loss: 0.1931

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 102/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 352ms/step - acc: 0.9136 - loss: 0.1931

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 352ms/step - acc: 0.9136 - loss: 0.1931

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 353ms/step - acc: 0.9136 - loss: 0.1931

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 355ms/step - acc: 0.9142 - loss: 0.1923

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 577ms/step - acc: 0.9142 - loss: 0.1923 - val_acc: 0.9177 - val_loss: 0.2070
Epoch 10/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  25s 341ms/step - acc: 0.9175 - loss: 0.1844

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 342ms/step - acc: 0.9181 - loss: 0.1835

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 344ms/step - acc: 0.9195 - loss: 0.1814

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 343ms/step - acc: 0.9196 - loss: 0.1812

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 343ms/step - acc: 0.9198 - loss: 0.1809

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 343ms/step - acc: 0.9199 - loss: 0.1808

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.9214 - loss: 0.1784

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 577ms/step - acc: 0.9214 - loss: 0.1784 - val_acc: 0.8422 - val_loss: 0.2917
Epoch 11/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 355ms/step - acc: 0.9283 - loss: 0.1659

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 354ms/step - acc: 0.9286 - loss: 0.1653

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9292 - loss: 0.1642

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 351ms/step - acc: 0.9292 - loss: 0.1641

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 352ms/step - acc: 0.9293 - loss: 0.1641

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 351ms/step - acc: 0.9293 - loss: 0.1640

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.9300 - loss: 0.1627

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 376ms/step - acc: 0.9300 - loss: 0.1627 - val_acc: 0.9230 - val_loss: 0.1889
Epoch 12/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  27s 358ms/step - acc: 0.9327 - loss: 0.1523

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 354ms/step - acc: 0.9326 - loss: 0.1522

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 352ms/step - acc: 0.9325 - loss: 0.1524

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 102/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 352ms/step - acc: 0.9324 - loss: 0.1524

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 352ms/step - acc: 0.9324 - loss: 0.1524

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 353ms/step - acc: 0.9324 - loss: 0.1525

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 347ms/step - acc: 0.9319 - loss: 0.1540

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 375ms/step - acc: 0.9319 - loss: 0.1540 - val_acc: 0.9227 - val_loss: 0.1862
Epoch 13/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 350ms/step - acc: 0.9424 - loss: 0.1428

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 358ms/step - acc: 0.9425 - loss: 0.1425

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  17s 356ms/step - acc: 0.9422 - loss: 0.1423

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 355ms/step - acc: 0.9422 - loss: 0.1423

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 355ms/step - acc: 0.9422 - loss: 0.1423

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 356ms/step - acc: 0.9422 - loss: 0.1423

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 349ms/step - acc: 0.9422 - loss: 0.1414

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 578ms/step - acc: 0.9422 - loss: 0.1414 - val_acc: 0.9053 - val_loss: 0.2298
Epoch 14/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 352ms/step - acc: 0.9406 - loss: 0.1447

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 350ms/step - acc: 0.9408 - loss: 0.1441

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.9412 - loss: 0.1430

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.9412 - loss: 0.1430

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 350ms/step - acc: 0.9413 - loss: 0.1429

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 349ms/step - acc: 0.9413 - loss: 0.1428

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 344ms/step - acc: 0.9416 - loss: 0.1420

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 374ms/step - acc: 0.9416 - loss: 0.1420 - val_acc: 0.9234 - val_loss: 0.2070
Epoch 15/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 357ms/step - acc: 0.9432 - loss: 0.1311

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 355ms/step - acc: 0.9432 - loss: 0.1310

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 353ms/step - acc: 0.9432 - loss: 0.1309

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 353ms/step - acc: 0.9432 - loss: 0.1309

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 352ms/step - acc: 0.9432 - loss: 0.1310

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 353ms/step - acc: 0.9432 - loss: 0.1310

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 346ms/step - acc: 0.9427 - loss: 0.1317

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 374ms/step - acc: 0.9427 - loss: 0.1317 - val_acc: 0.9364 - val_loss: 0.1668
Epoch 16/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 351ms/step - acc: 0.9490 - loss: 0.1259

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 350ms/step - acc: 0.9491 - loss: 0.1256

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 348ms/step - acc: 0.9493 - loss: 0.1250

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 349ms/step - acc: 0.9493 - loss: 0.1250

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 349ms/step - acc: 0.9493 - loss: 0.1249

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 349ms/step - acc: 0.9493 - loss: 0.1249

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 343ms/step - acc: 0.9491 - loss: 0.1245

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 372ms/step - acc: 0.9491 - loss: 0.1244 - val_acc: 0.9292 - val_loss: 0.1664
Epoch 17/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 358ms/step - acc: 0.9506 - loss: 0.1217

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 357ms/step - acc: 0.9504 - loss: 0.1214

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  17s 356ms/step - acc: 0.9503 - loss: 0.1205

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 356ms/step - acc: 0.9503 - loss: 0.1204

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 355ms/step - acc: 0.9504 - loss: 0.1203

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 355ms/step - acc: 0.9504 - loss: 0.1202

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 350ms/step - acc: 0.9504 - loss: 0.1193

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 380ms/step - acc: 0.9504 - loss: 0.1193 - val_acc: 0.9356 - val_loss: 0.1718
Epoch 18/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 360ms/step - acc: 0.9554 - loss: 0.1170

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 357ms/step - acc: 0.9554 - loss: 0.1167

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  17s 357ms/step - acc: 0.9552 - loss: 0.1163

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 357ms/step - acc: 0.9551 - loss: 0.1162

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 356ms/step - acc: 0.9551 - loss: 0.1162

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 356ms/step - acc: 0.9551 - loss: 0.1161

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 358ms/step - acc: 0.9546 - loss: 0.1152

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 579ms/step - acc: 0.9546 - loss: 0.1152 - val_acc: 0.8894 - val_loss: 0.2475
Epoch 19/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  25s 343ms/step - acc: 0.9513 - loss: 0.1233

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 344ms/step - acc: 0.9517 - loss: 0.1221

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 343ms/step - acc: 0.9523 - loss: 0.1199

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 344ms/step - acc: 0.9523 - loss: 0.1197

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 344ms/step - acc: 0.9524 - loss: 0.1195

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 344ms/step - acc: 0.9524 - loss: 0.1193

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 344ms/step - acc: 0.9534 - loss: 0.1162

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 372ms/step - acc: 0.9534 - loss: 0.1161 - val_acc: 0.9319 - val_loss: 0.1767
Epoch 20/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  25s 342ms/step - acc: 0.9524 - loss: 0.1172

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 343ms/step - acc: 0.9528 - loss: 0.1164

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 343ms/step - acc: 0.9538 - loss: 0.1145

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 343ms/step - acc: 0.9539 - loss: 0.1144

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 343ms/step - acc: 0.9539 - loss: 0.1142

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 343ms/step - acc: 0.9540 - loss: 0.1141

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 338ms/step - acc: 0.9552 - loss: 0.1117

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 578ms/step - acc: 0.9552 - loss: 0.1117 - val_acc: 0.9473 - val_loss: 0.1432
Epoch 21/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 349ms/step - acc: 0.9551 - loss: 0.1042

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 352ms/step - acc: 0.9553 - loss: 0.1038

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.9557 - loss: 0.1027

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 351ms/step - acc: 0.9558 - loss: 0.1026

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 350ms/step - acc: 0.9558 - loss: 0.1025

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 351ms/step - acc: 0.9559 - loss: 0.1024

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 345ms/step - acc: 0.9564 - loss: 0.1014

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 578ms/step - acc: 0.9564 - loss: 0.1013 - val_acc: 0.9366 - val_loss: 0.1476
Epoch 22/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 349ms/step - acc: 0.9614 - loss: 0.0954

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 349ms/step - acc: 0.9613 - loss: 0.0954

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 348ms/step - acc: 0.9611 - loss: 0.0957

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 102/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 348ms/step - acc: 0.9611 - loss: 0.0957

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 348ms/step - acc: 0.9611 - loss: 0.0957

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 347ms/step - acc: 0.9611 - loss: 0.0957

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 343ms/step - acc: 0.9606 - loss: 0.0960

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 371ms/step - acc: 0.9606 - loss: 0.0960 - val_acc: 0.9308 - val_loss: 0.1904
Epoch 23/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 354ms/step - acc: 0.9565 - loss: 0.1017

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 350ms/step - acc: 0.9571 - loss: 0.1008

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 347ms/step - acc: 0.9581 - loss: 0.0994

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 347ms/step - acc: 0.9582 - loss: 0.0993

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 347ms/step - acc: 0.9584 - loss: 0.0991

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 347ms/step - acc: 0.9584 - loss: 0.0991

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 343ms/step - acc: 0.9591 - loss: 0.0981

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 56s 372ms/step - acc: 0.9591 - loss: 0.0980 - val_acc: 0.9431 - val_loss: 0.1554
Epoch 24/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 357ms/step - acc: 0.9610 - loss: 0.0926

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 356ms/step - acc: 0.9612 - loss: 0.0923

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 354ms/step - acc: 0.9614 - loss: 0.0920

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 354ms/step - acc: 0.9614 - loss: 0.0920

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  15s 354ms/step - acc: 0.9615 - loss: 0.0919

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 354ms/step - acc: 0.9615 - loss: 0.0919

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 348ms/step - acc: 0.9614 - loss: 0.0916

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 86s 578ms/step - acc: 0.9614 - loss: 0.0916 - val_acc: 0.9223 - val_loss: 0.2491
Epoch 25/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  26s 348ms/step - acc: 0.9656 - loss: 0.0900

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  23s 347ms/step - acc: 0.9656 - loss: 0.0896

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 348ms/step - acc: 0.9657 - loss: 0.0888

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 347ms/step - acc: 0.9657 - loss: 0.0887

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 348ms/step - acc: 0.9657 - loss: 0.0886

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 347ms/step - acc: 0.9658 - loss: 0.0885

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 349ms/step - acc: 0.9657 - loss: 0.0878

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 57s 378ms/step - acc: 0.9657 - loss: 0.0878 - val_acc: 0.9037 - val_loss: 0.2279

<keras.src.callbacks.history.History at 0x7fd1a06f6f20>

```
</div>
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).

---
## Run inference on new data

Note that data augmentation and dropout are inactive at inference time.


```python
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
This image is 87.42% cat and 12.58% dog.

/var/tmp/ipykernel_16954/3443711317.py:8: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.
  score = float(keras.ops.sigmoid(predictions[0]))

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_29_2.png)
    

