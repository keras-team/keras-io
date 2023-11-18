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
100  786M  100  786M    0     0  11.1M      0  0:01:10  0:01:10 --:--:-- 11.8M

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




    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_24_0.png)
    



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
Epoch 1/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1700272973.676197 1678132 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 317ms/step - acc: 0.6092 - loss: 0.6542

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 317ms/step - acc: 0.6117 - loss: 0.6518

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 316ms/step - acc: 0.6179 - loss: 0.6457

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 315ms/step - acc: 0.6189 - loss: 0.6448

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 314ms/step - acc: 0.6197 - loss: 0.6439

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 314ms/step - acc: 0.6203 - loss: 0.6434

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 580ms/step - acc: 0.6307 - loss: 0.6324

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 224s 771ms/step - acc: 0.6310 - loss: 0.6321 - val_acc: 0.4958 - val_loss: 0.6935
Epoch 2/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  27s 367ms/step - acc: 0.7385 - loss: 0.5128

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  24s 360ms/step - acc: 0.7394 - loss: 0.5110

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 350ms/step - acc: 0.7417 - loss: 0.5066

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  16s 349ms/step - acc: 0.7419 - loss: 0.5061

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 348ms/step - acc: 0.7423 - loss: 0.5055

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 347ms/step - acc: 0.7425 - loss: 0.5050

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 336ms/step - acc: 0.7474 - loss: 0.4969

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 54s 360ms/step - acc: 0.7475 - loss: 0.4967 - val_acc: 0.4958 - val_loss: 0.7056
Epoch 3/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 322ms/step - acc: 0.8073 - loss: 0.4051

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 323ms/step - acc: 0.8073 - loss: 0.4047

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 321ms/step - acc: 0.8081 - loss: 0.4030

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 320ms/step - acc: 0.8081 - loss: 0.4029

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 320ms/step - acc: 0.8083 - loss: 0.4026

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 320ms/step - acc: 0.8083 - loss: 0.4024

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 315ms/step - acc: 0.8102 - loss: 0.3987

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 340ms/step - acc: 0.8102 - loss: 0.3986 - val_acc: 0.4958 - val_loss: 0.8008
Epoch 4/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 318ms/step - acc: 0.8379 - loss: 0.3511

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  80/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 316ms/step - acc: 0.8382 - loss: 0.3498

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 323ms/step - acc: 0.8390 - loss: 0.3472

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 324ms/step - acc: 0.8391 - loss: 0.3469

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 323ms/step - acc: 0.8393 - loss: 0.3465

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 323ms/step - acc: 0.8394 - loss: 0.3463

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 317ms/step - acc: 0.8418 - loss: 0.3405

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 52s 342ms/step - acc: 0.8418 - loss: 0.3403 - val_acc: 0.4958 - val_loss: 0.8938
Epoch 5/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 310ms/step - acc: 0.8641 - loss: 0.3061

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 308ms/step - acc: 0.8649 - loss: 0.3045

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 309ms/step - acc: 0.8660 - loss: 0.3019

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 308ms/step - acc: 0.8662 - loss: 0.3015

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 308ms/step - acc: 0.8663 - loss: 0.3012

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 309ms/step - acc: 0.8663 - loss: 0.3011

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 312ms/step - acc: 0.8684 - loss: 0.2960

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 336ms/step - acc: 0.8684 - loss: 0.2959 - val_acc: 0.7479 - val_loss: 0.4058
Epoch 6/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  22s 306ms/step - acc: 0.8914 - loss: 0.2597

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 308ms/step - acc: 0.8914 - loss: 0.2589

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 308ms/step - acc: 0.8915 - loss: 0.2573

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 308ms/step - acc: 0.8915 - loss: 0.2571

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 308ms/step - acc: 0.8915 - loss: 0.2569

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 309ms/step - acc: 0.8916 - loss: 0.2567

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 308ms/step - acc: 0.8922 - loss: 0.2537

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 336ms/step - acc: 0.8922 - loss: 0.2537 - val_acc: 0.6472 - val_loss: 0.7482
Epoch 7/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 311ms/step - acc: 0.8952 - loss: 0.2339

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 312ms/step - acc: 0.8954 - loss: 0.2334

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 311ms/step - acc: 0.8963 - loss: 0.2319

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 313ms/step - acc: 0.8964 - loss: 0.2317

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 314ms/step - acc: 0.8965 - loss: 0.2314

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 313ms/step - acc: 0.8966 - loss: 0.2314

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 311ms/step - acc: 0.8980 - loss: 0.2286

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 335ms/step - acc: 0.8980 - loss: 0.2286 - val_acc: 0.8988 - val_loss: 0.2326
Epoch 8/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 321ms/step - acc: 0.9172 - loss: 0.2058

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 322ms/step - acc: 0.9170 - loss: 0.2051

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 320ms/step - acc: 0.9164 - loss: 0.2039

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 319ms/step - acc: 0.9163 - loss: 0.2038

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 320ms/step - acc: 0.9163 - loss: 0.2037

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 319ms/step - acc: 0.9162 - loss: 0.2036

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9154 - loss: 0.2022

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 340ms/step - acc: 0.9154 - loss: 0.2022 - val_acc: 0.8865 - val_loss: 0.2269
Epoch 9/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 326ms/step - acc: 0.9174 - loss: 0.1947

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  22s 325ms/step - acc: 0.9177 - loss: 0.1936

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 322ms/step - acc: 0.9183 - loss: 0.1919

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 324ms/step - acc: 0.9184 - loss: 0.1918

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 324ms/step - acc: 0.9184 - loss: 0.1916

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 323ms/step - acc: 0.9185 - loss: 0.1915

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 317ms/step - acc: 0.9193 - loss: 0.1895

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 341ms/step - acc: 0.9193 - loss: 0.1894 - val_acc: 0.9149 - val_loss: 0.2096
Epoch 10/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 318ms/step - acc: 0.9235 - loss: 0.1772

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  22s 324ms/step - acc: 0.9235 - loss: 0.1771

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 323ms/step - acc: 0.9234 - loss: 0.1771

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 323ms/step - acc: 0.9234 - loss: 0.1771

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 324ms/step - acc: 0.9233 - loss: 0.1771

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 323ms/step - acc: 0.9233 - loss: 0.1771

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 318ms/step - acc: 0.9232 - loss: 0.1768

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 52s 342ms/step - acc: 0.9232 - loss: 0.1768 - val_acc: 0.9207 - val_loss: 0.1853
Epoch 11/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 315ms/step - acc: 0.9313 - loss: 0.1647

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 316ms/step - acc: 0.9315 - loss: 0.1642

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 315ms/step - acc: 0.9320 - loss: 0.1632

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 317ms/step - acc: 0.9320 - loss: 0.1631

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 318ms/step - acc: 0.9320 - loss: 0.1630

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 320ms/step - acc: 0.9321 - loss: 0.1629

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9323 - loss: 0.1615

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 341ms/step - acc: 0.9324 - loss: 0.1615 - val_acc: 0.9222 - val_loss: 0.1837
Epoch 12/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 317ms/step - acc: 0.9372 - loss: 0.1513

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 316ms/step - acc: 0.9375 - loss: 0.1507

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 315ms/step - acc: 0.9377 - loss: 0.1500

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 316ms/step - acc: 0.9377 - loss: 0.1500

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 315ms/step - acc: 0.9377 - loss: 0.1499

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 315ms/step - acc: 0.9377 - loss: 0.1499

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 318ms/step - acc: 0.9377 - loss: 0.1492

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 52s 342ms/step - acc: 0.9377 - loss: 0.1491 - val_acc: 0.8970 - val_loss: 0.2143
Epoch 13/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 314ms/step - acc: 0.9423 - loss: 0.1337

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 312ms/step - acc: 0.9424 - loss: 0.1340

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 314ms/step - acc: 0.9423 - loss: 0.1349

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 314ms/step - acc: 0.9423 - loss: 0.1350

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 313ms/step - acc: 0.9422 - loss: 0.1352

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 313ms/step - acc: 0.9422 - loss: 0.1353

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9414 - loss: 0.1372

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 53s 353ms/step - acc: 0.9414 - loss: 0.1373 - val_acc: 0.8986 - val_loss: 0.2395
Epoch 14/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 320ms/step - acc: 0.9414 - loss: 0.1373

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 319ms/step - acc: 0.9416 - loss: 0.1368

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 317ms/step - acc: 0.9418 - loss: 0.1367

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 316ms/step - acc: 0.9418 - loss: 0.1367

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 316ms/step - acc: 0.9418 - loss: 0.1367

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 316ms/step - acc: 0.9418 - loss: 0.1366

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 312ms/step - acc: 0.9421 - loss: 0.1358

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 336ms/step - acc: 0.9421 - loss: 0.1358 - val_acc: 0.8593 - val_loss: 0.3821
Epoch 15/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 325ms/step - acc: 0.9465 - loss: 0.1363

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  22s 326ms/step - acc: 0.9467 - loss: 0.1357

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 323ms/step - acc: 0.9469 - loss: 0.1347

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 322ms/step - acc: 0.9468 - loss: 0.1347

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 322ms/step - acc: 0.9468 - loss: 0.1346

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 321ms/step - acc: 0.9468 - loss: 0.1346

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9465 - loss: 0.1334

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 340ms/step - acc: 0.9465 - loss: 0.1333 - val_acc: 0.9310 - val_loss: 0.1730
Epoch 16/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 319ms/step - acc: 0.9506 - loss: 0.1212

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 318ms/step - acc: 0.9501 - loss: 0.1220

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 315ms/step - acc: 0.9490 - loss: 0.1241

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 315ms/step - acc: 0.9489 - loss: 0.1243

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 314ms/step - acc: 0.9487 - loss: 0.1246

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 314ms/step - acc: 0.9487 - loss: 0.1248

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 311ms/step - acc: 0.9469 - loss: 0.1280

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 335ms/step - acc: 0.9469 - loss: 0.1280 - val_acc: 0.9192 - val_loss: 0.1835
Epoch 17/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 307ms/step - acc: 0.9521 - loss: 0.1236

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 311ms/step - acc: 0.9522 - loss: 0.1230

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 316ms/step - acc: 0.9522 - loss: 0.1224

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 315ms/step - acc: 0.9522 - loss: 0.1224

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 103/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 315ms/step - acc: 0.9522 - loss: 0.1224

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 314ms/step - acc: 0.9522 - loss: 0.1223

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 311ms/step - acc: 0.9524 - loss: 0.1214

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 335ms/step - acc: 0.9524 - loss: 0.1214 - val_acc: 0.9360 - val_loss: 0.1764
Epoch 18/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 313ms/step - acc: 0.9520 - loss: 0.1121

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 314ms/step - acc: 0.9521 - loss: 0.1120

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 313ms/step - acc: 0.9524 - loss: 0.1118

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 314ms/step - acc: 0.9525 - loss: 0.1119

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 314ms/step - acc: 0.9525 - loss: 0.1119

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 316ms/step - acc: 0.9525 - loss: 0.1119

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9526 - loss: 0.1123

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 340ms/step - acc: 0.9526 - loss: 0.1123 - val_acc: 0.9307 - val_loss: 0.1950
Epoch 19/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 307ms/step - acc: 0.9511 - loss: 0.1081

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 306ms/step - acc: 0.9510 - loss: 0.1086

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 307ms/step - acc: 0.9509 - loss: 0.1089

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 307ms/step - acc: 0.9509 - loss: 0.1089

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 306ms/step - acc: 0.9509 - loss: 0.1089

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 306ms/step - acc: 0.9509 - loss: 0.1089

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 307ms/step - acc: 0.9512 - loss: 0.1091

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 331ms/step - acc: 0.9512 - loss: 0.1091 - val_acc: 0.9296 - val_loss: 0.1911
Epoch 20/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  22s 306ms/step - acc: 0.9495 - loss: 0.1230

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 308ms/step - acc: 0.9502 - loss: 0.1214

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 309ms/step - acc: 0.9515 - loss: 0.1182

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 309ms/step - acc: 0.9517 - loss: 0.1178

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 308ms/step - acc: 0.9518 - loss: 0.1174

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 308ms/step - acc: 0.9520 - loss: 0.1172

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 307ms/step - acc: 0.9535 - loss: 0.1135

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 331ms/step - acc: 0.9535 - loss: 0.1134 - val_acc: 0.9299 - val_loss: 0.1566
Epoch 21/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 314ms/step - acc: 0.9569 - loss: 0.1018

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 314ms/step - acc: 0.9571 - loss: 0.1017

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 311ms/step - acc: 0.9574 - loss: 0.1015

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 311ms/step - acc: 0.9574 - loss: 0.1015

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 103/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 313ms/step - acc: 0.9574 - loss: 0.1015

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 312ms/step - acc: 0.9574 - loss: 0.1016

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 307ms/step - acc: 0.9575 - loss: 0.1018

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 331ms/step - acc: 0.9575 - loss: 0.1018 - val_acc: 0.9416 - val_loss: 0.1651
Epoch 22/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  71/147 ━━━━━━━━━[37m━━━━━━━━━━━  24s 323ms/step - acc: 0.9576 - loss: 0.1048

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  78/147 ━━━━━━━━━━[37m━━━━━━━━━━  22s 323ms/step - acc: 0.9578 - loss: 0.1044

Warning: unknown JFIF revision number 0.00

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 321ms/step - acc: 0.9583 - loss: 0.1036

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 100/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 320ms/step - acc: 0.9583 - loss: 0.1035

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 103/147 ━━━━━━━━━━━━━━[37m━━━━━━  14s 320ms/step - acc: 0.9584 - loss: 0.1034

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 320ms/step - acc: 0.9584 - loss: 0.1034

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 314ms/step - acc: 0.9588 - loss: 0.1023

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 338ms/step - acc: 0.9588 - loss: 0.1023 - val_acc: 0.9034 - val_loss: 0.2785
Epoch 23/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 317ms/step - acc: 0.9637 - loss: 0.0911

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  21s 322ms/step - acc: 0.9637 - loss: 0.0911

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  15s 320ms/step - acc: 0.9637 - loss: 0.0914

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 319ms/step - acc: 0.9637 - loss: 0.0914

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 319ms/step - acc: 0.9637 - loss: 0.0915

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 318ms/step - acc: 0.9637 - loss: 0.0915

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - acc: 0.9637 - loss: 0.0918

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 51s 339ms/step - acc: 0.9637 - loss: 0.0918 - val_acc: 0.9439 - val_loss: 0.1511
Epoch 24/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  22s 302ms/step - acc: 0.9567 - loss: 0.1011

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 302ms/step - acc: 0.9572 - loss: 0.1004

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 308ms/step - acc: 0.9581 - loss: 0.0990

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 102/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 308ms/step - acc: 0.9582 - loss: 0.0989

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 104/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 310ms/step - acc: 0.9583 - loss: 0.0989

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 311ms/step - acc: 0.9584 - loss: 0.0988

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 307ms/step - acc: 0.9593 - loss: 0.0973

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 50s 331ms/step - acc: 0.9593 - loss: 0.0973 - val_acc: 0.9412 - val_loss: 0.1615
Epoch 25/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  23s 308ms/step - acc: 0.9645 - loss: 0.0891

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  79/147 ━━━━━━━━━━[37m━━━━━━━━━━  20s 308ms/step - acc: 0.9644 - loss: 0.0892

Warning: unknown JFIF revision number 0.00

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 308ms/step - acc: 0.9643 - loss: 0.0895

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 101/147 ━━━━━━━━━━━━━[37m━━━━━━━  14s 308ms/step - acc: 0.9643 - loss: 0.0895

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 105/147 ━━━━━━━━━━━━━━[37m━━━━━━  13s 310ms/step - acc: 0.9642 - loss: 0.0895

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 106/147 ━━━━━━━━━━━━━━[37m━━━━━━  12s 311ms/step - acc: 0.9642 - loss: 0.0896

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 330ms/step - acc: 0.9638 - loss: 0.0902

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 53s 354ms/step - acc: 0.9638 - loss: 0.0903 - val_acc: 0.9382 - val_loss: 0.1542

<keras.src.callbacks.history.History at 0x7f41003c24a0>

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
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
This image is 94.30% cat and 5.70% dog.

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_29_1.png)
    

