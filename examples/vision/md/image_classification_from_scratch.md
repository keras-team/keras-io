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
100  786M  100  786M    0     0   110M      0  0:00:07  0:00:07 --:--:--  123M

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
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
```

<div class="k-default-codeblock">
```
You must install pydot (`pip install pydot`) for `plot_model` to work.

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
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
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
I0000 00:00:1699570868.090230  724774 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5557 - loss: 0.7543

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5557 - loss: 0.7537

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5569 - loss: 0.7513

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5572 - loss: 0.7509

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5575 - loss: 0.7504

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5576 - loss: 0.7502

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 538ms/step - accuracy: 0.5596 - loss: 0.7623

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 215s 722ms/step - accuracy: 0.5596 - loss: 0.7629 - val_accuracy: 0.4958 - val_loss: 4.9329
Epoch 2/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.4893 - loss: 0.7471

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.4909 - loss: 0.7449

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.4935 - loss: 0.7405

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.4939 - loss: 0.7399

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.4944 - loss: 0.7393

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.4945 - loss: 0.7391

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.4995 - loss: 0.7318

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.4996 - loss: 0.7316 - val_accuracy: 0.5042 - val_loss: 0.7861
Epoch 3/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5250 - loss: 0.6978

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5244 - loss: 0.6980

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5236 - loss: 0.6985

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5235 - loss: 0.6985

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5235 - loss: 0.6986

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5235 - loss: 0.6986

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5236 - loss: 0.6987

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5236 - loss: 0.6987 - val_accuracy: 0.5248 - val_loss: 0.6897
Epoch 4/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5101 - loss: 0.7010

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5112 - loss: 0.7010

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5130 - loss: 0.7010

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5134 - loss: 0.7009

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5137 - loss: 0.7009

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5138 - loss: 0.7008

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5176 - loss: 0.7000

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5176 - loss: 0.6999 - val_accuracy: 0.5544 - val_loss: 0.6881
Epoch 5/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5255 - loss: 0.6984

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5254 - loss: 0.6980

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5258 - loss: 0.6973

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5259 - loss: 0.6972

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5261 - loss: 0.6971

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5261 - loss: 0.6971

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5270 - loss: 0.6962

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5270 - loss: 0.6961 - val_accuracy: 0.5581 - val_loss: 0.6855
Epoch 6/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5340 - loss: 0.6934

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5336 - loss: 0.6934

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5339 - loss: 0.6932

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5340 - loss: 0.6932

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5340 - loss: 0.6932

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5340 - loss: 0.6932

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5344 - loss: 0.6932

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5344 - loss: 0.6932 - val_accuracy: 0.5541 - val_loss: 0.6865
Epoch 7/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5324 - loss: 0.6911

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5320 - loss: 0.6912

Warning: unknown JFIF revision number 0.00

  92/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5324 - loss: 0.6912

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5325 - loss: 0.6912

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5327 - loss: 0.6912

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5327 - loss: 0.6912

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5341 - loss: 0.6913

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5341 - loss: 0.6913 - val_accuracy: 0.5478 - val_loss: 0.6958
Epoch 8/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5396 - loss: 0.6902

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5391 - loss: 0.6903

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5394 - loss: 0.6903

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5395 - loss: 0.6902

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5396 - loss: 0.6902

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5396 - loss: 0.6902

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5391 - loss: 0.6904

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5390 - loss: 0.6904 - val_accuracy: 0.5424 - val_loss: 0.7191
Epoch 9/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5535 - loss: 0.6883

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5524 - loss: 0.6884

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5511 - loss: 0.6886

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5510 - loss: 0.6886

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5509 - loss: 0.6886

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5509 - loss: 0.6886

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5504 - loss: 0.6885

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5504 - loss: 0.6885 - val_accuracy: 0.5548 - val_loss: 0.6924
Epoch 10/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5400 - loss: 0.6877

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5406 - loss: 0.6878

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5419 - loss: 0.6878

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5421 - loss: 0.6878

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5423 - loss: 0.6878

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5423 - loss: 0.6878

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 275ms/step - accuracy: 0.5434 - loss: 0.6880

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5434 - loss: 0.6880 - val_accuracy: 0.5695 - val_loss: 0.6789
Epoch 11/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5446 - loss: 0.6911

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5445 - loss: 0.6908

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5453 - loss: 0.6901

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5455 - loss: 0.6899

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5457 - loss: 0.6898

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5458 - loss: 0.6898

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5481 - loss: 0.6887

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5481 - loss: 0.6887 - val_accuracy: 0.5654 - val_loss: 0.6817
Epoch 12/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5713 - loss: 0.6817

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5702 - loss: 0.6818

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5686 - loss: 0.6817

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5685 - loss: 0.6817

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5684 - loss: 0.6817

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5683 - loss: 0.6817

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5658 - loss: 0.6818

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5657 - loss: 0.6818 - val_accuracy: 0.5735 - val_loss: 0.6757
Epoch 13/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5562 - loss: 0.6825

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5561 - loss: 0.6826

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5567 - loss: 0.6825

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5568 - loss: 0.6824

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5569 - loss: 0.6824

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5570 - loss: 0.6824

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5576 - loss: 0.6824

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5576 - loss: 0.6824 - val_accuracy: 0.5674 - val_loss: 0.6982
Epoch 14/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5650 - loss: 0.6798

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5636 - loss: 0.6805

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5618 - loss: 0.6815

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5616 - loss: 0.6816

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5614 - loss: 0.6817

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5614 - loss: 0.6818

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5590 - loss: 0.6832

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5589 - loss: 0.6832 - val_accuracy: 0.5810 - val_loss: 0.6778
Epoch 15/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5582 - loss: 0.6827

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5583 - loss: 0.6827

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5592 - loss: 0.6824

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5594 - loss: 0.6824

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5595 - loss: 0.6823

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5596 - loss: 0.6823

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 275ms/step - accuracy: 0.5609 - loss: 0.6821

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5609 - loss: 0.6821 - val_accuracy: 0.5715 - val_loss: 0.6940
Epoch 16/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5618 - loss: 0.6799

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5608 - loss: 0.6802

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5597 - loss: 0.6805

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5597 - loss: 0.6805

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5597 - loss: 0.6805

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5597 - loss: 0.6805

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5589 - loss: 0.6806

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5589 - loss: 0.6806 - val_accuracy: 0.5847 - val_loss: 0.6806
Epoch 17/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5679 - loss: 0.6749

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5682 - loss: 0.6751

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5689 - loss: 0.6752

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5690 - loss: 0.6752

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5690 - loss: 0.6752

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5691 - loss: 0.6752

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5696 - loss: 0.6752

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5696 - loss: 0.6752 - val_accuracy: 0.5902 - val_loss: 0.6713
Epoch 18/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5606 - loss: 0.6727

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5608 - loss: 0.6730

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5614 - loss: 0.6735

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5615 - loss: 0.6735

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5616 - loss: 0.6736

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5616 - loss: 0.6736

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5630 - loss: 0.6742

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5630 - loss: 0.6742 - val_accuracy: 0.5648 - val_loss: 0.6925
Epoch 19/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5641 - loss: 0.6737

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5634 - loss: 0.6743

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5629 - loss: 0.6748

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5629 - loss: 0.6748

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5629 - loss: 0.6749

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  99/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5629 - loss: 0.6749

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step - accuracy: 0.5631 - loss: 0.6754

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5631 - loss: 0.6755 - val_accuracy: 0.5730 - val_loss: 0.6753
Epoch 20/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5671 - loss: 0.6711

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5669 - loss: 0.6713

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5672 - loss: 0.6716

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 276ms/step - accuracy: 0.5673 - loss: 0.6716

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5674 - loss: 0.6717

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 276ms/step - accuracy: 0.5674 - loss: 0.6717

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5680 - loss: 0.6724

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 297ms/step - accuracy: 0.5680 - loss: 0.6724 - val_accuracy: 0.5969 - val_loss: 0.6603
Epoch 21/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5652 - loss: 0.6745

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5662 - loss: 0.6743

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5676 - loss: 0.6737

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5678 - loss: 0.6736

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5680 - loss: 0.6735

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5680 - loss: 0.6735

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5698 - loss: 0.6727

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5698 - loss: 0.6727 - val_accuracy: 0.5701 - val_loss: 0.6833
Epoch 22/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5807 - loss: 0.6624

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5807 - loss: 0.6628

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5810 - loss: 0.6633

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5811 - loss: 0.6633

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5812 - loss: 0.6634

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5812 - loss: 0.6634

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5809 - loss: 0.6644

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5809 - loss: 0.6644 - val_accuracy: 0.5822 - val_loss: 0.6679
Epoch 23/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5740 - loss: 0.6697

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5747 - loss: 0.6696

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5760 - loss: 0.6692

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5763 - loss: 0.6691

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5765 - loss: 0.6690

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5766 - loss: 0.6689

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step - accuracy: 0.5790 - loss: 0.6680

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5790 - loss: 0.6680 - val_accuracy: 0.5931 - val_loss: 0.6681
Epoch 24/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 275ms/step - accuracy: 0.5787 - loss: 0.6622

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 275ms/step - accuracy: 0.5792 - loss: 0.6623

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 275ms/step - accuracy: 0.5804 - loss: 0.6624

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5806 - loss: 0.6624

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5807 - loss: 0.6623

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5807 - loss: 0.6623

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step - accuracy: 0.5817 - loss: 0.6627

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5817 - loss: 0.6627 - val_accuracy: 0.5735 - val_loss: 0.6780
Epoch 25/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

  64/147 ━━━━━━━━[37m━━━━━━━━━━━━  22s 276ms/step - accuracy: 0.5832 - loss: 0.6633

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

  72/147 ━━━━━━━━━[37m━━━━━━━━━━━  20s 276ms/step - accuracy: 0.5831 - loss: 0.6632

Warning: unknown JFIF revision number 0.00

  91/147 ━━━━━━━━━━━━[37m━━━━━━━━  15s 276ms/step - accuracy: 0.5828 - loss: 0.6630

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

  94/147 ━━━━━━━━━━━━[37m━━━━━━━━  14s 275ms/step - accuracy: 0.5827 - loss: 0.6630

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

  97/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5826 - loss: 0.6630

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

  98/147 ━━━━━━━━━━━━━[37m━━━━━━━  13s 275ms/step - accuracy: 0.5826 - loss: 0.6630

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.5816 - loss: 0.6631

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

 147/147 ━━━━━━━━━━━━━━━━━━━━ 45s 296ms/step - accuracy: 0.5816 - loss: 0.6631 - val_accuracy: 0.5752 - val_loss: 0.6733

<keras.src.callbacks.history.History at 0x7fe29846df90>

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
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
This image is 57.10% cat and 42.90% dog.

/var/tmp/ipykernel_724616/2552698545.py:8: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  score = float(predictions[0])

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_29_2.png)
    

