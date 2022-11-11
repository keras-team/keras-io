# Image classification from scratch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/27<br>
**Last modified:** 2020/04/28<br>
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
100  786M  100  786M    0     0   182M      0  0:00:04  0:00:04 --:--:--  195M

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
import os

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)
```

<div class="k-default-codeblock">
```
Deleted 1590 images

```
</div>
---
## Generate a `Dataset`


```python
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
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

Here are the first 9 images in the training dataset. As you can see, label 1 is "dog"
and label 0 is "cat".


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
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
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
```

Let's visualize what the augmented samples look like, by applying `data_augmentation`
repeatedly to the first image in the dataset:


```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
```

<div class="k-default-codeblock">
```
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:5 out of the last 5 calls to <function pfor.<locals>.f at 0x7fd3971aed08> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:6 out of the last 6 calls to <function pfor.<locals>.f at 0x7fd397088d90> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.

```
</div>
    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_18_2.png)
    


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
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.

```
</div>
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
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.

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

 63/147 [===========>..................] - ETA: 57s - loss: 0.7032 - accuracy: 0.6023

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.6990 - accuracy: 0.6043

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.6853 - accuracy: 0.6146

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.6830 - accuracy: 0.6153

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.6811 - accuracy: 0.6164

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.6799 - accuracy: 0.6173

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.6531 - accuracy: 0.6416

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 116s 746ms/step - loss: 0.6531 - accuracy: 0.6416 - val_loss: 0.7669 - val_accuracy: 0.4957
Epoch 2/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 58s - loss: 0.5448 - accuracy: 0.7278

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 70/147 [=============>................] - ETA: 53s - loss: 0.5424 - accuracy: 0.7302

Warning: unknown JFIF revision number 0.00

 89/147 [=================>............] - ETA: 39s - loss: 0.5335 - accuracy: 0.7381

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 92/147 [=================>............] - ETA: 37s - loss: 0.5313 - accuracy: 0.7392

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 35s - loss: 0.5294 - accuracy: 0.7398

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 35s - loss: 0.5280 - accuracy: 0.7407

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.5026 - accuracy: 0.7559

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 737ms/step - loss: 0.5026 - accuracy: 0.7559 - val_loss: 1.3825 - val_accuracy: 0.4957
Epoch 3/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 62/147 [===========>..................] - ETA: 59s - loss: 0.4289 - accuracy: 0.8044 

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 70/147 [=============>................] - ETA: 53s - loss: 0.4267 - accuracy: 0.8056

Warning: unknown JFIF revision number 0.00

 87/147 [================>.............] - ETA: 41s - loss: 0.4216 - accuracy: 0.8076

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 91/147 [=================>............] - ETA: 38s - loss: 0.4197 - accuracy: 0.8088

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.4177 - accuracy: 0.8097

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 35s - loss: 0.4174 - accuracy: 0.8100

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.3928 - accuracy: 0.8243

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 738ms/step - loss: 0.3928 - accuracy: 0.8243 - val_loss: 1.6816 - val_accuracy: 0.4957
Epoch 4/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 58s - loss: 0.3434 - accuracy: 0.8516

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.3396 - accuracy: 0.8546

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.3425 - accuracy: 0.8536

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.3412 - accuracy: 0.8543

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 35s - loss: 0.3399 - accuracy: 0.8547

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.3393 - accuracy: 0.8551

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.3307 - accuracy: 0.8588

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 736ms/step - loss: 0.3307 - accuracy: 0.8588 - val_loss: 0.5025 - val_accuracy: 0.7520
Epoch 5/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.2900 - accuracy: 0.8806

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.2880 - accuracy: 0.8814

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.2825 - accuracy: 0.8844

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 36s - loss: 0.2822 - accuracy: 0.8845

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.2821 - accuracy: 0.8842

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.2819 - accuracy: 0.8843

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.2758 - accuracy: 0.8860

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.2758 - accuracy: 0.8860 - val_loss: 0.3462 - val_accuracy: 0.8545
Epoch 6/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 57s - loss: 0.2504 - accuracy: 0.8968

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.2503 - accuracy: 0.8972

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.2458 - accuracy: 0.8993

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.2455 - accuracy: 0.8995

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.2445 - accuracy: 0.9001

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.2446 - accuracy: 0.9000

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.2357 - accuracy: 0.9023

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.2357 - accuracy: 0.9023 - val_loss: 0.2712 - val_accuracy: 0.8825
Epoch 7/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.1999 - accuracy: 0.9186

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.2006 - accuracy: 0.9190

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.2015 - accuracy: 0.9200

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.2010 - accuracy: 0.9200

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.2011 - accuracy: 0.9202

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.2000 - accuracy: 0.9208

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.2011 - accuracy: 0.9201

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.2011 - accuracy: 0.9201 - val_loss: 0.2131 - val_accuracy: 0.9135
Epoch 8/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.1880 - accuracy: 0.9266

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.1831 - accuracy: 0.9285

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.1783 - accuracy: 0.9295

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.1796 - accuracy: 0.9288

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1797 - accuracy: 0.9281

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.1800 - accuracy: 0.9282

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1787 - accuracy: 0.9275

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.1787 - accuracy: 0.9275 - val_loss: 0.1969 - val_accuracy: 0.9227
Epoch 9/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 56s - loss: 0.1702 - accuracy: 0.9309

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.1682 - accuracy: 0.9315

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.1670 - accuracy: 0.9317

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 35s - loss: 0.1665 - accuracy: 0.9321

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.1662 - accuracy: 0.9325

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1650 - accuracy: 0.9321

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.1650 - accuracy: 0.9321 - val_loss: 0.2306 - val_accuracy: 0.9178
Epoch 10/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.1588 - accuracy: 0.9360

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.1553 - accuracy: 0.9384

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.1495 - accuracy: 0.9396

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 36s - loss: 0.1496 - accuracy: 0.9395

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.1489 - accuracy: 0.9396

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1492 - accuracy: 0.9396

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1474 - accuracy: 0.9408

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.1474 - accuracy: 0.9408 - val_loss: 0.2430 - val_accuracy: 0.9107
Epoch 11/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 56s - loss: 0.1383 - accuracy: 0.9446

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 50s - loss: 0.1350 - accuracy: 0.9468

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 37s - loss: 0.1337 - accuracy: 0.9468

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 35s - loss: 0.1356 - accuracy: 0.9456

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 32s - loss: 0.1358 - accuracy: 0.9459

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1352 - accuracy: 0.9461

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.1352 - accuracy: 0.9461 - val_loss: 0.2783 - val_accuracy: 0.8768
Epoch 12/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 56s - loss: 0.1236 - accuracy: 0.9514

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 50s - loss: 0.1225 - accuracy: 0.9521

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 37s - loss: 0.1259 - accuracy: 0.9494

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.1254 - accuracy: 0.9495

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1255 - accuracy: 0.9493

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.1251 - accuracy: 0.9495

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1291 - accuracy: 0.9474

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.1291 - accuracy: 0.9474 - val_loss: 0.4632 - val_accuracy: 0.8419
Epoch 13/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 58s - loss: 0.1235 - accuracy: 0.9499

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.1257 - accuracy: 0.9491

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.1295 - accuracy: 0.9477

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.1299 - accuracy: 0.9475

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.1299 - accuracy: 0.9478

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1295 - accuracy: 0.9481

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1208 - accuracy: 0.9521

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.1208 - accuracy: 0.9521 - val_loss: 0.3907 - val_accuracy: 0.8456
Epoch 14/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 58s - loss: 0.1221 - accuracy: 0.9530

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 52s - loss: 0.1212 - accuracy: 0.9528

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.1200 - accuracy: 0.9529

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.1192 - accuracy: 0.9533

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 35s - loss: 0.1192 - accuracy: 0.9535

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1185 - accuracy: 0.9539

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1162 - accuracy: 0.9553

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 110s 739ms/step - loss: 0.1162 - accuracy: 0.9553 - val_loss: 0.1503 - val_accuracy: 0.9417
Epoch 15/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 56s - loss: 0.1072 - accuracy: 0.9595

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.1068 - accuracy: 0.9592

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.1063 - accuracy: 0.9587

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 35s - loss: 0.1067 - accuracy: 0.9586

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1068 - accuracy: 0.9588

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.1066 - accuracy: 0.9587

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1037 - accuracy: 0.9598

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.1037 - accuracy: 0.9598 - val_loss: 0.1484 - val_accuracy: 0.9406
Epoch 16/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 57s - loss: 0.1037 - accuracy: 0.9606

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.1036 - accuracy: 0.9606

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.1028 - accuracy: 0.9604

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 36s - loss: 0.1025 - accuracy: 0.9607

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.1022 - accuracy: 0.9608

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.1022 - accuracy: 0.9609

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1018 - accuracy: 0.9605

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.1018 - accuracy: 0.9605 - val_loss: 0.2480 - val_accuracy: 0.9054
Epoch 17/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 57s - loss: 0.1064 - accuracy: 0.9585

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 74/147 [==============>...............] - ETA: 50s - loss: 0.1047 - accuracy: 0.9586

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.1017 - accuracy: 0.9595

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.1003 - accuracy: 0.9599

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 35s - loss: 0.0998 - accuracy: 0.9600

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0995 - accuracy: 0.9602

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0949 - accuracy: 0.9629

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 739ms/step - loss: 0.0949 - accuracy: 0.9629 - val_loss: 0.1585 - val_accuracy: 0.9378
Epoch 18/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 61/147 [===========>..................] - ETA: 59s - loss: 0.0983 - accuracy: 0.9590 

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 69/147 [=============>................] - ETA: 54s - loss: 0.0958 - accuracy: 0.9609

Warning: unknown JFIF revision number 0.00

 86/147 [================>.............] - ETA: 42s - loss: 0.0935 - accuracy: 0.9626

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 90/147 [=================>............] - ETA: 39s - loss: 0.0934 - accuracy: 0.9626

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 92/147 [=================>............] - ETA: 37s - loss: 0.0934 - accuracy: 0.9626

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.0933 - accuracy: 0.9626

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0941 - accuracy: 0.9622

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 736ms/step - loss: 0.0941 - accuracy: 0.9622 - val_loss: 0.1452 - val_accuracy: 0.9432
Epoch 19/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.0802 - accuracy: 0.9684

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.0806 - accuracy: 0.9686

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.0813 - accuracy: 0.9686

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.0826 - accuracy: 0.9679

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0830 - accuracy: 0.9676

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0862 - accuracy: 0.9668

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.0862 - accuracy: 0.9668 - val_loss: 0.2644 - val_accuracy: 0.8904
Epoch 20/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 57s - loss: 0.1006 - accuracy: 0.9608

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.0997 - accuracy: 0.9612

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.0962 - accuracy: 0.9628

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 36s - loss: 0.0964 - accuracy: 0.9626

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.0965 - accuracy: 0.9624

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0963 - accuracy: 0.9625

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0889 - accuracy: 0.9656

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.0889 - accuracy: 0.9656 - val_loss: 0.2335 - val_accuracy: 0.9182
Epoch 21/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 57s - loss: 0.0838 - accuracy: 0.9679

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.0802 - accuracy: 0.9692

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.0810 - accuracy: 0.9684

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.0804 - accuracy: 0.9685

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.0814 - accuracy: 0.9679

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0809 - accuracy: 0.9682

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0792 - accuracy: 0.9687

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.0792 - accuracy: 0.9687 - val_loss: 0.5037 - val_accuracy: 0.8751
Epoch 22/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 66/147 [============>.................] - ETA: 55s - loss: 0.0685 - accuracy: 0.9742

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 50s - loss: 0.0671 - accuracy: 0.9745

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 38s - loss: 0.0690 - accuracy: 0.9731

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.0689 - accuracy: 0.9730

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0685 - accuracy: 0.9729

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.0684 - accuracy: 0.9730

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0651 - accuracy: 0.9737

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 734ms/step - loss: 0.0651 - accuracy: 0.9737 - val_loss: 0.1103 - val_accuracy: 0.9551
Epoch 23/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 58s - loss: 0.0706 - accuracy: 0.9720

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.0684 - accuracy: 0.9725

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.0659 - accuracy: 0.9735

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.0656 - accuracy: 0.9736

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0653 - accuracy: 0.9738

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 32s - loss: 0.0652 - accuracy: 0.9739

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0641 - accuracy: 0.9751

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.0641 - accuracy: 0.9751 - val_loss: 0.1846 - val_accuracy: 0.9299
Epoch 24/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 56s - loss: 0.0719 - accuracy: 0.9739

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 51s - loss: 0.0721 - accuracy: 0.9735

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.0725 - accuracy: 0.9730

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 94/147 [==================>...........] - ETA: 36s - loss: 0.0734 - accuracy: 0.9726

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 34s - loss: 0.0730 - accuracy: 0.9728

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.0734 - accuracy: 0.9727

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0709 - accuracy: 0.9735

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 735ms/step - loss: 0.0709 - accuracy: 0.9735 - val_loss: 0.1151 - val_accuracy: 0.9575
Epoch 25/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 63/147 [===========>..................] - ETA: 58s - loss: 0.0589 - accuracy: 0.9779

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 71/147 [=============>................] - ETA: 52s - loss: 0.0575 - accuracy: 0.9781

Warning: unknown JFIF revision number 0.00

 90/147 [=================>............] - ETA: 39s - loss: 0.0589 - accuracy: 0.9774

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 93/147 [=================>............] - ETA: 37s - loss: 0.0589 - accuracy: 0.9775

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 34s - loss: 0.0591 - accuracy: 0.9774

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 33s - loss: 0.0593 - accuracy: 0.9774

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0612 - accuracy: 0.9768

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 109s 737ms/step - loss: 0.0612 - accuracy: 0.9768 - val_loss: 0.4759 - val_accuracy: 0.8710

<keras.callbacks.History at 0x7fd3941c87b8>

```
</div>
We get to ~96% validation accuracy after training for 25 epochs on the full dataset.

---
## Run inference on new data

Note that data augmentation and dropout are inactive at inference time.


```python
img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 446ms/step
This image is 45.28% cat and 54.72% dog.

```
</div>