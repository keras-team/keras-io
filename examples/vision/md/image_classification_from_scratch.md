# Image classification from scratch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/27<br>
**Last modified:** 2022/11/10<br>
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
100  786M  100  786M    0     0  53.3M      0  0:00:14  0:00:14 --:--:-- 53.9M

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
WARNING:tensorflow:5 out of the last 5 calls to <function pfor.<locals>.f at 0x7f33c90f2598> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:6 out of the last 6 calls to <function pfor.<locals>.f at 0x7f33c90817b8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
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




    
![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_24_0.png)
    



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

 66/147 [============>.................] - ETA: 1:04 - loss: 0.6949 - accuracy: 0.6035

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 74/147 [==============>...............] - ETA: 58s - loss: 0.6922 - accuracy: 0.6054

Warning: unknown JFIF revision number 0.00

 94/147 [==================>...........] - ETA: 42s - loss: 0.6788 - accuracy: 0.6150

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.6770 - accuracy: 0.6166

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.6449 - accuracy: 0.6453

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 146s 876ms/step - loss: 0.6449 - accuracy: 0.6453 - val_loss: 0.8450 - val_accuracy: 0.4957
Epoch 2/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 66/147 [============>.................] - ETA: 1:05 - loss: 0.5329 - accuracy: 0.7356

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 74/147 [==============>...............] - ETA: 59s - loss: 0.5289 - accuracy: 0.7382 

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.5154 - accuracy: 0.7486

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 42s - loss: 0.5155 - accuracy: 0.7491

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 39s - loss: 0.5118 - accuracy: 0.7519

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 38s - loss: 0.5114 - accuracy: 0.7523

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.4923 - accuracy: 0.7631

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 129s 867ms/step - loss: 0.4923 - accuracy: 0.7631 - val_loss: 1.2116 - val_accuracy: 0.4957
Epoch 3/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 66/147 [============>.................] - ETA: 1:05 - loss: 0.4027 - accuracy: 0.8182

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 75/147 [==============>...............] - ETA: 58s - loss: 0.4018 - accuracy: 0.8192

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.3952 - accuracy: 0.8227

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 42s - loss: 0.3936 - accuracy: 0.8237

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 39s - loss: 0.3919 - accuracy: 0.8250

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.3792 - accuracy: 0.8326

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 129s 868ms/step - loss: 0.3792 - accuracy: 0.8326 - val_loss: 1.2661 - val_accuracy: 0.4957
Epoch 4/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:06 - loss: 0.3351 - accuracy: 0.8566

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 74/147 [==============>...............] - ETA: 59s - loss: 0.3311 - accuracy: 0.8592 

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.3325 - accuracy: 0.8585

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.3323 - accuracy: 0.8581

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 38s - loss: 0.3305 - accuracy: 0.8589

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.3171 - accuracy: 0.8652

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 128s 862ms/step - loss: 0.3171 - accuracy: 0.8652 - val_loss: 0.8488 - val_accuracy: 0.6051
Epoch 5/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:04 - loss: 0.2649 - accuracy: 0.8876

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 58s - loss: 0.2638 - accuracy: 0.8876

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 43s - loss: 0.2612 - accuracy: 0.8886

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 39s - loss: 0.2599 - accuracy: 0.8892

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.2608 - accuracy: 0.8887

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

101/147 [===================>..........] - ETA: 36s - loss: 0.2608 - accuracy: 0.8885

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.2563 - accuracy: 0.8920

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 852ms/step - loss: 0.2563 - accuracy: 0.8920 - val_loss: 0.2993 - val_accuracy: 0.8744
Epoch 6/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:06 - loss: 0.2423 - accuracy: 0.8974

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:00 - loss: 0.2379 - accuracy: 0.8994

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.2322 - accuracy: 0.9023

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.2321 - accuracy: 0.9023

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 38s - loss: 0.2303 - accuracy: 0.9031

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

101/147 [===================>..........] - ETA: 37s - loss: 0.2304 - accuracy: 0.9029

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.2231 - accuracy: 0.9075

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 849ms/step - loss: 0.2231 - accuracy: 0.9075 - val_loss: 0.2684 - val_accuracy: 0.8804
Epoch 7/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:05 - loss: 0.1991 - accuracy: 0.9190

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.1992 - accuracy: 0.9185 

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.1996 - accuracy: 0.9175

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 40s - loss: 0.1991 - accuracy: 0.9174

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.1988 - accuracy: 0.9176

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 38s - loss: 0.1985 - accuracy: 0.9180

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1967 - accuracy: 0.9188

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 128s 862ms/step - loss: 0.1967 - accuracy: 0.9188 - val_loss: 0.2860 - val_accuracy: 0.8783
Epoch 8/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:05 - loss: 0.1806 - accuracy: 0.9282

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 58s - loss: 0.1767 - accuracy: 0.9292

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.1762 - accuracy: 0.9286

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 41s - loss: 0.1771 - accuracy: 0.9285

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.1774 - accuracy: 0.9279

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1756 - accuracy: 0.9296

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 856ms/step - loss: 0.1756 - accuracy: 0.9296 - val_loss: 0.2549 - val_accuracy: 0.9018
Epoch 9/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 66/147 [============>.................] - ETA: 1:05 - loss: 0.1688 - accuracy: 0.9306

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.1676 - accuracy: 0.9323

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.1684 - accuracy: 0.9314

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 40s - loss: 0.1695 - accuracy: 0.9307

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.1689 - accuracy: 0.9310

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

101/147 [===================>..........] - ETA: 36s - loss: 0.1695 - accuracy: 0.9309

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1665 - accuracy: 0.9317

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 856ms/step - loss: 0.1665 - accuracy: 0.9317 - val_loss: 0.2216 - val_accuracy: 0.9163
Epoch 10/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:06 - loss: 0.1532 - accuracy: 0.9415

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.1518 - accuracy: 0.9417 

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.1484 - accuracy: 0.9414

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 40s - loss: 0.1478 - accuracy: 0.9417

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.1486 - accuracy: 0.9412

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

101/147 [===================>..........] - ETA: 37s - loss: 0.1478 - accuracy: 0.9417

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1506 - accuracy: 0.9400

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 128s 862ms/step - loss: 0.1506 - accuracy: 0.9400 - val_loss: 0.2799 - val_accuracy: 0.8857
Epoch 11/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:06 - loss: 0.1338 - accuracy: 0.9471

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:00 - loss: 0.1308 - accuracy: 0.9480

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.1335 - accuracy: 0.9467

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.1329 - accuracy: 0.9473

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.1321 - accuracy: 0.9473

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1341 - accuracy: 0.9457

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 858ms/step - loss: 0.1341 - accuracy: 0.9457 - val_loss: 0.7291 - val_accuracy: 0.7443
Epoch 12/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:05 - loss: 0.1322 - accuracy: 0.9464

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.1340 - accuracy: 0.9452 

Warning: unknown JFIF revision number 0.00

 94/147 [==================>...........] - ETA: 42s - loss: 0.1346 - accuracy: 0.9447

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 38s - loss: 0.1333 - accuracy: 0.9454

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1349 - accuracy: 0.9462

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 125s 845ms/step - loss: 0.1349 - accuracy: 0.9462 - val_loss: 0.1958 - val_accuracy: 0.9191
Epoch 13/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:03 - loss: 0.1203 - accuracy: 0.9534

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 59s - loss: 0.1203 - accuracy: 0.9528 

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 43s - loss: 0.1253 - accuracy: 0.9501

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 40s - loss: 0.1245 - accuracy: 0.9504

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.1239 - accuracy: 0.9505

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1234 - accuracy: 0.9507

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 125s 846ms/step - loss: 0.1234 - accuracy: 0.9507 - val_loss: 0.1876 - val_accuracy: 0.9291
Epoch 14/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:06 - loss: 0.1160 - accuracy: 0.9553

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:00 - loss: 0.1155 - accuracy: 0.9554

Warning: unknown JFIF revision number 0.00

 94/147 [==================>...........] - ETA: 42s - loss: 0.1126 - accuracy: 0.9567

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 41s - loss: 0.1131 - accuracy: 0.9567

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.1131 - accuracy: 0.9566

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1111 - accuracy: 0.9574

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 856ms/step - loss: 0.1111 - accuracy: 0.9574 - val_loss: 0.2982 - val_accuracy: 0.8906
Epoch 15/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:04 - loss: 0.1132 - accuracy: 0.9557

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 75/147 [==============>...............] - ETA: 58s - loss: 0.1119 - accuracy: 0.9560

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.1104 - accuracy: 0.9561

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.1107 - accuracy: 0.9559

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 40s - loss: 0.1108 - accuracy: 0.9558

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.1106 - accuracy: 0.9560

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.1087 - accuracy: 0.9571

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 852ms/step - loss: 0.1087 - accuracy: 0.9571 - val_loss: 0.3337 - val_accuracy: 0.9018
Epoch 16/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:03 - loss: 0.1018 - accuracy: 0.9616

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 75/147 [==============>...............] - ETA: 57s - loss: 0.0979 - accuracy: 0.9632

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.0993 - accuracy: 0.9630

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 40s - loss: 0.0993 - accuracy: 0.9631

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.0990 - accuracy: 0.9630

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0979 - accuracy: 0.9638

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 849ms/step - loss: 0.0979 - accuracy: 0.9638 - val_loss: 0.1485 - val_accuracy: 0.9404
Epoch 17/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 1:06 - loss: 0.0893 - accuracy: 0.9650

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:00 - loss: 0.0911 - accuracy: 0.9638

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.0900 - accuracy: 0.9639

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.0894 - accuracy: 0.9642

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.0887 - accuracy: 0.9648

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0930 - accuracy: 0.9625

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 848ms/step - loss: 0.0930 - accuracy: 0.9625 - val_loss: 0.2074 - val_accuracy: 0.9276
Epoch 18/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:07 - loss: 0.1002 - accuracy: 0.9609

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:00 - loss: 0.0987 - accuracy: 0.9616

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 45s - loss: 0.0996 - accuracy: 0.9610

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 41s - loss: 0.0987 - accuracy: 0.9613

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 38s - loss: 0.0977 - accuracy: 0.9618

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0975 - accuracy: 0.9618

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 855ms/step - loss: 0.0975 - accuracy: 0.9618 - val_loss: 0.2466 - val_accuracy: 0.8934
Epoch 19/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:04 - loss: 0.0805 - accuracy: 0.9688

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 75/147 [==============>...............] - ETA: 58s - loss: 0.0800 - accuracy: 0.9685

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.0804 - accuracy: 0.9682

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.0806 - accuracy: 0.9683

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 38s - loss: 0.0813 - accuracy: 0.9680

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0824 - accuracy: 0.9679

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 128s 861ms/step - loss: 0.0824 - accuracy: 0.9679 - val_loss: 0.3342 - val_accuracy: 0.8911
Epoch 20/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:04 - loss: 0.0925 - accuracy: 0.9627

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 74/147 [==============>...............] - ETA: 58s - loss: 0.0918 - accuracy: 0.9632

Warning: unknown JFIF revision number 0.00

 91/147 [=================>............] - ETA: 44s - loss: 0.0911 - accuracy: 0.9636

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 95/147 [==================>...........] - ETA: 41s - loss: 0.0909 - accuracy: 0.9637

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 38s - loss: 0.0910 - accuracy: 0.9636

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0878 - accuracy: 0.9652

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 851ms/step - loss: 0.0878 - accuracy: 0.9652 - val_loss: 0.1628 - val_accuracy: 0.9317
Epoch 21/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 67/147 [============>.................] - ETA: 1:04 - loss: 0.0826 - accuracy: 0.9664

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 1:01 - loss: 0.0809 - accuracy: 0.9670

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.0803 - accuracy: 0.9676

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.0816 - accuracy: 0.9673

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 38s - loss: 0.0814 - accuracy: 0.9677

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.0812 - accuracy: 0.9679

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0809 - accuracy: 0.9680

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 858ms/step - loss: 0.0809 - accuracy: 0.9680 - val_loss: 0.1628 - val_accuracy: 0.9364
Epoch 22/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 65/147 [============>.................] - ETA: 1:05 - loss: 0.0742 - accuracy: 0.9709

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.0731 - accuracy: 0.9713 

Warning: unknown JFIF revision number 0.00

 94/147 [==================>...........] - ETA: 42s - loss: 0.0744 - accuracy: 0.9708

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.0743 - accuracy: 0.9709

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

 99/147 [===================>..........] - ETA: 38s - loss: 0.0742 - accuracy: 0.9709

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0731 - accuracy: 0.9708

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 860ms/step - loss: 0.0731 - accuracy: 0.9708 - val_loss: 0.1921 - val_accuracy: 0.9351
Epoch 23/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 1:06 - loss: 0.0740 - accuracy: 0.9731

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 59s - loss: 0.0760 - accuracy: 0.9720

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.0761 - accuracy: 0.9717

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 97/147 [==================>...........] - ETA: 40s - loss: 0.0760 - accuracy: 0.9714

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

 98/147 [===================>..........] - ETA: 39s - loss: 0.0755 - accuracy: 0.9716

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9

101/147 [===================>..........] - ETA: 36s - loss: 0.0751 - accuracy: 0.9718

Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0715 - accuracy: 0.9732

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 849ms/step - loss: 0.0715 - accuracy: 0.9732 - val_loss: 0.1802 - val_accuracy: 0.9419
Epoch 24/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 64/147 [============>.................] - ETA: 1:07 - loss: 0.0705 - accuracy: 0.9728

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 73/147 [=============>................] - ETA: 1:00 - loss: 0.0703 - accuracy: 0.9725

Warning: unknown JFIF revision number 0.00

 93/147 [=================>............] - ETA: 43s - loss: 0.0707 - accuracy: 0.9712

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 41s - loss: 0.0710 - accuracy: 0.9713

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 38s - loss: 0.0723 - accuracy: 0.9709

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0738 - accuracy: 0.9703

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 127s 861ms/step - loss: 0.0738 - accuracy: 0.9703 - val_loss: 0.1518 - val_accuracy: 0.9434
Epoch 25/25

Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9

 66/147 [============>.................] - ETA: 1:05 - loss: 0.0706 - accuracy: 0.9743

Corrupt JPEG data: 228 extraneous bytes before marker 0xd9

 72/147 [=============>................] - ETA: 59s - loss: 0.0693 - accuracy: 0.9747 

Warning: unknown JFIF revision number 0.00

 92/147 [=================>............] - ETA: 44s - loss: 0.0681 - accuracy: 0.9749

Corrupt JPEG data: 128 extraneous bytes before marker 0xd9

 96/147 [==================>...........] - ETA: 40s - loss: 0.0670 - accuracy: 0.9753

Corrupt JPEG data: 65 extraneous bytes before marker 0xd9

100/147 [===================>..........] - ETA: 37s - loss: 0.0668 - accuracy: 0.9754

Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9

147/147 [==============================] - ETA: 0s - loss: 0.0628 - accuracy: 0.9770

Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9

147/147 [==============================] - 126s 853ms/step - loss: 0.0628 - accuracy: 0.9770 - val_loss: 0.1464 - val_accuracy: 0.9517

<keras.callbacks.History at 0x7f33c892d710>

```
</div>
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).

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
1/1 [==============================] - 1s 514ms/step
This image is 77.94% cat and 22.06% dog.

```
</div>