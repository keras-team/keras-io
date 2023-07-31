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

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
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
147/147 [==============================] - 116s 746ms/step - loss: 0.6531 - accuracy: 0.6416 - val_loss: 0.7669 - val_accuracy: 0.4957
Epoch 2/25
147/147 [==============================] - 109s 737ms/step - loss: 0.5026 - accuracy: 0.7559 - val_loss: 1.3825 - val_accuracy: 0.4957
Epoch 3/25
147/147 [==============================] - 109s 738ms/step - loss: 0.3928 - accuracy: 0.8243 - val_loss: 1.6816 - val_accuracy: 0.4957
Epoch 4/25
147/147 [==============================] - 109s 736ms/step - loss: 0.3307 - accuracy: 0.8588 - val_loss: 0.5025 - val_accuracy: 0.7520
Epoch 5/25
147/147 [==============================] - 109s 734ms/step - loss: 0.2758 - accuracy: 0.8860 - val_loss: 0.3462 - val_accuracy: 0.8545
Epoch 6/25
147/147 [==============================] - 109s 735ms/step - loss: 0.2357 - accuracy: 0.9023 - val_loss: 0.2712 - val_accuracy: 0.8825
Epoch 7/25
147/147 [==============================] - 109s 734ms/step - loss: 0.2011 - accuracy: 0.9201 - val_loss: 0.2131 - val_accuracy: 0.9135
Epoch 8/25
147/147 [==============================] - 109s 735ms/step - loss: 0.1787 - accuracy: 0.9275 - val_loss: 0.1969 - val_accuracy: 0.9227
Epoch 9/25
147/147 [==============================] - 109s 734ms/step - loss: 0.1650 - accuracy: 0.9321 - val_loss: 0.2306 - val_accuracy: 0.9178
Epoch 10/25
147/147 [==============================] - 109s 734ms/step - loss: 0.1474 - accuracy: 0.9408 - val_loss: 0.2430 - val_accuracy: 0.9107
Epoch 11/25
147/147 [==============================] - 109s 735ms/step - loss: 0.1352 - accuracy: 0.9461 - val_loss: 0.2783 - val_accuracy: 0.8768
Epoch 12/25
147/147 [==============================] - 109s 734ms/step - loss: 0.1291 - accuracy: 0.9474 - val_loss: 0.4632 - val_accuracy: 0.8419
Epoch 13/25
147/147 [==============================] - 109s 735ms/step - loss: 0.1208 - accuracy: 0.9521 - val_loss: 0.3907 - val_accuracy: 0.8456
Epoch 14/25
147/147 [==============================] - 110s 739ms/step - loss: 0.1162 - accuracy: 0.9553 - val_loss: 0.1503 - val_accuracy: 0.9417
Epoch 15/25
147/147 [==============================] - 109s 735ms/step - loss: 0.1037 - accuracy: 0.9598 - val_loss: 0.1484 - val_accuracy: 0.9406
Epoch 16/25
147/147 [==============================] - 109s 734ms/step - loss: 0.1018 - accuracy: 0.9605 - val_loss: 0.2480 - val_accuracy: 0.9054
Epoch 17/25
147/147 [==============================] - 109s 739ms/step - loss: 0.0949 - accuracy: 0.9629 - val_loss: 0.1585 - val_accuracy: 0.9378
Epoch 18/25
147/147 [==============================] - 109s 736ms/step - loss: 0.0941 - accuracy: 0.9622 - val_loss: 0.1452 - val_accuracy: 0.9432
Epoch 19/25
147/147 [==============================] - 109s 734ms/step - loss: 0.0862 - accuracy: 0.9668 - val_loss: 0.2644 - val_accuracy: 0.8904
Epoch 20/25
147/147 [==============================] - 109s 734ms/step - loss: 0.0889 - accuracy: 0.9656 - val_loss: 0.2335 - val_accuracy: 0.9182
Epoch 21/25
147/147 [==============================] - 109s 735ms/step - loss: 0.0792 - accuracy: 0.9687 - val_loss: 0.5037 - val_accuracy: 0.8751
Epoch 22/25
147/147 [==============================] - 109s 734ms/step - loss: 0.0651 - accuracy: 0.9737 - val_loss: 0.1103 - val_accuracy: 0.9551
Epoch 23/25
147/147 [==============================] - 109s 735ms/step - loss: 0.0641 - accuracy: 0.9751 - val_loss: 0.1846 - val_accuracy: 0.9299
Epoch 24/25
147/147 [==============================] - 109s 735ms/step - loss: 0.0709 - accuracy: 0.9735 - val_loss: 0.1151 - val_accuracy: 0.9575
Epoch 25/25
147/147 [==============================] - 109s 737ms/step - loss: 0.0612 - accuracy: 0.9768 - val_loss: 0.1259 - val_accuracy: 0.9510

<keras.callbacks.History at 0x7fd3941c87b8>

```
</div>
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).

---
## Run inference on new data

```python
img = keras.utils.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 446ms/step
This image is 85.28% cat and 14.72% dog.

```
</div>