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
!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

```

```python
!unzip -q kagglecatsanddogs_3367a.zip
!ls

```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  786M  100  786M    0     0  44.4M      0  0:00:17  0:00:17 --:--:-- 49.6M

image_classification_from_scratch.ipynb  MSR-LA - 3467.docx  readme[1].txt
kagglecatsanddogs_3367a.zip		 PetImages

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
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

```

<div class="k-default-codeblock">
```
Found 23410 files belonging to 2 classes.
Using 18728 files for training.
Found 23410 files belonging to 2 classes.
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


![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_14_0.png)


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


![png](/img/examples/vision/image_classification_from_scratch/image_classification_from_scratch_18_0.png)


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

If you're training on GPU, this is the better option.

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

In our case, we'll go with the first option.


---
## Configure the dataset for performance

Let's make sure to use buffered prefetching so we can yield data from disk without
 having I/O becoming blocking:



```python
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

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
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
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
('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.')

```
</div>
---
## Train the model



```python
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

```

<div class="k-default-codeblock">
```
Epoch 1/50
586/586 [==============================] - 81s 139ms/step - loss: 0.6233 - accuracy: 0.6700 - val_loss: 0.7698 - val_accuracy: 0.6117
Epoch 2/50
586/586 [==============================] - 80s 137ms/step - loss: 0.4638 - accuracy: 0.7840 - val_loss: 0.4056 - val_accuracy: 0.8178
Epoch 3/50
586/586 [==============================] - 80s 137ms/step - loss: 0.3652 - accuracy: 0.8405 - val_loss: 0.3535 - val_accuracy: 0.8528
Epoch 4/50
586/586 [==============================] - 80s 137ms/step - loss: 0.3112 - accuracy: 0.8675 - val_loss: 0.2673 - val_accuracy: 0.8894
Epoch 5/50
586/586 [==============================] - 80s 137ms/step - loss: 0.2585 - accuracy: 0.8928 - val_loss: 0.6213 - val_accuracy: 0.7294
Epoch 6/50
586/586 [==============================] - 81s 138ms/step - loss: 0.2218 - accuracy: 0.9071 - val_loss: 0.2377 - val_accuracy: 0.8930
Epoch 7/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1992 - accuracy: 0.9169 - val_loss: 1.1273 - val_accuracy: 0.6254
Epoch 8/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1820 - accuracy: 0.9243 - val_loss: 0.1955 - val_accuracy: 0.9173
Epoch 9/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1694 - accuracy: 0.9308 - val_loss: 0.1602 - val_accuracy: 0.9314
Epoch 10/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1623 - accuracy: 0.9333 - val_loss: 0.1777 - val_accuracy: 0.9248
Epoch 11/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1522 - accuracy: 0.9365 - val_loss: 0.1562 - val_accuracy: 0.9400
Epoch 12/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1458 - accuracy: 0.9417 - val_loss: 0.1529 - val_accuracy: 0.9338
Epoch 13/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1368 - accuracy: 0.9433 - val_loss: 0.1694 - val_accuracy: 0.9259
Epoch 14/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1301 - accuracy: 0.9461 - val_loss: 0.1250 - val_accuracy: 0.9530
Epoch 15/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1261 - accuracy: 0.9483 - val_loss: 0.1548 - val_accuracy: 0.9353
Epoch 16/50
586/586 [==============================] - 81s 137ms/step - loss: 0.1241 - accuracy: 0.9497 - val_loss: 0.1376 - val_accuracy: 0.9464
Epoch 17/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1193 - accuracy: 0.9535 - val_loss: 0.1093 - val_accuracy: 0.9575
Epoch 18/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1107 - accuracy: 0.9558 - val_loss: 0.1488 - val_accuracy: 0.9432
Epoch 19/50
586/586 [==============================] - 80s 137ms/step - loss: 0.1175 - accuracy: 0.9532 - val_loss: 0.1380 - val_accuracy: 0.9421
Epoch 20/50
586/586 [==============================] - 81s 138ms/step - loss: 0.1026 - accuracy: 0.9584 - val_loss: 0.1293 - val_accuracy: 0.9485
Epoch 21/50
586/586 [==============================] - 80s 137ms/step - loss: 0.0977 - accuracy: 0.9606 - val_loss: 0.1105 - val_accuracy: 0.9573
Epoch 22/50
586/586 [==============================] - 80s 137ms/step - loss: 0.0983 - accuracy: 0.9610 - val_loss: 0.1023 - val_accuracy: 0.9633
Epoch 23/50
586/586 [==============================] - 80s 137ms/step - loss: 0.0776 - accuracy: 0.9694 - val_loss: 0.1176 - val_accuracy: 0.9530
Epoch 38/50
586/586 [==============================] - 80s 136ms/step - loss: 0.0596 - accuracy: 0.9768 - val_loss: 0.0967 - val_accuracy: 0.9633
Epoch 44/50
586/586 [==============================] - 80s 136ms/step - loss: 0.0504 - accuracy: 0.9792 - val_loss: 0.0984 - val_accuracy: 0.9663
Epoch 50/50
586/586 [==============================] - 80s 137ms/step - loss: 0.0486 - accuracy: 0.9817 - val_loss: 0.1157 - val_accuracy: 0.9609

<tensorflow.python.keras.callbacks.History at 0x7f1694135320>

```
</div>
We get to ~96% validation accuracy after training for 50 epochs on the full dataset.


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
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)

```

<div class="k-default-codeblock">
```
This image is 84.34 percent cat and 15.66 percent dog.

```
</div>
