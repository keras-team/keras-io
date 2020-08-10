# Pneumonia Classification on TPU

**Author:** Amy MiHyun Jang<br>
**Date created:** 2020/07/28<br>
**Last modified:** 2020/08/05<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/visionipynb/xray_classification_with_tpus.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/visionxray_classification_with_tpus.py)


**Description:** Medical image classification on TPU.

---
## Introduction + Set-up

This tutorial will explain how to build an X-ray image classification model
to predict whether an X-ray scan shows presence of pneumonia.


```python
import re
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)
```

<div class="k-default-codeblock">
```
Number of replicas: 8
```
</div>
We need a Google Cloud link to our data to load the data using a TPU.
Below, we define key configuration parameters we'll use in this example.
To run on TPU, this example must be on Colab with the TPU runtime selected.


```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307"
BATCH_SIZE = 25 * strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]
```

---
## Load the data

The Chest X-ray data we are using from
[*Cell*](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) divides the data
into training, validation, and test files. There are only 16 files in the validation folder,
and we would prefer to have a less extreme division between the training and the validation set.
We will append the validation files and create a new split that resembles the standard
80:20 division instead.


```python
filenames = tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*/*"))
filenames.extend(tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/val/*/*")))

random.shuffle(filenames)
split_ind = int(0.8 * len(filenames))

train_filenames, val_filenames = filenames[:split_ind], filenames[split_ind:]
```

Let's count how many healthy/normal chest X-rays we have and how many
pneumonia chest X-rays we have:


```python
COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len(
    [filename for filename in train_filenames if "PNEUMONIA" in filename]
)
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
```

<div class="k-default-codeblock">
```
Normal images count in training set: 1076
Pneumonia images count in training set: 3109

```
</div>
Notice that there are way more images that are classified as pneumonia than normal. This
shows that we have an imbalance in our data. We will correct for this imbalance later on
in our notebook.


```python
train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

for f in train_list_ds.take(5):
    print(f.numpy())
```

<div class="k-default-codeblock">
```
b'gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307/chest_xray/train/PNEUMONIA/person141_virus_287.jpeg'
b'gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307/chest_xray/train/NORMAL/IM-0586-0001.jpeg'
b'gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307/chest_xray/train/NORMAL/NORMAL2-IM-1126-0001.jpeg'
b'gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307/chest_xray/train/PNEUMONIA/person640_virus_1221.jpeg'
b'gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307/chest_xray/train/PNEUMONIA/person505_bacteria_2135.jpeg'

```
</div>
Run the following cell to see how many images we have in our training dataset and how
many images we have in our validation set. Verify that the ratio of images is 80:20.


```python
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))
```

<div class="k-default-codeblock">
```
Training images count: 4185
Validating images count: 1047

```
</div>
As expected, we have two labels for our images.


```python
CLASS_NAMES = [
    str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
    for item in tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*"))
]
print("Class names: %s" % (CLASS_NAMES,))
```

<div class="k-default-codeblock">
```
Class names: ['NORMAL', 'PNEUMONIA']

```
</div>
Currently, our dataset is just a list of filenames. We want to map each filename to the
corresponding (image, label) pair. The following methods will help us do that.

As we only have two labels, we will encode the label so that `1` or `True` indicates
pneumonia and `0` or `False` indicates normal.


```python

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
```

Let's visualize the shape of an (image, label) pair.


```python
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
```

<div class="k-default-codeblock">
```
Image shape:  (180, 180, 3)
Label:  True

```
</div>
Load and format the test data as well.


```python
test_list_ds = tf.data.Dataset.list_files(str(GCS_PATH + "/chest_xray/test/*/*"))
TEST_IMG_COUNT = test_list_ds.cardinality().numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

print("Testing images count: " + str(TEST_IMG_COUNT))
```

<div class="k-default-codeblock">
```
Testing images count: 624

```
</div>
---
## Visualize the dataset

First, let's use buffered prefetching so we can yield data from disk without having I/O
become blocking.

Please note that large image datasets should not be cached in memory. We do it here
because the dataset is not very large and we want to train on TPU.


```python

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

```

Call the next batch iteration of the training data.


```python
train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))
```

Define the method to show the images in the batch.


```python

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255)
        if label_batch[n]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")

```

As the method takes in NumPy arrays as its parameters, call the numpy function on the
batches to return the tensor in NumPy array form.


```python
show_batch(image_batch.numpy(), label_batch.numpy())
```


![png](/img/examples/vision/xray_classification_with_tpus/xray_classification_with_tpus_28_0.png)


---
## Build the CNN

To make our model more modular and easier to understand, let's define some blocks. As
we're building a convolution neural network, we'll create a convolution block and a dense
layer block.

The architecture for this CNN has been inspired by this
[article](https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8).


```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def conv_block(filters, inputs):
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.MaxPool2D()(x)

    return outputs


def dense_block(units, dropout_rate, inputs):
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dropout(dropout_rate)(x)

    return outputs

```

The following method will define the function to build our model for us.

The images originally have values that range from [0, 255]. CNNs work better with smaller
numbers so we will scale this down for our input.

The Dropout layers are important, as they
reduce the likelikhood of the model overfitting. We want to end the model with a `Dense`
layer with one node, as this will be the binary output that determines if an X-ray shows
presence of pneumonia.


```python

def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(32, x)
    x = conv_block(64, x)

    x = conv_block(128, x)
    x = layers.Dropout(0.2)(x)

    x = conv_block(256, x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = dense_block(512, 0.7, x)
    x = dense_block(128, 0.5, x)
    x = dense_block(64, 0.3, x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Correct for data imbalance

We saw earlier in this example that the data was imbalanced, with more images classified
as pneumonia than normal. We will correct for that by using class weighting:


```python
initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))
```

<div class="k-default-codeblock">
```
Initial bias: 1.06105
Weight for class 0: 1.94
Weight for class 1: 0.67

```
</div>
The weight for class `0` (Normal) is a lot higher than the weight for class `1`
(Pneumonia). Because there are less normal images, each normal image will be weighted
more to balance the data as the CNN works best when the training data is balanced.

---
## Train the model

### Defining callbacks

The checkpoint callback saves the best weights of the model, so next time we want to use
the model, we do not have to spend time training it. The early stopping callback stops
the training process when the model starts becoming stagnant, or even worse, when the
model starts overfitting.


```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
```

We also want to tune our learning rate. Too high of a learning rate will cause the model
to diverge. Too small of a learning rate will cause the model to be too slow. We
implement the exponential learning rate scheduling method below.


```python
initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
```

### Fit the model

For our metrics, we want to include precision and recall as they will provide use with a
more informed picture of how good our model is. Accuracy tells us what fraction of the
labels is correct. Since our data is not balanced, accuracy might give a skewed sense of
a good model (i.e. a model that always predicts PNEUMONIA will be 74% accurate but is not
a good model).

Precision is the number of true positives (TP) over the sum of TP and false positives
(FP). It shows what fraction of labeled positives are actually correct.

Recall is the number of TP over the sum of TP and false negatves (FN). It shows what
fraction of actual positives are correct.

Since there are only two possible labels for the image, we will be using the
binary crossentropy loss. When we fit the model, remember to specify the class weights,
which we defined earlier. Because we are using a TPU, training will be quick - less than
2 minutes.


```python
with strategy.scope():
    model = build_model()

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=METRICS,
    )

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
```

<div class="k-default-codeblock">
```
Epoch 1/100
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py:601: get_next_as_optional (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Iterator.get_next_as_optional()` instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py:601: get_next_as_optional (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Iterator.get_next_as_optional()` instead.

 2/21 [=>............................] - ETA: 4:40 - loss: 0.8468 - binary_accuracy: 0.5725 - precision: 0.8088 - recall: 0.5556WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0039s vs `on_train_batch_end` time: 29.5466s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0039s vs `on_train_batch_end` time: 29.5466s). Check your callbacks.

21/21 [==============================] - ETA: 0s - loss: 0.5928 - binary_accuracy: 0.6671 - precision: 0.8861 - recall: 0.6333 WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0026s vs `on_test_batch_end` time: 0.0369s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0026s vs `on_test_batch_end` time: 0.0369s). Check your callbacks.

21/21 [==============================] - 744s 35s/step - loss: 0.5928 - binary_accuracy: 0.6671 - precision: 0.8861 - recall: 0.6333 - val_loss: 0.9497 - val_binary_accuracy: 0.7393 - val_precision: 0.7393 - val_recall: 1.0000
Epoch 2/100
21/21 [==============================] - 3s 165ms/step - loss: 0.2691 - binary_accuracy: 0.8961 - precision: 0.9636 - recall: 0.8939 - val_loss: 0.6038 - val_binary_accuracy: 0.7383 - val_precision: 0.7395 - val_recall: 0.9974
Epoch 3/100
21/21 [==============================] - 3s 166ms/step - loss: 0.2079 - binary_accuracy: 0.9200 - precision: 0.9714 - recall: 0.9193 - val_loss: 0.5108 - val_binary_accuracy: 0.7708 - val_precision: 0.7638 - val_recall: 0.9987
Epoch 4/100
21/21 [==============================] - 3s 123ms/step - loss: 0.1723 - binary_accuracy: 0.9312 - precision: 0.9770 - recall: 0.9292 - val_loss: 1.1205 - val_binary_accuracy: 0.7393 - val_precision: 0.7393 - val_recall: 1.0000
Epoch 5/100
21/21 [==============================] - 3s 125ms/step - loss: 0.1617 - binary_accuracy: 0.9448 - precision: 0.9800 - recall: 0.9450 - val_loss: 1.3290 - val_binary_accuracy: 0.7402 - val_precision: 0.7400 - val_recall: 1.0000
Epoch 6/100
21/21 [==============================] - 3s 137ms/step - loss: 0.1568 - binary_accuracy: 0.9405 - precision: 0.9821 - recall: 0.9370 - val_loss: 1.3986 - val_binary_accuracy: 0.7402 - val_precision: 0.7400 - val_recall: 1.0000
Epoch 7/100
21/21 [==============================] - 3s 139ms/step - loss: 0.1255 - binary_accuracy: 0.9484 - precision: 0.9830 - recall: 0.9469 - val_loss: 1.4712 - val_binary_accuracy: 0.7393 - val_precision: 0.7393 - val_recall: 1.0000
Epoch 8/100
21/21 [==============================] - 4s 170ms/step - loss: 0.1194 - binary_accuracy: 0.9565 - precision: 0.9854 - recall: 0.9556 - val_loss: 0.4464 - val_binary_accuracy: 0.8586 - val_precision: 0.8395 - val_recall: 1.0000
Epoch 9/100
21/21 [==============================] - 4s 169ms/step - loss: 0.1232 - binary_accuracy: 0.9570 - precision: 0.9848 - recall: 0.9569 - val_loss: 0.3975 - val_binary_accuracy: 0.8462 - val_precision: 0.8278 - val_recall: 1.0000
Epoch 10/100
21/21 [==============================] - 3s 125ms/step - loss: 0.1207 - binary_accuracy: 0.9546 - precision: 0.9870 - recall: 0.9514 - val_loss: 2.5951 - val_binary_accuracy: 0.7393 - val_precision: 0.7393 - val_recall: 1.0000
Epoch 11/100
21/21 [==============================] - 3s 125ms/step - loss: 0.1138 - binary_accuracy: 0.9594 - precision: 0.9871 - recall: 0.9579 - val_loss: 0.6879 - val_binary_accuracy: 0.7727 - val_precision: 0.7648 - val_recall: 1.0000
Epoch 12/100
21/21 [==============================] - 3s 124ms/step - loss: 0.1003 - binary_accuracy: 0.9661 - precision: 0.9907 - recall: 0.9633 - val_loss: 0.4012 - val_binary_accuracy: 0.9083 - val_precision: 0.8915 - val_recall: 0.9974
Epoch 13/100
21/21 [==============================] - 3s 166ms/step - loss: 0.1110 - binary_accuracy: 0.9642 - precision: 0.9884 - recall: 0.9630 - val_loss: 0.2690 - val_binary_accuracy: 0.8978 - val_precision: 0.9826 - val_recall: 0.8773
Epoch 14/100
21/21 [==============================] - 3s 143ms/step - loss: 0.1127 - binary_accuracy: 0.9572 - precision: 0.9887 - recall: 0.9534 - val_loss: 0.5987 - val_binary_accuracy: 0.8357 - val_precision: 0.9934 - val_recall: 0.7829
Epoch 15/100
21/21 [==============================] - 3s 127ms/step - loss: 0.0995 - binary_accuracy: 0.9591 - precision: 0.9864 - recall: 0.9582 - val_loss: 2.8717 - val_binary_accuracy: 0.3830 - val_precision: 1.0000 - val_recall: 0.1654
Epoch 16/100
21/21 [==============================] - 3s 130ms/step - loss: 0.1019 - binary_accuracy: 0.9634 - precision: 0.9907 - recall: 0.9598 - val_loss: 7.2178 - val_binary_accuracy: 0.3734 - val_precision: 1.0000 - val_recall: 0.1525
Epoch 17/100
21/21 [==============================] - 3s 126ms/step - loss: 0.1027 - binary_accuracy: 0.9625 - precision: 0.9910 - recall: 0.9582 - val_loss: 2.4284 - val_binary_accuracy: 0.6495 - val_precision: 1.0000 - val_recall: 0.5258
Epoch 18/100
21/21 [==============================] - 3s 125ms/step - loss: 0.0887 - binary_accuracy: 0.9654 - precision: 0.9920 - recall: 0.9611 - val_loss: 0.3605 - val_binary_accuracy: 0.9265 - val_precision: 1.0000 - val_recall: 0.9005
Epoch 19/100
21/21 [==============================] - 3s 166ms/step - loss: 0.0664 - binary_accuracy: 0.9756 - precision: 0.9944 - recall: 0.9727 - val_loss: 0.2604 - val_binary_accuracy: 0.9580 - val_precision: 0.9932 - val_recall: 0.9496
Epoch 20/100
21/21 [==============================] - 3s 140ms/step - loss: 0.0689 - binary_accuracy: 0.9744 - precision: 0.9931 - recall: 0.9723 - val_loss: 2.5697 - val_binary_accuracy: 0.7622 - val_precision: 1.0000 - val_recall: 0.6783
Epoch 21/100
21/21 [==============================] - 3s 125ms/step - loss: 0.0639 - binary_accuracy: 0.9778 - precision: 0.9957 - recall: 0.9743 - val_loss: 0.8325 - val_binary_accuracy: 0.8682 - val_precision: 1.0000 - val_recall: 0.8217
Epoch 22/100
21/21 [==============================] - 3s 126ms/step - loss: 0.0793 - binary_accuracy: 0.9761 - precision: 0.9931 - recall: 0.9746 - val_loss: 12.2000 - val_binary_accuracy: 0.2961 - val_precision: 1.0000 - val_recall: 0.0478
Epoch 23/100
21/21 [==============================] - 3s 127ms/step - loss: 0.0760 - binary_accuracy: 0.9716 - precision: 0.9934 - recall: 0.9682 - val_loss: 3.6567 - val_binary_accuracy: 0.3992 - val_precision: 1.0000 - val_recall: 0.1873
Epoch 24/100
21/21 [==============================] - 3s 123ms/step - loss: 0.0820 - binary_accuracy: 0.9697 - precision: 0.9931 - recall: 0.9659 - val_loss: 0.7858 - val_binary_accuracy: 0.8281 - val_precision: 0.9934 - val_recall: 0.7726
Epoch 25/100
21/21 [==============================] - 3s 123ms/step - loss: 0.0725 - binary_accuracy: 0.9718 - precision: 0.9921 - recall: 0.9698 - val_loss: 1.7016 - val_binary_accuracy: 0.6781 - val_precision: 1.0000 - val_recall: 0.5646
Epoch 26/100
21/21 [==============================] - 3s 127ms/step - loss: 0.0574 - binary_accuracy: 0.9775 - precision: 0.9957 - recall: 0.9739 - val_loss: 1.5002 - val_binary_accuracy: 0.6963 - val_precision: 1.0000 - val_recall: 0.5891
Epoch 27/100
21/21 [==============================] - 3s 126ms/step - loss: 0.0549 - binary_accuracy: 0.9809 - precision: 0.9957 - recall: 0.9784 - val_loss: 0.6662 - val_binary_accuracy: 0.8863 - val_precision: 1.0000 - val_recall: 0.8463
Epoch 28/100
21/21 [==============================] - 3s 127ms/step - loss: 0.0573 - binary_accuracy: 0.9818 - precision: 0.9964 - recall: 0.9791 - val_loss: 2.6141 - val_binary_accuracy: 0.5444 - val_precision: 1.0000 - val_recall: 0.3837
Epoch 29/100
21/21 [==============================] - 4s 185ms/step - loss: 0.0644 - binary_accuracy: 0.9759 - precision: 0.9938 - recall: 0.9736 - val_loss: 5.0325 - val_binary_accuracy: 0.6189 - val_precision: 1.0000 - val_recall: 0.4845

```
</div>
---
## Visualizing model performance

Let's plot the model accuracy and loss for the training and the validating set. Note that
no random seed is specified for this notebook. For your notebook, there might be slight
variance.


```python
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])
```


![png](/img/examples/vision/xray_classification_with_tpus/xray_classification_with_tpus_44_0.png)


We see that the accuracy for our model is around 95%.

---
## Predict and evaluate results

Let's evaluate the model on our test data!


```python
model.evaluate(test_ds, return_dict=True)
```

<div class="k-default-codeblock">
```
2/4 [==============>...............] - ETA: 1:12 - loss: 2.4281 - binary_accuracy: 0.8250 - precision: 0.8025 - recall: 0.9734WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0025s vs `on_test_batch_end` time: 32.3267s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0025s vs `on_test_batch_end` time: 32.3267s). Check your callbacks.

4/4 [==============================] - 113s 28s/step - loss: 2.7205 - binary_accuracy: 0.8141 - precision: 0.7807 - recall: 0.9769

{'binary_accuracy': 0.8141025900840759,
 'loss': 2.720494508743286,
 'precision': 0.7807376384735107,
 'recall': 0.9769231081008911}

```
</div>
We see that our accuracy on our test data is lower than the accuracy for our validating
set. This may indicate overfitting.

Our recall is greater than our precision, indicating that almost all pneumonia images are
correctly identified but some normal images are falsely identified. We should aim to
increase our precision.


```python
img = tf.io.read_file(
    str(GCS_PATH + "/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg")
)
img = decode_img(img)
plt.imshow(img / 255)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

prediction = model.predict(img_array)[0]
scores = [1 - prediction, prediction]

for score, name in zip(scores, CLASS_NAMES):
    print("This image is %.2f percent %s" % ((100 * score), name))
```

<div class="k-default-codeblock">
```
This image is 0.00 percent NORMAL
This image is 100.00 percent PNEUMONIA

```
</div>
![png](/img/examples/vision/xray_classification_with_tpus/xray_classification_with_tpus_49_1.png)


Our model could accurately classify this image.
