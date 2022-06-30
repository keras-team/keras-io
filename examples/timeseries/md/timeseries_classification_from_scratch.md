# Timeseries classification from scratch

**Author:** [hfawaz](https://github.com/hfawaz/)<br>
**Date created:** 2020/07/21<br>
**Last modified:** 2021/07/16<br>
**Description:** Training a timeseries classifier from scratch on the FordA dataset from the UCR/UEA archive.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_classification_from_scratch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_classification_from_scratch.py)



---
## Introduction

This example shows how to do timeseries classification from scratch, starting from raw
CSV timeseries files on disk. We demonstrate the workflow on the FordA dataset from the
[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

---
## Setup


```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

---
## Load the data: the FordA dataset

### Dataset description

The dataset we are using here is called FordA.
The data comes from the UCR archive.
The dataset contains 3601 training instances and another 1320 testing instances.
Each timeseries corresponds to a measurement of engine noise captured by a motor sensor.
For this task, the goal is to automatically detect the presence of a specific issue with
the engine. The problem is a balanced binary classification task. The full description of
this dataset can be found [here](http://www.j-wichard.de/publications/FordPaper.pdf).

### Read the TSV data

We will use the `FordA_TRAIN` file for training and the
`FordA_TEST` file for testing. The simplicity of this dataset
allows us to demonstrate effectively how to use ConvNets for timeseries classification.
In this file, the first column corresponds to the label.


```python

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
```

---
## Visualize the data

Here we visualize one timeseries example for each class in the dataset.


```python
classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()
```


    
![png](/img/examples/timeseries/timeseries_classification_from_scratch/timeseries_classification_from_scratch_7_0.png)
    


---
## Standardize the data

Our timeseries are already in a single length (500). However, their values are
usually in various ranges. This is not ideal for a neural network;
in general we should seek to make the input values normalized.
For this specific dataset, the data is already z-normalized: each timeseries sample
has a mean equal to zero and a standard deviation equal to one. This type of
normalization is very common for timeseries classification problems, see
[Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).

Note that the timeseries data used here are univariate, meaning we only have one channel
per timeseries example.
We will therefore transform the timeseries into a multivariate one with one channel
using a simple reshaping via numpy.
This will allow us to construct a model that is easily applicable to multivariate time
series.


```python
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
```

Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
the number of classes beforehand.


```python
num_classes = len(np.unique(y_train))
```

Now we shuffle the training set because we will be using the `validation_split` option
later when training.


```python
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
```

Standardize the labels to positive integers.
The expected labels will then be 0 and 1.


```python
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
```

---
## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).
The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).


```python

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)
```

<div class="k-default-codeblock">
```
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.

```
</div>
---
## Train the model


```python
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
```

<div class="k-default-codeblock">
```
Epoch 1/500
90/90 [==============================] - 4s 19ms/step - loss: 0.5554 - sparse_categorical_accuracy: 0.6969 - val_loss: 0.7550 - val_sparse_categorical_accuracy: 0.4771 - lr: 0.0010
Epoch 2/500
90/90 [==============================] - 2s 17ms/step - loss: 0.4683 - sparse_categorical_accuracy: 0.7625 - val_loss: 0.8690 - val_sparse_categorical_accuracy: 0.4771 - lr: 0.0010
Epoch 3/500
90/90 [==============================] - 2s 17ms/step - loss: 0.4392 - sparse_categorical_accuracy: 0.7649 - val_loss: 1.0477 - val_sparse_categorical_accuracy: 0.4771 - lr: 0.0010
Epoch 4/500
90/90 [==============================] - 1s 17ms/step - loss: 0.4319 - sparse_categorical_accuracy: 0.7785 - val_loss: 0.8893 - val_sparse_categorical_accuracy: 0.4771 - lr: 0.0010
Epoch 5/500
90/90 [==============================] - 2s 17ms/step - loss: 0.4207 - sparse_categorical_accuracy: 0.7885 - val_loss: 0.6663 - val_sparse_categorical_accuracy: 0.5368 - lr: 0.0010
Epoch 6/500
90/90 [==============================] - 2s 17ms/step - loss: 0.4037 - sparse_categorical_accuracy: 0.7997 - val_loss: 0.4300 - val_sparse_categorical_accuracy: 0.8488 - lr: 0.0010
Epoch 7/500
90/90 [==============================] - 1s 16ms/step - loss: 0.4016 - sparse_categorical_accuracy: 0.8063 - val_loss: 1.4926 - val_sparse_categorical_accuracy: 0.5742 - lr: 0.0010
Epoch 8/500
90/90 [==============================] - 1s 15ms/step - loss: 0.4032 - sparse_categorical_accuracy: 0.7906 - val_loss: 0.3655 - val_sparse_categorical_accuracy: 0.8197 - lr: 0.0010
Epoch 9/500
90/90 [==============================] - 1s 15ms/step - loss: 0.3792 - sparse_categorical_accuracy: 0.8156 - val_loss: 0.4062 - val_sparse_categorical_accuracy: 0.7517 - lr: 0.0010
Epoch 10/500
90/90 [==============================] - 2s 17ms/step - loss: 0.3766 - sparse_categorical_accuracy: 0.8226 - val_loss: 0.6753 - val_sparse_categorical_accuracy: 0.6976 - lr: 0.0010
Epoch 11/500
90/90 [==============================] - 1s 16ms/step - loss: 0.3763 - sparse_categorical_accuracy: 0.8184 - val_loss: 0.3262 - val_sparse_categorical_accuracy: 0.8599 - lr: 0.0010
Epoch 12/500
90/90 [==============================] - 1s 16ms/step - loss: 0.3624 - sparse_categorical_accuracy: 0.8257 - val_loss: 0.4381 - val_sparse_categorical_accuracy: 0.7947 - lr: 0.0010
Epoch 13/500
90/90 [==============================] - 1s 16ms/step - loss: 0.3570 - sparse_categorical_accuracy: 0.8351 - val_loss: 0.4016 - val_sparse_categorical_accuracy: 0.7642 - lr: 0.0010
Epoch 14/500
90/90 [==============================] - 1s 17ms/step - loss: 0.3324 - sparse_categorical_accuracy: 0.8559 - val_loss: 0.5826 - val_sparse_categorical_accuracy: 0.6879 - lr: 0.0010
Epoch 15/500
90/90 [==============================] - 1s 15ms/step - loss: 0.3429 - sparse_categorical_accuracy: 0.8413 - val_loss: 1.9227 - val_sparse_categorical_accuracy: 0.4799 - lr: 0.0010
Epoch 16/500
90/90 [==============================] - 1s 16ms/step - loss: 0.3344 - sparse_categorical_accuracy: 0.8514 - val_loss: 0.3054 - val_sparse_categorical_accuracy: 0.8710 - lr: 0.0010
Epoch 17/500
90/90 [==============================] - 1s 15ms/step - loss: 0.3111 - sparse_categorical_accuracy: 0.8681 - val_loss: 1.0335 - val_sparse_categorical_accuracy: 0.5395 - lr: 0.0010
Epoch 18/500
90/90 [==============================] - 2s 17ms/step - loss: 0.3143 - sparse_categorical_accuracy: 0.8601 - val_loss: 0.5179 - val_sparse_categorical_accuracy: 0.7171 - lr: 0.0010
Epoch 19/500
90/90 [==============================] - 1s 15ms/step - loss: 0.3030 - sparse_categorical_accuracy: 0.8715 - val_loss: 0.4609 - val_sparse_categorical_accuracy: 0.7268 - lr: 0.0010
Epoch 20/500
90/90 [==============================] - 1s 16ms/step - loss: 0.3114 - sparse_categorical_accuracy: 0.8653 - val_loss: 0.2724 - val_sparse_categorical_accuracy: 0.8821 - lr: 0.0010
Epoch 21/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2900 - sparse_categorical_accuracy: 0.8788 - val_loss: 0.8168 - val_sparse_categorical_accuracy: 0.7143 - lr: 0.0010
Epoch 22/500
90/90 [==============================] - 1s 15ms/step - loss: 0.3042 - sparse_categorical_accuracy: 0.8660 - val_loss: 0.8978 - val_sparse_categorical_accuracy: 0.6297 - lr: 0.0010
Epoch 23/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2908 - sparse_categorical_accuracy: 0.8774 - val_loss: 0.3084 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 24/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2774 - sparse_categorical_accuracy: 0.8826 - val_loss: 0.2578 - val_sparse_categorical_accuracy: 0.8904 - lr: 0.0010
Epoch 25/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2752 - sparse_categorical_accuracy: 0.8896 - val_loss: 1.8307 - val_sparse_categorical_accuracy: 0.5229 - lr: 0.0010
Epoch 26/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2674 - sparse_categorical_accuracy: 0.8917 - val_loss: 1.1022 - val_sparse_categorical_accuracy: 0.5908 - lr: 0.0010
Epoch 27/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2665 - sparse_categorical_accuracy: 0.8944 - val_loss: 0.3530 - val_sparse_categorical_accuracy: 0.8419 - lr: 0.0010
Epoch 28/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2622 - sparse_categorical_accuracy: 0.8938 - val_loss: 0.3882 - val_sparse_categorical_accuracy: 0.8100 - lr: 0.0010
Epoch 29/500
90/90 [==============================] - 2s 18ms/step - loss: 0.2726 - sparse_categorical_accuracy: 0.8813 - val_loss: 0.2384 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
Epoch 30/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2799 - sparse_categorical_accuracy: 0.8785 - val_loss: 1.1490 - val_sparse_categorical_accuracy: 0.6824 - lr: 0.0010
Epoch 31/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2639 - sparse_categorical_accuracy: 0.8885 - val_loss: 1.1054 - val_sparse_categorical_accuracy: 0.6657 - lr: 0.0010
Epoch 32/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2608 - sparse_categorical_accuracy: 0.8899 - val_loss: 0.2576 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
Epoch 33/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2579 - sparse_categorical_accuracy: 0.8903 - val_loss: 0.5989 - val_sparse_categorical_accuracy: 0.7309 - lr: 0.0010
Epoch 34/500
90/90 [==============================] - 1s 15ms/step - loss: 0.2481 - sparse_categorical_accuracy: 0.9000 - val_loss: 0.2655 - val_sparse_categorical_accuracy: 0.8655 - lr: 0.0010
Epoch 35/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2461 - sparse_categorical_accuracy: 0.8979 - val_loss: 0.3423 - val_sparse_categorical_accuracy: 0.8225 - lr: 0.0010
Epoch 36/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2565 - sparse_categorical_accuracy: 0.8927 - val_loss: 0.5507 - val_sparse_categorical_accuracy: 0.7282 - lr: 0.0010
Epoch 37/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2420 - sparse_categorical_accuracy: 0.9031 - val_loss: 0.2979 - val_sparse_categorical_accuracy: 0.8488 - lr: 0.0010
Epoch 38/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2383 - sparse_categorical_accuracy: 0.9062 - val_loss: 2.4242 - val_sparse_categorical_accuracy: 0.5229 - lr: 0.0010
Epoch 39/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2278 - sparse_categorical_accuracy: 0.9080 - val_loss: 1.1454 - val_sparse_categorical_accuracy: 0.6089 - lr: 0.0010
Epoch 40/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2386 - sparse_categorical_accuracy: 0.9080 - val_loss: 0.4439 - val_sparse_categorical_accuracy: 0.8058 - lr: 0.0010
Epoch 41/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2349 - sparse_categorical_accuracy: 0.9049 - val_loss: 0.8547 - val_sparse_categorical_accuracy: 0.6366 - lr: 0.0010
Epoch 42/500
90/90 [==============================] - 2s 18ms/step - loss: 0.2322 - sparse_categorical_accuracy: 0.9056 - val_loss: 0.2770 - val_sparse_categorical_accuracy: 0.8793 - lr: 0.0010
Epoch 43/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2669 - sparse_categorical_accuracy: 0.8872 - val_loss: 0.6167 - val_sparse_categorical_accuracy: 0.7226 - lr: 0.0010
Epoch 44/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2283 - sparse_categorical_accuracy: 0.9104 - val_loss: 0.3539 - val_sparse_categorical_accuracy: 0.8031 - lr: 0.0010
Epoch 45/500
90/90 [==============================] - 2s 17ms/step - loss: 0.2285 - sparse_categorical_accuracy: 0.9090 - val_loss: 1.3117 - val_sparse_categorical_accuracy: 0.5659 - lr: 0.0010
Epoch 46/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2289 - sparse_categorical_accuracy: 0.9083 - val_loss: 1.9057 - val_sparse_categorical_accuracy: 0.5229 - lr: 0.0010
Epoch 47/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2174 - sparse_categorical_accuracy: 0.9125 - val_loss: 2.6058 - val_sparse_categorical_accuracy: 0.5756 - lr: 0.0010
Epoch 48/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2106 - sparse_categorical_accuracy: 0.9149 - val_loss: 4.0373 - val_sparse_categorical_accuracy: 0.5049 - lr: 0.0010
Epoch 49/500
90/90 [==============================] - 1s 16ms/step - loss: 0.2361 - sparse_categorical_accuracy: 0.9042 - val_loss: 0.9768 - val_sparse_categorical_accuracy: 0.6117 - lr: 0.0010
Epoch 50/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1953 - sparse_categorical_accuracy: 0.9281 - val_loss: 1.3880 - val_sparse_categorical_accuracy: 0.6976 - lr: 5.0000e-04
Epoch 51/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1872 - sparse_categorical_accuracy: 0.9292 - val_loss: 0.7788 - val_sparse_categorical_accuracy: 0.6935 - lr: 5.0000e-04
Epoch 52/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1801 - sparse_categorical_accuracy: 0.9337 - val_loss: 0.1860 - val_sparse_categorical_accuracy: 0.9348 - lr: 5.0000e-04
Epoch 53/500
90/90 [==============================] - 2s 18ms/step - loss: 0.1698 - sparse_categorical_accuracy: 0.9389 - val_loss: 0.3886 - val_sparse_categorical_accuracy: 0.8100 - lr: 5.0000e-04
Epoch 54/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1704 - sparse_categorical_accuracy: 0.9399 - val_loss: 0.1713 - val_sparse_categorical_accuracy: 0.9293 - lr: 5.0000e-04
Epoch 55/500
90/90 [==============================] - 2s 18ms/step - loss: 0.1622 - sparse_categorical_accuracy: 0.9410 - val_loss: 0.1911 - val_sparse_categorical_accuracy: 0.9196 - lr: 5.0000e-04
Epoch 56/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1625 - sparse_categorical_accuracy: 0.9389 - val_loss: 0.2347 - val_sparse_categorical_accuracy: 0.8890 - lr: 5.0000e-04
Epoch 57/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1605 - sparse_categorical_accuracy: 0.9448 - val_loss: 0.4846 - val_sparse_categorical_accuracy: 0.7531 - lr: 5.0000e-04
Epoch 58/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1568 - sparse_categorical_accuracy: 0.9465 - val_loss: 1.0308 - val_sparse_categorical_accuracy: 0.6449 - lr: 5.0000e-04
Epoch 59/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1500 - sparse_categorical_accuracy: 0.9479 - val_loss: 0.7054 - val_sparse_categorical_accuracy: 0.7184 - lr: 5.0000e-04
Epoch 60/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1434 - sparse_categorical_accuracy: 0.9497 - val_loss: 0.9202 - val_sparse_categorical_accuracy: 0.6574 - lr: 5.0000e-04
Epoch 61/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1309 - sparse_categorical_accuracy: 0.9538 - val_loss: 2.0657 - val_sparse_categorical_accuracy: 0.6824 - lr: 5.0000e-04
Epoch 62/500
90/90 [==============================] - 1s 17ms/step - loss: 0.1411 - sparse_categorical_accuracy: 0.9549 - val_loss: 0.7986 - val_sparse_categorical_accuracy: 0.6810 - lr: 5.0000e-04
Epoch 63/500
90/90 [==============================] - 1s 15ms/step - loss: 0.1377 - sparse_categorical_accuracy: 0.9472 - val_loss: 0.1914 - val_sparse_categorical_accuracy: 0.9209 - lr: 5.0000e-04
Epoch 64/500
90/90 [==============================] - 1s 15ms/step - loss: 0.1308 - sparse_categorical_accuracy: 0.9528 - val_loss: 0.6375 - val_sparse_categorical_accuracy: 0.7462 - lr: 5.0000e-04
Epoch 65/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1339 - sparse_categorical_accuracy: 0.9517 - val_loss: 0.1269 - val_sparse_categorical_accuracy: 0.9653 - lr: 5.0000e-04
Epoch 66/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1352 - sparse_categorical_accuracy: 0.9517 - val_loss: 0.3781 - val_sparse_categorical_accuracy: 0.8128 - lr: 5.0000e-04
Epoch 67/500
90/90 [==============================] - 1s 15ms/step - loss: 0.1246 - sparse_categorical_accuracy: 0.9580 - val_loss: 0.6750 - val_sparse_categorical_accuracy: 0.7226 - lr: 5.0000e-04
Epoch 68/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1247 - sparse_categorical_accuracy: 0.9590 - val_loss: 0.1933 - val_sparse_categorical_accuracy: 0.9279 - lr: 5.0000e-04
Epoch 69/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1330 - sparse_categorical_accuracy: 0.9517 - val_loss: 0.1645 - val_sparse_categorical_accuracy: 0.9265 - lr: 5.0000e-04
Epoch 70/500
90/90 [==============================] - 2s 19ms/step - loss: 0.1184 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.1139 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 71/500
90/90 [==============================] - 1s 17ms/step - loss: 0.1271 - sparse_categorical_accuracy: 0.9580 - val_loss: 0.1375 - val_sparse_categorical_accuracy: 0.9501 - lr: 5.0000e-04
Epoch 72/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1306 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.6631 - val_sparse_categorical_accuracy: 0.7406 - lr: 5.0000e-04
Epoch 73/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1193 - sparse_categorical_accuracy: 0.9608 - val_loss: 1.7042 - val_sparse_categorical_accuracy: 0.7046 - lr: 5.0000e-04
Epoch 74/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1240 - sparse_categorical_accuracy: 0.9594 - val_loss: 0.3667 - val_sparse_categorical_accuracy: 0.8336 - lr: 5.0000e-04
Epoch 75/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1317 - sparse_categorical_accuracy: 0.9521 - val_loss: 0.1496 - val_sparse_categorical_accuracy: 0.9362 - lr: 5.0000e-04
Epoch 76/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1209 - sparse_categorical_accuracy: 0.9563 - val_loss: 0.1316 - val_sparse_categorical_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 77/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1215 - sparse_categorical_accuracy: 0.9587 - val_loss: 1.8449 - val_sparse_categorical_accuracy: 0.5589 - lr: 5.0000e-04
Epoch 78/500
90/90 [==============================] - 2s 18ms/step - loss: 0.1183 - sparse_categorical_accuracy: 0.9604 - val_loss: 0.1069 - val_sparse_categorical_accuracy: 0.9653 - lr: 5.0000e-04
Epoch 79/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1158 - sparse_categorical_accuracy: 0.9628 - val_loss: 1.9783 - val_sparse_categorical_accuracy: 0.6810 - lr: 5.0000e-04
Epoch 80/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1216 - sparse_categorical_accuracy: 0.9587 - val_loss: 0.3074 - val_sparse_categorical_accuracy: 0.8516 - lr: 5.0000e-04
Epoch 81/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1118 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.8524 - val_sparse_categorical_accuracy: 0.7171 - lr: 5.0000e-04
Epoch 82/500
90/90 [==============================] - 2s 18ms/step - loss: 0.1074 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.6653 - val_sparse_categorical_accuracy: 0.7517 - lr: 5.0000e-04
Epoch 83/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1151 - sparse_categorical_accuracy: 0.9604 - val_loss: 0.2433 - val_sparse_categorical_accuracy: 0.8918 - lr: 5.0000e-04
Epoch 84/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1139 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1556 - val_sparse_categorical_accuracy: 0.9320 - lr: 5.0000e-04
Epoch 85/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1090 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.1614 - val_sparse_categorical_accuracy: 0.9265 - lr: 5.0000e-04
Epoch 86/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1109 - sparse_categorical_accuracy: 0.9642 - val_loss: 2.1912 - val_sparse_categorical_accuracy: 0.6824 - lr: 5.0000e-04
Epoch 87/500
90/90 [==============================] - 1s 15ms/step - loss: 0.1137 - sparse_categorical_accuracy: 0.9646 - val_loss: 1.0993 - val_sparse_categorical_accuracy: 0.7226 - lr: 5.0000e-04
Epoch 88/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1167 - sparse_categorical_accuracy: 0.9580 - val_loss: 0.1261 - val_sparse_categorical_accuracy: 0.9528 - lr: 5.0000e-04
Epoch 89/500
90/90 [==============================] - 2s 18ms/step - loss: 0.1086 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.1672 - val_sparse_categorical_accuracy: 0.9404 - lr: 5.0000e-04
Epoch 90/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1140 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.3000 - val_sparse_categorical_accuracy: 0.8641 - lr: 5.0000e-04
Epoch 91/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1080 - sparse_categorical_accuracy: 0.9635 - val_loss: 1.6657 - val_sparse_categorical_accuracy: 0.6893 - lr: 5.0000e-04
Epoch 92/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1032 - sparse_categorical_accuracy: 0.9649 - val_loss: 1.1590 - val_sparse_categorical_accuracy: 0.7309 - lr: 5.0000e-04
Epoch 93/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1105 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.4788 - val_sparse_categorical_accuracy: 0.7975 - lr: 5.0000e-04
Epoch 94/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0985 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1762 - val_sparse_categorical_accuracy: 0.9293 - lr: 5.0000e-04
Epoch 95/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0967 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.3000 - val_sparse_categorical_accuracy: 0.8641 - lr: 5.0000e-04
Epoch 96/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1071 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.1081 - val_sparse_categorical_accuracy: 0.9639 - lr: 5.0000e-04
Epoch 97/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1031 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.5081 - val_sparse_categorical_accuracy: 0.7920 - lr: 5.0000e-04
Epoch 98/500
90/90 [==============================] - 2s 17ms/step - loss: 0.1046 - sparse_categorical_accuracy: 0.9611 - val_loss: 0.2052 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
Epoch 99/500
90/90 [==============================] - 1s 16ms/step - loss: 0.1015 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.1345 - val_sparse_categorical_accuracy: 0.9445 - lr: 2.5000e-04
Epoch 100/500
90/90 [==============================] - 1s 17ms/step - loss: 0.1037 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1875 - val_sparse_categorical_accuracy: 0.9140 - lr: 2.5000e-04
Epoch 101/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0979 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.6255 - val_sparse_categorical_accuracy: 0.7739 - lr: 2.5000e-04
Epoch 102/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0961 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9584 - lr: 2.5000e-04
Epoch 103/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0982 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.2673 - val_sparse_categorical_accuracy: 0.8724 - lr: 2.5000e-04
Epoch 104/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0976 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.1509 - val_sparse_categorical_accuracy: 0.9348 - lr: 2.5000e-04
Epoch 105/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0951 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.1712 - val_sparse_categorical_accuracy: 0.9279 - lr: 2.5000e-04
Epoch 106/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0985 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.5290 - val_sparse_categorical_accuracy: 0.7490 - lr: 2.5000e-04
Epoch 107/500
90/90 [==============================] - 1s 17ms/step - loss: 0.1009 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1907 - val_sparse_categorical_accuracy: 0.9209 - lr: 2.5000e-04
Epoch 108/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0943 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.4829 - val_sparse_categorical_accuracy: 0.8003 - lr: 2.5000e-04
Epoch 109/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0994 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.3914 - val_sparse_categorical_accuracy: 0.8197 - lr: 2.5000e-04
Epoch 110/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1094 - val_sparse_categorical_accuracy: 0.9570 - lr: 2.5000e-04
Epoch 111/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0923 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.3189 - val_sparse_categorical_accuracy: 0.8544 - lr: 2.5000e-04
Epoch 112/500
90/90 [==============================] - 2s 19ms/step - loss: 0.0963 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1033 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 113/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0927 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1034 - val_sparse_categorical_accuracy: 0.9598 - lr: 2.5000e-04
Epoch 114/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0961 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1596 - val_sparse_categorical_accuracy: 0.9293 - lr: 2.5000e-04
Epoch 115/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0980 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1907 - val_sparse_categorical_accuracy: 0.9140 - lr: 2.5000e-04
Epoch 116/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0985 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 117/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0939 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.0868 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 118/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0954 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.0991 - val_sparse_categorical_accuracy: 0.9612 - lr: 2.5000e-04
Epoch 119/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0906 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 120/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0891 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.0887 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 121/500
90/90 [==============================] - 2s 19ms/step - loss: 0.0911 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.2260 - val_sparse_categorical_accuracy: 0.8946 - lr: 2.5000e-04
Epoch 122/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0897 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 123/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0891 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1216 - val_sparse_categorical_accuracy: 0.9556 - lr: 2.5000e-04
Epoch 124/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0936 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.2818 - val_sparse_categorical_accuracy: 0.8669 - lr: 2.5000e-04
Epoch 125/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0964 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.2752 - val_sparse_categorical_accuracy: 0.8696 - lr: 2.5000e-04
Epoch 126/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0890 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0924 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 127/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1247 - val_sparse_categorical_accuracy: 0.9515 - lr: 2.5000e-04
Epoch 128/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0895 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.1127 - val_sparse_categorical_accuracy: 0.9598 - lr: 2.5000e-04
Epoch 129/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0921 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1466 - val_sparse_categorical_accuracy: 0.9501 - lr: 2.5000e-04
Epoch 130/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.1108 - val_sparse_categorical_accuracy: 0.9584 - lr: 2.5000e-04
Epoch 131/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0955 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.0822 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 132/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0915 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1170 - val_sparse_categorical_accuracy: 0.9570 - lr: 2.5000e-04
Epoch 133/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0901 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.0876 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 134/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0992 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9792 - lr: 2.5000e-04
Epoch 135/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0916 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.2946 - val_sparse_categorical_accuracy: 0.8724 - lr: 2.5000e-04
Epoch 136/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0899 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1794 - val_sparse_categorical_accuracy: 0.9126 - lr: 2.5000e-04
Epoch 137/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0866 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.3517 - val_sparse_categorical_accuracy: 0.8322 - lr: 2.5000e-04
Epoch 138/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0895 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 139/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0945 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.1689 - val_sparse_categorical_accuracy: 0.9320 - lr: 2.5000e-04
Epoch 140/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0892 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.2925 - val_sparse_categorical_accuracy: 0.8585 - lr: 2.5000e-04
Epoch 141/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0911 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1387 - val_sparse_categorical_accuracy: 0.9404 - lr: 2.5000e-04
Epoch 142/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0914 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 143/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0849 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1157 - val_sparse_categorical_accuracy: 0.9584 - lr: 2.5000e-04
Epoch 144/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0970 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 145/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1225 - val_sparse_categorical_accuracy: 0.9584 - lr: 2.5000e-04
Epoch 146/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 147/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0873 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1188 - val_sparse_categorical_accuracy: 0.9501 - lr: 2.5000e-04
Epoch 148/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0964 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.0940 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 149/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0863 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.0897 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 150/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0861 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.2606 - val_sparse_categorical_accuracy: 0.8821 - lr: 2.5000e-04
Epoch 151/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0826 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.6422 - val_sparse_categorical_accuracy: 0.7337 - lr: 2.5000e-04
Epoch 152/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0823 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1239 - val_sparse_categorical_accuracy: 0.9570 - lr: 1.2500e-04
Epoch 153/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0831 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1206 - val_sparse_categorical_accuracy: 0.9570 - lr: 1.2500e-04
Epoch 154/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0870 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0831 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 155/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0809 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1099 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.2500e-04
Epoch 156/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0821 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1036 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 157/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0854 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.2088 - val_sparse_categorical_accuracy: 0.9126 - lr: 1.2500e-04
Epoch 158/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1329 - val_sparse_categorical_accuracy: 0.9556 - lr: 1.2500e-04
Epoch 159/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0815 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1140 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 160/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0826 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.0814 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 161/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0805 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 162/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0823 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1639 - val_sparse_categorical_accuracy: 0.9320 - lr: 1.2500e-04
Epoch 163/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.2226 - val_sparse_categorical_accuracy: 0.8918 - lr: 1.2500e-04
Epoch 164/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0807 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1018 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 165/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.2077 - val_sparse_categorical_accuracy: 0.9154 - lr: 1.2500e-04
Epoch 166/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0821 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.0912 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 167/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0858 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 168/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0826 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0808 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 169/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0875 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1032 - val_sparse_categorical_accuracy: 0.9639 - lr: 1.2500e-04
Epoch 170/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0819 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 171/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0792 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 172/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0817 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1236 - val_sparse_categorical_accuracy: 0.9515 - lr: 1.2500e-04
Epoch 173/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0816 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.2500e-04
Epoch 174/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0843 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0940 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.2500e-04
Epoch 175/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0793 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.0844 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 176/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.0830 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 177/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0853 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.0803 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 178/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0773 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.2500e-04
Epoch 179/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0844 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 180/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1043 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 181/500
90/90 [==============================] - 2s 19ms/step - loss: 0.0865 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.0996 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 182/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1993 - val_sparse_categorical_accuracy: 0.9196 - lr: 1.2500e-04
Epoch 183/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0831 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 184/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1043 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.2500e-04
Epoch 185/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1067 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.2500e-04
Epoch 186/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0839 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0904 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.2500e-04
Epoch 187/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0815 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 188/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.2500e-04
Epoch 189/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0790 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0786 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 190/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0885 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1109 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.2500e-04
Epoch 191/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0836 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 192/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0816 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.0880 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 193/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0766 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1058 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 194/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0872 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 195/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0811 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.2232 - val_sparse_categorical_accuracy: 0.8904 - lr: 1.2500e-04
Epoch 196/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0858 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1090 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 197/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 198/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0774 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1367 - val_sparse_categorical_accuracy: 0.9473 - lr: 1.2500e-04
Epoch 199/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 200/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0766 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0816 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 201/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0839 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 202/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0859 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0812 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 203/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0753 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0841 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 204/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0815 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1013 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 205/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0764 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 206/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0973 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 207/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0985 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.2500e-04
Epoch 208/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0732 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0784 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 209/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1183 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.2500e-04
Epoch 210/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1017 - val_sparse_categorical_accuracy: 0.9570 - lr: 1.2500e-04
Epoch 211/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0785 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 212/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0872 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 213/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0836 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 214/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1039 - val_sparse_categorical_accuracy: 0.9639 - lr: 1.2500e-04
Epoch 215/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0806 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0893 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.2500e-04
Epoch 216/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1266 - val_sparse_categorical_accuracy: 0.9528 - lr: 1.2500e-04
Epoch 217/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1097 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.2500e-04
Epoch 218/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0765 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 219/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0785 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.2249 - val_sparse_categorical_accuracy: 0.9071 - lr: 1.2500e-04
Epoch 220/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1039 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 221/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0813 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 222/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0769 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0827 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 223/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0803 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0796 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 224/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1011 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 225/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0767 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 226/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.0815 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 227/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1021 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 228/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 229/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0769 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0861 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 230/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1018 - val_sparse_categorical_accuracy: 0.9639 - lr: 1.0000e-04
Epoch 231/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0998 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 232/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.0896 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.0000e-04
Epoch 233/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0842 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 234/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0875 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 235/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0920 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 236/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0698 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.0000e-04
Epoch 237/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0728 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0834 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 238/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 239/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0751 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 240/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0716 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1447 - val_sparse_categorical_accuracy: 0.9473 - lr: 1.0000e-04
Epoch 241/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0751 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 242/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0799 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0882 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 243/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 244/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0722 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0830 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 245/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0881 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 246/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 247/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0820 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0887 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 248/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0847 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 249/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1059 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.0000e-04
Epoch 250/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0868 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 251/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0785 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0816 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 252/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0778 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0910 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 253/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0900 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 254/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0840 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 255/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0745 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.0000e-04
Epoch 256/500
90/90 [==============================] - 1s 17ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.0816 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 257/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0762 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 258/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0684 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0796 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 259/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0730 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1046 - val_sparse_categorical_accuracy: 0.9570 - lr: 1.0000e-04
Epoch 260/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0924 - val_sparse_categorical_accuracy: 0.9556 - lr: 1.0000e-04
Epoch 261/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0747 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0794 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 262/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0855 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 263/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0761 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 264/500
90/90 [==============================] - 2s 19ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0808 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 265/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0847 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 266/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0735 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0856 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 267/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0836 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 268/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0737 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0848 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.0000e-04
Epoch 269/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1046 - val_sparse_categorical_accuracy: 0.9639 - lr: 1.0000e-04
Epoch 270/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0706 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0823 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 271/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 272/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0772 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1056 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.0000e-04
Epoch 273/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 274/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0765 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1293 - val_sparse_categorical_accuracy: 0.9515 - lr: 1.0000e-04
Epoch 275/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0730 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0829 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 276/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0730 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0869 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 277/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1169 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.0000e-04
Epoch 278/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 279/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0897 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 280/500
90/90 [==============================] - 2s 18ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1171 - val_sparse_categorical_accuracy: 0.9584 - lr: 1.0000e-04
Epoch 281/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0735 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.0804 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 282/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0717 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1286 - val_sparse_categorical_accuracy: 0.9473 - lr: 1.0000e-04
Epoch 283/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0647 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 284/500
90/90 [==============================] - 2s 17ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0885 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 285/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0686 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1085 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.0000e-04
Epoch 286/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0707 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 287/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0691 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 288/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0702 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0795 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 289/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0683 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0865 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 290/500
90/90 [==============================] - 1s 16ms/step - loss: 0.0759 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 291/500
90/90 [==============================] - 1s 15ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1573 - val_sparse_categorical_accuracy: 0.9445 - lr: 1.0000e-04
Epoch 291: early stopping

```
</div>
---
## Evaluate model on test data


```python
model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
```

<div class="k-default-codeblock">
```
42/42 [==============================] - 0s 8ms/step - loss: 0.0913 - sparse_categorical_accuracy: 0.9727
Test accuracy 0.9727272987365723
Test loss 0.09134132415056229

```
</div>
---
## Plot the model's training and validation loss


```python
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
```


    
![png](/img/examples/timeseries/timeseries_classification_from_scratch/timeseries_classification_from_scratch_23_0.png)
    


We can see how the training accuracy reaches almost 0.95 after 100 epochs.
However, by observing the validation accuracy we can see how the network still needs
training until it reaches almost 0.97 for both the validation and the training accuracy
after 200 epochs. Beyond the 200th epoch, if we continue on training, the validation
accuracy will start decreasing while the training accuracy will continue on increasing:
the model starts overfitting.

Example available on HuggingFace:
| Trained Model | Demo |
|------|------|
| [![🤗 Model - Timeseries classification from scratch](https://img.shields.io/badge/🤗_Model-Timeseries_classification_from_scratch-black)](https://huggingface.co/keras-io/timeseries-classification-from-scratch) | [![🤗 Spaces - Timeseries classification from scratch](https://img.shields.io/badge/🤗_Spaces-Timeseries_classification_from_scratch-black)](https://huggingface.co/spaces/keras-io/timeseries-classification-from-scratch) |
