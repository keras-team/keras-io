# Timeseries classification from scratch

**Author:** [hfawaz](https://github.com/hfawaz/)<br>
**Date created:** 2020/07/21<br>
**Last modified:** 2020/07/16<br>
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
('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.')

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
90/90 [==============================] - 1s 8ms/step - loss: 0.5531 - sparse_categorical_accuracy: 0.7017 - val_loss: 0.7335 - val_sparse_categorical_accuracy: 0.4882
Epoch 2/500
90/90 [==============================] - 1s 6ms/step - loss: 0.4520 - sparse_categorical_accuracy: 0.7729 - val_loss: 0.7446 - val_sparse_categorical_accuracy: 0.4882
Epoch 3/500
90/90 [==============================] - 1s 6ms/step - loss: 0.4404 - sparse_categorical_accuracy: 0.7733 - val_loss: 0.7706 - val_sparse_categorical_accuracy: 0.4882
Epoch 4/500
90/90 [==============================] - 1s 6ms/step - loss: 0.4234 - sparse_categorical_accuracy: 0.7899 - val_loss: 0.9741 - val_sparse_categorical_accuracy: 0.4882
Epoch 5/500
90/90 [==============================] - 1s 6ms/step - loss: 0.4180 - sparse_categorical_accuracy: 0.7972 - val_loss: 0.6679 - val_sparse_categorical_accuracy: 0.5936
Epoch 6/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3988 - sparse_categorical_accuracy: 0.8066 - val_loss: 0.5399 - val_sparse_categorical_accuracy: 0.6990
Epoch 7/500
90/90 [==============================] - 1s 6ms/step - loss: 0.4012 - sparse_categorical_accuracy: 0.8024 - val_loss: 0.4051 - val_sparse_categorical_accuracy: 0.8225
Epoch 8/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3903 - sparse_categorical_accuracy: 0.8080 - val_loss: 0.9671 - val_sparse_categorical_accuracy: 0.5340
Epoch 9/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3948 - sparse_categorical_accuracy: 0.7986 - val_loss: 0.5778 - val_sparse_categorical_accuracy: 0.6436
Epoch 10/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3731 - sparse_categorical_accuracy: 0.8260 - val_loss: 0.4307 - val_sparse_categorical_accuracy: 0.7698
Epoch 11/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3645 - sparse_categorical_accuracy: 0.8260 - val_loss: 0.4010 - val_sparse_categorical_accuracy: 0.7698
Epoch 12/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3666 - sparse_categorical_accuracy: 0.8247 - val_loss: 0.3574 - val_sparse_categorical_accuracy: 0.8350
Epoch 13/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3618 - sparse_categorical_accuracy: 0.8271 - val_loss: 0.3942 - val_sparse_categorical_accuracy: 0.8044
Epoch 14/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3619 - sparse_categorical_accuracy: 0.8257 - val_loss: 0.4104 - val_sparse_categorical_accuracy: 0.7906
Epoch 15/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3353 - sparse_categorical_accuracy: 0.8521 - val_loss: 0.3819 - val_sparse_categorical_accuracy: 0.7684
Epoch 16/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3287 - sparse_categorical_accuracy: 0.8514 - val_loss: 0.3776 - val_sparse_categorical_accuracy: 0.8252
Epoch 17/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3299 - sparse_categorical_accuracy: 0.8545 - val_loss: 0.3555 - val_sparse_categorical_accuracy: 0.8350
Epoch 18/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3206 - sparse_categorical_accuracy: 0.8601 - val_loss: 0.4051 - val_sparse_categorical_accuracy: 0.7906
Epoch 19/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3125 - sparse_categorical_accuracy: 0.8608 - val_loss: 0.3792 - val_sparse_categorical_accuracy: 0.8114
Epoch 20/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3052 - sparse_categorical_accuracy: 0.8750 - val_loss: 0.3448 - val_sparse_categorical_accuracy: 0.8377
Epoch 21/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3023 - sparse_categorical_accuracy: 0.8736 - val_loss: 0.3325 - val_sparse_categorical_accuracy: 0.8363
Epoch 22/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2955 - sparse_categorical_accuracy: 0.8736 - val_loss: 0.3447 - val_sparse_categorical_accuracy: 0.8225
Epoch 23/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2934 - sparse_categorical_accuracy: 0.8788 - val_loss: 0.2943 - val_sparse_categorical_accuracy: 0.8779
Epoch 24/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2972 - sparse_categorical_accuracy: 0.8715 - val_loss: 0.4946 - val_sparse_categorical_accuracy: 0.7462
Epoch 25/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2800 - sparse_categorical_accuracy: 0.8865 - val_loss: 0.2860 - val_sparse_categorical_accuracy: 0.8821
Epoch 26/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2752 - sparse_categorical_accuracy: 0.8847 - val_loss: 0.2924 - val_sparse_categorical_accuracy: 0.8655
Epoch 27/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2769 - sparse_categorical_accuracy: 0.8847 - val_loss: 0.6254 - val_sparse_categorical_accuracy: 0.6879
Epoch 28/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2821 - sparse_categorical_accuracy: 0.8799 - val_loss: 0.2764 - val_sparse_categorical_accuracy: 0.8821
Epoch 29/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2713 - sparse_categorical_accuracy: 0.8892 - val_loss: 0.7015 - val_sparse_categorical_accuracy: 0.6422
Epoch 30/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2633 - sparse_categorical_accuracy: 0.8885 - val_loss: 0.8508 - val_sparse_categorical_accuracy: 0.7254
Epoch 31/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2673 - sparse_categorical_accuracy: 0.8896 - val_loss: 0.4354 - val_sparse_categorical_accuracy: 0.7725
Epoch 32/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2518 - sparse_categorical_accuracy: 0.8997 - val_loss: 0.9172 - val_sparse_categorical_accuracy: 0.6394
Epoch 33/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2484 - sparse_categorical_accuracy: 0.9024 - val_loss: 0.5055 - val_sparse_categorical_accuracy: 0.7531
Epoch 34/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2352 - sparse_categorical_accuracy: 0.9059 - val_loss: 0.6289 - val_sparse_categorical_accuracy: 0.7115
Epoch 35/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2389 - sparse_categorical_accuracy: 0.9104 - val_loss: 0.2776 - val_sparse_categorical_accuracy: 0.8946
Epoch 36/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9122 - val_loss: 1.3105 - val_sparse_categorical_accuracy: 0.6408
Epoch 37/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2237 - sparse_categorical_accuracy: 0.9125 - val_loss: 0.4860 - val_sparse_categorical_accuracy: 0.7628
Epoch 38/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2008 - sparse_categorical_accuracy: 0.9281 - val_loss: 0.5553 - val_sparse_categorical_accuracy: 0.7226
Epoch 39/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1999 - sparse_categorical_accuracy: 0.9233 - val_loss: 0.4511 - val_sparse_categorical_accuracy: 0.8058
Epoch 40/500
90/90 [==============================] - 0s 6ms/step - loss: 0.1857 - sparse_categorical_accuracy: 0.9330 - val_loss: 0.2912 - val_sparse_categorical_accuracy: 0.8516
Epoch 41/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1736 - sparse_categorical_accuracy: 0.9399 - val_loss: 0.9930 - val_sparse_categorical_accuracy: 0.5506
Epoch 42/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1649 - sparse_categorical_accuracy: 0.9396 - val_loss: 0.5852 - val_sparse_categorical_accuracy: 0.7198
Epoch 43/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1501 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.1911 - val_sparse_categorical_accuracy: 0.9168
Epoch 44/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1512 - sparse_categorical_accuracy: 0.9455 - val_loss: 0.8169 - val_sparse_categorical_accuracy: 0.6130
Epoch 45/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1358 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.4748 - val_sparse_categorical_accuracy: 0.7795
Epoch 46/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1401 - sparse_categorical_accuracy: 0.9535 - val_loss: 1.7678 - val_sparse_categorical_accuracy: 0.5881
Epoch 47/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1444 - sparse_categorical_accuracy: 0.9545 - val_loss: 1.7005 - val_sparse_categorical_accuracy: 0.5950
Epoch 48/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1320 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.1550 - val_sparse_categorical_accuracy: 0.9431
Epoch 49/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1333 - sparse_categorical_accuracy: 0.9576 - val_loss: 0.1665 - val_sparse_categorical_accuracy: 0.9362
Epoch 50/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1367 - sparse_categorical_accuracy: 0.9549 - val_loss: 0.4227 - val_sparse_categorical_accuracy: 0.8308
Epoch 51/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1391 - sparse_categorical_accuracy: 0.9503 - val_loss: 0.1729 - val_sparse_categorical_accuracy: 0.9390
Epoch 52/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1237 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.1338 - val_sparse_categorical_accuracy: 0.9487
Epoch 53/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1397 - sparse_categorical_accuracy: 0.9531 - val_loss: 0.1667 - val_sparse_categorical_accuracy: 0.9487
Epoch 54/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1205 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.2904 - val_sparse_categorical_accuracy: 0.8821
Epoch 55/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1302 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.9437 - val_sparse_categorical_accuracy: 0.7060
Epoch 56/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1241 - sparse_categorical_accuracy: 0.9580 - val_loss: 0.1346 - val_sparse_categorical_accuracy: 0.9501
Epoch 57/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1158 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.9489 - val_sparse_categorical_accuracy: 0.6907
Epoch 58/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1175 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.6089 - val_sparse_categorical_accuracy: 0.7212
Epoch 59/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1160 - sparse_categorical_accuracy: 0.9611 - val_loss: 0.1294 - val_sparse_categorical_accuracy: 0.9487
Epoch 60/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1096 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1527 - val_sparse_categorical_accuracy: 0.9417
Epoch 61/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1163 - sparse_categorical_accuracy: 0.9611 - val_loss: 0.5554 - val_sparse_categorical_accuracy: 0.7684
Epoch 62/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1090 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.2433 - val_sparse_categorical_accuracy: 0.8904
Epoch 63/500
90/90 [==============================] - 0s 6ms/step - loss: 0.1105 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.3426 - val_sparse_categorical_accuracy: 0.8571
Epoch 64/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1058 - sparse_categorical_accuracy: 0.9667 - val_loss: 2.1389 - val_sparse_categorical_accuracy: 0.5520
Epoch 65/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1037 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.3875 - val_sparse_categorical_accuracy: 0.8738
Epoch 66/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1135 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1783 - val_sparse_categorical_accuracy: 0.9459
Epoch 67/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1006 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1462 - val_sparse_categorical_accuracy: 0.9515
Epoch 68/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0994 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.1140 - val_sparse_categorical_accuracy: 0.9584
Epoch 69/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1095 - sparse_categorical_accuracy: 0.9635 - val_loss: 1.6500 - val_sparse_categorical_accuracy: 0.5589
Epoch 70/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1118 - sparse_categorical_accuracy: 0.9628 - val_loss: 1.3355 - val_sparse_categorical_accuracy: 0.6768
Epoch 71/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1155 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.3167 - val_sparse_categorical_accuracy: 0.8793
Epoch 72/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1041 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1329 - val_sparse_categorical_accuracy: 0.9417
Epoch 73/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1001 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1385 - val_sparse_categorical_accuracy: 0.9417
Epoch 74/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0997 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1369 - val_sparse_categorical_accuracy: 0.9473
Epoch 75/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1051 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.5135 - val_sparse_categorical_accuracy: 0.7781
Epoch 76/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0945 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1440 - val_sparse_categorical_accuracy: 0.9556
Epoch 77/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1081 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.2210 - val_sparse_categorical_accuracy: 0.9196
Epoch 78/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1109 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.2181 - val_sparse_categorical_accuracy: 0.9196
Epoch 79/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1047 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.2074 - val_sparse_categorical_accuracy: 0.9237
Epoch 80/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1035 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.3792 - val_sparse_categorical_accuracy: 0.8571
Epoch 81/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1040 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.7353 - val_sparse_categorical_accuracy: 0.7420
Epoch 82/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1106 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.2948 - val_sparse_categorical_accuracy: 0.9140
Epoch 83/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1066 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.1338 - val_sparse_categorical_accuracy: 0.9570
Epoch 84/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0988 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.1095 - val_sparse_categorical_accuracy: 0.9570
Epoch 85/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1065 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1717 - val_sparse_categorical_accuracy: 0.9417
Epoch 86/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1087 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.1206 - val_sparse_categorical_accuracy: 0.9570
Epoch 87/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.4285 - val_sparse_categorical_accuracy: 0.8474
Epoch 88/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0984 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1589 - val_sparse_categorical_accuracy: 0.9334
Epoch 89/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1023 - sparse_categorical_accuracy: 0.9701 - val_loss: 1.5442 - val_sparse_categorical_accuracy: 0.6782
Epoch 90/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0995 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1211 - val_sparse_categorical_accuracy: 0.9528
Epoch 91/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9556
Epoch 92/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0919 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.2109 - val_sparse_categorical_accuracy: 0.9140
Epoch 93/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0890 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1509 - val_sparse_categorical_accuracy: 0.9431
Epoch 94/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0958 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.1761 - val_sparse_categorical_accuracy: 0.9417
Epoch 95/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1000 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1466 - val_sparse_categorical_accuracy: 0.9293
Epoch 96/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0913 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.6963 - val_sparse_categorical_accuracy: 0.7725
Epoch 97/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0954 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.3042 - val_sparse_categorical_accuracy: 0.8738
Epoch 98/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0866 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1115 - val_sparse_categorical_accuracy: 0.9584
Epoch 99/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1017 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.1195 - val_sparse_categorical_accuracy: 0.9584
Epoch 100/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1012 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1975 - val_sparse_categorical_accuracy: 0.9196
Epoch 101/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1058 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1960 - val_sparse_categorical_accuracy: 0.9487
Epoch 102/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0914 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1086 - val_sparse_categorical_accuracy: 0.9598
Epoch 103/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0907 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1117 - val_sparse_categorical_accuracy: 0.9584
Epoch 104/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0959 - sparse_categorical_accuracy: 0.9674 - val_loss: 3.9192 - val_sparse_categorical_accuracy: 0.4993
Epoch 105/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.1232 - val_sparse_categorical_accuracy: 0.9473
Epoch 106/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0953 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.1328 - val_sparse_categorical_accuracy: 0.9584
Epoch 107/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1480 - val_sparse_categorical_accuracy: 0.9542
Epoch 108/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0865 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1095 - val_sparse_categorical_accuracy: 0.9598
Epoch 109/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9681 - val_loss: 3.4316 - val_sparse_categorical_accuracy: 0.6422
Epoch 110/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1015 - sparse_categorical_accuracy: 0.9632 - val_loss: 4.1126 - val_sparse_categorical_accuracy: 0.4965
Epoch 111/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0882 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.1968 - val_sparse_categorical_accuracy: 0.9390
Epoch 112/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0778 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1051 - val_sparse_categorical_accuracy: 0.9584
Epoch 113/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0784 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1120 - val_sparse_categorical_accuracy: 0.9612
Epoch 114/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0765 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1347 - val_sparse_categorical_accuracy: 0.9556
Epoch 115/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0771 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1268 - val_sparse_categorical_accuracy: 0.9556
Epoch 116/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1014 - val_sparse_categorical_accuracy: 0.9626
Epoch 117/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0995 - val_sparse_categorical_accuracy: 0.9695
Epoch 118/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1022 - val_sparse_categorical_accuracy: 0.9598
Epoch 119/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.2318 - val_sparse_categorical_accuracy: 0.9098
Epoch 120/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0751 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.3361 - val_sparse_categorical_accuracy: 0.8793
Epoch 121/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0708 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1739 - val_sparse_categorical_accuracy: 0.9362
Epoch 122/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1351 - val_sparse_categorical_accuracy: 0.9556
Epoch 123/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0724 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1064 - val_sparse_categorical_accuracy: 0.9556
Epoch 124/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1159 - val_sparse_categorical_accuracy: 0.9598
Epoch 125/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0806 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1268 - val_sparse_categorical_accuracy: 0.9612
Epoch 126/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1175 - val_sparse_categorical_accuracy: 0.9528
Epoch 127/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0741 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1049 - val_sparse_categorical_accuracy: 0.9612
Epoch 128/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1756 - val_sparse_categorical_accuracy: 0.9376
Epoch 129/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1165 - val_sparse_categorical_accuracy: 0.9639
Epoch 130/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1398 - val_sparse_categorical_accuracy: 0.9417
Epoch 131/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1193 - val_sparse_categorical_accuracy: 0.9459
Epoch 132/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0741 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1661 - val_sparse_categorical_accuracy: 0.9473
Epoch 133/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1016 - val_sparse_categorical_accuracy: 0.9612
Epoch 134/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0673 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1049 - val_sparse_categorical_accuracy: 0.9584
Epoch 135/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1109 - val_sparse_categorical_accuracy: 0.9515
Epoch 136/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0673 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1198 - val_sparse_categorical_accuracy: 0.9542
Epoch 137/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0679 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1130 - val_sparse_categorical_accuracy: 0.9528
Epoch 138/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0717 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1009 - val_sparse_categorical_accuracy: 0.9612
Epoch 139/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1046 - val_sparse_categorical_accuracy: 0.9528
Epoch 140/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0711 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0977 - val_sparse_categorical_accuracy: 0.9639
Epoch 141/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0719 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1071 - val_sparse_categorical_accuracy: 0.9612
Epoch 142/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0663 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9612
Epoch 143/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1131 - val_sparse_categorical_accuracy: 0.9626
Epoch 144/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9626
Epoch 145/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0653 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0935 - val_sparse_categorical_accuracy: 0.9653
Epoch 146/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0616 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9556
Epoch 147/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0643 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9584
Epoch 148/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0944 - val_sparse_categorical_accuracy: 0.9639
Epoch 149/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1311 - val_sparse_categorical_accuracy: 0.9501
Epoch 150/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1715 - val_sparse_categorical_accuracy: 0.9390
Epoch 151/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0658 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9612
Epoch 152/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0652 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9639
Epoch 153/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0640 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0996 - val_sparse_categorical_accuracy: 0.9598
Epoch 154/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0659 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9612
Epoch 155/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0666 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1490 - val_sparse_categorical_accuracy: 0.9501
Epoch 156/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0659 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9570
Epoch 157/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0650 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9570
Epoch 158/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0626 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0965 - val_sparse_categorical_accuracy: 0.9612
Epoch 159/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0645 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9570
Epoch 160/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0691 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9626
Epoch 161/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0615 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9612
Epoch 162/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0625 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9626
Epoch 163/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0601 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0989 - val_sparse_categorical_accuracy: 0.9584
Epoch 164/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0624 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1512 - val_sparse_categorical_accuracy: 0.9515
Epoch 165/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0641 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0986 - val_sparse_categorical_accuracy: 0.9584
Epoch 166/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9598
Epoch 167/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0607 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1085 - val_sparse_categorical_accuracy: 0.9626
Epoch 168/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9639
Epoch 169/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1078 - val_sparse_categorical_accuracy: 0.9626
Epoch 170/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0608 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0951 - val_sparse_categorical_accuracy: 0.9626
Epoch 171/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0612 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1004 - val_sparse_categorical_accuracy: 0.9612
Epoch 172/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0622 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9653
Epoch 173/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0622 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9639
Epoch 174/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0600 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1019 - val_sparse_categorical_accuracy: 0.9639
Epoch 175/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0591 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1238 - val_sparse_categorical_accuracy: 0.9626
Epoch 176/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0588 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9639
Epoch 177/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0598 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1138 - val_sparse_categorical_accuracy: 0.9626
Epoch 178/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0938 - val_sparse_categorical_accuracy: 0.9639
Epoch 179/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0634 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0966 - val_sparse_categorical_accuracy: 0.9639
Epoch 180/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1033 - val_sparse_categorical_accuracy: 0.9653
Epoch 181/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0601 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9626
Epoch 182/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9626
Epoch 183/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0987 - val_sparse_categorical_accuracy: 0.9626
Epoch 184/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0907 - val_sparse_categorical_accuracy: 0.9626
Epoch 185/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9626
Epoch 186/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0571 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9626
Epoch 187/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0577 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9626
Epoch 188/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0634 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1014 - val_sparse_categorical_accuracy: 0.9667
Epoch 189/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0906 - val_sparse_categorical_accuracy: 0.9639
Epoch 190/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0571 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9626
Epoch 191/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0602 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0903 - val_sparse_categorical_accuracy: 0.9626
Epoch 192/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0915 - val_sparse_categorical_accuracy: 0.9639
Epoch 193/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0574 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0914 - val_sparse_categorical_accuracy: 0.9639
Epoch 194/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0530 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9626
Epoch 195/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0925 - val_sparse_categorical_accuracy: 0.9653
Epoch 196/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1018 - val_sparse_categorical_accuracy: 0.9639
Epoch 197/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0562 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1003 - val_sparse_categorical_accuracy: 0.9626
Epoch 198/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9612
Epoch 199/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0602 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1001 - val_sparse_categorical_accuracy: 0.9667
Epoch 200/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0927 - val_sparse_categorical_accuracy: 0.9584
Epoch 201/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1226 - val_sparse_categorical_accuracy: 0.9612
Epoch 202/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9612
Epoch 203/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0602 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9639
Epoch 204/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0956 - val_sparse_categorical_accuracy: 0.9626
Epoch 205/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0561 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9639
Epoch 206/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1132 - val_sparse_categorical_accuracy: 0.9639
Epoch 207/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0930 - val_sparse_categorical_accuracy: 0.9653
Epoch 208/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0615 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9626
Epoch 209/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0555 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0900 - val_sparse_categorical_accuracy: 0.9626
Epoch 210/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9612
Epoch 211/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0615 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9626
Epoch 212/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9612
Epoch 213/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9653
Epoch 214/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1143 - val_sparse_categorical_accuracy: 0.9639
Epoch 215/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0965 - val_sparse_categorical_accuracy: 0.9626
Epoch 216/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0588 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9639
Epoch 217/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0549 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9612
Epoch 218/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9612
Epoch 219/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0555 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1217 - val_sparse_categorical_accuracy: 0.9598
Epoch 220/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0572 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9653
Epoch 221/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9653
Epoch 222/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1163 - val_sparse_categorical_accuracy: 0.9639
Epoch 223/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0556 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0955 - val_sparse_categorical_accuracy: 0.9626
Epoch 224/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9598
Epoch 225/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0915 - val_sparse_categorical_accuracy: 0.9667
Epoch 226/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9626
Epoch 227/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0528 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9639
Epoch 228/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1019 - val_sparse_categorical_accuracy: 0.9639
Epoch 229/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0572 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9639
Epoch 230/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9639
Epoch 231/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9626
Epoch 232/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0954 - val_sparse_categorical_accuracy: 0.9653
Epoch 233/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0965 - val_sparse_categorical_accuracy: 0.9626
Epoch 234/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0945 - val_sparse_categorical_accuracy: 0.9639
Epoch 235/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0528 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0925 - val_sparse_categorical_accuracy: 0.9639
Epoch 236/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0974 - val_sparse_categorical_accuracy: 0.9626
Epoch 237/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0529 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9612
Epoch 238/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0552 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0961 - val_sparse_categorical_accuracy: 0.9626
Epoch 239/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9598
Epoch 240/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0935 - val_sparse_categorical_accuracy: 0.9639
Epoch 241/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9626
Epoch 242/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0488 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9626
Epoch 243/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0935 - val_sparse_categorical_accuracy: 0.9626
Epoch 244/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0505 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0945 - val_sparse_categorical_accuracy: 0.9639
Epoch 245/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0483 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0952 - val_sparse_categorical_accuracy: 0.9584
Epoch 246/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0524 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9653
Epoch 247/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0507 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0934 - val_sparse_categorical_accuracy: 0.9653
Epoch 248/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0553 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9598
Epoch 249/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0577 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9612
Epoch 250/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9626
Epoch 251/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0509 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9626
Epoch 252/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0571 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9612
Epoch 253/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1017 - val_sparse_categorical_accuracy: 0.9639
Epoch 254/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0551 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0930 - val_sparse_categorical_accuracy: 0.9639
Epoch 255/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0896 - val_sparse_categorical_accuracy: 0.9653
Epoch 256/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9612
Epoch 257/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9612
Epoch 258/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9626
Epoch 259/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9612
Epoch 260/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0934 - val_sparse_categorical_accuracy: 0.9612
Epoch 261/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0510 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0914 - val_sparse_categorical_accuracy: 0.9639
Epoch 262/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9653
Epoch 263/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0956 - val_sparse_categorical_accuracy: 0.9653
Epoch 264/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9598
Epoch 265/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0529 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0928 - val_sparse_categorical_accuracy: 0.9626
Epoch 266/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0978 - val_sparse_categorical_accuracy: 0.9639
Epoch 267/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9639
Epoch 268/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0538 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9626
Epoch 269/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0928 - val_sparse_categorical_accuracy: 0.9598
Epoch 270/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0553 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9626
Epoch 271/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0501 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9639
Epoch 272/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0575 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0903 - val_sparse_categorical_accuracy: 0.9639
Epoch 273/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0490 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1155 - val_sparse_categorical_accuracy: 0.9626
Epoch 274/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0553 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9653
Epoch 275/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0915 - val_sparse_categorical_accuracy: 0.9598
Epoch 276/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9639
Epoch 277/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0606 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1049 - val_sparse_categorical_accuracy: 0.9639
Epoch 278/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0488 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9598
Epoch 279/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0934 - val_sparse_categorical_accuracy: 0.9639
Epoch 280/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0493 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0997 - val_sparse_categorical_accuracy: 0.9626
Epoch 281/500
90/90 [==============================] - 0s 5ms/step - loss: 0.0485 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9626
Epoch 282/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0493 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0906 - val_sparse_categorical_accuracy: 0.9626
Epoch 283/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0491 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0919 - val_sparse_categorical_accuracy: 0.9653
Epoch 284/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0482 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0895 - val_sparse_categorical_accuracy: 0.9639
Epoch 285/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0505 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9612
Epoch 286/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9639
Epoch 287/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0935 - val_sparse_categorical_accuracy: 0.9639
Epoch 288/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0527 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9639
Epoch 289/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0961 - val_sparse_categorical_accuracy: 0.9667
Epoch 290/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9612
Epoch 291/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9653
Epoch 292/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0489 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9653
Epoch 293/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9878 - val_loss: 0.0907 - val_sparse_categorical_accuracy: 0.9612
Epoch 294/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0892 - val_sparse_categorical_accuracy: 0.9639
Epoch 295/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0502 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9639
Epoch 296/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0504 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9639
Epoch 297/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0532 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9639
Epoch 298/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9584
Epoch 299/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0478 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1001 - val_sparse_categorical_accuracy: 0.9612
Epoch 300/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0929 - val_sparse_categorical_accuracy: 0.9639
Epoch 301/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0507 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0935 - val_sparse_categorical_accuracy: 0.9653
Epoch 302/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0897 - val_sparse_categorical_accuracy: 0.9612
Epoch 303/500
90/90 [==============================] - 0s 5ms/step - loss: 0.0480 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1003 - val_sparse_categorical_accuracy: 0.9612
Epoch 304/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0538 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0997 - val_sparse_categorical_accuracy: 0.9612
Epoch 305/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0528 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1028 - val_sparse_categorical_accuracy: 0.9626
Epoch 306/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0507 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9612
Epoch 307/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9639
Epoch 308/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0929 - val_sparse_categorical_accuracy: 0.9681
Epoch 309/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0510 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0904 - val_sparse_categorical_accuracy: 0.9626
Epoch 310/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0518 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0967 - val_sparse_categorical_accuracy: 0.9598
Epoch 311/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0521 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0945 - val_sparse_categorical_accuracy: 0.9626
Epoch 312/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0586 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9626
Epoch 313/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0470 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9598
Epoch 314/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0533 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9598
Epoch 315/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0502 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9639
Epoch 316/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0463 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0912 - val_sparse_categorical_accuracy: 0.9639
Epoch 317/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9626
Epoch 318/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0916 - val_sparse_categorical_accuracy: 0.9639
Epoch 319/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0919 - val_sparse_categorical_accuracy: 0.9639
Epoch 320/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0446 - sparse_categorical_accuracy: 0.9906 - val_loss: 0.0901 - val_sparse_categorical_accuracy: 0.9626
Epoch 321/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0527 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0910 - val_sparse_categorical_accuracy: 0.9598
Epoch 322/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0476 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.1029 - val_sparse_categorical_accuracy: 0.9598
Epoch 323/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0505 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0939 - val_sparse_categorical_accuracy: 0.9626
Epoch 324/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0505 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0900 - val_sparse_categorical_accuracy: 0.9612
Epoch 325/500
90/90 [==============================] - 0s 6ms/step - loss: 0.0516 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.1024 - val_sparse_categorical_accuracy: 0.9626
Epoch 326/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9598
Epoch 327/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0509 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9626
Epoch 328/500
90/90 [==============================] - 0s 5ms/step - loss: 0.0427 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9639
Epoch 329/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9612
Epoch 330/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9598
Epoch 331/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0485 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9626
Epoch 332/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9612
Epoch 333/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9639
Epoch 334/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0500 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1073 - val_sparse_categorical_accuracy: 0.9626
Epoch 335/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0506 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9612
Epoch 336/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9639
Epoch 337/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9626
Epoch 338/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0510 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0904 - val_sparse_categorical_accuracy: 0.9639
Epoch 339/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0521 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9584
Epoch 340/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9626
Epoch 341/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0468 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0990 - val_sparse_categorical_accuracy: 0.9626
Epoch 342/500
90/90 [==============================] - 0s 5ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9653
Epoch 343/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0518 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9598
Epoch 344/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0488 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9639
Epoch 345/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0523 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1467 - val_sparse_categorical_accuracy: 0.9515
Epoch 346/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0482 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0917 - val_sparse_categorical_accuracy: 0.9667
Epoch 347/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0492 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1134 - val_sparse_categorical_accuracy: 0.9626
Epoch 348/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0455 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9612
Epoch 349/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0462 - sparse_categorical_accuracy: 0.9896 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9667
Epoch 350/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0912 - val_sparse_categorical_accuracy: 0.9639
Epoch 351/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0462 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.0932 - val_sparse_categorical_accuracy: 0.9626
Epoch 352/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9653
Epoch 353/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0455 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0945 - val_sparse_categorical_accuracy: 0.9612
Epoch 354/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0452 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9598
Epoch 355/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0430 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0903 - val_sparse_categorical_accuracy: 0.9626
Epoch 356/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0471 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.1045 - val_sparse_categorical_accuracy: 0.9626
Epoch 357/500
90/90 [==============================] - 0s 5ms/step - loss: 0.0508 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9653
Epoch 358/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0468 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9639
Epoch 359/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9612
Epoch 360/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0440 - sparse_categorical_accuracy: 0.9899 - val_loss: 0.0988 - val_sparse_categorical_accuracy: 0.9626
Epoch 361/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0448 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0975 - val_sparse_categorical_accuracy: 0.9667
Epoch 362/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0914 - val_sparse_categorical_accuracy: 0.9639
Epoch 363/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0493 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0906 - val_sparse_categorical_accuracy: 0.9626
Epoch 364/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0488 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9626
Epoch 365/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0491 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9626
Epoch 366/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0891 - val_sparse_categorical_accuracy: 0.9612
Epoch 367/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0470 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.1026 - val_sparse_categorical_accuracy: 0.9626
Epoch 368/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0463 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9626
Epoch 369/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0459 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9639
Epoch 370/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.1036 - val_sparse_categorical_accuracy: 0.9626
Epoch 371/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0479 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9626
Epoch 372/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0516 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0932 - val_sparse_categorical_accuracy: 0.9653
Epoch 373/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0451 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0928 - val_sparse_categorical_accuracy: 0.9639
Epoch 374/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0461 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9612
Epoch 375/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0895 - val_sparse_categorical_accuracy: 0.9639
Epoch 376/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0466 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9639
Epoch 377/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0465 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0908 - val_sparse_categorical_accuracy: 0.9681
Epoch 378/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0430 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0906 - val_sparse_categorical_accuracy: 0.9626
Epoch 379/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0524 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0910 - val_sparse_categorical_accuracy: 0.9598
Epoch 380/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0467 - sparse_categorical_accuracy: 0.9872 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9639
Epoch 381/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0464 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9653
Epoch 382/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0449 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9639
Epoch 383/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0438 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.0905 - val_sparse_categorical_accuracy: 0.9612
Epoch 00383: early stopping

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
42/42 [==============================] - 0s 2ms/step - loss: 0.0936 - sparse_categorical_accuracy: 0.9682
Test accuracy 0.9681817889213562
Test loss 0.0935916006565094

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
