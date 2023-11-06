# Timeseries classification from scratch

**Author:** [hfawaz](https://github.com/hfawaz/)<br>
**Date created:** 2020/07/21<br>
**Last modified:** 2023/08/25<br>
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




    
![png](/img/examples/timeseries/timeseries_classification_from_scratch/timeseries_classification_from_scratch_17_0.png)
    



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
90/90 [==============================] - 10s 86ms/step - loss: 0.5351 - sparse_categorical_accuracy: 0.7323 - val_loss: 0.7737 - val_sparse_categorical_accuracy: 0.4813 - lr: 0.0010
Epoch 2/500

/home/codespace/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

90/90 [==============================] - 8s 86ms/step - loss: 0.4586 - sparse_categorical_accuracy: 0.7691 - val_loss: 0.8550 - val_sparse_categorical_accuracy: 0.4813 - lr: 0.0010
Epoch 3/500
90/90 [==============================] - 7s 81ms/step - loss: 0.4318 - sparse_categorical_accuracy: 0.7806 - val_loss: 0.8925 - val_sparse_categorical_accuracy: 0.4813 - lr: 0.0010
Epoch 4/500
90/90 [==============================] - 7s 75ms/step - loss: 0.4165 - sparse_categorical_accuracy: 0.7896 - val_loss: 0.7002 - val_sparse_categorical_accuracy: 0.4882 - lr: 0.0010
Epoch 5/500
90/90 [==============================] - 8s 84ms/step - loss: 0.4060 - sparse_categorical_accuracy: 0.7983 - val_loss: 0.5369 - val_sparse_categorical_accuracy: 0.7739 - lr: 0.0010
Epoch 6/500
90/90 [==============================] - 8s 89ms/step - loss: 0.4026 - sparse_categorical_accuracy: 0.7976 - val_loss: 0.4549 - val_sparse_categorical_accuracy: 0.7309 - lr: 0.0010
Epoch 7/500
90/90 [==============================] - 8s 88ms/step - loss: 0.3991 - sparse_categorical_accuracy: 0.8059 - val_loss: 0.3808 - val_sparse_categorical_accuracy: 0.8322 - lr: 0.0010
Epoch 8/500
90/90 [==============================] - 7s 80ms/step - loss: 0.3935 - sparse_categorical_accuracy: 0.8059 - val_loss: 0.3582 - val_sparse_categorical_accuracy: 0.8308 - lr: 0.0010
Epoch 9/500
90/90 [==============================] - 8s 92ms/step - loss: 0.3833 - sparse_categorical_accuracy: 0.8212 - val_loss: 0.3681 - val_sparse_categorical_accuracy: 0.8322 - lr: 0.0010
Epoch 10/500
90/90 [==============================] - 7s 75ms/step - loss: 0.3768 - sparse_categorical_accuracy: 0.8208 - val_loss: 0.3543 - val_sparse_categorical_accuracy: 0.8252 - lr: 0.0010
Epoch 11/500
90/90 [==============================] - 7s 79ms/step - loss: 0.3634 - sparse_categorical_accuracy: 0.8344 - val_loss: 0.3439 - val_sparse_categorical_accuracy: 0.8516 - lr: 0.0010
Epoch 12/500
90/90 [==============================] - 8s 91ms/step - loss: 0.3615 - sparse_categorical_accuracy: 0.8316 - val_loss: 0.4049 - val_sparse_categorical_accuracy: 0.7503 - lr: 0.0010
Epoch 13/500
90/90 [==============================] - 7s 73ms/step - loss: 0.3690 - sparse_categorical_accuracy: 0.8222 - val_loss: 0.4415 - val_sparse_categorical_accuracy: 0.7614 - lr: 0.0010
Epoch 14/500
90/90 [==============================] - 6s 71ms/step - loss: 0.3733 - sparse_categorical_accuracy: 0.8222 - val_loss: 0.3228 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 15/500
90/90 [==============================] - 8s 91ms/step - loss: 0.3446 - sparse_categorical_accuracy: 0.8503 - val_loss: 0.3102 - val_sparse_categorical_accuracy: 0.8655 - lr: 0.0010
Epoch 16/500
90/90 [==============================] - 8s 84ms/step - loss: 0.3446 - sparse_categorical_accuracy: 0.8434 - val_loss: 0.3696 - val_sparse_categorical_accuracy: 0.8280 - lr: 0.0010
Epoch 17/500
90/90 [==============================] - 7s 77ms/step - loss: 0.3365 - sparse_categorical_accuracy: 0.8455 - val_loss: 0.3563 - val_sparse_categorical_accuracy: 0.8225 - lr: 0.0010
Epoch 18/500
90/90 [==============================] - 6s 72ms/step - loss: 0.3359 - sparse_categorical_accuracy: 0.8483 - val_loss: 0.3560 - val_sparse_categorical_accuracy: 0.8169 - lr: 0.0010
Epoch 19/500
90/90 [==============================] - 7s 75ms/step - loss: 0.3160 - sparse_categorical_accuracy: 0.8667 - val_loss: 0.2975 - val_sparse_categorical_accuracy: 0.8710 - lr: 0.0010
Epoch 20/500
90/90 [==============================] - 8s 90ms/step - loss: 0.3163 - sparse_categorical_accuracy: 0.8552 - val_loss: 0.3040 - val_sparse_categorical_accuracy: 0.8655 - lr: 0.0010
Epoch 21/500
90/90 [==============================] - 7s 75ms/step - loss: 0.3103 - sparse_categorical_accuracy: 0.8698 - val_loss: 0.2945 - val_sparse_categorical_accuracy: 0.8530 - lr: 0.0010
Epoch 22/500
90/90 [==============================] - 7s 81ms/step - loss: 0.3346 - sparse_categorical_accuracy: 0.8465 - val_loss: 0.3628 - val_sparse_categorical_accuracy: 0.8350 - lr: 0.0010
Epoch 23/500
90/90 [==============================] - 7s 82ms/step - loss: 0.3061 - sparse_categorical_accuracy: 0.8681 - val_loss: 0.2894 - val_sparse_categorical_accuracy: 0.8890 - lr: 0.0010
Epoch 24/500
90/90 [==============================] - 8s 92ms/step - loss: 0.3087 - sparse_categorical_accuracy: 0.8660 - val_loss: 0.2962 - val_sparse_categorical_accuracy: 0.8655 - lr: 0.0010
Epoch 25/500
90/90 [==============================] - 7s 81ms/step - loss: 0.2983 - sparse_categorical_accuracy: 0.8715 - val_loss: 0.3761 - val_sparse_categorical_accuracy: 0.7961 - lr: 0.0010
Epoch 26/500
90/90 [==============================] - 7s 74ms/step - loss: 0.2966 - sparse_categorical_accuracy: 0.8767 - val_loss: 0.6136 - val_sparse_categorical_accuracy: 0.6630 - lr: 0.0010
Epoch 27/500
90/90 [==============================] - 7s 73ms/step - loss: 0.3010 - sparse_categorical_accuracy: 0.8698 - val_loss: 0.2758 - val_sparse_categorical_accuracy: 0.8669 - lr: 0.0010
Epoch 28/500
90/90 [==============================] - 7s 83ms/step - loss: 0.2979 - sparse_categorical_accuracy: 0.8726 - val_loss: 0.3870 - val_sparse_categorical_accuracy: 0.8044 - lr: 0.0010
Epoch 29/500
90/90 [==============================] - 7s 79ms/step - loss: 0.2905 - sparse_categorical_accuracy: 0.8767 - val_loss: 0.3122 - val_sparse_categorical_accuracy: 0.8419 - lr: 0.0010
Epoch 30/500
90/90 [==============================] - 7s 76ms/step - loss: 0.2794 - sparse_categorical_accuracy: 0.8823 - val_loss: 0.2786 - val_sparse_categorical_accuracy: 0.8710 - lr: 0.0010
Epoch 31/500
90/90 [==============================] - 7s 73ms/step - loss: 0.2933 - sparse_categorical_accuracy: 0.8785 - val_loss: 0.2552 - val_sparse_categorical_accuracy: 0.8946 - lr: 0.0010
Epoch 32/500
90/90 [==============================] - 8s 88ms/step - loss: 0.2728 - sparse_categorical_accuracy: 0.8938 - val_loss: 0.2419 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
Epoch 33/500
90/90 [==============================] - 7s 78ms/step - loss: 0.2596 - sparse_categorical_accuracy: 0.8979 - val_loss: 0.2294 - val_sparse_categorical_accuracy: 0.9154 - lr: 0.0010
Epoch 34/500
90/90 [==============================] - 8s 90ms/step - loss: 0.2517 - sparse_categorical_accuracy: 0.8951 - val_loss: 0.4143 - val_sparse_categorical_accuracy: 0.7878 - lr: 0.0010
Epoch 35/500
90/90 [==============================] - 7s 74ms/step - loss: 0.2755 - sparse_categorical_accuracy: 0.8875 - val_loss: 0.2499 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
Epoch 36/500
90/90 [==============================] - 7s 73ms/step - loss: 0.2522 - sparse_categorical_accuracy: 0.8986 - val_loss: 0.4125 - val_sparse_categorical_accuracy: 0.7725 - lr: 0.0010
Epoch 37/500
90/90 [==============================] - 6s 71ms/step - loss: 0.2624 - sparse_categorical_accuracy: 0.8944 - val_loss: 0.3298 - val_sparse_categorical_accuracy: 0.8350 - lr: 0.0010
Epoch 38/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2491 - sparse_categorical_accuracy: 0.9045 - val_loss: 0.3497 - val_sparse_categorical_accuracy: 0.8114 - lr: 0.0010
Epoch 39/500
90/90 [==============================] - 7s 72ms/step - loss: 0.2611 - sparse_categorical_accuracy: 0.8931 - val_loss: 0.4030 - val_sparse_categorical_accuracy: 0.8044 - lr: 0.0010
Epoch 40/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2476 - sparse_categorical_accuracy: 0.8993 - val_loss: 0.2347 - val_sparse_categorical_accuracy: 0.8974 - lr: 0.0010
Epoch 41/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2406 - sparse_categorical_accuracy: 0.9017 - val_loss: 0.4557 - val_sparse_categorical_accuracy: 0.7601 - lr: 0.0010
Epoch 42/500
90/90 [==============================] - 6s 71ms/step - loss: 0.2609 - sparse_categorical_accuracy: 0.8892 - val_loss: 0.2413 - val_sparse_categorical_accuracy: 0.9043 - lr: 0.0010
Epoch 43/500
90/90 [==============================] - 7s 73ms/step - loss: 0.2474 - sparse_categorical_accuracy: 0.9017 - val_loss: 0.3103 - val_sparse_categorical_accuracy: 0.8585 - lr: 0.0010
Epoch 44/500
90/90 [==============================] - 6s 70ms/step - loss: 0.2355 - sparse_categorical_accuracy: 0.9035 - val_loss: 0.2055 - val_sparse_categorical_accuracy: 0.9168 - lr: 0.0010
Epoch 45/500
90/90 [==============================] - 7s 79ms/step - loss: 0.2504 - sparse_categorical_accuracy: 0.8958 - val_loss: 0.2314 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
Epoch 46/500
90/90 [==============================] - 7s 79ms/step - loss: 0.2370 - sparse_categorical_accuracy: 0.9094 - val_loss: 0.2817 - val_sparse_categorical_accuracy: 0.8585 - lr: 0.0010
Epoch 47/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2315 - sparse_categorical_accuracy: 0.9111 - val_loss: 0.3770 - val_sparse_categorical_accuracy: 0.8100 - lr: 0.0010
Epoch 48/500
90/90 [==============================] - 7s 73ms/step - loss: 0.2351 - sparse_categorical_accuracy: 0.9069 - val_loss: 0.2037 - val_sparse_categorical_accuracy: 0.9223 - lr: 0.0010
Epoch 49/500
90/90 [==============================] - 8s 88ms/step - loss: 0.2249 - sparse_categorical_accuracy: 0.9101 - val_loss: 0.2153 - val_sparse_categorical_accuracy: 0.9154 - lr: 0.0010
Epoch 50/500
90/90 [==============================] - 6s 71ms/step - loss: 0.2143 - sparse_categorical_accuracy: 0.9149 - val_loss: 0.1979 - val_sparse_categorical_accuracy: 0.9251 - lr: 0.0010
Epoch 51/500
90/90 [==============================] - 8s 89ms/step - loss: 0.2305 - sparse_categorical_accuracy: 0.9066 - val_loss: 0.2462 - val_sparse_categorical_accuracy: 0.8849 - lr: 0.0010
Epoch 52/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2257 - sparse_categorical_accuracy: 0.9083 - val_loss: 0.2705 - val_sparse_categorical_accuracy: 0.8821 - lr: 0.0010
Epoch 53/500
90/90 [==============================] - 6s 71ms/step - loss: 0.2254 - sparse_categorical_accuracy: 0.9083 - val_loss: 0.4320 - val_sparse_categorical_accuracy: 0.7725 - lr: 0.0010
Epoch 54/500
90/90 [==============================] - 7s 75ms/step - loss: 0.2316 - sparse_categorical_accuracy: 0.9049 - val_loss: 0.2079 - val_sparse_categorical_accuracy: 0.9182 - lr: 0.0010
Epoch 55/500
90/90 [==============================] - 6s 72ms/step - loss: 0.2251 - sparse_categorical_accuracy: 0.9108 - val_loss: 0.2433 - val_sparse_categorical_accuracy: 0.9182 - lr: 0.0010
Epoch 56/500
90/90 [==============================] - 7s 74ms/step - loss: 0.1988 - sparse_categorical_accuracy: 0.9253 - val_loss: 0.2070 - val_sparse_categorical_accuracy: 0.9209 - lr: 0.0010
Epoch 57/500
90/90 [==============================] - 7s 73ms/step - loss: 0.2023 - sparse_categorical_accuracy: 0.9205 - val_loss: 0.1827 - val_sparse_categorical_accuracy: 0.9334 - lr: 0.0010
Epoch 58/500
90/90 [==============================] - 7s 81ms/step - loss: 0.1896 - sparse_categorical_accuracy: 0.9337 - val_loss: 0.1673 - val_sparse_categorical_accuracy: 0.9362 - lr: 0.0010
Epoch 59/500
90/90 [==============================] - 8s 89ms/step - loss: 0.1763 - sparse_categorical_accuracy: 0.9351 - val_loss: 0.1618 - val_sparse_categorical_accuracy: 0.9404 - lr: 0.0010
Epoch 60/500
90/90 [==============================] - 8s 86ms/step - loss: 0.1660 - sparse_categorical_accuracy: 0.9451 - val_loss: 0.2406 - val_sparse_categorical_accuracy: 0.8849 - lr: 0.0010
Epoch 61/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1625 - sparse_categorical_accuracy: 0.9434 - val_loss: 0.1559 - val_sparse_categorical_accuracy: 0.9390 - lr: 0.0010
Epoch 62/500
90/90 [==============================] - 8s 92ms/step - loss: 0.1630 - sparse_categorical_accuracy: 0.9465 - val_loss: 0.1931 - val_sparse_categorical_accuracy: 0.9209 - lr: 0.0010
Epoch 63/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1457 - sparse_categorical_accuracy: 0.9514 - val_loss: 0.2167 - val_sparse_categorical_accuracy: 0.9057 - lr: 0.0010
Epoch 64/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1680 - sparse_categorical_accuracy: 0.9410 - val_loss: 0.1946 - val_sparse_categorical_accuracy: 0.9362 - lr: 0.0010
Epoch 65/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1368 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.1415 - val_sparse_categorical_accuracy: 0.9542 - lr: 0.0010
Epoch 66/500
90/90 [==============================] - 8s 84ms/step - loss: 0.1479 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.2398 - val_sparse_categorical_accuracy: 0.8918 - lr: 0.0010
Epoch 67/500
90/90 [==============================] - 7s 76ms/step - loss: 0.1355 - sparse_categorical_accuracy: 0.9531 - val_loss: 0.1603 - val_sparse_categorical_accuracy: 0.9473 - lr: 0.0010
Epoch 68/500
90/90 [==============================] - 7s 77ms/step - loss: 0.1274 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.1779 - val_sparse_categorical_accuracy: 0.9307 - lr: 0.0010
Epoch 69/500
90/90 [==============================] - 7s 74ms/step - loss: 0.1252 - sparse_categorical_accuracy: 0.9556 - val_loss: 0.3398 - val_sparse_categorical_accuracy: 0.8447 - lr: 0.0010
Epoch 70/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1212 - sparse_categorical_accuracy: 0.9583 - val_loss: 0.4555 - val_sparse_categorical_accuracy: 0.8044 - lr: 0.0010
Epoch 71/500
90/90 [==============================] - 6s 71ms/step - loss: 0.1232 - sparse_categorical_accuracy: 0.9601 - val_loss: 2.1422 - val_sparse_categorical_accuracy: 0.5257 - lr: 0.0010
Epoch 72/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1230 - sparse_categorical_accuracy: 0.9594 - val_loss: 0.1999 - val_sparse_categorical_accuracy: 0.9237 - lr: 0.0010
Epoch 73/500
90/90 [==============================] - 7s 75ms/step - loss: 0.1144 - sparse_categorical_accuracy: 0.9608 - val_loss: 1.4856 - val_sparse_categorical_accuracy: 0.6491 - lr: 0.0010
Epoch 74/500
90/90 [==============================] - 6s 71ms/step - loss: 0.1157 - sparse_categorical_accuracy: 0.9611 - val_loss: 0.8342 - val_sparse_categorical_accuracy: 0.6782 - lr: 0.0010
Epoch 75/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1264 - sparse_categorical_accuracy: 0.9569 - val_loss: 0.1556 - val_sparse_categorical_accuracy: 0.9515 - lr: 0.0010
Epoch 76/500
90/90 [==============================] - 7s 74ms/step - loss: 0.1185 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.3898 - val_sparse_categorical_accuracy: 0.8225 - lr: 0.0010
Epoch 77/500
90/90 [==============================] - 6s 70ms/step - loss: 0.1183 - sparse_categorical_accuracy: 0.9590 - val_loss: 0.1241 - val_sparse_categorical_accuracy: 0.9528 - lr: 0.0010
Epoch 78/500
90/90 [==============================] - 7s 80ms/step - loss: 0.1113 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1586 - val_sparse_categorical_accuracy: 0.9404 - lr: 0.0010
Epoch 79/500
90/90 [==============================] - 7s 79ms/step - loss: 0.1036 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.2251 - val_sparse_categorical_accuracy: 0.9168 - lr: 0.0010
Epoch 80/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1075 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.2458 - val_sparse_categorical_accuracy: 0.9168 - lr: 0.0010
Epoch 81/500
90/90 [==============================] - 7s 74ms/step - loss: 0.1114 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1403 - val_sparse_categorical_accuracy: 0.9473 - lr: 0.0010
Epoch 82/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1113 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.1248 - val_sparse_categorical_accuracy: 0.9598 - lr: 0.0010
Epoch 83/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1048 - sparse_categorical_accuracy: 0.9653 - val_loss: 1.1956 - val_sparse_categorical_accuracy: 0.7074 - lr: 0.0010
Epoch 84/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1091 - sparse_categorical_accuracy: 0.9625 - val_loss: 1.4386 - val_sparse_categorical_accuracy: 0.7268 - lr: 0.0010
Epoch 85/500
90/90 [==============================] - 6s 71ms/step - loss: 0.1025 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.2521 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
Epoch 86/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1078 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.3390 - val_sparse_categorical_accuracy: 0.8530 - lr: 0.0010
Epoch 87/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0980 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.2838 - val_sparse_categorical_accuracy: 0.8669 - lr: 0.0010
Epoch 88/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0969 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.6388 - val_sparse_categorical_accuracy: 0.7573 - lr: 0.0010
Epoch 89/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1038 - sparse_categorical_accuracy: 0.9642 - val_loss: 2.7099 - val_sparse_categorical_accuracy: 0.5631 - lr: 0.0010
Epoch 90/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0968 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.3669 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 91/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0988 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1757 - val_sparse_categorical_accuracy: 0.9445 - lr: 0.0010
Epoch 92/500
90/90 [==============================] - 7s 72ms/step - loss: 0.0999 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.8467 - val_sparse_categorical_accuracy: 0.7060 - lr: 0.0010
Epoch 93/500
90/90 [==============================] - 6s 70ms/step - loss: 0.1091 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.1692 - val_sparse_categorical_accuracy: 0.9334 - lr: 0.0010
Epoch 94/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0941 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.1392 - val_sparse_categorical_accuracy: 0.9487 - lr: 0.0010
Epoch 95/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1004 - sparse_categorical_accuracy: 0.9660 - val_loss: 1.0792 - val_sparse_categorical_accuracy: 0.6158 - lr: 0.0010
Epoch 96/500
90/90 [==============================] - 7s 73ms/step - loss: 0.1153 - sparse_categorical_accuracy: 0.9594 - val_loss: 0.2934 - val_sparse_categorical_accuracy: 0.9043 - lr: 0.0010
Epoch 97/500
90/90 [==============================] - 6s 72ms/step - loss: 0.1004 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.2830 - val_sparse_categorical_accuracy: 0.8766 - lr: 0.0010
Epoch 98/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1831 - val_sparse_categorical_accuracy: 0.9459 - lr: 5.0000e-04
Epoch 99/500
90/90 [==============================] - 7s 76ms/step - loss: 0.0909 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.1598 - val_sparse_categorical_accuracy: 0.9376 - lr: 5.0000e-04
Epoch 100/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0873 - sparse_categorical_accuracy: 0.9681 - val_loss: 0.2001 - val_sparse_categorical_accuracy: 0.9320 - lr: 5.0000e-04
Epoch 101/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0936 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.7560 - val_sparse_categorical_accuracy: 0.7573 - lr: 5.0000e-04
Epoch 102/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0850 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.2384 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
Epoch 103/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.3161 - val_sparse_categorical_accuracy: 0.8779 - lr: 5.0000e-04
Epoch 104/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.3570 - val_sparse_categorical_accuracy: 0.8599 - lr: 5.0000e-04
Epoch 105/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0857 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.2856 - val_sparse_categorical_accuracy: 0.8752 - lr: 5.0000e-04
Epoch 106/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0837 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1330 - val_sparse_categorical_accuracy: 0.9556 - lr: 5.0000e-04
Epoch 107/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0858 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1063 - val_sparse_categorical_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 108/500
90/90 [==============================] - 8s 89ms/step - loss: 0.0904 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9584 - lr: 5.0000e-04
Epoch 109/500
90/90 [==============================] - 7s 72ms/step - loss: 0.0825 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1209 - val_sparse_categorical_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 110/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1364 - val_sparse_categorical_accuracy: 0.9487 - lr: 5.0000e-04
Epoch 111/500
90/90 [==============================] - 7s 76ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9639 - lr: 5.0000e-04
Epoch 112/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0885 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1916 - val_sparse_categorical_accuracy: 0.9182 - lr: 5.0000e-04
Epoch 113/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0883 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1552 - val_sparse_categorical_accuracy: 0.9251 - lr: 5.0000e-04
Epoch 114/500
90/90 [==============================] - 7s 78ms/step - loss: 0.0808 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1496 - val_sparse_categorical_accuracy: 0.9417 - lr: 5.0000e-04
Epoch 115/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0790 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.2551 - val_sparse_categorical_accuracy: 0.8974 - lr: 5.0000e-04
Epoch 116/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1704 - val_sparse_categorical_accuracy: 0.9445 - lr: 5.0000e-04
Epoch 117/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0791 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1119 - val_sparse_categorical_accuracy: 0.9584 - lr: 5.0000e-04
Epoch 118/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0814 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.3336 - val_sparse_categorical_accuracy: 0.8627 - lr: 5.0000e-04
Epoch 119/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0804 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1168 - val_sparse_categorical_accuracy: 0.9528 - lr: 5.0000e-04
Epoch 120/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0801 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1662 - val_sparse_categorical_accuracy: 0.9265 - lr: 5.0000e-04
Epoch 121/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0918 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.1312 - val_sparse_categorical_accuracy: 0.9445 - lr: 5.0000e-04
Epoch 122/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0871 - sparse_categorical_accuracy: 0.9694 - val_loss: 0.1142 - val_sparse_categorical_accuracy: 0.9653 - lr: 5.0000e-04
Epoch 123/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0839 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.0979 - val_sparse_categorical_accuracy: 0.9709 - lr: 5.0000e-04
Epoch 124/500
90/90 [==============================] - 7s 79ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9667 - lr: 5.0000e-04
Epoch 125/500
90/90 [==============================] - 8s 89ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1676 - val_sparse_categorical_accuracy: 0.9348 - lr: 5.0000e-04
Epoch 126/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0781 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1174 - val_sparse_categorical_accuracy: 0.9556 - lr: 5.0000e-04
Epoch 127/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1159 - val_sparse_categorical_accuracy: 0.9528 - lr: 5.0000e-04
Epoch 128/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0733 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1167 - val_sparse_categorical_accuracy: 0.9556 - lr: 5.0000e-04
Epoch 129/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0998 - val_sparse_categorical_accuracy: 0.9626 - lr: 5.0000e-04
Epoch 130/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0759 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1382 - val_sparse_categorical_accuracy: 0.9570 - lr: 5.0000e-04
Epoch 131/500
90/90 [==============================] - 7s 72ms/step - loss: 0.0735 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1849 - val_sparse_categorical_accuracy: 0.9390 - lr: 5.0000e-04
Epoch 132/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0775 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1084 - val_sparse_categorical_accuracy: 0.9653 - lr: 5.0000e-04
Epoch 133/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.2815 - val_sparse_categorical_accuracy: 0.8932 - lr: 5.0000e-04
Epoch 134/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1231 - val_sparse_categorical_accuracy: 0.9556 - lr: 5.0000e-04
Epoch 135/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0746 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1884 - val_sparse_categorical_accuracy: 0.9223 - lr: 5.0000e-04
Epoch 136/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0801 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1375 - val_sparse_categorical_accuracy: 0.9584 - lr: 5.0000e-04
Epoch 137/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0726 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.2792 - val_sparse_categorical_accuracy: 0.8807 - lr: 5.0000e-04
Epoch 138/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0745 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1292 - val_sparse_categorical_accuracy: 0.9487 - lr: 5.0000e-04
Epoch 139/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0786 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.2664 - val_sparse_categorical_accuracy: 0.8918 - lr: 5.0000e-04
Epoch 140/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0762 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1365 - val_sparse_categorical_accuracy: 0.9570 - lr: 5.0000e-04
Epoch 141/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1011 - val_sparse_categorical_accuracy: 0.9556 - lr: 5.0000e-04
Epoch 142/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0838 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.2273 - val_sparse_categorical_accuracy: 0.9057 - lr: 5.0000e-04
Epoch 143/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0769 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1539 - val_sparse_categorical_accuracy: 0.9487 - lr: 5.0000e-04
Epoch 144/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1625 - val_sparse_categorical_accuracy: 0.9501 - lr: 5.0000e-04
Epoch 145/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0678 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1192 - val_sparse_categorical_accuracy: 0.9598 - lr: 2.5000e-04
Epoch 146/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1012 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 147/500
90/90 [==============================] - 7s 76ms/step - loss: 0.0697 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1005 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 148/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0636 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1077 - val_sparse_categorical_accuracy: 0.9612 - lr: 2.5000e-04
Epoch 149/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0667 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1087 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 150/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 151/500
90/90 [==============================] - 8s 86ms/step - loss: 0.0668 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0948 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 152/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0665 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 153/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 154/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0685 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1243 - val_sparse_categorical_accuracy: 0.9556 - lr: 2.5000e-04
Epoch 155/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1146 - val_sparse_categorical_accuracy: 0.9528 - lr: 2.5000e-04
Epoch 156/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1260 - val_sparse_categorical_accuracy: 0.9501 - lr: 2.5000e-04
Epoch 157/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1830 - val_sparse_categorical_accuracy: 0.9293 - lr: 2.5000e-04
Epoch 158/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0624 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1481 - val_sparse_categorical_accuracy: 0.9376 - lr: 2.5000e-04
Epoch 159/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0676 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1137 - val_sparse_categorical_accuracy: 0.9598 - lr: 2.5000e-04
Epoch 160/500
90/90 [==============================] - 7s 81ms/step - loss: 0.0631 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 161/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0718 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 162/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0659 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0990 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 163/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0632 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 164/500
90/90 [==============================] - 7s 76ms/step - loss: 0.0649 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1189 - val_sparse_categorical_accuracy: 0.9473 - lr: 2.5000e-04
Epoch 165/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0663 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 166/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0640 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0895 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 167/500
90/90 [==============================] - 7s 79ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.2038 - val_sparse_categorical_accuracy: 0.9196 - lr: 2.5000e-04
Epoch 168/500
90/90 [==============================] - 7s 78ms/step - loss: 0.0682 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0938 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 169/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0606 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 170/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0666 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1187 - val_sparse_categorical_accuracy: 0.9542 - lr: 2.5000e-04
Epoch 171/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1012 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 172/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0615 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 173/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0649 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1448 - val_sparse_categorical_accuracy: 0.9348 - lr: 2.5000e-04
Epoch 174/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0719 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0981 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 175/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.1438 - val_sparse_categorical_accuracy: 0.9501 - lr: 2.5000e-04
Epoch 176/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0636 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1331 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 177/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0663 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1179 - val_sparse_categorical_accuracy: 0.9598 - lr: 2.5000e-04
Epoch 178/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0688 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1012 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 179/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0646 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1474 - val_sparse_categorical_accuracy: 0.9431 - lr: 2.5000e-04
Epoch 180/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0637 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 181/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0641 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1392 - val_sparse_categorical_accuracy: 0.9459 - lr: 2.5000e-04
Epoch 182/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0611 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1113 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 183/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0614 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0883 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 184/500
90/90 [==============================] - 7s 79ms/step - loss: 0.0628 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0994 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 185/500
90/90 [==============================] - 7s 78ms/step - loss: 0.0597 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1042 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 186/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0972 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 187/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1181 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 188/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0612 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1416 - val_sparse_categorical_accuracy: 0.9570 - lr: 2.5000e-04
Epoch 189/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0612 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 190/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0933 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 191/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0617 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0888 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 192/500
90/90 [==============================] - 7s 76ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0944 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 193/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0609 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 194/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0614 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1032 - val_sparse_categorical_accuracy: 0.9667 - lr: 2.5000e-04
Epoch 195/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0581 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 196/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1184 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 197/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1840 - val_sparse_categorical_accuracy: 0.9417 - lr: 2.5000e-04
Epoch 198/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0654 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0961 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 199/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1074 - val_sparse_categorical_accuracy: 0.9570 - lr: 2.5000e-04
Epoch 200/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.1238 - val_sparse_categorical_accuracy: 0.9584 - lr: 2.5000e-04
Epoch 201/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0648 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 202/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0603 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 203/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0641 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0919 - val_sparse_categorical_accuracy: 0.9695 - lr: 2.5000e-04
Epoch 204/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0544 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 205/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0575 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1041 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.2500e-04
Epoch 206/500
90/90 [==============================] - 7s 78ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 207/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 208/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0516 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 209/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.1069 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.2500e-04
Epoch 210/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1004 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 211/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0553 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0896 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 212/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1202 - val_sparse_categorical_accuracy: 0.9612 - lr: 1.2500e-04
Epoch 213/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0597 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0882 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 214/500
90/90 [==============================] - 8s 86ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 215/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0511 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0904 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 216/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1147 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.2500e-04
Epoch 217/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.1021 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 218/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0564 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0951 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 219/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0991 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 220/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1117 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.2500e-04
Epoch 221/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0551 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1048 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 222/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0994 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 223/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0587 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 224/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0912 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 225/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0570 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0916 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 226/500
90/90 [==============================] - 7s 72ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0920 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 227/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0561 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0940 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 228/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0907 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.2500e-04
Epoch 229/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0934 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 230/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0551 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0951 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 231/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0572 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 232/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0519 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.1066 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.2500e-04
Epoch 233/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.1630 - val_sparse_categorical_accuracy: 0.9445 - lr: 1.2500e-04
Epoch 234/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0897 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 235/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0586 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.1010 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 236/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0475 - sparse_categorical_accuracy: 0.9868 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 237/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0545 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0930 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 238/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0549 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 239/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0531 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.1169 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 240/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 241/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.0990 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 242/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 243/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0889 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 244/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0560 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.1069 - val_sparse_categorical_accuracy: 0.9570 - lr: 1.0000e-04
Epoch 245/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9598 - lr: 1.0000e-04
Epoch 246/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0519 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0986 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 247/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 248/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0529 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 249/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0490 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 250/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0889 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 251/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0558 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.1064 - val_sparse_categorical_accuracy: 0.9639 - lr: 1.0000e-04
Epoch 252/500
90/90 [==============================] - 7s 79ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 253/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0924 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 254/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0526 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0939 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 255/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0569 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 256/500
90/90 [==============================] - 7s 73ms/step - loss: 0.0494 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0915 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 257/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0488 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0899 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 258/500
90/90 [==============================] - 6s 72ms/step - loss: 0.0521 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9667 - lr: 1.0000e-04
Epoch 259/500
90/90 [==============================] - 7s 75ms/step - loss: 0.0578 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 260/500
90/90 [==============================] - 6s 70ms/step - loss: 0.0481 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0913 - val_sparse_categorical_accuracy: 0.9681 - lr: 1.0000e-04
Epoch 261/500
90/90 [==============================] - 6s 71ms/step - loss: 0.0531 - sparse_categorical_accuracy: 0.9861 - val_loss: 0.1129 - val_sparse_categorical_accuracy: 0.9653 - lr: 1.0000e-04
Epoch 262/500
90/90 [==============================] - 7s 74ms/step - loss: 0.0510 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0903 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.0000e-04
Epoch 263/500
90/90 [==============================] - 7s 77ms/step - loss: 0.0513 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0968 - val_sparse_categorical_accuracy: 0.9626 - lr: 1.0000e-04
Epoch 263: early stopping

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
42/42 [==============================] - 1s 14ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9644
Test accuracy 0.9643939137458801
Test loss 0.09282448887825012

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
| [![🤗 Model - Timeseries classification from scratch](https://img.shields.io/badge/🤗_Model-Timeseries_classification_from_scratch-black)](https://huggingface.co/keras-io/timeseries-classification-from-scratch) | [![🤗  Spaces - Timeseries classification from scratch](https://img.shields.io/badge/🤗_Spaces-Timeseries_classification_from_scratch-black)](https://huggingface.co/spaces/keras-io/timeseries-classification-from-scratch) |
