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
90/90 [==============================] - 8s 11ms/step - loss: 0.5377 - sparse_categorical_accuracy: 0.7042 - val_loss: 0.7685 - val_sparse_categorical_accuracy: 0.4619 - lr: 0.0010
Epoch 2/500
21/90 [======>.......................] - ETA: 0s - loss: 0.4934 - sparse_categorical_accuracy: 0.7574

/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

90/90 [==============================] - 1s 6ms/step - loss: 0.4494 - sparse_categorical_accuracy: 0.7760 - val_loss: 0.8208 - val_sparse_categorical_accuracy: 0.4619 - lr: 0.0010
Epoch 3/500
90/90 [==============================] - 1s 7ms/step - loss: 0.4248 - sparse_categorical_accuracy: 0.7823 - val_loss: 0.7471 - val_sparse_categorical_accuracy: 0.4771 - lr: 0.0010
Epoch 4/500
90/90 [==============================] - 1s 7ms/step - loss: 0.4272 - sparse_categorical_accuracy: 0.7833 - val_loss: 0.7369 - val_sparse_categorical_accuracy: 0.4840 - lr: 0.0010
Epoch 5/500
90/90 [==============================] - 1s 8ms/step - loss: 0.4075 - sparse_categorical_accuracy: 0.7937 - val_loss: 0.5881 - val_sparse_categorical_accuracy: 0.7087 - lr: 0.0010
Epoch 6/500
90/90 [==============================] - 1s 7ms/step - loss: 0.4091 - sparse_categorical_accuracy: 0.7951 - val_loss: 0.4135 - val_sparse_categorical_accuracy: 0.8377 - lr: 0.0010
Epoch 7/500
90/90 [==============================] - 1s 7ms/step - loss: 0.3891 - sparse_categorical_accuracy: 0.8069 - val_loss: 0.4047 - val_sparse_categorical_accuracy: 0.8141 - lr: 0.0010
Epoch 8/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3891 - sparse_categorical_accuracy: 0.8062 - val_loss: 0.4396 - val_sparse_categorical_accuracy: 0.7725 - lr: 0.0010
Epoch 9/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3924 - sparse_categorical_accuracy: 0.8094 - val_loss: 0.5073 - val_sparse_categorical_accuracy: 0.7517 - lr: 0.0010
Epoch 10/500
90/90 [==============================] - 1s 7ms/step - loss: 0.3763 - sparse_categorical_accuracy: 0.8260 - val_loss: 0.3786 - val_sparse_categorical_accuracy: 0.7836 - lr: 0.0010
Epoch 11/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3714 - sparse_categorical_accuracy: 0.8229 - val_loss: 0.3965 - val_sparse_categorical_accuracy: 0.7573 - lr: 0.0010
Epoch 12/500
90/90 [==============================] - 1s 6ms/step - loss: 0.3592 - sparse_categorical_accuracy: 0.8250 - val_loss: 0.6440 - val_sparse_categorical_accuracy: 0.6824 - lr: 0.0010
Epoch 13/500
90/90 [==============================] - 1s 7ms/step - loss: 0.3556 - sparse_categorical_accuracy: 0.8323 - val_loss: 0.4310 - val_sparse_categorical_accuracy: 0.7462 - lr: 0.0010
Epoch 14/500
90/90 [==============================] - 1s 7ms/step - loss: 0.3652 - sparse_categorical_accuracy: 0.8264 - val_loss: 0.3687 - val_sparse_categorical_accuracy: 0.8169 - lr: 0.0010
Epoch 15/500
90/90 [==============================] - 1s 7ms/step - loss: 0.3460 - sparse_categorical_accuracy: 0.8399 - val_loss: 0.3250 - val_sparse_categorical_accuracy: 0.8502 - lr: 0.0010
Epoch 16/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3326 - sparse_categorical_accuracy: 0.8427 - val_loss: 0.3230 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 17/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3279 - sparse_categorical_accuracy: 0.8542 - val_loss: 0.3883 - val_sparse_categorical_accuracy: 0.7892 - lr: 0.0010
Epoch 18/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3277 - sparse_categorical_accuracy: 0.8535 - val_loss: 0.4432 - val_sparse_categorical_accuracy: 0.7795 - lr: 0.0010
Epoch 19/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3165 - sparse_categorical_accuracy: 0.8566 - val_loss: 0.3186 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 20/500
90/90 [==============================] - 1s 9ms/step - loss: 0.3122 - sparse_categorical_accuracy: 0.8642 - val_loss: 0.4596 - val_sparse_categorical_accuracy: 0.7587 - lr: 0.0010
Epoch 21/500
90/90 [==============================] - 1s 9ms/step - loss: 0.3222 - sparse_categorical_accuracy: 0.8514 - val_loss: 0.3016 - val_sparse_categorical_accuracy: 0.8793 - lr: 0.0010
Epoch 22/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3144 - sparse_categorical_accuracy: 0.8587 - val_loss: 0.3164 - val_sparse_categorical_accuracy: 0.8641 - lr: 0.0010
Epoch 23/500
90/90 [==============================] - 1s 8ms/step - loss: 0.3044 - sparse_categorical_accuracy: 0.8687 - val_loss: 0.5242 - val_sparse_categorical_accuracy: 0.7268 - lr: 0.0010
Epoch 24/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2988 - sparse_categorical_accuracy: 0.8615 - val_loss: 0.3776 - val_sparse_categorical_accuracy: 0.8252 - lr: 0.0010
Epoch 25/500
90/90 [==============================] - 1s 8ms/step - loss: 0.2888 - sparse_categorical_accuracy: 0.8705 - val_loss: 0.3069 - val_sparse_categorical_accuracy: 0.8655 - lr: 0.0010
Epoch 26/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2817 - sparse_categorical_accuracy: 0.8833 - val_loss: 0.2988 - val_sparse_categorical_accuracy: 0.8682 - lr: 0.0010
Epoch 27/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2923 - sparse_categorical_accuracy: 0.8705 - val_loss: 0.5434 - val_sparse_categorical_accuracy: 0.7004 - lr: 0.0010
Epoch 28/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2773 - sparse_categorical_accuracy: 0.8830 - val_loss: 0.2982 - val_sparse_categorical_accuracy: 0.8544 - lr: 0.0010
Epoch 29/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2832 - sparse_categorical_accuracy: 0.8767 - val_loss: 0.4046 - val_sparse_categorical_accuracy: 0.7850 - lr: 0.0010
Epoch 30/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2748 - sparse_categorical_accuracy: 0.8844 - val_loss: 0.7592 - val_sparse_categorical_accuracy: 0.7254 - lr: 0.0010
Epoch 31/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2790 - sparse_categorical_accuracy: 0.8788 - val_loss: 0.6965 - val_sparse_categorical_accuracy: 0.6990 - lr: 0.0010
Epoch 32/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2695 - sparse_categorical_accuracy: 0.8840 - val_loss: 0.3256 - val_sparse_categorical_accuracy: 0.8460 - lr: 0.0010
Epoch 33/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2585 - sparse_categorical_accuracy: 0.8910 - val_loss: 0.2512 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
Epoch 34/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2655 - sparse_categorical_accuracy: 0.8878 - val_loss: 0.3264 - val_sparse_categorical_accuracy: 0.8405 - lr: 0.0010
Epoch 35/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2577 - sparse_categorical_accuracy: 0.8965 - val_loss: 0.3065 - val_sparse_categorical_accuracy: 0.8502 - lr: 0.0010
Epoch 36/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2650 - sparse_categorical_accuracy: 0.8878 - val_loss: 0.2594 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
Epoch 37/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2636 - sparse_categorical_accuracy: 0.8927 - val_loss: 0.3101 - val_sparse_categorical_accuracy: 0.8710 - lr: 0.0010
Epoch 38/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2558 - sparse_categorical_accuracy: 0.8948 - val_loss: 0.3148 - val_sparse_categorical_accuracy: 0.8474 - lr: 0.0010
Epoch 39/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2622 - sparse_categorical_accuracy: 0.8899 - val_loss: 0.3131 - val_sparse_categorical_accuracy: 0.8433 - lr: 0.0010
Epoch 40/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2442 - sparse_categorical_accuracy: 0.9003 - val_loss: 0.4162 - val_sparse_categorical_accuracy: 0.7878 - lr: 0.0010
Epoch 41/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2513 - sparse_categorical_accuracy: 0.8958 - val_loss: 0.3925 - val_sparse_categorical_accuracy: 0.8155 - lr: 0.0010
Epoch 42/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2531 - sparse_categorical_accuracy: 0.8906 - val_loss: 0.2434 - val_sparse_categorical_accuracy: 0.9001 - lr: 0.0010
Epoch 43/500
90/90 [==============================] - 1s 8ms/step - loss: 0.2459 - sparse_categorical_accuracy: 0.8969 - val_loss: 0.9065 - val_sparse_categorical_accuracy: 0.6976 - lr: 0.0010
Epoch 44/500
90/90 [==============================] - 1s 8ms/step - loss: 0.2416 - sparse_categorical_accuracy: 0.8979 - val_loss: 0.2872 - val_sparse_categorical_accuracy: 0.8696 - lr: 0.0010
Epoch 45/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2368 - sparse_categorical_accuracy: 0.9024 - val_loss: 0.4596 - val_sparse_categorical_accuracy: 0.7670 - lr: 0.0010
Epoch 46/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2348 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.3439 - val_sparse_categorical_accuracy: 0.8585 - lr: 0.0010
Epoch 47/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2340 - sparse_categorical_accuracy: 0.9056 - val_loss: 0.3808 - val_sparse_categorical_accuracy: 0.8211 - lr: 0.0010
Epoch 48/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2329 - sparse_categorical_accuracy: 0.9021 - val_loss: 0.3502 - val_sparse_categorical_accuracy: 0.8211 - lr: 0.0010
Epoch 49/500
90/90 [==============================] - 1s 10ms/step - loss: 0.2480 - sparse_categorical_accuracy: 0.8934 - val_loss: 0.2291 - val_sparse_categorical_accuracy: 0.9085 - lr: 0.0010
Epoch 50/500
90/90 [==============================] - 1s 8ms/step - loss: 0.2489 - sparse_categorical_accuracy: 0.9007 - val_loss: 0.2590 - val_sparse_categorical_accuracy: 0.8849 - lr: 0.0010
Epoch 51/500
90/90 [==============================] - 1s 9ms/step - loss: 0.2378 - sparse_categorical_accuracy: 0.8979 - val_loss: 0.2540 - val_sparse_categorical_accuracy: 0.8974 - lr: 0.0010
Epoch 52/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2244 - sparse_categorical_accuracy: 0.9125 - val_loss: 0.2356 - val_sparse_categorical_accuracy: 0.9237 - lr: 0.0010
Epoch 53/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2145 - sparse_categorical_accuracy: 0.9097 - val_loss: 0.4238 - val_sparse_categorical_accuracy: 0.7878 - lr: 0.0010
Epoch 54/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2148 - sparse_categorical_accuracy: 0.9142 - val_loss: 0.2462 - val_sparse_categorical_accuracy: 0.8946 - lr: 0.0010
Epoch 55/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2154 - sparse_categorical_accuracy: 0.9132 - val_loss: 0.2465 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
Epoch 56/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2177 - sparse_categorical_accuracy: 0.9118 - val_loss: 0.2813 - val_sparse_categorical_accuracy: 0.8849 - lr: 0.0010
Epoch 57/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2205 - sparse_categorical_accuracy: 0.9142 - val_loss: 0.2058 - val_sparse_categorical_accuracy: 0.9196 - lr: 0.0010
Epoch 58/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2083 - sparse_categorical_accuracy: 0.9194 - val_loss: 0.2344 - val_sparse_categorical_accuracy: 0.8877 - lr: 0.0010
Epoch 59/500
90/90 [==============================] - 1s 7ms/step - loss: 0.2034 - sparse_categorical_accuracy: 0.9233 - val_loss: 0.2056 - val_sparse_categorical_accuracy: 0.9140 - lr: 0.0010
Epoch 60/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1921 - sparse_categorical_accuracy: 0.9236 - val_loss: 0.2836 - val_sparse_categorical_accuracy: 0.8724 - lr: 0.0010
Epoch 61/500
90/90 [==============================] - 1s 6ms/step - loss: 0.2000 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.6413 - val_sparse_categorical_accuracy: 0.7309 - lr: 0.0010
Epoch 62/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1960 - sparse_categorical_accuracy: 0.9257 - val_loss: 0.3199 - val_sparse_categorical_accuracy: 0.8502 - lr: 0.0010
Epoch 63/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1877 - sparse_categorical_accuracy: 0.9285 - val_loss: 0.3911 - val_sparse_categorical_accuracy: 0.8322 - lr: 0.0010
Epoch 64/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1722 - sparse_categorical_accuracy: 0.9392 - val_loss: 0.2382 - val_sparse_categorical_accuracy: 0.9140 - lr: 0.0010
Epoch 65/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1656 - sparse_categorical_accuracy: 0.9413 - val_loss: 0.1893 - val_sparse_categorical_accuracy: 0.9348 - lr: 0.0010
Epoch 66/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1700 - sparse_categorical_accuracy: 0.9330 - val_loss: 0.4316 - val_sparse_categorical_accuracy: 0.8100 - lr: 0.0010
Epoch 67/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1545 - sparse_categorical_accuracy: 0.9424 - val_loss: 0.1977 - val_sparse_categorical_accuracy: 0.9362 - lr: 0.0010
Epoch 68/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1477 - sparse_categorical_accuracy: 0.9420 - val_loss: 0.2205 - val_sparse_categorical_accuracy: 0.9029 - lr: 0.0010
Epoch 69/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1394 - sparse_categorical_accuracy: 0.9493 - val_loss: 0.4338 - val_sparse_categorical_accuracy: 0.8114 - lr: 0.0010
Epoch 70/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1308 - sparse_categorical_accuracy: 0.9521 - val_loss: 0.1949 - val_sparse_categorical_accuracy: 0.9209 - lr: 0.0010
Epoch 71/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1444 - sparse_categorical_accuracy: 0.9521 - val_loss: 0.3488 - val_sparse_categorical_accuracy: 0.8363 - lr: 0.0010
Epoch 72/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1292 - sparse_categorical_accuracy: 0.9528 - val_loss: 0.2802 - val_sparse_categorical_accuracy: 0.8738 - lr: 0.0010
Epoch 73/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1365 - sparse_categorical_accuracy: 0.9517 - val_loss: 0.1396 - val_sparse_categorical_accuracy: 0.9528 - lr: 0.0010
Epoch 74/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1338 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.1790 - val_sparse_categorical_accuracy: 0.9404 - lr: 0.0010
Epoch 75/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1318 - sparse_categorical_accuracy: 0.9559 - val_loss: 0.2393 - val_sparse_categorical_accuracy: 0.8974 - lr: 0.0010
Epoch 76/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1334 - sparse_categorical_accuracy: 0.9528 - val_loss: 1.2081 - val_sparse_categorical_accuracy: 0.6963 - lr: 0.0010
Epoch 77/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1229 - sparse_categorical_accuracy: 0.9608 - val_loss: 3.0677 - val_sparse_categorical_accuracy: 0.6602 - lr: 0.0010
Epoch 78/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1254 - sparse_categorical_accuracy: 0.9556 - val_loss: 0.8715 - val_sparse_categorical_accuracy: 0.7295 - lr: 0.0010
Epoch 79/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1175 - sparse_categorical_accuracy: 0.9628 - val_loss: 0.5537 - val_sparse_categorical_accuracy: 0.7864 - lr: 0.0010
Epoch 80/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1129 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.8497 - val_sparse_categorical_accuracy: 0.6893 - lr: 0.0010
Epoch 81/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1250 - sparse_categorical_accuracy: 0.9524 - val_loss: 0.3366 - val_sparse_categorical_accuracy: 0.8558 - lr: 0.0010
Epoch 82/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1313 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.6747 - val_sparse_categorical_accuracy: 0.7476 - lr: 0.0010
Epoch 83/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1165 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.1477 - val_sparse_categorical_accuracy: 0.9431 - lr: 0.0010
Epoch 84/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1134 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.1024 - val_sparse_categorical_accuracy: 0.9653 - lr: 0.0010
Epoch 85/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1100 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.1122 - val_sparse_categorical_accuracy: 0.9570 - lr: 0.0010
Epoch 86/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1130 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.3038 - val_sparse_categorical_accuracy: 0.8627 - lr: 0.0010
Epoch 87/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1185 - sparse_categorical_accuracy: 0.9563 - val_loss: 1.8756 - val_sparse_categorical_accuracy: 0.5770 - lr: 0.0010
Epoch 88/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1214 - sparse_categorical_accuracy: 0.9583 - val_loss: 1.3279 - val_sparse_categorical_accuracy: 0.6366 - lr: 0.0010
Epoch 89/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1106 - sparse_categorical_accuracy: 0.9608 - val_loss: 4.0726 - val_sparse_categorical_accuracy: 0.6227 - lr: 0.0010
Epoch 90/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1094 - sparse_categorical_accuracy: 0.9632 - val_loss: 1.0566 - val_sparse_categorical_accuracy: 0.6921 - lr: 0.0010
Epoch 91/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1074 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.2565 - val_sparse_categorical_accuracy: 0.8904 - lr: 0.0010
Epoch 92/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1172 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.1518 - val_sparse_categorical_accuracy: 0.9515 - lr: 0.0010
Epoch 93/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1138 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.1244 - val_sparse_categorical_accuracy: 0.9542 - lr: 0.0010
Epoch 94/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1137 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.3159 - val_sparse_categorical_accuracy: 0.8571 - lr: 0.0010
Epoch 95/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1100 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.3246 - val_sparse_categorical_accuracy: 0.8516 - lr: 0.0010
Epoch 96/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1130 - sparse_categorical_accuracy: 0.9580 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9723 - lr: 0.0010
Epoch 97/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1110 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.7698 - val_sparse_categorical_accuracy: 0.7393 - lr: 0.0010
Epoch 98/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1040 - sparse_categorical_accuracy: 0.9663 - val_loss: 1.1191 - val_sparse_categorical_accuracy: 0.6935 - lr: 0.0010
Epoch 99/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1075 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.1783 - val_sparse_categorical_accuracy: 0.9334 - lr: 0.0010
Epoch 100/500
90/90 [==============================] - 1s 10ms/step - loss: 0.0987 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9695 - lr: 0.0010
Epoch 101/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1147 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9695 - lr: 0.0010
Epoch 102/500
90/90 [==============================] - 1s 8ms/step - loss: 0.1033 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.7451 - val_sparse_categorical_accuracy: 0.7115 - lr: 0.0010
Epoch 103/500
90/90 [==============================] - 1s 9ms/step - loss: 0.1011 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1700 - val_sparse_categorical_accuracy: 0.9348 - lr: 0.0010
Epoch 104/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0962 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.0945 - val_sparse_categorical_accuracy: 0.9681 - lr: 0.0010
Epoch 105/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1037 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1198 - val_sparse_categorical_accuracy: 0.9570 - lr: 0.0010
Epoch 106/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1103 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.3393 - val_sparse_categorical_accuracy: 0.8571 - lr: 0.0010
Epoch 107/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1004 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.8413 - val_sparse_categorical_accuracy: 0.7587 - lr: 0.0010
Epoch 108/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0957 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1240 - val_sparse_categorical_accuracy: 0.9612 - lr: 0.0010
Epoch 109/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0968 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.1338 - val_sparse_categorical_accuracy: 0.9528 - lr: 0.0010
Epoch 110/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1048 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.1082 - val_sparse_categorical_accuracy: 0.9653 - lr: 0.0010
Epoch 111/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1068 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.2642 - val_sparse_categorical_accuracy: 0.8904 - lr: 0.0010
Epoch 112/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1044 - sparse_categorical_accuracy: 0.9632 - val_loss: 0.1587 - val_sparse_categorical_accuracy: 0.9362 - lr: 0.0010
Epoch 113/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0967 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.2346 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
Epoch 114/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1009 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.2763 - val_sparse_categorical_accuracy: 0.8793 - lr: 0.0010
Epoch 115/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.4130 - val_sparse_categorical_accuracy: 0.8322 - lr: 0.0010
Epoch 116/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0985 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1494 - val_sparse_categorical_accuracy: 0.9487 - lr: 0.0010
Epoch 117/500
90/90 [==============================] - 1s 7ms/step - loss: 0.1035 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.9355 - val_sparse_categorical_accuracy: 0.7046 - lr: 0.0010
Epoch 118/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0993 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1639 - val_sparse_categorical_accuracy: 0.9445 - lr: 0.0010
Epoch 119/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0955 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1815 - val_sparse_categorical_accuracy: 0.9223 - lr: 0.0010
Epoch 120/500
90/90 [==============================] - 1s 6ms/step - loss: 0.1041 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.1142 - val_sparse_categorical_accuracy: 0.9695 - lr: 0.0010
Epoch 121/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0932 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.1263 - val_sparse_categorical_accuracy: 0.9542 - lr: 5.0000e-04
Epoch 122/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0869 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1404 - val_sparse_categorical_accuracy: 0.9473 - lr: 5.0000e-04
Epoch 123/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0836 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.1327 - val_sparse_categorical_accuracy: 0.9542 - lr: 5.0000e-04
Epoch 124/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0945 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.2469 - val_sparse_categorical_accuracy: 0.8918 - lr: 5.0000e-04
Epoch 125/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0890 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1340 - val_sparse_categorical_accuracy: 0.9487 - lr: 5.0000e-04
Epoch 126/500
90/90 [==============================] - 1s 10ms/step - loss: 0.0874 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.0901 - val_sparse_categorical_accuracy: 0.9764 - lr: 5.0000e-04
Epoch 127/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0818 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0975 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 128/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0825 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.1198 - val_sparse_categorical_accuracy: 0.9639 - lr: 5.0000e-04
Epoch 129/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0850 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1145 - val_sparse_categorical_accuracy: 0.9667 - lr: 5.0000e-04
Epoch 130/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0872 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1116 - val_sparse_categorical_accuracy: 0.9612 - lr: 5.0000e-04
Epoch 131/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 132/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0823 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9667 - lr: 5.0000e-04
Epoch 133/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0868 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9736 - lr: 5.0000e-04
Epoch 134/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.1553 - val_sparse_categorical_accuracy: 0.9376 - lr: 5.0000e-04
Epoch 135/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0920 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 136/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0807 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1719 - val_sparse_categorical_accuracy: 0.9320 - lr: 5.0000e-04
Epoch 137/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0789 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0868 - val_sparse_categorical_accuracy: 0.9709 - lr: 5.0000e-04
Epoch 138/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1056 - val_sparse_categorical_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 139/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1345 - val_sparse_categorical_accuracy: 0.9473 - lr: 5.0000e-04
Epoch 140/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0822 - val_sparse_categorical_accuracy: 0.9736 - lr: 5.0000e-04
Epoch 141/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0778 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1161 - val_sparse_categorical_accuracy: 0.9626 - lr: 5.0000e-04
Epoch 142/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0834 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.2792 - val_sparse_categorical_accuracy: 0.8849 - lr: 5.0000e-04
Epoch 143/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0852 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1147 - val_sparse_categorical_accuracy: 0.9653 - lr: 5.0000e-04
Epoch 144/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0820 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.0992 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 145/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0821 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.0954 - val_sparse_categorical_accuracy: 0.9723 - lr: 5.0000e-04
Epoch 146/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.1140 - val_sparse_categorical_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 147/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1454 - val_sparse_categorical_accuracy: 0.9431 - lr: 5.0000e-04
Epoch 148/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0737 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.1489 - val_sparse_categorical_accuracy: 0.9390 - lr: 5.0000e-04
Epoch 149/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0853 - val_sparse_categorical_accuracy: 0.9723 - lr: 5.0000e-04
Epoch 150/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.0792 - val_sparse_categorical_accuracy: 0.9764 - lr: 5.0000e-04
Epoch 151/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0792 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1190 - val_sparse_categorical_accuracy: 0.9584 - lr: 5.0000e-04
Epoch 152/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0760 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0876 - val_sparse_categorical_accuracy: 0.9736 - lr: 5.0000e-04
Epoch 153/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0747 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1696 - val_sparse_categorical_accuracy: 0.9404 - lr: 5.0000e-04
Epoch 154/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0847 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9695 - lr: 5.0000e-04
Epoch 155/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9705 - val_loss: 0.1492 - val_sparse_categorical_accuracy: 0.9362 - lr: 5.0000e-04
Epoch 156/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0839 - val_sparse_categorical_accuracy: 0.9750 - lr: 5.0000e-04
Epoch 157/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0784 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.1268 - val_sparse_categorical_accuracy: 0.9487 - lr: 5.0000e-04
Epoch 158/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.2980 - val_sparse_categorical_accuracy: 0.8821 - lr: 5.0000e-04
Epoch 159/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0763 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1550 - val_sparse_categorical_accuracy: 0.9334 - lr: 5.0000e-04
Epoch 160/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1058 - val_sparse_categorical_accuracy: 0.9626 - lr: 5.0000e-04
Epoch 161/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9709 - lr: 5.0000e-04
Epoch 162/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0709 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0807 - val_sparse_categorical_accuracy: 0.9736 - lr: 5.0000e-04
Epoch 163/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0875 - sparse_categorical_accuracy: 0.9698 - val_loss: 0.2034 - val_sparse_categorical_accuracy: 0.9223 - lr: 5.0000e-04
Epoch 164/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0800 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0843 - val_sparse_categorical_accuracy: 0.9723 - lr: 5.0000e-04
Epoch 165/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0864 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.0846 - val_sparse_categorical_accuracy: 0.9709 - lr: 5.0000e-04
Epoch 166/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.4184 - val_sparse_categorical_accuracy: 0.8460 - lr: 5.0000e-04
Epoch 167/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0897 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.1920 - val_sparse_categorical_accuracy: 0.9209 - lr: 5.0000e-04
Epoch 168/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0731 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9764 - lr: 5.0000e-04
Epoch 169/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0756 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1499 - val_sparse_categorical_accuracy: 0.9404 - lr: 5.0000e-04
Epoch 170/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9764 - lr: 5.0000e-04
Epoch 171/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9750 - lr: 2.5000e-04
Epoch 172/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0800 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 173/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0811 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 174/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0664 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0867 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 175/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0629 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 176/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9750 - lr: 2.5000e-04
Epoch 177/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0828 - val_sparse_categorical_accuracy: 0.9750 - lr: 2.5000e-04
Epoch 178/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0710 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.1355 - val_sparse_categorical_accuracy: 0.9528 - lr: 2.5000e-04
Epoch 179/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0650 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0812 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 180/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0726 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0802 - val_sparse_categorical_accuracy: 0.9778 - lr: 2.5000e-04
Epoch 181/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0947 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 182/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 183/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0648 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0804 - val_sparse_categorical_accuracy: 0.9750 - lr: 2.5000e-04
Epoch 184/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0621 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0962 - val_sparse_categorical_accuracy: 0.9681 - lr: 2.5000e-04
Epoch 185/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0790 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 186/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0666 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0859 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 187/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0630 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9778 - lr: 2.5000e-04
Epoch 188/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0594 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0977 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 189/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0800 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 190/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0705 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0757 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 191/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0629 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0783 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 192/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0673 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0966 - val_sparse_categorical_accuracy: 0.9612 - lr: 2.5000e-04
Epoch 193/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0619 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0944 - val_sparse_categorical_accuracy: 0.9736 - lr: 2.5000e-04
Epoch 194/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0838 - val_sparse_categorical_accuracy: 0.9778 - lr: 2.5000e-04
Epoch 195/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0659 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0792 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 196/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0624 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9709 - lr: 2.5000e-04
Epoch 197/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.1255 - val_sparse_categorical_accuracy: 0.9612 - lr: 2.5000e-04
Epoch 198/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0635 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 199/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0628 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0780 - val_sparse_categorical_accuracy: 0.9806 - lr: 2.5000e-04
Epoch 200/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0621 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0783 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 201/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0618 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.1168 - val_sparse_categorical_accuracy: 0.9626 - lr: 2.5000e-04
Epoch 202/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0686 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0855 - val_sparse_categorical_accuracy: 0.9750 - lr: 2.5000e-04
Epoch 203/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0623 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0989 - val_sparse_categorical_accuracy: 0.9723 - lr: 2.5000e-04
Epoch 204/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0636 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0811 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 205/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0641 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0793 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 206/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1018 - val_sparse_categorical_accuracy: 0.9653 - lr: 2.5000e-04
Epoch 207/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0630 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 208/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1163 - val_sparse_categorical_accuracy: 0.9639 - lr: 2.5000e-04
Epoch 209/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0675 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9778 - lr: 2.5000e-04
Epoch 210/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0577 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0894 - val_sparse_categorical_accuracy: 0.9764 - lr: 2.5000e-04
Epoch 211/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0623 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0760 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 212/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0567 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0757 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 213/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0527 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 214/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0556 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0874 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 215/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0546 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0771 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 216/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0594 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0786 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 217/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0753 - val_sparse_categorical_accuracy: 0.9806 - lr: 1.2500e-04
Epoch 218/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0596 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0805 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 219/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0574 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0996 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 220/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1028 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 221/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0615 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0799 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 222/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0823 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 223/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0634 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0776 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 224/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0573 - sparse_categorical_accuracy: 0.9847 - val_loss: 0.0771 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 225/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0593 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0785 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 226/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0607 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0752 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 227/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0521 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 228/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0567 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0804 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.2500e-04
Epoch 229/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 230/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0548 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0764 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 231/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0565 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0741 - val_sparse_categorical_accuracy: 0.9806 - lr: 1.2500e-04
Epoch 232/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0759 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 233/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0547 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0772 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 234/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0592 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0765 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 235/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0547 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0761 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 236/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0763 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 237/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0555 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0721 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 238/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0585 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0757 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 239/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0532 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.2500e-04
Epoch 240/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0617 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0769 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 241/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0781 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 242/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0567 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0878 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.2500e-04
Epoch 243/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0571 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0771 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 244/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0776 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 245/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0533 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0755 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 246/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0769 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 247/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0532 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 248/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0574 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9695 - lr: 1.2500e-04
Epoch 249/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0562 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0761 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 250/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0774 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 251/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0540 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0782 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.2500e-04
Epoch 252/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0537 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0769 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 253/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0599 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0774 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 254/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0530 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0907 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.2500e-04
Epoch 255/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0538 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0791 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 256/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0566 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0748 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.2500e-04
Epoch 257/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0759 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.2500e-04
Epoch 258/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0578 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0763 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 259/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0525 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0850 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 260/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0549 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0742 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 261/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0562 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.0761 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 262/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0785 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 263/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0551 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 264/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0541 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0742 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 265/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0506 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0801 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 266/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0528 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0767 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 267/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0762 - val_sparse_categorical_accuracy: 0.9806 - lr: 1.0000e-04
Epoch 268/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0486 - sparse_categorical_accuracy: 0.9840 - val_loss: 0.0754 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 269/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0547 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0768 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 270/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0828 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 271/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0543 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0738 - val_sparse_categorical_accuracy: 0.9806 - lr: 1.0000e-04
Epoch 272/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0523 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0767 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 273/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0535 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0749 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 274/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0512 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0749 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 275/500
90/90 [==============================] - 1s 8ms/step - loss: 0.0533 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0740 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 276/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0533 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0870 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 277/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0557 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0795 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 278/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0809 - val_sparse_categorical_accuracy: 0.9736 - lr: 1.0000e-04
Epoch 279/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0751 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 280/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0530 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9709 - lr: 1.0000e-04
Epoch 281/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0523 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0787 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 282/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0539 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0891 - val_sparse_categorical_accuracy: 0.9723 - lr: 1.0000e-04
Epoch 283/500
90/90 [==============================] - 1s 9ms/step - loss: 0.0491 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0753 - val_sparse_categorical_accuracy: 0.9820 - lr: 1.0000e-04
Epoch 284/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0587 - sparse_categorical_accuracy: 0.9830 - val_loss: 0.0783 - val_sparse_categorical_accuracy: 0.9792 - lr: 1.0000e-04
Epoch 285/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0508 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0745 - val_sparse_categorical_accuracy: 0.9764 - lr: 1.0000e-04
Epoch 286/500
90/90 [==============================] - 1s 6ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9851 - val_loss: 0.0817 - val_sparse_categorical_accuracy: 0.9750 - lr: 1.0000e-04
Epoch 287/500
90/90 [==============================] - 1s 7ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0735 - val_sparse_categorical_accuracy: 0.9778 - lr: 1.0000e-04
Epoch 287: early stopping

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
42/42 [==============================] - 0s 5ms/step - loss: 0.0942 - sparse_categorical_accuracy: 0.9689
Test accuracy 0.9689394235610962
Test loss 0.09415469318628311

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
