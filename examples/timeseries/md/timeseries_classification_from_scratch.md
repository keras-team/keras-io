# Timeseries classification from scratch

**Author:** [hfawaz](https://github.com/hfawaz/)<br>
**Date created:** 2020/07/21<br>
**Last modified:** 2023/11/10<br>
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
import keras
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
        "best_model.keras", save_best_only=True, monitor="val_loss"
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
 90/90 ━━━━━━━━━━━━━━━━━━━━ 5s 32ms/step - loss: 0.6056 - sparse_categorical_accuracy: 0.6818 - val_loss: 0.9692 - val_sparse_categorical_accuracy: 0.4591 - learning_rate: 0.0010
Epoch 2/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4623 - sparse_categorical_accuracy: 0.7619 - val_loss: 0.9336 - val_sparse_categorical_accuracy: 0.4591 - learning_rate: 0.0010
Epoch 3/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4383 - sparse_categorical_accuracy: 0.7888 - val_loss: 0.6842 - val_sparse_categorical_accuracy: 0.5409 - learning_rate: 0.0010
Epoch 4/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4295 - sparse_categorical_accuracy: 0.7826 - val_loss: 0.6632 - val_sparse_categorical_accuracy: 0.5118 - learning_rate: 0.0010
Epoch 5/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4311 - sparse_categorical_accuracy: 0.7831 - val_loss: 0.5693 - val_sparse_categorical_accuracy: 0.6935 - learning_rate: 0.0010
Epoch 6/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4250 - sparse_categorical_accuracy: 0.7832 - val_loss: 0.5001 - val_sparse_categorical_accuracy: 0.7712 - learning_rate: 0.0010
Epoch 7/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4179 - sparse_categorical_accuracy: 0.8079 - val_loss: 0.5151 - val_sparse_categorical_accuracy: 0.7379 - learning_rate: 0.0010
Epoch 8/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3929 - sparse_categorical_accuracy: 0.8073 - val_loss: 0.3992 - val_sparse_categorical_accuracy: 0.8377 - learning_rate: 0.0010
Epoch 9/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4074 - sparse_categorical_accuracy: 0.7947 - val_loss: 0.4053 - val_sparse_categorical_accuracy: 0.8225 - learning_rate: 0.0010
Epoch 10/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4067 - sparse_categorical_accuracy: 0.7984 - val_loss: 0.3727 - val_sparse_categorical_accuracy: 0.8377 - learning_rate: 0.0010
Epoch 11/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3910 - sparse_categorical_accuracy: 0.8083 - val_loss: 0.3687 - val_sparse_categorical_accuracy: 0.8363 - learning_rate: 0.0010
Epoch 12/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3872 - sparse_categorical_accuracy: 0.8001 - val_loss: 0.3773 - val_sparse_categorical_accuracy: 0.8169 - learning_rate: 0.0010
Epoch 13/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3684 - sparse_categorical_accuracy: 0.8138 - val_loss: 0.3566 - val_sparse_categorical_accuracy: 0.8474 - learning_rate: 0.0010
Epoch 14/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3843 - sparse_categorical_accuracy: 0.8102 - val_loss: 0.3674 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
Epoch 15/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3774 - sparse_categorical_accuracy: 0.8260 - val_loss: 0.4040 - val_sparse_categorical_accuracy: 0.7614 - learning_rate: 0.0010
Epoch 16/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3547 - sparse_categorical_accuracy: 0.8351 - val_loss: 0.6609 - val_sparse_categorical_accuracy: 0.6671 - learning_rate: 0.0010
Epoch 17/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3797 - sparse_categorical_accuracy: 0.8194 - val_loss: 0.3379 - val_sparse_categorical_accuracy: 0.8599 - learning_rate: 0.0010
Epoch 18/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3544 - sparse_categorical_accuracy: 0.8373 - val_loss: 0.3363 - val_sparse_categorical_accuracy: 0.8613 - learning_rate: 0.0010
Epoch 19/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3372 - sparse_categorical_accuracy: 0.8477 - val_loss: 0.4554 - val_sparse_categorical_accuracy: 0.7545 - learning_rate: 0.0010
Epoch 20/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3509 - sparse_categorical_accuracy: 0.8330 - val_loss: 0.4411 - val_sparse_categorical_accuracy: 0.7490 - learning_rate: 0.0010
Epoch 21/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3771 - sparse_categorical_accuracy: 0.8195 - val_loss: 0.3526 - val_sparse_categorical_accuracy: 0.8225 - learning_rate: 0.0010
Epoch 22/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3448 - sparse_categorical_accuracy: 0.8373 - val_loss: 0.3296 - val_sparse_categorical_accuracy: 0.8669 - learning_rate: 0.0010
Epoch 23/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3400 - sparse_categorical_accuracy: 0.8455 - val_loss: 0.3938 - val_sparse_categorical_accuracy: 0.7656 - learning_rate: 0.0010
Epoch 24/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3243 - sparse_categorical_accuracy: 0.8626 - val_loss: 0.8280 - val_sparse_categorical_accuracy: 0.5534 - learning_rate: 0.0010
Epoch 25/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3263 - sparse_categorical_accuracy: 0.8518 - val_loss: 0.3881 - val_sparse_categorical_accuracy: 0.8031 - learning_rate: 0.0010
Epoch 26/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3424 - sparse_categorical_accuracy: 0.8491 - val_loss: 0.3140 - val_sparse_categorical_accuracy: 0.8766 - learning_rate: 0.0010
Epoch 27/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3236 - sparse_categorical_accuracy: 0.8551 - val_loss: 0.3138 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 0.0010
Epoch 28/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3161 - sparse_categorical_accuracy: 0.8605 - val_loss: 0.3419 - val_sparse_categorical_accuracy: 0.8294 - learning_rate: 0.0010
Epoch 29/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3077 - sparse_categorical_accuracy: 0.8660 - val_loss: 0.3326 - val_sparse_categorical_accuracy: 0.8460 - learning_rate: 0.0010
Epoch 30/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3257 - sparse_categorical_accuracy: 0.8527 - val_loss: 0.2964 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 31/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2990 - sparse_categorical_accuracy: 0.8754 - val_loss: 0.3273 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 0.0010
Epoch 32/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3046 - sparse_categorical_accuracy: 0.8618 - val_loss: 0.2882 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 33/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2998 - sparse_categorical_accuracy: 0.8759 - val_loss: 0.3532 - val_sparse_categorical_accuracy: 0.7989 - learning_rate: 0.0010
Epoch 34/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2750 - sparse_categorical_accuracy: 0.8753 - val_loss: 0.5120 - val_sparse_categorical_accuracy: 0.7365 - learning_rate: 0.0010
Epoch 35/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2784 - sparse_categorical_accuracy: 0.8862 - val_loss: 0.3159 - val_sparse_categorical_accuracy: 0.8752 - learning_rate: 0.0010
Epoch 36/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2661 - sparse_categorical_accuracy: 0.8982 - val_loss: 0.3643 - val_sparse_categorical_accuracy: 0.8433 - learning_rate: 0.0010
Epoch 37/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2769 - sparse_categorical_accuracy: 0.8814 - val_loss: 0.4004 - val_sparse_categorical_accuracy: 0.7947 - learning_rate: 0.0010
Epoch 38/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2963 - sparse_categorical_accuracy: 0.8679 - val_loss: 0.4778 - val_sparse_categorical_accuracy: 0.7323 - learning_rate: 0.0010
Epoch 39/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2688 - sparse_categorical_accuracy: 0.8851 - val_loss: 0.2490 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 40/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2696 - sparse_categorical_accuracy: 0.8872 - val_loss: 0.2792 - val_sparse_categorical_accuracy: 0.8821 - learning_rate: 0.0010
Epoch 41/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2880 - sparse_categorical_accuracy: 0.8868 - val_loss: 0.2775 - val_sparse_categorical_accuracy: 0.8752 - learning_rate: 0.0010
Epoch 42/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2884 - sparse_categorical_accuracy: 0.8774 - val_loss: 0.3545 - val_sparse_categorical_accuracy: 0.8128 - learning_rate: 0.0010
Epoch 43/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2840 - sparse_categorical_accuracy: 0.8709 - val_loss: 0.3292 - val_sparse_categorical_accuracy: 0.8350 - learning_rate: 0.0010
Epoch 44/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3000 - sparse_categorical_accuracy: 0.8569 - val_loss: 1.5013 - val_sparse_categorical_accuracy: 0.5479 - learning_rate: 0.0010
Epoch 45/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2618 - sparse_categorical_accuracy: 0.8896 - val_loss: 0.2766 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
Epoch 46/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2604 - sparse_categorical_accuracy: 0.8955 - val_loss: 0.2397 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 47/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2520 - sparse_categorical_accuracy: 0.8975 - val_loss: 0.3794 - val_sparse_categorical_accuracy: 0.7975 - learning_rate: 0.0010
Epoch 48/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2521 - sparse_categorical_accuracy: 0.9067 - val_loss: 0.2871 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 49/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2554 - sparse_categorical_accuracy: 0.8904 - val_loss: 0.8962 - val_sparse_categorical_accuracy: 0.7115 - learning_rate: 0.0010
Epoch 50/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2501 - sparse_categorical_accuracy: 0.8989 - val_loss: 0.4592 - val_sparse_categorical_accuracy: 0.7864 - learning_rate: 0.0010
Epoch 51/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2362 - sparse_categorical_accuracy: 0.8944 - val_loss: 0.4599 - val_sparse_categorical_accuracy: 0.7684 - learning_rate: 0.0010
Epoch 52/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2538 - sparse_categorical_accuracy: 0.8986 - val_loss: 0.2748 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
Epoch 53/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2648 - sparse_categorical_accuracy: 0.8934 - val_loss: 0.2725 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 0.0010
Epoch 54/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2292 - sparse_categorical_accuracy: 0.9117 - val_loss: 0.2617 - val_sparse_categorical_accuracy: 0.8766 - learning_rate: 0.0010
Epoch 55/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2704 - sparse_categorical_accuracy: 0.8826 - val_loss: 0.2929 - val_sparse_categorical_accuracy: 0.8488 - learning_rate: 0.0010
Epoch 56/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2388 - sparse_categorical_accuracy: 0.9022 - val_loss: 0.2365 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 57/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2309 - sparse_categorical_accuracy: 0.9087 - val_loss: 1.1993 - val_sparse_categorical_accuracy: 0.5784 - learning_rate: 0.0010
Epoch 58/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2639 - sparse_categorical_accuracy: 0.8893 - val_loss: 0.2410 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 59/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2229 - sparse_categorical_accuracy: 0.9104 - val_loss: 0.6126 - val_sparse_categorical_accuracy: 0.7212 - learning_rate: 0.0010
Epoch 60/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2451 - sparse_categorical_accuracy: 0.9084 - val_loss: 0.3189 - val_sparse_categorical_accuracy: 0.8655 - learning_rate: 0.0010
Epoch 61/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2200 - sparse_categorical_accuracy: 0.9169 - val_loss: 0.7695 - val_sparse_categorical_accuracy: 0.7212 - learning_rate: 0.0010
Epoch 62/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2249 - sparse_categorical_accuracy: 0.9149 - val_loss: 0.2900 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
Epoch 63/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2476 - sparse_categorical_accuracy: 0.8988 - val_loss: 0.2863 - val_sparse_categorical_accuracy: 0.8682 - learning_rate: 0.0010
Epoch 64/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2263 - sparse_categorical_accuracy: 0.9010 - val_loss: 0.4034 - val_sparse_categorical_accuracy: 0.7961 - learning_rate: 0.0010
Epoch 65/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2404 - sparse_categorical_accuracy: 0.9041 - val_loss: 0.2965 - val_sparse_categorical_accuracy: 0.8696 - learning_rate: 0.0010
Epoch 66/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2257 - sparse_categorical_accuracy: 0.9051 - val_loss: 0.2227 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
Epoch 67/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9088 - val_loss: 0.2274 - val_sparse_categorical_accuracy: 0.9154 - learning_rate: 0.0010
Epoch 68/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2106 - sparse_categorical_accuracy: 0.9159 - val_loss: 0.2703 - val_sparse_categorical_accuracy: 0.8877 - learning_rate: 0.0010
Epoch 69/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1945 - sparse_categorical_accuracy: 0.9278 - val_loss: 0.2688 - val_sparse_categorical_accuracy: 0.8724 - learning_rate: 0.0010
Epoch 70/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2269 - sparse_categorical_accuracy: 0.9108 - val_loss: 0.2003 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
Epoch 71/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2312 - sparse_categorical_accuracy: 0.9041 - val_loss: 0.3678 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
Epoch 72/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1828 - sparse_categorical_accuracy: 0.9290 - val_loss: 0.2397 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 73/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1723 - sparse_categorical_accuracy: 0.9364 - val_loss: 0.2070 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 74/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1830 - sparse_categorical_accuracy: 0.9317 - val_loss: 0.3114 - val_sparse_categorical_accuracy: 0.8391 - learning_rate: 0.0010
Epoch 75/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1786 - sparse_categorical_accuracy: 0.9345 - val_loss: 0.7721 - val_sparse_categorical_accuracy: 0.6824 - learning_rate: 0.0010
Epoch 76/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1680 - sparse_categorical_accuracy: 0.9444 - val_loss: 0.1898 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
Epoch 77/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1606 - sparse_categorical_accuracy: 0.9426 - val_loss: 0.1803 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
Epoch 78/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1705 - sparse_categorical_accuracy: 0.9292 - val_loss: 0.6892 - val_sparse_categorical_accuracy: 0.7226 - learning_rate: 0.0010
Epoch 79/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1428 - sparse_categorical_accuracy: 0.9534 - val_loss: 0.2448 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 80/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1527 - sparse_categorical_accuracy: 0.9441 - val_loss: 0.3191 - val_sparse_categorical_accuracy: 0.8377 - learning_rate: 0.0010
Epoch 81/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1398 - sparse_categorical_accuracy: 0.9447 - val_loss: 0.9834 - val_sparse_categorical_accuracy: 0.6366 - learning_rate: 0.0010
Epoch 82/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1615 - sparse_categorical_accuracy: 0.9405 - val_loss: 0.3857 - val_sparse_categorical_accuracy: 0.8391 - learning_rate: 0.0010
Epoch 83/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1371 - sparse_categorical_accuracy: 0.9525 - val_loss: 0.1597 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 0.0010
Epoch 84/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1377 - sparse_categorical_accuracy: 0.9548 - val_loss: 0.4212 - val_sparse_categorical_accuracy: 0.8058 - learning_rate: 0.0010
Epoch 85/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1315 - sparse_categorical_accuracy: 0.9585 - val_loss: 0.3091 - val_sparse_categorical_accuracy: 0.8447 - learning_rate: 0.0010
Epoch 86/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1381 - sparse_categorical_accuracy: 0.9517 - val_loss: 0.1539 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
Epoch 87/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1169 - sparse_categorical_accuracy: 0.9581 - val_loss: 0.1927 - val_sparse_categorical_accuracy: 0.9168 - learning_rate: 0.0010
Epoch 88/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1438 - sparse_categorical_accuracy: 0.9512 - val_loss: 0.1696 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
Epoch 89/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1471 - sparse_categorical_accuracy: 0.9464 - val_loss: 0.2523 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 90/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1389 - sparse_categorical_accuracy: 0.9535 - val_loss: 0.2452 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
Epoch 91/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1209 - sparse_categorical_accuracy: 0.9599 - val_loss: 0.3986 - val_sparse_categorical_accuracy: 0.8183 - learning_rate: 0.0010
Epoch 92/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1278 - sparse_categorical_accuracy: 0.9520 - val_loss: 0.2153 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 0.0010
Epoch 93/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1080 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.1532 - val_sparse_categorical_accuracy: 0.9459 - learning_rate: 0.0010
Epoch 94/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1236 - sparse_categorical_accuracy: 0.9671 - val_loss: 0.1580 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 0.0010
Epoch 95/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0982 - sparse_categorical_accuracy: 0.9645 - val_loss: 0.1922 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 0.0010
Epoch 96/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1165 - sparse_categorical_accuracy: 0.9630 - val_loss: 0.3719 - val_sparse_categorical_accuracy: 0.8377 - learning_rate: 0.0010
Epoch 97/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1207 - sparse_categorical_accuracy: 0.9655 - val_loss: 0.2266 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 98/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1431 - sparse_categorical_accuracy: 0.9530 - val_loss: 0.1165 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 0.0010
Epoch 99/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1262 - sparse_categorical_accuracy: 0.9553 - val_loss: 0.1814 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 0.0010
Epoch 100/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0983 - sparse_categorical_accuracy: 0.9714 - val_loss: 0.1264 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 0.0010
Epoch 101/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1366 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.1222 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 0.0010
Epoch 102/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1156 - sparse_categorical_accuracy: 0.9602 - val_loss: 0.3325 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 0.0010
Epoch 103/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1231 - sparse_categorical_accuracy: 0.9544 - val_loss: 0.7861 - val_sparse_categorical_accuracy: 0.7074 - learning_rate: 0.0010
Epoch 104/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1081 - sparse_categorical_accuracy: 0.9653 - val_loss: 0.1329 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 0.0010
Epoch 105/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1171 - sparse_categorical_accuracy: 0.9585 - val_loss: 0.1094 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 0.0010
Epoch 106/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1110 - sparse_categorical_accuracy: 0.9633 - val_loss: 0.1403 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
Epoch 107/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1308 - sparse_categorical_accuracy: 0.9523 - val_loss: 0.2915 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
Epoch 108/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1062 - sparse_categorical_accuracy: 0.9662 - val_loss: 0.1033 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 0.0010
Epoch 109/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1167 - sparse_categorical_accuracy: 0.9614 - val_loss: 0.1259 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 0.0010
Epoch 110/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1037 - sparse_categorical_accuracy: 0.9676 - val_loss: 0.1180 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 0.0010
Epoch 111/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1156 - sparse_categorical_accuracy: 0.9626 - val_loss: 0.1534 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 0.0010
Epoch 112/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1165 - sparse_categorical_accuracy: 0.9559 - val_loss: 0.2067 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 0.0010
Epoch 113/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1163 - sparse_categorical_accuracy: 0.9574 - val_loss: 0.4253 - val_sparse_categorical_accuracy: 0.8044 - learning_rate: 0.0010
Epoch 114/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1148 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.1323 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 0.0010
Epoch 115/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1055 - sparse_categorical_accuracy: 0.9627 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 0.0010
Epoch 116/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0910 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.7235 - val_sparse_categorical_accuracy: 0.6963 - learning_rate: 0.0010
Epoch 117/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1308 - sparse_categorical_accuracy: 0.9597 - val_loss: 0.1575 - val_sparse_categorical_accuracy: 0.9348 - learning_rate: 0.0010
Epoch 118/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1368 - sparse_categorical_accuracy: 0.9433 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 0.0010
Epoch 119/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0995 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.1788 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
Epoch 120/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1221 - sparse_categorical_accuracy: 0.9506 - val_loss: 0.1161 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 0.0010
Epoch 121/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0921 - sparse_categorical_accuracy: 0.9741 - val_loss: 0.1154 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 0.0010
Epoch 122/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1081 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.1153 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 0.0010
Epoch 123/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0962 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1808 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
Epoch 124/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1115 - sparse_categorical_accuracy: 0.9634 - val_loss: 0.1017 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 0.0010
Epoch 125/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1032 - sparse_categorical_accuracy: 0.9657 - val_loss: 0.1763 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
Epoch 126/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1088 - sparse_categorical_accuracy: 0.9628 - val_loss: 0.1823 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
Epoch 127/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1095 - sparse_categorical_accuracy: 0.9637 - val_loss: 0.1089 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 0.0010
Epoch 128/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1316 - sparse_categorical_accuracy: 0.9547 - val_loss: 0.1416 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
Epoch 129/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1051 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.2307 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 0.0010
Epoch 130/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1051 - sparse_categorical_accuracy: 0.9692 - val_loss: 1.0068 - val_sparse_categorical_accuracy: 0.6338 - learning_rate: 0.0010
Epoch 131/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1052 - sparse_categorical_accuracy: 0.9620 - val_loss: 0.2687 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 132/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1045 - sparse_categorical_accuracy: 0.9647 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 0.0010
Epoch 133/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0953 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1996 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
Epoch 134/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1149 - sparse_categorical_accuracy: 0.9612 - val_loss: 0.4479 - val_sparse_categorical_accuracy: 0.8044 - learning_rate: 0.0010
Epoch 135/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0913 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0993 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 0.0010
Epoch 136/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1211 - sparse_categorical_accuracy: 0.9586 - val_loss: 0.1036 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 0.0010
Epoch 137/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0910 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.1525 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 0.0010
Epoch 138/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0986 - sparse_categorical_accuracy: 0.9633 - val_loss: 0.1699 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 0.0010
Epoch 139/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0886 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.0957 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 0.0010
Epoch 140/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1050 - sparse_categorical_accuracy: 0.9652 - val_loss: 1.6603 - val_sparse_categorical_accuracy: 0.6366 - learning_rate: 0.0010
Epoch 141/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0922 - sparse_categorical_accuracy: 0.9676 - val_loss: 0.1741 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
Epoch 142/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1383 - sparse_categorical_accuracy: 0.9476 - val_loss: 0.2704 - val_sparse_categorical_accuracy: 0.8821 - learning_rate: 0.0010
Epoch 143/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1104 - sparse_categorical_accuracy: 0.9576 - val_loss: 0.3363 - val_sparse_categorical_accuracy: 0.8447 - learning_rate: 0.0010
Epoch 144/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1037 - sparse_categorical_accuracy: 0.9666 - val_loss: 0.4437 - val_sparse_categorical_accuracy: 0.8169 - learning_rate: 0.0010
Epoch 145/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0939 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.2474 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
Epoch 146/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1130 - sparse_categorical_accuracy: 0.9564 - val_loss: 0.1531 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 0.0010
Epoch 147/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1022 - sparse_categorical_accuracy: 0.9626 - val_loss: 0.1573 - val_sparse_categorical_accuracy: 0.9348 - learning_rate: 0.0010
Epoch 148/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0815 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1416 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 0.0010
Epoch 149/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0937 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.2065 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 150/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0955 - sparse_categorical_accuracy: 0.9672 - val_loss: 0.1146 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 0.0010
Epoch 151/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1097 - sparse_categorical_accuracy: 0.9560 - val_loss: 0.3142 - val_sparse_categorical_accuracy: 0.8599 - learning_rate: 0.0010
Epoch 152/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1017 - sparse_categorical_accuracy: 0.9636 - val_loss: 0.3406 - val_sparse_categorical_accuracy: 0.8433 - learning_rate: 0.0010
Epoch 153/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0930 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.0928 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 154/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0969 - sparse_categorical_accuracy: 0.9685 - val_loss: 0.2657 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 5.0000e-04
Epoch 155/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1045 - sparse_categorical_accuracy: 0.9634 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 156/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0915 - sparse_categorical_accuracy: 0.9699 - val_loss: 0.1175 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 5.0000e-04
Epoch 157/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0949 - sparse_categorical_accuracy: 0.9634 - val_loss: 0.1001 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 158/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0899 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 159/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0827 - sparse_categorical_accuracy: 0.9758 - val_loss: 0.1171 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
Epoch 160/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9686 - val_loss: 0.1056 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 161/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0765 - sparse_categorical_accuracy: 0.9777 - val_loss: 0.1604 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
Epoch 162/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0848 - sparse_categorical_accuracy: 0.9707 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 163/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0891 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.0882 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 164/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9721 - val_loss: 0.0989 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 165/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0810 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.2738 - val_sparse_categorical_accuracy: 0.8655 - learning_rate: 5.0000e-04
Epoch 166/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.0985 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 167/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1081 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
Epoch 168/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1182 - sparse_categorical_accuracy: 0.9519 - val_loss: 0.1212 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
Epoch 169/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0909 - sparse_categorical_accuracy: 0.9666 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 170/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0882 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.0912 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 171/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0863 - sparse_categorical_accuracy: 0.9735 - val_loss: 0.1391 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 5.0000e-04
Epoch 172/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0853 - sparse_categorical_accuracy: 0.9692 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 173/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0922 - sparse_categorical_accuracy: 0.9679 - val_loss: 0.0924 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 174/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0954 - sparse_categorical_accuracy: 0.9699 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 175/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0823 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1449 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 5.0000e-04
Epoch 176/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0853 - sparse_categorical_accuracy: 0.9692 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 177/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0834 - sparse_categorical_accuracy: 0.9692 - val_loss: 0.2338 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 5.0000e-04
Epoch 178/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.1609 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 5.0000e-04
Epoch 179/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0965 - sparse_categorical_accuracy: 0.9628 - val_loss: 0.5213 - val_sparse_categorical_accuracy: 0.7947 - learning_rate: 5.0000e-04
Epoch 180/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0926 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.0898 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 181/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0854 - sparse_categorical_accuracy: 0.9732 - val_loss: 0.0949 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 182/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0691 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0841 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 183/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9766 - val_loss: 0.1021 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 184/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0842 - sparse_categorical_accuracy: 0.9692 - val_loss: 0.1105 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 185/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0731 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.1487 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 5.0000e-04
Epoch 186/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0762 - sparse_categorical_accuracy: 0.9724 - val_loss: 0.1126 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 187/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9723 - val_loss: 0.0954 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 188/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9671 - val_loss: 0.1957 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 5.0000e-04
Epoch 189/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0831 - sparse_categorical_accuracy: 0.9695 - val_loss: 0.1711 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 5.0000e-04
Epoch 190/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0881 - sparse_categorical_accuracy: 0.9693 - val_loss: 0.0861 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 191/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0735 - sparse_categorical_accuracy: 0.9769 - val_loss: 0.1154 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
Epoch 192/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0877 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.0845 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 5.0000e-04
Epoch 193/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0899 - sparse_categorical_accuracy: 0.9709 - val_loss: 0.0977 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 194/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0843 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0969 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 195/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9765 - val_loss: 0.1345 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 5.0000e-04
Epoch 196/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0844 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 197/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0751 - sparse_categorical_accuracy: 0.9801 - val_loss: 0.2736 - val_sparse_categorical_accuracy: 0.8793 - learning_rate: 5.0000e-04
Epoch 198/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0860 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0843 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 199/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0835 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.1799 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 5.0000e-04
Epoch 200/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0760 - sparse_categorical_accuracy: 0.9745 - val_loss: 0.1790 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 5.0000e-04
Epoch 201/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0714 - sparse_categorical_accuracy: 0.9742 - val_loss: 0.0918 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 202/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.1168 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
Epoch 203/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0654 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0825 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 204/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0717 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.1186 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
Epoch 205/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9679 - val_loss: 0.0847 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 206/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0897 - sparse_categorical_accuracy: 0.9687 - val_loss: 0.0820 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 207/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9763 - val_loss: 0.0790 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 208/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0683 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0991 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 209/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.1057 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 2.5000e-04
Epoch 210/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0858 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 211/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0856 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 212/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.0835 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 213/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9761 - val_loss: 0.0894 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 214/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9800 - val_loss: 0.0788 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 215/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9744 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 216/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0655 - sparse_categorical_accuracy: 0.9819 - val_loss: 0.1158 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 217/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0722 - sparse_categorical_accuracy: 0.9777 - val_loss: 0.0940 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 218/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0750 - sparse_categorical_accuracy: 0.9761 - val_loss: 0.0966 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 219/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0695 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.1727 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 2.5000e-04
Epoch 220/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.1067 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 2.5000e-04
Epoch 221/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0848 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 222/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0675 - sparse_categorical_accuracy: 0.9808 - val_loss: 0.0931 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 223/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0695 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0785 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 224/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9822 - val_loss: 0.0820 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 225/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0637 - sparse_categorical_accuracy: 0.9772 - val_loss: 0.1084 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
Epoch 226/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.1029 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 227/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0821 - sparse_categorical_accuracy: 0.9704 - val_loss: 0.1545 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 2.5000e-04
Epoch 228/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0826 - sparse_categorical_accuracy: 0.9714 - val_loss: 0.0819 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 229/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0788 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 230/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0702 - sparse_categorical_accuracy: 0.9776 - val_loss: 0.1514 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
Epoch 231/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0749 - sparse_categorical_accuracy: 0.9775 - val_loss: 0.1150 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 232/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0732 - sparse_categorical_accuracy: 0.9794 - val_loss: 0.1110 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 233/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0667 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1451 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
Epoch 234/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9793 - val_loss: 0.0954 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 235/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0629 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 236/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0843 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 237/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0722 - sparse_categorical_accuracy: 0.9775 - val_loss: 0.1315 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 2.5000e-04
Epoch 238/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9744 - val_loss: 0.0969 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 239/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0697 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0890 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 240/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0640 - sparse_categorical_accuracy: 0.9811 - val_loss: 0.0812 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 241/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0637 - sparse_categorical_accuracy: 0.9852 - val_loss: 0.0750 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 242/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0645 - sparse_categorical_accuracy: 0.9772 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 2.5000e-04
Epoch 243/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9746 - val_loss: 0.0885 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 244/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0635 - sparse_categorical_accuracy: 0.9835 - val_loss: 0.1270 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 2.5000e-04
Epoch 245/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9761 - val_loss: 0.0803 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 246/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0635 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.0791 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 247/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0622 - sparse_categorical_accuracy: 0.9801 - val_loss: 0.0928 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 248/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9756 - val_loss: 0.0817 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 249/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0652 - sparse_categorical_accuracy: 0.9821 - val_loss: 0.0804 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 250/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0765 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 251/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.1128 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
Epoch 252/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0896 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 253/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9776 - val_loss: 0.1141 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
Epoch 254/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0648 - sparse_categorical_accuracy: 0.9783 - val_loss: 0.1578 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 2.5000e-04
Epoch 255/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0554 - sparse_categorical_accuracy: 0.9862 - val_loss: 0.0835 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 256/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0645 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.0930 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 257/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0645 - sparse_categorical_accuracy: 0.9838 - val_loss: 0.0784 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 258/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0733 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0867 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 259/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0601 - sparse_categorical_accuracy: 0.9836 - val_loss: 0.1279 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
Epoch 260/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9742 - val_loss: 0.1646 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
Epoch 261/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9755 - val_loss: 0.0781 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 262/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9798 - val_loss: 0.0775 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
Epoch 263/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0671 - sparse_categorical_accuracy: 0.9777 - val_loss: 0.1033 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.2500e-04
Epoch 264/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9831 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
Epoch 265/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9828 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
Epoch 266/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0653 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0834 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
Epoch 267/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0646 - sparse_categorical_accuracy: 0.9808 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
Epoch 268/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.0795 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
Epoch 269/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9737 - val_loss: 0.0812 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
Epoch 270/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0613 - sparse_categorical_accuracy: 0.9843 - val_loss: 0.0905 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
Epoch 271/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0624 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.1130 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 1.2500e-04
Epoch 272/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0654 - sparse_categorical_accuracy: 0.9794 - val_loss: 0.0784 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
Epoch 273/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9804 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
Epoch 274/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0627 - sparse_categorical_accuracy: 0.9842 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.2500e-04
Epoch 275/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0713 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0956 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.2500e-04
Epoch 276/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0631 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0805 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
Epoch 277/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0613 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.0982 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.2500e-04
Epoch 278/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0649 - sparse_categorical_accuracy: 0.9818 - val_loss: 0.0857 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.2500e-04
Epoch 279/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0668 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0845 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.2500e-04
Epoch 280/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0679 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0835 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.2500e-04
Epoch 281/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0766 - sparse_categorical_accuracy: 0.9734 - val_loss: 0.0810 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
Epoch 282/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9815 - val_loss: 0.0829 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
Epoch 283/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0676 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0856 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 1.0000e-04
Epoch 284/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0607 - sparse_categorical_accuracy: 0.9832 - val_loss: 0.0850 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
Epoch 285/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0723 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.0844 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
Epoch 286/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0620 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.1347 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 1.0000e-04
Epoch 287/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0641 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0765 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
Epoch 288/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0654 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.1081 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 1.0000e-04
Epoch 289/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1734 - val_sparse_categorical_accuracy: 0.9362 - learning_rate: 1.0000e-04
Epoch 290/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0771 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
Epoch 291/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0605 - sparse_categorical_accuracy: 0.9839 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
Epoch 291: early stopping

```
</div>
---
## Evaluate model on test data


```python
model = keras.models.load_model("best_model.keras")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
```

<div class="k-default-codeblock">
```
 42/42 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - loss: 0.0997 - sparse_categorical_accuracy: 0.9687
Test accuracy 0.9696969985961914
Test loss 0.09916326403617859

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
