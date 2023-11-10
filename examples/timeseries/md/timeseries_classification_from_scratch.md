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

<div class="k-default-codeblock">
```
You must install graphviz (see instructions at https://graphviz.gitlab.io/download/) for `plot_model` to work.

```
</div>
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
 90/90 ━━━━━━━━━━━━━━━━━━━━ 5s 32ms/step - loss: 0.6255 - sparse_categorical_accuracy: 0.6396 - val_loss: 0.8089 - val_sparse_categorical_accuracy: 0.4827 - learning_rate: 0.0010
Epoch 2/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4928 - sparse_categorical_accuracy: 0.7621 - val_loss: 0.7455 - val_sparse_categorical_accuracy: 0.4827 - learning_rate: 0.0010
Epoch 3/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4495 - sparse_categorical_accuracy: 0.7677 - val_loss: 0.7198 - val_sparse_categorical_accuracy: 0.4840 - learning_rate: 0.0010
Epoch 4/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4413 - sparse_categorical_accuracy: 0.7795 - val_loss: 0.7572 - val_sparse_categorical_accuracy: 0.4078 - learning_rate: 0.0010
Epoch 5/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4269 - sparse_categorical_accuracy: 0.7812 - val_loss: 0.5151 - val_sparse_categorical_accuracy: 0.7406 - learning_rate: 0.0010
Epoch 6/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4160 - sparse_categorical_accuracy: 0.7895 - val_loss: 0.5557 - val_sparse_categorical_accuracy: 0.6865 - learning_rate: 0.0010
Epoch 7/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4137 - sparse_categorical_accuracy: 0.8025 - val_loss: 0.8247 - val_sparse_categorical_accuracy: 0.5395 - learning_rate: 0.0010
Epoch 8/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3865 - sparse_categorical_accuracy: 0.8135 - val_loss: 0.4834 - val_sparse_categorical_accuracy: 0.7226 - learning_rate: 0.0010
Epoch 9/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3992 - sparse_categorical_accuracy: 0.7994 - val_loss: 0.3799 - val_sparse_categorical_accuracy: 0.8350 - learning_rate: 0.0010
Epoch 10/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3792 - sparse_categorical_accuracy: 0.8117 - val_loss: 0.3640 - val_sparse_categorical_accuracy: 0.8266 - learning_rate: 0.0010
Epoch 11/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3979 - sparse_categorical_accuracy: 0.7980 - val_loss: 0.6575 - val_sparse_categorical_accuracy: 0.6976 - learning_rate: 0.0010
Epoch 12/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3539 - sparse_categorical_accuracy: 0.8339 - val_loss: 0.3742 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 0.0010
Epoch 13/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3657 - sparse_categorical_accuracy: 0.8086 - val_loss: 0.4935 - val_sparse_categorical_accuracy: 0.7226 - learning_rate: 0.0010
Epoch 14/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3820 - sparse_categorical_accuracy: 0.8281 - val_loss: 0.4078 - val_sparse_categorical_accuracy: 0.7864 - learning_rate: 0.0010
Epoch 15/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3506 - sparse_categorical_accuracy: 0.8395 - val_loss: 0.3565 - val_sparse_categorical_accuracy: 0.8391 - learning_rate: 0.0010
Epoch 16/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3627 - sparse_categorical_accuracy: 0.8368 - val_loss: 0.5151 - val_sparse_categorical_accuracy: 0.7171 - learning_rate: 0.0010
Epoch 17/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3488 - sparse_categorical_accuracy: 0.8484 - val_loss: 0.3129 - val_sparse_categorical_accuracy: 0.8793 - learning_rate: 0.0010
Epoch 18/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3568 - sparse_categorical_accuracy: 0.8204 - val_loss: 0.3254 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 19/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3534 - sparse_categorical_accuracy: 0.8269 - val_loss: 0.3707 - val_sparse_categorical_accuracy: 0.8100 - learning_rate: 0.0010
Epoch 20/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3456 - sparse_categorical_accuracy: 0.8403 - val_loss: 0.3311 - val_sparse_categorical_accuracy: 0.8363 - learning_rate: 0.0010
Epoch 21/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3324 - sparse_categorical_accuracy: 0.8547 - val_loss: 0.3159 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
Epoch 22/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3288 - sparse_categorical_accuracy: 0.8547 - val_loss: 0.3665 - val_sparse_categorical_accuracy: 0.8336 - learning_rate: 0.0010
Epoch 23/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3414 - sparse_categorical_accuracy: 0.8445 - val_loss: 0.3156 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 24/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3240 - sparse_categorical_accuracy: 0.8528 - val_loss: 0.3587 - val_sparse_categorical_accuracy: 0.8017 - learning_rate: 0.0010
Epoch 25/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3403 - sparse_categorical_accuracy: 0.8356 - val_loss: 0.2816 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
Epoch 26/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3205 - sparse_categorical_accuracy: 0.8633 - val_loss: 0.3821 - val_sparse_categorical_accuracy: 0.7753 - learning_rate: 0.0010
Epoch 27/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3211 - sparse_categorical_accuracy: 0.8547 - val_loss: 0.3082 - val_sparse_categorical_accuracy: 0.8516 - learning_rate: 0.0010
Epoch 28/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3144 - sparse_categorical_accuracy: 0.8592 - val_loss: 0.2815 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 29/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2989 - sparse_categorical_accuracy: 0.8832 - val_loss: 0.6136 - val_sparse_categorical_accuracy: 0.6893 - learning_rate: 0.0010
Epoch 30/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2816 - sparse_categorical_accuracy: 0.8868 - val_loss: 0.2644 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 31/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3028 - sparse_categorical_accuracy: 0.8773 - val_loss: 0.3855 - val_sparse_categorical_accuracy: 0.8044 - learning_rate: 0.0010
Epoch 32/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3164 - sparse_categorical_accuracy: 0.8664 - val_loss: 0.2708 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
Epoch 33/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3071 - sparse_categorical_accuracy: 0.8669 - val_loss: 0.4591 - val_sparse_categorical_accuracy: 0.8044 - learning_rate: 0.0010
Epoch 34/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3454 - sparse_categorical_accuracy: 0.8442 - val_loss: 0.3072 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 0.0010
Epoch 35/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2895 - sparse_categorical_accuracy: 0.8814 - val_loss: 0.3076 - val_sparse_categorical_accuracy: 0.8613 - learning_rate: 0.0010
Epoch 36/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2800 - sparse_categorical_accuracy: 0.8799 - val_loss: 0.2553 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 37/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2819 - sparse_categorical_accuracy: 0.8872 - val_loss: 0.2646 - val_sparse_categorical_accuracy: 0.8766 - learning_rate: 0.0010
Epoch 38/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2771 - sparse_categorical_accuracy: 0.8864 - val_loss: 0.2869 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
Epoch 39/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2575 - sparse_categorical_accuracy: 0.8928 - val_loss: 0.2742 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 40/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2753 - sparse_categorical_accuracy: 0.8765 - val_loss: 0.2618 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
Epoch 41/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2748 - sparse_categorical_accuracy: 0.8791 - val_loss: 0.2734 - val_sparse_categorical_accuracy: 0.8807 - learning_rate: 0.0010
Epoch 42/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2863 - sparse_categorical_accuracy: 0.8695 - val_loss: 0.2689 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 43/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2776 - sparse_categorical_accuracy: 0.8879 - val_loss: 0.2732 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
Epoch 44/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2957 - sparse_categorical_accuracy: 0.8759 - val_loss: 0.2839 - val_sparse_categorical_accuracy: 0.8627 - learning_rate: 0.0010
Epoch 45/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2820 - sparse_categorical_accuracy: 0.8740 - val_loss: 0.3987 - val_sparse_categorical_accuracy: 0.7864 - learning_rate: 0.0010
Epoch 46/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2566 - sparse_categorical_accuracy: 0.8989 - val_loss: 0.2167 - val_sparse_categorical_accuracy: 0.9182 - learning_rate: 0.0010
Epoch 47/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2635 - sparse_categorical_accuracy: 0.8904 - val_loss: 0.2659 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 0.0010
Epoch 48/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2580 - sparse_categorical_accuracy: 0.8835 - val_loss: 0.6046 - val_sparse_categorical_accuracy: 0.6990 - learning_rate: 0.0010
Epoch 49/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2561 - sparse_categorical_accuracy: 0.8969 - val_loss: 0.2286 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 50/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2458 - sparse_categorical_accuracy: 0.8969 - val_loss: 0.3737 - val_sparse_categorical_accuracy: 0.7989 - learning_rate: 0.0010
Epoch 51/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2545 - sparse_categorical_accuracy: 0.9022 - val_loss: 0.5977 - val_sparse_categorical_accuracy: 0.7157 - learning_rate: 0.0010
Epoch 52/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2650 - sparse_categorical_accuracy: 0.8805 - val_loss: 0.2496 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 53/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2593 - sparse_categorical_accuracy: 0.8917 - val_loss: 0.2122 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 0.0010
Epoch 54/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2539 - sparse_categorical_accuracy: 0.8909 - val_loss: 0.2366 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
Epoch 55/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2497 - sparse_categorical_accuracy: 0.9029 - val_loss: 0.2901 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 56/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2375 - sparse_categorical_accuracy: 0.9063 - val_loss: 0.2572 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
Epoch 57/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2458 - sparse_categorical_accuracy: 0.9079 - val_loss: 0.2223 - val_sparse_categorical_accuracy: 0.9126 - learning_rate: 0.0010
Epoch 58/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2392 - sparse_categorical_accuracy: 0.8965 - val_loss: 0.2703 - val_sparse_categorical_accuracy: 0.8793 - learning_rate: 0.0010
Epoch 59/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2404 - sparse_categorical_accuracy: 0.9030 - val_loss: 0.2322 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
Epoch 60/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2324 - sparse_categorical_accuracy: 0.9098 - val_loss: 0.5351 - val_sparse_categorical_accuracy: 0.7476 - learning_rate: 0.0010
Epoch 61/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2400 - sparse_categorical_accuracy: 0.8974 - val_loss: 0.8388 - val_sparse_categorical_accuracy: 0.6186 - learning_rate: 0.0010
Epoch 62/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2229 - sparse_categorical_accuracy: 0.9110 - val_loss: 0.3339 - val_sparse_categorical_accuracy: 0.8350 - learning_rate: 0.0010
Epoch 63/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2572 - sparse_categorical_accuracy: 0.8836 - val_loss: 0.2189 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 64/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2466 - sparse_categorical_accuracy: 0.8975 - val_loss: 0.3449 - val_sparse_categorical_accuracy: 0.8322 - learning_rate: 0.0010
Epoch 65/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2214 - sparse_categorical_accuracy: 0.9102 - val_loss: 0.2581 - val_sparse_categorical_accuracy: 0.8918 - learning_rate: 0.0010
Epoch 66/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2225 - sparse_categorical_accuracy: 0.8995 - val_loss: 0.5282 - val_sparse_categorical_accuracy: 0.7365 - learning_rate: 0.0010
Epoch 67/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2399 - sparse_categorical_accuracy: 0.9066 - val_loss: 0.2102 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 68/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2317 - sparse_categorical_accuracy: 0.8988 - val_loss: 0.2531 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 0.0010
Epoch 69/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2421 - sparse_categorical_accuracy: 0.8861 - val_loss: 0.2033 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
Epoch 70/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2520 - sparse_categorical_accuracy: 0.8981 - val_loss: 0.2841 - val_sparse_categorical_accuracy: 0.8613 - learning_rate: 0.0010
Epoch 71/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2307 - sparse_categorical_accuracy: 0.9067 - val_loss: 0.2289 - val_sparse_categorical_accuracy: 0.9140 - learning_rate: 0.0010
Epoch 72/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2316 - sparse_categorical_accuracy: 0.9112 - val_loss: 0.2042 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 73/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2409 - sparse_categorical_accuracy: 0.8951 - val_loss: 0.2155 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 74/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2327 - sparse_categorical_accuracy: 0.9046 - val_loss: 0.5060 - val_sparse_categorical_accuracy: 0.7725 - learning_rate: 0.0010
Epoch 75/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2139 - sparse_categorical_accuracy: 0.9169 - val_loss: 0.2029 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
Epoch 76/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2121 - sparse_categorical_accuracy: 0.9098 - val_loss: 0.2191 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
Epoch 77/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2260 - sparse_categorical_accuracy: 0.9071 - val_loss: 0.2682 - val_sparse_categorical_accuracy: 0.8752 - learning_rate: 0.0010
Epoch 78/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2334 - sparse_categorical_accuracy: 0.9022 - val_loss: 0.2683 - val_sparse_categorical_accuracy: 0.8710 - learning_rate: 0.0010
Epoch 79/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2137 - sparse_categorical_accuracy: 0.9062 - val_loss: 0.2375 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
Epoch 80/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2058 - sparse_categorical_accuracy: 0.9160 - val_loss: 0.2506 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
Epoch 81/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2272 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.2341 - val_sparse_categorical_accuracy: 0.8974 - learning_rate: 0.0010
Epoch 82/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2077 - sparse_categorical_accuracy: 0.9192 - val_loss: 0.2693 - val_sparse_categorical_accuracy: 0.8807 - learning_rate: 0.0010
Epoch 83/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2027 - sparse_categorical_accuracy: 0.9119 - val_loss: 0.2059 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 84/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2139 - sparse_categorical_accuracy: 0.9218 - val_loss: 0.2310 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 0.0010
Epoch 85/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2128 - sparse_categorical_accuracy: 0.9040 - val_loss: 0.2414 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 0.0010
Epoch 86/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2064 - sparse_categorical_accuracy: 0.9163 - val_loss: 0.2106 - val_sparse_categorical_accuracy: 0.9182 - learning_rate: 0.0010
Epoch 87/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2028 - sparse_categorical_accuracy: 0.9160 - val_loss: 0.5569 - val_sparse_categorical_accuracy: 0.7531 - learning_rate: 0.0010
Epoch 88/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2228 - sparse_categorical_accuracy: 0.9041 - val_loss: 0.2497 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 89/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2104 - sparse_categorical_accuracy: 0.9188 - val_loss: 0.1816 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
Epoch 90/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1876 - sparse_categorical_accuracy: 0.9284 - val_loss: 0.3861 - val_sparse_categorical_accuracy: 0.8197 - learning_rate: 0.0010
Epoch 91/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2272 - sparse_categorical_accuracy: 0.9098 - val_loss: 0.2145 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 0.0010
Epoch 92/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1956 - sparse_categorical_accuracy: 0.9277 - val_loss: 0.3385 - val_sparse_categorical_accuracy: 0.8488 - learning_rate: 0.0010
Epoch 93/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2030 - sparse_categorical_accuracy: 0.9210 - val_loss: 0.2766 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
Epoch 94/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2457 - sparse_categorical_accuracy: 0.8945 - val_loss: 0.2112 - val_sparse_categorical_accuracy: 0.9182 - learning_rate: 0.0010
Epoch 95/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2007 - sparse_categorical_accuracy: 0.9269 - val_loss: 0.5014 - val_sparse_categorical_accuracy: 0.7822 - learning_rate: 0.0010
Epoch 96/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1953 - sparse_categorical_accuracy: 0.9230 - val_loss: 0.4796 - val_sparse_categorical_accuracy: 0.7961 - learning_rate: 0.0010
Epoch 97/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2123 - sparse_categorical_accuracy: 0.9126 - val_loss: 0.1730 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 0.0010
Epoch 98/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1913 - sparse_categorical_accuracy: 0.9181 - val_loss: 0.1821 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
Epoch 99/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1753 - sparse_categorical_accuracy: 0.9285 - val_loss: 0.1762 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
Epoch 100/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1924 - sparse_categorical_accuracy: 0.9171 - val_loss: 0.2866 - val_sparse_categorical_accuracy: 0.8502 - learning_rate: 0.0010
Epoch 101/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1887 - sparse_categorical_accuracy: 0.9299 - val_loss: 0.1520 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 0.0010
Epoch 102/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1719 - sparse_categorical_accuracy: 0.9360 - val_loss: 0.2198 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 103/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9105 - val_loss: 0.2186 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 104/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1753 - sparse_categorical_accuracy: 0.9332 - val_loss: 0.2836 - val_sparse_categorical_accuracy: 0.8724 - learning_rate: 0.0010
Epoch 105/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1596 - sparse_categorical_accuracy: 0.9462 - val_loss: 0.1551 - val_sparse_categorical_accuracy: 0.9348 - learning_rate: 0.0010
Epoch 106/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1524 - sparse_categorical_accuracy: 0.9493 - val_loss: 0.2069 - val_sparse_categorical_accuracy: 0.9112 - learning_rate: 0.0010
Epoch 107/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1307 - sparse_categorical_accuracy: 0.9616 - val_loss: 0.1257 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 0.0010
Epoch 108/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1478 - sparse_categorical_accuracy: 0.9465 - val_loss: 0.4959 - val_sparse_categorical_accuracy: 0.7725 - learning_rate: 0.0010
Epoch 109/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1353 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.1916 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 0.0010
Epoch 110/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1393 - sparse_categorical_accuracy: 0.9571 - val_loss: 0.1569 - val_sparse_categorical_accuracy: 0.9293 - learning_rate: 0.0010
Epoch 111/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1267 - sparse_categorical_accuracy: 0.9613 - val_loss: 0.1288 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
Epoch 112/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1658 - sparse_categorical_accuracy: 0.9409 - val_loss: 0.3085 - val_sparse_categorical_accuracy: 0.8571 - learning_rate: 0.0010
Epoch 113/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1319 - sparse_categorical_accuracy: 0.9536 - val_loss: 0.1407 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 0.0010
Epoch 114/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1318 - sparse_categorical_accuracy: 0.9553 - val_loss: 0.1275 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 0.0010
Epoch 115/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1331 - sparse_categorical_accuracy: 0.9545 - val_loss: 1.2944 - val_sparse_categorical_accuracy: 0.5825 - learning_rate: 0.0010
Epoch 116/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1485 - sparse_categorical_accuracy: 0.9448 - val_loss: 0.2111 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 117/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1312 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.1022 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 0.0010
Epoch 118/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1223 - sparse_categorical_accuracy: 0.9617 - val_loss: 0.1538 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 0.0010
Epoch 119/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1346 - sparse_categorical_accuracy: 0.9596 - val_loss: 0.1269 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 0.0010
Epoch 120/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1262 - sparse_categorical_accuracy: 0.9561 - val_loss: 0.1026 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 0.0010
Epoch 121/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1258 - sparse_categorical_accuracy: 0.9500 - val_loss: 0.1482 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 0.0010
Epoch 122/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1169 - sparse_categorical_accuracy: 0.9588 - val_loss: 0.1103 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 0.0010
Epoch 123/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1343 - sparse_categorical_accuracy: 0.9498 - val_loss: 0.1884 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
Epoch 124/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1221 - sparse_categorical_accuracy: 0.9543 - val_loss: 0.1024 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 0.0010
Epoch 125/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1129 - sparse_categorical_accuracy: 0.9641 - val_loss: 0.0939 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 0.0010
Epoch 126/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1041 - sparse_categorical_accuracy: 0.9683 - val_loss: 0.3132 - val_sparse_categorical_accuracy: 0.8544 - learning_rate: 0.0010
Epoch 127/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1175 - sparse_categorical_accuracy: 0.9624 - val_loss: 0.1463 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 0.0010
Epoch 128/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1047 - sparse_categorical_accuracy: 0.9664 - val_loss: 0.2700 - val_sparse_categorical_accuracy: 0.8752 - learning_rate: 0.0010
Epoch 129/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0973 - sparse_categorical_accuracy: 0.9664 - val_loss: 0.1738 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 0.0010
Epoch 130/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1125 - sparse_categorical_accuracy: 0.9603 - val_loss: 0.1088 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 0.0010
Epoch 131/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1021 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.1881 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
Epoch 132/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1101 - sparse_categorical_accuracy: 0.9658 - val_loss: 0.1243 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 0.0010
Epoch 133/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0985 - sparse_categorical_accuracy: 0.9672 - val_loss: 0.1710 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
Epoch 134/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1150 - sparse_categorical_accuracy: 0.9657 - val_loss: 0.1725 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 0.0010
Epoch 135/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9652 - val_loss: 0.1589 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 0.0010
Epoch 136/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1021 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.1002 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 0.0010
Epoch 137/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1067 - sparse_categorical_accuracy: 0.9646 - val_loss: 0.1732 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 0.0010
Epoch 138/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1057 - sparse_categorical_accuracy: 0.9648 - val_loss: 0.2940 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 0.0010
Epoch 139/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1119 - sparse_categorical_accuracy: 0.9584 - val_loss: 0.1457 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 0.0010
Epoch 140/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1132 - sparse_categorical_accuracy: 0.9614 - val_loss: 0.4335 - val_sparse_categorical_accuracy: 0.8044 - learning_rate: 0.0010
Epoch 141/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1028 - sparse_categorical_accuracy: 0.9637 - val_loss: 0.1623 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
Epoch 142/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.1810 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 0.0010
Epoch 143/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0924 - sparse_categorical_accuracy: 0.9685 - val_loss: 0.1296 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 0.0010
Epoch 144/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0978 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.1835 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 0.0010
Epoch 145/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1095 - sparse_categorical_accuracy: 0.9609 - val_loss: 0.2358 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 146/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0909 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 147/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0992 - sparse_categorical_accuracy: 0.9695 - val_loss: 0.0941 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 148/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1019 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1508 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 5.0000e-04
Epoch 149/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0844 - sparse_categorical_accuracy: 0.9717 - val_loss: 0.1021 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 150/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0886 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1110 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 151/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0802 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0859 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 152/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.2167 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 5.0000e-04
Epoch 153/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0920 - sparse_categorical_accuracy: 0.9654 - val_loss: 0.1498 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 5.0000e-04
Epoch 154/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0840 - sparse_categorical_accuracy: 0.9717 - val_loss: 0.1077 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 155/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1000 - sparse_categorical_accuracy: 0.9689 - val_loss: 0.1054 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 156/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0939 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.1419 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 5.0000e-04
Epoch 157/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0889 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 158/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0959 - sparse_categorical_accuracy: 0.9657 - val_loss: 0.1807 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 5.0000e-04
Epoch 159/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0880 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0992 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 160/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0905 - sparse_categorical_accuracy: 0.9709 - val_loss: 0.1104 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 161/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0948 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.1065 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 162/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0865 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.1030 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 163/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0856 - sparse_categorical_accuracy: 0.9721 - val_loss: 0.1128 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 164/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0867 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0889 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 5.0000e-04
Epoch 165/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9682 - val_loss: 0.1201 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
Epoch 166/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0846 - sparse_categorical_accuracy: 0.9713 - val_loss: 0.1428 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 5.0000e-04
Epoch 167/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0912 - sparse_categorical_accuracy: 0.9683 - val_loss: 0.1105 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 168/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0857 - sparse_categorical_accuracy: 0.9711 - val_loss: 0.1075 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 169/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0762 - sparse_categorical_accuracy: 0.9737 - val_loss: 0.0874 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 170/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0959 - sparse_categorical_accuracy: 0.9734 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 171/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0885 - sparse_categorical_accuracy: 0.9697 - val_loss: 0.0951 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 172/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0804 - sparse_categorical_accuracy: 0.9714 - val_loss: 0.1106 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 173/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0832 - sparse_categorical_accuracy: 0.9731 - val_loss: 0.2379 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 2.5000e-04
Epoch 174/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.1567 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 2.5000e-04
Epoch 175/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0808 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0994 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 176/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0765 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0895 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 177/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0850 - sparse_categorical_accuracy: 0.9775 - val_loss: 0.1154 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
Epoch 178/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0759 - sparse_categorical_accuracy: 0.9763 - val_loss: 0.0885 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 179/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0750 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.1018 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 180/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 181/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0742 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0964 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 182/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9727 - val_loss: 0.0932 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 183/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0849 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 2.5000e-04
Epoch 184/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0765 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0876 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 2.5000e-04
Epoch 185/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9725 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 186/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0865 - sparse_categorical_accuracy: 0.9650 - val_loss: 0.0872 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 187/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0794 - sparse_categorical_accuracy: 0.9769 - val_loss: 0.0992 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 188/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0691 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.0959 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 189/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0767 - sparse_categorical_accuracy: 0.9776 - val_loss: 0.0927 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 190/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0665 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0853 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 191/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0832 - sparse_categorical_accuracy: 0.9699 - val_loss: 0.0862 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 192/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1030 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
Epoch 193/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0732 - sparse_categorical_accuracy: 0.9761 - val_loss: 0.0942 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 194/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0784 - sparse_categorical_accuracy: 0.9743 - val_loss: 0.1023 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 195/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.0929 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 196/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0760 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0973 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 197/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0923 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 198/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0713 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 199/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0890 - sparse_categorical_accuracy: 0.9689 - val_loss: 0.0845 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 200/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9713 - val_loss: 0.0914 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 201/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0726 - sparse_categorical_accuracy: 0.9800 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 202/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0834 - sparse_categorical_accuracy: 0.9707 - val_loss: 0.0872 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 203/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9784 - val_loss: 0.0920 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 204/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0891 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 205/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0725 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 206/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0851 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 207/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.0862 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 208/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0809 - sparse_categorical_accuracy: 0.9721 - val_loss: 0.0848 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 2.5000e-04
Epoch 209/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0760 - sparse_categorical_accuracy: 0.9735 - val_loss: 0.0870 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 210/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0846 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 211/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0648 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.1144 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 212/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9770 - val_loss: 0.1156 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 2.5000e-04
Epoch 213/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9793 - val_loss: 0.0958 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 214/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0889 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 215/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0717 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 216/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0685 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0839 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 217/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0776 - sparse_categorical_accuracy: 0.9699 - val_loss: 0.1052 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
Epoch 218/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9731 - val_loss: 0.1314 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 2.5000e-04
Epoch 219/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.1113 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
Epoch 220/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9763 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 2.5000e-04
Epoch 221/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0740 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.0994 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 222/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.1023 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 223/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9770 - val_loss: 0.1172 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
Epoch 224/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0785 - sparse_categorical_accuracy: 0.9665 - val_loss: 0.0946 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 225/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0724 - sparse_categorical_accuracy: 0.9763 - val_loss: 0.0922 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 226/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0653 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.1050 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 227/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0684 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.1491 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
Epoch 228/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0637 - sparse_categorical_accuracy: 0.9772 - val_loss: 0.0905 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 229/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0678 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0971 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 230/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0719 - sparse_categorical_accuracy: 0.9758 - val_loss: 0.0832 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 2.5000e-04
Epoch 231/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0687 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0871 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 232/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0810 - sparse_categorical_accuracy: 0.9744 - val_loss: 0.0902 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 233/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0794 - sparse_categorical_accuracy: 0.9732 - val_loss: 0.1202 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 2.5000e-04
Epoch 234/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0741 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0857 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 235/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0678 - sparse_categorical_accuracy: 0.9810 - val_loss: 0.0875 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 236/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0741 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 237/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0660 - sparse_categorical_accuracy: 0.9804 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 238/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0597 - sparse_categorical_accuracy: 0.9808 - val_loss: 0.0852 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 239/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0851 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 240/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.0934 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 241/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0892 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 242/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0817 - sparse_categorical_accuracy: 0.9737 - val_loss: 0.0943 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 243/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9783 - val_loss: 0.0830 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 2.5000e-04
Epoch 244/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0841 - sparse_categorical_accuracy: 0.9714 - val_loss: 0.0969 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 245/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0809 - sparse_categorical_accuracy: 0.9706 - val_loss: 0.0919 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 246/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0705 - sparse_categorical_accuracy: 0.9704 - val_loss: 0.0880 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 247/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9790 - val_loss: 0.0951 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 248/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0666 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.0833 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 249/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0637 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.0885 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 250/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0645 - sparse_categorical_accuracy: 0.9803 - val_loss: 0.1264 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 2.5000e-04
Epoch 251/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0629 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 252/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0790 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 253/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0663 - sparse_categorical_accuracy: 0.9805 - val_loss: 0.1172 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 254/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9757 - val_loss: 0.0980 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 255/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0684 - sparse_categorical_accuracy: 0.9779 - val_loss: 0.0872 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 256/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.2256 - val_sparse_categorical_accuracy: 0.9126 - learning_rate: 2.5000e-04
Epoch 257/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0583 - sparse_categorical_accuracy: 0.9800 - val_loss: 0.0860 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 258/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0739 - sparse_categorical_accuracy: 0.9745 - val_loss: 0.1226 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 2.5000e-04
Epoch 259/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0771 - sparse_categorical_accuracy: 0.9745 - val_loss: 0.0861 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 260/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0657 - sparse_categorical_accuracy: 0.9811 - val_loss: 0.1063 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 2.5000e-04
Epoch 261/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9751 - val_loss: 0.1429 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 2.5000e-04
Epoch 262/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9810 - val_loss: 0.1071 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 2.5000e-04
Epoch 263/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0542 - sparse_categorical_accuracy: 0.9867 - val_loss: 0.0861 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 264/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9790 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 265/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.1116 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 1.2500e-04
Epoch 266/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0698 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0811 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 267/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0549 - sparse_categorical_accuracy: 0.9865 - val_loss: 0.0920 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 268/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.0880 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
Epoch 269/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9759 - val_loss: 0.0897 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 270/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0725 - sparse_categorical_accuracy: 0.9711 - val_loss: 0.0825 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 271/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0666 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0853 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 272/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0632 - sparse_categorical_accuracy: 0.9825 - val_loss: 0.0832 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
Epoch 273/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0664 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0879 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 274/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0600 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 275/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0586 - sparse_categorical_accuracy: 0.9835 - val_loss: 0.0827 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 276/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0631 - sparse_categorical_accuracy: 0.9825 - val_loss: 0.1040 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.2500e-04
Epoch 277/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0659 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.2500e-04
Epoch 278/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9704 - val_loss: 0.0860 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 279/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0568 - sparse_categorical_accuracy: 0.9806 - val_loss: 0.1007 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
Epoch 280/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0628 - sparse_categorical_accuracy: 0.9829 - val_loss: 0.0892 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 281/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0697 - sparse_categorical_accuracy: 0.9765 - val_loss: 0.0823 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.2500e-04
Epoch 282/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0707 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0848 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.2500e-04
Epoch 283/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0621 - sparse_categorical_accuracy: 0.9786 - val_loss: 0.0855 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 284/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0603 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0814 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 285/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9820 - val_loss: 0.0836 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 286/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0605 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1126 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.2500e-04
Epoch 287/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0546 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 288/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0631 - sparse_categorical_accuracy: 0.9817 - val_loss: 0.0879 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 289/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0621 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.1059 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 1.0000e-04
Epoch 290/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0577 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0878 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 291/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0588 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0826 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 292/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0607 - sparse_categorical_accuracy: 0.9817 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 293/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0595 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.1114 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 1.0000e-04
Epoch 294/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0676 - sparse_categorical_accuracy: 0.9780 - val_loss: 0.0842 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 295/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0595 - sparse_categorical_accuracy: 0.9811 - val_loss: 0.0839 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 296/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9791 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 297/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0586 - sparse_categorical_accuracy: 0.9810 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 298/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0565 - sparse_categorical_accuracy: 0.9839 - val_loss: 0.0952 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 299/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0655 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.0825 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 300/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0626 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.0855 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 301/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.0827 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 302/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0693 - sparse_categorical_accuracy: 0.9768 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 303/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0606 - sparse_categorical_accuracy: 0.9778 - val_loss: 0.0960 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 304/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.0831 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 305/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9780 - val_loss: 0.0999 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
Epoch 306/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0667 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.0866 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 307/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0662 - sparse_categorical_accuracy: 0.9790 - val_loss: 0.1042 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
Epoch 308/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0603 - sparse_categorical_accuracy: 0.9821 - val_loss: 0.0813 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 309/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0576 - sparse_categorical_accuracy: 0.9815 - val_loss: 0.0986 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
Epoch 310/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0594 - sparse_categorical_accuracy: 0.9821 - val_loss: 0.0817 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 311/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0591 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0838 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 312/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0632 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.0830 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 313/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0618 - sparse_categorical_accuracy: 0.9779 - val_loss: 0.1093 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
Epoch 314/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0786 - sparse_categorical_accuracy: 0.9737 - val_loss: 0.0824 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 315/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0560 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.0901 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 316/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0509 - sparse_categorical_accuracy: 0.9856 - val_loss: 0.0824 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 316: early stopping

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
 42/42 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - loss: 0.0888 - sparse_categorical_accuracy: 0.9719
Test accuracy 0.9704545736312866
Test loss 0.0922362208366394

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
