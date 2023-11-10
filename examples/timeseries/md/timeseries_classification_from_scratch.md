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
You must install pydot (`pip install pydot`) for `plot_model` to work.

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
 90/90 ━━━━━━━━━━━━━━━━━━━━ 5s 33ms/step - loss: 0.6176 - sparse_categorical_accuracy: 0.6534 - val_loss: 0.7964 - val_sparse_categorical_accuracy: 0.5173 - learning_rate: 0.0010
Epoch 2/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.5158 - sparse_categorical_accuracy: 0.7391 - val_loss: 0.8469 - val_sparse_categorical_accuracy: 0.5173 - learning_rate: 0.0010
Epoch 3/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4653 - sparse_categorical_accuracy: 0.7727 - val_loss: 0.8551 - val_sparse_categorical_accuracy: 0.5173 - learning_rate: 0.0010
Epoch 4/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4519 - sparse_categorical_accuracy: 0.7731 - val_loss: 0.8786 - val_sparse_categorical_accuracy: 0.5173 - learning_rate: 0.0010
Epoch 5/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4703 - sparse_categorical_accuracy: 0.7560 - val_loss: 0.6655 - val_sparse_categorical_accuracy: 0.5312 - learning_rate: 0.0010
Epoch 6/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4180 - sparse_categorical_accuracy: 0.8027 - val_loss: 0.5685 - val_sparse_categorical_accuracy: 0.6935 - learning_rate: 0.0010
Epoch 7/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4338 - sparse_categorical_accuracy: 0.7713 - val_loss: 0.6558 - val_sparse_categorical_accuracy: 0.5825 - learning_rate: 0.0010
Epoch 8/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4083 - sparse_categorical_accuracy: 0.8105 - val_loss: 0.5433 - val_sparse_categorical_accuracy: 0.7060 - learning_rate: 0.0010
Epoch 9/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4196 - sparse_categorical_accuracy: 0.7781 - val_loss: 0.4088 - val_sparse_categorical_accuracy: 0.8072 - learning_rate: 0.0010
Epoch 10/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4303 - sparse_categorical_accuracy: 0.7941 - val_loss: 0.9413 - val_sparse_categorical_accuracy: 0.6796 - learning_rate: 0.0010
Epoch 11/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4029 - sparse_categorical_accuracy: 0.7968 - val_loss: 0.4785 - val_sparse_categorical_accuracy: 0.7309 - learning_rate: 0.0010
Epoch 12/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4209 - sparse_categorical_accuracy: 0.7860 - val_loss: 0.7936 - val_sparse_categorical_accuracy: 0.6976 - learning_rate: 0.0010
Epoch 13/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4047 - sparse_categorical_accuracy: 0.7978 - val_loss: 0.5113 - val_sparse_categorical_accuracy: 0.7462 - learning_rate: 0.0010
Epoch 14/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4150 - sparse_categorical_accuracy: 0.7768 - val_loss: 0.3888 - val_sparse_categorical_accuracy: 0.7933 - learning_rate: 0.0010
Epoch 15/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4132 - sparse_categorical_accuracy: 0.7991 - val_loss: 0.4027 - val_sparse_categorical_accuracy: 0.7850 - learning_rate: 0.0010
Epoch 16/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3867 - sparse_categorical_accuracy: 0.8110 - val_loss: 0.4061 - val_sparse_categorical_accuracy: 0.8100 - learning_rate: 0.0010
Epoch 17/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3964 - sparse_categorical_accuracy: 0.7954 - val_loss: 0.6036 - val_sparse_categorical_accuracy: 0.6990 - learning_rate: 0.0010
Epoch 18/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3681 - sparse_categorical_accuracy: 0.8191 - val_loss: 0.6130 - val_sparse_categorical_accuracy: 0.6921 - learning_rate: 0.0010
Epoch 19/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3855 - sparse_categorical_accuracy: 0.8158 - val_loss: 1.9731 - val_sparse_categorical_accuracy: 0.4827 - learning_rate: 0.0010
Epoch 20/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3603 - sparse_categorical_accuracy: 0.8296 - val_loss: 1.0285 - val_sparse_categorical_accuracy: 0.5964 - learning_rate: 0.0010
Epoch 21/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3562 - sparse_categorical_accuracy: 0.8372 - val_loss: 0.3432 - val_sparse_categorical_accuracy: 0.8294 - learning_rate: 0.0010
Epoch 22/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3604 - sparse_categorical_accuracy: 0.8415 - val_loss: 0.4124 - val_sparse_categorical_accuracy: 0.7420 - learning_rate: 0.0010
Epoch 23/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3595 - sparse_categorical_accuracy: 0.8323 - val_loss: 0.4349 - val_sparse_categorical_accuracy: 0.7864 - learning_rate: 0.0010
Epoch 24/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3454 - sparse_categorical_accuracy: 0.8399 - val_loss: 0.4272 - val_sparse_categorical_accuracy: 0.7573 - learning_rate: 0.0010
Epoch 25/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3380 - sparse_categorical_accuracy: 0.8520 - val_loss: 0.4990 - val_sparse_categorical_accuracy: 0.7351 - learning_rate: 0.0010
Epoch 26/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3716 - sparse_categorical_accuracy: 0.8227 - val_loss: 0.4124 - val_sparse_categorical_accuracy: 0.7933 - learning_rate: 0.0010
Epoch 27/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3604 - sparse_categorical_accuracy: 0.8182 - val_loss: 0.6465 - val_sparse_categorical_accuracy: 0.7434 - learning_rate: 0.0010
Epoch 28/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3657 - sparse_categorical_accuracy: 0.8150 - val_loss: 0.5638 - val_sparse_categorical_accuracy: 0.6976 - learning_rate: 0.0010
Epoch 29/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3344 - sparse_categorical_accuracy: 0.8520 - val_loss: 0.4443 - val_sparse_categorical_accuracy: 0.7531 - learning_rate: 0.0010
Epoch 30/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3197 - sparse_categorical_accuracy: 0.8562 - val_loss: 0.3326 - val_sparse_categorical_accuracy: 0.8363 - learning_rate: 0.0010
Epoch 31/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3166 - sparse_categorical_accuracy: 0.8679 - val_loss: 0.3151 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 0.0010
Epoch 32/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3131 - sparse_categorical_accuracy: 0.8714 - val_loss: 0.3041 - val_sparse_categorical_accuracy: 0.8710 - learning_rate: 0.0010
Epoch 33/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3188 - sparse_categorical_accuracy: 0.8718 - val_loss: 0.2921 - val_sparse_categorical_accuracy: 0.8835 - learning_rate: 0.0010
Epoch 34/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3055 - sparse_categorical_accuracy: 0.8664 - val_loss: 0.4658 - val_sparse_categorical_accuracy: 0.7587 - learning_rate: 0.0010
Epoch 35/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2968 - sparse_categorical_accuracy: 0.8695 - val_loss: 0.2949 - val_sparse_categorical_accuracy: 0.8516 - learning_rate: 0.0010
Epoch 36/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3095 - sparse_categorical_accuracy: 0.8647 - val_loss: 0.2920 - val_sparse_categorical_accuracy: 0.8544 - learning_rate: 0.0010
Epoch 37/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3047 - sparse_categorical_accuracy: 0.8720 - val_loss: 1.0986 - val_sparse_categorical_accuracy: 0.6311 - learning_rate: 0.0010
Epoch 38/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3022 - sparse_categorical_accuracy: 0.8723 - val_loss: 0.5850 - val_sparse_categorical_accuracy: 0.7212 - learning_rate: 0.0010
Epoch 39/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2929 - sparse_categorical_accuracy: 0.8861 - val_loss: 0.8902 - val_sparse_categorical_accuracy: 0.6158 - learning_rate: 0.0010
Epoch 40/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3063 - sparse_categorical_accuracy: 0.8534 - val_loss: 0.2870 - val_sparse_categorical_accuracy: 0.8779 - learning_rate: 0.0010
Epoch 41/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3002 - sparse_categorical_accuracy: 0.8635 - val_loss: 0.3584 - val_sparse_categorical_accuracy: 0.8031 - learning_rate: 0.0010
Epoch 42/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2625 - sparse_categorical_accuracy: 0.8951 - val_loss: 0.6331 - val_sparse_categorical_accuracy: 0.6463 - learning_rate: 0.0010
Epoch 43/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2629 - sparse_categorical_accuracy: 0.8930 - val_loss: 0.2613 - val_sparse_categorical_accuracy: 0.8932 - learning_rate: 0.0010
Epoch 44/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2503 - sparse_categorical_accuracy: 0.8988 - val_loss: 0.2485 - val_sparse_categorical_accuracy: 0.9015 - learning_rate: 0.0010
Epoch 45/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2624 - sparse_categorical_accuracy: 0.8898 - val_loss: 0.2502 - val_sparse_categorical_accuracy: 0.8918 - learning_rate: 0.0010
Epoch 46/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2777 - sparse_categorical_accuracy: 0.8903 - val_loss: 0.2760 - val_sparse_categorical_accuracy: 0.8585 - learning_rate: 0.0010
Epoch 47/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2617 - sparse_categorical_accuracy: 0.8901 - val_loss: 0.2813 - val_sparse_categorical_accuracy: 0.8821 - learning_rate: 0.0010
Epoch 48/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2693 - sparse_categorical_accuracy: 0.8938 - val_loss: 0.8611 - val_sparse_categorical_accuracy: 0.7462 - learning_rate: 0.0010
Epoch 49/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2715 - sparse_categorical_accuracy: 0.8947 - val_loss: 0.3997 - val_sparse_categorical_accuracy: 0.7947 - learning_rate: 0.0010
Epoch 50/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2668 - sparse_categorical_accuracy: 0.8944 - val_loss: 0.2784 - val_sparse_categorical_accuracy: 0.8738 - learning_rate: 0.0010
Epoch 51/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2617 - sparse_categorical_accuracy: 0.8941 - val_loss: 0.2371 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 0.0010
Epoch 52/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2483 - sparse_categorical_accuracy: 0.9062 - val_loss: 0.2417 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 0.0010
Epoch 53/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2644 - sparse_categorical_accuracy: 0.8805 - val_loss: 0.3824 - val_sparse_categorical_accuracy: 0.8086 - learning_rate: 0.0010
Epoch 54/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2583 - sparse_categorical_accuracy: 0.8960 - val_loss: 0.2364 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 55/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2514 - sparse_categorical_accuracy: 0.8923 - val_loss: 0.4527 - val_sparse_categorical_accuracy: 0.7614 - learning_rate: 0.0010
Epoch 56/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2526 - sparse_categorical_accuracy: 0.8978 - val_loss: 0.3718 - val_sparse_categorical_accuracy: 0.8086 - learning_rate: 0.0010
Epoch 57/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2458 - sparse_categorical_accuracy: 0.8951 - val_loss: 0.3448 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 0.0010
Epoch 58/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2391 - sparse_categorical_accuracy: 0.9110 - val_loss: 0.6488 - val_sparse_categorical_accuracy: 0.7032 - learning_rate: 0.0010
Epoch 59/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2450 - sparse_categorical_accuracy: 0.8970 - val_loss: 0.2307 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 60/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2301 - sparse_categorical_accuracy: 0.9033 - val_loss: 0.6413 - val_sparse_categorical_accuracy: 0.7323 - learning_rate: 0.0010
Epoch 61/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2448 - sparse_categorical_accuracy: 0.9043 - val_loss: 0.2325 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 0.0010
Epoch 62/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2610 - sparse_categorical_accuracy: 0.8945 - val_loss: 0.2570 - val_sparse_categorical_accuracy: 0.8890 - learning_rate: 0.0010
Epoch 63/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2476 - sparse_categorical_accuracy: 0.9037 - val_loss: 0.2181 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
Epoch 64/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2367 - sparse_categorical_accuracy: 0.9006 - val_loss: 0.2180 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 0.0010
Epoch 65/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2455 - sparse_categorical_accuracy: 0.9018 - val_loss: 0.2214 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 0.0010
Epoch 66/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2629 - sparse_categorical_accuracy: 0.8956 - val_loss: 0.3576 - val_sparse_categorical_accuracy: 0.8294 - learning_rate: 0.0010
Epoch 67/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2555 - sparse_categorical_accuracy: 0.8964 - val_loss: 0.2108 - val_sparse_categorical_accuracy: 0.9140 - learning_rate: 0.0010
Epoch 68/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2339 - sparse_categorical_accuracy: 0.9037 - val_loss: 0.2117 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
Epoch 69/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2476 - sparse_categorical_accuracy: 0.8969 - val_loss: 0.2150 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
Epoch 70/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2465 - sparse_categorical_accuracy: 0.8970 - val_loss: 0.2635 - val_sparse_categorical_accuracy: 0.8863 - learning_rate: 0.0010
Epoch 71/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2244 - sparse_categorical_accuracy: 0.9139 - val_loss: 0.2375 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 72/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2172 - sparse_categorical_accuracy: 0.9183 - val_loss: 0.8133 - val_sparse_categorical_accuracy: 0.6546 - learning_rate: 0.0010
Epoch 73/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2472 - sparse_categorical_accuracy: 0.8938 - val_loss: 0.3039 - val_sparse_categorical_accuracy: 0.8599 - learning_rate: 0.0010
Epoch 74/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2284 - sparse_categorical_accuracy: 0.9114 - val_loss: 0.2198 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 0.0010
Epoch 75/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2256 - sparse_categorical_accuracy: 0.9063 - val_loss: 0.2236 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 76/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2322 - sparse_categorical_accuracy: 0.9032 - val_loss: 0.2525 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
Epoch 77/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2411 - sparse_categorical_accuracy: 0.9039 - val_loss: 0.4705 - val_sparse_categorical_accuracy: 0.7628 - learning_rate: 0.0010
Epoch 78/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2463 - sparse_categorical_accuracy: 0.8915 - val_loss: 0.2148 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 0.0010
Epoch 79/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2294 - sparse_categorical_accuracy: 0.9060 - val_loss: 0.2366 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 0.0010
Epoch 80/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2198 - sparse_categorical_accuracy: 0.9102 - val_loss: 0.2928 - val_sparse_categorical_accuracy: 0.8669 - learning_rate: 0.0010
Epoch 81/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2339 - sparse_categorical_accuracy: 0.9042 - val_loss: 0.2230 - val_sparse_categorical_accuracy: 0.9043 - learning_rate: 0.0010
Epoch 82/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2459 - sparse_categorical_accuracy: 0.9082 - val_loss: 0.2576 - val_sparse_categorical_accuracy: 0.8960 - learning_rate: 0.0010
Epoch 83/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2258 - sparse_categorical_accuracy: 0.9055 - val_loss: 0.2126 - val_sparse_categorical_accuracy: 0.9209 - learning_rate: 0.0010
Epoch 84/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2186 - sparse_categorical_accuracy: 0.9178 - val_loss: 0.2620 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 85/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2242 - sparse_categorical_accuracy: 0.9096 - val_loss: 0.5812 - val_sparse_categorical_accuracy: 0.7268 - learning_rate: 0.0010
Epoch 86/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2371 - sparse_categorical_accuracy: 0.8980 - val_loss: 0.2489 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 0.0010
Epoch 87/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2144 - sparse_categorical_accuracy: 0.9150 - val_loss: 0.2117 - val_sparse_categorical_accuracy: 0.9071 - learning_rate: 0.0010
Epoch 88/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2086 - sparse_categorical_accuracy: 0.9202 - val_loss: 0.1916 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 5.0000e-04
Epoch 89/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2178 - sparse_categorical_accuracy: 0.9141 - val_loss: 0.1848 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 5.0000e-04
Epoch 90/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2072 - sparse_categorical_accuracy: 0.9223 - val_loss: 0.2023 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 5.0000e-04
Epoch 91/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2221 - sparse_categorical_accuracy: 0.9070 - val_loss: 0.2345 - val_sparse_categorical_accuracy: 0.9001 - learning_rate: 5.0000e-04
Epoch 92/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2040 - sparse_categorical_accuracy: 0.9207 - val_loss: 0.1975 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 5.0000e-04
Epoch 93/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2170 - sparse_categorical_accuracy: 0.9135 - val_loss: 0.1904 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 5.0000e-04
Epoch 94/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2028 - sparse_categorical_accuracy: 0.9271 - val_loss: 0.1865 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 5.0000e-04
Epoch 95/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2116 - sparse_categorical_accuracy: 0.9181 - val_loss: 0.1996 - val_sparse_categorical_accuracy: 0.9140 - learning_rate: 5.0000e-04
Epoch 96/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2077 - sparse_categorical_accuracy: 0.9255 - val_loss: 0.1920 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 5.0000e-04
Epoch 97/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2110 - sparse_categorical_accuracy: 0.9082 - val_loss: 0.2076 - val_sparse_categorical_accuracy: 0.9126 - learning_rate: 5.0000e-04
Epoch 98/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2017 - sparse_categorical_accuracy: 0.9267 - val_loss: 0.2646 - val_sparse_categorical_accuracy: 0.8904 - learning_rate: 5.0000e-04
Epoch 99/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1932 - sparse_categorical_accuracy: 0.9273 - val_loss: 0.1990 - val_sparse_categorical_accuracy: 0.9154 - learning_rate: 5.0000e-04
Epoch 100/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1982 - sparse_categorical_accuracy: 0.9232 - val_loss: 0.2803 - val_sparse_categorical_accuracy: 0.8682 - learning_rate: 5.0000e-04
Epoch 101/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1848 - sparse_categorical_accuracy: 0.9326 - val_loss: 0.1839 - val_sparse_categorical_accuracy: 0.9237 - learning_rate: 5.0000e-04
Epoch 102/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1954 - sparse_categorical_accuracy: 0.9219 - val_loss: 0.1783 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 5.0000e-04
Epoch 103/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2057 - sparse_categorical_accuracy: 0.9190 - val_loss: 0.2886 - val_sparse_categorical_accuracy: 0.8779 - learning_rate: 5.0000e-04
Epoch 104/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1906 - sparse_categorical_accuracy: 0.9274 - val_loss: 0.2424 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 5.0000e-04
Epoch 105/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2163 - sparse_categorical_accuracy: 0.9151 - val_loss: 0.2035 - val_sparse_categorical_accuracy: 0.9265 - learning_rate: 5.0000e-04
Epoch 106/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2119 - sparse_categorical_accuracy: 0.9132 - val_loss: 0.2000 - val_sparse_categorical_accuracy: 0.9085 - learning_rate: 5.0000e-04
Epoch 107/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2086 - sparse_categorical_accuracy: 0.9164 - val_loss: 0.2458 - val_sparse_categorical_accuracy: 0.8946 - learning_rate: 5.0000e-04
Epoch 108/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1900 - sparse_categorical_accuracy: 0.9253 - val_loss: 0.1817 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
Epoch 109/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2106 - sparse_categorical_accuracy: 0.9160 - val_loss: 0.1883 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 5.0000e-04
Epoch 110/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1945 - sparse_categorical_accuracy: 0.9307 - val_loss: 0.1912 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 5.0000e-04
Epoch 111/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1963 - sparse_categorical_accuracy: 0.9166 - val_loss: 0.1769 - val_sparse_categorical_accuracy: 0.9251 - learning_rate: 5.0000e-04
Epoch 112/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1984 - sparse_categorical_accuracy: 0.9233 - val_loss: 0.1688 - val_sparse_categorical_accuracy: 0.9348 - learning_rate: 5.0000e-04
Epoch 113/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1835 - sparse_categorical_accuracy: 0.9300 - val_loss: 0.1689 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 5.0000e-04
Epoch 114/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1785 - sparse_categorical_accuracy: 0.9334 - val_loss: 0.1955 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 5.0000e-04
Epoch 115/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1779 - sparse_categorical_accuracy: 0.9336 - val_loss: 0.3694 - val_sparse_categorical_accuracy: 0.8405 - learning_rate: 5.0000e-04
Epoch 116/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1923 - sparse_categorical_accuracy: 0.9239 - val_loss: 0.1687 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 5.0000e-04
Epoch 117/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1854 - sparse_categorical_accuracy: 0.9307 - val_loss: 0.1603 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 5.0000e-04
Epoch 118/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2090 - sparse_categorical_accuracy: 0.9220 - val_loss: 0.1787 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
Epoch 119/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1744 - sparse_categorical_accuracy: 0.9415 - val_loss: 0.3293 - val_sparse_categorical_accuracy: 0.8530 - learning_rate: 5.0000e-04
Epoch 120/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1765 - sparse_categorical_accuracy: 0.9377 - val_loss: 0.1663 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 5.0000e-04
Epoch 121/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1788 - sparse_categorical_accuracy: 0.9270 - val_loss: 0.1705 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 5.0000e-04
Epoch 122/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1741 - sparse_categorical_accuracy: 0.9333 - val_loss: 0.1726 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
Epoch 123/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1796 - sparse_categorical_accuracy: 0.9311 - val_loss: 0.1497 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 5.0000e-04
Epoch 124/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1679 - sparse_categorical_accuracy: 0.9399 - val_loss: 0.2153 - val_sparse_categorical_accuracy: 0.9223 - learning_rate: 5.0000e-04
Epoch 125/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1719 - sparse_categorical_accuracy: 0.9428 - val_loss: 0.2525 - val_sparse_categorical_accuracy: 0.8849 - learning_rate: 5.0000e-04
Epoch 126/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1710 - sparse_categorical_accuracy: 0.9327 - val_loss: 0.3206 - val_sparse_categorical_accuracy: 0.8474 - learning_rate: 5.0000e-04
Epoch 127/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1670 - sparse_categorical_accuracy: 0.9347 - val_loss: 0.1519 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 5.0000e-04
Epoch 128/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1752 - sparse_categorical_accuracy: 0.9365 - val_loss: 0.1608 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
Epoch 129/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1576 - sparse_categorical_accuracy: 0.9441 - val_loss: 0.1441 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 5.0000e-04
Epoch 130/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1604 - sparse_categorical_accuracy: 0.9441 - val_loss: 0.1389 - val_sparse_categorical_accuracy: 0.9487 - learning_rate: 5.0000e-04
Epoch 131/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1808 - sparse_categorical_accuracy: 0.9349 - val_loss: 0.2013 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 5.0000e-04
Epoch 132/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1552 - sparse_categorical_accuracy: 0.9411 - val_loss: 0.1448 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
Epoch 133/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1576 - sparse_categorical_accuracy: 0.9445 - val_loss: 0.1847 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 5.0000e-04
Epoch 134/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1557 - sparse_categorical_accuracy: 0.9432 - val_loss: 0.1514 - val_sparse_categorical_accuracy: 0.9404 - learning_rate: 5.0000e-04
Epoch 135/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1599 - sparse_categorical_accuracy: 0.9444 - val_loss: 0.2285 - val_sparse_categorical_accuracy: 0.9168 - learning_rate: 5.0000e-04
Epoch 136/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1427 - sparse_categorical_accuracy: 0.9528 - val_loss: 0.1800 - val_sparse_categorical_accuracy: 0.9279 - learning_rate: 5.0000e-04
Epoch 137/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1516 - sparse_categorical_accuracy: 0.9483 - val_loss: 0.2976 - val_sparse_categorical_accuracy: 0.8641 - learning_rate: 5.0000e-04
Epoch 138/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1404 - sparse_categorical_accuracy: 0.9441 - val_loss: 0.1188 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 5.0000e-04
Epoch 139/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1656 - sparse_categorical_accuracy: 0.9386 - val_loss: 0.1185 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 140/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1360 - sparse_categorical_accuracy: 0.9528 - val_loss: 0.2340 - val_sparse_categorical_accuracy: 0.8988 - learning_rate: 5.0000e-04
Epoch 141/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1384 - sparse_categorical_accuracy: 0.9488 - val_loss: 0.1114 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 142/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1307 - sparse_categorical_accuracy: 0.9590 - val_loss: 0.1165 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 143/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1303 - sparse_categorical_accuracy: 0.9561 - val_loss: 0.1155 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 144/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1296 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.1455 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 5.0000e-04
Epoch 145/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1348 - sparse_categorical_accuracy: 0.9519 - val_loss: 0.1485 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
Epoch 146/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1407 - sparse_categorical_accuracy: 0.9501 - val_loss: 0.1389 - val_sparse_categorical_accuracy: 0.9584 - learning_rate: 5.0000e-04
Epoch 147/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1436 - sparse_categorical_accuracy: 0.9486 - val_loss: 0.1176 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 148/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1240 - sparse_categorical_accuracy: 0.9594 - val_loss: 0.1218 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 149/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1377 - sparse_categorical_accuracy: 0.9556 - val_loss: 0.1008 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 150/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1198 - sparse_categorical_accuracy: 0.9569 - val_loss: 0.1440 - val_sparse_categorical_accuracy: 0.9417 - learning_rate: 5.0000e-04
Epoch 151/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1381 - sparse_categorical_accuracy: 0.9509 - val_loss: 0.1596 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
Epoch 152/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1286 - sparse_categorical_accuracy: 0.9607 - val_loss: 0.1255 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
Epoch 153/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1315 - sparse_categorical_accuracy: 0.9483 - val_loss: 0.1591 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 5.0000e-04
Epoch 154/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1157 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.1157 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 155/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1114 - sparse_categorical_accuracy: 0.9645 - val_loss: 0.1059 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 156/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1405 - sparse_categorical_accuracy: 0.9497 - val_loss: 0.1080 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 157/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1071 - sparse_categorical_accuracy: 0.9654 - val_loss: 0.2372 - val_sparse_categorical_accuracy: 0.9098 - learning_rate: 5.0000e-04
Epoch 158/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1372 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.0981 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 159/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1193 - sparse_categorical_accuracy: 0.9634 - val_loss: 0.1184 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 160/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1151 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.1054 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 161/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1215 - sparse_categorical_accuracy: 0.9620 - val_loss: 0.1875 - val_sparse_categorical_accuracy: 0.9126 - learning_rate: 5.0000e-04
Epoch 162/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1081 - sparse_categorical_accuracy: 0.9648 - val_loss: 0.1244 - val_sparse_categorical_accuracy: 0.9542 - learning_rate: 5.0000e-04
Epoch 163/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1184 - sparse_categorical_accuracy: 0.9583 - val_loss: 0.1180 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 164/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1093 - sparse_categorical_accuracy: 0.9636 - val_loss: 0.1701 - val_sparse_categorical_accuracy: 0.9307 - learning_rate: 5.0000e-04
Epoch 165/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1124 - sparse_categorical_accuracy: 0.9577 - val_loss: 0.0924 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 5.0000e-04
Epoch 166/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1226 - sparse_categorical_accuracy: 0.9578 - val_loss: 0.1151 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 167/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1064 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.1183 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 5.0000e-04
Epoch 168/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1079 - sparse_categorical_accuracy: 0.9659 - val_loss: 0.1493 - val_sparse_categorical_accuracy: 0.9473 - learning_rate: 5.0000e-04
Epoch 169/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1015 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1471 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
Epoch 170/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1136 - sparse_categorical_accuracy: 0.9611 - val_loss: 0.0895 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 171/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1086 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.0886 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 5.0000e-04
Epoch 172/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1022 - sparse_categorical_accuracy: 0.9630 - val_loss: 0.1097 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 173/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1054 - sparse_categorical_accuracy: 0.9666 - val_loss: 0.0911 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 174/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1006 - sparse_categorical_accuracy: 0.9642 - val_loss: 0.1057 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 175/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1150 - sparse_categorical_accuracy: 0.9607 - val_loss: 0.3058 - val_sparse_categorical_accuracy: 0.8669 - learning_rate: 5.0000e-04
Epoch 176/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1226 - sparse_categorical_accuracy: 0.9563 - val_loss: 0.3008 - val_sparse_categorical_accuracy: 0.8766 - learning_rate: 5.0000e-04
Epoch 177/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1032 - sparse_categorical_accuracy: 0.9627 - val_loss: 0.1166 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 5.0000e-04
Epoch 178/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1032 - sparse_categorical_accuracy: 0.9734 - val_loss: 0.1399 - val_sparse_categorical_accuracy: 0.9515 - learning_rate: 5.0000e-04
Epoch 179/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0981 - sparse_categorical_accuracy: 0.9636 - val_loss: 0.1535 - val_sparse_categorical_accuracy: 0.9390 - learning_rate: 5.0000e-04
Epoch 180/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1000 - sparse_categorical_accuracy: 0.9676 - val_loss: 0.1213 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 181/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1007 - sparse_categorical_accuracy: 0.9668 - val_loss: 0.1058 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 182/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0990 - sparse_categorical_accuracy: 0.9648 - val_loss: 0.1197 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 183/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0983 - sparse_categorical_accuracy: 0.9713 - val_loss: 0.1365 - val_sparse_categorical_accuracy: 0.9431 - learning_rate: 5.0000e-04
Epoch 184/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1001 - sparse_categorical_accuracy: 0.9608 - val_loss: 0.0870 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 185/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1062 - sparse_categorical_accuracy: 0.9684 - val_loss: 0.1739 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 5.0000e-04
Epoch 186/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1125 - sparse_categorical_accuracy: 0.9623 - val_loss: 0.1058 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 187/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1061 - sparse_categorical_accuracy: 0.9634 - val_loss: 0.1127 - val_sparse_categorical_accuracy: 0.9570 - learning_rate: 5.0000e-04
Epoch 188/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1024 - sparse_categorical_accuracy: 0.9636 - val_loss: 0.1041 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 189/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9696 - val_loss: 0.0914 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 5.0000e-04
Epoch 190/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0979 - sparse_categorical_accuracy: 0.9664 - val_loss: 0.0871 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 191/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1107 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.3318 - val_sparse_categorical_accuracy: 0.8669 - learning_rate: 5.0000e-04
Epoch 192/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1210 - sparse_categorical_accuracy: 0.9612 - val_loss: 0.0927 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 193/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0922 - sparse_categorical_accuracy: 0.9695 - val_loss: 0.0994 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 5.0000e-04
Epoch 194/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1100 - sparse_categorical_accuracy: 0.9615 - val_loss: 0.0856 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 5.0000e-04
Epoch 195/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1107 - sparse_categorical_accuracy: 0.9614 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 196/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1048 - sparse_categorical_accuracy: 0.9686 - val_loss: 0.3066 - val_sparse_categorical_accuracy: 0.8669 - learning_rate: 5.0000e-04
Epoch 197/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0956 - sparse_categorical_accuracy: 0.9651 - val_loss: 0.1379 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 5.0000e-04
Epoch 198/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9682 - val_loss: 0.0841 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 5.0000e-04
Epoch 199/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0931 - sparse_categorical_accuracy: 0.9712 - val_loss: 0.0884 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 5.0000e-04
Epoch 200/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0996 - sparse_categorical_accuracy: 0.9666 - val_loss: 0.0856 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 5.0000e-04
Epoch 201/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1011 - sparse_categorical_accuracy: 0.9630 - val_loss: 0.1599 - val_sparse_categorical_accuracy: 0.9334 - learning_rate: 5.0000e-04
Epoch 202/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1064 - sparse_categorical_accuracy: 0.9626 - val_loss: 0.1019 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 203/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0980 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.1702 - val_sparse_categorical_accuracy: 0.9348 - learning_rate: 5.0000e-04
Epoch 204/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1047 - sparse_categorical_accuracy: 0.9655 - val_loss: 0.2068 - val_sparse_categorical_accuracy: 0.9196 - learning_rate: 5.0000e-04
Epoch 205/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0921 - sparse_categorical_accuracy: 0.9696 - val_loss: 0.1166 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 5.0000e-04
Epoch 206/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1028 - sparse_categorical_accuracy: 0.9639 - val_loss: 0.1011 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 5.0000e-04
Epoch 207/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1204 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 5.0000e-04
Epoch 208/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0936 - sparse_categorical_accuracy: 0.9688 - val_loss: 0.0804 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 5.0000e-04
Epoch 209/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0907 - sparse_categorical_accuracy: 0.9703 - val_loss: 0.1012 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 210/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0983 - sparse_categorical_accuracy: 0.9676 - val_loss: 0.1750 - val_sparse_categorical_accuracy: 0.9320 - learning_rate: 5.0000e-04
Epoch 211/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0878 - sparse_categorical_accuracy: 0.9690 - val_loss: 0.1114 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 212/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1004 - sparse_categorical_accuracy: 0.9678 - val_loss: 0.0829 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 213/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0974 - sparse_categorical_accuracy: 0.9675 - val_loss: 0.3115 - val_sparse_categorical_accuracy: 0.8779 - learning_rate: 5.0000e-04
Epoch 214/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.2600 - val_sparse_categorical_accuracy: 0.9029 - learning_rate: 5.0000e-04
Epoch 215/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0879 - sparse_categorical_accuracy: 0.9752 - val_loss: 0.0921 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 5.0000e-04
Epoch 216/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0942 - sparse_categorical_accuracy: 0.9679 - val_loss: 0.1025 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 217/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0916 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.1124 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 5.0000e-04
Epoch 218/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.0950 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 5.0000e-04
Epoch 219/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1000 - sparse_categorical_accuracy: 0.9593 - val_loss: 0.0859 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 5.0000e-04
Epoch 220/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1187 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.0893 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 5.0000e-04
Epoch 221/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0905 - sparse_categorical_accuracy: 0.9691 - val_loss: 0.1528 - val_sparse_categorical_accuracy: 0.9376 - learning_rate: 5.0000e-04
Epoch 222/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0923 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.1189 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 5.0000e-04
Epoch 223/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1064 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.1031 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 224/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1020 - sparse_categorical_accuracy: 0.9669 - val_loss: 0.1172 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 5.0000e-04
Epoch 225/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0923 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1007 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 5.0000e-04
Epoch 226/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0947 - sparse_categorical_accuracy: 0.9668 - val_loss: 0.0896 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 5.0000e-04
Epoch 227/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0844 - sparse_categorical_accuracy: 0.9756 - val_loss: 0.1046 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 5.0000e-04
Epoch 228/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0799 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.1096 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 5.0000e-04
Epoch 229/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0890 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.1249 - val_sparse_categorical_accuracy: 0.9445 - learning_rate: 2.5000e-04
Epoch 230/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0834 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0819 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 231/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0864 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.1339 - val_sparse_categorical_accuracy: 0.9501 - learning_rate: 2.5000e-04
Epoch 232/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.0953 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 233/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9742 - val_loss: 0.0993 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 234/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0766 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.1430 - val_sparse_categorical_accuracy: 0.9459 - learning_rate: 2.5000e-04
Epoch 235/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0819 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.0807 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 2.5000e-04
Epoch 236/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0992 - sparse_categorical_accuracy: 0.9662 - val_loss: 0.0936 - val_sparse_categorical_accuracy: 0.9653 - learning_rate: 2.5000e-04
Epoch 237/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9745 - val_loss: 0.0829 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 2.5000e-04
Epoch 238/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0852 - sparse_categorical_accuracy: 0.9755 - val_loss: 0.0842 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 2.5000e-04
Epoch 239/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0782 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.1123 - val_sparse_categorical_accuracy: 0.9612 - learning_rate: 2.5000e-04
Epoch 240/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0911 - sparse_categorical_accuracy: 0.9682 - val_loss: 0.0854 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 2.5000e-04
Epoch 241/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0749 - sparse_categorical_accuracy: 0.9738 - val_loss: 0.0890 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 242/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0829 - sparse_categorical_accuracy: 0.9732 - val_loss: 0.1039 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 243/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0807 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 2.5000e-04
Epoch 244/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9741 - val_loss: 0.0853 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 2.5000e-04
Epoch 245/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0773 - sparse_categorical_accuracy: 0.9734 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 246/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0907 - sparse_categorical_accuracy: 0.9720 - val_loss: 0.0967 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 2.5000e-04
Epoch 247/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0867 - sparse_categorical_accuracy: 0.9751 - val_loss: 0.1254 - val_sparse_categorical_accuracy: 0.9528 - learning_rate: 2.5000e-04
Epoch 248/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0767 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0874 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 2.5000e-04
Epoch 249/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0787 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.2500e-04
Epoch 250/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0779 - sparse_categorical_accuracy: 0.9722 - val_loss: 0.0782 - val_sparse_categorical_accuracy: 0.9806 - learning_rate: 1.2500e-04
Epoch 251/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9777 - val_loss: 0.0905 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 252/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9779 - val_loss: 0.0784 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 253/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0775 - sparse_categorical_accuracy: 0.9707 - val_loss: 0.0794 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.2500e-04
Epoch 254/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0772 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0807 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 255/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0774 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0871 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 256/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0728 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0846 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.2500e-04
Epoch 257/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0799 - sparse_categorical_accuracy: 0.9728 - val_loss: 0.0801 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 258/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0837 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0831 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 259/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0754 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.0814 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.2500e-04
Epoch 260/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0878 - sparse_categorical_accuracy: 0.9758 - val_loss: 0.0815 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 261/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0841 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.0814 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.2500e-04
Epoch 262/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0826 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0858 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
Epoch 263/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0798 - val_sparse_categorical_accuracy: 0.9806 - learning_rate: 1.2500e-04
Epoch 264/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0814 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.0827 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.2500e-04
Epoch 265/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0681 - sparse_categorical_accuracy: 0.9790 - val_loss: 0.1069 - val_sparse_categorical_accuracy: 0.9556 - learning_rate: 1.2500e-04
Epoch 266/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9797 - val_loss: 0.0882 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 267/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9798 - val_loss: 0.0811 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.2500e-04
Epoch 268/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0813 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.2500e-04
Epoch 269/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0778 - sparse_categorical_accuracy: 0.9721 - val_loss: 0.0800 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.2500e-04
Epoch 270/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9731 - val_loss: 0.0792 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.2500e-04
Epoch 271/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0808 - sparse_categorical_accuracy: 0.9727 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 272/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0786 - sparse_categorical_accuracy: 0.9746 - val_loss: 0.0864 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 273/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0753 - sparse_categorical_accuracy: 0.9746 - val_loss: 0.0814 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 274/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9793 - val_loss: 0.0794 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 275/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0655 - sparse_categorical_accuracy: 0.9805 - val_loss: 0.0777 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 276/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9755 - val_loss: 0.0909 - val_sparse_categorical_accuracy: 0.9667 - learning_rate: 1.0000e-04
Epoch 277/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0784 - sparse_categorical_accuracy: 0.9774 - val_loss: 0.0822 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 278/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0871 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.0800 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 279/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9768 - val_loss: 0.0795 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 280/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9807 - val_loss: 0.0796 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 281/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0763 - sparse_categorical_accuracy: 0.9786 - val_loss: 0.0779 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 282/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0728 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.0796 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 283/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9817 - val_loss: 0.0816 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 284/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0728 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 285/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9733 - val_loss: 0.0910 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
Epoch 286/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.0832 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 287/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0740 - sparse_categorical_accuracy: 0.9752 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 288/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0726 - sparse_categorical_accuracy: 0.9736 - val_loss: 0.0809 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 289/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0695 - sparse_categorical_accuracy: 0.9773 - val_loss: 0.0889 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 290/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0837 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.0801 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 291/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0633 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.0851 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
Epoch 292/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0604 - sparse_categorical_accuracy: 0.9845 - val_loss: 0.0775 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 293/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0881 - sparse_categorical_accuracy: 0.9699 - val_loss: 0.0764 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 294/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0751 - sparse_categorical_accuracy: 0.9716 - val_loss: 0.0834 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 295/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9723 - val_loss: 0.0798 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 296/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0762 - sparse_categorical_accuracy: 0.9735 - val_loss: 0.0795 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 297/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9829 - val_loss: 0.0780 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 298/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9753 - val_loss: 0.0955 - val_sparse_categorical_accuracy: 0.9639 - learning_rate: 1.0000e-04
Epoch 299/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0777 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0775 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 300/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9756 - val_loss: 0.0788 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 301/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0814 - sparse_categorical_accuracy: 0.9782 - val_loss: 0.0787 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 302/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0682 - sparse_categorical_accuracy: 0.9769 - val_loss: 0.0829 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 303/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0685 - sparse_categorical_accuracy: 0.9765 - val_loss: 0.0823 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 304/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9783 - val_loss: 0.0852 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
Epoch 305/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9738 - val_loss: 0.0771 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 306/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9745 - val_loss: 0.0791 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 307/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0808 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.0784 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 308/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9786 - val_loss: 0.0785 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 309/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0677 - sparse_categorical_accuracy: 0.9762 - val_loss: 0.0774 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 310/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0707 - sparse_categorical_accuracy: 0.9765 - val_loss: 0.0794 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 311/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0644 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.0809 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 312/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0760 - sparse_categorical_accuracy: 0.9755 - val_loss: 0.0910 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 313/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9748 - val_loss: 0.0777 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 314/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0873 - sparse_categorical_accuracy: 0.9697 - val_loss: 0.0779 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 315/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9766 - val_loss: 0.0788 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 316/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0738 - sparse_categorical_accuracy: 0.9758 - val_loss: 0.0787 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 317/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0745 - sparse_categorical_accuracy: 0.9756 - val_loss: 0.0827 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 318/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0679 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0791 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 319/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0586 - sparse_categorical_accuracy: 0.9829 - val_loss: 0.0806 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 320/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.1026 - val_sparse_categorical_accuracy: 0.9598 - learning_rate: 1.0000e-04
Epoch 321/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9732 - val_loss: 0.0783 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 322/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0826 - val_sparse_categorical_accuracy: 0.9695 - learning_rate: 1.0000e-04
Epoch 323/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0937 - sparse_categorical_accuracy: 0.9659 - val_loss: 0.0788 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 324/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.0780 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 325/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0780 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0819 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 326/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0658 - sparse_categorical_accuracy: 0.9788 - val_loss: 0.0926 - val_sparse_categorical_accuracy: 0.9626 - learning_rate: 1.0000e-04
Epoch 327/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0779 - sparse_categorical_accuracy: 0.9771 - val_loss: 0.0874 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 328/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9794 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 329/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0718 - sparse_categorical_accuracy: 0.9737 - val_loss: 0.0818 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 330/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0739 - sparse_categorical_accuracy: 0.9752 - val_loss: 0.0776 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 331/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0763 - sparse_categorical_accuracy: 0.9751 - val_loss: 0.0797 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 332/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0648 - sparse_categorical_accuracy: 0.9783 - val_loss: 0.0837 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
Epoch 333/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0715 - sparse_categorical_accuracy: 0.9784 - val_loss: 0.0807 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 334/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0820 - sparse_categorical_accuracy: 0.9739 - val_loss: 0.0786 - val_sparse_categorical_accuracy: 0.9778 - learning_rate: 1.0000e-04
Epoch 335/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0640 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.0881 - val_sparse_categorical_accuracy: 0.9681 - learning_rate: 1.0000e-04
Epoch 336/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0680 - sparse_categorical_accuracy: 0.9817 - val_loss: 0.0821 - val_sparse_categorical_accuracy: 0.9750 - learning_rate: 1.0000e-04
Epoch 337/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0652 - sparse_categorical_accuracy: 0.9795 - val_loss: 0.0777 - val_sparse_categorical_accuracy: 0.9806 - learning_rate: 1.0000e-04
Epoch 338/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9796 - val_loss: 0.0783 - val_sparse_categorical_accuracy: 0.9764 - learning_rate: 1.0000e-04
Epoch 339/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0638 - sparse_categorical_accuracy: 0.9804 - val_loss: 0.0772 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 340/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0674 - sparse_categorical_accuracy: 0.9754 - val_loss: 0.0904 - val_sparse_categorical_accuracy: 0.9723 - learning_rate: 1.0000e-04
Epoch 341/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0807 - sparse_categorical_accuracy: 0.9714 - val_loss: 0.0877 - val_sparse_categorical_accuracy: 0.9736 - learning_rate: 1.0000e-04
Epoch 342/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0679 - sparse_categorical_accuracy: 0.9794 - val_loss: 0.0770 - val_sparse_categorical_accuracy: 0.9792 - learning_rate: 1.0000e-04
Epoch 343/500
 90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9719 - val_loss: 0.0817 - val_sparse_categorical_accuracy: 0.9709 - learning_rate: 1.0000e-04
Epoch 343: early stopping

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
 42/42 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - loss: 0.0943 - sparse_categorical_accuracy: 0.9668
Test accuracy 0.9689394235610962
Test loss 0.09581823647022247

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
