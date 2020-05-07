# A Quasi-SVM in Keras

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/17<br>
**Last modified:** 2020/04/17<br>
**Description:** Demonstration of how to train a Keras model that approximates a SVM.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/quasi_svm.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/quasi_svm.py)



---
## Introduction

This example demonstrates how to train a Keras model that approximates a Support Vector
 Machine (SVM).

The key idea is to stack a `RandomFourierFeatures` layer with a linear layer.

The `RandomFourierFeatures` layer can be used to "kernelize" linear models by applying
 a non-linear transformation to the input
features and then training a linear model on top of the transformed features. Depending
on the loss function of the linear model, the composition of this layer and the linear
model results to models that are equivalent (up to approximation) to kernel SVMs (for
hinge loss), kernel logistic regression (for logistic loss), kernel linear regression
 (for MSE loss), etc.

In our case, we approximate SVM using a hinge loss.

---
## Setup


```python

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

```

---
## Build the model


```python

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

```

---
## Prepare the data


```python

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data by flattening & scaling it
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

# Categorical (one hot) encoding of the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

```

---
## Train the model


```python

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

```

<div class="k-default-codeblock">
```
Epoch 1/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0829 - acc: 0.8681 - val_loss: 0.0432 - val_acc: 0.9358
Epoch 2/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0423 - acc: 0.9364 - val_loss: 0.0364 - val_acc: 0.9471
Epoch 3/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0340 - acc: 0.9502 - val_loss: 0.0360 - val_acc: 0.9488
Epoch 4/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0292 - acc: 0.9572 - val_loss: 0.0286 - val_acc: 0.9582
Epoch 5/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0251 - acc: 0.9637 - val_loss: 0.0261 - val_acc: 0.9643
Epoch 6/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0231 - acc: 0.9684 - val_loss: 0.0259 - val_acc: 0.9639
Epoch 7/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0215 - acc: 0.9710 - val_loss: 0.0247 - val_acc: 0.9662
Epoch 8/20
375/375 [==============================] - 2s 7ms/step - loss: 0.0197 - acc: 0.9740 - val_loss: 0.0263 - val_acc: 0.9629
Epoch 9/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0190 - acc: 0.9749 - val_loss: 0.0222 - val_acc: 0.9704
Epoch 10/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0177 - acc: 0.9767 - val_loss: 0.0224 - val_acc: 0.9689
Epoch 11/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0168 - acc: 0.9781 - val_loss: 0.0231 - val_acc: 0.9661
Epoch 12/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0158 - acc: 0.9804 - val_loss: 0.0232 - val_acc: 0.9688
Epoch 13/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0153 - acc: 0.9814 - val_loss: 0.0227 - val_acc: 0.9682
Epoch 14/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0140 - acc: 0.9829 - val_loss: 0.0228 - val_acc: 0.9678
Epoch 15/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0143 - acc: 0.9820 - val_loss: 0.0230 - val_acc: 0.9676
Epoch 16/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0134 - acc: 0.9840 - val_loss: 0.0246 - val_acc: 0.9675
Epoch 17/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0127 - acc: 0.9853 - val_loss: 0.0224 - val_acc: 0.9697
Epoch 18/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0124 - acc: 0.9855 - val_loss: 0.0248 - val_acc: 0.9659
Epoch 19/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0117 - acc: 0.9867 - val_loss: 0.0207 - val_acc: 0.9722
Epoch 20/20
375/375 [==============================] - 2s 6ms/step - loss: 0.0113 - acc: 0.9870 - val_loss: 0.0205 - val_acc: 0.9724

<tensorflow.python.keras.callbacks.History at 0x168558710>

```
</div>
I can't say that it works well or that it is indeed a good idea, but you can probably
 get decent results by tuning your hyperparameters.

You can use this setup to add a "SVM layer" on top of a deep learning model, and train
 the whole model end-to-end.
