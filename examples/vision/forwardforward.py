"""
Title: Using Forward-Forward Algorithm for Image Classification
Author: [Suvaditya Mukherjee](https://twitter.com/halcyonrayes)
Date created: 2022/12/21
Last modified: 2022/12/23
Description: Training a Dense-layer based model using the Forward-Forward algorithm.

"""

"""
## Introduction

The following example explores how to use the Forward-Forward algorithm to perform
training instead of the traditionally-used method of backpropagation, as proposed by
[Prof. Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
The concept was inspired by the understanding behind [Boltzmann
Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf). Backpropagation involves
calculating loss via a cost function and propagating the error across the network. On the
other hand, the FF Algorithm suggests the analogy of neurons which get "excited" based on
looking at a certain recognized combination of an image and its correct corresponding
label.
This method takes certain inspiration from the biological learning process that occurs in
the cortex. A significant advantage that this method brings is the fact that
backpropagation through the network does not need to be performed anymore, and that
weight updates are local to the layer itself.
As this is yet still an experimental method, it does not yield state-of-the-art results.
But with proper tuning, it is supposed to come close to the same.
Through this example, we will examine a process that allows us to implement the
Forward-Forward algorithm within the layers themselves, instead of the traditional method
of relying on the global loss functions and optimizers.
The process is as follows:
- Perform necessary imports
- Load the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- Visualize Random samples from the MNIST dataset
- Define a `FFDense` Layer to override `call` and implement a custom `forwardforward`
method which performs weight updates.
- Define a `FFNetwork` Layer to override `train_step`, `predict` and implement 2 custom
functions for per-sample prediction and overlaying labels
- Convert MNIST from `NumPy` arrays to `tf.data.Dataset`
- Fit the network
- Visualize results
- Perform inference testing

As this example requires the customization of certain core functions with
`tf.keras.layers.Layer` and `tf.keras.models.Model`, refer to the following resources for
a primer on how to do so
- [Customizing what happens in
`model.fit()`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [Making new Layers and Models via
subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
"""

"""
## Setup imports
"""

import tensorflow as tf
from tensorflow import keras
from tqdm.notebook import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

tf.config.run_functions_eagerly(True)

"""
## Load dataset and visualize

We use the `keras.datasets.mnist.load_data()` utility to directly pull the MNIST dataset
in the form of `NumPy` arrays. We then arrange it in the form of the train and test
splits.

Following loading the dataset, we select 4 random samples from within the training set
and visualize them using `matplotlib.pyplot`
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("4 Random Training samples and labels")
idx1, idx2, idx3, idx4 = random.sample(range(0, x_train.shape[0]), 4)

img1 = (x_train[idx1], y_train[idx1])
img2 = (x_train[idx2], y_train[idx2])
img3 = (x_train[idx3], y_train[idx3])
img4 = (x_train[idx4], y_train[idx4])

imgs = [img1, img2, img3, img4]

plt.figure(figsize=(10, 10))

for idx, item in enumerate(imgs):
    image, label = item[0], item[1]
    plt.subplot(2, 2, idx + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Label : {label}")
plt.show()

"""
## Define `FFDense` Custom Layer

In this custom layer, we have a base `tf.keras.layers.Dense` object which acts as the
base `Dense` layer within. Since weight updates will happen within the layer itself, we
add an `tf.keras.optimizers.Optimizer` object that is accepted from the user. Here, we
use `Adam` as our optimizer with a rather higher learning rate of `0.03`.
Following the algorithm's specifics, we must set a `threshold` parameter that will be
used to make the positive-negative decision in each prediction. This is set to a default
of 2.0
As the epochs are localized to the layer itself, we also set a `num_epochs` parameter
(default at 2000).

We override the `call` function in order to perform a normalization over the complete
input space followed by running it through the base `Dense` layer as would happen in a
normal `Dense` layer call.

We implement the Forward-Forward algorithm which accepts 2 kinds of input tensors, each
representing the positive and negative samples respectively. We write a custom training
loop here with the use of `tf.GradientTape()`, within which we calculate a loss per
sample by taking the distance of the prediction from the threshold to understand the
error and taking its mean to get a `mean_loss` metric.
With the help of `tf.GradientTape()` we calculate the gradient updates for the trainable
base `Dense` layer and apply them using the layer's local optimizer.

Finally, we return the `call` result as the `Dense` results of the positive and negative
samples while also returning the last `mean_loss` metric and all the loss values over a
certain all-epoch run.
"""


class FFDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        optimizer,
        num_epochs=2000,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(FFDense, self).__init__()
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            **kwargs,
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = optimizer
        self.threshold = 2.0
        self.num_epochs = num_epochs

    def call(self, x):
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return self.relu(res)

    def forwardforward(self, x_pos, x_neg):
        loss_list = []
        for i in trange(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)

                loss = tf.math.log(
                    1
                    + tf.math.exp(
                        tf.concat([-g_pos + self.threshold, g_neg - self.threshold], 0)
                    )
                )
                mean_loss = tf.math.reduce_mean(loss)
                loss_list.append(mean_loss.numpy())
            gradients = tape.gradient(mean_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            tf.stop_gradient(self.call(x_pos)),
            tf.stop_gradient(self.call(x_neg)),
            mean_loss,
            loss_list,
        )


"""
## Define the `FFNetwork` Custom Model

With our custom layer defined, we also need to override the `train_step` method and
define a custom `tf.keras.models.Model` that works with our `FFDense` layer.

For this algorithm, we must 'embed' the labels onto the original image. To do so, we
exploit the structure of MNIST images where the top-left 10 pixels are always zeros. We
use that as a label space in order to visually one-hot-encode the labels within the image
itself. This action is performed by the `overlay_y_on_x` function.

We break down the prediction function with a per-sample prediction function which is then
called over the entire test set by the overriden `predict()` function. The prediction is
performed here with the help of measuring the `excitation` of the neurons per layer for
each image. This is then summed over all layers to calculate a network-wide 'goodness
score'. The label with the highest 'goodness score' is then chosen as the sample
prediction.

The `train_step` function is overriden to act as the main controlling loop for running
training on each layer as per the number of epochs per layer.
"""


class FFNetwork(keras.Model):
    def __init__(self, dims, layer_optimizer=keras.optimizers.Adam(learning_rate=0.03)):
        super().__init__()
        self.layer_optimizer = layer_optimizer
        self.mean_loss = keras.metrics.Mean()
        self.flatten_layer = keras.layers.Flatten()
        self.layer_list = [keras.Input(shape=(dims[0],))]
        for d in range(len(dims) - 1):
            self.layer_list += [FFDense(dims[d + 1], optimizer=self.layer_optimizer)]

    @tf.function()
    def overlay_y_on_x(self, X, y):
        x_res = X.numpy()
        x_npy = X.numpy()
        x_res[:, :10] *= 0.0
        if not isinstance(y, int):
            y_npy = y.numpy()
            x_res[range(x_npy.shape[0]), y.numpy()] = x_npy.max()
        else:
            x_res[range(x_npy.shape[0]), y] = x_npy.max()
        return tf.convert_to_tensor(x_res)

    @tf.function()
    def predict_one_sample(self, x):
        goodness_per_label = []
        x = tf.expand_dims(x, axis=0)
        for label in range(10):
            h = self.overlay_y_on_x(x, label)
            h = self.flatten_layer(h)
            goodness = []
            for layer_idx in range(1, len(self.layer_list)):
                layer = self.layer_list[layer_idx]
                h = layer(h)
                goodness += [tf.math.reduce_mean(tf.math.pow(h, 2), 1)]
            goodness_per_label += [
                tf.expand_dims(tf.reduce_sum(goodness, keepdims=True), 1)
            ]
        goodness_per_label = tf.concat(goodness_per_label, 1)
        return tf.argmax(goodness_per_label, 1)

    def predict(self, data):
        x = data
        preds = list()
        for idx in trange(x.shape[0]):
            sample = x[idx]
            result = self.predict_one_sample(sample)
            preds.append(result)
        return np.asarray(preds, dtype=int)

    def train_step(self, data):
        x, y = data
        x = self.flatten_layer(x)
        perm_array = tf.range(start=0, limit=x.get_shape()[0], delta=1)
        x_pos = self.overlay_y_on_x(x, y)
        y_numpy = y.numpy()
        random_y_tensor = y_numpy[tf.random.shuffle(perm_array)]
        x_neg = self.overlay_y_on_x(x, tf.convert_to_tensor(random_y_tensor))
        h_pos, h_neg = x_pos, x_neg
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                print("Input layer : No training")
                continue
            print(f"Training layer {idx+1} now : ")
            if isinstance(layer, FFDense):
                h_pos, h_neg, loss, loss_list = layer.forwardforward(h_pos, h_neg)
                plt.plot(range(len(loss_list)), loss_list)
                plt.title(f"Loss over training on layer {idx+1}")
                plt.show()
            else:
                x = layer(x)
        return {"FinalLoss": loss}


"""
## Convert MNIST `NumPy` arrays to `tf.data.Dataset`

We now perform some preliminary processing on the `NumPy` arrays and then convert them
into the `tf.data.Dataset` format which allows for optimized loading.
"""

x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.batch(60000)
test_dataset = test_dataset.batch(10000)

"""
## Fit the network and visualize results

Having performed all previous set-up, we are now going to run `model.fit()` and run 1
model epoch, which will perform 2000 epochs on each layer. We get to see the plotted loss
curve as each layer is trained.
"""

model = FFNetwork(dims=[784, 500, 500])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.03), loss="mse", run_eagerly=True
)

history = model.fit(train_dataset, epochs=1)

"""
## Perform inference and testing

Having trained the model to a large extent, we now see how it performs on the test set.
We calculate the Accuracy Score to understand the results closely.
"""

preds = model.predict(tf.convert_to_tensor(x_test))

preds = preds.reshape((preds.shape[0], preds.shape[1]))

results = accuracy_score(preds, y_test)

print(f"Accuracy score : {results*100}%")

"""
## Conclusion:

This example has hereby demonstrated how the Forward-Forward algorithm works using
TensorFlow and Keras modules. While the investigation results presented by Prof. Hinton
in their paper are currently still limited to smaller models and datasets like MNIST and
Fashion-MNIST, subsequent results on larger models like LLMs are expected in future
papers.

Through the paper, Prof. Hinton has reported results of 1.36% test error with a
2000-units, 4 hidden-layer, fully-connected network run over 60 epochs (while mentioning
that backpropagation takes only 20 epochs to achieve similar performance). Another run of
doubling the learning rate and training for 40 epochs yields a slightly worse error rate
of 1.46%

The current example does not yield state-of-the-art results. But with proper tuning of
the Learning Rate, model architecture (no. of units in `Dense` layers, kernel
activations, initializations, regularization etc.), the results can be improved
drastically to match the claims of the paper.
"""
