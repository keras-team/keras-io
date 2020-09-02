"""
Title: Knowledge Distillation
Author: [Kenneth Borup](https://twitter.com/Kennethborup)
Date created: 2020/09/01
Last modified: 2020/09/01
Description: Implementation of classical Knowledge Distillation.
"""

"""
## Introduction to Knowledge Distillation

Knowledge Distillation (or student-teacher training) is a procedure for model
compression, in which a small (student) model is trained to match a large pre-trained
(teacher) model. Intrinsic knowledge is transferred from the teacher model to the student
by minimizing a loss function, aimed at matching softened teacher logits as well as
ground-truth labels.
The logits are softened by applying a temperature in the softmax, effectively smoothing
out the probability distribution and revealing inter-class relationships learned by the
teacher.

- [Paper](https://arxiv.org/abs/1503.02531)
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
## Construct `Distiller()` class
The custom `Distiller()` class, merges custom `train_step` and `test_step` with classical
and custom Keras `compile()` elements. In order to apply the distiller, we need:

- A trained teacher model
- A student model to train
- A student loss function on the difference between student predictions and ground-truth
- A distillation loss function, along with a temperature `T`, on the difference between
the soft student predictions and the soft teacher labels
- `alpha` to weight the student and distillation loss
- An optimizer for the student and (optional) metrics to evaluate performance

In `train_step` we perform a forward pass of both the teacher and student, calculate the
loss with weighting of the student loss `s_loss` and distillation loss `d_loss` by
`alpha` and `1-alpha`, respectively, and update weights. Note, only the student weights
are updated and therefore, we merely calculate the gradients for the student weights. In
the `test_step` we evaluate the student model on the provided dataset as usual.
"""


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = student
        self.student = teacher

    def compile(self, optimizer, metrics, s_loss_fn, d_loss_fn, alpha, T):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.s_loss_fn = s_loss_fn
        self.d_loss_fn = d_loss_fn
        self.alpha = alpha
        self.T = T

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        t_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            s_pred = self.student(x, training=True)

            # Compute losses
            s_loss = self.s_loss_fn(y, s_pred)
            d_loss = self.d_loss_fn(
                tf.nn.softmax(t_pred / self.T, axis=1),
                tf.nn.softmax(s_pred / self.T, axis=1),
            )
            loss = self.alpha * s_loss + (1 - self.alpha) * d_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, s_pred)

        # Return a dict of performance
        return {
            **{"s_loss": s_loss, "d_loss": d_loss},
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_pred = self.student(x, training=False)

        # Calculate the loss
        s_loss = self.s_loss_fn(y, y_pred)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict of performance
        return {**{"s_loss": s_loss}, **{m.name: m.result() for m in self.metrics}}


"""
## Create student and teacher models
Initialy, we create a teacher model and a smaller student model. Both models are
convolutional neural networks and created using `Sequential()`, but could be any model
created with the Keras API.
"""

# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

"""
## Prepare dataset
The dataset used for training the teacher and distilling the teacher is
[MNIST](https://keras.io/api/datasets/mnist/). Both the student and teacher are trained
on the training set and evaluated on the test set.
"""

# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# Construct datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

"""
## Train teacher
In knowledge distillation we assume that the teacher is trained and fixed. Thus, we start
by training the teacher model on the training set in the usual way.
"""

# Train teacher as usual
teacher.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train and evaluate on subset of data to reduce computations.
teacher.fit(train_dataset.take(10), epochs=1)
loss, accuracy = teacher.evaluate(test_dataset.take(10))

"""
## Distill teacher to student
We have already trained the teacher model, and we only need to initialize a
`Distiller(student, teacher)` instance, `compile()` it with the desired losses,
hyperparameters and optimizer, and distill the teacher to the student.
"""

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer="adam",
    metrics=["accuracy"],
    s_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    d_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    T=3,
)

# Distill teacher to student
distiller.fit(train_dataset.take(10), epochs=1)

# Evaluate student on test dataset
accuracy = distiller.evaluate(test_dataset.take(10))

"""
If the teacher is trained for 5 full epochs and the student is distilled on this teacher
for 5 epochs, you should experience a performance boost compared to training the same
student model from scratch, i.e. in the classical manner.
"""
