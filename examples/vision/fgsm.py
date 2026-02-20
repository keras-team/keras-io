"""
Title: Adversarial examples generation from scratch
Author: [Zhuoyi Mou](https://github.com/Alexair059)
Date created: 2023/09/02
Last modified: 2023/09/02
Description: Implement FGSM attack with keras from scratch.
Accelerator: None
"""
"""
## Introduction

The Fast Gradient Sign Attack (FGSM) is one of the earliest and most popular adversarial
attacks. The idea of FGSM is intuitive and effective: the attack is carried out using the
way neural networks learn, i.e., the gradient, using the back-propagation gradient of the
model and adjusting the sample data to maximize losses.

We utilize keras' built-in `MNIST` dataset to implement a FGSM attack on a trained LeNet
model. In this tutorial GPU devices are not necessary and you can complete the training
in a shorter time.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt

"""
## Epsilon Values for FGSM Attack

The larger the `epsilon`, the more noticable the perturbations are.
"""

epsilon = [0, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35]

"""
## Prepare the Data (MNIST)
"""

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.mnist.load_data()

train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_images = train_images.reshape((*train_images.shape, 1))
test_images = test_images.reshape((*test_images.shape, 1))

"""
## Build and Train the Model (LeNet Modified)
"""

input = keras.Input((28, 28, 1))
x = layers.Conv2D(32, 3, strides=1, activation="relu")(input)
x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=input, outputs=output)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

"""
## Sample Datas for FGSM
"""

datas = tf.constant(test_images[:256])
targets = tf.constant(test_labels[:256].astype("float16"))

"""
## Drawing One Epsilon for FGSM Attack

Here we choose 0.1, `epsilon[1]`
"""

with tf.GradientTape() as tape:
    tape.watch(datas)
    tape.watch(targets)
    outs = model(datas)
    sparse_crossentropy_loss = tf.losses.SparseCategoricalCrossentropy()
    loss = sparse_crossentropy_loss(targets, outs)

datas_grad = tape.gradient(loss, datas)
perturbed_datas = datas + epsilon[1] * tf.sign(datas_grad)

"""
## Test and Visualization
"""

prediction = model(datas).numpy().argmax(axis=1)
adv_prediction = model(perturbed_datas).numpy().argmax(axis=1)
result = prediction == adv_prediction
inconsistent_result = np.argwhere(result == False).squeeze()
print(f"Attck Success Rate: {len(inconsistent_result) / len(result): .4f}")

index = inconsistent_result[0]

ex = tf.squeeze(datas[index]).numpy()
pre = prediction[index]
adv = tf.squeeze(perturbed_datas[index]).numpy()
adv_pre = adv_prediction[index]

plt.figure(figsize=(8, 10))

plt.subplot(1, 2, 1)
plt.title(f"Predict: {pre}")
plt.xticks([], [])
plt.yticks([], [])
plt.imshow(ex, cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"Attack: {adv_pre}")
plt.xticks([], [])
plt.yticks([], [])
plt.imshow(adv, cmap="gray")
plt.show()
