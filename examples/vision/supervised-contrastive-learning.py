"""
Title: Supervised Contrastive Learning
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/11/01
Last modified: 2020/11/01
Description: Using supervised contrastive learning for image classification.


## Introduction

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) (Prannay Khosla et al.) is a training methodology that outperforms cross-entropy on supervised learning tasks. 

Essentially, training an image classification model with Supervised Contrastive Learning is performed in two phases: 

1. Pre-training an encoder to generate feature vectors for input images such that feature vectors of images in the same class will be more similar compared feature vectors of images in other classes.
2. Training a classifier on top of the freezed encoder.

## Setup
"""

"""shell
pip install -q tensorflow-addons
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import keras
from keras import layers

"""## Prepare the data"""

num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

"""## Build the encoder model

The encoder model takes the image as an input and produce a 128-dimension feature vector.
"""

def create_encoder():
  inputs = keras.Input(shape=input_shape)

  x = layers.Conv2D(32, 3)(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(64, 3)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)

  x = layers.Conv2D(128, 3)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(256, 3)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)

  outputs = layers.GlobalAveragePooling2D()(x)

  model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
  return model

encoder = create_encoder()
encoder.summary()

batch_size = 256
num_epochs = 25
dropout_rate = 0.5
temperature = 0.05

"""## Build the classification model

The classification model adds a fully-connected layer on top of the encoder, plus a softmax layer with the target classes.
"""

def create_classifier(encoder, trainable=True):

  for layer in encoder.layers:
    layer.trainable = trainable

  inputs = keras.Input(shape=input_shape)
  features = encoder(inputs)
  features = layers.Dropout(dropout_rate)(features)
  outputs =  layers.Dense(num_classes, activation="softmax")(features)
  model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
  model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=keras.metrics.SparseCategoricalAccuracy()
  )
  return model

"""## Experiment 1: Train the baseline classification model 

In this experiment, a baseline classifier is trained normally, i.e., the encoder and the classifier parts are trained together as a single model to minimize cross-entropy loss.
"""

encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {round(accuracy * 100, 2)}%')

"""We get to ~70.5% test accuracy.

## Experiment 2: Use supervised contrastive learning
In this experiment, the model is trained in two phases. In the first phase, the encoder is pretrained to optimize the supervised contrastive loss, described in [Prannay Khosla et al.](https://arxiv.org/abs/2004.11362). In the second phase, the classifier is trained using the trained encoder with its weights freezed; only the weights of fully-connected layers with the softmax are optimized.

### 1. Supervised contrastive learning loss function
"""

class SupervisedContrastiveLoss(keras.losses.Loss):
  def __init__(self, temperature=1, name=None):
    super(SupervisedContrastiveLoss,self).__init__(name=name)
    self.temperature = temperature
    
  def __call__(self, labels, feature_vectors, sample_weight=None):
    # Normalize feature vectors
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    # Compute logits 
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, 
            tf.transpose(feature_vectors_normalized)),
        temperature
    )
    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

"""### 2. Pretrain the encoder"""

encoder = create_encoder()
encoder.compile(
    optimizer=keras.optimizers.Adam(),
    loss=SupervisedContrastiveLoss(temperature)
)

history = encoder.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=num_epochs)

"""### 3. Train the classifier with the freezed encoder"""

classifier = create_classifier(encoder, trainable=False)
history = classifier.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {round(accuracy * 100, 2)}%')

"""We get to ~79.1% test accuracy."""

