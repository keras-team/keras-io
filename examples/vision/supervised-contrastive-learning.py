"""
Title: Supervised Contrastive Learning
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/11/30
Last modified: 2026/04/04
Description: Using supervised contrastive learning for image classification.
Accelerator: GPU
Converted to Keras 3 by: [LakshmiKalaKadali](https://github.com/LakshmiKalaKadali)
"""

"""
## Introduction

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
(Prannay Khosla et al.) is a training methodology that outperforms
supervised training with crossentropy on classification tasks.

Essentially, training an image classification model with Supervised Contrastive
Learning is performed in two phases:

1. Training an encoder to learn to produce vector representations of input images such
that representations of images in the same class will be more similar compared to
representations of images in different classes.
2. Training a classifier on top of the frozen encoder.

## Setup
"""

import os

# Set backend: "jax", "torch", or "tensorflow"
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers, ops

"""
## Prepare the data
"""

num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


"""
## Using image data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_train)

"""
## Build the encoder model

The encoder model takes the image as input and turns it into a 2048-dimensional
feature vector.
"""


def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


encoder = create_encoder()
encoder.summary()

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05

"""
## Build the classification model

The classification model adds a fully-connected layer on top of the encoder,
plus a softmax layer with the target classes.
"""


def create_classifier(encoder, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


"""
## Experiment 1: Train the baseline classification model

In this experiment, a baseline classifier is trained as usual, i.e., the
encoder and the classifier parts are trained together as a single model
to minimize the crossentropy loss.
"""

encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

# history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")


"""
## Experiment 2: Use supervised contrastive learning

In this experiment, the model is trained in two phases. In the first phase,
the encoder is pretrained to optimize the supervised contrastive loss,
described in [Prannay Khosla et al.](https://arxiv.org/abs/2004.11362).

In the second phase, the classifier is trained using the trained encoder with
its weights freezed; only the weights of fully-connected layers with the
softmax are optimized.

### 1. Supervised contrastive learning loss function
"""


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, labels, feature_vectors):
        feature_vectors = ops.normalize(feature_vectors, axis=1)

        logits = ops.divide(
            ops.matmul(feature_vectors, ops.transpose(feature_vectors)),
            self.temperature,
        )

        # Create a mask to find positive pairs (images of same class)
        labels = ops.cast(labels, "int32")
        labels = ops.reshape(labels, (-1, 1))
        mask = ops.cast(ops.equal(labels, ops.transpose(labels)), "float32")

        batch_size = ops.shape(logits)[0]
        logits_mask = 1.0 - ops.eye(batch_size)
        mask = mask * logits_mask

        logits_max = ops.max(logits, axis=1, keepdims=True)
        logits_exp = ops.exp(logits - logits_max) * logits_mask

        log_prob = (logits - logits_max) - ops.log(
            ops.sum(logits_exp, axis=1, keepdims=True) + 1e-8
        )

        mean_log_prob_pos = ops.sum(mask * log_prob, axis=1) / (
            ops.sum(mask, axis=1) + 1e-8
        )

        return ops.subtract(0.0, ops.mean(mean_log_prob_pos))


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


"""
### 2. Pretrain the encoder
"""

encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=num_epochs,
)

"""
### 3. Train the classifier with the frozen encoder
"""

classifier = create_classifier(encoder, trainable=False)

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

"""
We get to an improved test accuracy.
"""

"""
## Conclusion

As shown in the experiments, using the supervised contrastive learning technique
outperformed the conventional technique in terms of the test accuracy. Note that
the same training budget (i.e., number of epochs) was given to each technique.
Supervised contrastive learning pays off when the encoder involves a complex
architecture, like ResNet, and multi-class problems with many labels.
In addition, large batch sizes and multi-layer projection heads
improve its effectiveness. See the [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
paper for more details.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/supervised-contrastive-learning-cifar10)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/supervised-contrastive-learning).
"""
