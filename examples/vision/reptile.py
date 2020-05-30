"""
Title: REPTILE
Author: [ADMoreau](https://github.com/ADMoreau)
Date created: 2020/05/21
Last modified: 2020/05/29
Description: Reptile meta learning algorithm implemented in Keras

"""

"""
The Reptile algorithm (https://arxiv.org/abs/1803.02999) was developed by OpenAI to
perform model agnostic meta-learning. Specifically, this algorithm was designed to
quickly learn to perform new tasks with minimal training, a task known as few-shot
learning. The algorithm works by taking metagradients from the training of mini-batches
representing new tasks, and using these gradients for a meta training step.
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

"""
## Define the Parameters
"""

learning_rate = 0.003
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 5
classes = 5

"""
## Dataset Functions
"""


class Dataset:
    # This dataset function will fascilitate the creation of a few shot dataset
    # from the Omniglot dataset that can be sampled from quickly while also
    # allowing to create new labels at the same time.
    def __init__(self, training):

        # Download the tfrecord files containing the omniglot data and convert to a
        # dataset.
        if training:
            ds = tfds.load(
                "omniglot", split="train", as_supervised=True, shuffle_files=False
            )
            ds = ds.batch(1)
        else:
            ds = tfds.load(
                "omniglot", split="test", as_supervised=True, shuffle_files=False
            )
            ds = ds.batch(1)
        # Iterate over the dataset to get each individual image and accompanying class
        # and put that data into a dictionary.
        self.data = {}

        def extraction(image, label):
            # This function will shrink the omniglot images into the desired size,
            # scale the pixel values and convert the RGB image to graysclae
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            return image, label

        for image, label in ds.map(extraction):
            image = image.numpy()[0]
            label = str(label.numpy()[0])
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
            self.labels = list(self.data.keys())

    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, split=False
    ):

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 28, 28, 1))
        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 28, 28, 1))
        # get a random subset of labels from the entire label set
        label_subset = random.choices(self.labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            # use enumerated index value as a temporary label for mini_batch in
            # few shot learning
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            # if creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset
            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
            else:
# for each index in the randomly selected label_subset, sample the
necessary
                # number of images
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        if split:
            return dataset, test_images, test_labels
        return dataset


import urllib3

urllib3.disable_warnings()
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

"""
## Build the Model
"""


class ModelLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(ModelLayer, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")
        self.bnorm = layers.BatchNormalization()
        self.activation = layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bnorm(x)
        return self.activation(x)


inputs = layers.Input(shape=(28, 28, 1))
layer1 = ModelLayer()(inputs)
layer2 = ModelLayer()(layer1)
layer3 = ModelLayer()(layer2)
layer4 = ModelLayer()(layer3)
flatten = layers.Flatten()(layer4)
dense = layers.Dense(classes, activation="softmax")(flatten)
# Log softmax
outputs = tf.math.log(layers.Softmax()(dense))
model = keras.Model(inputs=inputs, outputs=dense)
model.compile()
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

"""
## Training Loop
"""

training = []
testing = []
for meta_iter in range(meta_iters):
    frac_done = meta_iter / meta_iters
    cur_meta_step_size = (1 - frac_done) * meta_step_size
    # Temporarily save the weights from the model.
    old_vars = model.get_weights()
    # Get a sample from the full dataset.
    mini_dataset = train_dataset.get_mini_dataset(
        inner_batch_size, inner_iters, train_shots, classes
    )
    for images, labels in mini_dataset:
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = keras.losses.categorical_crossentropy(
                tf.one_hot(labels, depth=classes), logits
            )
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    new_vars = model.get_weights()
    # Perform SGD for the meta step.
    for var in range(len(new_vars)):
        new_vars[var] = old_vars[var] + (
            (new_vars[var] - old_vars[var]) * cur_meta_step_size
        )
    # After the meta-learning step, reload the newly trained weights into the model.
    model.set_weights(new_vars)
    # Evaluation loop
    if meta_iter % eval_interval == 0:
        accuracies = []
        for dataset in (train_dataset, test_dataset):
            # Sample a mini dataset from the full dataset.
            train_set, test_images, test_labels = dataset.get_mini_dataset(
                eval_batch_size, eval_iters, shots, classes, split=True
            )
            old_vars = model.get_weights()
            # Train on the samples and get the resulting accuracies.
            for images, labels in train_set:
                with tf.GradientTape() as tape:
                    logits = model(images)
                    loss = keras.losses.categorical_crossentropy(
                        tf.one_hot(labels, depth=classes), logits
                    )
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            test_preds = model.predict(test_images)
            test_preds = tf.argmax(test_preds).numpy()
            num_correct = (test_preds == test_labels).sum()
            # And reset the weights after getting the evaluation accuracies.
            model.set_weights(old_vars)
            accuracies.append(num_correct / classes)
        training.append(accuracies[0])
        testing.append(accuracies[1])
        if meta_iter % 100 == 0:
            print(
                "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
            )

"""
## Visualize Results
"""

# First, some preprocessing to smooth the training and testing arrays for display.
window_len = 100
w = eval("np.hamming(window_len)")
train_s = np.r_[
    training[window_len - 1 : 0 : -1], training, training[-1:-window_len:-1]
]
test_s = np.r_[testing[window_len - 1 : 0 : -1], testing, testing[-1:-window_len:-1]]
train_y = np.convolve(w / w.sum(), train_s, mode="valid")
test_y = np.convolve(w / w.sum(), test_s, mode="valid")

import matplotlib.pyplot as plt

# And display the training accuracies.
x = np.arange(0, len(test_y), 1)
plt.plot(x, test_y, x, train_y)
plt.legend(["test", "train"])
plt.grid()
