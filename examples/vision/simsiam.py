"""
Title: Self-supervised contrastive learning with SimSiam
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/03/13
Last modified: 2021/03/17
Description: Implementation of a self-supervised learning method for computer vision.
"""
"""
Self-supervised learning (SSL) is an interesting branch of study in the field of
representation learning. SSL systems try to formulate a supervised signal from a corpus
of unlabeled data points.  An example is we train a deep neural network to predict the
next word from a given set of words. In literature, these tasks are known as *pretext
tasks* or *auxiliary tasks*. If we [train such a
network](https://arxiv.org/abs/1801.06146) on a huge dataset (such as the [Wikipedia text
corpus](https://www.corpusdata.org/wikipedia.asp)) it learns very effective
representations that transfer well to downstream tasks. Language models like
[BERT](https://arxiv.org/abs/1810.04805), [GPT-3](https://arxiv.org/abs/2005.14165),
[ELMo](https://allennlp.org/elmo) all benefit from this. 

Much like the language models we can train computer vision models using similar
approaches. To make things work in computer vision, we need to formulate the learning
tasks such that the underlying model (a deep neural network) is able to make sense of the
semantic information present in vision data. One such task is to a model to _contrast_
between two different versions of the same image. The hope is that in this way the model
will have learn representations where the similar images are grouped as together possible
while the dissimilar images are further away. 

In this example, we will be implementing one such system called **SimSiam** proposed in
[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566). It
is implemented as the following:
1. We create two different versions of the same dataset with a stochastic data
augmentation pipeline. Note that the random initialization seed needs to be the same
during create these versions. 
2. We take a ResNet50 without any classification head (**backbone**) and we add a shallow
fully-connected network (**projection head**) on top of it. Collectively, this is known
as the **encoder**. 
3. We pass the output of the encoder through a **predictor** which is again a shallow
fully-connected network having an
[AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder) like structure. 
4. We then train our encoder to maximize the cosine similarity between the two different
versions of our dataset. 

This example requires TensorFlow 2.4 or higher.
"""

"""
## Setup
"""

from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

tfds.disable_progress_bar()

"""
## Define hyperparameters
"""

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 5
RESIZE_TO = 260
CROP_TO = 224
SEED = 26

PROJECT_DIM = 2048
LATENT_DIM = 512

"""
## Load the [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/) dataset
"""

train_ds = tfds.load("tf_flowers", as_supervised=False, split=["train[:85%]"])[0]

"""
## Defining our data augmentation pipeline

As studied in [SimCLR](https://arxiv.org/abs/2002.05709) having the right data
augmentation pipeline is critical for SSL systems to work effectively in computer vision.
Two particular augmentation transforms that seem to matter the most are: 1.) Random
resized crops and 2.) Color distortions. Most of the other SSL systems for computer
vision (such as [BYOL](https://arxiv.org/abs/2006.07733),
[MoCov2](https://arxiv.org/abs/2003.04297), [SwAV](https://arxiv.org/abs/2006.09882), etc.)
include these in their training pipelines.  
"""


def random_resize_crop(image):
    # We first resize our image to a bigger dimension and then
    # we take random crops from it fitting to our desired
    # dimension.
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    return image


def color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


# This function is referred from here:
# https://github.com/google-research/simclr/blob/master/data_util.py.
def random_apply(func, x, p):
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )


def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = image["image"]
    image = random_resize_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image


"""
It should be noted that an augmentation pipeline is generally dependent on various
properties of the dataset we are dealing with. For example, if images in the dataset are
heavily object-centric then taking random crops with a very high probability may hurt the
training performance.

Let's now apply our augmentation pipeline to our dataset and visualize a few outputs. 
"""

"""
## Convert the data into TensorFlow `Dataset` objects

Here we create two different versions of our dataset *without* any ground-truth labels. 
"""

ssl_ds_one = (
    train_ds.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

ssl_ds_two = (
    train_ds.shuffle(1024, seed=SEED)
    .map(custom_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# We then zip both of these datasets.
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# Visualize a few augmented images.
sample_images_one = next(iter(ssl_ds_one))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_one[n].numpy().astype("int"))
    plt.axis("off")
plt.show()

# Ensure that the different versions of the dataset actually contain
# identical images.
sample_images_two = next(iter(ssl_ds_two))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_two[n].numpy().astype("int"))
    plt.axis("off")
plt.show()

"""
Notice that the images in `samples_images_one` and `sample_images_two` are essentially
the same but are augmented differently.
"""

"""
## Defining the encoder and the predictor

The configurations of these architectures have been referred from Section 3 of [the
paper](https://arxiv.org/abs/2011.10566). 
"""


def get_encoder():
    resnet50_v2 = tf.keras.applications.ResNet50V2(
        weights=None, include_top=False, input_shape=(CROP_TO, CROP_TO, 3)
    )
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1)(
        inputs
    )
    x = resnet50_v2(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(PROJECT_DIM, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(PROJECT_DIM, use_bias=False)(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")


def get_predictor():
    model = tf.keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            layers.Input((PROJECT_DIM,)),
            layers.Dense(LATENT_DIM, use_bias=False),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dense(PROJECT_DIM),
        ],
        name="predictor",
    )
    return model


print(get_encoder().summary())
print(get_predictor().summary())

"""
## Defining the (pre-)training loop

One of the main reasons behind training networks with these kinds of approaches is to
utilize the learned representations for downstream tasks like classification. This is why
this particular training phase is also referred to as _pre-training_. 

We start by defining the loss function.
"""


def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


"""
We then define our training loop by overriding the `train_step()` function of the
`tf.keras.Model` class. 
"""


class SimSiam(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


"""
## Pre-training our networks

In the interest of this example, we will train the model for only 5 epochs.
"""

# Create a cosine decay learning scheduler.
num_training_samples = tf.data.experimental.cardinality(train_ds).numpy()
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.03, decay_steps=steps
)

# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

# Start training.
simsiam = SimSiam(get_encoder(), get_predictor())
simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])

plt.plot(history.history["loss"])
plt.grid()
plt.show()

"""
Although our solution quickly gets very close to the minimum loss value (-1 in this case)
this is likely happening because of *representation collapse* where the encoder projects
all the images to identical representations. This suggests that additional hyperparameter
tuning is required especially in the following areas:
* Strength of the color distortions and their probabilities. 
* Learning rate and its schedule. 
* Architecture of both the backbone and their projection head. 
"""

"""
## Evaluating our SSL method

The most popularly used method to evaluate a SSL method in computer vision (or any other
pre-training method as such) is to learn a linear classifier on the frozen features of
the trained backbone model (in this case it is `ResNet50V2`) and evaluate the classifier
on unseen images. Other methods include
[fine-tuning](https://keras.io/guides/transfer_learning/) on the source dataset or even a
target dataset with 5% or 10% labels present. Practically, we can use the backbone model
for any downstream task such as semantic segmentation, object detection, and so on where
the backbone models are usually pre-trained with *pure supervised learning*. 
"""

# We first load our supervised dataset.
train_ds, validation_ds = tfds.load(
    "tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True
)

# Then we shuffle, batch, and prefetch this dataset for performance. We
# also apply random resized crops as an augmentation but only to the
# training set.
train_ds = (
    train_ds.shuffle(1024)
    .map(lambda x, y: (random_resize_crop(x), y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
validation_ds = (
    validation_ds.shuffle(1024)
    .map(
        lambda x, y: (tf.image.resize(x, (CROP_TO, CROP_TO)), y),
        num_parallel_calls=AUTO,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Extract the backbone ResNet50V2.
rn50 = tf.keras.Model(simsiam.encoder.input, simsiam.encoder.layers[3].output)

# We then create our linear classifier and train it.
rn50.trainable = False
inputs = layers.Input((CROP_TO, CROP_TO, 3))
x = rn50(inputs, training=False)
outputs = layers.Dense(5, activation="softmax")(x)
linear_model = tf.keras.Model(inputs, outputs)

# Start training.
linear_model.compile(
    loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
)
history = linear_model.fit(
    train_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[early_stopping]
)

"""
## Notes
* More data and longer pre-training schedule benefit SSL in general.
* SSL is particularly very helpful when you do not have access to very limited *labeled*
training data but you can manage to build a large corpus of unlabeled data. Recently,
using an SSL method called [SwAV](https://arxiv.org/abs/2006.09882), a group of
researchers at Facebook trained a [RegNet](https://arxiv.org/abs/2006.09882) on 2 Billion
images. They were able to achieve downstream performance very close to those achieved by
pure supervised pre-training. For some downstream tasks, their method even outperformed
the supervised counterparts. You can check out [their
paper](https://arxiv.org/pdf/2103.01988.pdf) to know the details. 
* If you are interested to understand why contrastive SSL helps networks learn meaningful
representations, you can check out the following resources:
   * [Self-supervised learning: The dark matter of
intelligence](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)
   * [Understanding self-supervised learning using controlled datasets with known
structure](https://sslneuips20.github.io/files/CameraReadys%203-77/64/CameraReady/Understanding_self_supervised_learning.pdf)
"""
