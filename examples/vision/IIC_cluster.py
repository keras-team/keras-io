"""
Title: Invariant information clustering 
Author: [Anish](https://twitter.com/anishhacko)
Date created: 2020/08/03
Last modified: 2020/08/13
Description: Unsupervised information learning
"""
"""


## Introduction
This example helps to get started with unsupervised learning, code implemented here is
based on [Invariant Information Clustering](https://arxiv.org/pdf/1807.06653.pdf).<br>In
a nutshell 
this algorithm tries to maximize the mutal information between two images, most
importantly we will<br>
use fit method to implement in keras which makes it more simpler to break and understand
the algorithm.

"""

"""
## Setup
"""

import requests
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

"""
## Tuning Paramters
#### Reading the paper would help in more detail in choosing these parameters.
1. subheads - total number of subheads for clustering.
2. classes  - total number classes to classify (guess the value)
3. aux_classes - Number of auxilary classses for overclusteing(can be ignored during
inference)
"""

sub_heads = 3
classes_ = 4
aux_classes = 8
imh, imw = 224, 224
BATCH_SIZE = 8
BUFFER = 1024
LR_RATE = 1e-5
EPOCHS = 180

"""
## Download the dataset.
##### Download and unzip it
"""

"""shell
!wget https://github.com/anish9/Tutorials/raw/master/toy.zip
!unzip toy.zip
"""

"""
 ## Build Data Pipeline
 It just requires original image and its augumented version,
 apply some random transformations to generate real and its augumented version.
"""


def x_gen(im):
    # random crops
    if np.random.randint(1, 8, 1)[0] > 4:
        image = tf.image.random_crop(im, (imh // 2, imw // 2, 3))
    if np.random.randint(1, 8, 1)[0] > 4:
        image = tf.image.random_brightness(im, 0.4)  # random brightness
        image = tf.image.rot90(image, k=1)
    if np.random.randint(1, 8, 1)[0] > 4:
        image = tf.image.random_contrast(im, 0.2, 0.7)  # random contrast
        image = tf.image.rot90(image, k=3)
    else:
        image = tf.image.random_flip_up_down(im)  # random flips vertical
    image = tf.image.resize(im, (imh, imw))
    return image


def read_file(im):
    image = tf.io.read_file(im)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (imh * 2, imw * 2))
    image_ = x_gen(image)
    image = tf.image.resize(image, (imh, imw))
    return image / 255.0, image_ / 255.0


image_path = sorted(glob("toy/*"))
TRAIN = tf.data.Dataset.from_tensor_slices((image_path))
TRAIN = TRAIN.map(read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
TRAIN = TRAIN.batch(BATCH_SIZE, drop_remainder=True).shuffle(BUFFER)

"""
## Visualize the data
"""

plt.figure(figsize=(12, 8))
for batch in TRAIN.take(1):
    img = batch[0]
    for w in range(1, BATCH_SIZE + 1):
        plt.subplot(4, 4, w)
        image = img[w - 1, :, :, :] * 255.0
        image = image.numpy().astype(np.uint8)
        plt.imshow(image)

"""
## Building backbone network and cluster model
VGG 16 is used as backbone network
"""


def VGG16_bb(subheads, classes, aux_branch=True, aux_classes=15):
    base_full = VGG16(include_top=False, input_shape=(imh, imw, 3))
    base_outp = base_full.output
    gpool = keras.layers.GlobalAveragePooling2D()(base_outp)
    outmaps = []
    auxmaps = []
    for w in range(subheads):
        outpu = keras.layers.Dense(classes, activation="softmax")(gpool)
        outmaps.append(outpu)
    if aux_branch:

        for w in range(subheads):
            auxou = keras.layers.Dense(aux_classes, activation="softmax")(gpool)
            auxmaps.append(auxou)
        fmodel = keras.models.Model(base_full.input, [outmaps, auxmaps], name="IIC_db")
        return fmodel

    else:
        fmodel = keras.models.Model(base_full.input, outmaps, name="IIC_sb")
        return fmodel


class cluster_IIC(keras.models.Model):
    def __init__(self, backbone, class_, aux_class, aux=False):
        super(cluster_IIC, self).__init__()
        self.backbone = backbone
        self.aux = aux
        self.class_ = class_
        self.aux_class = aux_class

    def compile(self, optimizer, loss):
        super(cluster_IIC, self).compile()
        self.optimizer_fn = optimizer
        self.loss_fn = loss

    def train_step(self, x):
        xin, xin_ = x
        with tf.GradientTape() as t:
            if self.aux:
                # trains with auxilary branch combined
                m1, a1 = self.backbone(xin)
                m2, a2 = self.backbone(xin_)
                iicm = tf.reduce_mean([self.loss_fn(x, y) for (x, y) in zip(m1, m2)])
                iica = tf.reduce_mean([self.loss_fn(x, y) for (x, y) in zip(a1, a2)])
                iicc = iicm + iica
            else:
                m1 = self.backbone(xin)
                m2 = self.backbone(xin_)
                iicc = tf.reduce_mean([self.loss_fn(x, y) for (x, y) in zip(m1, m2)])

        grad = t.gradient(iicc, self.backbone.trainable_weights)
        self.optimizer_fn.apply_gradients(zip(grad, self.backbone.trainable_weights))
        return {"loss": iicc}


"""
## Building custom loss function
"""


class IIC_loss(keras.losses.Loss):
    def __init__(self, epsilon=1e-6, lamb=1.0, reduction=keras.losses.Reduction.AUTO):
        super(IIC_loss, self).__init__(reduction=reduction)
        self.epsilon = epsilon
        self.lamb = lamb

    def call(self, x, x_):
        c = tf.shape(x)[-1]  # class labels
        epsilon = self.epsilon
        lamb = self.lamb
        i = tf.reshape(x, (-1, c, 1))
        j = tf.reshape(x_, (-1, 1, c))
        j_prob = tf.matmul(i, j)  # compute joint probability
        j_prob = tf.reduce_sum(j_prob, axis=0)  # sum the sub_heads
        j_prob = (j_prob + tf.transpose(j_prob)) / 2  # impose invariant compute
        j_prob = tf.clip_by_value(j_prob, epsilon, 1e9)
        j_prob /= tf.reduce_sum(j_prob)
        pi_ = tf.reduce_sum(j_prob, axis=0)
        pi_ = tf.reshape(pi_, (-1, 1))
        pi_ = tf.broadcast_to(pi_, (c, c))
        pj_ = tf.reduce_sum(j_prob, axis=1)
        pj_ = tf.reshape(pj_, (1, -1))
        pj_ = tf.broadcast_to(pj_, (c, c))
        loss_ = -tf.reduce_sum(
            j_prob * (tf.math.log(j_prob) - tf.math.log(pi_) - tf.math.log(pj_))
        )
        return loss_


"""
## Custom callback
custom callback to save only inference model
"""


class custom_checkpoint(keras.callbacks.Callback):
    def __init__(self, model_func, loss_track=0):
        self.model_ = model_func
        self.track = loss_track

    def on_epoch_end(self, epoch, logs=None):
        track_obj = logs["loss"]
        if track_obj < self.track:
            print(f" Model saved at for best loss: {track_obj}")
            self.model_.save_weights("saved_best.h5")
            self.track = track_obj


"""
## Initialization and training
"""

backbone = VGG16_bb(subheads=sub_heads, classes=classes_, aux_classes=aux_classes)
clu = cluster_IIC(backbone, classes_, aux_classes, aux=True)
loss_c = IIC_loss()

logdir = "./logs"
tb = keras.callbacks.TensorBoard(logdir)
ckpt = custom_checkpoint(backbone)

optim = tf.keras.optimizers.Adam(LR_RATE)
clu.compile(optimizer=optim, loss=loss_c)
history = clu.fit(
    TRAIN,
    batch_size=BATCH_SIZE,
    epochs=5,
    steps_per_epoch=len(image_path) // BATCH_SIZE,
    callbacks=[tb, ckpt],
)

"""
## Infernce and Visualization
 This section holds prediction time clustering and visualization codes
"""


class infernce:
    def __init__(self, model, path, mode):
        self.model = model  # bakcbone as infernce model
        self.path = path  # image_dir path
        self.mode = mode  # predict using aux or main branch
        self.clusters = defaultdict(list)

    def predict(self, image_name):
        """for each image predict their clusters"""
        if self.mode not in {"aux", "main"}:
            raise ValueError("allowed modes {aux or main}")
        im = load_img(image_name, target_size=(imh, imw))
        im = img_to_array(im) / 255.0
        im = np.expand_dims(im, axis=0)
        pred1, pred2 = self.model.predict(im)
        if self.mode == "aux":
            predf = np.array(pred2)
        if self.mode == "main":
            predf = np.array(pred1)
        predf = np.sum(predf, axis=0)  # sum in branch axis
        predf = np.argmax(predf, axis=-1)
        return predf[0]

    def get_cluster_keys(self):
        dirs = self.path
        for files in dirs:
            pred = self.predict(files)
            self.clusters[pred].append(files)
        cluster_keys = list(self.clusters.keys())
        print(f"available predicted keys... {cluster_keys}")
        return cluster_keys

    def visualize_cluster(self, cluster_id):
        """helps to visualize images in  clsuters based on the 
        predicted cluster branch values"""
        predictions = self.clusters[cluster_id]
        plt.figure(figsize=(8, 12))
        for w in range(1, BATCH_SIZE + 1):
            try:
                image = load_img(predictions[w], target_size=(imh, imw))
                image = img_to_array(image, dtype=np.uint8)
                plt.subplot(4, 4, w)
                plt.imshow(image)
            except:
                None


cluster_inf = infernce(backbone, image_path, "main")
keys_out = cluster_inf.get_cluster_keys()

"""
## Visualize the clusters using predicted cluster keys
"""

cluster_inf.visualize_cluster(0)
cluster_inf.visualize_cluster(1)
cluster_inf.visualize_cluster(3)
