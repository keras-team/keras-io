"""
<<<<<<< HEAD
Title: Invariant Information Clusering(IIC)
Author: [Anish Josh](https://twitter.com/anishhacko)
Date created: 2020/08/12
Last modified: 2020/08/19
Description: Implementation of unsupervised learning algorithm using IIC.
=======
Title: Invariant Information Clustering(IIC)
Author: [Anish Josh](https://twitter.com/anishhacko)
Date created: 2020/08/05
Last modified: 2020/06/18
Description: Implementation of unsupervised learning example using IIC algorithm.
>>>>>>> 8b685b388d2b9d1c6cccaf83b4809bf3e829af95
"""
"""


## Introduction
This example helps to get started with unsupervised learning methods. Code implemented
here is
based on paper [Invariant Information Clustering](https://arxiv.org/abs/1807.06653).In a
nutshell 
this algorithm tries to maximize the mutual information between two images. Most
excitingly we will use fit method to run the experiment.
"""

"""
## Setup
"""

import requests
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50

"""
## Tuning Paramters
#### Reading the paper would help in more detail, choosing these parameters.
1. subheads - Total number of subheads for clustering.
2. main_classes  - Total number classes to classify (guess the value).
3. auxilary_classes - Number of auxilary classses for overclusteing(can be ignored during
inference).
"""

sub_heads = 4
main_classes = 6
auxilary_classes = 10
image_h, image_w = 256, 256
batch_size = 8
learning_rate = 5e-6
epochs = 200
dataset = "toy_data"
<<<<<<< HEAD
logdir = "./logs"  # Tensorboard directory
=======
logdir = "./logs" #tensorboard directory path
>>>>>>> 8b685b388d2b9d1c6cccaf83b4809bf3e829af95

"""
## Downloading Dataset
##### Download and unzip it.
"""

# !wget https://github.com/anish9/Tutorials/raw/master/toy_data.zip
# !unzip toy_data.zip

"""
 ## Build Data Pipeline
"""

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset, label_mode=None, batch_size=batch_size, image_size=(image_h, image_w)
)
train_ds = train_ds.prefetch(32).repeat()
test_path = sorted(glob(dataset + "/copys/*"))

"""
## Visualizing The Loaded Dataset
"""

plt.figure(figsize=(12, 8))
for img in train_ds.take(1):
    for w in range(1, batch_size + 1):
        plt.subplot(4, 4, w)
        image = img[w - 1, :, :, :]
        image = image.numpy().astype(np.uint8)
        plt.imshow(image)
        plt.axis("off")

"""
## Building Model: 
#### 1 - Defining Augumentation Layers (only target image is augumented, source is
scaled).
#### 2 - Defining Backbone Network.
#### 3 - Defining Clustering Network.
"""

<<<<<<< HEAD
# Source augument has only normalizing effect
=======
# Source augument has only normalizing effect.
>>>>>>> 8b685b388d2b9d1c6cccaf83b4809bf3e829af95
source_augument = keras.Sequential(
    [keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),]
)

<<<<<<< HEAD
# Target augment can be augmented with respect to the dataset pattern
=======
# Target augment can contain multiple augumentations with respect to the dataset.
>>>>>>> 8b685b388d2b9d1c6cccaf83b4809bf3e829af95
target_augment = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.4),
        keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
    ]
)

# Backbone Network
def resnet_backbone(subheads, classes, auxilary_branch=True, auxilary_classes=10):
    base_full = ResNet50(include_top=False, input_shape=(image_h, image_w, 3))
    base_outp = base_full.output
    gpool = keras.layers.GlobalAveragePooling2D()(base_outp)
    outmaps = []
    auxmaps = []
    for w in range(subheads):
        main_out = keras.layers.Dense(classes, activation="softmax")(gpool)
        outmaps.append(main_out)
    if auxilary_branch:
        for w in range(subheads):
            aux_out = keras.layers.Dense(auxilary_classes, activation="softmax")(gpool)
            auxmaps.append(aux_out)
        fmodel = keras.models.Model(
            base_full.input, [outmaps, auxmaps], name="IIC_aux_node"
        )
        return fmodel, auxilary_branch

    else:
        fmodel = Model(base_full.input, outmaps, name="IIC_main_node")
        return fmodel, auxilary_branch


<<<<<<< HEAD
# Clustering network
=======
# Clustering Network
>>>>>>> 8b685b388d2b9d1c6cccaf83b4809bf3e829af95
class cluster_IIC(keras.models.Model):
    def __init__(self, backbone, auxilary_flag=False):
        super(cluster_IIC, self).__init__()
        self.backbone = backbone
        self.auxilary = auxilary_flag

    def compile(self, optimizer, loss):
        super(cluster_IIC, self).compile()
        self.optimizer_fn = optimizer
        self.loss_fn = loss

    def train_step(self, x):
        xin, xin_ = source_augument(x), target_augment(x)
        with tf.GradientTape() as t:
            if self.auxilary:
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
## Building Loss Function
"""


class IIC_loss(keras.losses.Loss):
    def __init__(self, epsilon=1e-6, lamb=1.0, reduction=keras.losses.Reduction.AUTO):
        super(IIC_loss, self).__init__(reduction=reduction)
        self.epsilon = epsilon
        self.lamb = lamb

    def call(self, x, x_):
        c = tf.shape(x)[-1]  # class labels
        i = tf.reshape(x, (-1, c, 1))
        j = tf.reshape(x_, (-1, 1, c))
        j_prob = tf.matmul(i, j)  # compute joint probability
        j_prob = tf.reduce_sum(j_prob, axis=0)  # sum the sub_heads
        j_prob = (j_prob + tf.transpose(j_prob)) / 2  # impose invariant compute
        j_prob = tf.clip_by_value(j_prob, self.epsilon, 1e9)
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
## Custom Callback
Custom callback to save only Inference Model
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
## Initialization and Training
"""

backbone, auxilary_flag = resnet_backbone(
    subheads=sub_heads,
    classes=main_classes,
    auxilary_branch=True,
    auxilary_classes=auxilary_classes,
)
cluster_model = cluster_IIC(backbone, auxilary_flag=auxilary_flag)
loss = IIC_loss()
tensor_board = keras.callbacks.TensorBoard(logdir)
ckpt = custom_checkpoint(backbone)
optimizer = tf.keras.optimizers.Adam(learning_rate)
cluster_model.compile(optimizer=optimizer, loss=loss)
history = cluster_model.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=len(test_path) // batch_size,
    callbacks=[tensor_board, ckpt],
)

"""
## Inference and Visualization
 This section holds codes for prediction time clustering and visualization.
"""


class infernce:
    def __init__(self, model, path, mode):
        self.model = model  # Load bakcbone as infernce model
        self.path = path  # Image_directory path
        self.mode = mode  # Predict using auxilary or main branch
        self.clusters = defaultdict(list)

    def predict(self, image_name):
        """for each image predict their clusters"""
        if self.mode not in {"aux", "main"}:
            raise ValueError("allowed modes {aux or main}")
        im = load_img(image_name, target_size=(image_h, image_w))
        im = img_to_array(im) / 255.0
        im = np.expand_dims(im, axis=0)
        pred1, pred2 = self.model.predict(im)
        if self.mode == "aux":
            predf = np.array(pred2)
        if self.mode == "main":
            predf = np.array(pred1)
        predf = np.sum(predf, axis=0)
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
        for w in range(1, batch_size + 1):
            try:
                image = load_img(predictions[w], target_size=(image_h, image_w))
                image = img_to_array(image, dtype=np.uint8)
                plt.subplot(4, 4, w)
                plt.imshow(image)
            except:
                None


cluster_inf = infernce(backbone, test_path, "main")  #
keys_out = cluster_inf.get_cluster_keys()

"""
### Visualize using predicted cluster keys
"""

cluster_inf.visualize_cluster(3)
# cluster_inf.visualize_cluster(1)
# cluster_inf.visualize_cluster(7)
