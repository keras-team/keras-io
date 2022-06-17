# VICReg: Variance-Invariance-Covariance Regularization for SSL

**Author:** Abhiraam Eranti<br>
**Date created:** 4/13/2022<br>
**Last modified:** 4/13/2022<br>
**Description:** We implement VICReg using Tensorflow Similarity and train on CIFAR-10.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/vicreg.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/vicreg.py)



---
## Introduction

**Problem**
VicReg, created by Adrien Bardes, Jean Ponce, and Yann LeCun, is a self-supervised method
to generate high-quality embeddings that
maximize the amount of dataset-related information inside them.
Previously, the main way to get these kinds of embeddings was to just calculate
the distance between representations of similar and different images.
Ideally, similar images would have similar embeddings and different images
have different ones. However, there was one problem: they
would *collapse*, or try to "cheat the system". Let's look at an example:
Suppose we had an image of a cat and a dog. The embeddings should
primarily store information from the images that differentiate the cat from its canine
counterpart. For example, it could keep the shape of the ears of both images, or maybe
the tail length, and so on. When used in a downstream task,
like a classification model, these embeddings(which have the curated measurements
mentioned above) should assist the model.
However, instead of this occurring, these approaches would produce embeddings that did
not help the downstream model as much as they should have. This is because they would
become redundant, meaning they repeated information more
than once. This led to less information being passed to the downstream model.
The previous solutions were to carefully and precisely tune the weights and
augmentations of the model and data such that collapse does not occur. However,
this was a finicky task, and even then, redundancy was still an issue.
**Solutions**
VicReg was not the first solution to this. Barlow Twins is
another similar method that was designed to reduce redundancy by measuring both
the invariance and covariance of embeddings. It works pretty well at
doing this, and is generally better in performance to contrastive models like
SimCLR.
VicReg is inspired by Barlow Twins and shares a similar performance to it
on tasks like image classification, the example shown here. Instead of just
measuring the invariance and covariance, it measures *similarity*, *variance*,
and *covariance* concerning the embeddings instead. However, they share the
same model composition and training loop, and both are substantially simpler
than other methods to train.
However, VicReg outperforms Barlow Twins on multimodal tasks like
image-to-text and text-to-image translation.

We will also utilize **Tensorflow Similarity**, a library designed to make metric and
self-supervised learning easier for practical use. Using this
library, we do not need to make the augmentations, model architectures,
training loop, and visualization code ourselves.

### References
[VicReg Paper](https://arxiv.org/abs/2105.04906)
[VicReg PyTorch Implementation](https://github.com/facebookresearch/vicreg)
[Barlow Twins Paper](https://arxiv.org/abs/2103.03230)
[Barlow Twins Example(Some of the architecture code is copied from
it)](https://keras.io/examples/vision/barlow_twins/)
[Tensorflow Similarity](https://github.com/tensorflow/similarity)

---
## Installation and Imports

We need `tensorflow-addons` for the LAMB loss function and
`tensorflow-similarity` for our augmenting, model building, and training setup.


```python
!!pip install tensorflow-addons
!!pip install tensorflow-similarity
```




```python
import os

# slightly faster improvements, on the first epoch 30 second decrease and a 1-2 second
# decrease in epoch time. Overall saves approx. 5 min of training time

# Allocates two threads for a gpu private which allows more operations to be
# done faster
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

import tensorflow as tf  # framework
from tensorflow import keras  # for tf.keras
import tensorflow_addons as tfa  # LAMB optimizer and gaussian_blur_2d function
import numpy as np  # np.random.random
import matplotlib.pyplot as plt  # graphs
import datetime  # tensorboard logs naming
import tensorflow_similarity  # loss function module
from functools import partial

# XLA optimization for faster performance(up to 10-15 minutes total time saved)
tf.config.optimizer.set_jit(True)
```
<div class="k-default-codeblock">
```
['Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/',
 'Collecting tensorflow-similarity',
 '  Downloading tensorflow_similarity-0.16.3-py3-none-any.whl (204 kB)',
 '\x1b[?25l',
 '\x1b[K     |█▋                              | 10 kB 27.6 MB/s eta 0:00:01',
 '\x1b[K     |███▏                            | 20 kB 18.5 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 30 kB 10.4 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 40 kB 8.4 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 51 kB 4.5 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 61 kB 5.3 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 71 kB 5.4 MB/s eta 0:00:01',
 '\x1b[K     |████████████▉                   | 81 kB 5.4 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 92 kB 6.0 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 102 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▋              | 112 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▏            | 122 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 133 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▍         | 143 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████████        | 153 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████████▋      | 163 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████████▏    | 174 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████████████▉   | 184 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████████████▍ | 194 kB 5.1 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████████████████| 204 kB 5.1 MB/s ',
 '\x1b[?25hCollecting nmslib',
 '  Downloading nmslib-2.1.1-cp37-cp37m-manylinux2010_x86_64.whl (13.5 MB)',
 '\x1b[?25l',
 '\x1b[K     |                                | 10 kB 33.9 MB/s eta 0:00:01',
 '\x1b[K     |                                | 20 kB 40.4 MB/s eta 0:00:01',
 '\x1b[K     |                                | 30 kB 44.9 MB/s eta 0:00:01',
 '\x1b[K     |                                | 40 kB 48.9 MB/s eta 0:00:01',
 '\x1b[K     |▏                               | 51 kB 51.1 MB/s eta 0:00:01',
 '\x1b[K     |▏                               | 61 kB 53.9 MB/s eta 0:00:01',
 '\x1b[K     |▏                               | 71 kB 55.7 MB/s eta 0:00:01',
 '\x1b[K     |▏                               | 81 kB 57.4 MB/s eta 0:00:01',
 '\x1b[K     |▏                               | 92 kB 57.0 MB/s eta 0:00:01',
 '\x1b[K     |▎                               | 102 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▎                               | 112 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▎                               | 122 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▎                               | 133 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▍                               | 143 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▍                               | 153 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▍                               | 163 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▍                               | 174 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▍                               | 184 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▌                               | 194 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▌                               | 204 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▌                               | 215 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▌                               | 225 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▋                               | 235 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▋                               | 245 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▋                               | 256 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▋                               | 266 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▋                               | 276 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▊                               | 286 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▊                               | 296 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▊                               | 307 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▊                               | 317 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▊                               | 327 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▉                               | 337 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▉                               | 348 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▉                               | 358 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |▉                               | 368 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 378 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 389 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 399 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 409 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 419 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 430 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 440 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 450 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█                               | 460 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▏                              | 471 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▏                              | 481 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▏                              | 491 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▏                              | 501 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▏                              | 512 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▎                              | 522 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▎                              | 532 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▎                              | 542 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▎                              | 552 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▍                              | 563 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▍                              | 573 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▍                              | 583 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▍                              | 593 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▍                              | 604 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▌                              | 614 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▌                              | 624 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▌                              | 634 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▌                              | 645 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▌                              | 655 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▋                              | 665 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▋                              | 675 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▋                              | 686 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▋                              | 696 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▊                              | 706 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▊                              | 716 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▊                              | 727 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▊                              | 737 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▊                              | 747 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▉                              | 757 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▉                              | 768 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▉                              | 778 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█▉                              | 788 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 798 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 808 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 819 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 829 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 839 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 849 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 860 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 870 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██                              | 880 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▏                             | 890 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▏                             | 901 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▏                             | 911 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▏                             | 921 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▏                             | 931 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▎                             | 942 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▎                             | 952 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▎                             | 962 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▎                             | 972 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▎                             | 983 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▍                             | 993 kB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▍                             | 1.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▍                             | 1.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▍                             | 1.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▌                             | 1.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▌                             | 1.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▌                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▌                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▌                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▋                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▋                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▋                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▋                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▊                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▊                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▊                             | 1.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▊                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▊                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▉                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▉                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▉                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▉                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██▉                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███                             | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▏                            | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▏                            | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▏                            | 1.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▏                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▎                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▎                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▎                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▎                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▎                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▍                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▍                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▍                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▍                            | 1.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▌                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▌                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▌                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▌                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▌                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▋                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▋                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▋                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▋                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▋                            | 1.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▊                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▊                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▊                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▊                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▉                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▉                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▉                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▉                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███▉                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████                            | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▏                           | 1.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▏                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▏                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▏                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▎                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▎                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▎                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▎                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▎                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▍                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▍                           | 1.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▍                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▍                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▍                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▌                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▌                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▌                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▌                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▋                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▋                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▋                           | 1.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▋                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▋                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▊                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▊                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▊                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▊                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 2.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████▉                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████                           | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▏                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▏                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▏                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▏                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▏                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▎                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▎                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▎                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▎                          | 2.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▍                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▍                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▍                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▍                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▍                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▌                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▌                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▌                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▌                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▋                          | 2.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▋                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▋                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▋                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▋                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▊                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▊                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▊                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▊                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▊                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▉                          | 2.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▉                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▉                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████▉                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████                          | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▏                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▏                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▏                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▏                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▏                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▎                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▎                         | 2.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▎                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▎                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▍                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▌                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▌                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▌                         | 2.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▌                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▌                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▋                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▋                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▋                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▋                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▊                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▊                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▊                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▊                         | 2.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▊                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▉                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▉                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▉                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████▉                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 2.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████                         | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▏                        | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▏                        | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▏                        | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▏                        | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▏                        | 3.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▎                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▎                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▎                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▎                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▎                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▍                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▍                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▍                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▍                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▌                        | 3.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▌                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▌                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▌                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▌                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▋                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▋                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▋                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▋                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▊                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▊                        | 3.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▊                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▊                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▊                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▉                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▉                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▉                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▉                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████▉                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████                        | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▏                       | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▏                       | 3.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▏                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▏                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▎                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▎                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▎                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▎                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▎                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▍                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▍                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▍                       | 3.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▍                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▌                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▌                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▌                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▌                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▌                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▋                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▋                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▋                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▋                       | 3.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▋                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▊                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▊                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▊                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▊                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▉                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▉                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▉                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▉                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████▉                       | 3.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████                       | 3.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▏                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▏                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▏                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▏                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▎                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▎                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▎                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▎                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▎                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▍                      | 3.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▍                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▍                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▍                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▍                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▌                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▌                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▌                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▌                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 4.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▋                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▊                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▊                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▊                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▊                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▉                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▉                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▉                      | 4.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▉                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████▉                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████                      | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▏                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▏                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▏                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▏                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▏                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▎                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▎                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▎                     | 4.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▎                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▍                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▍                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▍                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▍                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▍                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▌                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▌                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▌                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▌                     | 4.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▋                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▋                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▋                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▋                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▋                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▊                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▊                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▊                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▊                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▊                     | 4.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▉                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▉                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▉                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████▉                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████                     | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▏                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▎                    | 4.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▎                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▎                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▎                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▍                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▍                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▍                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▍                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▍                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▌                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▌                    | 4.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▌                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▌                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▌                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▋                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▋                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▋                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▋                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▊                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▊                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▊                    | 4.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▊                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▊                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▉                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▉                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▉                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████▉                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████                    | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▏                   | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▏                   | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▏                   | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▏                   | 5.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▏                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▎                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▎                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▎                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▎                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▎                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▍                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▍                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▍                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▍                   | 5.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▌                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▌                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▌                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▌                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▌                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▋                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▋                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▋                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▋                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▊                   | 5.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▊                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▊                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▊                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▊                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▉                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▉                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▉                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████▉                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████                   | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▏                  | 5.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▏                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▏                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▏                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▎                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▎                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▎                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▎                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▎                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▍                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▍                  | 5.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▍                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▍                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▌                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▌                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▌                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▌                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▌                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▋                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▋                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▋                  | 5.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▋                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▋                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▊                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▊                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▊                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▊                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▉                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▉                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▉                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▉                  | 5.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████▉                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████                  | 5.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▏                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▏                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▏                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▏                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▎                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▎                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▎                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▎                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▎                 | 6.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▍                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▌                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▌                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▌                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▌                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▋                 | 6.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▋                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▋                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▋                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▋                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▊                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▊                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▊                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▊                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▉                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▉                 | 6.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▉                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▉                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████▉                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████                 | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▏                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▏                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▏                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▏                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▏                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▎                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▎                | 6.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▎                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▎                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▍                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▍                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▍                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▍                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▍                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▌                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▌                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▌                | 6.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▌                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▋                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▋                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▋                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▋                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▋                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▊                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▊                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▊                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▊                | 6.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▊                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▉                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▉                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▉                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████▉                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████                | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▏               | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▏               | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▏               | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▏               | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▏               | 6.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▎               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▎               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▎               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▎               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▍               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▍               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▍               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▍               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▍               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▌               | 6.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▌               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▌               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▌               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▌               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▋               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▋               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▋               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▋               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▊               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▊               | 7.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▊               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▊               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▊               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▉               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▉               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▉               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████▉               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████               | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▏              | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▏              | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▏              | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▏              | 7.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▏              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▎              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▎              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▎              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▎              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▎              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▍              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▍              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▍              | 7.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▍              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▌              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▌              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▌              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▌              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▌              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▋              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▋              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▋              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▋              | 7.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▊              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▊              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▊              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▊              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▊              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▉              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▉              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▉              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████▉              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████              | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▏             | 7.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▏             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▏             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▏             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▎             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▎             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▎             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▎             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▎             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▍             | 7.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▍             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▍             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▍             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▌             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▌             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▌             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▌             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▌             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▋             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▋             | 7.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▋             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▋             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▋             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▊             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▊             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▊             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▊             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▉             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▉             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▉             | 7.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▉             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████▉             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████             | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▏            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▏            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▏            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▏            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▎            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▎            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▎            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▎            | 8.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▎            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▍            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▍            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▍            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▍            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▍            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▌            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▌            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▌            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▌            | 8.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▋            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▋            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▋            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▋            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▋            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▊            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▊            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▊            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▊            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▉            | 8.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▉            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▉            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▉            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████▉            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████            | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▏           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▏           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▏           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▏           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▏           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▎           | 8.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▎           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▎           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▎           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▍           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▍           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▍           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▍           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▍           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▌           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▌           | 8.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▌           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▌           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▋           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▋           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▋           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▋           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▋           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▊           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▊           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▊           | 8.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▊           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |████████████████████▉           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████           | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▏          | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▏          | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▏          | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▏          | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▏          | 8.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▎          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▎          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▎          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▎          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▍          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▍          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▍          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▍          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▍          | 9.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▌          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▌          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▌          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▌          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▌          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▋          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▋          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▋          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▋          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▊          | 9.1 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▊          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▊          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▊          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▊          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▉          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▉          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▉          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |█████████████████████▉          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.2 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████          | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▏         | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▏         | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▏         | 9.3 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▏         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▏         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▎         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▎         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▎         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▎         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▎         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▍         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▍         | 9.4 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▍         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▍         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▌         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▌         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▌         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▌         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▌         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▋         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▋         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▋         | 9.5 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▋         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▊         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▊         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▊         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▊         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▊         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▉         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▉         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▉         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |██████████████████████▉         | 9.6 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████         | 9.7 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▏        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▏        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▏        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▏        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▎        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▎        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▎        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▎        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▎        | 9.8 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▍        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▍        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▍        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▍        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▌        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▌        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▌        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▌        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▌        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▋        | 9.9 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▋        | 10.0 MB 57.6 MB/s eta 0:00:01',
 '\x1b[K     |███████████████████████▋        | 10.0 MB 57.6 MB/s eta 0:00:01',
 ...]

```
</div>
---
## Dataset preparation

### Data loading

We will be using CIFAR-10 as it is a nice baseline for our task. Because it has been used
for several other models, we can compare our results with other methods.
For the sake of time, we will only use 30% of the dataset, or around 18000 images for
this experiment. 15000 will be unlabeled images used during the VicReg process, and only
3000 labeled images will be used to train our linear evaluation model. Because of this,
we will see subpar results from our model. Try running this project in an interactive
notebook while changing that `DATASET_PERCENTAGE` constant to be higher.


```python
# Batch size of dataset
BATCH_SIZE = 512
# Width and height of image
IMAGE_SIZE = 32

[
    (train_features, train_labels),
    (test_features, test_labels),
] = keras.datasets.cifar10.load_data()

DATASET_PERCENTAGE = 0.3
train_features = train_features[: int(len(train_features) * DATASET_PERCENTAGE)]
test_features = test_features[: int(len(test_features) * DATASET_PERCENTAGE)]
train_labels = train_labels[: int(len(train_labels) * DATASET_PERCENTAGE)]
test_labels = test_labels[: int(len(test_labels) * DATASET_PERCENTAGE)]

train_features = train_features / 255.0
test_features = test_features / 255.0
```

<div class="k-default-codeblock">
```
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 2s 0us/step
170508288/170498071 [==============================] - 2s 0us/step

```
</div>
### Augmentation

VicReg uses the same augmentation pipeline as both Barlow Twins and BYOL. These
augmentations occur probabilistically, which allows for even more variation to help the
model learn.
<details>
<summary>Augmentation Pipeline Details</summary>
The pipeline is as follows:
* ***Random Crop***: We crop a random part of the image out. This resulting cropped image
is between 75% and 100% of the image size. Then, the cropped image is resized to the
original image width and height.
* ***Random Horizontal Flip*** (*50%*): There is a *50%* probability that the image will
be flipped horizontally
* ***Random Color Jitter*** (*80%*): There is an *80%* probability that the image will be
discolored. This process includes:
  * Random brightness (additive), ranging from `-0.8` to `+0.8`.
  * Random contrast (multiplicative), ranging from `0.4` to `1.6`
  * Random saturation (multiplicative), ranging from `0.4` to `1.6`
  * Random hue (multiplicative), ranging from `0.8` to `1.2`
* ***Random Greyscale*** (*20%*)
* ***Random Gaussian Blur***(*20%*): The blur amount σ ranges from `0` to
`1`
* ***Random Solarization***(*20%*): Solarization is when very low pixels get
inverted to do to irregularities in the camera. The solarization threshold for this
pipeline is `10`. If a pixel(not normalized) is below `10`, it will be
flipped to `255-pixel`.
Instead of implementing these pipelines ourselves, Tensorflow Similarity has
a collection of augmenters that we can use instead. In this case, we will be
using the pipeline function
`tensorflow_similarity.augmenters.barlow.augment_barlow` that takes in an image
and returns an augmented version using these transforms.
</details>
<details>
<summary> Dataset method </summary>
We'll use this function in the `tf.data.Dataset` API due to its ease of
use when batching and mapping. However, Tensorflow Similarity offers a simpler method
with it's augmenter library.
You can use `tensorflow_similarity.augmenters.BarlowAugmenter()` as a callable.
However, be aware that it *does* load the dataset into RAM, and you may have to
handle extra preprocessing (like batching) separately.
</details>


```python
# Saves a few minutes of performance - disables intra-op parallelism
performance_options = tf.data.Options()
performance_options.threading.max_intra_op_parallelism = 1

# Adding image width and height to augmenter
configed_augmenter = partial(
    tensorflow_similarity.augmenters.barlow.augment_barlow,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
)


def make_version():
    augment_version = (
        tf.data.Dataset.from_tensor_slices(train_features)
        .map(configed_augmenter, tf.data.AUTOTUNE)
        .shuffle(1000, seed=1024)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .with_options(performance_options)
    )

    return augment_version


augment_version_a, augment_version_b = make_version(), make_version()
augment_versions = tf.data.Dataset.zip(
    (augment_version_a, augment_version_b)
).with_options(performance_options)
```

We can use `tensorflow_similarity.visualization.visualize_views` to check out a
few sample images. Let's verify that each pair of images have a different set of
transforms.


```python
sample = next(iter(augment_versions))

print("Augmented Views")
tensorflow_similarity.visualization.visualize_views(
    sample, num_imgs=20, views_per_col=4, max_pixel_value=1.0, fig_size=(10, 10)
)
```

<div class="k-default-codeblock">
```
Augmented Views

```
</div>
![png](/img/examples/vision/vicreg/vicreg_17_1.png)


---
## Model Training

### Architecture

VicReg - Like Barlow Twins, requires a backbone (encoder) and a projector. The
projector is responsible for creating the embeddings that represent the dataset
as a whole.
We will be using a ResNet-18 with an output of length 512 and attach that to a
projector which will return embeddings of length 5000. The projector takes the
output of the backbone, and applies a series of Dense, Batch Normalization, and
Relu transformations.
![Model Structure](https://i.imgur.com/GuVyyJW.png)

### Loss

VicReg differs from other self-supervised methods like Barlow Twins due to its
unique way of calculating the loss function. It checks the variance, covariance,
and similarity between the embeddings per each image. We will be using
`tensorflow_similarity.losses.Vicreg()` to do this for us.
![Vicreg Loss](https://i.imgur.com/9Xa6tYD.png)
When the image says to "minimize the similarity", it means to minimize the
mean squared error.
<details>
<summary> Details about VicReg Loss </summary>
The VicReg loss aims to:
* ***Maximize*** the **variance** between corresponding elements of *different*
embeddings within a
batch. The notion is that different images should have different representations
from other ones. One way to measure this is by taking the variance, which checks
how varied, or scattered a dataset is. In this case, if the variance is high, we
can assume that the embeddings for different images are going to be different.
* ***Minimize*** the internal **covariance** of each embedding in the batch.
Covariance is when the individual values in the embeddings "correlate" with each
other. For example, if in an embedding there are two different variables that
always have the same value with each other, we say they are covariant. This is
bad because we want our embeddings to carry as much information about the
dataset as possible, so that downstream tasks have a lot more to work with. If
two different values in the embedding share correlations with each other, we
wouldn't need two separate values; we can just have one embedding that carries
both of their information together. Having two embeddings that always carry the
same information is *redundant*, and we want to remove this redundancy to get
the maximum information we can from these embeddings.
* ***Minimize*** the **distance** between embeddings that are for the same image. Two
similar images must have similar embeddings. To check this we can just use the
Mean Squared Error to find the distance between them.
Each of these losses are weighted summed with each other to get one loss number
</details>
<details>
<summary> VicReg pseudocode of variance, covariance, similarity </summary>
Variance Pseudocode:
```
z = mean_center(z)
std_z = sqrt(var(a, axis=0) + SMALL_CONSTANT)
std_z = mean(max(std_z, 0))
*same for z' and std_z'*
std_loss = average(std_z, std_z')
```
Similarity Pseudocode:
```
sim_loss = mse(z, z')
```
Covariance Pseudocode:
```
z = mean_center(z)
cov_loss_z = mm(transpose(z), z).get_off_diagonal()
cov_loss_z = sum(cov_loss_z) / embedding_size
*do same for z' and cov_loss_z'*
cov_loss = cov_loss_z + cov_loss_z'
```
</details>

### Implementation

We will be using Tensorflow Similarity's `ResNet18Sim` as our backbone and will implement
our custom projector. The backbone and projector will be combined via `ContrastiveModel`,
an API that manages our model composition and training
loop.


```python
# Code for defining our projector
def projector_layers(input_size, n_dense_neurons) -> keras.Model:
    """projector_layers method.
    Builds the projector network for VicReg, which are a series of Dense,
    BatchNormalization, and ReLU layers stacked on top of each other.
    Returns:
        returns the projector layers for VicReg
    """

    # number of dense neurons in the projector
    input_layer = tf.keras.layers.Input(input_size)

    # intermediate layers of the projector network
    n_layers = 2
    for i in range(n_layers):
        dense = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{i}")
        x = dense(input_layer) if i == 0 else dense(x)
        x = tf.keras.layers.BatchNormalization(name=f"projector_bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"projector_relu_{i}")(x)

    x = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{n_layers}")(x)

    model = keras.Model(input_layer, x)
    return model


backbone = tensorflow_similarity.architectures.ResNet18Sim(
    (IMAGE_SIZE, IMAGE_SIZE, 3), embedding_size=512
)
projector = projector_layers(backbone.output.shape[-1], n_dense_neurons=5000)

model = tensorflow_similarity.models.ContrastiveModel(
    backbone=backbone,
    projector=projector,
    algorithm="barlow",  # VicReg uses same architecture + training loop as Barlow Twins
)

# LAMB optimizer converges faster than ADAM or SGD when using large batch sizes.
optimizer = tfa.optimizers.LAMB()
loss = tensorflow_similarity.losses.VicReg()
model.compile(optimizer=optimizer, loss=loss)

# Expected training time: 1 hour
history = model.fit(augment_versions, epochs=75)
plt.plot(history.history["loss"])
plt.show()
```

<div class="k-default-codeblock">
```
Epoch 1/75
29/29 [==============================] - 56s 480ms/step - loss: 30.1285
Epoch 2/75
29/29 [==============================] - 15s 477ms/step - loss: 23.3109
Epoch 3/75
29/29 [==============================] - 15s 478ms/step - loss: 21.7725
Epoch 4/75
29/29 [==============================] - 15s 477ms/step - loss: 21.4164
Epoch 5/75
29/29 [==============================] - 15s 478ms/step - loss: 21.1626
Epoch 6/75
29/29 [==============================] - 15s 478ms/step - loss: 20.9882
Epoch 7/75
29/29 [==============================] - 15s 482ms/step - loss: 20.8279
Epoch 8/75
29/29 [==============================] - 15s 479ms/step - loss: 20.6876
Epoch 9/75
29/29 [==============================] - 15s 481ms/step - loss: 20.5789
Epoch 10/75
29/29 [==============================] - 15s 480ms/step - loss: 20.4825
Epoch 11/75
29/29 [==============================] - 15s 480ms/step - loss: 20.3810
Epoch 12/75
29/29 [==============================] - 15s 480ms/step - loss: 20.3028
Epoch 13/75
29/29 [==============================] - 15s 481ms/step - loss: 20.2172
Epoch 14/75
29/29 [==============================] - 15s 480ms/step - loss: 20.1469
Epoch 15/75
29/29 [==============================] - 15s 482ms/step - loss: 20.0781
Epoch 16/75
29/29 [==============================] - 15s 480ms/step - loss: 20.0433
Epoch 17/75
29/29 [==============================] - 15s 484ms/step - loss: 19.9838
Epoch 18/75
29/29 [==============================] - 15s 481ms/step - loss: 19.9308
Epoch 19/75
29/29 [==============================] - 15s 482ms/step - loss: 19.8758
Epoch 20/75
29/29 [==============================] - 15s 483ms/step - loss: 19.8068
Epoch 21/75
29/29 [==============================] - 15s 482ms/step - loss: 19.7600
Epoch 22/75
29/29 [==============================] - 15s 481ms/step - loss: 19.6903
Epoch 23/75
29/29 [==============================] - 15s 481ms/step - loss: 19.6791
Epoch 24/75
29/29 [==============================] - 16s 482ms/step - loss: 19.6296
Epoch 25/75
29/29 [==============================] - 15s 481ms/step - loss: 19.5968
Epoch 26/75
29/29 [==============================] - 15s 482ms/step - loss: 19.5917
Epoch 27/75
29/29 [==============================] - 15s 482ms/step - loss: 19.5282
Epoch 28/75
29/29 [==============================] - 15s 483ms/step - loss: 19.4809
Epoch 29/75
29/29 [==============================] - 15s 482ms/step - loss: 19.4508
Epoch 30/75
29/29 [==============================] - 15s 481ms/step - loss: 19.4637
Epoch 31/75
29/29 [==============================] - 15s 484ms/step - loss: 19.4135
Epoch 32/75
29/29 [==============================] - 15s 482ms/step - loss: 19.3691
Epoch 33/75
29/29 [==============================] - 15s 481ms/step - loss: 19.3314
Epoch 34/75
29/29 [==============================] - 15s 482ms/step - loss: 19.3100
Epoch 35/75
29/29 [==============================] - 15s 482ms/step - loss: 19.2882
Epoch 36/75
29/29 [==============================] - 15s 483ms/step - loss: 19.2543
Epoch 37/75
29/29 [==============================] - 15s 482ms/step - loss: 19.2335
Epoch 38/75
29/29 [==============================] - 15s 483ms/step - loss: 19.2171
Epoch 39/75
29/29 [==============================] - 15s 482ms/step - loss: 19.1670
Epoch 40/75
29/29 [==============================] - 15s 490ms/step - loss: 19.1639
Epoch 41/75
29/29 [==============================] - 15s 482ms/step - loss: 19.1320
Epoch 42/75
29/29 [==============================] - 15s 482ms/step - loss: 19.0959
Epoch 43/75
29/29 [==============================] - 15s 482ms/step - loss: 19.0985
Epoch 44/75
29/29 [==============================] - 15s 481ms/step - loss: 19.0551
Epoch 45/75
29/29 [==============================] - 15s 480ms/step - loss: 19.0615
Epoch 46/75
29/29 [==============================] - 15s 483ms/step - loss: 19.0310
Epoch 47/75
29/29 [==============================] - 15s 481ms/step - loss: 19.0374
Epoch 48/75
29/29 [==============================] - 15s 487ms/step - loss: 19.0326
Epoch 49/75
29/29 [==============================] - 15s 481ms/step - loss: 18.9722
Epoch 50/75
29/29 [==============================] - 15s 481ms/step - loss: 18.9463
Epoch 51/75
29/29 [==============================] - 15s 481ms/step - loss: 18.9157
Epoch 52/75
29/29 [==============================] - 15s 482ms/step - loss: 18.9280
Epoch 53/75
29/29 [==============================] - 15s 483ms/step - loss: 18.8804
Epoch 54/75
29/29 [==============================] - 15s 482ms/step - loss: 18.8796
Epoch 55/75
29/29 [==============================] - 15s 488ms/step - loss: 18.8789
Epoch 56/75
29/29 [==============================] - 15s 482ms/step - loss: 18.8454
Epoch 57/75
29/29 [==============================] - 15s 482ms/step - loss: 18.8507
Epoch 58/75
29/29 [==============================] - 15s 481ms/step - loss: 18.8068
Epoch 59/75
29/29 [==============================] - 15s 481ms/step - loss: 18.7941
Epoch 60/75
29/29 [==============================] - 15s 482ms/step - loss: 18.7627
Epoch 61/75
29/29 [==============================] - 15s 482ms/step - loss: 18.7858
Epoch 62/75
29/29 [==============================] - 15s 482ms/step - loss: 18.7324
Epoch 63/75
29/29 [==============================] - 15s 491ms/step - loss: 18.7217
Epoch 64/75
29/29 [==============================] - 15s 481ms/step - loss: 18.7395
Epoch 65/75
29/29 [==============================] - 15s 482ms/step - loss: 18.7169
Epoch 66/75
29/29 [==============================] - 15s 480ms/step - loss: 18.6908
Epoch 67/75
29/29 [==============================] - 15s 482ms/step - loss: 18.6712
Epoch 68/75
29/29 [==============================] - 15s 483ms/step - loss: 18.6613
Epoch 69/75
29/29 [==============================] - 15s 480ms/step - loss: 18.6357
Epoch 70/75
29/29 [==============================] - 15s 483ms/step - loss: 18.6181
Epoch 71/75
29/29 [==============================] - 15s 481ms/step - loss: 18.6062
Epoch 72/75
29/29 [==============================] - 15s 482ms/step - loss: 18.5946
Epoch 73/75
29/29 [==============================] - 15s 481ms/step - loss: 18.6312
Epoch 74/75
29/29 [==============================] - 15s 481ms/step - loss: 18.5810
Epoch 75/75
29/29 [==============================] - 15s 483ms/step - loss: 18.5929

```
</div>
![png](/img/examples/vision/vicreg/vicreg_25_1.png)


---
## Evaluation

### Evaluation Method

We will use *linear evaluation* to see how well our model learned embeddings.
This is where we freeze our trained backbone and projector, and just add a
single Dense + Softmax layer. Then, we train our model using the test images and
labels. Because we took 30% of CIFAR-10 when sampling, we are only training this
model with 3000 labeled images. However, remember that we trained the backbone
and projector using 12000 images, though they were unlabeled.

### Code


```python

def gen_lin_ds(features, labels):
    ds = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(1000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


xy_ds = gen_lin_ds(train_features, train_labels)
test_ds = gen_lin_ds(test_features, test_labels)

evaluator = keras.models.Sequential(
    [
        model.backbone,
        model.projector,
        keras.layers.Dense(
            10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.02)
        ),
    ]
)

# Need to test the backbone
evaluator.layers[0].trainable = False
evaluator.layers[1].trainable = False

linear_optimizer = tfa.optimizers.LAMB()
evaluator.compile(
    optimizer=linear_optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

evaluator.fit(xy_ds, epochs=35, validation_data=test_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/35
29/29 [==============================] - 7s 129ms/step - loss: 2.7620 - accuracy: 0.1334 - val_loss: 2.6538 - val_accuracy: 0.1727
Epoch 2/35
29/29 [==============================] - 2s 78ms/step - loss: 2.5818 - accuracy: 0.2196 - val_loss: 2.4794 - val_accuracy: 0.2703
Epoch 3/35
29/29 [==============================] - 2s 77ms/step - loss: 2.4181 - accuracy: 0.3083 - val_loss: 2.3263 - val_accuracy: 0.3664
Epoch 4/35
29/29 [==============================] - 2s 77ms/step - loss: 2.2700 - accuracy: 0.3893 - val_loss: 2.1959 - val_accuracy: 0.4305
Epoch 5/35
29/29 [==============================] - 2s 77ms/step - loss: 2.1376 - accuracy: 0.4508 - val_loss: 2.0747 - val_accuracy: 0.4770
Epoch 6/35
29/29 [==============================] - 2s 78ms/step - loss: 2.0210 - accuracy: 0.4995 - val_loss: 1.9727 - val_accuracy: 0.5109
Epoch 7/35
29/29 [==============================] - 2s 77ms/step - loss: 1.9206 - accuracy: 0.5346 - val_loss: 1.8800 - val_accuracy: 0.5500
Epoch 8/35
29/29 [==============================] - 2s 78ms/step - loss: 1.8319 - accuracy: 0.5610 - val_loss: 1.8128 - val_accuracy: 0.5574
Epoch 9/35
29/29 [==============================] - 2s 77ms/step - loss: 1.7544 - accuracy: 0.5807 - val_loss: 1.7375 - val_accuracy: 0.5723
Epoch 10/35
29/29 [==============================] - 2s 77ms/step - loss: 1.6900 - accuracy: 0.5967 - val_loss: 1.6886 - val_accuracy: 0.5820
Epoch 11/35
29/29 [==============================] - 2s 78ms/step - loss: 1.6335 - accuracy: 0.6094 - val_loss: 1.6398 - val_accuracy: 0.5938
Epoch 12/35
29/29 [==============================] - 2s 77ms/step - loss: 1.5825 - accuracy: 0.6210 - val_loss: 1.6029 - val_accuracy: 0.5926
Epoch 13/35
29/29 [==============================] - 2s 78ms/step - loss: 1.5393 - accuracy: 0.6314 - val_loss: 1.5749 - val_accuracy: 0.6016
Epoch 14/35
29/29 [==============================] - 2s 77ms/step - loss: 1.5016 - accuracy: 0.6373 - val_loss: 1.5362 - val_accuracy: 0.6070
Epoch 15/35
29/29 [==============================] - 2s 78ms/step - loss: 1.4665 - accuracy: 0.6433 - val_loss: 1.5070 - val_accuracy: 0.6109
Epoch 16/35
29/29 [==============================] - 2s 77ms/step - loss: 1.4388 - accuracy: 0.6488 - val_loss: 1.4739 - val_accuracy: 0.6238
Epoch 17/35
29/29 [==============================] - 2s 77ms/step - loss: 1.4108 - accuracy: 0.6549 - val_loss: 1.4685 - val_accuracy: 0.6176
Epoch 18/35
29/29 [==============================] - 2s 77ms/step - loss: 1.3854 - accuracy: 0.6579 - val_loss: 1.4496 - val_accuracy: 0.6211
Epoch 19/35
29/29 [==============================] - 2s 77ms/step - loss: 1.3639 - accuracy: 0.6606 - val_loss: 1.4406 - val_accuracy: 0.6273
Epoch 20/35
29/29 [==============================] - 2s 76ms/step - loss: 1.3445 - accuracy: 0.6639 - val_loss: 1.4311 - val_accuracy: 0.6223
Epoch 21/35
29/29 [==============================] - 2s 77ms/step - loss: 1.3238 - accuracy: 0.6672 - val_loss: 1.4115 - val_accuracy: 0.6258
Epoch 22/35
29/29 [==============================] - 2s 77ms/step - loss: 1.3053 - accuracy: 0.6708 - val_loss: 1.3936 - val_accuracy: 0.6285
Epoch 23/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2916 - accuracy: 0.6722 - val_loss: 1.3765 - val_accuracy: 0.6336
Epoch 24/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2751 - accuracy: 0.6758 - val_loss: 1.3788 - val_accuracy: 0.6281
Epoch 25/35
29/29 [==============================] - 2s 80ms/step - loss: 1.2611 - accuracy: 0.6772 - val_loss: 1.3462 - val_accuracy: 0.6359
Epoch 26/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2466 - accuracy: 0.6793 - val_loss: 1.3566 - val_accuracy: 0.6320
Epoch 27/35
29/29 [==============================] - 2s 78ms/step - loss: 1.2355 - accuracy: 0.6805 - val_loss: 1.3302 - val_accuracy: 0.6387
Epoch 28/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2240 - accuracy: 0.6814 - val_loss: 1.3263 - val_accuracy: 0.6375
Epoch 29/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2130 - accuracy: 0.6831 - val_loss: 1.3252 - val_accuracy: 0.6398
Epoch 30/35
29/29 [==============================] - 2s 77ms/step - loss: 1.2008 - accuracy: 0.6846 - val_loss: 1.3132 - val_accuracy: 0.6383
Epoch 31/35
29/29 [==============================] - 2s 77ms/step - loss: 1.1910 - accuracy: 0.6853 - val_loss: 1.3085 - val_accuracy: 0.6352
Epoch 32/35
29/29 [==============================] - 2s 78ms/step - loss: 1.1817 - accuracy: 0.6869 - val_loss: 1.3051 - val_accuracy: 0.6387
Epoch 33/35
29/29 [==============================] - 2s 77ms/step - loss: 1.1727 - accuracy: 0.6889 - val_loss: 1.2863 - val_accuracy: 0.6418
Epoch 34/35
29/29 [==============================] - 2s 78ms/step - loss: 1.1639 - accuracy: 0.6900 - val_loss: 1.2888 - val_accuracy: 0.6379
Epoch 35/35
29/29 [==============================] - 2s 77ms/step - loss: 1.1550 - accuracy: 0.6909 - val_loss: 1.2795 - val_accuracy: 0.6418

<keras.callbacks.History at 0x7f07e9e4bdd0>

```
</div>
Our accuracy should be between 60%-63%. This shows that our VicReg model was
able to learn a lot from the dataset, and can get better results than just the
10% one may get with random guessing.
**Things To try**
* If you change `DATASET_PERCENTAGE` to 1, meaning that it would use all the
dataset, accuracy should increase to about 70%
* If the number of epochs is changed from 75 to 150, accuracy may also increase
by a few points as well.

---
## Conclusion

* VicReg is a method of self-supervised partially-contrastive learning to
generate high-quality embeddings that contain dataset relationships.
* Using VicReg on 30% of our dataset, out of which 80% is unlabeled, we can get
an accuracy of around 62% when freezing all layers except a small Dense layer
at the end.

* VicReg, and other similar algorithms, have several use cases.
  * Can be used in semi-supervised learning, as shown in this demo. This is
  where you have a lot of unlabeled data and very little labeled data. You can
  use the unlabeled data to generate embeddings to assist the labeled data when
  training.
* VicReg vs Barlow Twins (Predecessor)
  * VicReg performs similarly to Barlow Twins on CIFAR-10 and other
  Image classification datasets
  * However it significantly outperforms Barlow Twins on multi-modal tasks like
  Image-to-Text and Text-to-Image
  ![Table](https://i.imgur.com/GuWIssF.png)
