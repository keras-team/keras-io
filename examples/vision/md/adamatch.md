# Semi-supervision and domain adaptation with AdaMatch

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/06/19<br>
**Last modified:** 2026/05/12<br>
**Description:** Unifying semi-supervised learning and unsupervised domain adaptation with AdaMatch.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/adamatch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/adamatch.py)



---
## Introduction

In this example, we will implement the AdaMatch algorithm, proposed in
[AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732)
by Berthelot et al. It sets a new state-of-the-art in unsupervised domain adaptation (as of
June 2021). AdaMatch is particularly interesting because it
unifies semi-supervised learning (SSL) and unsupervised domain adaptation
(UDA) under one framework. It thereby provides a way to perform semi-supervised domain
adaptation (SSDA).

Before we proceed, let's review a few preliminary concepts underlying this example.

---
## Preliminaries

In **semi-supervised learning (SSL)**, we use a small amount of labeled data to
train models on a bigger unlabeled dataset. Popular semi-supervised learning methods
for computer vision include [FixMatch](https://arxiv.org/abs/2001.07685),
[MixMatch](https://arxiv.org/abs/1905.02249),
[Noisy Student Training](https://arxiv.org/abs/1911.04252), etc. You can refer to
[this example](https://keras.io/examples/vision/consistency_training/) to get an idea
of what a standard SSL workflow looks like.

In **unsupervised domain adaptation**, we have access to a source labeled dataset and
a target *unlabeled* dataset. Then the task is to learn a model that can generalize well
to the target dataset. The source and the target datasets vary in terms of distribution.
The following figure provides an illustration of this idea. In the present example, we use the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) as the source dataset, while the target dataset is
[SVHN](http://ufldl.stanford.edu/housenumbers/), which consists of images of house
numbers. Both datasets have various varying factors in terms of texture, viewpoint,
appearance, etc.: their domains, or distributions, are different from one
another.

![](https://i.imgur.com/dJFSJuT.png)

Popular domain adaptation algorithms in deep learning include
[Deep CORAL](https://arxiv.org/abs/1612.01939),
[Moment Matching](https://arxiv.org/abs/1812.01754), etc.

---
## Setup

To run this example, ensure you have the following dependencies installed:

```shell
pip install scipy pillow
```


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # "jax" or "torch"

import keras

keras.utils.set_random_seed(42)

import numpy as np
from keras import layers
from keras import ops
import scipy.io
from PIL import Image
```

---
## Prepare the data


```python
# MNIST
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = (
    keras.datasets.mnist.load_data()
)

# Add a channel dimension
mnist_x_train = np.expand_dims(mnist_x_train, -1)
mnist_x_test = np.expand_dims(mnist_x_test, -1)

# Convert the labels to one-hot encoded vectors
mnist_y_train = keras.utils.to_categorical(mnist_y_train, 10)


# SVHN
def load_svhn_data():
    path = keras.utils.get_file(
        "train_32x32.mat",
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    )
    data = scipy.io.loadmat(path)
    x = np.transpose(data["X"], (3, 0, 1, 2))
    return x


svhn_x_train = load_svhn_data()
```

---
## Define constants and hyperparameters


```python
RESIZE_TO = 32

SOURCE_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 3 * SOURCE_BATCH_SIZE  # Reference: Section 3.2
EPOCHS = 2
STEPS_PER_EPOCH = len(mnist_x_train) // SOURCE_BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

LEARNING_RATE = 0.03

WEIGHT_DECAY = 0.0005
INIT = "he_normal"
DEPTH = 28
WIDTH_MULT = 2

```

---
## Data Loading and Augmentation Utilities

For custom data loading and preprocessing in Keras 3, it is recommended to
use `keras.utils.PyDataset`. It ensures thread-safe data iteration and
multi-backend compatibility.

A standard element of SSL algorithms is to feed weakly and strongly augmented
versions of the same images to the model to ensure consistent predictions.
For strong augmentation, [RandAugment](https://arxiv.org/abs/1909.13719) is
used. For weak augmentation, we use horizontal flipping and random translation.


```python

class AdaMatchDataset(keras.utils.PyDataset):
    def __init__(self, source_x, source_y, target_x, target_size=32, **kwargs):
        """
        Dataset for AdaMatch training.
        Performs resize-and-pad on source images to preserve aspect ratio,
        then tiles them to 3 channels if needed.
        """
        super().__init__(**kwargs)
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_size = target_size

    def __len__(self):
        return STEPS_PER_EPOCH

    def resize_and_pad(self, images):
        """
        Resize images to target_size x target_size while preserving aspect ratio.
        Pads with zeros if necessary.
        """
        resized_images = []
        for img in images:
            img = np.squeeze(img)
            if img.ndim == 2:
                img = np.expand_dims(img, -1)  # grayscale to (H,W,1)
            h, w = img.shape[:2]
            scale = self.target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            if img.shape[2] == 1:
                pil_img = Image.fromarray(img[:, :, 0])
            else:
                pil_img = Image.fromarray(img.astype(np.uint8))
            pil_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized = (
                np.expand_dims(np.array(pil_resized), -1)
                if img.shape[2] == 1
                else np.array(pil_resized)
            )
            # Pad
            pad_h = (self.target_size - new_h) // 2
            pad_w = (self.target_size - new_w) // 2
            padded = np.zeros(
                (self.target_size, self.target_size, img.shape[2]), dtype=img.dtype
            )
            padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = resized
            resized_images.append(padded)
        return np.array(resized_images, dtype="float32")

    def __getitem__(self, idx):
        s_idx = np.random.choice(len(self.source_x), SOURCE_BATCH_SIZE)
        t_idx = np.random.choice(len(self.target_x), TARGET_BATCH_SIZE)
        s_imgs = self.resize_and_pad(self.source_x[s_idx])
        s_imgs = np.tile(s_imgs, (1, 1, 1, 3))

        t_imgs = self.target_x[t_idx].astype("float32")

        return (s_imgs, self.source_y[s_idx].astype("float32"), t_imgs), np.zeros(
            (SOURCE_BATCH_SIZE,)
        )


train_ds = AdaMatchDataset(mnist_x_train, mnist_y_train, svhn_x_train)
```

---
## Subclassed model for AdaMatch training

The figure below presents the overall workflow of AdaMatch (taken from the
[original paper](https://arxiv.org/abs/2106.04732)):

![](https://i.imgur.com/1QsEm2M.png)

Here's a brief step-by-step breakdown of the workflow:

1. We first retrieve the weakly and strongly augmented pairs of images from the source and
target datasets.
2. We prepare two concatenated copies:
    i. One where both pairs are concatenated.
    ii. One where only the source data image pair is concatenated.
3. We run two forward passes through the model:
    i. The first forward pass uses the concatenated copy obtained from **2.i**. In
this forward pass, the [Batch Normalization](https://arxiv.org/abs/1502.03167) statistics
are updated.
    ii. In the second forward pass, we only use the concatenated copy obtained from **2.ii**.
    Batch Normalization layers are run in inference mode.
4. The respective logits are computed for both the forward passes.
5. The logits go through a series of transformations, introduced in the paper (which
we will discuss shortly).
6. We compute the loss and update the gradients of the underlying model.


```python

class AdaMatch(keras.Model):
    def __init__(self, model, total_steps, tau=0.9, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tau = tau
        self.total_steps = total_steps
        self.current_step = keras.Variable(0.0, dtype="float32", trainable=False)
        self.loss_tracker = keras.metrics.Mean(name="loss")

        self.weak_augment = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
            ]
        )
        rand_aug = layers.RandAugment(value_range=(0, 255), num_ops=2, factor=0.5)
        self.strong_aug = rand_aug

    def build(self, input_shape):
        self.model.build(input_shape[0])  # input_shape[0] is the source_imgs shape
        self.weak_augment.build(input_shape[0])
        super().build(input_shape)

    @property
    def metrics(self):
        return [self.loss_tracker]

    # This is a warmup schedule to update the weight of the
    # loss contributed by the target unlabeled samples. More
    # on this in the text.
    def compute_mu(self):
        pi = ops.cast(np.pi, "float32")
        return (
            0.5
            - ops.cos(ops.minimum(pi, (2 * pi * self.current_step) / self.total_steps))
            / 2
        )

    def call(self, inputs, training=False):
        source_imgs, _, _ = inputs
        return self.model(source_imgs, training=training)

    def compute_loss(self, x=None, y_true=None, y_pred=None, sample_weight=None):
        source_imgs, source_labels, target_imgs = x

        source_ds_w = self.weak_augment(source_imgs, training=True)
        source_ds_s = self.strong_aug(source_imgs, training=True)

        target_ds_w = self.weak_augment(target_imgs, training=True)
        target_ds_s = self.strong_aug(target_imgs, training=True)

        combined_images = ops.concatenate(
            [source_ds_w, source_ds_s, target_ds_w, target_ds_s], axis=0
        )

        combined_source = ops.concatenate([source_ds_w, source_ds_s], axis=0)
        ## Forward passes ##
        combined_logits = self.model(combined_images, training=True)
        z_d_prime_source = self.model(
            combined_source, training=False
        )  # No BatchNorm update.

        total_source = ops.shape(combined_source)[0]
        z_prime_source = combined_logits[:total_source]

        ## 1. Random logit interpolation for the source images ##
        lambd = keras.random.uniform(ops.shape(z_prime_source), 0, 1)

        final_source_logits = (lambd * z_prime_source) + (
            (1 - lambd) * z_d_prime_source
        )

        ## 2. Distribution alignment (only consider weakly augmented images) ##
        # Compute softmax for logits of the WEAKLY augmented SOURCE images.
        y_hat_source_w = ops.softmax(final_source_logits[:SOURCE_BATCH_SIZE])

        # Extract logits for the WEAKLY augmented TARGET images and compute softmax.
        logits_target = combined_logits[total_source:]
        logits_target_w = logits_target[:TARGET_BATCH_SIZE]
        y_hat_target_w = ops.softmax(logits_target_w)

        source_dist = ops.mean(y_hat_source_w, axis=0) + 1e-8
        target_dist = ops.mean(y_hat_target_w, axis=0) + 1e-8

        # Align the target label distribution to that of the source.
        expectation_ratio = source_dist / target_dist
        expectation_ratio = ops.clip(expectation_ratio, 0.1, 10.0)
        y_tilde_target_w = ops.stop_gradient(
            ops.normalize(y_hat_target_w * expectation_ratio, axis=-1, order=1)
        )

        ## 3. Relative confidence thresholding ##
        row_wise_max = ops.max(y_hat_source_w, axis=-1)
        c_tau = self.tau * ops.mean(row_wise_max)
        mask = ops.cast(ops.max(y_tilde_target_w, axis=-1) >= c_tau, "float32")

        loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)

        ## Compute losses (pay attention to the indexing) ##
        source_loss = loss_func(
            source_labels, final_source_logits[:SOURCE_BATCH_SIZE]
        ) + loss_func(
            source_labels, final_source_logits[SOURCE_BATCH_SIZE:total_source]
        )

        target_loss = ops.mean(
            keras.losses.categorical_crossentropy(
                y_tilde_target_w,
                logits_target[TARGET_BATCH_SIZE:],
                from_logits=True,
            )
            * mask
        )

        target_loss_weight = self.compute_mu()  # Compute weight for the target loss
        total_loss = source_loss + (
            target_loss_weight * target_loss
        )  # Update current training step for the scheduler

        self.current_step.assign_add(1.0)

        self.loss_tracker.update_state(total_loss)
        return total_loss

```

The authors introduce three improvements in the paper:

* In AdaMatch, we perform two forward passes, and only one of them is respsonsible for
updating the Batch Normalization statistics. This is done to account for distribution
shifts in the target dataset. In the other forward pass, we only use the source sample,
and the Batch Normalization layers are run in inference mode. Logits for the source
samples (weakly and strongly augmented versions) from these two passes are slightly
different from one another because of how Batch Normalization layers are run. Final
logits for the source samples are computed by linearly interpolating between these two
different pairs of logits. This induces a form of consistency regularization. This step
is referred to as **random logit interpolation**.
* **Distribution alignment** is used to align the source and target label distributions.
This further helps the underlying model learn *domain-invariant representations*. In case
of unsupervised domain adaptation, we don't have access to any labels of the target
dataset. This is why pseudo labels are generated from the underlying model.
* The underlying model generates pseudo-labels for the target samples. It's likely that
the model would make faulty predictions. Those can propagate back as we make progress in
the training, and hurt the overall performance. To compensate for that, we filter the
high-confidence predictions based on a threshold (hence the use of `mask` inside
`compute_loss_target()`). In AdaMatch, this threshold is relatively adjusted which is why
it is called **relative confidence thresholding**.

For more details on these methods and to know how each of them contribute please refer to
[the paper](https://arxiv.org/abs/2106.04732).

**About `compute_mu()`**:

Rather than using a fixed scalar quantity, a varying scalar is used in AdaMatch. It
denotes the weight of the loss contibuted by the target samples. Visually, the weight
scheduler look like so:

![](https://i.imgur.com/dG7i9uH.png)

This scheduler increases the weight of the target domain loss from 0 to 1 for the first
half of the training. Then it keeps that weight at 1 for the second half of the training.

---
## Instantiate a Wide-ResNet-28-2

The authors use a [WideResNet-28-2](https://arxiv.org/abs/1605.07146) for the dataset
pairs we are using in this example. Most of the following code has been referred from
[this script](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py). Note
that the following model has a scaling layer inside it that scales the pixel values to
[0, 1].


```python

def wide_basic(x, n_input_plane, n_output_plane, stride):

    # Shortcut connection: identity function or 1x1
    # convolutional
    #  (depends on difference between input & output shape - this
    #   corresponds to whether we are using the first block in
    #   each
    #   group; see `block_series()`).

    if n_input_plane != n_output_plane:

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        shortcut = layers.Conv2D(
            n_output_plane,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        )(x)

        convs = layers.Conv2D(
            n_output_plane,
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        )(x)

    else:

        shortcut = x

        convs = layers.BatchNormalization()(x)
        convs = layers.Activation("relu")(convs)

        convs = layers.Conv2D(
            n_output_plane,
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        )(convs)

    convs = layers.BatchNormalization()(convs)
    convs = layers.Activation("relu")(convs)

    convs = layers.Conv2D(
        n_output_plane,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=INIT,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(convs)

    return layers.Add()([convs, shortcut])


def get_network():
    n = (DEPTH - 4) // 6
    stages = [16, 16 * WIDTH_MULT, 32 * WIDTH_MULT, 64 * WIDTH_MULT]
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(
        stages[0],
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=INIT,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)

    for i in range(1, 4):
        x = wide_basic(x, stages[i - 1], stages[i], stride=(1 if i == 1 else 2))
        for _ in range(n - 1):
            x = wide_basic(x, stages[i], stages[i], stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        10,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)

    return keras.Model(inputs, outputs)

```

We can now instantiate a Wide ResNet model like so. Note that the purpose of using a
Wide ResNet here is to keep the implementation as close to the original one
as possible.


```python
wrn_model = get_network()

print(f"Model has {wrn_model.count_params()/1e6} Million parameters.")
```

<div class="k-default-codeblock">
```
Model has 1.471226 Million parameters.
```
</div>

---
## Instantiate AdaMatch model and compile it


```python
reduce_lr = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, TOTAL_STEPS, 0.25)
optimizer = keras.optimizers.Adam(reduce_lr)
adamatch_trainer = AdaMatch(model=wrn_model, total_steps=TOTAL_STEPS)

adamatch_trainer.compile(optimizer=optimizer)
sample_batch = train_ds[0][0]
_ = adamatch_trainer(sample_batch)
```

---
## Model training


```python
adamatch_trainer.fit(train_ds, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/2

937/937 ━━━━━━━━━━━━━━━━━━━━ 4385s 5s/step - loss: 2604439552.0000

Epoch 2/2

937/937 ━━━━━━━━━━━━━━━━━━━━ 4841s 5s/step - loss: 1.0786

<keras.src.callbacks.history.History at 0x31fd8d550>
```
</div>

---
## Evaluation on the target and source test sets


```python
# Compile the AdaMatch model to yield accuracy.
adamatch_trained_model = adamatch_trainer.model
adamatch_trained_model.compile(metrics=[keras.metrics.SparseCategoricalAccuracy()])

test_path = keras.utils.get_file(
    "test_32x32.mat",
    "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
)

# Score on the target test set.
svhn_test = scipy.io.loadmat(test_path)

x_test = np.transpose(svhn_test["X"], (3, 0, 1, 2)).astype("float32")
y_test = svhn_test["y"].flatten()
y_test[y_test == 10] = 0
results = adamatch_trained_model.evaluate(x_test, y_test, verbose=0)
accuracy = results[1]

print(f"SVHN Accuracy: {accuracy *100:.2f}%")
```

<div class="k-default-codeblock">
```
SVHN Accuracy: 26.13%
```
</div>

With more training, this score improves. When this same network is trained with
standard classification objective, it yields an accuracy of **7.20%** which is
significantly lower than what we got with AdaMatch. You can check out
[this notebook](https://colab.research.google.com/github/sayakpaul/AdaMatch-TF/blob/main/Vanilla_WideResNet.ipynb)
to learn more about the hyperparameters and other experimental details.


```python

# Utility function for preprocessing the source test set.
def prepare_test_ds_source(images):
    resizer = layers.Resizing(RESIZE_TO, RESIZE_TO)
    images = images.astype("float32")
    images = resizer(images)
    images = ops.tile(images, (1, 1, 1, 3))
    return images


x_source_test = prepare_test_ds_source(mnist_x_test)
results = adamatch_trained_model.evaluate(
    x_source_test,
    mnist_y_test,
    batch_size=TARGET_BATCH_SIZE,
    verbose=0,
)

accuracy = results[1]

print(f"Accuracy on source test set: {accuracy * 100:.2f}%")
```

<div class="k-default-codeblock">
```
Accuracy on source test set: 96.20%
```
</div>

You can reproduce the results by using these
[model weights](https://github.com/sayakpaul/AdaMatch-TF/releases/tag/v1.0.0).

**Example available on HuggingFace**
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AdaMatch%20Domain%20Adaption-black.svg)](https://huggingface.co/keras-io/adamatch-domain-adaption) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-AdaMatch%20Domain%20Adaption-black.svg)](https://huggingface.co/spaces/keras-io/adamatch-domain-adaption) |
