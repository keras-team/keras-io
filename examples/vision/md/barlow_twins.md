# Barlow Twins for Contrastive SSL

**Author:** [Abhiraam Eranti](https://github.com/dewball345)<br>
**Date created:** 11/4/21<br>
**Last modified:** 26/04/02<br>
**Description:** A keras implementation of Barlow Twins (contrastive SSL with redundancy reduction).


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/barlow_twins.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/barlow_twins.py)



---
## Introduction

Self-supervised learning (SSL) is a relatively novel technique in which a model
learns from unlabeled data, and is often used when the data is corrupted or
if there is very little of it. A practical use for SSL is to create
intermediate embeddings that are learned from the data. These embeddings are
based on the dataset itself, with similar images having similar embeddings, and
vice versa. They are then attached to the rest of the model, which uses those
embeddings as information and effectively learns and makes predictions properly.
These embeddings, ideally, should contain as much information and insight about
the data as possible, so that the model can make better predictions. However,
a common problem that arises is that the model creates embeddings that are
redundant. For example, if two images are similar, the model will create
embeddings that are just a string of 1's, or some other value that
contains repeating bits of information. This is no better than a one-hot
encoding or just having one bit as the model’s representations; it defeats the
purpose of the embeddings, as they do not learn as much about the dataset as
possible. For other approaches, the solution to the problem was to carefully
configure the model such that it tries not to be redundant.

Barlow Twins is a new approach to this problem; while other solutions mainly
tackle the first goal of invariance (similar images have similar embeddings),
the Barlow Twins method also prioritizes the goal of reducing redundancy.

It also has the advantage of being much simpler than other methods, and its
model architecture is symmetric, meaning that both twins in the model do the
same thing. It is also near state-of-the-art on imagenet, even exceeding methods
like SimCLR.

One disadvantage of Barlow Twins is that it is heavily dependent on
augmentation, suffering major performance decreases in accuracy without them.

TL, DR: Barlow twins creates representations that are:

*   Invariant.
*   Not redundant, and carry as much info about the dataset.

Also, it is simpler than other methods.

This notebook can train a Barlow Twins model and reach up to
64% validation accuracy on the CIFAR-10 dataset.

![image](https://i.imgur.com/G6LnEPT.png)

### High-Level Theory

The model takes two versions of the same image(with different augmentations) as
input. Then it takes a prediction of each of them, creating representations.
They are then used to make a cross-correlation matrix.

Cross-correlation matrix:
```
(pred_1.T @ pred_2) / batch_size
```

The cross-correlation matrix measures the correlation between the output
neurons in the two representations made by the model predictions of the two
augmented versions of data. Ideally, a cross-correlation matrix should look
like an identity matrix if the two images are the same.

When this happens, it means that the representations:

1.   Are invariant. The diagonal shows the correlation between each
representation's neurons and its corresponding augmented one. Because the two
versions come from the same image, the diagonal of the matrix should show that
there is a strong correlation between them. If the images are different, there
shouldn't be a diagonal.
2.   Do not show signs of redundancy. If the neurons show correlation with a
non-diagonal neuron, it means that it is not correctly identifying similarities
between the two augmented images. This means that it is redundant.

Here is a good way of understanding in pseudocode(information from the original
paper):

```
c[i][i] = 1
c[i][j] = 0

where:
  c is the cross-correlation matrix
  i is the index of one representation's neuron
  j is the index of the second representation's neuron
```

Taken from the original paper: [Barlow Twins: Self-Supervised Learning via Redundancy
Reduction](https://arxiv.org/abs/2103.03230)

### References

Paper:
[Barlow Twins: Self-Supervised Learning via Redundancy
Reduction](https://arxiv.org/abs/2103.03230)

Original Implementation:
 [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)

---
## Setup


```python
import os

# slightly faster improvements, on the first epoch 30 second decrease and a 1-2 second
# decrease in epoch time. Overall saves approx. 5 min of training time

# Allocates two threads for a gpu private which allows more operations to be
# done faster
os.environ["KERAS_BACKEND"] = "tensorflow" #or "jax" or "torch"

import keras
import numpy as np  # np.random.random
import matplotlib.pyplot as plt  # graphs
from keras import layers
from keras import random
from keras import losses
from keras import ops
from keras.utils import Sequence

# XLA optimization for faster performance(up to 10-15 minutes total time saved)
# tf.config.optimizer.set_jit(True)
```

---
## Load the CIFAR-10 dataset


```python
[
    (train_features, train_labels),
    (test_features, test_labels),
] = keras.datasets.cifar10.load_data()

train_features = train_features / 255.0
test_features = test_features / 255.0
```

---
## Necessary Hyperparameters


```python
# Batch size
BATCH_SIZE = 512

IMAGE_SIZE = 32
```

---
## Augmentation Utilities
The Barlow twins algorithm is heavily reliant on
Augmentation. One unique feature of the method is that sometimes, augmentations
probabilistically occur.

**Augmentations**

*   *RandomToGrayscale*: randomly applies grayscale to image 20% of the time
*   *RandomColorJitter*: randomly applies color jitter 80% of the time
*   *RandomFlip*: randomly flips image horizontally 50% of the time
*   *RandomResizedCrop*: randomly crops an image to a random size then resizes. This
happens 100% of the time
*   *RandomSolarize*: randomly applies solarization to an image 20% of the time
*   *RandomBlur*: randomly blurs an image 20% of the time


```python

class Augmentation(keras.layers.Layer):
    """Base augmentation class.

    Base augmentation class. Contains the random_execute method.

    Methods:
        random_execute: method that returns true or false based
          on a probability. Used to determine whether an augmentation
          will be run.
    """

    def __init__(self):
        super().__init__()

    def random_execute(self, prob: float) -> bool:
        """random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the
              probability.

        Returns:
            returns true or false based on the probability.
        """

        return random.uniform([], minval=0, maxval=1) < prob


class RandomToGrayscale:
    """RandomToGrayscale class.

    RandomToGrayscale class. Randomly makes an image
    grayscaled based on the random_execute method. There
    is a 20% chance that an image will be grayscaled.

    Methods:
        call: method that grayscales an image 20% of
          the time.
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() < self.prob:
            # average channels to get grayscale
            gray = np.mean(x, axis=-1, keepdims=True)
            x = np.repeat(gray, 3, axis=-1)
        return x


class RandomColorJitter(Augmentation):
    """RandomColorJitter class.

    RandomColorJitter class. Randomly adds color jitter to an image.
    Color jitter means to add random brightness, contrast,
    saturation, and hue to an image. There is a 80% chance that an
    image will be randomly color-jittered.

    Methods:
        call: method that color-jitters an image 80% of
          the time.
    """

    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() < self.prob:
            x = x + np.random.uniform(-0.2, 0.2)
            x = (x - 0.5) * np.random.uniform(0.8, 1.2) + 0.5
            gray = np.mean(x, axis=-1, keepdims=True)
            x = gray + (x - gray) * np.random.uniform(0.8, 1.2)
            x = np.clip(x, 0, 1)
        return x


class RandomFlip:
    """RandomFlip class.

    RandomFlip class. Randomly flips image horizontally. There is a 50%
    chance that an image will be randomly flipped.

    Methods:
        call: method that flips an image 50% of
          the time.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        """call function.

        Randomly flips the image.

        Arguments:
            x: a Tensor representing the image.

        Returns:
            returns a flipped version of the image 50% of the time
              and the original image 50% of the time.
        """

        if np.random.rand() < self.prob:
            # flip horizontally
            x = np.fliplr(x)
        return x


class RandomResizedCrop(Augmentation):
    def __init__(self, image_size):
        self.image_size = image_size
        self.resize_layer = layers.Resizing(image_size, image_size)

    def __call__(self, x):
        h, w, _ = x.shape
        rand_size = int(np.random.uniform(0.75 * self.image_size, self.image_size))
        top = np.random.randint(0, max(h - rand_size + 1, 1))
        left = np.random.randint(0, max(w - rand_size + 1, 1))
        crop = x[top : top + rand_size, left : left + rand_size, :]
        crop_resized = self.resize_layer(np.expand_dims(crop, 0))[0].numpy()
        return crop_resized


class RandomSolarize(Augmentation):
    """RandomSolarize class.

    RandomSolarize class. Randomly solarizes an image.
    Solarization is when pixels accidentally flip to an inverted state.

    Methods:
        call: method that does random solarization 20% of the time.
    """

    def __init__(self, prob=0.2, threshold=0.5):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, x):
        """call function.

        Randomly solarizes the image.

        Arguments:
            x: a Tensor representing the image.

        Returns:
            returns a solarized version of the image 20% of the time
              and the original image 80% of the time.
        """
        if np.random.rand() < self.prob:
            x = np.where(x < self.threshold, x, 1.0 - x)
        return x


class RandomBlur(Augmentation):
    """RandomBlur class.

    RandomBlur class. Randomly blurs an image.

    Methods:
        call: method that does random blur 20% of the time.
    """

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, x):
        """call function.

        Randomly solarizes the image.

        Arguments:
            x: a Tensor representing the image.

        Returns:
            returns a blurred version of the image 20% of the time
              and the original image 80% of the time.
        """
        if np.random.rand() < self.prob:
            sigma = np.random.rand() * 1.0
            x = ops.image.gaussian_blur(x, sigma=(sigma, sigma, 0))
        return x


class RandomAugmentor:
    """RandomAugmentor class.

    RandomAugmentor class. Chains all the augmentations into
    one pipeline.

    Attributes:
        image_size: An integer represing the width and height
          of the image. Designed to be used for square images.
        random_resized_crop: Instance variable representing the
          RandomResizedCrop layer.
        random_flip: Instance variable representing the
          RandomFlip layer.
        random_color_jitter: Instance variable representing the
          RandomColorJitter layer.
        random_blur: Instance variable representing the
          RandomBlur layer
        random_to_grayscale: Instance variable representing the
          RandomToGrayscale layer
        random_solarize: Instance variable representing the
          RandomSolarize layer

    Methods:
        call: chains layers in pipeline together
    """

    def __init__(self, image_size):
        self.random_resized_crop = RandomResizedCrop(image_size)
        self.random_flip = RandomFlip()
        self.random_color_jitter = RandomColorJitter()
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def __call__(self, x):
        x = self.random_resized_crop(x)
        x = self.random_flip(x)
        x = self.random_color_jitter(x)
        x = self.random_blur(x)
        x = self.random_to_grayscale(x)
        x = self.random_solarize(x)
        return np.clip(x, 0, 1)


bt_augmentor = RandomAugmentor(IMAGE_SIZE)
```

---
## Data Loading

A class that creates the barlow twins' dataset.

The dataset consists of two copies of each image, with each copy receiving different
augmentations.


```python

class BTDatasetCreator:
    """Barlow twins dataset creator class.

    BTDatasetCreator class. Responsible for creating the
    barlow twins' dataset.

    Attributes:
        options: data.Options needed to configure a setting
          that may improve performance.
        seed: random seed for shuffling. Used to synchronize two
          augmented versions.
        augmentor: augmentor used for augmentation.

    Methods:
        __call__: creates barlow dataset.
        augmented_version: creates 1 half of the dataset.
    """

    """Creates Barlow Twins augmentation datasets using pure Python Sequence."""

    def __init__(self, augmentor, batch_size=BATCH_SIZE, seed=1024):
        self.augmentor = augmentor
        self.batch_size = batch_size
        self.seed = seed

    def augmented_version(self, x):
        """Return a Sequence that yields two augmented versions of each batch."""

        class AugmentedSequence(Sequence):
            def __init__(self, x, augmentor, batch_size, seed):
                super().__init__()
                self.x = x
                self.augmentor = augmentor
                self.batch_size = batch_size
                self.seed = seed
                self.on_epoch_end()

            def __len__(self):
                return int(np.ceil(len(self.x) / self.batch_size))

            def __getitem__(self, index):
                batch = self.x[index * self.batch_size : (index + 1) * self.batch_size]
                a1 = np.stack([self.augmentor(xi) for xi in batch])
                a2 = np.stack([self.augmentor(xi) for xi in batch])
                return a1, a2

            def on_epoch_end(self):
                np.random.seed(self.seed)
                idx = np.arange(len(self.x))
                np.random.shuffle(idx)
                self.x = self.x[idx]

        return AugmentedSequence(x, self.augmentor, self.batch_size, self.seed)

    def __call__(self, x):
        """Return a zipped Sequence with two augmented views per batch."""
        seq1 = self.augmented_version(x)
        seq2 = self.augmented_version(x)

        class ZippedSequence(Sequence):
            def __init__(self, seq1, seq2):
                super().__init__()
                self.seq1 = seq1
                self.seq2 = seq2

            def __len__(self):
                return min(len(self.seq1), len(self.seq2))

            def __getitem__(self, index):
                a1, _ = self.seq1[index]
                a2, _ = self.seq2[index]
                batch_size = len(a1)
                sam = np.zeros((batch_size, 1))
                return (a1, a2), sam

            def on_epoch_end(self):
                self.seq1.on_epoch_end()
                self.seq2.on_epoch_end()

        return ZippedSequence(seq1, seq2)


bt_augmentor = RandomAugmentor(IMAGE_SIZE)
augment_versions = BTDatasetCreator(bt_augmentor)(train_features)
```

View examples of dataset.


```python
sample_augment_versions = iter(augment_versions)


def plot_values(batch):
    (a1, a2), _ = batch

    fig, axs = plt.subplots(3, 3, figsize=(6, 6))

    for i in range(3):
        for j in range(3):
            img = a1[3 * i + j]  # shape (32, 32, 3)
            axs[i][j].imshow(img)
            axs[i][j].axis("off")

    plt.tight_layout()
    plt.show()


plot_values(next(sample_augment_versions))
```


    
![png](/img/examples/vision/barlow_twins/barlow_twins_20_0.png)
    


---
## Pseudocode of loss and model
The following sections follow the original author's pseudocode containing both model and
loss functions(see diagram below). Also contains a reference of variables used.

![pseudocode](https://i.imgur.com/Tlrootj.png)

Reference:

```
y_a: first augmented version of original image.
y_b: second augmented version of original image.
z_a: model representation(embeddings) of y_a.
z_b: model representation(embeddings) of y_b.
z_a_norm: normalized z_a.
z_b_norm: normalized z_b.
c: cross correlation matrix.
c_diff: diagonal portion of loss(invariance term).
off_diag: off-diagonal portion of loss(redundancy reduction term).
```

---
## BarlowLoss: barlow twins model's loss function

Barlow Twins uses the cross correlation matrix for its loss. There are two parts to the
loss function:

*   ***The invariance term***(diagonal). This part is used to make the diagonals of the
matrix into 1s. When this is the case, the matrix shows that the images are
correlated(same).
  * The loss function subtracts 1 from the diagonal and squares the values.
*   ***The redundancy reduction term***(off-diagonal). Here, the barlow twins loss
function aims to make these values zero. As mentioned before, it is redundant if the
representation neurons are correlated with values that are not on the diagonal.
  * Off diagonals are squared.

After this the two parts are summed together.


```python

class BarlowLoss(losses.Loss):
    """BarlowLoss class.

    BarlowLoss class. Creates a loss function based on the cross-correlation
    matrix.

    Attributes:
        batch_size: the batch size of the dataset
        lambda_amt: the value for lambda(used in cross_corr_matrix_loss)

    Methods:
        __init__: gets instance variables
        call: gets the loss based on the cross-correlation matrix
          make_diag_zeros: Used in calculating off-diagonal section
          of loss function; makes diagonals zeros.
        cross_corr_matrix_loss: creates loss based on cross correlation
          matrix.
    """

    def __init__(self, batch_size, lambd=0.0051, **kwargs):
        """__init__ method.

        Gets the instance variables

        Arguments:
            batch_size: An integer value representing the batch size of the
              dataset. Used for cross correlation matrix calculation.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.lambd = lambd

    def normalize(self, z):
        """normalize method.

        Normalizes the model prediction.

        Arguments:
            output: the model prediction.

        Returns:
            Returns a normalized version of the model prediction.
        """
        mean = ops.mean(z, axis=0, keepdims=True)
        std = ops.std(z, axis=0, keepdims=True)
        return (z - mean) / (std + 1e-10)

    def cross_corr_matrix_loss(self, c):
        """cross_corr_matrix_loss method.

        Gets the loss based on the cross correlation matrix.
        We want the diagonals to be 1's and everything else to be
        zeros to show that the two augmented images are similar.

        Loss function procedure:
        take the diagonal of the cross-correlation matrix, subtract by 1,
        and square that value so no negatives.

        Take the off-diagonal of the cc-matrix(see get_off_diag()),
        square those values to get rid of negatives and increase the value,
        and multiply it by a lambda to weight it such that it is of equal
        value to the optimizer as the diagonal(there are more values off-diag
        then on-diag)

        Take the sum of the first and second parts and then sum them together.

        Arguments:
            c: A tensor that represents the cross correlation
              matrix

        Returns:
            Returns a tensor which represents the cross correlation
            matrix with its diagonals as zeros.
        """

        # Diagonal: c_ii - 1
        diag = ops.diagonal(c)
        diag_loss = ops.sum(ops.square(ops.subtract(diag, 1)))

        # Off-diagonal: c_ij for i != j
        off_diag = c - ops.diag(diag)
        off_diag_loss = ops.sum(ops.square(off_diag))

        return diag_loss + self.lambd * off_diag_loss

    def call(self, y_true, y_pred):
        """call method.

        Makes the cross-correlation loss. Uses the CreateCrossCorr
        class to make the cross corr matrix, then finds the loss and
        returns it(see cross_corr_matrix_loss()).

        Arguments:
            z_a: The prediction of the first set of augmented data.
            z_b: the prediction of the second set of augmented data.

        Returns:
            Returns a (rank-0) Tensor that represents the loss.
        """
        # Normalize projections
        proj_dim = ops.shape(y_pred)[1] // 2

        z_a = y_pred[:, :proj_dim]
        z_b = y_pred[:, proj_dim:]

        z_a_norm = self.normalize(z_a)
        z_b_norm = self.normalize(z_b)

        c = ops.matmul(ops.transpose(z_a_norm), z_b_norm)
        c /= ops.cast(ops.shape(z_a_norm)[0], c.dtype)

        loss = self.cross_corr_matrix_loss(c)

        return loss

```

---
## Barlow Twins' Model Architecture
The model has two parts:

*   The encoder network, which is a resnet-34.
*   The projector network, which creates the model embeddings.
   * This consists of an MLP with 3 dense-batchnorm-relu layers.

Resnet encoder network implementation:


```python

class ResNet34:
    """Resnet34 class.

        Responsible for the Resnet 34 architecture.
    Modified from
    https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/#h2_2.
    https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/#h2_2.
        View their website for more information.
    """

    def identity_block(self, x, filter):
        x_skip = x
        # Layer 1
        x = keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization(axis=3)(x)
        x = keras.layers.Activation("relu")(x)
        # Layer 2
        x = keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization(axis=3)(x)
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.Activation("relu")(x)
        return x

    def convolutional_block(self, x, filter):
        x_skip = x
        # Layer 1
        x = keras.layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3)(x)
        x = keras.layers.Activation("relu")(x)
        # Layer 2
        x = keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization(axis=3)(x)
        # Processing Residue with conv(1,1)
        x_skip = keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
        # Add Residue
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.Activation("relu")(x)
        return x

    def __call__(self, shape=(32, 32, 3)):
        x_input = keras.layers.Input(shape)
        x = keras.layers.ZeroPadding2D((3, 3))(x_input)
        x = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
        block_layers = [3, 4, 6, 3]
        filter_size = 64
        for i in range(4):
            if i == 0:
                for j in range(block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size * 2
                x = self.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)
        x = keras.layers.AveragePooling2D((2, 2), padding="same")(x)
        x = keras.layers.Flatten()(x)
        model = keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
        return model

```

Projector network:


```python

def build_twin() -> keras.Model:
    """build_twin method.

    Builds a barlow twins model consisting of an encoder(resnet-34)
    and a projector, which generates embeddings for the images

    Returns:
        returns a barlow twins model
    """

    # number of dense neurons in the projector
    n_dense_neurons = 5000

    # encoder network
    resnet = ResNet34()()
    last_layer = resnet.layers[-1].output

    # intermediate layers of the projector network
    n_layers = 2
    for i in range(n_layers):
        dense = keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{i}")
        if i == 0:
            x = dense(last_layer)
        else:
            x = dense(x)
        x = keras.layers.BatchNormalization(name=f"projector_bn_{i}")(x)
        x = keras.layers.ReLU(name=f"projector_relu_{i}")(x)

    x = keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{n_layers}")(x)

    model = keras.Model(resnet.input, x)
    return model

```

---
## Training Loop Model

See pseudocode for reference.


```python

def build_barlow_model(image_shape=(32, 32, 3)):
    """
    Builds the full Barlow Twins training model.

    The model takes two augmented images as input,
    passes both through the same encoder + projector,
    then concatenates their projections.
    """

    encoder = build_twin()

    input1 = keras.Input(shape=image_shape)
    input2 = keras.Input(shape=image_shape)

    z1 = encoder(input1)
    z2 = encoder(input2)

    z = layers.Concatenate(axis=1)([z1, z2])

    return keras.Model([input1, input2], z)

```

---
## Model Training

* Used the LAMB optimizer, instead of ADAM or SGD.
* Similar to the LARS optimizer used in the paper, and lets the model converge much
faster than other methods.
* Expected training time: 1 hour 30 min. Go and eat a snack or take a nap or something.


```python
# sets up model, optimizer, loss
barlow_model = build_barlow_model()

barlow_model.compile(
    optimizer=keras.optimizers.Lamb(),
    loss=BarlowLoss(BATCH_SIZE),
)

history = barlow_model.fit(augment_versions, epochs=2)
plt.plot(history.history["loss"])
plt.show()

```

<div class="k-default-codeblock">
```
Epoch 1/2

98/98 ━━━━━━━━━━━━━━━━━━━━ 1157s 12s/step - loss: 2743.9092

Epoch 2/2

98/98 ━━━━━━━━━━━━━━━━━━━━ 1306s 13s/step - loss: 1657.6096
```
</div>

![png](/img/examples/vision/barlow_twins/barlow_twins_34_200.png)
    


---
## Evaluation

**Linear evaluation:** to evaluate the model's performance, we add
a linear dense layer at the end and freeze the main model's weights, only letting the
dense layer to be tuned. If the model actually learned something, then the accuracy would
be significantly higher than random chance.

**Accuracy on CIFAR-10** : 64% for this notebook. This is much better than the 10% we get
from random guessing.

---
## PyDataset for Linear Evaluation


```python

class XYDataset(Sequence):
    def __init__(self, x, y, batch_size):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x_batch = self.x[index * self.batch_size : (index + 1) * self.batch_size]
        y_batch = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        return x_batch, y_batch

    def on_epoch_end(self):
        idx = np.arange(len(self.x))
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]


xy_ds = XYDataset(train_features, train_labels, BATCH_SIZE)
test_ds = XYDataset(test_features, test_labels, BATCH_SIZE)
```

---
## Evaluation

**Linear evaluation:** to evaluate the model's performance, we add
a linear dense layer at the end and freeze the main model's weights, only letting the
dense layer to be tuned. If the model actually learned something, then the accuracy would
be significantly higher than random chance.

**Accuracy on CIFAR-10** : 64% for this notebook. This is much better than the 10% we get
from random guessing.


```python
# Approx: 64% accuracy with this barlow twins model.

model = keras.models.Sequential(
    [
        barlow_model.layers[2],
        keras.layers.Dense(
            10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.02)
        ),
    ]
)
linear_optimizer = keras.optimizers.Lamb()
model.compile(
    optimizer=linear_optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.layers[0].trainable = False
model.fit(xy_ds, epochs=2, validation_data=test_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/2

98/98 ━━━━━━━━━━━━━━━━━━━━ 139s 1s/step - accuracy: 0.2275 - loss: 2.5433 - val_accuracy: 0.3606 - val_loss: 2.1783

Epoch 2/2

98/98 ━━━━━━━━━━━━━━━━━━━━ 131s 1s/step - accuracy: 0.3923 - loss: 2.0781 - val_accuracy: 0.4228 - val_loss: 1.9901

<keras.src.callbacks.history.History at 0x30e1e3920>
```
</div>

---
## Conclusion

*   Barlow Twins is a simple and concise method for contrastive and self-supervised
learning.
*   With this resnet-34 model architecture, we were able to reach 62-64% validation
accuracy.

---
## Use-Cases of Barlow-Twins(and contrastive learning in General)

*   Semi-supervised learning: You can see that this model gave a 62-64% boost in accuracy
when it wasn't even trained with the labels. It can be used when you have little labeled
data but a lot of unlabeled data.
* You do barlow twins training on the unlabeled data, and then you do secondary training
with the labeled data.

---
## Helpful links

* [Paper](https://arxiv.org/abs/2103.03230)
* [Original Pytorch Implementation](https://github.com/facebookresearch/barlowtwins)
* [Sayak Paul's Implementation](https://colab.research.google.com/github/sayakpaul/Barlow-Twins-TF/blob/main/Barlow_Twins.ipynb#scrollTo=GlWepkM8_prl).
* Thanks to Sayak Paul for his implementation. It helped me with debugging and
comparisons of accuracy, loss.
* [resnet34 implementation](https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/#h2_2)
  * Thanks to Yashowardhan Shinde for writing the article.
