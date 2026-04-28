# Barlow Twins for Contrastive SSL

**Author:** [Abhiraam Eranti](https://github.com/dewball345)<br>
**Date created:** 11/4/21<br>
**Last modified:** 26/04/28<br>
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
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" or "torch"

import keras
import numpy as np
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
# BATCH_SIZE = 1024

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

class Augmentation(layers.Layer):
    """Base augmentation class.

    Base augmentation class. Contains the random_execute method.

    Methods:
        random_execute: method that returns true or false based
          on a probability. Used to determine whether an augmentation
          will be run.
    """

    def __init__(self):
        super().__init__()

    def random_execute(self, prob: float):
        """random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the
              probability.

        Returns:
            returns a boolean mask tensor.
        """
        return random.uniform(()) < prob


class RandomToGrayscale(Augmentation):
    """RandomToGrayscale class.

    RandomToGrayscale class. Randomly makes an image
    grayscaled based on the random_execute method. There
    is a 20% chance that an image will be grayscaled.

    Methods:
        call: method that grayscales an image 20% of
          the time.
    """

    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

    def call(self, x):
        mask = self.random_execute(self.prob)
        # average channels to get grayscale
        gray = ops.mean(x, axis=-1, keepdims=True)
        gray = ops.repeat(gray, 3, axis=-1)
        x = ops.where(mask, gray, x)
        return x


class RandomFlip(Augmentation):
    """RandomFlip class.

    RandomFlip class. Randomly flips image horizontally. There is a 50%
    chance that an image will be randomly flipped.

    Methods:
        call: method that flips an image 50% of
          the time.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def call(self, x):
        """call function.

        Randomly flips the image.

        Arguments:
            x: a Tensor representing the image.

        Returns:
            returns a flipped version of the image 50% of the time
              and the original image 50% of the time.
        """
        mask = self.random_execute(self.prob)
        # flip horizontally
        flipped = ops.flip(x, axis=2)  # batch-aware if x is (B,H,W,C)
        return ops.where(mask, flipped, x)


class RandomResizedCrop(Augmentation):
    """RandomResizedCrop class.

    RandomResizedCrop applies a random crop to an image and then
    resizes it back to the target image size. This is a key augmentation
    used in self-supervised learning methods such as Barlow Twins,
    as it encourages invariance to spatial transformations.

    The crop size and position are randomly sampled, and the cropped
    region is resized back to the original image dimensions.

    Supports both single images (H, W, C) and batched inputs
    (B, H, W, C).

    Attributes:
        image_size: Integer representing the target height and width
            of the output image.

    Methods:
        _crop_one: Applies random resized cropping to a single image.
        call: Applies the augmentation to either a single image or a batch.
    """

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def _crop_one(self, x):
        h, w = x.shape[0], x.shape[1]

        crop_size = random.randint((), int(0.75 * self.image_size), self.image_size)

        top = random.randint((), 0, h - crop_size + 1)
        left = random.randint((), 0, w - crop_size + 1)

        crop = x[top : top + crop_size, left : left + crop_size, :]
        return ops.image.resize(crop, (self.image_size, self.image_size))

    def call(self, x):
        # x can be (H,W,C) or (B,H,W,C)
        if len(ops.shape(x)) == 3:
            return self._crop_one(x)

        crop = ops.stack([self._crop_one(img) for img in x])

        return crop


class RandomSolarize(Augmentation):
    """RandomSolarize class.

    RandomSolarize class. Randomly solarizes an image.
    Solarization is when pixels accidentally flip to an inverted state.

    Methods:
        call: method that does random solarization 20% of the time.
    """

    def __init__(self, prob=0.2, threshold=0.4):
        super().__init__()
        self.prob = prob
        self.threshold = threshold

    def call(self, x):
        """call function.

        Randomly solarizes the image.

        Arguments:
            x: a Tensor representing the image.

        Returns:
            returns a solarized version of the image 20% of the time
              and the original image 80% of the time.
        """
        mask = self.random_execute(self.prob)
        solarized = ops.where(x < self.threshold, x, 1.0 - x)
        return ops.where(mask, solarized, x)


class RandomBlur(Augmentation):
    """RandomBlur class.

    RandomBlur class. Randomly blurs an image.

    Methods:
        call: method that does random blur 20% of the time.
    """

    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype="float32")
        self.kernel = kernel / np.sum(kernel)

    def call(self, x):
        mask = self.random_execute(self.prob)

        k = ops.convert_to_tensor(self.kernel)
        k = ops.reshape(k, (3, 3, 1, 1))
        k = ops.tile(k, (1, 1, 3, 1))

        blurred = ops.nn.depthwise_conv(x, k, strides=1, padding="SAME")

        return ops.where(mask, blurred, x)


class RandomAugmentor(layers.Layer):
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

    def __init__(self, image_size: int):
        super().__init__()
        self.random_resized_crop = RandomResizedCrop(image_size)
        self.random_flip = RandomFlip()
        self.random_color_jitter = layers.RandomColorJitter(
            brightness_factor=0.8,
            contrast_factor=(0.4, 1.6),
            saturation_factor=(0.4, 1.6),
            hue_factor=0.2,
            value_range=(0, 1),
        )
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def call(self, x):
        x = self.random_resized_crop(x)
        x = self.random_flip(x)
        x = self.random_color_jitter(x)
        x = self.random_blur(x)
        x = self.random_to_grayscale(x)
        x = self.random_solarize(x)

        return ops.clip(x, 0, 1)


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

    def __call__(self, x):
        class BTDataset(Sequence):
            def __init__(self, x, augmentor, batch_size, seed):
                super().__init__()
                self.x = x
                self.augmentor = augmentor
                self.batch_size = batch_size
                self.seed = seed
                self.on_epoch_end()

            def __len__(self):
                return int(np.ceil(len(self.x) / self.batch_size))

            def __getitem__(self, idx):
                batch = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

                a1 = self.augmentor(batch)
                a2 = self.augmentor(batch)

                y_dummy = np.zeros((len(batch), 1))
                return (a1, a2), y_dummy

            def on_epoch_end(self):
                np.random.seed(self.seed)
                idx = np.arange(len(self.x))
                np.random.shuffle(idx)
                self.x = self.x[idx]

        return BTDataset(x, self.augmentor, self.batch_size, self.seed)


augment_versions = BTDatasetCreator(bt_augmentor)(train_features)
```

View examples of dataset.


```python
sample_augment_versions = iter(augment_versions)


def plot_values(batch: tuple):
    fig, axs = plt.subplots(3, 3)
    fig1, axs1 = plt.subplots(3, 3)

    fig.suptitle("Augmentation 1")
    fig1.suptitle("Augmentation 2")

    (a1, a2), _ = batch
    # Convert to numpy
    a1 = ops.convert_to_numpy(a1)
    a2 = ops.convert_to_numpy(a2)

    for i in range(3):
        for j in range(3):
            axs[i][j].imshow(a1[3 * i + j])  # shape (32, 32, 3)
            axs[i][j].axis("off")

            axs1[i][j].imshow(a2[3 * i + j])
            axs1[i][j].axis("off")

    plt.show()


plot_values(next(sample_augment_versions))
```


    
![png](/img/examples/vision/barlow_twins/barlow_twins_20_0.png)
    



    
![png](/img/examples/vision/barlow_twins/barlow_twins_20_1.png)
    


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
        diag_loss = ops.sum(ops.square(diag - 1))

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

def build_encoder():
    """build_twin method.

    Builds a barlow twins model consisting of an encoder(resnet-34)
    and a projector, which generates embeddings for the images

    Returns:
        returns a barlow twins model
    """

    # encoder network
    resnet = ResNet34()()
    return keras.Model(
        inputs=resnet.input, outputs=resnet.layers[-1].output, name="encoder_resnet34"
    )


def build_projector(input_dim):
    inputs = keras.Input(shape=(input_dim,))

    x = inputs
    for i in range(2):
        x = keras.layers.Dense(5000)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    outputs = keras.layers.Dense(5000)(x)

    model = keras.Model(inputs, outputs, name="projector")

    return model


def build_twin():
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

    x = keras

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
    encoder = build_encoder()

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

history = barlow_model.fit(augment_versions, epochs=5)
plt.plot(history.history["loss"])
plt.show()

```

<div class="k-default-codeblock">
```
Epoch 1/5

98/98 ━━━━━━━━━━━━━━━━━━━━ 1606s 16s/step - loss: 345.1386

Epoch 2/5

98/98 ━━━━━━━━━━━━━━━━━━━━ 1714s 17s/step - loss: 229.7889

Epoch 3/5

98/98 ━━━━━━━━━━━━━━━━━━━━ 1914s 19s/step - loss: 182.4243

Epoch 4/5

98/98 ━━━━━━━━━━━━━━━━━━━━ 1732s 18s/step - loss: 163.0861

Epoch 5/5

98/98 ━━━━━━━━━━━━━━━━━━━━ 1473s 15s/step - loss: 157.1498
```
</div>

![png](/img/examples/vision/barlow_twins/barlow_twins_34_500.png)
    


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

encoder = build_encoder()

# Load pretrained weights
encoder.set_weights(barlow_model.get_layer("encoder_resnet34").get_weights())
# Freeze encoder
encoder.trainable = False

# Build model
model = keras.models.Sequential(
    [
        encoder,
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
model.fit(xy_ds, epochs=20, validation_data=test_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 155s 2s/step - accuracy: 0.1576 - loss: 3.4602 - val_accuracy: 0.2131 - val_loss: 2.7183

Epoch 2/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 147s 2s/step - accuracy: 0.2763 - loss: 2.4487 - val_accuracy: 0.3292 - val_loss: 2.2542

Epoch 3/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 145s 1s/step - accuracy: 0.3383 - loss: 2.2019 - val_accuracy: 0.3424 - val_loss: 2.1568

Epoch 4/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 146s 1s/step - accuracy: 0.3452 - loss: 2.1399 - val_accuracy: 0.3473 - val_loss: 2.1108

Epoch 5/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 147s 2s/step - accuracy: 0.3472 - loss: 2.1032 - val_accuracy: 0.3468 - val_loss: 2.0822

Epoch 6/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3500 - loss: 2.0761 - val_accuracy: 0.3495 - val_loss: 2.0577

Epoch 7/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 145s 1s/step - accuracy: 0.3508 - loss: 2.0512 - val_accuracy: 0.3539 - val_loss: 2.0334

Epoch 8/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 145s 1s/step - accuracy: 0.3518 - loss: 2.0307 - val_accuracy: 0.3538 - val_loss: 2.0128

Epoch 9/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 145s 1s/step - accuracy: 0.3543 - loss: 2.0111 - val_accuracy: 0.3530 - val_loss: 1.9935

Epoch 10/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 145s 1s/step - accuracy: 0.3516 - loss: 1.9935 - val_accuracy: 0.3559 - val_loss: 1.9782

Epoch 11/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3556 - loss: 1.9781 - val_accuracy: 0.3546 - val_loss: 1.9629

Epoch 12/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3547 - loss: 1.9637 - val_accuracy: 0.3584 - val_loss: 1.9511

Epoch 13/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 146s 1s/step - accuracy: 0.3580 - loss: 1.9515 - val_accuracy: 0.3597 - val_loss: 1.9353

Epoch 14/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 147s 2s/step - accuracy: 0.3585 - loss: 1.9398 - val_accuracy: 0.3643 - val_loss: 1.9243

Epoch 15/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3580 - loss: 1.9294 - val_accuracy: 0.3642 - val_loss: 1.9164

Epoch 16/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 144s 1s/step - accuracy: 0.3584 - loss: 1.9197 - val_accuracy: 0.3620 - val_loss: 1.9085

Epoch 17/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 144s 1s/step - accuracy: 0.3589 - loss: 1.9109 - val_accuracy: 0.3640 - val_loss: 1.8994

Epoch 18/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 144s 1s/step - accuracy: 0.3600 - loss: 1.9032 - val_accuracy: 0.3618 - val_loss: 1.8982

Epoch 19/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3601 - loss: 1.8966 - val_accuracy: 0.3652 - val_loss: 1.8848

Epoch 20/20

98/98 ━━━━━━━━━━━━━━━━━━━━ 143s 1s/step - accuracy: 0.3612 - loss: 1.8902 - val_accuracy: 0.3660 - val_loss: 1.8769

<keras.src.callbacks.history.History at 0x31ecfcf20>
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
