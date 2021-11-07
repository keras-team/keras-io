"""
Title: Neural Style Transfer with AdaIN
Author: [Aritra Roy Gosthipaty](https://twitter.com/arig23498), [Ritwik Raha](https://twitter.com/ritwik_raha)
Date created: 2021/11/08
Last modified: 2021/11/08
Description: Implementing Neural Style Transfer with Adaptive Instance Normalization.
"""
"""
# Introduction

[Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)
is the process of transferring the style of one image onto the content
of another. This was first introduced in the seminal paper
["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)
by Gatys et al. This work was however limited in its scope because it
required a *slow iterative optimization process*.

This was tackled by future papers that introduced
[Batch Normalization](https://arxiv.org/abs/1502.03167),
[Instance Normalization](https://arxiv.org/abs/1701.02096) and
[Conditional Instance Normalization](https://arxiv.org/abs/1610.07629).

In line with this interpretation the authors Xun Huang and Serge
Belongie propose
[Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
that allows arbitrary style transfer in real time.

In this example we implement Adapative Instance Normalization
for Neural Style Transfer. We show in the below figure the inference
on our AdaIN (Adaptive Instance Normalization) model trained for
only **30 epochs**.

![img](https://i.imgur.com/zDjDuea.png)

You can also try out the model with your own images with this
[Hugging Face demo](https://huggingface.co/spaces/ariG23498/nst).
"""

"""
# Setup

We begin with importing the necessary packages. We also set the
seed for reproducibility. The global variables are hyperparameters
which we can change as we like.
"""

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# For reproducibility setting seed.
SEED = 111
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Defining the global variables.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
# Training for single epoch for time constraint.
# Please use atleast 30 epochs to see good results.
EPOCHS = 1
AUTOTUNE = tf.data.AUTOTUNE

"""
# Downloading the dataset

For Neural Style Transfer we need style images and content images. In
this example we will use the
[Best Artworks of All Time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
as our style dataset and
[Pascal VOC](https://www.tensorflow.org/datasets/catalog/voc)
as our content dataset.

This is a deviation from the original paper implementation by the
authors, where they use
[WIKI-Art](https://paperswithcode.com/dataset/wikiart) as style and
[MSCOCO](https://cocodataset.org/#home) as content datasets
respectively. We do this to create a minimal yet reproducible example
for the community.

## Pointers for downloading dataset from Kaggle

The
[Best Artworks of All Time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
dataset is hosted in Kaggle and one can easily download it in colab by
following these steps:

- Follow the instructions [here](https://github.com/Kaggle/kaggle-api)
in order to obtain your Kaggle API keys in case you don't have them.
- Use the following command to upload the Kaggle API keys.
```python
from google.colab import files
files.upload()
```
- Use the following commands to move the API keys to the proper
directory and download the dataset.
```shell
$ mkdir ~/.kaggle
$ cp kaggle.json ~/.kaggle/
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle datasets download ikarus777/best-artworks-of-all-time
$ unzip -qq best-artworks-of-all-time.zip
$ rm -rf images
$ mv resized artwork
$ rm best-artworks-of-all-time.zip artists.csv
```
"""

"""
# `tf.data` pipeline

In this section, we will build the `tf.data` pipeline for the project.
For the style dataset, we decode, convert and resize the images from
the folder. For the content images we are already presented with a
`tf.data` dataset as we use the `tfds` module.

After we have out style and content data pipeline ready, we zip the
two together to obtain the data pipeline that our model will consume.
"""


def decode_and_resize(image_path):
    """Decodes and resizes an image from the image file path.

    Args:
        image_path: The image file path.
        size: The size of the image to be resized to.

    Returns:
        A resized image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def extract_image_from_voc(element):
    """Extracts image from the pascal voc dataset.

    Args:
        element: A dictionary of data.
        size: The size of the image to be resized to.

    Returns:
        A resized image.
    """
    image = element["image"]
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


# Get the image file paths for the style images.
style_images = os.listdir("artwork/resized")
style_images = [os.path.join("artwork/resized", path) for path in style_images]

# split the style images in train, val and test
total_style_images = len(style_images)
train_style = style_images[: int(0.8 * total_style_images)]
val_style = style_images[int(0.8 * total_style_images) : int(0.9 * total_style_images)]
test_style = style_images[int(0.9 * total_style_images) :]

# Build the style and content tf.data datasets.
train_style_ds = (
    tf.data.Dataset.from_tensor_slices(train_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)
train_content_ds = tfds.load("voc", split="train").map(extract_image_from_voc).repeat()

val_style_ds = (
    tf.data.Dataset.from_tensor_slices(val_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)
val_content_ds = (
    tfds.load("voc", split="validation").map(extract_image_from_voc).repeat()
)

test_style_ds = (
    tf.data.Dataset.from_tensor_slices(test_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)
test_content_ds = (
    tfds.load("voc", split="test")
    .map(extract_image_from_voc, num_parallel_calls=AUTOTUNE)
    .repeat()
)

# Zipping the style and content datasets.
train_ds = (
    tf.data.Dataset.zip((train_style_ds, train_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.zip((val_style_ds, val_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.zip((test_style_ds, test_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

"""
# Visualizing the data

It is always better to visualize the data while going ahead. Here we
iterate over the zipped dataset and take 10 pairs of images depicting
the style and content images.
"""

style, content = next(iter(train_ds))
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(5, 30))
[ax.axis("off") for ax in np.ravel(axes)]

for (
    (
        ax_s,
        ax_c,
    ),
    s_im,
    c_im,
) in zip(axes, style[0:10], content[0:10]):
    ax_s.imshow(s_im)
    ax_s.set_title("Style Image")

    ax_c.imshow(c_im)
    ax_c.set_title("Content Image")

"""
# Architecture

The style transfer network takes a content image and a style image as
inputs and outputs the style transfered image. The authors of AdaIN
propose a simple encoder-decoder structure for achieving this.

![AdaIN architecture](https://i.imgur.com/JbIfoyE.png)

The content image (`C`) and the style image (`S`) are both fed to the
encoder networks. The output from these encoder networks (feature maps)
are then fed to the AdaIN layer. The AdaIN layer computes a combined
feature map. This feature map is then fed into a randomly initialized
decoder network that serves as the generator for the neural style
transfered image.

![AdaIn equation](https://i.imgur.com/hqhcBQS.png)

The style feature map (`fs`) and the content feature map (`fc`) are
fed to the AdaIN layer. This layer produced the combined feature map
`t`. The function `g` represents the decoder (generator) network.
"""

"""
## Encoder

The encoder is a part of the pretrained (pretrianed on
[imagenet](https://www.image-net.org/)) VGG19 model. We slice the
model from the `block4-conv1` layer. The choice of the output layer
is as suggested by the authors in their paper.
"""


def get_encoder():
    vgg19 = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    vgg19.trainable = False
    mini_vgg19 = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")


"""
## Adaptive Instance Normalization

The AdIN (Adaptive Instance Normalization) layer takes in the features
of the content and style image. The layer can be defined using the
following equation:

![img](https://i.imgur.com/tWq3VKP.png)

where `sigma` is the standard deviation and `mu` is the mean for the
concerned variable. In the above equation the mean and variance of the
content feature map `fc` is aligned with the mean and variance of the
style feature maps `fs`.

It is important to note that the AdaIN layer proposed by the authors
uses no other parameters apart from mean and variance. The layer also
does not have any learning parameters. This is why we use a
*python function* instead of using a *keras layer*. The function takes
style and content feature maps, computes the mean and standard deviation
of the images and returns the adaptive instance normalized feature map.
"""


def get_mean_std(tensor, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    tensor_mean, tensor_var = tf.nn.moments(tensor, axes=axes, keepdims=True)
    tensor_std = tf.sqrt(tensor_var + epsilon)

    # Return the mean and standard deviation
    return tensor_mean, tensor_std


def ada_in(style, content, epsilon=1e-5):
    """Computes the adaptive instance normalized feature map from the style and
    content feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    c_mean, c_std = get_mean_std(content)
    s_mean, s_std = get_mean_std(style)
    t = s_std * (content - c_mean) / c_std + s_mean
    return t


"""
## Decoder

The authors specify that the deocder network must mirror the encoder
network.  We have symmetrically inverted the encoder to build our
decoder. We have used `UpSampling2D` layers to increase the spatial
resolution of the feature maps.

Note that the authors warn against using any normalization layer
in the deocder network and do indeed go onto show that including
batch normalization or instance normalization hurts the performance
of the overall network.

Out of the entire architecture, this is the only portion that is
trainable. 
"""


def get_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),
            layers.Conv2D(filters=512, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=128, **config),
            layers.Conv2D(filters=128, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=64, **config),
            layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
            ),
        ]
    )
    return decoder


"""
## Loss Network

Here we build the loss functions for the neural style transfer model.
The authors propose to use a pretrained VGG-19 to compute the loss
function of the network. It is important to keep in mind that this
will be used for training only the decoder netwrok. The total
loss (`Lt`) is a weighted combination of content loss (`Lc`) and style
loss (`Ls`). The `lambda` term is used to vary the amount of style
transfered.

![The total loss](https://i.imgur.com/Q5y1jUM.png)

### Content Loss

This is the Euclidean distance between the content image features
and the features of the neural style transferred image.

![The content loss](https://i.imgur.com/dZ0uD0N.png)

Here the authors propose to use the output from the AdaIn layer `t` as
the content target rather than using features of the original image as
target. This is done to optimize faster convergence of the loss.

### Style Loss

Rather than using the more commonly used [Gram
Matrix](https://mathworld.wolfram.com/GramMatrix.html) the authors
propose to compute the difference between the statistical features
(mean and variance) which makes it conceptually cleaner. This can
easily visualized in the following equation

![The style loss](https://i.imgur.com/Ctclhn3.png)

where `theta` denotes the layers in VGG-19 used to compute the loss.
In this case this corresponds to:

- `block1_conv1`
- `block1_conv2`
- `block1_conv3`
- `block1_conv4`

"""


def get_loss_net():
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")


"""
## Neural Style Transfer

This is the trainer module. We wrap the encoder and decoder inside a
trainer module so that we can harness the beautiful `model.fit` API. 
"""


class NST(tf.keras.Model):
    def __init__(self, encoder, decoder, ada_in, loss_net, lamb):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ada_in = ada_in
        self.loss_net = loss_net
        self.lamb = lamb

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def train_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            # Encode the style and content image.
            style_enc = self.encoder(style)
            content_enc = self.encoder(content)

            # Compute the AdaIN feature map.
            t = self.ada_in(style=style_enc, content=content_enc)

            # Generate the neural style transferred image.
            recons_image = self.decoder(t)

            # Compute the losses.
            recons_vgg_feats = self.loss_net(recons_image)
            style_vgg_feats = self.loss_net(style)
            loss_content = self.loss_fn(t, recons_vgg_feats[-1])
            for inp, out in zip(style_vgg_feats, recons_vgg_feats):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                    std_inp, std_out
                )
            loss_style = self.lamb * loss_style
            total_loss = loss_content + loss_style

        # Compute gradients and optimize the decoder.
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def test_step(self, inputs):
        style, content = inputs
        loss_content = 0.0
        loss_style = 0.0
        style_enc = self.encoder(style)
        content_enc = self.encoder(content)
        t = self.ada_in(style=style_enc, content=content_enc)
        recons_image = self.decoder(t)
        recons_vgg_feats = self.loss_net(recons_image)
        style_vgg_feats = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_feats[-1])
        for inp, out in zip(style_vgg_feats, recons_vgg_feats):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.lamb * loss_style
        total_loss = loss_content + loss_style
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]


"""
## Train Monitor callback

This callback is used to visualize the style transfer capability of
the model on each epoch. The objective of style transfer is not
quantified properly and is mostly upto the audience to call is good
or bad. For this reason, visualization falls into a key aspect of
training the model.
"""

test_style, test_content = next(iter(test_ds))


class TrainMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Encode the style and content image.
        test_style_enc = self.model.encoder(test_style)
        test_content_enc = self.model.encoder(test_content)

        # Compute the AdaIN features.
        test_t = self.model.ada_in(style=test_style_enc, content=test_content_enc)
        test_recons_image = self.model.decoder(test_t)

        # Plot the Style, Content and the NST image.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(tf.keras.preprocessing.image.array_to_img(test_style[0]))
        ax[0].set_title(f"Style: {epoch:03d}")

        ax[1].imshow(tf.keras.preprocessing.image.array_to_img(test_content[0]))
        ax[1].set_title(f"Content: {epoch:03d}")

        ax[2].imshow(tf.keras.preprocessing.image.array_to_img(test_recons_image[0]))
        ax[2].set_title(f"NST: {epoch:03d}")

        plt.show()
        plt.close()


"""
# Train the model

In this section, we initialize the optimizer, loss funtion and the
trainer module. We compile the trainer module with the optimizer and
the loss function and then train it.

*Note*: We train the model for a single epoch for time constranints,
but we will need to train is for atleast 30 epochs to see good results.
"""

optimizer = keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = keras.losses.MeanSquaredError()

encoder = get_encoder()
loss_net = get_loss_net()
decoder = get_decoder()

model = NST(
    encoder=encoder, decoder=decoder, ada_in=ada_in, loss_net=loss_net, lamb=4.0
)

model.compile(optimizer=optimizer, loss_fn=loss_fn)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=50,
    validation_data=val_ds,
    validation_steps=50,
    callbacks=[TrainMonitor()],
)

"""
# Inference

After we train the model, we now need to run inference on it. We will
pass arbitrary content and style images from the test dataset and see
the NST images.

*NOTE*: To try out the model on your own images you can go to this
[Hugging Face demo](https://huggingface.co/spaces/ariG23498/nst).
"""

for style, content in test_ds.take(1):
    style_enc = model.encoder(style)
    content_enc = model.encoder(content)
    t = model.ada_in(style=style_enc, content=content_enc)
    recons_image = model.decoder(t)
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(10, 30))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (ax_s, ax_c, ax_r), s_im, c_im, r_im in zip(
        axes, style[0:10], content[0:10], recons_image[0:10]
    ):
        ax_s.imshow(s_im)
        ax_s.set_title("Style Image")
        ax_c.imshow(c_im)
        ax_c.set_title("Content Image")
        ax_r.imshow(r_im)
        ax_r.set_title("NST Image")

"""
# Conclusion

Adaptive Instance Normalization is an important step in the direction
of Neural Style Transfer research beacuse it allows arbitrary style
transfer in real time. It is also important to note that the novel
proposition of the authors is to achive this only by aligning the
statistical features (mean and standard deviation) of the style and the
content images.

*Note*: Despite being this simple, AdaIN forms the base for
[Style-GANs](https://arxiv.org/abs/1812.04948).

# Reference

- https://github.com/ftokarev/tf-adain
"""
