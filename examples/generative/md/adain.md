# Neural Style Transfer with AdaIN

**Author:** [Aritra Roy Gosthipaty](https://twitter.com/arig23498), [Ritwik Raha](https://twitter.com/ritwik_raha)<br>
**Date created:** 2021/11/08<br>
**Last modified:** 2026/04/13<br>
**Description:** Neural Style Transfer with Adaptive Instance Normalization.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/adain.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/adain.py)



# Introduction

[Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)
is the process of transferring the style of one image onto the content
of another. This was first introduced in the seminal paper
["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)
by Gatys et al. A major limitation of the technique proposed in this
work is in its runtime, as the algorithm uses a slow iterative
optimization process.

Follow-up papers that introduced
[Batch Normalization](https://arxiv.org/abs/1502.03167),
[Instance Normalization](https://arxiv.org/abs/1701.02096) and
[Conditional Instance Normalization](https://arxiv.org/abs/1610.07629)
allowed Style Transfer to be performed in new ways, no longer
requiring a slow iterative process.

Following these papers, the authors Xun Huang and Serge
Belongie propose
[Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) (AdaIN),
which allows arbitrary style transfer in real time.

In this example we implement Adaptive Instance Normalization
for Neural Style Transfer. We show in the below figure the output
of our AdaIN model trained for
only **30 epochs**.

![Style transfer sample gallery](https://i.imgur.com/zDjDuea.png)

You can also try out the model with your own images with this
[Hugging Face demo](https://huggingface.co/spaces/ariG23498/nst).

# Setup

We begin by importing the packages used throughout the example and
defining a few global hyperparameters that control image size,
batch size, and training duration.


```python
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers

# Defining the global variables.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
# Training for single epoch for time constraint.
# Please use atleast 30 epochs to see good results.
EPOCHS = 30
```

---
## Style transfer sample gallery

For Neural Style Transfer we need style images and content images. In
this example we will use the
[Best Artworks of All Time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
as our style dataset and
[Pascal VOC](https://host.robots.ox.ac.uk/pascal/VOC/) images
as our content dataset.

This is a deviation from the original paper implementation by the
authors, where they use
[WIKI-Art](https://paperswithcode.com/dataset/wikiart) as style and
[MSCOCO](https://cocodataset.org/#home) as content datasets
respectively. We do this to create a minimal yet reproducible example.

---
## Downloading the dataset from Kaggle

The [Best Artworks of All Time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
dataset is hosted on Kaggle and one can easily download it in Colab by
following these steps:

- Follow the instructions [here](https://github.com/Kaggle/kaggle-api)
in order to obtain your Kaggle API keys in case you don't have them.
- Use the following command to upload the Kaggle API keys.

```python
from google.colab import files
files.upload()
```

- Use the following commands to move the API keys to the proper
directory and download the style dataset.

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

- Download Pascal VOC 2012 images and extract them under `/content/VOCdevkit/VOC2012/JPEGImages`.

---
## `PyDataset` pipeline

In this section, we build a `keras.utils.PyDataset` input pipeline
with no `tensorflow_datasets` dependency.

Both style and content images are read from local folders, decoded, resized,
and paired inside a custom `PyDataset` implementation.

After loading image paths, we create train, validation, and test
`PyDataset` instances that the model will consume.


```python

class StyleContentPyDataset(keras.utils.PyDataset):
    def __init__(
        self, style_paths, content_paths, batch_size, image_size, shuffle=True
    ):
        super().__init__()
        self.style_paths = style_paths
        self.content_paths = content_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        self.num_style = len(self.style_paths)
        self.num_content = len(self.content_paths)
        self.num_samples = max(self.num_style, self.num_content)

        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, image_path):
        image = keras.utils.load_img(image_path, target_size=self.image_size)
        image = keras.utils.img_to_array(image).astype("float32") / 255.0
        return image

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]

        style_batch = [
            self._load_image(self.style_paths[i % self.num_style])
            for i in batch_indices
        ]
        content_batch = [
            self._load_image(self.content_paths[i % self.num_content])
            for i in batch_indices
        ]

        # Return x only as a length-1 tuple so fit() treats (style, content) as model input.
        return ((np.stack(style_batch), np.stack(content_batch)),)


def list_image_paths(image_root):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = [
        os.path.join(image_root, name)
        for name in os.listdir(image_root)
        if os.path.isfile(os.path.join(image_root, name))
        and name.lower().endswith(valid_exts)
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError(f"No images found under: {image_root}")
    return image_paths


def split_paths(paths):
    total = len(paths)
    train_end = max(1, int(0.8 * total))
    val_end = max(train_end + 1, int(0.9 * total))

    train = paths[:train_end]
    val = paths[train_end:val_end]
    test = paths[val_end:]

    # Ensure no split is empty for very small fallback datasets.
    if not val:
        val = train[: min(len(train), 1)]
    if not test:
        test = train[: min(len(train), 1)]

    return train, val, test


def ensure_fallback_image_dirs(style_root, content_root, image_size):
    if os.path.isdir(style_root) and os.path.isdir(content_root):
        if os.listdir(style_root) and os.listdir(content_root):
            return style_root, content_root

    fallback_style = "/tmp/adain_data/style"
    fallback_content = "/tmp/adain_data/content"
    os.makedirs(fallback_style, exist_ok=True)
    os.makedirs(fallback_content, exist_ok=True)

    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train[:1000]

    for i, image in enumerate(x_train):
        pil_img = keras.utils.array_to_img(image)
        pil_img = pil_img.resize((image_size[1], image_size[0]))
        if i % 2 == 0:
            pil_img.save(os.path.join(fallback_style, f"style_{i:04d}.jpg"))
        else:
            pil_img.save(os.path.join(fallback_content, f"content_{i:04d}.jpg"))

    print(
        "Using fallback local data at /tmp/adain_data because /content paths were not found."
    )
    return fallback_style, fallback_content


# Dataset roots
style_root = "/content/artwork/resized"
content_root = "/content/VOCdevkit/VOC2012/JPEGImages"
style_root, content_root = ensure_fallback_image_dirs(
    style_root, content_root, IMAGE_SIZE
)

# Build path lists
style_paths = list_image_paths(style_root)
content_paths = list_image_paths(content_root)

# Split style/content paths
train_style, val_style, test_style = split_paths(style_paths)
train_content, val_content, test_content = split_paths(content_paths)

# Build PyDataset objects
train_ds = StyleContentPyDataset(
    train_style,
    train_content,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
)
val_ds = StyleContentPyDataset(
    val_style, val_content, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True
)
test_ds = StyleContentPyDataset(
    test_style, test_content, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True
)
```

<div class="k-default-codeblock">
Using fallback local data at /tmp/adain_data because /content paths were not found.
</div>

---
## Visualizing the data

It is always better to visualize the data before training. To ensure
the correctness of our preprocessing pipeline, we visualize 10 samples
from our dataset.


```python
((style, content),) = next(iter(train_ds))
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(5, 30))
[ax.axis("off") for ax in np.ravel(axes)]

for axis, style_image, content_image in zip(axes, style[0:10], content[0:10]):
    ax_style, ax_content = axis
    ax_style.imshow(style_image)
    ax_style.set_title("Style Image")

    ax_content.imshow(content_image)
    ax_content.set_title("Content Image")
```


    
![png](/img/examples/generative/adain/adain_8_0.png)
    


---
## Architecture

The style transfer network takes a content image and a style image as
inputs and outputs the style transferred image. The authors of AdaIN
propose a simple encoder-decoder structure for achieving this.

![AdaIN architecture](https://i.imgur.com/JbIfoyE.png)

The content image (`C`) and the style image (`S`) are both fed to the
encoder networks. The output from these encoder networks (feature maps)
are then fed to the AdaIN layer. The AdaIN layer computes a combined
feature map. This feature map is then fed into a randomly initialized
decoder network that serves as the generator for the neural style
transferred image.

![AdaIn equation](https://i.imgur.com/hqhcBQS.png)

The style feature map (`fs`) and the content feature map (`fc`) are
fed to the AdaIN layer. This layer produced the combined feature map
`t`. The function `g` represents the decoder (generator) network.

### Encoder

The encoder is a part of the pretrained (pretrained on
[imagenet](https://www.image-net.org/)) VGG19 model. We slice the
model from the `block4-conv1` layer. The output layer is as suggested
by the authors in their paper.


```python

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

```

### Adaptive Instance Normalization

The AdaIN layer takes in the features
of the content and style image. The layer can be defined via the
following equation:

![AdaIn formula](https://i.imgur.com/tWq3VKP.png)

where `sigma` is the standard deviation and `mu` is the mean for the
concerned variable. In the above equation the mean and variance of the
content feature map `fc` is aligned with the mean and variance of the
style feature maps `fs`.

It is important to note that the AdaIN layer proposed by the authors
uses no other parameters apart from mean and variance. The layer also
does not have any trainable parameters. This is why we use a
*Python function* instead of using a *Keras layer*. The function takes
style and content feature maps, computes the mean and standard deviation
of the images and returns the adaptive instance normalized feature map.


```python

def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = keras.ops.moments(x, axes=axes, keepdims=True)
    standard_deviation = keras.ops.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t

```

### Decoder

The authors specify that the decoder network must mirror the encoder
network.  We have symmetrically inverted the encoder to build our
decoder. We have used `UpSampling2D` layers to increase the spatial
resolution of the feature maps.

Note that the authors warn against using any normalization layer
in the decoder network, and do indeed go on to show that including
batch normalization or instance normalization hurts the performance
of the overall network.

This is the only portion of the entire architecture that is trainable.


```python

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

```

### Loss functions

Here we build the loss functions for the neural style transfer model.
The authors propose to use a pretrained VGG-19 to compute the loss
function of the network. It is important to keep in mind that this
will be used for training only the decoder network. The total
loss (`Lt`) is a weighted combination of content loss (`Lc`) and style
loss (`Ls`). The `lambda` term is used to vary the amount of style
transferred.

![The total loss](https://i.imgur.com/Q5y1jUM.png)

### Content Loss

This is the Euclidean distance between the content image features
and the features of the neural style transferred image.

![The content loss](https://i.imgur.com/dZ0uD0N.png)

Here the authors propose to use the output from the AdaIn layer `t` as
the content target rather than using features of the original image as
target. This is done to speed up convergence.

### Style Loss

Rather than using the more commonly used
[Gram Matrix](https://mathworld.wolfram.com/GramMatrix.html),
the authors propose to compute the difference between the statistical features
(mean and variance) which makes it conceptually cleaner. This can be
easily visualized via the following equation:

![The style loss](https://i.imgur.com/Ctclhn3.png)

where `theta` denotes the layers in VGG-19 used to compute the loss.
In this case this corresponds to:

- `block1_conv1`
- `block1_conv2`
- `block1_conv3`
- `block1_conv4`


```python

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

```

---
## Neural Style Transfer

This is the trainer module. We wrap the encoder and decoder inside
a `keras.Model` subclass and migrate training logic into `compute_loss()`.
This lets Keras run the backend-native training step while we keep
custom loss computation for AdaIN.


```python

class NeuralStyleTransfer(keras.Model):
    def __init__(self, encoder, decoder, loss_net, style_weight, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight

        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def compile(self, optimizer, loss_fn, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.loss_fn = loss_fn

    def call(self, inputs, training=False):
        style, content = inputs

        # Encode style and content, then construct AdaIN target.
        style_encoded = self.encoder(style, training=False)
        content_encoded = self.encoder(content, training=False)
        target_features = ada_in(style=style_encoded, content=content_encoded)

        # Decode back into image space.
        reconstructed_image = self.decoder(target_features, training=training)

        return {
            "reconstructed_image": reconstructed_image,
            "target_features": target_features,
            "style": style,
        }

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Loss content comes from target AdaIN features vs reconstructed features.
        reconstructed_vgg_features = self.loss_net(
            y_pred["reconstructed_image"], training=False
        )
        style_vgg_features = self.loss_net(y_pred["style"], training=False)

        loss_content = self.loss_fn(
            y_pred["target_features"], reconstructed_vgg_features[-1]
        )

        # Style loss matches means/variances at each VGG feature level.
        loss_style = 0.0
        for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style

        total_loss = loss_content + loss_style

        # Update trackers for progress logs.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)

        return total_loss

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]

```

---
## Train Monitor callback

This callback is used to visualize the style transfer output of
the model at the end of each epoch. The objective of style transfer cannot be
quantified properly, and is to be subjectively evaluated by an audience.
For this reason, visualization is a key aspect of evaluating the model.


```python
((test_style, test_content),) = next(iter(test_ds))


class TrainMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Encode the style and content image.
        test_style_encoded = self.model.encoder(test_style)
        test_content_encoded = self.model.encoder(test_content)

        # Compute the AdaIN features.
        test_t = ada_in(style=test_style_encoded, content=test_content_encoded)
        test_reconstructed_image = self.model.decoder(test_t)

        # Plot the Style, Content and the NST image.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_style[0]))
        ax[0].set_title(f"Style: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(test_content[0]))
        ax[1].set_title(f"Content: {epoch:03d}")

        ax[2].imshow(keras.utils.array_to_img(test_reconstructed_image[0]))
        ax[2].set_title(f"NST: {epoch:03d}")

        plt.show()
        plt.close()

```

---
## Train the model

In this section, we define the optimizer, the loss function, and the
trainer module. We compile the trainer module with the optimizer and
the loss function and then train it.

*Note*: We train the model for a single epoch for time constraints,
but you will typically need to train it for at least 30 epochs to see
good results.


```python
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = keras.losses.MeanSquaredError()

encoder = get_encoder()
loss_net = get_loss_net()
decoder = get_decoder()

model = NeuralStyleTransfer(
    encoder=encoder, decoder=decoder, loss_net=loss_net, style_weight=4.0
)

model.compile(optimizer=optimizer, loss_fn=loss_fn)

train_steps = min(50, len(train_ds))
val_steps = min(50, len(val_ds))

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[TrainMonitor()],
)
```

<div class="k-default-codeblock">
Epoch 1/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 69.3327 - style_loss: 140.9464 - total_loss: 210.2792
</div>

![png](/img/examples/generative/adain/adain_23_8.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 25s 3s/step - content_loss: 64.4736 - style_loss: 129.5553 - total_loss: 194.0289 - val_content_loss: 61.5842 - val_style_loss: 121.2355 - val_total_loss: 182.8197

Epoch 2/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 58.3413 - style_loss: 110.8445 - total_loss: 169.1858
</div>

![png](/img/examples/generative/adain/adain_23_18.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 58.0459 - style_loss: 108.7533 - total_loss: 166.7992 - val_content_loss: 58.5479 - val_style_loss: 110.1745 - val_total_loss: 168.7224

Epoch 3/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 55.2051 - style_loss: 100.6297 - total_loss: 155.8348
</div>

![png](/img/examples/generative/adain/adain_23_28.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 54.5550 - style_loss: 97.7767 - total_loss: 152.3318 - val_content_loss: 56.7784 - val_style_loss: 99.6632 - val_total_loss: 156.4415

Epoch 4/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 52.8162 - style_loss: 89.3740 - total_loss: 142.1902
</div>

![png](/img/examples/generative/adain/adain_23_38.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 52.2626 - style_loss: 88.3876 - total_loss: 140.6502 - val_content_loss: 53.1038 - val_style_loss: 90.5593 - val_total_loss: 143.6631

Epoch 5/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - content_loss: 48.9089 - style_loss: 79.1898 - total_loss: 128.0987
</div>

![png](/img/examples/generative/adain/adain_23_48.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 48.3021 - style_loss: 77.6323 - total_loss: 125.9344 - val_content_loss: 49.3859 - val_style_loss: 76.2716 - val_total_loss: 125.6575

Epoch 6/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 46.5736 - style_loss: 66.7209 - total_loss: 113.2945
</div>

![png](/img/examples/generative/adain/adain_23_58.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 46.2035 - style_loss: 64.3224 - total_loss: 110.5259 - val_content_loss: 46.8296 - val_style_loss: 55.3400 - val_total_loss: 102.1695

Epoch 7/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 45.0892 - style_loss: 49.0773 - total_loss: 94.1665
</div>

![png](/img/examples/generative/adain/adain_23_68.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 43.8027 - style_loss: 45.6226 - total_loss: 89.4253 - val_content_loss: 44.6162 - val_style_loss: 47.6259 - val_total_loss: 92.2422

Epoch 8/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 41.9670 - style_loss: 42.2324 - total_loss: 84.1994
</div>

![png](/img/examples/generative/adain/adain_23_78.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 41.9080 - style_loss: 41.7055 - total_loss: 83.6135 - val_content_loss: 42.9081 - val_style_loss: 44.1262 - val_total_loss: 87.0343

Epoch 9/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 41.0440 - style_loss: 39.7052 - total_loss: 80.7492
</div>

![png](/img/examples/generative/adain/adain_23_88.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 40.8874 - style_loss: 38.1873 - total_loss: 79.0748 - val_content_loss: 42.6833 - val_style_loss: 38.9655 - val_total_loss: 81.6487

Epoch 10/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 40.2837 - style_loss: 33.6737 - total_loss: 73.9575
</div>

![png](/img/examples/generative/adain/adain_23_98.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 39.9428 - style_loss: 34.1020 - total_loss: 74.0449 - val_content_loss: 40.4202 - val_style_loss: 36.6194 - val_total_loss: 77.0396

Epoch 11/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 37.7901 - style_loss: 32.1251 - total_loss: 69.9152
</div>

![png](/img/examples/generative/adain/adain_23_108.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 37.6326 - style_loss: 32.0121 - total_loss: 69.6447 - val_content_loss: 38.7432 - val_style_loss: 33.8374 - val_total_loss: 72.5806

Epoch 12/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 36.8127 - style_loss: 30.6076 - total_loss: 67.4203
</div>

![png](/img/examples/generative/adain/adain_23_118.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 36.4167 - style_loss: 29.9773 - total_loss: 66.3941 - val_content_loss: 36.4050 - val_style_loss: 30.8240 - val_total_loss: 67.2290

Epoch 13/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 33.8249 - style_loss: 27.3473 - total_loss: 61.1722
</div>

![png](/img/examples/generative/adain/adain_23_128.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 33.0191 - style_loss: 26.5015 - total_loss: 59.5206 - val_content_loss: 33.8826 - val_style_loss: 28.0350 - val_total_loss: 61.9177

Epoch 14/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 30.7907 - style_loss: 24.0908 - total_loss: 54.8815
</div>

![png](/img/examples/generative/adain/adain_23_138.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 30.9472 - style_loss: 24.6625 - total_loss: 55.6098 - val_content_loss: 31.0826 - val_style_loss: 25.6283 - val_total_loss: 56.7110

Epoch 15/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 28.0676 - style_loss: 22.0865 - total_loss: 50.1541
</div>

![png](/img/examples/generative/adain/adain_23_148.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 27.8455 - style_loss: 22.1264 - total_loss: 49.9719 - val_content_loss: 28.5455 - val_style_loss: 23.5957 - val_total_loss: 52.1412

Epoch 16/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 26.2761 - style_loss: 21.0945 - total_loss: 47.3706
</div>

![png](/img/examples/generative/adain/adain_23_158.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 25.7858 - style_loss: 21.1223 - total_loss: 46.9081 - val_content_loss: 26.5362 - val_style_loss: 21.9274 - val_total_loss: 48.4636

Epoch 17/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - content_loss: 23.6390 - style_loss: 18.7739 - total_loss: 42.4129
</div>

![png](/img/examples/generative/adain/adain_23_168.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 23.7391 - style_loss: 19.0580 - total_loss: 42.7971 - val_content_loss: 24.9193 - val_style_loss: 20.4144 - val_total_loss: 45.3337

Epoch 18/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - content_loss: 22.4734 - style_loss: 17.9847 - total_loss: 40.4581
</div>

![png](/img/examples/generative/adain/adain_23_178.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 22.4346 - style_loss: 18.0033 - total_loss: 40.4380 - val_content_loss: 23.3949 - val_style_loss: 19.1585 - val_total_loss: 42.5533

Epoch 19/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 21.3290 - style_loss: 17.0995 - total_loss: 38.4285
</div>

![png](/img/examples/generative/adain/adain_23_188.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 21.0454 - style_loss: 16.8622 - total_loss: 37.9076 - val_content_loss: 22.4199 - val_style_loss: 18.0265 - val_total_loss: 40.4464

Epoch 20/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 19.8949 - style_loss: 15.9196 - total_loss: 35.8146
</div>

![png](/img/examples/generative/adain/adain_23_198.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 20.2994 - style_loss: 16.1616 - total_loss: 36.4610 - val_content_loss: 21.6341 - val_style_loss: 17.2532 - val_total_loss: 38.8873

Epoch 21/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 19.9996 - style_loss: 15.7381 - total_loss: 35.7377
</div>

![png](/img/examples/generative/adain/adain_23_208.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 19.5869 - style_loss: 15.2180 - total_loss: 34.8049 - val_content_loss: 20.7572 - val_style_loss: 16.6049 - val_total_loss: 37.3621

Epoch 22/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 19.2298 - style_loss: 15.3685 - total_loss: 34.5983
</div>

![png](/img/examples/generative/adain/adain_23_218.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 18.9491 - style_loss: 14.7553 - total_loss: 33.7044 - val_content_loss: 20.2151 - val_style_loss: 15.9647 - val_total_loss: 36.1798

Epoch 23/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 18.6367 - style_loss: 14.3806 - total_loss: 33.0173

</div>

![png](/img/examples/generative/adain/adain_23_228.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 18.5593 - style_loss: 14.2651 - total_loss: 32.8244 - val_content_loss: 19.5625 - val_style_loss: 15.5629 - val_total_loss: 35.1255

Epoch 24/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 18.5452 - style_loss: 14.5071 - total_loss: 33.0524

</div>

![png](/img/examples/generative/adain/adain_23_238.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 18.0261 - style_loss: 14.1054 - total_loss: 32.1316 - val_content_loss: 19.1599 - val_style_loss: 15.1159 - val_total_loss: 34.2758

Epoch 25/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 17.6185 - style_loss: 13.5009 - total_loss: 31.1194

</div>

![png](/img/examples/generative/adain/adain_23_248.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 17.7612 - style_loss: 13.4830 - total_loss: 31.2442 - val_content_loss: 18.9258 - val_style_loss: 14.6126 - val_total_loss: 33.5383

Epoch 26/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - content_loss: 17.5533 - style_loss: 12.5722 - total_loss: 30.1255

</div>

![png](/img/examples/generative/adain/adain_23_258.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 17.3663 - style_loss: 13.0496 - total_loss: 30.4158 - val_content_loss: 18.3139 - val_style_loss: 14.6291 - val_total_loss: 32.9430

Epoch 27/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 17.6558 - style_loss: 13.8729 - total_loss: 31.5287

</div>

![png](/img/examples/generative/adain/adain_23_268.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 17.2391 - style_loss: 13.4659 - total_loss: 30.7049 - val_content_loss: 18.0461 - val_style_loss: 14.3104 - val_total_loss: 32.3565

Epoch 28/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 17.0192 - style_loss: 13.0646 - total_loss: 30.0838

</div>

![png](/img/examples/generative/adain/adain_23_278.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 16.6298 - style_loss: 12.8305 - total_loss: 29.4603 - val_content_loss: 17.8203 - val_style_loss: 14.0111 - val_total_loss: 31.8314

Epoch 29/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - content_loss: 16.4851 - style_loss: 12.4063 - total_loss: 28.8914

</div>

![png](/img/examples/generative/adain/adain_23_288.png)
    


<div class="k-default-codeblock">

7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 16.2729 - style_loss: 12.4816 - total_loss: 28.7545 - val_content_loss: 17.5812 - val_style_loss: 13.7577 - val_total_loss: 31.3390

Epoch 30/30

7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - content_loss: 15.9445 - style_loss: 12.1650 - total_loss: 28.1095

</div>

![png](/img/examples/generative/adain/adain_23_298.png)
    


<div class="k-default-codeblock">
7/7 ━━━━━━━━━━━━━━━━━━━━ 20s 3s/step - content_loss: 16.0177 - style_loss: 12.2860 - total_loss: 28.3037 - val_content_loss: 17.2777 - val_style_loss: 13.4660 - val_total_loss: 30.7437
</div>

---
## Inference

After we train the model, we now need to run inference with it. We will
pass arbitrary content and style images from the test dataset and take a look at
the output images.

*NOTE*: To try out the model on your own images, you can use this
[Hugging Face demo](https://huggingface.co/spaces/ariG23498/nst).


```python
((style, content),) = test_ds[0]
style_encoded = model.encoder(style)
content_encoded = model.encoder(content)
t = ada_in(style=style_encoded, content=content_encoded)
reconstructed_image = model.decoder(t)
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(10, 30))
[ax.axis("off") for ax in np.ravel(axes)]

for axis, style_image, content_image, reconstructed_image in zip(
    axes, style[0:10], content[0:10], reconstructed_image[0:10]
):
    ax_style, ax_content, ax_reconstructed = axis
    ax_style.imshow(style_image)
    ax_style.set_title("Style Image")
    ax_content.imshow(content_image)
    ax_content.set_title("Content Image")
    ax_reconstructed.imshow(reconstructed_image)
    ax_reconstructed.set_title("NST Image")
```


    
![png](/img/examples/generative/adain/adain_25_0.png)
    


---
## Conclusion

Adaptive Instance Normalization allows arbitrary style transfer in
real time. It is also important to note that the novel proposition of
the authors is to achieve this only by aligning the statistical
features (mean and standard deviation) of the style and the content
images.

*Note*: AdaIN also serves as the base for
[Style-GANs](https://arxiv.org/abs/1812.04948).

---
## Reference

- [TF implementation](https://github.com/ftokarev/tf-adain)

---
## Acknowledgement

We thank [Luke Wood](https://lukewood.xyz) for his
detailed review.

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
