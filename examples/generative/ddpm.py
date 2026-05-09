"""
Title: Denoising Diffusion Probabilistic Model
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2022/11/30
Last modified: 2026/05/09
Description: Generating images of flowers with denoising diffusion probabilistic models.

Converted to Keras 3 by: [Maitry Sinha](https://github.com/maitry63)
"""

"""
## Introduction

Generative modeling experienced tremendous growth in the last five years. Models like
VAEs, GANs, and flow-based models proved to be a great success in generating
high-quality content, especially images. Diffusion models are a new type of generative
model that has proven to be better than previous approaches.

Diffusion models are inspired by non-equilibrium thermodynamics, and they learn to
generate by denoising. Learning by denoising consists of two processes,
each of which is a Markov Chain. These are:

1. The forward process: In the forward process, we slowly add random noise to the data
in a series of time steps `(t1, t2, ..., tn )`. Samples at the current time step are
drawn from a Gaussian distribution where the mean of the distribution is conditioned
on the sample at the previous time step, and the variance of the distribution follows
a fixed schedule. At the end of the forward process, the samples end up with a pure
noise distribution.

2. The reverse process: During the reverse process, we try to undo the added noise at
every time step. We start with the pure noise distribution (the last step of the
forward process) and try to denoise the samples in the backward direction
`(tn, tn-1, ..., t1)`.

We implement the [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
paper or DDPMs for short in this code example. It was the first paper demonstrating
the use of diffusion models for generating high-quality images. The authors proved
that a certain parameterization of diffusion models reveals an equivalence with
denoising score matching over multiple noise levels during training and with annealed
Langevin dynamics during sampling that generates the best quality results.

This paper replicates both the Markov chains (forward process and reverse process)
involved in the diffusion process but for images. The forward process is fixed and
gradually adds Gaussian noise to the images according to a fixed variance schedule
denoted by beta in the paper. This is what the diffusion process looks like in case
of images: (image -> noise::noise -> image)

![diffusion process gif](https://imgur.com/Yn7tho9.gif)


The paper describes two algorithms, one for training the model, and the other for
sampling from the trained model. Training is performed by optimizing the usual
variational bound on negative log-likelihood. The objective function is further
simplified, and the network is treated as a noise prediction network. Once optimized,
we can sample from the network to generate new images from noise samples. Here is an
overview of both algorithms as presented in the paper:

![ddpms](https://i.imgur.com/S7KH5hZ.png)


**Note:** DDPM is just one way of implementing a diffusion model. Also, the sampling
algorithm in the DDPM replicates the complete Markov chain. Hence, it's slow in
generating new samples compared to other generative models like GANs. Lots of research
efforts have been made to address this issue. One such example is Denoising Diffusion
Implicit Models, or DDIM for short, where the authors replaced the Markov chain with a
non-Markovian process to sample faster. You can find the code example for DDIM
[here](https://keras.io/examples/generative/ddim/)

Implementing a DDPM model is simple. We define a model that takes
two inputs: Images and the randomly sampled time steps. At each training step, we
perform the following operations to train our model:

1. Sample random noise to be added to the inputs.
2. Apply the forward process to diffuse the inputs with the sampled noise.
3. Your model takes these noisy samples as inputs and outputs the noise
prediction for each time step.
4. Given true noise and predicted noise, we calculate the loss values
5. We then calculate the gradients and update the model weights.

Given that our model knows how to denoise a noisy sample at a given time step,
we can leverage this idea to generate new samples, starting from a pure noise
distribution.
"""

"""
## Setup
"""

import os
import tarfile
import requests
from pathlib import Path
from PIL import Image
import random as py_random
import numpy as np
import matplotlib.pyplot as plt
import keras

keras.backend.set_floatx("float32")
from keras import layers
from keras import ops
from keras import random

"""
## Hyperparameters
"""

# batch_size = 32
batch_size = 4
num_epochs = 1  # Just for the sake of demonstration
# total_timesteps = 1000
total_timesteps = 10
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

dataset_name = "oxford_flowers102"
splits = ["train"]


"""
## Dataset

We use the [Oxford Flowers 102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
dataset for generating images of flowers. In terms of preprocessing, we use center
cropping for resizing the images to the desired image size, and we rescale the pixel
values in the range `[-1.0, 1.0]`. This is in line with the range of the pixel values that
was applied by the authors of the [DDPMs paper](https://arxiv.org/abs/2006.11239). For
augmenting training data, we randomly flip the images left/right.
"""


# Load the dataset
dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
dataset_dir = "oxford_flowers102"

os.makedirs(dataset_dir, exist_ok=True)
tar_path = os.path.join(dataset_dir, "102flowers.tgz")

if not os.path.exists(tar_path):
    print("Downloading dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Extract images
extracted_path = os.path.join(dataset_dir, "jpg")
if not os.path.exists(extracted_path):
    print("Extracting images...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dataset_dir)
    print("Extraction complete.")

# Load images into a list
image_paths = sorted(Path(extracted_path).glob("*.jpg"))
print(f"Found {len(image_paths)} images.")

images = []
for path in image_paths:
    img = Image.open(path).convert("RGB")
    images.append(np.array(img))


# Preprocessing
def augment(img):
    """Flips an image left/right randomly."""
    if py_random.random() < 0.5:
        img = np.fliplr(img)
    return img


def resize_and_rescale(img, size=(img_size, img_size)):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [-1.0, 1.0].

    Args:
        img: Image tensor
        size: Desired image size for resizing
    Returns:
        Resized and rescaled image tensor
    """
    h, w = img.shape[:2]
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    img = img[start_h : start_h + crop_size, start_w : start_w + crop_size]

    img = np.array(Image.fromarray(img).resize(size, Image.BILINEAR))

    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.clip(img, clip_min, clip_max)
    return img


def preprocess_image(img):
    img = resize_and_rescale(img)
    img = augment(img)
    return img


# PyDataset Generator
class PyDataset:
    def __init__(self, images, batch_size, shuffle=True):
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        self.indices = np.arange(self.num_samples)
        self.current_idx = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        idxs = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch = np.array(
            [preprocess_image(self.images[i]) for i in idxs], dtype=np.float32
        )
        self.current_idx += self.batch_size
        return batch


train_ds = PyDataset(images, batch_size=batch_size, shuffle=True)


"""
## Gaussian diffusion utilities

We define the forward process and the reverse process
as a separate utility. Most of the code in this utility has been borrowed
from the original implementation with some slight modifications.
"""


class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        betas = np.linspace(beta_start, beta_end, timesteps).astype("float32")
        alphas = (1.0 - betas).astype("float32")
        alphas_cumprod = np.cumprod(alphas, axis=0).astype("float32")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]).astype("float32")

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod).astype("float32")
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod).astype(
            "float32"
        )
        self.log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod).astype(
            "float32"
        )
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod).astype("float32")
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0).astype(
            "float32"
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).astype("float32")
        self.posterior_variance = posterior_variance

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.maximum(posterior_variance, 1e-20)
        ).astype("float32")

        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).astype("float32")

        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).astype("float32")

    def _extract(self, arr, t, batch_size):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        out = ops.take(arr, t)
        # Reshape for broadcasting: [batch_size, 1, 1, 1]
        return ops.cast(ops.reshape(out, (batch_size, 1, 1, 1)), dtype="float32")

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        batch_size = x_start.shape[0]
        mean = self._extract(self.sqrt_alphas_cumprod, t, batch_size) * x_start
        var = self._extract(1.0 - self.alphas_cumprod, t, batch_size)
        log_var = self._extract(self.log_one_minus_alphas_cumprod, t, batch_size)
        return mean, var, log_var

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        batch_size = x_start.shape[0]
        return (
            self._extract(self.sqrt_alphas_cumprod, t, batch_size) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_size) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        batch_size = x_t.shape[0]
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, batch_size) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, batch_size) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """
        batch_size = x_t.shape[0]
        mean = (
            self._extract(self.posterior_mean_coef1, t, batch_size) * x_start
            + self._extract(self.posterior_mean_coef2, t, batch_size) * x_t
        )
        var = self._extract(self.posterior_variance, t, batch_size)
        log_var = self._extract(self.posterior_log_variance_clipped, t, batch_size)
        return mean, var, log_var

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x_recon = ops.clip(x_recon, self.clip_min, self.clip_max)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        mean, var, log_var = self.p_mean_variance(pred_noise, x, t, clip_denoised)
        noise = keras.random.normal(shape=x.shape, dtype="float32")
        # No noise when t == 0
        # nonzero_mask = (t != 0).astype(x.dtype).reshape(-1, 1, 1, 1)
        nonzero_mask = ops.cast(t != 0, dtype=ops.dtype(x))
        nonzero_mask = ops.reshape(nonzero_mask, (-1, 1, 1, 1))
        return mean + nonzero_mask * ops.exp(0.5 * log_var) * noise


"""
## Network architecture

U-Net, originally developed for semantic segmentation, is an architecture that is
widely used for implementing diffusion models but with some slight modifications:

1. The network accepts two inputs: Image and time step
2. Self-attention between the convolution blocks once we reach a specific resolution
(16x16 in the paper)
3. Group Normalization instead of weight normalization

We implement most of the things as used in the original paper. We use the
`swish` activation function throughout the network. We use the variance scaling
kernel initializer.

The only difference here is the number of groups used for the
`GroupNormalization` layer. For the flowers dataset,
we found that a value of `groups=8` produces better results
compared to the default value of `groups=32`. Dropout is optional and should be
used where chances of over fitting is high. In the paper, the authors used dropout
only when training on CIFAR10.
"""


# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    def __init__(self, query_dim, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.query_dim = query_dim
        self.groups = groups

        self.group_norm = layers.GroupNormalization(groups=groups)
        self.q_proj = layers.Dense(query_dim)
        self.k_proj = layers.Dense(query_dim)
        self.v_proj = layers.Dense(query_dim)
        self.out_proj = layers.Dense(query_dim)

    def build(self, input_shape):
        channels = input_shape[-1]

        self.group_norm.build(input_shape)

        flat_shape = (input_shape[0], input_shape[1] * input_shape[2], channels)
        self.q_proj.build(flat_shape)
        self.k_proj.build(flat_shape)
        self.v_proj.build(flat_shape)
        self.out_proj.build(flat_shape)

        self.built = True

    def call(self, x):
        batch_size = ops.shape(x)[0]
        h = ops.shape(x)[1]
        w = ops.shape(x)[2]

        res = x
        x = self.group_norm(x)

        x_flat = ops.reshape(x, (batch_size, h * w, -1))

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        scale = ops.sqrt(ops.cast(self.query_dim, x.dtype))
        k_t = ops.transpose(k, axes=(0, 2, 1))
        attn_logits = ops.matmul(q, k_t) / scale
        attn_weights = ops.softmax(attn_logits, axis=-1)

        attn_out = ops.matmul(attn_weights, v)
        x = self.out_proj(attn_out)
        x = ops.reshape(x, (batch_size, h, w, -1))

        return x + res

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_dim": self.query_dim,
                "groups": self.groups,
            }
        )
        return config


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2

        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (np.arange(self.half_dim) / self.half_dim))
        self.inv_freq_const = inv_freq.astype("float32")

    def call(self, t):
        t = ops.cast(t, dtype="float32")
        t = ops.expand_dims(t, axis=-1)
        emb = t * self.inv_freq_const
        return ops.concatenate([ops.sin(emb), ops.cos(emb)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t_emb = inputs
        input_width = x.shape[3]

        residual = (
            x
            if input_width == width
            else layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)
        )

        temb = activation_fn(t_emb)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)
        temb = temb[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)

        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(t_emb):
        t_emb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(t_emb)
        t_emb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(t_emb)
        return t_emb

    return apply


def build_model(
    img_size,
    img_channels,
    widths,
    has_attention,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):

    image_input = layers.Input(shape=(img_size, img_size, img_channels))
    time_input = layers.Input(shape=(), dtype=np.int32)

    x = layers.Conv2D(
        widths[0], kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
    )(image_input)

    t_emb = TimeEmbedding(dim=widths[0] * 4)(time_input)
    t_emb = TimeMLP(units=widths[0] * 4, activation_fn=activation_fn)(t_emb)

    skips = [x]

    # DownBlock
    for i, width in enumerate(widths):
        for _ in range(num_res_blocks):
            x = ResidualBlock(width, groups=norm_groups, activation_fn=activation_fn)(
                [x, t_emb]
            )
            if has_attention[i]:
                x = AttentionBlock(width, groups=norm_groups)(x)
            skips.append(x)
        if i != len(widths) - 1:
            x = DownSample(width)(x)
            skips.append(x)

    # Middle block
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, t_emb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, t_emb]
    )

    # Up block
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, t_emb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(
        img_channels, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
    )(x)

    return keras.Model([image_input, time_input], x, name="unet")


"""
## Training

We follow the same setup for training the diffusion model as described
in the paper. We use `Adam` optimizer with a learning rate of `2e-4`.
We use EMA on model parameters with a decay factor of 0.999. We
treat our model as noise prediction network i.e. at every training step, we
input a batch of images and corresponding time steps to our UNet,
and the network outputs the noise as predictions.

The only difference is that we aren't using the Kernel Inception Distance (KID)
or Frechet Inception Distance (FID) for evaluating the quality of generated
samples during training. This is because both these metrics are compute heavy
and are skipped for the brevity of implementation.

**Note: ** We are using mean squared error as the loss function which is aligned with
the paper, and theoretically makes sense. In practice, though, it is also common to
use mean absolute error or Huber loss as the loss function.
"""


class DiffusionModel:
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, lr=1e-4):
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )

    def update_ema(self):
        model_weights = self.network.get_weights()
        ema_weights = self.ema_network.get_weights()

        new_weights = [
            self.ema * ew + (1.0 - self.ema) * w
            for w, ew in zip(model_weights, ema_weights)
        ]
        self.ema_network.set_weights(new_weights)

    def train_step(self, images):
        batch_size = ops.shape(images)[0]

        # 1. Sample timesteps
        t = random.randint(
            shape=(batch_size,),
            minval=0,
            maxval=self.timesteps,
        )

        # 2. Sample Gaussian noise
        noise = random.normal(shape=ops.shape(images))

        # 3. Forward diffusion (get noisy images)
        images_t = self.gdf_util.q_sample(images, t, noise)

        # 4 & 5 & 6. Train on batch (runs forward + backprop + returns loss)
        # This is much faster than running predict() then training separately
        loss = self.network.train_on_batch([images_t, t], noise)

        # 7. EMA update
        self.update_ema()

        return loss

    def generate_images(self, num_images=16):
        img_shape = self.network.input_shape[0][1:]

        # Start from pure noise
        samples = random.normal(shape=(num_images, *img_shape))

        for t in reversed(range(self.timesteps)):
            tt = ops.full((num_images,), t, dtype="int32")

            # Use EMA model
            pred_noise = self.ema_network.predict([samples, tt], verbose=0)

            # Reverse diffusion step
            samples = self.gdf_util.p_sample(pred_noise, samples, tt)

        return ops.convert_to_numpy(samples)

    def plot_images(self, num_rows=2, num_cols=8, figsize=(12, 5)):
        generated = self.generate_images(num_rows * num_cols)
        generated = np.clip((generated + 1.0) * 127.5, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        ax = ax.flatten()

        for i, img in enumerate(generated):
            ax[i].imshow(img)
            ax[i].axis("off")

        plt.tight_layout()
        plt.show()


# Build models
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

# Initialize EMA network with same weights
ema_network.set_weights(network.get_weights())

# Initialize model wrapper
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=GaussianDiffusion(timesteps=total_timesteps),
    timesteps=total_timesteps,
    ema=0.999,
    lr=learning_rate,
)

# Training Loop
for epoch in range(num_epochs):
    losses = []
    for batch in train_ds:
        loss = model.train_step(batch)
        losses.append(loss)

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        model.plot_images()

"""
## Results

We trained this model for 800 epochs on a V100 GPU,
and each epoch took almost 8 seconds to finish. We load those weights
here, and we generate a few samples starting from pure noise.
"""

"""shell
curl -LO https://github.com/AakashKumarNain/ddpms/releases/download/v3.0.0/checkpoints.zip
unzip -qq checkpoints.zip
"""

# # Load the model weights
# model.ema_network.load_weights("checkpoints/diffusion_model_checkpoint")

# Generate and plot some samples
model.plot_images(num_rows=4, num_cols=8)


"""
## Conclusion

We successfully implemented and trained a diffusion model exactly in the same
fashion as implemented by the authors of the DDPMs paper. You can find the
original implementation [here](https://github.com/hojonathanho/diffusion).

There are a few things that you can try to improve the model:

1. Increasing the width of each block. A bigger model can learn to denoise
in fewer epochs, though you may have to take care of overfitting.

2. We implemented the linear schedule for variance scheduling. You can implement
other schemes like cosine scheduling and compare the performance.
"""

"""
## References

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [Author's implementation](https://github.com/hojonathanho/diffusion)
3. [A deep dive into DDPMs](https://magic-with-latents.github.io/latent/posts/ddpms/part3/)
4. [Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/)
5. [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
6. [AIAIART](https://www.youtube.com/watch?v=XTs7M6TSK9I&t=14s)
"""

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
"""
