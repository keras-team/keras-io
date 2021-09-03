"""
Title: MelGAN Based Spectrogram Inversion using Feature Matching
Author: [Darshan Deshpande](https://twitter.com/getdarshan)
Date created: 02/09/2021
Last modified: 02/09/2021
Description: Inversion of audio from mel-spectrograms using the MelGAN architecture and feature matching.
"""
"""
## Introduction

Autoregressive vocoders have been ubiquitous for a majority of the speech processing history
but they lack parallelism. [MelGAN](https://arxiv.org/pdf/1910.06711v3.pdf) is a
non-autoregressive, fully convolutional vocoder architecture used for purposes ranging
from spectral inversion and speech enhancement to speech synthesis when used as a decoder
with models like Tacotron2 or FastSpeech.

In this tutorial, we will have a look at the MelGAN architecture and how it can achieve
fast spectral inversion, i.e. conversion of spectrograms to audio waves. The MelGAN
implemented in this tutorial is similar to the original implementation with only the
difference of method of padding for convolutions where we will use 'same' instead of
reflect padding.
"""

"""
## Imports
"""

"""shell
!pip install -qqq tensorflow_addons
!pip install -qqq tensorflow-io
"""

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import librosa.display
import glob
from tensorflow.keras.layers import *
from tensorflow_addons.layers import WeightNormalization
from sklearn.model_selection import train_test_split

# Defining hyperparameters

DESIRED_SAMPLES = 8192
TEST_SPLIT = 0.2
LEARNING_RATE_GEN = 1e-5
LEARNING_RATE_DISC = 1e-6
BATCH_SIZE = 16

mse = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
)
mae = tf.keras.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
)

"""
## Loading the Dataset

This example uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

The LJSpeech dataset is primarily used for text-to-speech and consists of 13,100 discrete
speech samples taken from 7 non-fiction books, having a total length of approximately 24
hours. The MelGAN training is only concerned with the audio waves so we process only the
WAV files and ignore the audio annotations.
"""

"""shell
!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xf /content/LJSpeech-1.1.tar.bz2
"""

# Splitting the dataset into training and testing splits
wavs = glob.glob("LJSpeech-1.1/wavs/*.wav")
train, test = train_test_split(wavs, test_size=TEST_SPLIT)

print(f"Number of training audio files: {len(train)}")
print(f"Number of testing audio files: {len(test)}")

# Mapper function for loading the audio. This function returns two instances of the wave


def preprocess(filename):
    audio = tf.audio.decode_wav(tf.io.read_file(filename), 1, DESIRED_SAMPLES).audio
    return audio, audio


# Create tf.data.Datasets and mapping the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train,)
train_dataset = train_dataset.map(preprocess, tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(test,)
test_dataset = test_dataset.map(preprocess, tf.data.AUTOTUNE)

"""
## Defining custom layers for MelGAN

The MelGAN architecture consists of 3 main modules:
1. The Residual Block
2. Dilated Conv Block
3. Discriminator Block
![MelGAN](https://i.imgur.com/ZdxwzPG.png)
"""

"""
Since the network inputs a mel-spectrogram, we will create an additional custom layer
which can convert the raw audio wave to a spectrogram on-the-fly.
"""

# Custom keras layer for on-the-fly audio to spectrogram conversion


class MelSpec(tf.keras.layers.Layer):
    def __init__(
        self,
        frame_length=1024,
        frame_step=256,
        fft_length=None,
        sampling_rate=22050,
        n_mel_channels=80,
        fmin=0,
        fmax=7600,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.fmin = fmin
        self.fmax = fmax
        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mel_channels,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
        )

    def call(self, audio):
        # Taking the Short Time Fourier Transform. Ensure that the audio is padded.
        # In the paper, the STFT output is padded using the 'REFLECT' strategy.
        stft = tf.signal.stft(
            tf.squeeze(audio, -1),
            self.frame_length,
            self.frame_step,
            self.fft_length,
            pad_end=True,
        )

        # Taking the magnitude of the STFT output
        magnitude = tf.abs(stft)

        # Multiplying the Mel-filterbank with the magnitude and scaling it using the db scale
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        log_mel_spec = tfio.audio.dbscale(mel, 80)
        return log_mel_spec

    def get_config(self):
        config = super(MelSpec, self).get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "n_mel_channels": self.n_mel_channels,
                "fmin": self.fmin,
                "fmax": self.fmax,
            }
        )
        return config

"""
The Residual Convolutional Block extensively uses dilations and has a total receptive
field of 27 timesteps per block.
"""

# Creating the Residual Stack Layer

class ResidualStack(tf.keras.layers.Layer):
    """
  Convolutional residual stack with weight normalization.

  input: filter: int, determines filter size for the residual stack.
	output: convolutional stack output.
  """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.c1 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
        )
        self.c2 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
        )
        self.c3 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=3, padding="same"), data_init=False
        )
        self.c4 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
        )
        self.c5 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=9, padding="same"), data_init=False
        )
        self.c6 = WeightNormalization(
            Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
        )
        self.lrelu = LeakyReLU

    def call(self, input):
        x1 = self.c1(input)
        x2 = self.lrelu()(x1)
        x3 = self.c2(x2)
        x4 = x3 + input

        x5 = self.lrelu()(x4)
        x6 = self.c3(x5)
        x7 = self.lrelu()(x6)
        x8 = self.c4(x7)
        x9 = x4 + x8

        x10 = self.lrelu()(x9)
        x11 = self.c5(x10)
        x12 = self.lrelu()(x11)
        x13 = self.c6(x12)
        x14 = x9 + x13

        return x14

    def get_config(self):
        config = super(ResidualStack, self).get_config()
        config.update({"filters": self.filters})
        return config


"""
The Dilated Convolutional Block uses the dilations offered by the residual stack
and upsamples the input data by the `upsampling_factor`.
"""

# Dilated Convolutional Block


class DilatedConvBlock(tf.keras.layers.Layer):
    """
  Dilated Convolutional Block with weight normalization.

  input:  conv_dim: int, determines filter size for the block
          upsampling_factor: int, scale for upsampling

  output: keras.layer, convolutional stack output
  """

    def __init__(self, conv_dim, upsampling_factor, **kwargs):
        super().__init__(**kwargs)
        self.conv_dim = conv_dim
        self.upsampling_factor = upsampling_factor
        self.conv_t = WeightNormalization(
            Conv1DTranspose(self.conv_dim, 16, self.upsampling_factor, padding="same"),
            data_init=False,
        )
        self.lrelu1 = LeakyReLU()
        self.residual_stack = ResidualStack(self.conv_dim)
        self.lrelu2 = LeakyReLU()

    def call(self, input):
        out = self.conv_t(input)
        out = self.lrelu1(out)
        out = self.residual_stack(out)
        out = self.lrelu2(out)
        return out

    def get_config(self):
        config = super(DilatedConvBlock, self).get_config()
        config.update(
            {"conv_dim": self.conv_dim, "upsampling_factor": self.upsampling_factor}
        )
        return config


"""
The discriminator block consists of convolutions and downsampling layers. This block is
essential for the implementation of the feature matching technique.

Each discriminator outputs a list of feature maps that will be compared during training
to calculate the feature matching loss.
"""


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.convs = []
        self.convs.append(
            WeightNormalization(Conv1D(16, 15, 1, "same"), data_init=False)
        )
        self.convs.append(
            WeightNormalization(Conv1D(64, 41, 4, "same", groups=4), data_init=False)
        )
        self.convs.append(
            WeightNormalization(Conv1D(256, 41, 4, "same", groups=16), data_init=False)
        )
        self.convs.append(
            WeightNormalization(Conv1D(1024, 41, 4, "same", groups=64), data_init=False)
        )
        self.convs.append(
            WeightNormalization(
                Conv1D(1024, 41, 4, "same", groups=256), data_init=False
            )
        )
        self.convs.append(
            WeightNormalization(Conv1D(1024, 5, 1, "same"), data_init=False)
        )
        self.convs.append(WeightNormalization(Conv1D(1, 3, 1, "same"), data_init=False))
        self.lrelu = LeakyReLU

    def call(self, input):
        out_list = []
        for index, conv in enumerate(self.convs):
            out = conv(input if index == 0 else out)
            if index != len(self.convs) - 1:
                out = self.lrelu()(out)
            out_list.append(out)
        return out_list


"""
### Create the generator
"""


def create_generator(input_shape):
    inp = Input(input_shape)
    x = MelSpec()(inp)
    x = Conv1D(512, 7, padding="same")(x)
    x = LeakyReLU()(x)
    x = DilatedConvBlock(256, 8)(x)
    x = DilatedConvBlock(128, 8)(x)
    x = DilatedConvBlock(64, 2)(x)
    x = DilatedConvBlock(32, 2)(x)
    x = WeightNormalization(Conv1D(1, 7, padding="same", activation="tanh"))(x)
    return tf.keras.models.Model(inp, x)


# We use a dynamic input shape for the generator since the model is fully convolutional
generator = create_generator((None, 1))
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True)

"""
### Create the discriminator
"""

# We use a dynamic input shape for the discriminator
# This is done because the input shape for the generator is unknown


def create_discriminator(input_shape):
    inp = Input(input_shape)
    out_map1 = DiscriminatorBlock()(inp)
    pool1 = AveragePooling1D()(inp)
    out_map2 = DiscriminatorBlock()(pool1)
    pool2 = AveragePooling1D()(pool1)
    out_map3 = DiscriminatorBlock()(pool2)
    return tf.keras.models.Model(inp, [out_map1, out_map2, out_map3])


discriminator = create_discriminator((None, 1))
discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

"""
## Defining the loss functions

**Generator Loss**

The generator architecture uses a combination of two losses

1. MeanSquaredError:

This is the standard MSE generator loss calculated between ones and the outputs from the
discriminator

\begin{align}
Loss = ||1 - D(G(s))||^2
\end{align}

2. Feature Matching Loss:

This loss involves extracting the outputs of every layer from the discriminator for both
the generator and ground truth and compare each layer output using Mean Squared Error.

\begin{align}
Loss_{fm} = \frac{1}{N} \sum_{i=0}^N ||D_k^{(i)}(x) - D_k^{(i)}(G(s))||_1
\end{align}

**Discriminator Loss**

The discriminator uses the Mean Absolute Error and compares the real data predictions
with ones and generated predictions with zeros.

\begin{align}
Loss = \frac{1}{N} \sum_{i=0}^N (||1 -D_k^{(i)}(x)||^2 + ||0 - D_k^{(i)}(G(s))||)
\end{align}


"""

# Generator loss


def generator_loss(real_pred, fake_pred):
    """
inputs: real_pred: tensor, Output of the ground truth wave passed through the
discriminator.
fake_pred: tensor, Output of the generator prediction passed through the
discriminator.

  outputs: generator loss: int, Loss for the generator
  """
    gen_loss = 0
    for i in range(len(fake_pred)):
        gen_loss += mse(tf.ones_like(fake_pred[i][-1]), fake_pred[i][-1])

    # Scaling the loss by the number of layers
    gen_loss /= i + 1

    # Feature matching loss
    fm_loss = 0
    lambda_feature = 10
    for i in range(len(fake_pred)):
        for j in range(len(fake_pred[i]) - 1):
            fm_loss += mae(real_pred[i][j], fake_pred[i][j])

    # Calculating final generator loss
    gen_loss += lambda_feature * fm_loss / ((i + 1) * (j + 1))

    return gen_loss


def discriminator_loss(real_pred, fake_pred):
    """
inputs: real_pred: tensor, Output of the ground truth wave passed through the
discriminator.
fake_pred: tensor, Output of the generator prediction passed through the
discriminator.

  outputs: discriminator loss: int, Loss for the discriminator
  """
    real_loss, fake_loss = 0, 0
    for i in range(len(real_pred)):
        real_loss += mse(tf.ones_like(real_pred[i][-1]), real_pred[i][-1])
        fake_loss += mse(tf.zeros_like(fake_pred[i][-1]), fake_pred[i][-1])

    # Calculating the final discriminator loss after scaling
    disc_loss = real_loss / (i + 1) + fake_loss / (i + 1)
    return disc_loss


class MelGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, **kwargs):
        """
    inputs: generator: keras.model.Model, Generator model
            discriminator: keras.model.Model, Discriminator model
    """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(
        self, gen_optimizer, disc_optimizer, generator_loss, discriminator_loss
    ):
        """
    inputs: gen_optimizer: keras.optimizer, Optimizer to be used for training
            disc_optimizer: keras.optimizer, Optimizer to be used for training
            generator_loss: callable, Loss function for generator
            discriminator_loss: callable, Loss function for discriminator
    """
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def train_step(self, batch):
        x_batch_train, y_batch_train = batch

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generating the audio wave
            gen_audio_wave = generator(x_batch_train, training=True)

            # Generating the features using the discriminator
            fake_pred = discriminator(y_batch_train)
            real_pred = discriminator(gen_audio_wave)

            # Calculating the losses
            gen_loss = generator_loss(real_pred, fake_pred)
            disc_loss = discriminator_loss(real_pred, fake_pred)

        # Calculating and applying the gradients for generator and discriminator
        grads_gen = gen_tape.gradient(gen_loss, generator.trainable_weights)
        grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
        gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_weights))
        disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


"""
## Training

The paper suggests that the training with dynamic shapes takes around 400,000 steps. For
this example, we will run it only for three epochs
"""

gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_GEN, beta_1=0.5, beta_2=0.9, clipnorm=1)
disc_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_DISC, beta_1=0.5, beta_2=0.9, clipnorm=1)

# Start training
generator = create_generator((None, 1))
discriminator = create_discriminator((None, 1))

mel_gan = MelGAN(generator, discriminator)
mel_gan.compile(gen_optimizer, disc_optimizer, generator_loss, discriminator_loss)
mel_gan.fit(train_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), epochs=3)

"""
# Conclusion

The MelGAN is a highly effective algorithm for spectral inversion and beats the Griffin
Lim algorithm by a Mean Opinion Score difference of 2.95 which is a considerable
improvement. In contrast with the purely algortithmic techniques, the MelGAN outperforms
the WaveGlow and WaveNet architectures on text-to-speech and speech enhancement tasks on
the LJSpeech and VCTK datasets.

Further reading

1. MelGAN paper to understand the reasoning behind the architecture and training process

2. For in depth understanding of the feature matching loss, you can refer to [Improved
Techniques for Training GANs](https://arxiv.org/pdf/1606.03498v1.pdf) (Tim Salimans et
al.).
"""
