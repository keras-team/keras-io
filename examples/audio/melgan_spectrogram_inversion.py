# NOTE: place this file under `examples/` in your repository
# Minimal, self-contained improvements to the MelGAN example:
# - Fixes train_step to use instance attributes (self.generator, self.gen_optimizer, ...)
# - Adds checkpointing and TensorBoard callback examples
# - Adds a simple CLI to train or run a quick inference
# - Adds @tf.function to train_step
#
# Use: python examples/melgan_improved.py --mode train
#      python examples/melgan_improved.py --mode infer

import os
import argparse
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import layers as addon_layers

tf.get_logger().setLevel("ERROR")
DESIRED_SAMPLES = 8192
LEARNING_RATE_GEN = 1e-5
LEARNING_RATE_DISC = 1e-6
BATCH_SIZE = 16
EPOCHS = 1

mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()


def load_wav_paths(root="LJSpeech-1.1/wavs/*.wav"):
    return tf.io.gfile.glob(root)


def preprocess(filename):
    audio = tf.audio.decode_wav(tf.io.read_file(
        filename), desired_channels=1, desired_samples=DESIRED_SAMPLES).audio
    return audio, audio


class MelSpec(layers.Layer):
    def __init__(
        self,
        frame_length=1024,
        frame_step=256,
        fft_length=None,
        sampling_rate=22050,
        num_mel_channels=80,
        freq_min=125,
        freq_max=7600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, audio, training=True):
        # During training the input is raw audio (shape [batch, samples, 1])
        # During inference we expect a precomputed mel-spectrogram (shape [batch, time, mel_bins])
        if training:
            # Remove channel dim and compute STFT
            stft = tf.signal.stft(
                tf.squeeze(audio, -1),
                self.frame_length,
                self.frame_step,
                self.fft_length,
                pad_end=True,
            )
            magnitude = tf.abs(stft)
            mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
            log_mel_spec = tfio.audio.dbscale(mel, top_db=80)
            return log_mel_spec
        else:
            # Pass-through: assume the input is already a mel-spectrogram
            return audio

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_channels": self.num_mel_channels,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config


def residual_stack(input, filters):
    c1 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
    )(input)
    lrelu1 = layers.LeakyReLU()(c1)
    c2 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
    )(lrelu1)
    add1 = layers.Add()([c2, input])

    lrelu2 = layers.LeakyReLU()(add1)
    c3 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=3, padding="same"), data_init=False
    )(lrelu2)
    lrelu3 = layers.LeakyReLU()(c3)
    c4 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
    )(lrelu3)
    add2 = layers.Add()([add1, c4])

    lrelu4 = layers.LeakyReLU()(add2)
    c5 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=9, padding="same"), data_init=False
    )(lrelu4)
    lrelu5 = layers.LeakyReLU()(c5)
    c6 = addon_layers.WeightNormalization(
        layers.Conv1D(filters, 3, dilation_rate=1, padding="same"), data_init=False
    )(lrelu5)
    add3 = layers.Add()([c6, add2])

    return add3


def conv_block(input, conv_dim, upsampling_factor):
    conv_t = addon_layers.WeightNormalization(
        layers.Conv1DTranspose(
            conv_dim, 16, upsampling_factor, padding="same"),
        data_init=False,
    )(input)
    lrelu1 = layers.LeakyReLU()(conv_t)
    res_stack = residual_stack(lrelu1, conv_dim)
    lrelu2 = layers.LeakyReLU()(res_stack)
    return lrelu2


def discriminator_block(input):
    conv1 = addon_layers.WeightNormalization(
        layers.Conv1D(16, 15, 1, "same"), data_init=False
    )(input)
    lrelu1 = layers.LeakyReLU()(conv1)
    conv2 = addon_layers.WeightNormalization(
        layers.Conv1D(64, 41, 4, "same", groups=4), data_init=False
    )(lrelu1)
    lrelu2 = layers.LeakyReLU()(conv2)
    conv3 = addon_layers.WeightNormalization(
        layers.Conv1D(256, 41, 4, "same", groups=16), data_init=False
    )(lrelu2)
    lrelu3 = layers.LeakyReLU()(conv3)
    conv4 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 41, 4, "same", groups=64), data_init=False
    )(lrelu3)
    lrelu4 = layers.LeakyReLU()(conv4)
    conv5 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 41, 4, "same", groups=256), data_init=False
    )(lrelu4)
    lrelu5 = layers.LeakyReLU()(conv5)
    conv6 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 5, 1, "same"), data_init=False
    )(lrelu5)
    lrelu6 = layers.LeakyReLU()(conv6)
    conv7 = addon_layers.WeightNormalization(
        layers.Conv1D(1, 3, 1, "same"), data_init=False
    )(lrelu6)
    return [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]


def create_generator(input_shape):
    inp = keras.Input(input_shape)
    x = MelSpec()(inp)
    x = layers.Conv1D(512, 7, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = conv_block(x, 256, 8)
    x = conv_block(x, 128, 8)
    x = conv_block(x, 64, 2)
    x = conv_block(x, 32, 2)
    x = addon_layers.WeightNormalization(
        layers.Conv1D(1, 7, padding="same", activation="tanh")
    )(x)
    return keras.Model(inp, x, name="generator")


def create_discriminator(input_shape):
    inp = keras.Input(input_shape)
    out_map1 = discriminator_block(inp)
    pool1 = layers.AveragePooling1D()(inp)
    out_map2 = discriminator_block(pool1)
    pool2 = layers.AveragePooling1D()(pool1)
    out_map3 = discriminator_block(pool2)
    return keras.Model(inp, [out_map1, out_map2, out_map3], name="discriminator")


def generator_loss_fn(real_pred, fake_pred):
    losses = [mse(tf.ones_like(fp[-1]), fp[-1]) for fp in fake_pred]
    return tf.reduce_mean(losses)


def feature_matching_loss_fn(real_pred, fake_pred):
    fm_loss = []
    for r, f in zip(real_pred, fake_pred):
        for j in range(len(f) - 1):
            fm_loss.append(mae(r[j], f[j]))
    return tf.reduce_mean(fm_loss)


def discriminator_loss_fn(real_pred, fake_pred):
    real_loss = [mse(tf.ones_like(r[-1]), r[-1]) for r in real_pred]
    fake_loss = [mse(tf.zeros_like(f[-1]), f[-1]) for f in fake_pred]
    return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)


class MelGAN(keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
        generator_loss,
        feature_matching_loss,
        discriminator_loss,
    ):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.generator_loss = generator_loss
        self.feature_matching_loss = feature_matching_loss
        self.discriminator_loss = discriminator_loss

    @property
    def metrics(self):
        # Required to make Keras reset metrics at each epoch
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    @tf.function
    def train_step(self, batch):
        x_batch_train, y_batch_train = batch

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_audio_wave = self.generator(x_batch_train, training=True)

            real_pred = self.discriminator(y_batch_train, training=True)
            fake_pred = self.discriminator(gen_audio_wave, training=True)

            gen_loss = self.generator_loss(real_pred, fake_pred)
            fm_loss = self.feature_matching_loss(real_pred, fake_pred)
            gen_fm_loss = gen_loss + 10.0 * fm_loss

            disc_loss = self.discriminator_loss(real_pred, fake_pred)

        grads_gen = gen_tape.gradient(
            gen_fm_loss, self.generator.trainable_variables)
        grads_disc = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # Apply gradients using instance optimizers
        self.gen_optimizer.apply_gradients(
            zip(grads_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(
            zip(grads_disc, self.discriminator.trainable_variables))

        self.gen_loss_tracker.update_state(gen_fm_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {"gen_loss": self.gen_loss_tracker.result(), "disc_loss": self.disc_loss_tracker.result()}


def make_datasets(wav_glob="LJSpeech-1.1/wavs/*.wav"):
    wavs = load_wav_paths(wav_glob)
    ds = tf.data.Dataset.from_tensor_slices((wavs,))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def train_main(args):
    ds = make_datasets(args.wav_glob)
    ds = ds.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    generator = create_generator((None, 1))
    discriminator = create_discriminator((None, 1))

    mel_gan = MelGAN(generator, discriminator)
    gen_optimizer = keras.optimizers.Adam(
        LEARNING_RATE_GEN, beta_1=0.5, beta_2=0.9, clipnorm=1)
    disc_optimizer = keras.optimizers.Adam(
        LEARNING_RATE_DISC, beta_1=0.5, beta_2=0.9, clipnorm=1)

    mel_gan.compile(
        gen_optimizer,
        disc_optimizer,
        generator_loss_fn,
        feature_matching_loss_fn,
        discriminator_loss_fn,
    )

    # Callbacks: checkpoint and tensorboard
    ckpt_dir = args.checkpoint_dir or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "melgan_ckpt_{epoch:02d}.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="gen_loss", save_best_only=False, save_weights_only=True, verbose=1
    )
    tb_cb = keras.callbacks.TensorBoard(
        log_dir=args.log_dir or "logs", update_freq="batch")

    mel_gan.fit(ds, epochs=args.epochs or EPOCHS,
                callbacks=[checkpoint_cb, tb_cb])


def infer_main(args):
    # Simple inference test: create a generator that expects mel input (so bypass MelSpec)
    # The create_generator contains MelSpec, which returns the input unchanged if training=False.
    generator = create_generator((None, 1))
    # load weights if provided
    if args.weights:
        generator.load_weights(args.weights)
    # example: use a random mel-spectrogram batch for speed test
    audio_sample = tf.random.uniform([128, 50, 80])
    # call generator with training=False so MelSpec is bypassed (we pass mel directly)
    pred = generator.predict(audio_sample, batch_size=32, verbose=1)
    print("Predicted waveform batch shape:", pred.shape)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--wav_glob", default="LJSpeech-1.1/wavs/*.wav")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--weights", default=None)
    p.add_argument("--epochs", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_main(args)
    else:
        infer_main(args)
