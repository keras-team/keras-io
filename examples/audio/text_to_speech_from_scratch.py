# -*- coding: utf-8 -*-
"""
**Title:** Text-to-Speech from Scratch <br>
**Author:** [Abdulraheem Kasim](https://www.linkedin.com/in/kasim-abdulrahim/) <br>
**Date created:** 2026/10/14<br>
**Description:** Training a Tacotron model for text-to-speech. <br>
**Accelerator:** GPU

## Introduction

Text-to-speech, also known as speech synthesis, is the task of converting written text into natural-sounding speech. Modern neural TTS systems learn this mapping directly from paired text and audio using deep learning, eliminating the need for handcrafted linguistic and acoustic features.

This example demonstrates how to build an attention-based neural TTS system from scratch using **Keras**. The model is inspired by Tacotron, combining a CBHG encoder, an attention mechanism, and a recurrent decoder to predict mel spectrograms. During inference, the predicted mel spectrograms are converted into waveforms using Griffin-Lim, illustrating the workflow used by modern end-to-end speech synthesis systems.

The model is trained on the **LJSpeech** dataset, a publicly available corpus containing approximately 24 hours of English speech a single female speaker.

### References

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)

- [Tacotron: Towards End-to-End Speech Synthesis (Wang et al., 2017)](https://research.google/pubs/tacotron-towards-end-to-end-speech-synthesis/)

- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions((Shen et al., 2017))](https://arxiv.org/abs/1712.05884)

- [Griffin & Lim (1984), Signal estimation from modified short-time Fourier transform](https://ieeexplore.ieee.org/document/1164317)

## Setup
First, let's import the necessary libraries and configure hyperparameters for our model.
"""

# import os

import numpy as np
import pandas as pd

# import torch
import tensorflow as tf
import keras
import librosa
from keras import ops, layers
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# from speechbrain.inference.vocoders import HIFIGAN

# ── Hyperparameters ──
N_FFT, SR, HOP, WIN, N_MELS = 1024, 22050, 256, 1024, 80
R, MAX_MEL_LEN, NB_CHARS_MAX, K1 = 5, 200, 200, 16
F_MIN, F_MAX, BATCH_SIZE, EPOCHS, TRAIN_RATIO = 0.0, 8000.0, 32, 100, 0.15

"""## Hyperparameters
- **N_FFT**: Fast Fourier Transform window size for spectrogram computation.
- **SR**: Audio sample rate.
- **HOP**: Hop length between frames.
- **N_MELS**: Number of mel filterbanks.
- **R**: Reduction factor(frames per decoder step).
- **MAX_MEL_LEN**: Maximum mel spectrogram length.

## Load the LJSpeech Dataset

Let's download the [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/).
The dataset contains 13,100 audio files as `wav` files in the `/wavs/` folder.
The label (transcript) for each audio file is a string
given in the `metadata.csv` file. The fields are:

- **ID**: this is the name of the corresponding .wav file
- **Transcription**: words spoken by the reader (UTF-8)
- **Normalized transcription**: transcription with numbers,
ordinals, and monetary units expanded into full words (UTF-8).

For this demo we take 15% of the data and will use only the "Normalized transcription" field.

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22,050 Hz.
"""

data_path = keras.utils.get_file(
    "LJSpeech-1.1",
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    untar=True,
)
WAVS_PATH = data_path + "/LJSpeech-1.1/wavs/"

metadata_df = (
    pd.read_csv(
        data_path + "/LJSpeech-1.1/metadata.csv",
        sep="|",
        header=None,
        quoting=3,
        names=["file_name", "transcription", "normalized_transcription"],
    )[["file_name", "normalized_transcription"]]
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

train_df = metadata_df.iloc[: int(len(metadata_df) * TRAIN_RATIO)]

# 10% of training data for validation
val_df = metadata_df.iloc[len(train_df) : len(train_df) + int(len(train_df) * 0.1)]

"""## Data Preprocessing
### Character Vocabulary and Mel Filterbank
We create a character vocabulary and define a mel filterbank matrix for mel spectrogram computation.
"""

# ── Character vocabulary and log-mel filterbank ──
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz'?!;-, ")
char_to_num = keras.layers.StringLookup(vocabulary=CHARACTERS, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
MEL_FB_TF = tf.constant(
    librosa.filters.mel(  # [N_FFT // 2 + 1, N_MELS]
        sr=SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        htk=False,
        norm="slaney",
    ).T.astype(np.float32)
)

"""## Audio and Text Processing Functions
These functions transform raw audio and text into inputs and targets.

1. **Audio to Mel Spectrogram**: Converts raw audio to log-mel spectrograms using Short-Time Fourier Transform(STFT).

2. **Padding and Reshaping**: Pads mel frames to multiples of R and fixes total length to MAX_MEL_LEN.

3. **Stop Token Generation**: creates stop targets indicating where padding begins.

4. **Decoder Input Preparation**: Creates right-shifted mel spectrograms for teacher forcing during training.

5. **Text Encoding**: Converts text to character IDs with padding to maximum text length.

This preprocessing ensures uniform input sizes for batch training and proper target formatting for the autoregressive decoder.


"""


def compute_mel_tf(audio):
    """Waveform [samples] -> log-mel spectrogram [T, N_MELS]."""
    audio = tf.pad(
        tf.cast(audio, tf.float32), [[N_FFT // 2, N_FFT // 2]], mode="REFLECT"
    )
    stft = tf.signal.stft(audio, WIN, HOP, N_FFT, tf.signal.hann_window)
    return tf.math.log(tf.maximum(tf.matmul(tf.abs(stft), MEL_FB_TF), 1e-5))


def encode_single_sample(wav_file, label):
    """(wav filename, transcript) -> ((text ids, decoder in), (mel, stop))."""
    audio, _ = tf.audio.decode_wav(tf.io.read_file(WAVS_PATH + wav_file + ".wav"))
    mel = ops.cast(compute_mel_tf(ops.reshape(audio, [-1])), "float32")
    mel.set_shape([None, N_MELS])
    # Pad to a multiple of R, then fix length at MAX_MEL_LEN decoder steps.
    t = ops.shape(mel)[0]
    mel = ops.pad(mel, [[0, (R - t % R) % R], [0, 0]])
    real_len = ops.shape(mel)[0] // R  # true decoder length, in steps
    max_frames = MAX_MEL_LEN * R
    mel = ops.pad(mel, [[0, ops.maximum(0, max_frames - ops.shape(mel)[0])], [0, 0]])
    mel = ops.reshape(mel[:max_frames], (MAX_MEL_LEN, N_MELS * R))
    # Stop target (1 = padding); GO frame + right-shifted decoder input.
    stop = ops.cast(
        ops.arange(MAX_MEL_LEN) >= ops.minimum(real_len, MAX_MEL_LEN), "float32"
    )
    dec = ops.pad(mel[:-1], [[1, 0], [0, 0]])
    label = char_to_num(tf.strings.unicode_split(tf.strings.lower(label), "UTF-8"))
    label = ops.pad(label[:NB_CHARS_MAX], [[0, NB_CHARS_MAX - ops.shape(label)[0]]])
    return (label, dec), (mel, stop[:, None])


"""## Creating `Dataset` objects

We create a `tf.data.Dataset` object that yields
the transformed elements, in the same order as they
appeared in the input.
"""


def make_dataset(df):
    return (
        tf.data.Dataset.from_tensor_slices(
            (df["file_name"].tolist(), df["normalized_transcription"].tolist())
        )
        .map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(4096)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)

"""## Visualize the data
Let's visualize a sample from the dataset to understand the data better.
"""


def visualize_sample(dataset, df):
    batch = next(iter(dataset))
    spec_raw = batch[1][0][0].numpy()
    nz = np.where(np.any(spec_raw != 0, axis=1))[0]
    spec = np.transpose(spec_raw[: nz[-1] + 1] if nz.size else spec_raw)
    label = tf.strings.reduce_join(num_to_char(batch[0][0][0])).numpy().decode("utf-8")

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spec, vmax=1)
    ax.set_title(label)
    ax.axis("off")

    audio, _ = tf.audio.decode_wav(
        tf.io.read_file(WAVS_PATH + df["file_name"].iloc[0] + ".wav")
    )
    audio = audio.numpy()
    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display(Audio(np.transpose(audio), rate=SR))
    plt.show()


visualize_sample(train_dataset, train_df)

"""The visualization shows both the mel spectrogram and waveform for a sample, along with it's text transcription. The spectrogram displays time on the x-axis and frequency bands on the y-axis, with intensity representing audio energy.

## Model Architecture

### CBHG Encoder
The CBHG (Convolutional Bank + Highway Network + Bidirectional GRU); encoder processes character embeddings into rich representations suitable for the attention mechanism.

- **Convolutional Bank**: Applies multiple conv1D filters with different kernel sizes to capture local context.

- **Highway Network**: Allows information flow across layers.

- **Bidirectional GRU**: Captures long range dependencies in both directions.
"""


def pre_net(x):
    for d in (256, 128):
        x = layers.Dropout(0.5)(layers.Dense(d, activation="relu")(x))
    return x


def conv1d_bank(K, x):
    """Conv1D bank, kernel sizes 1..K, each 128 filters, BN + ReLU."""
    bank = [
        layers.BatchNormalization()(
            layers.Activation("relu")(
                layers.Conv1D(128, k, padding="same", use_bias=False)(x)
            )
        )
        for k in range(1, K + 1)
    ]
    return ops.concatenate(bank, axis=-1)


def highway_output(x, nb_layers, activation="tanh", bias=-3):
    dim = x.shape[-1]
    for _ in range(nb_layers):
        gate = layers.Dense(
            dim,
            activation="sigmoid",
            bias_initializer=keras.initializers.Constant(bias),
        )(x)
        x = layers.Dense(dim, activation=activation)(x) * gate + x * (1.0 - gate)
    return x


def cbhg_encoder(inp, K):
    x = conv1d_bank(K, inp)
    x = layers.MaxPooling1D(2, 1, padding="same")(x)
    x = layers.BatchNormalization()(
        layers.Conv1D(128, 3, padding="same", use_bias=False, activation="relu")(x)
    )
    x = layers.BatchNormalization()(
        layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    )
    x = x + inp
    x = highway_output(x, 4, activation="relu")
    return layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)


"""**Attention Mechanism**: The location-sensitive attention works as follows:
1. **Location features**: Uses convolutional filters on the cumulative attention to focus on relevant encoder positions.

2. **Energy calculation**: Computes attention energies using encoder outputs, decoder state, and location features.

3. **Softmax Normalization**: Converts energies to attention weights.

4. **Context Vector**: Computes weighted sum of encoder outputs using attention weights.

The decoder uses to GRU cells to generate mel spectrogram frames and stop tokens autoregressively. During training, it uses teacher-forcing (feeding ground truth), while during inference, it feeds back its own predictions.
"""


class TacotronDecoder(keras.layers.Layer):
    """Autoregressive decoder with location-sensitive attention: teacher
    forcing feeds dec_in[:, t] at step t, while an all-zero dec_in switches to
    autoregressive mode (the model feeds back its own previous prediction)."""

    def __init__(
        self,
        T_dec,
        n_mels,
        r,
        attn_dim=128,
        loc_filters=32,
        loc_kernel_size=31,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.T_dec, self.n_mels, self.r = T_dec, n_mels, r
        self.attn_dim, self.loc_filters = attn_dim, loc_filters
        self.loc_kernel_size = loc_kernel_size

    def build(self, input_shape):
        enc_dim = input_shape[0][-1]
        self.prenet = keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
            ]
        )
        self.prenet.build((None, self.n_mels * self.r))
        self.attn_rnn = layers.GRUCell(256)
        self.dec_rnn_1 = layers.GRUCell(256)
        self.dec_rnn_2 = layers.GRUCell(256)
        self.W_enc = self.add_weight((enc_dim, self.attn_dim), name="W_enc")
        self.W_dec = self.add_weight((256, self.attn_dim), name="W_dec")
        self.W_loc = self.add_weight((self.loc_filters, self.attn_dim), name="W_loc")
        self.v = self.add_weight((self.attn_dim, 1), name="v")
        self.loc_conv = layers.Conv1D(
            self.loc_filters, self.loc_kernel_size, padding="same", use_bias=False
        )
        self.mel_proj = layers.Dense(self.n_mels * self.r, name="mel_out")
        self.stop_proj = layers.Dense(1, activation="sigmoid", name="stop_out")

    def call(self, inputs, enc_mask=None, teacher_forcing=None, training=None):
        enc_out, dec_in = inputs
        dtype = enc_out.dtype
        B, T_enc = ops.shape(enc_out)[0], ops.shape(enc_out)[1]
        enc_mask = ops.ones((B, T_enc), dtype=dtype) if enc_mask is None else enc_mask
        if teacher_forcing is None:  # all-zero dec_in -> autoregressive mode
            teacher_forcing = ops.any(ops.not_equal(dec_in, 0.0))
        mask_f32 = ops.cast(enc_mask, "float32")
        enc_proj = ops.matmul(enc_out, self.W_enc)
        zeros = lambda *shape: ops.zeros(shape, dtype=dtype)
        mel_ta = tf.TensorArray("float32", size=self.T_dec)
        stop_ta = tf.TensorArray("float32", size=self.T_dec)
        attn_ta = tf.TensorArray("float32", size=self.T_dec)

        def body(
            t, prev_frame, cum_attn, attn_h, h1, h2, ctx, mel_ta, stop_ta, attn_ta
        ):
            # Teacher forcing: ground-truth frame; AR: own previous prediction.
            frame = ops.where(teacher_forcing, dec_in[:, t, :], prev_frame)
            x = self.prenet(frame, training=training)
            attn_out, [attn_h] = self.attn_rnn(
                ops.concatenate([x, ctx], -1), [attn_h], training=training
            )
            loc = ops.matmul(self.loc_conv(cum_attn), self.W_loc)  # location feature
            e = tf.tanh(enc_proj + ops.matmul(attn_out, self.W_dec)[:, None] + loc)
            score = ops.cast(tf.squeeze(ops.matmul(e, self.v), -1), "float32")
            score += (1.0 - mask_f32) * -1e9  # mask out padded characters
            attn = tf.nn.softmax(score, axis=-1)
            attn_c = ops.cast(attn, dtype)
            cum_attn += tf.expand_dims(attn_c, -1)
            ctx = ops.sum(tf.expand_dims(attn_c, -1) * enc_out, axis=1)
            out1, [h1] = self.dec_rnn_1(
                ops.concatenate([attn_out, ctx], -1), [h1], training=training
            )
            out2, [h2] = self.dec_rnn_2(out1, [h2], training=training)
            mel_out = self.mel_proj(out2)
            mel_ta = mel_ta.write(t, ops.cast(mel_out, "float32"))
            stop_ta = stop_ta.write(t, ops.cast(self.stop_proj(out2), "float32"))
            attn_ta = attn_ta.write(t, attn)
            return (
                t + 1,
                mel_out,
                cum_attn,
                attn_h,
                h1,
                h2,
                ctx,
                mel_ta,
                stop_ta,
                attn_ta,
            )

        *_, mel_ta, stop_ta, attn_ta = tf.while_loop(
            cond=lambda t, *_: t < self.T_dec,
            body=body,
            loop_vars=(
                tf.constant(0),
                zeros(B, self.n_mels * self.r),
                zeros(B, T_enc, 1),
                zeros(B, 256),
                zeros(B, 256),
                zeros(B, 256),
                zeros(B, enc_out.shape[-1]),
                mel_ta,
                stop_ta,
                attn_ta,
            ),
            parallel_iterations=10,
            swap_memory=True,
            maximum_iterations=self.T_dec,
        )
        stack = lambda ta: ops.cast(tf.transpose(ta.stack(), [1, 0, 2]), dtype)
        return stack(mel_ta), stack(stop_ta), stack(attn_ta)


def get_tacotron_model(n_mels, r, k1, mel_time_length, vocabulary):
    """Builds (train_model, attn_model); attn_model also returns alignments."""
    enc_in = layers.Input(shape=(NB_CHARS_MAX,), name="enc_in")
    enc_mask = ops.cast(ops.not_equal(enc_in, 0), "float32")
    enc_out = cbhg_encoder(pre_net(layers.Embedding(len(vocabulary), 256)(enc_in)), k1)
    dec_in = layers.Input(shape=(mel_time_length, n_mels * r), name="dec_in")
    mel_hat, stop_hat, attn_w = TacotronDecoder(
        T_dec=mel_time_length, n_mels=n_mels, r=r
    )([enc_out, dec_in], enc_mask=enc_mask)
    # Output names match the loss dictionary passed to compile().
    mel_hat = layers.Activation("linear", name="mel_out")(mel_hat)
    stop_hat = layers.Activation("linear", name="stop_out")(stop_hat)
    train_model = keras.Model([enc_in, dec_in], [mel_hat, stop_hat])
    attn_model = keras.Model(
        [enc_in, dec_in],
        [mel_hat, stop_hat, layers.Activation("linear", name="attn_out")(attn_w)],
    )
    return train_model, attn_model


"""## Custom Learning Rate Schedules and Loss Functions"""


class NoamSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """LR = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)."""

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model, self.warmup_steps = float(d_model), float(warmup_steps)

    def __call__(self, step):
        step = ops.cast(step, "float32") + 1.0  # avoid div-by-zero at step 0
        return self.d_model**-0.5 * ops.minimum(
            step**-0.5, step * self.warmup_steps**-1.5
        )

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


def masked_mae(y_true, y_pred):
    """MAE over non-padded frames only."""
    mask = ops.cast(
        ops.any(ops.not_equal(y_true, 0.0), axis=-1, keepdims=True), "float32"
    )
    mask = ops.broadcast_to(mask, ops.shape(y_true))
    return ops.sum(ops.abs(y_true - y_pred) * mask) / (ops.sum(mask) + 1e-7)


def stop_token_bce(y_true, y_pred):
    """BCE with 2x weight on stop frames."""
    bce = keras.ops.binary_crossentropy(y_true, y_pred)
    return ops.mean(bce * ops.where(ops.equal(y_true, 1.0), 2.0, 1.0))


"""## Training the Model
Now we compile and train the model with custom loss functions and learning rate schedule.
"""

model, attn_model = get_tacotron_model(
    N_MELS, R, K1, MAX_MEL_LEN, vocabulary=char_to_num.get_vocabulary()
)
model.compile(
    optimizer=keras.optimizers.Adam(NoamSchedule(d_model=256)),
    loss={"mel_out": masked_mae, "stop_out": stop_token_bce},
    loss_weights={"mel_out": 1.0, "stop_out": 0.5},
)
model.summary()

"""## Training Callback for Visualization
We create a callback to visualize attention alignments during training.
"""


class AttentionCallback(keras.callbacks.Callback):
    """Plot alignment cropped to the ground-truth stop position."""

    def __init__(self, attn_model, batch, every_n=1):
        super().__init__()
        self.attn_model, self.batch, self.every_n = attn_model, batch, every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        x, y = self.batch
        A = self.attn_model.predict(x, verbose=0)[2][0]  # [T_dec, T_enc]
        n_chars = int(np.count_nonzero(x[0][0].numpy()))
        stop = y[1][0].numpy()[:, 0]
        dec_end = int(np.argmax(stop > 0.5)) if (stop > 0.5).any() else A.shape[0]
        plt.imshow(A[:dec_end, :n_chars].T, aspect="auto", origin="upper")
        plt.colorbar()
        plt.title(f"{n_chars} chars x {dec_end} steps")
        plt.xlabel("Decoder timestep")
        plt.ylabel("Encoder timestep")
        plt.show()
        print("sharpness (→1.0):", float(A[:dec_end].max(-1).mean()))
        print(
            "park position:",
            int(A[dec_end - 1].argmax()),
            f"/ last char = {n_chars - 1}",
        )


"""During training, we monitor several losses:

- **Mel Spectrogram Loss**: Masked MAE that only considers non-padded frames.

- **Stop Token Loss**: Binary cross-entropy with 2x weight, teaching the decoder to reliably predict when the generation should halt.

The attention visualizations show how the model learns to align characters with mel spectrogram frames. During training ,The plot goes from blurry to a sharp diagonal.

"""

model.load_weights("/kaggle/working/checkpoints/tacotron_v2_epoch_250.weights.h5")

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="tacotron_best.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=1,
)

batch = next(iter(train_dataset.take(1)))
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[AttentionCallback(attn_model, batch), checkpoint_cb],
)

"""## Inference: Text-to-Speech Synthesis
Finally, we demonstrate how to synthesize speech from text using the trained model.

To synthesize speech from text, we run the trained model to predict a mel spectrogram, then invert the preprocessing on the spectrogram to recover a waveform. We use the Griffin-Lim algorithm: the predicted spectrogram is mapped back to linear-frequency magnitudes, and Griffin-Lim iteratively reconstructs the missing phase.
"""

MEL_FB_INVERSE = np.linalg.pinv(MEL_FB_TF.numpy().T)  # [N_MELS, N_FFT//2+1]


def mel_to_linear_mag(mel_spectrogram):
    if len(mel_spectrogram.shape) == 2:
        mel_spectrogram = mel_spectrogram[None, ...]
    mel_np = keras.ops.convert_to_numpy(mel_spectrogram)
    linear_mag = np.matmul(mel_np, MEL_FB_INVERSE)
    linear_mag = np.maximum(linear_mag, 0.0)
    return linear_mag


def mel_hat_to_audio_librosa(mel_hat, n_iter=32):
    # 1. Ensure batch dimension and convert to numpy
    if len(mel_hat.shape) == 2:
        mel_hat = mel_hat[None, ...]
    mel = keras.ops.convert_to_numpy(mel_hat)
    B, T_dec, _ = mel.shape

    # 2. Ungroup R frames
    mel = mel.reshape(B, T_dec * R, N_MELS)

    # 3. Undo log operation
    mel = np.exp(mel)

    # 4. Convert mel to linear magnitude using pseudo-inverse
    linear_mag = np.matmul(mel, MEL_FB_INVERSE)
    linear_mag = np.maximum(linear_mag, 0.0)

    # 5. Apply Griffin-Lim
    audio_batch = []
    for i in range(B):
        # librosa expects [N_FFT//2+1, T] so we transpose
        audio = librosa.griffinlim(
            linear_mag[i].T,
            n_iter=n_iter,
            hop_length=HOP,
            win_length=WIN,
            window="hann",
        )
        audio_batch.append(audio)

    return audio_batch[0] if B == 1 else np.stack(audio_batch)


# ── Test with both methods ──
text = "At that station, the safe were given out, heavy with shot and gold, the thieves went on to dover and by-and-by"
print(f'Synthesizing audio for: "{text}"\n')

mel_hat = synthesize(text, model)

audio_librosa = mel_hat_to_audio_librosa(mel_hat, n_iter=32)
print("\nLibrosa Griffin-Lim output:")
display(Audio(audio_librosa, rate=SR))

"""The sound sounds robotic but how much of the "robotic" sound is the vocoder and not the model? Let's preprocess a ground-truth audio into a mel-spectrogram and use the Griffin-Lim algorithm to vocode it."""

audio, _ = tf.audio.decode_wav(
    tf.io.read_file(WAVS_PATH + train_df["file_name"].iloc[0] + ".wav")
)
# compute ground truth mel
gt_mel = compute_mel_tf(audio, [-1])
display(
    Audio(
        librosa.griffin_lim(
            tf.constant(np.maximum(np.exp(gt_mel.numpy()) @ MEL_FB_INV, 0.0))
        ),
        rate=SR,
    )
)

"""## Conclusion
This example demonstrated how to build a complete Tacotron-based text-to-speech system using Keras. This architecture represents the foundation of modern end-to-end TTS systems and can be extended with more advanced features like multi-speaker support, prosody modeling and vocoding with a neural vocoder.
"""
