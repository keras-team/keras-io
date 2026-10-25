"""
Title: Text-to-Speech from Scratch
Author: [Abdulraheem Kasim](https://www.linkedin.com/in/kasim-abdulrahim/)
Date created: 2026/10/25
Last modified: 2026/10/25
Description: Training a Tacotron model for text-to-speech.
Accelerator: GPU
"""

"""
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

To switch backend, change `KERAS_BACKEND` below and **restart the kernel** —
the backend is fixed at the first `import keras` (and importing `tensorflow`
or `jax` first can silently pin the backend to TensorFlow).
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # any of "jax", "torch", "tensorflow"

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from keras import layers, ops

# Hyperparameters
N_FFT, SR, HOP, WIN, N_MELS = 1024, 22050, 256, 1024, 80
R, MAX_MEL_LEN, NB_CHARS_MAX, K1 = 5, 200, 200, 16
F_MIN, F_MAX, BATCH_SIZE, EPOCHS, TRAIN_RATIO = 0.0, 8000.0, 32, 1, 0.15

"""
## Hyperparameters
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

"""
## Data Preprocessing
### Character Vocabulary and Mel Filterbank
We create a character vocabulary and define a mel filterbank matrix for mel spectrogram computation.
"""

# Character vocabulary and log-mel filterbank
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz'?!;-, ")
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARACTERS)}
ID_TO_CHAR = [""] + CHARACTERS
VOCAB_SIZE = 1 + len(CHARACTERS)

MEL_FB = librosa.filters.mel(  # [N_FFT // 2 + 1, N_MELS]
    sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, htk=False, norm="slaney"
).T.astype(np.float32)

"""
## Audio and Text Processing Functions
These functions transform raw audio and text into inputs and targets.

1. **Audio to Mel Spectrogram**: Converts raw audio to log-mel spectrograms using Short-Time Fourier Transform(STFT).

2. **Padding and Reshaping**: Pads mel frames to multiples of R and fixes total length to MAX_MEL_LEN.

3. **Stop Token Generation**: creates stop targets indicating where padding begins.

4. **Decoder Input Preparation**: Creates right-shifted mel spectrograms for teacher forcing during training.

5. **Text Encoding**: Converts text to character IDs with padding to maximum text length.

This preprocessing ensures uniform input sizes for batch training and proper target formatting for the autoregressive decoder.
"""


def compute_mel(audio):
    """Waveform [samples] -> log-mel spectrogram [T, N_MELS]."""
    S = np.abs(
        librosa.stft(  # [N_FFT // 2 + 1, T]
            audio.astype(np.float32),
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    )
    return np.log(np.maximum(S.T @ MEL_FB, 1e-5)).astype("float32")


def encode_single_sample(wav_file, label):
    """(wav filename, transcript) -> ((text ids, decoder in), (mel, stop))."""
    audio, _ = librosa.load(WAVS_PATH + wav_file + ".wav", sr=SR)
    mel = compute_mel(audio)
    # Pad to a multiple of R, then fix length at MAX_MEL_LEN decoder steps.
    t = mel.shape[0]
    mel = np.pad(mel, ((0, (R - t % R) % R), (0, 0)))
    real_len = mel.shape[0] // R  # true decoder length, in steps
    max_frames = MAX_MEL_LEN * R
    mel = np.pad(mel, ((0, max(0, max_frames - mel.shape[0])), (0, 0)))
    mel = mel[:max_frames].reshape(MAX_MEL_LEN, N_MELS * R)
    # Stop target (1 = padding); GO frame + right-shifted decoder input.
    stop = (np.arange(MAX_MEL_LEN) >= min(real_len, MAX_MEL_LEN)).astype("float32")
    dec = np.pad(mel[:-1], ((1, 0), (0, 0)))
    ids = np.asarray([CHAR_TO_ID.get(c, 0) for c in label.lower()], dtype="int32")[
        :NB_CHARS_MAX
    ]
    ids = np.pad(ids, (0, NB_CHARS_MAX - ids.shape[0]))
    return (ids, dec.astype("float32")), (mel.astype("float32"), stop[:, None])


"""
## Creating `Dataset` objects

We preprocess each split into NumPy arrays. These arrays feed `model.fit`
"""


def make_arrays(df):
    """Preprocess a split into ([ids, dec_in], [mel, stop]) NumPy arrays."""
    enc, dec, mel, stop = [], [], [], []
    for i, (f, txt) in enumerate(zip(df["file_name"], df["normalized_transcription"])):
        (ids, d), (m, s) = encode_single_sample(f, txt)
        enc.append(ids)
        dec.append(d)
        mel.append(m)
        stop.append(s)
        if (i + 1) % 500 == 0:
            print(f"  preprocessed {i + 1}/{len(df)}")
    return [np.stack(enc), np.stack(dec)], [np.stack(mel), np.stack(stop)]


train_x, train_y = make_arrays(train_df)
val_x, val_y = make_arrays(val_df)

"""
## Visualize the data
Let's visualize a sample from the dataset to understand the data better.
"""


def visualize_sample(x, y, df):
    spec_raw = y[0][0]  # first sample's mel target [MAX_MEL_LEN, N_MELS * R]
    nz = np.where(np.any(spec_raw != 0, axis=1))[0]
    spec = np.transpose(spec_raw[: nz[-1] + 1] if nz.size else spec_raw)
    label = "".join(ID_TO_CHAR[i] for i in x[0][0] if i != 0)

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spec, vmax=1)
    ax.set_title(label)
    ax.axis("off")

    audio, _ = librosa.load(WAVS_PATH + df["file_name"].iloc[0] + ".wav", sr=SR)
    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display(Audio(audio, rate=SR))
    plt.show()


visualize_sample(train_x, train_y, train_df)

"""
The visualization shows both the mel spectrogram and waveform for a sample, along with it's text transcription. The spectrogram displays time on the x-axis and frequency bands on the y-axis, with intensity representing audio energy.

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


"""
**Attention Mechanism**: The location-sensitive attention works as follows:
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
        prenet_flat = self.prenet(
            ops.reshape(dec_in, (-1, self.n_mels * self.r)), training=training
        )  # [B*T_dec, 128]
        prenet_dec = ops.reshape(prenet_flat, (-1, self.T_dec, 128))  # [B, T_dec, 128]
        mask_f32 = ops.cast(enc_mask, "float32")
        enc_proj = ops.matmul(enc_out, self.W_enc)

        def step(carry, x):
            # Teacher forcing: precomputed prenet features (x);
            # AR: prenet of own previous prediction (dropout off).
            prev_frame, cum_attn, attn_h, h1, h2, ctx = carry[:6]
            h = ops.where(teacher_forcing, x, self.prenet(prev_frame, training=False))
            attn_out, [attn_h] = self.attn_rnn(
                ops.concatenate([h, ctx], -1), [attn_h], training=training
            )
            loc = ops.matmul(self.loc_conv(cum_attn), self.W_loc)  # location feature
            e = ops.tanh(enc_proj + ops.matmul(attn_out, self.W_dec)[:, None] + loc)
            score = ops.cast(ops.squeeze(ops.matmul(e, self.v), axis=-1), "float32")
            score = score + (1.0 - mask_f32) * -1e9  # mask out padded characters
            attn = ops.softmax(score, axis=-1)
            attn_c = ops.cast(attn, dtype)
            cum_attn = cum_attn + ops.expand_dims(attn_c, -1)
            ctx = ops.sum(ops.expand_dims(attn_c, -1) * enc_out, axis=1)
            out1, [h1] = self.dec_rnn_1(
                ops.concatenate([attn_out, ctx], -1), [h1], training=training
            )
            out2, [h2] = self.dec_rnn_2(out1, [h2], training=training)
            mel_out = self.mel_proj(out2)
            stop_t = self.stop_proj(out2)
            state = (mel_out, cum_attn, attn_h, h1, h2, ctx, stop_t, attn)
            return state, state

        init = (
            ops.zeros((B, self.n_mels * self.r), dtype),
            ops.zeros((B, T_enc, 1), dtype),
            ops.zeros((B, 256), dtype),
            ops.zeros((B, 256), dtype),
            ops.zeros((B, 256), dtype),
            ops.zeros((B, enc_out.shape[-1]), dtype),
            ops.zeros((B, 1), dtype),
            ops.zeros((B, T_enc), "float32"),
        )
        _, (mel_t, _, _, _, _, _, stop_t, attn_t) = ops.scan(
            step, init, xs=ops.transpose(prenet_dec, [1, 0, 2])
        )
        to_bmt = lambda x: ops.transpose(x, [1, 0, 2])  # [T, B, ...] -> [B, T, ...]
        return to_bmt(mel_t), to_bmt(stop_t), to_bmt(attn_t)


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


"""
## Training the Model
Now we compile and train the model with custom loss functions and learning rate schedule.
"""

model, attn_model = get_tacotron_model(
    N_MELS, R, K1, MAX_MEL_LEN, vocabulary=ID_TO_CHAR
)
model.compile(
    optimizer=keras.optimizers.Adam(NoamSchedule(d_model=256)),
    loss={"mel_out": masked_mae, "stop_out": stop_token_bce},
    loss_weights={"mel_out": 1.0, "stop_out": 0.5},
)
model.summary()

"""
## Training Callback for Visualization
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
        n_chars = int(np.count_nonzero(x[0][0]))
        stop = y[1][0][:, 0]
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


"""
During training, we monitor several losses:

- **Mel Spectrogram Loss**: Masked MAE that only considers non-padded frames.

- **Stop Token Loss**: Binary cross-entropy with 2x weight, teaching the decoder to reliably predict when the generation should halt.

The attention visualizations show how the model learns to align characters with mel spectrogram frames. During training ,The plot goes from blurry to a sharp diagonal.
"""

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="tacotron_best.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=1,
)

cb_batch = ([a[:BATCH_SIZE] for a in train_x], [b[:BATCH_SIZE] for b in train_y])
history = model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[AttentionCallback(attn_model, cb_batch), checkpoint_cb],
)

"""
## Inference: Text-to-Speech Synthesis
Finally, we demonstrate how to synthesize speech from text using the trained model.

To synthesize speech from text, we run the trained model to predict a mel spectrogram, then invert the preprocessing on the spectrogram to recover a waveform. We use the Griffin-Lim algorithm: the predicted spectrogram is mapped back to linear-frequency magnitudes, and Griffin-Lim iteratively reconstructs the missing phase.
"""

MEL_FB_INVERSE = np.linalg.pinv(MEL_FB)  # [N_MELS, N_FFT//2+1]


def mel_to_linear_mag(mel_spectrogram):
    if len(mel_spectrogram.shape) == 2:
        mel_spectrogram = mel_spectrogram[None, ...]
    mel_np = keras.ops.convert_to_numpy(mel_spectrogram)
    linear_mag = np.matmul(mel_np, MEL_FB_INVERSE)
    linear_mag = np.maximum(linear_mag, 0.0)
    return linear_mag


def synthesize(text, model):
    """Text -> predicted log-mel spectrogram [T_dec, N_MELS * R].

    Feeds an all-zero decoder input, which switches TacotronDecoder into
    autoregressive mode (the model feeds back its own predictions), then
    crops the padding after the predicted stop point.
    """
    ids = np.asarray([CHAR_TO_ID.get(c, 0) for c in text.lower()], dtype="int32")[
        :NB_CHARS_MAX
    ]
    ids = np.pad(ids, (0, NB_CHARS_MAX - ids.shape[0]))[None, ...]
    dec_in = np.zeros((1, MAX_MEL_LEN, N_MELS * R), dtype="float32")
    mel_hat, stop_hat = model.predict([ids, dec_in], verbose=0)
    stop = stop_hat[0, :, 0]
    end = int(np.argmax(stop > 0.5)) if (stop > 0.5).any() else MAX_MEL_LEN
    return mel_hat[0, : max(end, 1)]


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


# Test with both methods
text = "Hello, This is a text-to-speech example demonstrating speech sythesis through end to end neural architecture"
print(f'Synthesizing audio for: "{text}"\n')

mel_hat = synthesize(text, model)

audio_librosa = mel_hat_to_audio_librosa(mel_hat, n_iter=32)
print("\nLibrosa Griffin-Lim output:")
display(Audio(audio_librosa, rate=SR))
# The sound sounds robotic but how much of the "robotic" sound is the vocoder and not the model? Let's preprocess a ground-truth audio into a mel-spectrogram and use the Griffin-Lim algorithm to vocode it

# Ground-truth sanity check, backend-agnostic (no tf.* calls): vocode a
# ground-truth mel with Griffin-Lim to isolate vocoder error from model error.
audio, _ = librosa.load(WAVS_PATH + train_df["file_name"].iloc[0] + ".wav", sr=SR)
gt_mel = compute_mel(audio)  # [T, N_MELS]
gt_linear = np.maximum(np.exp(gt_mel) @ MEL_FB_INVERSE, 0.0)  # [T, N_FFT//2+1]
display(
    Audio(
        librosa.griffinlim(
            gt_linear.T, n_iter=32, hop_length=HOP, win_length=WIN, window="hann"
        ),
        rate=SR,
    )
)

"""
## Conclusion
This example demonstrated how to build a complete Tacotron-based text-to-speech system using Keras. This architecture represents the foundation of modern end-to-end TTS systems and can be extended with more advanced features like multi-speaker support, prosody modeling and vocoding with a neural vocoder.
"""
