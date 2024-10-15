"""
Title: Audio Classification with the STFTSpectrogram layer
Author: [Mostafa M. Amin](https://mostafa-amin.com)
Date created: 2024/10/04
Last modified: 2024/10/04
Description: Introducing the `STFTSpectrogram` layer to extract spectrograms for audio classification.
Accelerator: GPU
"""

"""
## Introduction

Preprocessing audio as spectrograms is an essential step in the vast majority
of audio-based applications. Spectrograms represent the frequency content of a
signal over time, are widely used for this purpose. In this tutorial, we'll
demonstrate how to use the `STFTSpectrogram` layer in Keras to convert raw
audio waveforms into spectrograms **within the model**. We'll then feed
these spectrograms into an LSTM network followed by Dense layers to perform
audio classification on the Speech Commands dataset.

We will:

- Load the ESC-10 dataset.
- Preprocess the raw audio waveforms and generate spectrograms using
   `STFTSpectrogram`.
- Build two models, one using spectrograms as 1D signals and the other is using
   as images (2D signals) with a pretrained image model.
- Train and evaluate the models.

## Setup

### Importing the necessary libraries
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile
from keras import layers
from scipy.signal import resample

keras.utils.set_random_seed(41)

"""
### Define some variables
"""

BASE_DATA_DIR = "./datasets/esc-50_extracted/ESC-50-master/"
BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 200
SAMPLE_RATE = 16000

"""
## Download and Preprocess the ESC-10 Dataset

We'll use the Dataset for Environmental Sound Classification dataset (ESC-10).
This dataset consists of five-second .wav files of environmental sounds.

### Download and Extract the dataset
"""

keras.utils.get_file(
    "esc-50.zip",
    "https://github.com/karoldvl/ESC-50/archive/master.zip",
    cache_dir="./",
    cache_subdir="datasets",
    extract=True,
)

"""
### Read the CSV file
"""

pd_data = pd.read_csv(os.path.join(BASE_DATA_DIR, "meta", "esc50.csv"))
# filter ESC-50 to ESC-10 and reassign the targets
pd_data = pd_data[pd_data["esc10"]]
targets = sorted(pd_data["target"].unique().tolist())
assert len(targets) == NUM_CLASSES
old_target_to_new_target = {old: new for new, old in enumerate(targets)}
pd_data["target"] = pd_data["target"].map(lambda t: old_target_to_new_target[t])
pd_data

"""
### Define functions to read and preprocess the WAV files
"""


def read_wav_file(path, target_sr=SAMPLE_RATE):
    sr, wav = scipy.io.wavfile.read(os.path.join(BASE_DATA_DIR, "audio", path))
    wav = wav.astype(np.float32) / 32768.0  # normalize to [-1, 1]
    num_samples = int(len(wav) * target_sr / sr)  # resample to 16 kHz
    wav = resample(wav, num_samples)
    return wav[:, None]  # Add a channel dimension (of size 1)


"""
Create a function that uses the `STFTSpectrogram` to compute a spectrogram,
then plots it.
"""


def plot_single_spectrogram(sample_wav_data):
    spectrogram = layers.STFTSpectrogram(
        mode="log",
        frame_length=SAMPLE_RATE * 20 // 1000,
        frame_step=SAMPLE_RATE * 5 // 1000,
        fft_length=1024,
        trainable=False,
    )(sample_wav_data[None, ...])[0, ...]

    # Plot the spectrogram
    plt.imshow(spectrogram.T, origin="lower")
    plt.title("Single Channel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


"""
Create a function that uses the `STFTSpectrogram` to compute three
spectrograms with multiple bandwidths, then aligns them as an image
with different channels, to get a multi-bandwith spectrogram,
then plots the spectrogram.
"""


def plot_multi_bandwidth_spectrogram(sample_wav_data):
    # All spectrograms must use the same `fft_length`, `frame_step`, and
    # `padding="same"` in order to produce spectrograms with identical shapes,
    # hence aligning them together. `expand_dims` ensures that the shapes are
    # compatible with image models.

    spectrograms = np.concatenate(
        [
            layers.STFTSpectrogram(
                mode="log",
                frame_length=SAMPLE_RATE * x // 1000,
                frame_step=SAMPLE_RATE * 5 // 1000,
                fft_length=1024,
                padding="same",
                expand_dims=True,
            )(sample_wav_data[None, ...])[0, ...]
            for x in [5, 10, 20]
        ],
        axis=-1,
    ).transpose([1, 0, 2])

    # normalize each color channel for better viewing
    mn = spectrograms.min(axis=(0, 1), keepdims=True)
    mx = spectrograms.max(axis=(0, 1), keepdims=True)
    spectrograms = (spectrograms - mn) / (mx - mn)

    plt.imshow(spectrograms, origin="lower")
    plt.title("Multi-bandwidth Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


"""
Demonstrate a sample wav file.
"""

sample_wav_data = read_wav_file(pd_data["filename"].tolist()[52])
plt.plot(sample_wav_data[:, 0])
plt.show()

"""
Plot a Spectrogram
"""

plot_single_spectrogram(sample_wav_data)

"""
Plot a multi-bandwidth spectrogram
"""

plot_multi_bandwidth_spectrogram(sample_wav_data)

"""
### Define functions to construct a TF Dataset
"""


def read_dataset(df, folds):
    msk = df["fold"].isin(folds)
    filenames = df["filename"][msk]
    targets = df["target"][msk].values
    waves = np.array([read_wav_file(fil) for fil in filenames], dtype=np.float32)
    return waves, targets


"""
### Create the datasets
"""

train_x, train_y = read_dataset(pd_data, [1, 2, 3])
valid_x, valid_y = read_dataset(pd_data, [4])
test_x, test_y = read_dataset(pd_data, [5])

"""
## Training the Models

In this tutorial we demonstrate the different usecases of the `STFTSpectrogram`
layer.

The first model will use a non-trainable `STFTSpectrogram` layer, so it is
intended purely for preprocessing. Additionally, the model will use 1D signals,
hence it make use of Conv1D layers.

The second model will use a trainable `STFTSpectrogram` layer with the
`expand_dims` option, which expands the shapes to be compatible with image
models.

### Create the 1D model

1. Create a non-trainable spectrograms, extracting a 1D time signal.
2. Apply `Conv1D` layers with `LayerNormalization` simialar to the
   classic VGG design.
4. Apply global maximum pooling to have fixed set of features.
5. Add `Dense` layers to make the final predictions based on the features.
"""

model1d = keras.Sequential(
    [
        layers.InputLayer((None, 1)),
        layers.STFTSpectrogram(
            mode="log",
            frame_length=SAMPLE_RATE * 40 // 1000,
            frame_step=SAMPLE_RATE * 15 // 1000,
            trainable=False,
        ),
        layers.Conv1D(64, 64, activation="relu"),
        layers.Conv1D(128, 16, activation="relu"),
        layers.LayerNormalization(),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 8, activation="relu"),
        layers.Conv1D(256, 8, activation="relu"),
        layers.Conv1D(512, 4, activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ],
    name="model_1d_non_trainble_stft",
)
model1d.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model1d.summary()

"""
Train the model and restore the best weights.
"""

history_model1d = model1d.fit(
    train_x,
    train_y,
    batch_size=BATCH_SIZE,
    validation_data=(valid_x, valid_y),
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EPOCHS,
            restore_best_weights=True,
        )
    ],
)

"""
### Create the 2D model

1. Create three spectrograms with multiple band-widths from the raw input.
2. Concatenate the three spectrograms to have three channels.
3. Load `MobileNet` and set the weights from the weights trained on `ImageNet`.
4. Apply global maximum pooling to have fixed set of features.
5. Add `Dense` layers to make the final predictions based on the features.
"""

input = layers.Input((None, 1))
spectrograms = [
    layers.STFTSpectrogram(
        mode="log",
        frame_length=SAMPLE_RATE * frame_size // 1000,
        frame_step=SAMPLE_RATE * 15 // 1000,
        fft_length=2048,
        padding="same",
        expand_dims=True,
        # trainable=True,  # trainable by default
    )(input)
    for frame_size in [30, 40, 50]  # frame size in milliseconds
]

multi_spectrograms = layers.Concatenate(axis=-1)(spectrograms)

img_model = keras.applications.MobileNet(include_top=False, pooling="max")
output = img_model(multi_spectrograms)

output = layers.Dropout(0.5)(output)
output = layers.Dense(256, activation="relu")(output)
output = layers.Dense(256, activation="relu")(output)
output = layers.Dense(NUM_CLASSES, activation="softmax")(output)
model2d = keras.Model(input, output, name="model_2d_trainble_stft")

model2d.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model2d.summary()

"""
Train the model and restore the best weights.
"""

history_model2d = model2d.fit(
    train_x,
    train_y,
    batch_size=BATCH_SIZE,
    validation_data=(valid_x, valid_y),
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EPOCHS,
            restore_best_weights=True,
        )
    ],
)

"""
### Plot Training History
"""

epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    epochs_range,
    history_model1d.history["accuracy"],
    label="Training Accuracy,1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model1d.history["val_accuracy"],
    label="Validation Accuracy, 1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["accuracy"],
    label="Training Accuracy, 2D model with trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["val_accuracy"],
    label="Validation Accuracy, 2D model with trainable STFT",
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(
    epochs_range,
    history_model1d.history["loss"],
    label="Training Loss,1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model1d.history["val_loss"],
    label="Validation Loss, 1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["loss"],
    label="Training Loss, 2D model with trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["val_loss"],
    label="Validation Loss, 2D model with trainable STFT",
)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

"""
### Evaluate on Test Data

Running the models on the test set.
"""

_, test_acc = model1d.evaluate(test_x, test_y)
print(f"1D model wit non-trainable STFT -> Test Accuracy: {test_acc * 100:.2f}%")

_, test_acc = model2d.evaluate(test_x, test_y)
print(f"2D model with trainable STFT -> Test Accuracy: {test_acc * 100:.2f}%")
