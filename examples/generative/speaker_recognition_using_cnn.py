"""
Title: Speaker Recognition Using a Convolutional Network
Author: Fadi Badine
Date created: 14/06/2020
Last modified: 14/06/2020
Description: A CNN to classify speakers using Fast Fourier Transformation.
"""

"""
## Introduction

This example demostrates how to create a model to classify speakers based on the transformation
of their speech into the frequency domain using Fast Fourier Transformation.

It shows the following:
- How to use Keras Sequence in order to create a data generator that loads data from file while the model trains
- How to create a VGG16-like model using Conv1D and MaxPool1D

Note that this example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import librosa
from scipy.io import wavfile
from IPython.display import Audio

import tensorflow as tf
from tensorflow import keras

# Where the dataset is located
# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
DATASET_ROOT = "/home/fadi/Downloads/16000_pcm_speeches"

# The folders to which we will put both the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Folders containing the noise samples in the
# original dataset
NOISE_FOLDERS = ["other", "_background_noise_"]

# Supported audio file format
FILE_FORMATS = ".wav"

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
DESIRED_SR = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 100

"""
## Preparing Data
"""

"""
The dataset is composed of 7 folders that can be divided into 2 groups:
- One for the speeches containing 5 folders for 5 different speakers. Each folder contain
1500 audio files each of which is 1 second long and sampled at 16000 samples per second
- One for different noises containing 2 folders and an overall of 6 files. Those files
are longer than 1 second and not all of them sampled at 16000. We will use those 6 files
to create 354 1-second-long noise samples to be used for training
"""

# The dataset is composed of 7 folders that can be divided into 2 groups:
# - Speakers' audio folders:
#   5 folders containing each 1500 audio files each of which is 1 second long
#   and sampled at 16000 samples per second
# - Noise folders:
#   2 folders and an overall of 6 noise audio files.
#   These files are longer than 1 second and not all of them sampled at 16000.
#   We will use those 6 files to create 354 1-second-long noise samples

# Let's separate these 2 categories into 2 folders:
# - 'audio' which will contain all the speakers' audio folders and files
# - 'noise' which will contain all the noise sounds and folders

if os.path.exists(DATASET_AUDIO_PATH) is False:
    os.makedirs(DATASET_AUDIO_PATH)

if os.path.exists(DATASET_NOISE_PATH) is False:
    os.makedirs(DATASET_NOISE_PATH)

if (
    len(os.listdir(DATASET_AUDIO_PATH)) == 0
    and len(os.listdir(DATASET_NOISE_PATH)) == 0
):
    for folder in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
            if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                continue
            elif folder in NOISE_FOLDERS:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_NOISE_PATH, folder),
                )
            else:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_AUDIO_PATH, folder),
                )

# Let's generate the list of audio files and their corresponding label (speaker)
(
    file_paths,
    labels,
    class_names,
) = tf.python.keras.preprocessing.dataset_utils.index_directory(
    DATASET_AUDIO_PATH, "inferred", FILE_FORMATS, shuffle=True, seed=SHUFFLE_SEED
)

"""
## Generating and Loading Noise
"""

# Let's first get the list of all noise files
noise_paths, _, _ = tf.python.keras.preprocessing.dataset_utils.index_directory(
    DATASET_NOISE_PATH, "inferred", FILE_FORMATS, shuffle=False
)

# - Read each of those files
# - Resample it at the desired sampling rate
# - Split it into chuncks of 1 second long each

noise_stream = np.array([]).reshape(0, DESIRED_SR)

for noise_path in noise_paths:
    # Read and resample the noise audio file
    noise, _ = librosa.load(noise_path, sr=DESIRED_SR)

    # Split the audio into segments of 1 second long each
    nb_samples = len(noise) // DESIRED_SR
    arr = np.split(noise[: nb_samples * DESIRED_SR], nb_samples)

    # Add the sample to the Noises array
    noise_stream = np.concatenate((noise_stream, arr), axis=0)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noise_stream.shape[0], noise_stream.shape[1] // DESIRED_SR
    )
)

# Create a random array of the same size as file_paths and with values
# ranging from 0 to the number of noise stream samples that we have.
# For every audio sample in file_paths, there will be a noise sample
# identified by the value of noises at the same location as file_paths
# This array will be shuffled at the end of every epoch in order to train
# each sample with a wide range of noise samples.

noises = np.random.randint(0, noise_stream.shape[0], len(file_paths))

"""
## Train / Validation Split
"""

valid_size = int(len(file_paths) * VALID_SPLIT)

valid_file_paths, train_file_paths = file_paths[:valid_size], file_paths[valid_size:]
valid_labels, train_labels = labels[:valid_size], labels[valid_size:]

print("Training size: {}".format(len(train_file_paths)))
print("Validation size: {}".format(len(valid_file_paths)))

valid_noises, train_noises = noises[:valid_size], noises[valid_size:]

print("Training noise size: {}".format(train_noises.shape[0]))
print("Validation noise size: {}".format(valid_noises.shape[0]))

"""
## Data Generator
"""


class SpeechGenerator(keras.utils.Sequence):
    def __init__(
        self,
        file_paths,
        labels,
        class_names,
        noise_stream,
        noises=None,
        batch_size=64,
        shuffle_noises=True,
        shuffle_seed=None,
        scale=0.5,
    ):
        """
        file_paths:     List of audio file paths that the generator will load
                        during training, evaluation or prediction
        labels:         List of labels. Must be of the same length as file_paths
                        Each entry in labels is the speaker identification for the
                        corresponding audio file in file_paths
        class_names:    Labels' names which is the speakers' names' list
        noise_stream:   An array of preloaded noise streams.
                        It is of shape (number, length)
                        where:
                            number = number of noise samples that we got by chopping
                                     each noise to segments of 1 sec each at a sampling
                                     rate of 16000 and stacking them all together
                            length = number of samples in noise audio which in the case
                                     of sr = 16000 and duration of 1 sec gives 16000
                                     samples
                        It was preloaded in order to reduce the number of reading
                        from disk while training.
        noises:         An array of randomly chosen numbers between 0 and
                        noise_stream.shape[0]
                        The length of this array is equal to len(file_paths)
                        For each audio stream in file_path at index idx,
                        we will add noise_stream[noises[idx]] with some amplitude
                        manipulation
                        Those noises will be shuffled at every epoch end (if
                        shuffle_noises is not None) in order to train each audio sample
                        with different noise samples.
                        If noises is None, the samples will be trained without
                        adding noise
        batch_size:     The training or validation batch size
        shuffle_noises: Whether to shuffle noise at the end of each epoch
        shuffle_seed:   The seed to use for shuffling the noises
        scale:          Used to create the noisy audio sample as follows:
                        noisy_sample = sample + noise * prop * scale
                            where:
                                prop = max_sample_amplitude / max_noise_amplitude
                        By maximising scale, we are creating audio files that
                        are more noisy
        """
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names
        self.batch_size = batch_size
        self.noise_stream = noise_stream
        self.noises = noises
        self.shuffle_noises = shuffle_noises
        self.shuffle_seed = 1e6 if shuffle_seed is None else shuffle_seed
        self.scale = scale

        self.input_size = self.noise_stream.shape[1] // 2
        self.input_dim = (self.input_size, 1)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size

        # Getting the batches of files
        batch_file_paths = self.file_paths[i : i + self.batch_size]

        # Reading audio file
        Xt = [wavfile.read(file)[1] for file in batch_file_paths]
        Xt = np.array(Xt)

        # Adding noise
        if self.noises is not None:
            prop = np.max(Xt, axis=-1, keepdims=True) / np.max(
                self.noise_stream[self.noises[i : i + self.batch_size], :],
                axis=-1,
                keepdims=True,
            )
            Xt = (
                Xt
                + self.noise_stream[self.noises[i : i + self.batch_size], :]
                * prop
                * self.scale
            )

        # Let's now get the Fast Fourier Transformation (FFT)
        # in order to transform our signal into the frequency domain.
        # We will take only the positive frequencies hence we are taking
        # only Xf[:,:self.input_size] where input_size is half the desired sr
        Xf = np.fft.fft(Xt, axis=-1)
        Xf = np.expand_dims(np.abs(Xf[:, : self.input_size]), axis=-1)

        # Labels
        y = self.labels[i : i + self.batch_size]

        return Xf, y

    def on_epoch_end(self):
        # At the end of each epoch, we shuffle the noises so that
        # the same audio sample to be trained with different noise
        # samples in order to reduce overfitting
        if self.noises is not None and self.shuffle_noises is True:
            rng = np.random.RandomState(self.shuffle_seed)
            rng.shuffle(self.noises)


# Training sample generator with noises added to the sample
train_gen = SpeechGenerator(
    file_paths=train_file_paths,
    labels=train_labels,
    class_names=class_names,
    noise_stream=noise_stream,
    noises=train_noises,
    batch_size=BATCH_SIZE,
    shuffle_noises=True,
    shuffle_seed=SHUFFLE_SEED,
    scale=SCALE,
)

# Validation sample generator with no noise added
valid_gen = SpeechGenerator(
    file_paths=valid_file_paths,
    labels=valid_labels,
    class_names=class_names,
    noise_stream=noise_stream,
    noises=None,
    batch_size=32,
    shuffle_noises=True,
    shuffle_seed=SHUFFLE_SEED,
    scale=SCALE,
)

"""
## Model Definition

I am using a model that is very similar to the `VGG16` for image classification where I
replaced the `Conv2D` and `MaxPool2D` by `Conv1D` and `MaxPool1D` respectively.
"""

keras.backend.clear_session()

np.random.seed(42)
tf.random.set_seed(42)


def build_model(input_shape, num_classes):
    model = keras.models.Sequential(
        [
            # Block 1
            keras.layers.Conv1D(
                16,
                3,
                padding="same",
                activation="relu",
                name="block1_conv1",
                input_shape=input_shape,
            ),
            keras.layers.Conv1D(
                16, 3, padding="same", activation="relu", name="block1_conv2"
            ),
            keras.layers.MaxPool1D(pool_size=2, strides=2, name="block1_pool"),
            # Block 2
            keras.layers.Conv1D(
                32, 3, padding="same", activation="relu", name="block2_conv1"
            ),
            keras.layers.Conv1D(
                32, 3, padding="same", activation="relu", name="block2_conv2"
            ),
            keras.layers.MaxPool1D(pool_size=2, strides=2, name="block2_pool"),
            # Block 3
            keras.layers.Conv1D(
                64, 3, padding="same", activation="relu", name="block3_conv1"
            ),
            keras.layers.Conv1D(
                64, 3, padding="same", activation="relu", name="block3_conv2"
            ),
            keras.layers.Conv1D(
                64, 3, padding="same", activation="relu", name="block3_conv3"
            ),
            keras.layers.MaxPool1D(pool_size=2, strides=2, name="block3_pool"),
            # Block 4
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block4_conv1"
            ),
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block4_conv2"
            ),
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block4_conv3"
            ),
            keras.layers.MaxPool1D(pool_size=2, strides=2, name="block4_pool"),
            # Block 5
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block5_conv1"
            ),
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block5_conv2"
            ),
            keras.layers.Conv1D(
                128, 3, padding="same", activation="relu", name="block5_conv3"
            ),
            keras.layers.MaxPool1D(pool_size=2, strides=2, name="block5_pool"),
            keras.layers.Flatten(name="flatten"),
            # Dense layers
            keras.layers.Dense(256, activation="relu", name="dense1"),
            keras.layers.Dense(128, activation="relu", name="dense2"),
            # Output layer
            keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    return model


model = build_model(train_gen.input_dim, len(train_gen.class_names))

model.summary()

# Compile the model using Adam's default learning rate
model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Let's add callbacks to stop training when the model is not enhancing anymore
# 'EarlyStopping' and always keep the model that has the best val_accuracy
# 'ModelCheckPoint'

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    "model.h5", monitor="val_accuracy", save_best_only=True
)

"""
## Training
"""

# Time to train the model
history = model.fit(
    train_gen,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)

"""
## Evaluation
"""

# Let's restore the best model
best_model = keras.models.load_model("model.h5")

print(best_model.evaluate(valid_gen))

"""
Excellent accuracy!
"""

"""
Let's now consider calculating the accuracy when adding noise to the validation set and
set the noise amplitude to the same magnitude as the audio itself. It will become really
noisy ...
"""

test_gen = SpeechGenerator(
    file_paths=valid_file_paths,
    labels=valid_labels,
    class_names=class_names,
    noise_stream=noise_stream,
    noises=valid_noises,
    batch_size=32,
    shuffle_noises=True,
    shuffle_seed=SHUFFLE_SEED,
    scale=1.0,
)

print(best_model.evaluate(test_gen))

"""
Despite being quite noisy, it is still able to achieve a very good accuracy
"""
