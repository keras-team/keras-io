"""
Title: Speaker Recognition Using a Convolutional Network
Author: Fadi Badine
Date created: 14/06/2020
Last modified: 23/06/2020
Description: A CNN to classify speakers using Fast Fourier Transformation.
"""
"""
## Introduction

This example demostrates how to create a model to classify speakers based on the
transformation of their speech into the frequency domain using Fast Fourier
Transformation.

It shows the following:
- How to use tf.data to load, preprocess and feed audio stream into the model
- How to create a convolutional model using Conv1D and Max1Pool along with residual
connections.

Note:
- This example should be run with `TensorFlow 2.3` or higher, or `tf-nightly`.
- The noise samples in the dataset need to be resampled to a sampling rate of 16000
before using the code in this example
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.preprocessing import dataset_utils

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
EPOCHS = 1

"""
## Data Preparation
"""

"""
The dataset is composed of 7 folders that can be divided into 2 groups:
- One for the speeches containing 5 folders for 5 different speakers. Each folder contain
1500 audio files each of which is 1 second long and sampled at 16000 samples per second
- One for different noises containing 2 folders and an overall of 6 files. Those files
are longer than 1 second (and originally not sampled at 16000, but we assume that by at
this stage, you would have resampled them to 16000). We will use those 6 files to create
354 1-second-long noise samples to be used for training
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

"""
## Noise Preparation
"""

# Let's first get the list of all noise files
noise_paths, _, _ = tf.python.keras.preprocessing.dataset_utils.index_directory(
    DATASET_NOISE_PATH, "inferred", FILE_FORMATS, shuffle=False
)

# In this section:
# - We load all noise samples which should have been resampled to 16000
# - We split those noise samples to chuncks of 16000 samples which
#   correspond to 1 second duration each

i = tf.constant(0)

noises = tf.Variable([])


def read_and_process_noise(i, noises):
    noise, sr = tf.audio.decode_wav(tf.io.read_file(noise_paths[i]), desired_channels=1)

    if sr == DESIRED_SR:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(noise.shape[0] / DESIRED_SR)

        # Split the noise sample, stack and concatenate them
        if noises.shape[0] == 0:
            noises = tf.stack(tf.split(noise[: slices * DESIRED_SR], slices))
        else:
            noises = tf.concat(
                [noises, tf.stack(tf.split(noise[: slices * DESIRED_SR], slices))],
                axis=0,
            )
    else:
        print("SR for {} is different than desired. Ignoring it".format(noise_paths[i]))

    return i + 1, noises


cond = lambda i, noises: tf.less(i, len(noise_paths))

_, noises = tf.while_loop(cond, read_and_process_noise, [i, noises])

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] // DESIRED_SR
    )
)

"""
## Data Generation
"""


def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    desired_channels=-1,
    desired_samples=-1,
    batch_size=32,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """
    Generates a `tf.data.Dataset` from audio files in a directory.
    If your directory structure is:
    ```
    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav
    ```
    Then calling `audio_dataset_from_directory (main_directory, labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of audios from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
    Supported audio formats: wav.
    Arguments:
        directory: Directory where the data is located.
            If `labels` is "inferred", it should contain
            subdirectories, each containing audios for a class.
            Otherwise, the directory structure is ignored.
        labels: Either "inferred"
            (labels are generated from the directory structure),
            or a list/tuple of integer labels of the same size as the number of
            audio files found in the directory. Labels should be sorted according
            to the alphanumeric order of the audio file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode:
            - 'int': means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - 'categorical' means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - 'binary' means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - None (no labels).
        class_names: Only valid if "labels" is "inferred". This is the explict
            list of class names (must match names of subdirectories). Used
            to control the order of the classes
            (otherwise alphanumerical order is used).
        desired_channels: Number of audio channels to read.
        desired_samples: Number of samples to read from the audio file
        batch_size: Size of the batches of data. Default: 32.
        shuffle: Whether to shuffle the data. Default: True.
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation".
            Only used if `validation_split` is set.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to False.
    Returns:
        A `tf.data.Dataset` object.
        - If `label_mode` is None, it yields `float32` tensors of shape
            `(batch_size, desired_samples, desired_channels)`.
        - Otherwise, it yields a tuple `(audios, labels)`, where `audios`
            has shape `(batch_size, desired_samples, desired_channels)`,
            and `labels` follows the format described below.
    Rules regarding labels format:
        - if `label_mode` is `int`, the labels are an `int32` tensor of shape
            `(batch_size,)`.
        - if `label_mode` is `binary`, the labels are a `float32` tensor of
            1s and 0s of shape `(batch_size, 1)`.
        - if `label_mode` is `categorial`, the labels are a `float32` tensor
            of shape `(batch_size, num_classes)`, representing a one-hot
            encoding of the class index.
    """
    if labels != "inferred":
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "`labels` argument should be a list/tuple of integer labels, of "
                "the same size as the number of audio files in the target directory."
            )
    if class_names:
        raise ValueError(
            "You can only pass `class_names` if the labels are inferred from the "
            'subdirectory names in the target directory (`labels="inferred"`).'
        )

    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            "or None. Received: %s" % (label_mode,)
        )

    dataset_utils.check_validation_split_arg(validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)

    # Get the list of audio file paths along with the corresponding labels
    audio_paths, labels, class_names = dataset_utils.index_directory(
        directory, labels, FILE_FORMATS, class_names, shuffle, seed, follow_links
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
        )

    audio_paths, labels = dataset_utils.get_training_or_validation_split(
        audio_paths, labels, validation_split, subset
    )

    dataset = paths_and_labels_to_dataset(
        audio_paths, desired_channels, desired_samples, labels, label_mode, len(labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

    dataset = dataset.batch(batch_size)

    dataset.class_names = class_names

    return dataset


def paths_and_labels_to_dataset(
    audio_paths, desired_channels, desired_samples, labels, label_mode, num_classes
):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)

    audio_ds = path_ds.map(
        lambda x: path_to_audio(x, desired_channels, desired_samples)
    )
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        audio_ds = tf.data.Dataset.zip((audio_ds, label_ds))

    return audio_ds


def path_to_audio(path, desired_channels, desired_samples):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(
        audio, desired_channels=desired_channels, desired_samples=desired_samples
    )
    audio.set_shape((desired_samples, desired_channels))

    return audio


def audio_to_fft(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Let's now create 2 datasets, one for training and the other for validation
train_ds = audio_dataset_from_directory(
    DATASET_AUDIO_PATH,
    desired_channels=1,
    desired_samples=DESIRED_SR,
    batch_size=BATCH_SIZE,
    seed=SHUFFLE_SEED,
    validation_split=VALID_SPLIT,
    subset="training",
)

class_names = train_ds.class_names

valid_ds = audio_dataset_from_directory(
    DATASET_AUDIO_PATH,
    desired_channels=1,
    desired_samples=DESIRED_SR,
    seed=SHUFFLE_SEED,
    validation_split=VALID_SPLIT,
    subset="validation",
)

# The map functions will call audio_to_fft that will transform the audio wave into the
# frequency domain.
# For the training data, we are adding random noise
# For the validation data, we are not adding any noise (by setting scale = 0)

train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x, noises, scale=SCALE), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
train_ds = train_ds.prefetch(BATCH_SIZE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x, None, scale=0), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
valid_ds = valid_ds.prefetch(32)

"""
## Model Definition
"""


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)

    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv1D(filters, 3, padding="same")(x)

    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)

    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((DESIRED_SR // 2, 1), len(class_names))

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
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)

"""
## Evaluation
"""

print(model.evaluate(valid_ds))

"""
Excellent accuracy!
"""

"""
Let's now consider calculating the accuracy when adding noise to the validation set and
set the noise amplitude to the same magnitude as the audio itself. It will become really
noisy ...
"""

valid_ds = audio_dataset_from_directory(
    DATASET_AUDIO_PATH,
    desired_channels=1,
    desired_samples=DESIRED_SR,
    seed=SHUFFLE_SEED,
    validation_split=VALID_SPLIT,
    subset="validation",
)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x, noises, scale=1.0), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

print(model.evaluate(valid_ds))

"""
Despite being quite noisy, it is still able to achieve a very good accuracy
"""
