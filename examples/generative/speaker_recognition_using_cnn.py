"""
Title: Speaker Recognition Using a Convolutional Network
Author: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 14/06/2020
Last modified: 28/06/2020
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

from IPython.display import display, Audio

# Where the dataset is located
# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save them into folder 'Downloads' in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")

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
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 100

"""
## Useful Functions
"""


def index_directory(directory, formats, shuffle=True, seed=SHUFFLE_SEED):
    """
    Function that reads directory and all its subdirectories
    searching for files whose format is defined by `formats`
    Arguments:
        directory:  Root directory that contains all the files
        formats:    Files formats to look for in directory and
                    all its subdirectories
        shuffle:    Whether to shuffle the data. Default: True.
        seed:       Optional random seed for shuffling.
    Returns:
        - List of all file_paths
        - The corresponding labels
        - The different class names
    """
    class_names = []

    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            class_names.append(subdir)

    class_indices = dict(zip(class_names, range(len(class_names))))

    results = []
    filenames = []

    for dirpath in (os.path.join(directory, subdir) for subdir in class_names):
        results.append(index_subdirectory(dirpath, class_indices, formats))

    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res
        labels_list.append(partial_labels)
        filenames += partial_filenames

    index = 0
    labels = np.zeros((len(filenames),), dtype="int32")
    for partial_labels in labels_list:
        labels[index : index + len(partial_labels)] = partial_labels
        index += len(partial_labels)

    print(
        "Found {} files belonging to {} classes.".format(
            len(filenames), len(class_names)
        )
    )

    file_paths = [os.path.join(directory, fname) for fname in filenames]

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)

    return file_paths, labels, class_names


def index_subdirectory(directory, class_indices, formats):
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)
    return filenames, labels


def iter_valid_files(directory, formats):
    walk = os.walk(directory)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname


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

noise_paths, _, _ = index_directory(DATASET_NOISE_PATH, FILE_FORMATS, shuffle=False)

# In this section:
# - We load all noise samples which should have been resampled to 16000
# - We split those noise samples to chuncks of 16000 samples which
#   correspond to 1 second duration each

index = tf.constant(0)

noises = tf.Variable([])


def read_and_process_noise(index, noises):
    noise, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(noise_paths[index]), desired_channels=1
    )

    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(noise.shape[0] / SAMPLING_RATE)

        # Split the noise sample, stack and concatenate them
        if noises.shape[0] == 0:
            noises = tf.stack(tf.split(noise[: slices * SAMPLING_RATE], slices))
        else:
            noises = tf.concat(
                [noises, tf.stack(tf.split(noise[: slices * SAMPLING_RATE], slices))],
                axis=0,
            )
    else:
        print("SR for {} is different than desired. Ignoring it".format(noise_paths[i]))

    return index + 1, noises


cond = lambda index, noises: tf.less(index, len(noise_paths))

_, noises = tf.while_loop(cond, read_and_process_noise, [index, noises])

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
    )
)

"""
## Data Generation
"""


def audio_dataset_from_directory(
    directory,
    batch_size=32,
    shuffle=True,
    seed=SHUFFLE_SEED,
    validation_split=None,
    subset=None,
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
    Then calling `audio_dataset_from_directory` will return a `tf.data.Dataset`
    that yields batches of audios from the subdirectories `class_a` and `class_b`
    together with labels 0 and 1 (0 corresponding to `class_a` and 1
    corresponding to `class_b`).
    Supported audio formats: wav.
    Arguments:
        directory: Directory where the data is located.
            It should contain subdirectories, each containing audios for a class.
        batch_size: Size of the batches of data. Default: 32.
        shuffle: Whether to shuffle the data. Default: True.
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation".
            Only used if `validation_split` is set.
    Returns:
        A `tf.data.Dataset` object that yields a tuple (audios, labels)
        where audios have the shape (batch_size, DESIRED_SAMPLING_RATE, 1)
    """
    # Get the list of audio file paths along with their corresponding labels
    audio_paths, labels, class_names = index_directory(directory, FILE_FORMATS)

    # Split into training or validation
    if validation_split:
        num_val_samples = int(validation_split * len(audio_paths))
        if subset == "training":
            num_train_samples = len(audio_paths) - num_val_samples
            print("Using {} files for training.".format(num_train_samples))
            audio_paths = audio_paths[:-num_val_samples]
            labels = labels[:-num_val_samples]
        elif subset == "validation":
            print("Using {} files for validation.".format(num_val_samples))
            audio_paths = audio_paths[-num_val_samples:]
            labels = labels[-num_val_samples:]

    # Create the dataset
    dataset = paths_and_labels_to_dataset(audio_paths, labels)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

    dataset = dataset.batch(batch_size)

    dataset.class_names = class_names

    return dataset


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)

    audio_ds = path_ds.map(lambda x: path_to_audio(x))

    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    audio.set_shape((SAMPLING_RATE, 1))

    return audio


def add_noise(audio, noises=None, scale=0.5):
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

    return audio


def audio_to_fft(audio):
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
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    subset="training",
)

class_names = train_ds.class_names

valid_ds = audio_dataset_from_directory(
    DATASET_AUDIO_PATH, validation_split=VALID_SPLIT, subset="validation",
)

# Add noise to the training set
train_ds = train_ds.map(
    lambda x, y: (add_noise(x, noises, scale=SCALE), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

# Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

"""
## Model Definition
"""

keras.backend.clear_session()

np.random.seed(42)
tf.random.set_seed(42)


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


model = build_model((SAMPLING_RATE // 2, 1), len(class_names))

model.summary()

# Compile the model using Adam's default learning rate
model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Let's add callbacks to stop training when the model is not enhancing anymore
# 'EarlyStopping' and always keep the model that has the best val_accuracy
# 'ModelCheckPoint'
model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
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
Excellent results! ~ 98% validation accuracy
"""

"""
## Examples
Let's take some examples:
- predict the speaker
- compare the prediction with the real speaker
- listen to the audio to see that despite being noisy, the model is still quite accurate
"""

SAMPLES_TO_DISPLAY = 10

test_ds = audio_dataset_from_directory(
    DATASET_AUDIO_PATH,
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    subset="validation",
)

test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))

for audios, labels in test_ds.take(1):
    # Get the signal FFT
    ffts = audio_to_fft(audios)
    # Predict
    y_pred = model.predict(ffts)
    # Take random samples
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(SAMPLES_TO_DISPLAY):
        # For every sample, print the true and predicted label
        # as well as run the voice with the noise
        print(
            "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[labels[index]],
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[y_pred[index]],
            )
        )
        display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))
