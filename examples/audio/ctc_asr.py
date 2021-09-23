"""
Title: Automatic Speech Recognition using CTC
Author: [Mohamed Reda Bouadjenek](https://rbouadjenek.github.io/)
Date created: 2021/09/23
Last modified: 2021/09/23
Description: Training a CTC-based model for automatic speech recognition.
"""

"""
## Introduction

Speech recognition is an interdisciplinary subfield of computer science
and computational linguistics that develops methodologies and technologies
that enable the recognition and translation of spoken language into text
by computers. It is also known as automatic speech recognition (ASR),
computer speech recognition or speech to text (STT). It incorporates
knowledge and research in the computer science, linguistics and computer
engineering fields.



This demonstration shows how to combine 1D CNN, RNN and a CTC loss to
build an ASR. We will use the LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.



**References:**

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Connectionist temporal
classification](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)
- [Speech recognition](https://en.wikipedia.org/wiki/Speech_recognition)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
"""

"""
## Setup
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    InputLayer,
)
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from IPython import display

"""
## Load the data: [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
Let's download the data.
"""

data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
wavs_path = data_path + "/wavs/"
metadata = data_path + "/metadata.csv"
# Read metadata file and parse it
metadata_df = pd.read_csv(metadata, sep="|", header=None)

"""
## Preprocessing
"""

"""
The dataset contains 13,100 audio files as `wav` files in the `/wavs/`.
The label (transcript) for each audio file is a string
given in the `metadata.csv` file.

"""

"""
Let's first get the max lenght of the audio which will be used for padding.
Also, we need to get a list of audio files.
"""

# Get the max lenght of the audio which will be used for padding.
# This requires to first parse all audio data.
maxlen_x = 0
with tqdm(total=len(metadata_df)) as pbar:
    for index, row in metadata_df.iterrows():
        wav_file = wavs_path + row[0] + ".wav"
        wav_file = tf.io.read_file(wav_file)
        audio, _ = tf.audio.decode_wav(wav_file)
        audio = tf.squeeze(audio, axis=-1)
        maxlen_x = max(maxlen_x, tf.shape(audio).numpy()[0])
        pbar.update(1)

# Get list of all wav files
wav_files = [wavs_path + x + ".wav" for x in metadata_df[0]]

print("")
print("=" * 50)
print("The maximum length of the audio file is maxlen_x: ", maxlen_x)
print("Number of wav files found: ", len(wav_files))
print("=" * 50)

"""
Let's now preprocess the labels.
"""

# Get a list of all labels
labels = list(metadata_df[1])
# Remove punctuation from the labels
labels = [
    "".join(l for l in stringIn if l not in "!?.:;|[](),’”“\",'-").lower()
    for stringIn in labels
]
# Replace letters with accent marks
labels = [
    label.replace("ê", "e")
    .replace("é", "e")
    .replace("è", "e")
    .replace("ü", "u")
    .replace("â", "a")
    .replace("à", "a")
    .replace("ü", "u")
    for label in labels
]

# Get maximum length of any transcription in the dataset.
# This will be used for padding during training.
maxlen_y = max([len(label) for label in labels])
# Pad the labels
padded_labels = []
for label in labels:
    while len(label) < maxlen_y:
        label += "#"
    padded_labels.append(label)
characters = set(char for label in padded_labels for char in label)

print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)
print("maxlen_x:", maxlen_x)
print("maxlen_y:", maxlen_y)

"""
Let's get mappings for the labels.
"""

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

"""
We now split the data into training and validation set.
"""


def split_data(wav_files, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(wav_files)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = (
        wav_files[indices[:train_samples]],
        labels[indices[:train_samples]],
    )
    x_valid, y_valid = (
        wav_files[indices[train_samples:]],
        labels[indices[train_samples:]],
    )
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(
    np.array(wav_files), np.array(padded_labels)
)

"""
## Create `Dataset` objects

We first create the function that describes the transformation that we apply to each
element of our dataset.
"""

frame_length = 255
frame_step = 128


def encode_single_sample(wav_file, label):
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Compute the padding tensor
    zero_padding = tf.zeros([maxlen_x] - tf.shape(audio), dtype=tf.float32)
    # 4. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 5. Pad the audio
    equal_length = tf.concat([audio, zero_padding], 0)
    # 6. Get the spectrogram
    spectrogram = tf.signal.stft(
        equal_length, frame_length=frame_length, frame_step=frame_step
    )
    # 7. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    # 8. Add dimension
    # 9. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label


"""
We now create our `tf.data.Dataset` object returns a new dataset
containing the transformed elements, in the same order as they
appeared in the input. The function `encode_single_sample` is
used to change both the values and the structure of a dataset's elements.

"""

# Define the batch size
batch_size = 128

# Define the trainig dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

"""
## Visualize the data

Let's visualize an example in our dataset, including the
audio clip, the spectrogram and the corresponding label.
"""

fig = plt.figure(figsize=(20, 5))
for batch in train_dataset.shuffle(buffer_size=16).take(1):
    spectrograms = batch[0]
    labels = batch[1]
    for i in range(1):
        input_shape = (spectrograms[i].shape[0], spectrograms[i].shape[1])
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax = plt.subplot(2, 1, 1)
        ax.imshow(np.transpose(spectrograms[i]), vmax=1)
        ax.set_title(label.replace("#", ""))
        ax.axis("off")
        # Convert to wave format
        spectrograms_ = tf.expand_dims(spectrograms[i], axis=2)
        audio = tf.signal.inverse_stft(
            tf.cast(spectrograms[i], tf.complex128), frame_length, frame_step
        )
        ax = plt.subplot(2, 1, 2)
        plt.plot(audio)
        ax.set_title("Signal Wave")
        ax.set_xlim(0, len(audio))
        display.display(display.Audio(audio, rate=16000))
plt.show()

"""
## Model
"""

"""
We first define the CTC Loss function.
"""


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


"""
We now define our model.
"""


def build_model():
    # Inputs to the model
    input_spectrogram = keras.layers.Input(
        shape=input_shape, name="spectrogram", dtype="float32"
    )
    # First conv block
    x = keras.layers.Conv1D(
        64,
        (32),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_spectrogram)
    x = keras.layers.BatchNormalization()(x)
    # Second conv block
    x = keras.layers.Conv1D(
        64,
        (32),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    # RNNs
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    # Compression layer
    x = keras.layers.Permute((2, 1))(x)
    x = keras.layers.Dense(768, activation="relu", name="compress_layer_1")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(768, activation="relu", name="compress_layer_2")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Permute((2, 1))(x)
    # Output layer
    output = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="output_layer"
    )(x)
    # Define the model
    model = keras.models.Model(
        inputs=input_spectrogram, outputs=output, name="speech_transcript_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model()
model.summary(line_length=110)


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :maxlen_y
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


class DisplayOutputs(keras.callbacks.Callback):
    """
    Displays a batch of outputs after every epoch.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0 or epoch == 0:
            return
        # Get the prediction model by extracting layers till the output layer
        for batch in self.dataset.shuffle(buffer_size=512).take(1):
            spectrograms = batch[0]
            labels = batch[1]
            preds = model.predict(spectrograms)
            pred_texts = decode_batch_predictions(preds)
            orig_texts = []
            for label in labels:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                orig_texts.append(label)
            # Print two elements for verification.
            for i in range(2):
                print("-" * 100)
                print(f"Target    : {orig_texts[i].replace('#', '')}")
                print(f"Prediction: {pred_texts[i].replace('#', '')}")


"""
## Training
"""

# Define the number of epochs
epochs = 100
# Callback function to check transcription on the val set.
validation_DisplayOutputs = DisplayOutputs(validation_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_DisplayOutputs],
)

"""
## Learning plot to monitor the model's learning
"""


def learning_plots(history):
    plt.figure(figsize=(17, 4))
    ax1 = plt.subplot(1, 1, 1)
    for l in history.history:
        if l == "loss" or l == "val_loss":
            loss = history.history[l]
            plt.plot(range(1, len(loss) + 1), loss, label=l)

    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("CTC Loss")
    plt.grid()
    plt.legend()
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    plt.show()


learning_plots(history)

"""
## Inference
"""

#  Let's check results on more validation samples
for batch in validation_dataset.take(1):
    spectrograms = batch[0]
    labels = batch[1]
    preds = model.predict(spectrograms)
    pred_texts = decode_batch_predictions(preds)
    orig_texts = []
    for label in labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    for i in range(16):
        print("-" * 50)
        print(f"Target    : {orig_texts[i].replace('#', '')}")
        print(f"Prediction: {pred_texts[i].replace('#', '')}")
