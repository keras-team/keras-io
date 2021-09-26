"""
Title: Automatic Speech Recognition using CTC
Author: [Ngoc Dung Huynh](https://www.linkedin.com/in/parkerhuynh/) and [Mohamed Reda Bouadjenek](https://rbouadjenek.github.io/)
Date created: 2021/09/26
Last modified: 2021/09/26
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



This demonstration shows how to combine 2D CNN, RNN and a Connectionist
Temporal Classification (CTC) loss to build an ASR. CTC is an algorithm
used to train deep neural networks in speech recognition, handwriting
recognition and other sequence problems. CTC is used when  we don’t know
how the input alligns with the outputs (how the characters in the transcript
align to the audio). The model we create is similar to the
[DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html).

To extract speech features we will used the
[`python_speech_features's`
package](https://python-speech-features.readthedocs.io/en/latest/)
This library provides common speech features for ASR including
Mel-frequency cepstral coefficients (MFCCs) and filterbank energies.
To install `python_speech_features's` package, use the following command line:

```
> pip install python_speech_features

```


We will use the LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.

We will evaluate the quality of the model using [Word Error Rate
(WER)](https://en.wikipedia.org/wiki/Word_error_rate). WER is obtained by adding up
the substitutions, insertions, and deletions that occur in a sequence of
recognized words. Divide that number by the total number of words originally
spoken. The result is the WER. To get the WER score you need to install the
[jiwer](https://pypi.org/project/jiwer/) package. Use the following command line:

```
> pip install jiwer

```





**References:**

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Speech recognition](https://en.wikipedia.org/wiki/Speech_recognition)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
- [DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)

"""

"""
## Setup
"""

import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from scipy.io import wavfile
import python_speech_features
from jiwer import wer


"""
## Load the LJSpeech Dataset

Let's download the [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/).
The dataset contains 13,100 audio files as `wav` files in the `/wavs/`.
The label (transcript) for each audio file is a string
given in the `metadata.csv` file. The fields are:

- `ID:` this is the name of the corresponding .wav file
- `Transcription:` words spoken by the reader (UTF-8)
- `Normalized Transcription:` transcription with numbers,
ordinals, and monetary units expanded into full words b(UTF-8).

For this demo we will use on the `Normalized Transcription` field.

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22050 Hz.


"""

data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
wavs_path = data_path + "/wavs/"
metadata_path = data_path + "/metadata.csv"


# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
metadata_df.head(3)


"""
We now split the data into training and validation set.

"""

split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")


"""
## Preprocessing
"""

"""
Now, let's create a class DataGenerator, which will be used for
real-time data feeding to the Keras model. We make the latter
inherit the properties of `keras.utils.Sequence` class so that
we can leverage nice functionalities.

"""


class DataGenerator(keras.utils.Sequence):
    """
    Create a data generator to  be used for real-time data feeding to the Keras model.
    """

    def __init__(self, dataset, wavs_path, vocabulary, nfilt, batch_size=32):
        """
        Initialization.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_len_train = 0
        self.vocabulary = vocabulary
        self.on_epoch_end()
        self.wavs_path = wavs_path
        self.nfilt = nfilt

    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_data = [self.dataset.iloc[k] for k in indexes]
        audios, labels = self.__data_generation(batch_data)
        return audios, labels

    def text_to_idx(self, text):
        """
        Convert a label to indexes.
        """
        text = text.lower()
        idx = []
        for char in text:
            if char in self.vocabulary:
                idx.append(self.vocabulary.index(char))
        return idx

    def normalize(self, audio):
        """
        Normalize raw audio by deviding by the max value.
        """
        gain = 1.0 / (np.max(np.abs(audio)) + 1e-5)
        return audio * gain

    def standardize(self, features):
        """
        Standardize the Mel-filter bank coefficients (FBANK).
        """
        mean = np.mean(features)
        std = np.std(features)
        return (features - mean) / std

    def audio_to_features(self, audio):
        """
        Generat the logarithmic Mel-filter bank coefficients (FBANK).
        """
        sf, audio = wavfile.read(f"{self.wavs_path}{audio}.wav")
        audio = self.normalize(audio.astype(np.float32))
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        features = python_speech_features.logfbank(
            audio, nfilt=self.nfilt, winlen=0.02, winstep=0.01
        )
        return self.standardize(features)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.dataset))

    def __data_generation(self, batch_data):
        """
        Returns one batch of data.
        """
        audios = []
        labels = []
        label_len = []
        audio_len = []

        for filename, transcript in batch_data:
            # Process audio
            audio = self.audio_to_features(filename)
            audios.append(audio)
            audio_len.append(len(audio))
            # Process labels
            label = self.text_to_idx(transcript)
            labels.append(label)
            label_len.append(len(label))

        max_audio_len = max(audio_len)
        max_label_len = max(label_len)
        audios = keras.preprocessing.sequence.pad_sequences(
            audios, maxlen=max_audio_len, dtype="float32", value=0, padding="post"
        )
        labels = keras.preprocessing.sequence.pad_sequences(
            labels,
            maxlen=max_label_len,
            value=self.vocabulary.index(""),
            padding="post",
        )
        return audios, labels


"""
## Create `DataGenerator` objects

We first prepare the vocabulary to be used.

"""

# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Add a special characters to be used for padding
characters.append("")
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(
    vocabulary=characters, mask_token=None, num_oov_indices=0
)
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True,
    num_oov_indices=0,
)

print(f"The vocabulary is: {char_to_num.get_vocabulary()}")


"""
We now create our `DataGenerator` objects that contain
the transformed elements, in the same order as they
appeared in the input.
"""

# Batch size used
batch_size = 32
#  The number of filters in the filterbank.
nfilt = 160
# Create our two data generators
train_dataset = DataGenerator(
    df_train, wavs_path, char_to_num.get_vocabulary(), nfilt, batch_size
)
validation_dataset = DataGenerator(
    df_val, wavs_path, char_to_num.get_vocabulary(), nfilt, batch_size
)


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
We now define our model. We will define a model similar to
[DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
"""


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """
    This model is similar to DeepSpeech 2
    https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html
    """
    # Model's input
    input_spectrogram = layers.Input([None, input_dim], name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape([-1, input_dim // 4 * 32])(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.TimeDistributed(layers.Dense(units=rnn_units * 2))(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=nfilt, output_dim=char_to_num.vocabulary_size(), rnn_units=512
)
model.summary(line_length=110)


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class callback(keras.callbacks.Callback):
    """
    Displays a batch of outputs after every epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


"""
## Training and Evaluating
"""

# Define the number of epochs.
epochs = 1
# Callback function to check transcription on the val set.
validation_callback = callback(validation_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)


"""
## Inference
"""

#  Let's check results on more validation samples
predictions = []
targets = []
for batch in validation_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 5):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)


"""
## Conclusion


In practice, you should train for around 20-30 epochs or more. Each epoch
took approximately 5-6mn using a `GeForce RTX 2080 Ti` GPU.
The model we trained at 30 epochs has a `Word Error Rate (WER) = 16.45%`.

Some of the transcriptions around epoch 30 (results keep improving after that):



**Audio file: LJ017-0048.wav**
```
- Target    : people came to stare at the supposed coldblooded prisoner
- Prediction: people came to stair at the supposed cold blooded prisoner

```


**Audio file: LJ007-0204.wav**
```
- Target    : which has been demonstrably proved to be the fruitful source of all the
abuses and irregularities which have so long disgraced newgate
- Prediction: which has been demostrably proved to be the footful source of allf the
abuses and irregularities which have so long desquraced newgate
```

**Audio file: LJ019-0233.wav**
```
- Target    : and when it was completed both sides of the prison were brought into
harmony with modern ideas
- Prediction: and when at was completed both side of the prison were brought into harmany
with modern ideas

```


"""
