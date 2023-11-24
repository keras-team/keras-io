# Audio Classification with Hugging Face Transformers

**Author:** Sreyan Ghosh<br>
**Date created:** 2022/07/01<br>
**Last modified:** 2022/08/27<br>
**Description:** Training Wav2Vec 2.0 using Hugging Face Transformers for Audio Classification.


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/wav2vec2_audiocls.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/audio/wav2vec2_audiocls.py)



---
## Introduction

Identification of speech commands, also known as *keyword spotting* (KWS),
is important from an engineering perspective for a wide range of applications,
from indexing audio databases and indexing keywords, to running speech models locally
on microcontrollers. Currently, many human-computer interfaces (HCI) like Google
Assistant, Microsoft Cortana, Amazon Alexa, Apple Siri and others rely on keyword
spotting. There is a significant amount of research in the field by all major companies,
notably Google and Baidu.

In the past decade, deep learning has led to significant performance
gains on this task. Though low-level audio features extracted from raw audio like MFCC or
mel-filterbanks have been used for decades, the design of these low-level features
are [flawed by biases](https://arxiv.org/abs/2101.08596). Moreover, deep learning models
trained on these low-level features can easily overfit to noise or signals irrelevant to the
task.  This makes it is essential for any system to learn speech representations that make
high-level information, such as acoustic and linguistic content, including phonemes,
words, semantic meanings, tone, speaker characteristics from speech signals available to
solve the downstream task. [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), which solves a
self-supervised contrastive learning task to learn high-level speech representations,
provides a great alternative to traditional low-level features for training deep learning
models for KWS.

In this notebook, we train the Wav2Vec 2.0 (base) model, built on the
Hugging Face Transformers library, in an end-to-end fashion on the keyword spotting task and
achieve state-of-the-art results on the Google Speech Commands Dataset.

---
## Setup

### Installing the requirements


```python
pip install git+https://github.com/huggingface/transformers.git
pip install datasets
pip install huggingface-hub
pip install joblib
pip install librosa
```

### Importing the necessary libraries


```python
import random
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)
```

### Define certain variables


```python
# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 1
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 32  # Batch-size for training and evaluating our model.
NUM_CLASSES = 10  # Number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 49
MAX_EPOCHS = 2  # Maximum number of training epochs.

MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Name of pretrained model from Hugging Face Model Hub
```

---
## Load the Google Speech Commands Dataset

We now download the [Google Speech Commands V1 Dataset](https://arxiv.org/abs/1804.03209),
a popular benchmark for training and evaluating deep learning models built for solving the KWS task.
The dataset consists of a total of 60,973 audio files, each of 1 second duration,
divided into ten classes of keywords ("Yes", "No", "Up", "Down", "Left", "Right", "On",
"Off", "Stop", and "Go"), a class for silence, and an unknown class to include the false
positive. We load the dataset from [Hugging Face Datasets](https://github.com/huggingface/datasets).
This can be easily done with the `load_dataset` function.


```python
from datasets import load_dataset

speech_commands_v1 = load_dataset("superb", "ks")
```

The dataset has the following fields:

- **file**: the path to the raw .wav file of the audio
- **audio**: the audio file sampled at 16kHz
- **label**: label ID of the audio utterance


```python
print(speech_commands_v1)
```

<div class="k-default-codeblock">
```
DatasetDict({
    train: Dataset({
        features: ['file', 'audio', 'label'],
        num_rows: 51094
    })
    validation: Dataset({
        features: ['file', 'audio', 'label'],
        num_rows: 6798
    })
    test: Dataset({
        features: ['file', 'audio', 'label'],
        num_rows: 3081
    })
})

```
</div>
---
## Data Pre-processing

For the sake of demonstrating the workflow, in this notebook we only take
small stratified balanced splits (50%) of the train as our training and test sets.
We can easily split the dataset using the `train_test_split` method which expects
the split size and the name of the column relative to which you want to stratify.

Post splitting the dataset, we remove the `unknown` and `silence` classes and only
focus on the ten main classes. The `filter` method does that easily for you.

Next we sample our train and test splits to a multiple of the `BATCH_SIZE` to
facilitate smooth training and inference. You can achieve that using the `select`
method which expects the indices of the samples you want to keep. Rest all are
discarded.


```python
speech_commands_v1 = speech_commands_v1["train"].train_test_split(
    train_size=0.5, test_size=0.5, stratify_by_column="label"
)

speech_commands_v1 = speech_commands_v1.filter(
    lambda x: x["label"]
    != (
        speech_commands_v1["train"].features["label"].names.index("_unknown_")
        and speech_commands_v1["train"].features["label"].names.index("_silence_")
    )
)

speech_commands_v1["train"] = speech_commands_v1["train"].select(
    [i for i in range((len(speech_commands_v1["train"]) // BATCH_SIZE) * BATCH_SIZE)]
)
speech_commands_v1["test"] = speech_commands_v1["test"].select(
    [i for i in range((len(speech_commands_v1["test"]) // BATCH_SIZE) * BATCH_SIZE)]
)

print(speech_commands_v1)
```

<div class="k-default-codeblock">
```
DatasetDict({
    train: Dataset({
        features: ['file', 'audio', 'label'],
        num_rows: 896
    })
    test: Dataset({
        features: ['file', 'audio', 'label'],
        num_rows: 896
    })
})

```
</div>
Additionally, you can check the actual labels corresponding to each label ID.


```python
labels = speech_commands_v1["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(id2label)
```

<div class="k-default-codeblock">
```
{'0': 'yes', '1': 'no', '2': 'up', '3': 'down', '4': 'left', '5': 'right', '6': 'on', '7': 'off', '8': 'stop', '9': 'go', '10': '_silence_', '11': '_unknown_'}

```
</div>
Before we can feed the audio utterance samples to our model, we need to
pre-process them. This is done by a Hugging Face Transformers "Feature Extractor"
which will (as the name indicates) re-sample your inputs to the sampling rate
the model expects (in-case they exist with a different sampling rate), as well
as generate the other inputs that model requires.

To do all of this, we instantiate our `Feature Extractor` with the
`AutoFeatureExtractor.from_pretrained`, which will ensure:

We get a `Feature Extractor` that corresponds to the model architecture we want to use.
We download the config that was used when pretraining this specific checkpoint.
This will be cached so that it's not downloaded again the next time we run the cell.

The `from_pretrained()` method expects the name of a model from the Hugging Face Hub. This is
exactly similar to `MODEL_CHECKPOINT` and we just pass that.

We write a simple function that helps us in the pre-processing that is compatible
with Hugging Face Datasets. To summarize, our pre-processing function should:

- Call the audio column to load and if necessary resample the audio file.
- Check the sampling rate of the audio file matches the sampling rate of the audio data a
model was pretrained with. You can find this information on the Wav2Vec 2.0 model card.
- Set a maximum input length so longer inputs are batched without being truncated.


```python
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_CHECKPOINT, return_attention_mask=True
)


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=True,
    )
    return inputs


# This line with pre-process our speech_commands_v1 dataset. We also remove the "audio"
# and "file" columns as they will be of no use to us while training.
processed_speech_commands_v1 = speech_commands_v1.map(
    preprocess_function, remove_columns=["audio", "file"], batched=True
)

# Load the whole dataset splits as a dict of numpy arrays
train = processed_speech_commands_v1["train"].shuffle(seed=42).with_format("numpy")[:]
test = processed_speech_commands_v1["test"].shuffle(seed=42).with_format("numpy")[:]
```

---
## Defining the Wav2Vec 2.0 with Classification-Head

We now define our model. To be precise, we define a Wav2Vec 2.0 model and add a
Classification-Head on top to output a probability distribution of all classes for each
input audio sample. Since the model might get complex we first define the Wav2Vec
2.0 model with Classification-Head as a Keras layer and then build the model using that.

We instantiate our main Wav2Vec 2.0 model using the `TFWav2Vec2Model` class. This will
instantiate a model which will output 768 or 1024 dimensional embeddings according to
the config you choose (BASE or LARGE). The `from_pretrained()` additionally helps you
load pre-trained weights from the Hugging Face Model Hub. It will download the pre-trained weights
together with the config corresponding to the name of the model you have mentioned when
calling the method. For our task, we choose the BASE variant of the model that has
just been pre-trained, since we fine-tune over it.


```python
from transformers import TFWav2Vec2Model


def mean_pool(hidden_states, feature_lengths):
    attenion_mask = tf.sequence_mask(
        feature_lengths, maxlen=MAX_FRAMES, dtype=tf.dtypes.int64
    )
    padding_mask = tf.cast(
        tf.reverse(tf.cumsum(tf.reverse(attenion_mask, [-1]), -1), [-1]),
        dtype=tf.dtypes.bool,
    )
    hidden_states = tf.where(
        tf.broadcast_to(
            tf.expand_dims(~padding_mask, -1), (BATCH_SIZE, MAX_FRAMES, HIDDEN_DIM)
        ),
        0.0,
        hidden_states,
    )
    pooled_state = tf.math.reduce_sum(hidden_states, axis=1) / tf.reshape(
        tf.math.reduce_sum(tf.cast(padding_mask, dtype=tf.dtypes.float32), axis=1),
        [-1, 1],
    )
    return pooled_state


class TFWav2Vec2ForAudioClassification(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, model_checkpoint, num_classes):
        super().__init__()
        # Instantiate the Wav2Vec 2.0 model without the Classification-Head
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            model_checkpoint, apply_spec_augment=False, from_pt=True
        )
        self.pooling = layers.GlobalAveragePooling1D()
        # Drop-out layer before the final Classification-Head
        self.intermediate_layer_dropout = layers.Dropout(0.5)
        # Classification-Head
        self.final_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # We take only the first output in the returned dictionary corresponding to the
        # output of the last layer of Wav2vec 2.0
        hidden_states = self.wav2vec2(inputs["input_values"])[0]

        # If attention mask does exist then mean-pool only un-masked output frames
        if tf.is_tensor(inputs["attention_mask"]):
            # Get the length of each audio input by summing up the attention_mask
            # (attention_mask = (BATCH_SIZE x MAX_SEQ_LENGTH) ∈ {1,0})
            audio_lengths = tf.cumsum(inputs["attention_mask"], -1)[:, -1]
            # Get the number of Wav2Vec 2.0 output frames for each corresponding audio input
            # length
            feature_lengths = self.wav2vec2.wav2vec2._get_feat_extract_output_lengths(
                audio_lengths
            )
            pooled_state = mean_pool(hidden_states, feature_lengths)
        # If attention mask does not exist then mean-pool only all output frames
        else:
            pooled_state = self.pooling(hidden_states)

        intermediate_state = self.intermediate_layer_dropout(pooled_state)
        final_state = self.final_layer(intermediate_state)

        return final_state

```

---
## Building and Compiling the model

We now build and compile our model. We use the `SparseCategoricalCrossentropy`
to train our model since it is a classification task. Following much of literature
we evaluate our model on the `accuracy` metric.


```python

def build_model():
    # Model's input
    inputs = {
        "input_values": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="float32"),
        "attention_mask": tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="int32"),
    }
    # Instantiate the Wav2Vec 2.0 model with Classification-Head using the desired
    # pre-trained checkpoint
    wav2vec2_model = TFWav2Vec2ForAudioClassification(MODEL_CHECKPOINT, NUM_CLASSES)(
        inputs
    )
    # Model
    model = tf.keras.Model(inputs, wav2vec2_model)
    # Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    # Compile and return
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


model = build_model()
```

---
## Training the model

Before we start training our model, we divide the inputs into its
dependent and independent variables.


```python
# Remove targets from training dictionaries
train_x = {x: y for x, y in train.items() if x != "label"}
test_x = {x: y for x, y in test.items() if x != "label"}
```

And now we can finally start training our model.


```python
model.fit(
    train_x,
    train["label"],
    validation_data=(test_x, test["label"]),
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
)
```

<div class="k-default-codeblock">
```
Epoch 1/2
28/28 [==============================] - 25s 338ms/step - loss: 2.3122 - accuracy: 0.1205 - val_loss: 2.2023 - val_accuracy: 0.2176
Epoch 2/2
28/28 [==============================] - 5s 189ms/step - loss: 2.0533 - accuracy: 0.2868 - val_loss: 1.8177 - val_accuracy: 0.5089

<keras.callbacks.History at 0x7fcee542dc50>

```
</div>
Great! Now that we have trained our model, we predict the classes
for audio samples in the test set using the `model.predict()` method! We see
the model predictions are not that great as it has been trained on a very small
number of samples for just 1 epoch. For best results, we recommend training on
the complete dataset for at least 5 epochs!


```python
preds = model.predict(test_x)
```

<div class="k-default-codeblock">
```
28/28 [==============================] - 4s 44ms/step

```
</div>
Now we try to infer the model we trained on a randomly sampled audio file.
We hear the audio file and then also see how well our model was able to predict!


```python
import IPython.display as ipd

rand_int = random.randint(0, len(test_x))

ipd.Audio(data=np.asarray(test_x["input_values"][rand_int]), autoplay=True, rate=16000)

print("Original Label is ", id2label[str(test["label"][rand_int])])
print("Predicted Label is ", id2label[str(np.argmax((preds[rand_int])))])
```

<div class="k-default-codeblock">
```
Original Label is  up
Predicted Label is  on

```
</div>
Now you can push this model to Hugging Face Model Hub and also share it with all your friends,
family, favorite pets: they can all load it with the identifier
`"your-username/the-name-you-picked"`, for instance:

```python
model.push_to_hub("wav2vec2-ks", organization="keras-io")
tokenizer.push_to_hub("wav2vec2-ks", organization="keras-io")
```
And after you push your model this is how you can load it in the future!

```python
from transformers import TFWav2Vec2Model

model = TFWav2Vec2Model.from_pretrained("your-username/my-awesome-model", from_pt=True)
```

