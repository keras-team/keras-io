"""
Title: Finetuning Wav2Vec 2.0 with Hugging Face Transformers.
Author: Sreyan Ghosh
Date created: 2022/07/10
Last modified: 2022/07/10
Description: Finetuning Wav2Vec 2.0 with Hugging Face Transformers on CTC.
"""

"""
## Introduction
"""

"""
### Automatic Speech Recognition (ASR)

Automatic Speech recognition is an subfield of computer science, computational
linguistics and electrical engineering that develops methodologies and technologies that
enable the recognition and translation of spoken utterances into text by computers. With
the recent advances in Deep Neural Network (DNN) powered AI, ASR has achieved impressive
performance in the recent past.

ASR systems can be built in a variety of ways, including Connectionist Temporal
Classification (CTC), RNN-T and Encoder-Decoder models. In this notebook, we focus on
end-to-end CTC finetuning. CTC is an algorithm used to train DNNs in speech recognition,
handwriting recognition and other sequence problems. CTC is used when we don’t know how the
input aligns with the output.To get a better understanding, we refer our reader to this
[awesome blog](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)!

### Wav2Vec 2.0

[Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), is a Transformer model which solves
a Masked Acoustic Modelling (MAM) task by minimizing [InfoNCE](https://paperswithcode.com/method/infonce)
loss (which simply put solves a contratsive task) to learn high-level speech reprentations.
Wav2Vec 2.0 pre-training is done in a self-supervised fashion and doesn't need any labels
while pre-training. One can use this pre-trained model and finetune it on almost any
speech processing task. Wav2Vec 2.0 has achived state-of-the-art performance on a variety
of speech processing task including ASR, Speech Emotion Recognition, etc.

In this notebook, we will train the Wav2Vec 2.0 (Base) model, built on the
🤗 Transformers library, in an end-to-end fashion on the task of ASR. We will use CTC to
finetune our model.
"""

"""
## Setup
"""

"""
### Installing the requirements
"""

"""shell
pip install git+https://github.com/huggingface/transformers.git
pip install datasets
pip install huggingface-hub
pip install librosa
pip install jiwer
pip install pydub
pip install gdown
pip install soundfile
pip install librosa
pip install Levenshtein
"""

"""
### Importing the necessary libraries
"""

import json
import random
import logging
import numpy as np

# import tensorflow as tf
from tensorflow import keras
import IPython.display as ipd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

tf.get_logger().setLevel(logging.ERROR)

"""
### Define certain variables
"""

BATCH_SIZE = 2  # Batch-size to train the model on
TEST_SIZE = 0.1  # Size of the test set relative to entire dataset
MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-5  # Learning rate for training the model
MAX_INPUT_LENGTH_IN_SEC = 10  # Maximum length of audio input to the model

MODEL_CHECKPOINT = (
    "facebook/wav2vec2-base"  # Name of pre-trained model from 🤗 Model Hub
)

"""
## Load the LibriSpeech Dataset
"""

"""
We will now download the LibriSpeech dataset.
[LibriSpeech](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7178964&casa_token=xanx
_dg4s_0AAAAA:3i9SNXR2UBpMa9LhZben7jS9JVWHDaSsk8DiF9j6Cn_Y7z0hxfRG1V2ZQ2xZvj5EUfpSDQQY4qLA&
tag=1) is a corpus of approximately 1000 hours of 16kHz read English speech, prepared
by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.

We will load the dataset from [🤗 Datasets](https://github.com/huggingface/datasets).The
original LirbiSpeech dataset has 2 main splits, namely `clean` and `other`.
Further, each split has it's dedicated `train`, `validation` and `test` splits.
For the purpose of demonstration in this notebook, we will work with
only with the `test` split of the `clean` split of the dataset. This can
be easily done with the `load_dataset` function. The `split` argument expects
the dataset split you want to use. You can ignore the argument if you want to use
the pre-defined splits.

Additionally, we will also load our evaluation metric from 🤗 Datasets using the
`load_metric()` function. Since Word Error Rate (WER) is the most common metric to
evaluate ASR systems, we will use that for this notebook too! For more detailed
description on how the WER is calculated, we would like to refer our readers to this
[excellent blog](https://www.rev.com/blog/resources/what-is-wer-what-does-word-error-rate-mean)!
"""

from datasets import load_dataset, load_metric

# Replace the dataset name with "librispeech_asr" if you want to download the entire
# dataset
librispeech = load_dataset(
    "andreagasparini/librispeech_test_only", "clean", split="test"
)
wer_metric = load_metric("wer")

"""
Now let us see how the dataset looks like.

The dataset has the following fields:
- **file**: the path to the raw .wav file of the audio utterance
- **audio**: the audio file sampled at 16kHz
- **text**: the gold human-annotated transcript for the utterance
- **speaker_id**: The ID of the speaker for the utterance
- **chapter_id**: The chapter ID of the book being read out
- **id**: The unique ID for the utterance
"""

librispeech

"""
Many ASR datasets only provide the target text, `'text'` for each audio `'audio'` and file
`'file'`. As we see above, LibriSpeech actually provides much more information about each
audio file which many researchers use to evaluate other tasks instead of ASR
when working with LibriSpeech. However, since we are only interested in ASR in this
notebook, we will only consider the audio transcribed text for fine-tuning and remove
irrelevant columns.
"""

librispeech = librispeech.remove_columns(["speaker_id", "chapter_id", "id"])

"""
## Splitting the Dataset
"""

"""
Since we will only be working with the test-clean of LibriSpeech in this
notebook, we will divide the test-clean dataset into training and test
splits. We will not have a seperate validation split and will use the
test set to validate our model.
"""

librispeech = librispeech.train_test_split(test_size=TEST_SIZE)

"""
## Data Pre-processing
"""

"""
ASR models transcribe speech to text, which means that we both need a feature extractor
that processes the speech signal to the model's input format, *e.g.* a feature vector,
and a tokenizer that processes the model's output format to text.

In 🤗 Transformers, speech recognition models are thus accompanied by both a tokenizer,
and a feature extractor.

Before starting, lets see some examples of the text transcripts!

Let's write a short function to display some random samples of the dataset and run it a
couple of times to get a feeling for the transcriptions.
"""

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


show_random_elements(
    librispeech["train"].remove_columns(["audio", "file"]), num_examples=10
)

"""
Looks perfect! There are no special characters to be removed, and all sentences are
already normalized to upper-case!

Now let's start by creating the tokenizer responsible for decoding the model's
predictions.
"""

"""
### Create Wav2Vec2CTC Tokenizer
"""

"""
In CTC, it is common to classify speech chunks into letters, so we will do the same here.
Let's extract all distinct letters of the training and test data and build our vocabulary
from this set of letters.

We write a mapping function that concatenates all transcriptions into one long
transcription and then transforms the string into a set of chars.

First we will define a function to accomplish this and then use the `map()` function in
🤗 Datasets to execute it on our dataset. It is important to pass the argument
`batched=True` to the `map()` function so that the mapping function has access to
all transcriptions at once.
"""


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = librispeech.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=librispeech.column_names["train"],
)

"""
Now, we create the union of all distinct letters in the training dataset and test dataset
and convert the resulting list into an enumerated dictionary.
"""

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict

"""
Cool, we see that almost all letters of the alphabet occur in the dataset
and we also extracted the special characters `" "` and `'`. Note that we did
not exclude those special characters because:

- The model has to learn to predict when a word finished or else the model prediction
would always be a sequence of chars which would make it impossible to separate words from
each other.
- In English, we need to keep the `'` character to differentiate between words, *e.g.*,
`"it's"` and `"its"` which have very different meanings.
"""

"""
To make it clearer that `" "` has its own token class, we give it a more visible
character `|`. In addition, we also add an "unknown" token so that the model can later
deal with characters not encountered in Timit's training set.

Finally, we also add a padding token that corresponds to CTC's "*blank token*". The
"blank token" is a core component of the CTC algorithm. For more information, please take
a look at the "Alignment" section [here](https://distill.pub/2017/ctc/).
"""

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

vocab_dict

"""
Cool, now our vocabulary is complete and consists of 30 tokens, which means that the
linear layer that we will add on top of the pretrained speech checkpoint will have an
output dimension of 30.
"""

"""
Let's now save the vocabulary as a json file for use in our 🤗 Tokenizer.
"""

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

"""
In a final step, we use the json file to instantiate a tokenizer object with the just
created vocabulary file.

For our `Tokenizer` we use the 🤗 Wav2Vec2CTCTokenizer. Note the Tokenizer expects to know
which characters in your vocab file correspond to `unknown`, `padding` and `word
delimiter`, so you need to specify values for these arguments specifically!
"""

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)

"""
### Pre-process Audio Data

So far, we have not looked at the actual values of the speech signal but just the
transcription. In addition to `'text'`, our datasets include two more column names
`'file'` and `'audio'`. `'file'` states the absolute path of the audio file. Let's take a
look.
"""

librispeech["train"][0]["file"]

"""
Wav2Vec 2.0 expects the input in the format of a 1-dimensional array of 16 kHz.
This means that the audio file has to be loaded and resampled.

Thankfully, datasets does this automatically when calling the column audio. Let try it
out.
"""

librispeech["train"][0]["audio"]

"""
We can see that the audio file has automatically been loaded. 🤗 Datasets automatically
loads and resamples audio files on-the-fly upon calling.

The sampling rate is set to 16kHz which is what Wav2Vec 2.0 expects as an input.
"""

"""
Great, let's listen to a couple of audio files to better understand the dataset and
verify that the audio was correctly loaded.

**Note**: *You can click the following cell a couple of times to listen to different
speech samples.*
"""

rand_int = random.randint(0, len(librispeech["train"]))

print(librispeech["train"][rand_int]["text"])
ipd.Audio(
    data=np.asarray(librispeech["train"][rand_int]["audio"]["array"]),
    autoplay=True,
    rate=16000,
)

"""
It can be heard, that the speakers change along with their speaking rate, accent, etc.
Overall, the recordings sound relatively clear though, which is to be expected from a
read speech corpus.

Let's do a final check that the data is correctly prepared, by printing the shape of the
speech input, its transcription, and the corresponding sampling rate.

**Note**: *You can click the following cell a couple of times to verify multiple samples.*
"""

"""
Though the audio data is already processed in a format we want, but this is not
always the case.

Before we can feed the audio utterance samples to our model, we need to
preprocess them. This is done by a 🤗 Transformers `Feature Extractor`
which will (as the name indicates) re-sample your the inputs to sampling rate
the the model expects (in-case they exist with a different sampling rate), as well
as generate the other inputs that model requires.

To do all of this, we instantiate our `Feature Extractor` with the
AutoFeatureExtractor.from_pretrained, which will ensure:

We get a `Feature Extractor` that corresponds to the model architecture we want to use.
We download the config that was used when pretraining this specific checkpoint.
This will be cached so it's not downloaded again the next time we run the cell.
The `from_pretrained()` method expects the name of a model from the 🤗 Hub. This is
exactly similar to MODEL_CHECKPOINT and we will just pass that.
"""

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

"""
and wrap it into a `Wav2Vec2Processor` together with the tokenizer. Now we can use the
`Wav2Vec2Processor` to do pre-process both audio and targets to the format expected by
the model for CTC training!
"""

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

"""
Next, to use our `processor` we will write a simple function that helps us in the
pre-processing that is compatible with 🤗 Datasets. To summarize, our pre-processing
function should:

- Call the audio column to load and if necessary resample the audio file.
- Check the sampling rate of the audio file matches the sampling rate of the audio data a
model was pretrained with. You can find this information on the Wav2Vec 2.0 model card.
- Set a maximum input length so longer inputs are batched without being truncated.
- Encode the transcriptions to label ids


The function makes use of 🤗 Dataset's
[`map()`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlig
ht=map#datasets.DatasetDict.map) function.
"""


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


"""
Let's apply the data preparation function to all examples.
"""

librispeech = librispeech.map(
    prepare_dataset, remove_columns=librispeech.column_names["train"], num_proc=4
)

"""
This is an extra step where we will filter out the training and testing samples with
a maximum length of 10 seconds. Long input sequences require a lot of memory. Since
`Wav2Vec 2.0` is based on `self-attention` the memory requirement scales quadratically
with the input length for long input sequences. This is useful if you have GPU memory size
constraints.

**Note**: You can feel free to ignore this step if you don't have such a constraint.
"""

librispeech["train"] = librispeech["train"].filter(
    lambda x: x < MAX_INPUT_LENGTH_IN_SEC * processor.feature_extractor.sampling_rate,
    input_columns=["input_length"],
)

librispeech["test"] = librispeech["test"].filter(
    lambda x: x < MAX_INPUT_LENGTH_IN_SEC * processor.feature_extractor.sampling_rate,
    input_columns=["input_length"],
)

"""
Before we can feed the audio samples to our model, we need to pre-process them and get
them ready for the task, i.e., CTC fine-tuning. To achieve this, we define a custom `collator`
called `DataCollatorCTCWithPadding` compatible with 🤗 Datasets which does the following:

- pad the batch of audio inputs with a pre-defined padding strategy
- pad the batch of target character sequence
- replace the padded elements of target character sequence with -100
"""


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="tf",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="tf",
            )

        # replace padding with -100 to ignore loss correctly
        condition = tf.equal(
            labels_batch["input_ids"], tokenizer.convert_tokens_to_ids("[PAD]")
        )
        case_true = -100
        case_false = labels_batch["input_ids"]
        labels = tf.where(condition, case_true, case_false)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

"""
Next we define our training and testing set with which we will train our model. Again,
🤗 Datasets provides us with the `to_tf_dataset` method which will help us integrate our
dataset with the `collator` defined above. The method expects certain parameters:

- **columns**: the columns which will serve as our independant variables
- **label_cols**: the columns which will serve as our labels or dependant variables
- **batch_size**: our batch size for training
- **shuffle**: whether we want to shuffle your dataset
- **collate_fn**: our collator function
"""

train = librispeech["train"].to_tf_dataset(
    columns=["input_values", "labels"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)
test = librispeech["test"].to_tf_dataset(
    columns=["input_values", "labels"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

"""
## Defining the model
"""

"""
Now we can download the pre-trained model and fine-tune it. For finetuning
Wav2Vec 2.0 on CTC 🤗 Transformers provides us with `TFWav2Vec2ForCTC`. The
`from_pretrained` method will download and cache the model for us.

The `from_pretrained()` method expects the name of a model from the 🤗 Model Hub. As
mentioned earlier, we will use the `facebook/wav2vec2-base` model checkpoint. This
checkpoint provided by Facebook was pre-trainen on the MAM task on the complete 960hrs
of LibriSpeech data in self-supervised fashion.
"""

from transformers import TFWav2Vec2ForCTC

model = TFWav2Vec2ForCTC.from_pretrained(
    MODEL_CHECKPOINT,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    apply_spec_augment=False,
    from_pt=True,
)

"""
## Building and Compiling the the Model
"""

"""
First, before we compile our model, to evaluate it on-the-fly while training,
we will define `compute_wer()` function which will calculate the `wer` score between
the groud-truth and predictions.
"""


def compute_wer(y_true, y_pred):
    condition = tf.equal(y_true, -100)
    case_true = tokenizer.pad_token_id
    case_false = y_true
    y_true = tf.where(condition, case_true, case_false)
    pred_str = processor.batch_decode(tf.argmax(y_pred, axis=-1))
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(y_true, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return wer


"""
Now we will define our optimizer and compile the model. The loss calculation is handled
internally and so we need not worry about that!
"""

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, metrics=[compute_wer])

"""
## Training and Evaluating the Model
"""

"""
Now we can finally start training our model!
"""

tf.config.run_functions_eagerly(True)
model.fit(train, validation_data=test, epochs=MAX_EPOCHS)

"""
The `wer` isn't great, but that is primarily because the model wasn't trained well! For
best performace, one should train the model on the complete clean train split (100+360
hrs) for atleast 5 epochs!
"""

"""
## Inference
"""

"""
Now we will try to infer the model we trained on some sample audio files.
To do so, now will define a small function that will take the first 2 samples
of our test set and predict.
"""


def infer():
    model_input = next(iter(test))
    raw_logits = model(model_input).logits
    outputs = processor.batch_decode(tf.argmax(raw_logits, axis=-1))
    condition = tf.equal(model_input["labels"], -100)
    case_true = tokenizer.pad_token_id
    case_false = model_input["labels"]
    y_true = tf.where(condition, case_true, case_false)
    label_str = processor.batch_decode(y_true, group_tokens=True)

    for i in range(len(label_str)):
        print("Predicted Transcrition was: ", outputs[i])
        print("Ground Truth Transcrition was: ", label_str[i])

    return


infer()

"""
Not too bad! This is also why Wav2Vec 2.0 is famous! It can learn in extreme low-resource
data too!

**Note**: You can run it as many number of times you want and play with it!
"""

"""
Now you can push this model to 🤗 Model Hub and also share it with with all your friends,
family, favorite pets: they can all load it with the identifier
`"your-username/the-name-you-picked"` so for instance:

```python
model.push_to_hub("finetuned-wav2vec2", organization="keras-io")
tokenizer.push_to_hub("finetuned-wav2vec2", organization="keras-io")
```
And after you push your model this is how you can load it in the future!

```python
from transformers import TFWav2Vec2ForCTC

model = TFWav2Vec2ForCTC.from_pretrained("your-username/my-awesome-model")
```
"""
