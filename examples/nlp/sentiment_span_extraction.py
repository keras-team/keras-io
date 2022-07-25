"""
Title: Sentiment Span Extraction with 🤗 Transformers.
Author: Sreyan Ghosh
Date created: 2022/07/24
Last modified: 2022/07/24
Description: Training RoBERTa using 🤗 Transformers for Sentiment Span Extraction.
"""

"""
## Introduction
"""

"""
Recent years have seen the paradigm shift of Entity Recognition (ER) system
from sequence labeling to span classification. Span prediction based systems
are a relatively newly-explored framework, and have achieved state-of-the-art
in tasks like [Named Entity Recognition](https://arxiv.org/abs/2106.00641).

Beyond ER, span extraction and prediction also find it's use in several other
important tasks like [sentiment span extraction](https://aclanthology.org/P19-1051/),
[toxicity span extraction](https://aclanthology.org/2021.semeval-1.6/) etc. For example,
sentiment span extraction can aid in opinion mining from tweets or reviews and and toxic
span extraction can help in more efficient tweet moderation in long posts. Thus the question
that needs to be answered is, can we identify a sequence of contiguous words, also known as a
span, that actually highlights the sentiment or toxicity of a post?

In this notebook, we will build a model which will help us pick out the span from a tweet
that reflects it's sentiment. For accomplising this,  we will fine-tune a pre-trained RoBERTa
model to which we will add a span classification head. For this task, we will use 🤗 Transformers
on the dataset released in [this Kaggle Competition](https://www.kaggle.com/competitions/tweet-sentiment-extraction)
loaded from 🤗 Datasets.

Span extraction in general can be done in 2 ways. 1) Predicting the start and end token
of the span in a sentence and 2) Classifying each exhaustive contiguous group of words
represented by a single vector. In this notebook, we will implement the first kind.
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
"""

"""
### Importing the necessary libraries
"""

import random
import logging

import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(48)

"""
### Define certain variables
"""

MAX_LEN = 96
MAX_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
TRAIN_TEST_SPLIT = 0.1

MODEL_CHECKPOINT = "roberta-base"
BASE_URL = "https://huggingface.co/datasets/SetFit/tweet_sentiment_extraction/raw/main/raw_data/"

"""
## Load the Kaggle Tweet Sentiment Extraction Dataset
"""

"""
We will now download the [Kaggle Tweet Sentiment Extraction
Dataset](https://www.kaggle.com/competitions/tweet-sentiment-extraction). This
competition had been hosted by Kaggle in 2020 and since then the dataset
has been quite popular for training and evaluating neural models built for solving the
task of span extraction/classification. The dataset consists of two pre-defined splits, the
train and the test splits, with a a total of 31,015 tweets, their corresponding sentiment labels
and the part of the tweet that reflects the sentiment.

We will load the dataset from [🤗 Datasets](https://github.com/huggingface/datasets).
This can be easily done with the `load_dataset` function. We will first download the raw
`csv` and then load it with the 🤗 Dataset class for further processing. Both these actions can
be easily done using appropriate arguments in `load_dataset` function. The `csv` argument
specifies that the dataset will be dowloaded is in `csv` format. The `data_files` argument
helps map the name of the split to the url from which the `csv` will be downloaded.
The original `test` split does not have sentiment spans and thus in this notebook we will
only work with the `train` split.
"""

from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train": [BASE_URL + "train_csv"]})

"""
The dataset has the following fields:

- **textID**: unique ID assigned to a tweet entry
- **text**: the original tweet
- **selected_text**: the part of the tweet that reflects the sentiment
- **sentiment**: the sentiment label for the corresponding tweet
"""

dataset

"""
## Data Pre-processing
"""

"""
Before we go ahead with pre-processing our data to a format expected by the model,
we will first split our downloaded `train` data into smaller `train` and `test` splits.
For the sake of of demonstrating the workflow, in this notebook we will only take
small subsets of the entire dataset. We will not have a seperate validation split
and will use the test set to validate our model.
"""

dataset = dataset["train"].train_test_split(
    train_size=TRAIN_TEST_SPLIT, test_size=TRAIN_TEST_SPLIT
)

"""
Before we can feed those texts to our model, we need to pre-process them and get them
ready for the task and the model. This is done by a 🤗 Transformers `Tokenizer` which will
tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained
vocabulary) and put it in a format the model expects, as well as generate the other inputs
that model requires.

The `from_pretrained()` method expects the name of a model from the 🤗 Model Hub. This is
exactly similar to MODEL_CHECKPOINT declared earlier and we will just pass that.
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

"""
Before we move on to the next step, it is important to understand the inputs that we will
be feeding to our model and our final model architecture. As we discussed ealier, in this
task and for `span extraction` in general, the main task of the model is to predict the
`start index` and the `end index` of the span, where indices correspond to `words` or
`tokens` in the sentence. Thus, the final model should have two softmax outputs each taking the
shape of the total number of tokens in the sentence (with padding and generally kept fixed for
`span extraction` tasks). The input to the model is just tokenized words.

In this specific task, since a sentence can have all 3 of `positive`, `negative` and
`neutral` spans, we additionally append sentiment information, in the form of a tokenized
sentiment string label to our sentences. This will inform the model regarding the type of
span to extract from the sentence. In this notebook, we just append the original sentiment
label while training and inference.

Let's see a random example of how our input to the model will look like. We will write a
simple function `show_random_example()` to do that:
"""


def show_random_example():

    i = random.randint(0, dataset["train"].num_rows)
    example = dataset["train"][i]
    tweet = tokenizer.tokenize(example["text"])
    sentiment = example["sentiment"]
    modified_tweet = ["<s>"] + tweet + ["</s>", "</s>"] + [sentiment] + ["</s>"]

    print("The original tweet is:", example["text"])
    print("The modified tweet is:", " ".join(modified_tweet))


show_random_example()

"""
The "Ġ" in front of each token is specific to RoBERTa tokenizer which highlights the start
of a new token with this specific unicode character.

Next, let's tokenize the sentiment labels in our dataset for use in data preparation.
"""

sentiment_id = {}

for sentiment in list(set(dataset["train"]["sentiment"])):
    sentiment_id[sentiment] = tokenizer.convert_tokens_to_ids("Ġ" + sentiment)

sentiment_id

"""
We will now write a simple function called `prepare_train_features()` that helps us
in the pre-processing and that is compatible with 🤗 Datasets. To summarize, our
pre-processing function should:

- Tokenize the tweet into it's corresponding token ids that will be used for embedding
look-up in RoBERTa. Additionally add the tokenized sentiment label to each tweet.
- Create additional inputs for the model like `token_type_ids`, `attention_mask`, etc.
- Create targets for training the model. Our model has 2 targets for each training sample,
namely the `start_tokens` and `end_tokens`, which correspond to the starting and ending
indices
of the span to be extracted from each tweet. These indices are then one-hot encoded which
transforms it into a list of size MAX_LEN.
"""


def prepare_train_features(examples):

    ct = len(examples["textID"])

    # Define empty numpy arrays
    examples["input_ids"] = np.ones((ct, MAX_LEN), dtype="int32")
    examples["attention_mask"] = np.zeros((ct, MAX_LEN), dtype="int32")
    examples["token_type_ids"] = np.zeros((ct, MAX_LEN), dtype="int32")
    examples["start_tokens"] = np.zeros((ct, MAX_LEN), dtype="int32")
    examples["end_tokens"] = np.zeros((ct, MAX_LEN), dtype="int32")

    for k in range(ct):

        # Find Overlap
        text1 = " ".join(examples["text"][k].split())
        text2 = " ".join(examples["selected_text"][k].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx : idx + len(text2)] = 1
        if text1[idx - 1] == " ":
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)[1:-1]

        # Find ID Offsets
        offsets = []
        idx = 0
        for t in enc:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # Find start and end tokens
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = sentiment_id[examples["sentiment"][k]]
        examples["input_ids"][k, : len(enc) + 5] = [0] + enc + [2, 2] + [s_tok] + [2]
        examples["attention_mask"][k, : len(enc) + 5] = 1
        if len(toks) > 0:
            examples["start_tokens"][k, toks[0] + 1] = 1
            examples["end_tokens"][k, toks[-1] + 1] = 1

    return examples


"""
To apply this function on all the tweets in our dataset, we just use the
`map` method of our `dataset` object we created earlier. This will apply the function
on all the elements of all the splits in `dataset`, so our training and testing
data will be preprocessed in one single command.

The `batched` argument specifies that input to the fucntion will be batches of samples
from the dataset. We set `batch_size` to be `-1` because we want to process the entire
dataset at once! Additionally, we ask our method to remove all original columns from the
dataset as they will be of no use to us after tokenization. We do this through the
`remove_columns` argument.
"""

tokenized_dataset = dataset.map(
    prepare_train_features,
    batched=True,
    batch_size=-1,
    remove_columns=dataset["train"].column_names,
)

"""
Next we define a data `collator` to batch our dataset while training and inference. We
use the `collator` called the `DefaultDataCollator` provided by the 🤗 Transformers
library on our dataset. The `return_tensors='tf'` argument ensures that we get `tf.Tensor`
objects back.
"""

from transformers import DefaultDataCollator

collater = DefaultDataCollator(return_tensors="tf")

"""
Next we define our training and validation set with which we will train and evaluate our
model. Again, 🤗 Datasets provides us with the `to_tf_dataset` method which will help us
integrate our dataset with the `collator` defined above. The method expects certain parameters:

- **columns**: the columns which will serve as our independant variables
- **label_cols**: the columns which will serve as our labels or dependant variables
- **batch_size**: our batch size for training
- **shuffle**: whether we want to shuffle our training dataset
- **collate_fn**: our collator function
"""

train = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=["start_tokens", "end_tokens"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collater,
)

test = tokenized_dataset["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=["start_tokens", "end_tokens"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collater,
)

"""
## Defining the model
"""

"""
We now define our model. To be precise, we define a `RoBERTa` model (base architecture)
and add two Classification-Heads on top to output two probability ditributions across all
tokens in the padded sample. Here each of the two Classification-Heads denotes the
probability of each token in a tweet to be the starting and ending token of the span.
Since the model might get complex we will first define the RoBERTa model with
Classification-Head as a Keras layer and then build the model using that.

We define our main RoBERTa BASE model using the `TFRobertaModel` class. This will
instantiate a model which will output 768 or 1024 dimensional embeddings according to
the config you choose (BASE or LARGE). The `from_pretrained()` additionally helps you
load pre-trained weights from the 🤗 Model Hub. It will download the pre-trained weights
together with the config corresponding to the name of the model you have mentioned when
calling the method. For our task, we will choose the BASE variant of the model that has
just been pre-trained, since we will fine-tune over it.

We keep our Classification-Heads to be independant Convolution layers followed by flatten
and softmax operations. The Conv1D with 1 filter and kernel size as (1,1) acts as a linear
layer and down-projects the (BATCH_SIZE x MAX_LEN x 768) embedding to (BATCH_SIZE x
MAX_LEN x 1). Next the `flatten` operation converts it to (BATCH_SIZE x MAX_LEN) and finally
the `softmax` operation converts it to exactly what we want! A probability distribution
across our tokens!
"""

from transformers import TFRobertaModel


class RoBERTa_SpanExtraction(tf.keras.layers.Layer):

    """Combines the RoBERTa encoder with a span extraction
       Classification-Head
    """

    def __init__(self, model_checkpoint):
        super(RoBERTa_SpanExtraction, self).__init__()
        self.roberta = TFRobertaModel.from_pretrained(model_checkpoint)
        self.conv_1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_2 = tf.keras.layers.Conv1D(1, 1)
        self.dropout_1 = tf.keras.layers.Dropout(0.1)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, input_sample):
        # We take only the first output in the returned dictionary corresponding to the
        # output of the last layer of the RoBERTa model
        hidden_states = self.roberta(
            input_sample["input_ids"],
            attention_mask=input_sample["attention_mask"],
            token_type_ids=input_sample["token_type_ids"],
        )[0]

        start_tokens = self.dropout_1(hidden_states)
        start_tokens = self.conv_1(start_tokens)
        start_tokens = self.flatten(start_tokens)
        start_tokens = self.softmax(start_tokens)

        end_tokens = self.dropout_2(hidden_states)
        end_tokens = self.conv_2(end_tokens)
        end_tokens = self.flatten(end_tokens)
        end_tokens = self.softmax(end_tokens)

        return {"start_tokens": start_tokens, "end_tokens": end_tokens}


"""
## Building and Compiling the model
"""

"""
We now build and compile our model. We will use the `categorical_crossentropy`
to train our model since it is a classification task.
"""


def build_model():
    # Model's input
    inputs = {
        "input_ids": tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32),
        "attention_mask": tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32),
        "token_type_ids": tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32),
    }

    # Instantiate the RoBERTa model with Classification-Head using the desired
    # pre-trained checkpoint
    roberta_model = RoBERTa_SpanExtraction("roberta-base")(inputs)
    # Model
    model = tf.keras.Model(inputs, roberta_model)
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Compile and Return
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


model = build_model()

"""
## Training and Evaluating the Model
"""

"""
Now we can finally start training our model!
"""

model.fit(train, validation_data=test, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)

"""
Now that our model is trained, we would like to evaluate it's performance! To evaluate
our predicted spans against the ground truth spans, we will make use of the
[word-level Jaccard Score](https://en.wikipedia.org/wiki/Jaccard_index). Next we define a
simple function to define our evaluation metric.
"""


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


"""
Now that we have defined our evaluation metric, we will now predict the spans for tweets
in the test set using the `model.predict()` method!
"""

preds = model.predict(test)

"""
Finally we are all set now! We can now calculate the word-level `Jaccard Score` for our
test set! We will write a small function that helps us calculate the same.
"""


def calculate_jacard(predictions, dataset):

    preds_start = predictions["start_tokens"]
    preds_end = predictions["end_tokens"]

    prediction_strings = []
    jaccard_scores = []

    for k in range(dataset["test"].num_rows):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a > b:
            st = ""
        else:
            text1 = " ".join(dataset["test"]["text"][k].split())
            enc = tokenizer.encode(text1)[1:-1]
            st = tokenizer.decode(enc[a - 1 : b])
        # Store results and original sentences
        prediction_strings.append(
            [
                st,
                dataset["test"]["selected_text"][k],
                dataset["test"]["text"][k],
                dataset["test"]["sentiment"][k],
            ]
        )
        # Store Jaccard Scores for each sample in the test set
        jaccard_scores.append(jaccard(st, dataset["test"]["selected_text"][k]))

    return jaccard_scores, prediction_strings


jaccard_scores, prediction_strings = calculate_jacard(preds, dataset)

"""
Great! Now that we have defined our function we are all set. The code snippet below
will show you the average `Jaccard Score` and the model inference for a random example!
"""

print(
    "The average Jaccard Similarity Score for the test set is: ",
    np.mean(jaccard_scores),
)

k = random.randint(0, dataset["test"].num_rows)
print("Original Sentence: ", prediction_strings[k][2])
print("Original Sentiment: ", prediction_strings[k][3])
print("Original Span: ", prediction_strings[k][1])
print("Predicted Span: ", prediction_strings[k][0])

"""
Now you can push this model to 🤗 Model Hub and also share it with with all your friends,
family, favorite pets: they can all load it with the identifier
`"your-username/the-name-you-picked"` so for instance:
```python
model.push_to_hub("roberta-span-extraction", organization="keras-io")
tokenizer.push_to_hub("roberta-span-extraction", organization="keras-io")
```
"""