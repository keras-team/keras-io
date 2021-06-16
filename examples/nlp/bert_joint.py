"""
Title: Joint NER and sentence classification
Author: Varun Singh
Date created: Jun 16, 2021
Last modified: Jun 16, 2021
Description: Joint Named Entity Recognition (NER) and sentence classification from 2 different datasets.
"""
"""
## Introduction

Named Entity Recognition (NER) is the process of identifying named entities in text. Example of named entities are: "Person", "Location", "organization", "Dates" etc. NER is essentially a token classification process where every token is classified into one or more categories depending upon the number of named entities we want to identify.

Sentence classification is the process of assigning a label to an entire sentence. Examples include sentiment analysis and restaurant reviews.

For both of these tasks, we need labeled datasets. Ideally we want a dataset that has both sentence level labels as well as word level labels. However, in practice this is not very common. A lot of datasets are either one or the other. Further, if you have only one dataset and you want to use an open source public dataset for the other task, it's not very straight-forward to do so.


In this exercise, we will train a BERT model to do both NER and sentence classification using 2 different datasets - one that contains only word level labels and the other only sentence level.
"""
"""
## Install the open source transformers and dataset package from Huggingface
"""

"""shell
!pip3 install transformers
!pip3 install datasets
!wget https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py
"""

"""
## Setup
"""

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizer, TFBertForTokenClassification
from tqdm import trange, tqdm
from collections import Counter
from conlleval import evaluate

os.environ["HTTP_PROXY"] = "http://yqk798:Ajk791dp3@entproxy.kdc.capitalone.com:8099"
os.environ["HTTPS_PROXY"] = "http://yqk798:Ajk791dp3@entproxy.kdc.capitalone.com:8099"
os.environ["http_proxy"] = "http://yqk798:Ajk791dp3@entproxy.kdc.capitalone.com:8099"
os.environ["https_proxy"] = "http://yqk798:Ajk791dp3@entproxy.kdc.capitalone.com:8099"

"""
## Setup BertTokenizer
"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

"""
## Load the datasets
"""

"""
We will be using the CoNLL 2003 NER dataset and the IMDB movie review dataset. For
convenience, we will use Hugginggface's dataset library to download the datasets.
"""

ner_dataset = load_dataset("conll2003")
classify_dataset = load_dataset("imdb")

# Shuffle the datasets
ner_train_ds = ner_dataset["train"].shuffle()
classify_train_ds = classify_dataset["train"].shuffle()

ner_test_ds = ner_dataset["test"]
classify_test_ds = classify_dataset["test"]

# There are 9 labels in the CoNLL dataset
print(set(sum(ner_dataset["train"]["ner_tags"], [])))

# There are equal nunber of positive and negative reviews in the IMDB dataset
print(Counter(classify_train_ds["label"]))

ner_train_ds.num_rows, classify_train_ds.num_rows

"""
## Make the NER label lookup table
"""

"""
NER labels are usually provided in IOB, IOB2 or IOBES formats. Checkout this link for
more information:
[Wikipedia](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
[Wikipedia](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))

Note that we start our label numbering from 1 since 0 will be reserved for padding. We
have a total of 12 labels: 9 from the NER dataset, 2 from the classification dataset and
1 extra label 'X'. The purpose of 'X' is to mark tokens that should not be used for the
calculation of loss. So for e.g. for the NER, we don't have a sentence level
classification tag. So any sentence level information present in the [CLS] token of BERT
should be not be used for the calculation of loss. Simiarly for the sentence
classification dataset, we don't have any token level labels, so all token level labels
should be ignored for the calculation of loss.
"""


def make_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels + ["negative", "positive", "X"]
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


mapping = make_tag_lookup_table()
mapping

"""
## Create a keras Sequence representing the joint dataset
"""

"""
We will create a Keras Sequence class that will contain both of the datasets. In the
__getitem__ method of this class, we will check the structure of the data to ensure we
process it appropriately. Note that the CoNLL dataset contains 9 labels ranging from 0-8.
Since we reserve label 0 for the padding token, we will increment each value by 1. The
IMDB dataset contains 2 labels - 0 and 1 and we will increment those by 10 since our
combined label space contains labels from both the datasets.
"""


class NERDataset(tf.keras.utils.Sequence):
    def __init__(self, ner_ds, classify_ds, maxlen=64, batch_size=32, shuffle=True):
        self.data = list(ner_ds)
        max_labels = 9

        self.data += list(classify_ds)
        max_labels += 2

        self.data = np.array(self.data, dtype=object)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.all_labels = list(range(max_labels + 2))

        # ignore_label corresponds to label 'X' with an index of 12
        self.ignore_label = self.all_labels[-1]

        # Shuffle the dataset
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = self.data[inds]
        all_tokens, all_tags, all_masks = [], [], []

        for data in batch_data:
            tokens, tags = [], []

            # The CoNLL dataset is already split in words. Since we will be using the BERT tokenizer,
            # we will have to tokenize each word to produce sub-word tokens
            if "ner_tags" in data:
                for token, tag in zip(data["tokens"], data["ner_tags"]):
                    tag += 1
                    bert_tokens = tokenizer.tokenize(token)
                    tokens.extend(bert_tokens)
                    tags.extend([tag] + [self.ignore_label] * (len(bert_tokens) - 1))
                tokens = ["[CLS]"] + tokens[: self.maxlen - 2] + ["[SEP]"]
                tags = (
                    [self.ignore_label] + tags[: self.maxlen - 2] + [self.ignore_label]
                )

            else:
                review, sentiment = data["text"], data["label"]
                sentiment = sentiment + 10
                tokens = tokenizer.tokenize(review)
                tokens = ["[CLS]"] + tokens[: self.maxlen - 2] + ["[SEP]"]

                # We will ignore all the tokens except the [CLS] token for the IMDB  dataset
                # since the dataset only has a sentence level label
                tags = [sentiment] + [self.ignore_label] * (len(tokens) - 1)

            # Attention mask
            masks = [1] * len(tokens)
            encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)

            all_tokens.append(encoded_tokens)
            all_tags.append(tags)
            all_masks.append(masks)

        padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            all_tokens, padding="post"
        )
        padded_tags = tf.keras.preprocessing.sequence.pad_sequences(
            all_tags, padding="post"
        )
        padded_masks = tf.keras.preprocessing.sequence.pad_sequences(
            all_masks, padding="post"
        )
        return {"input_ids": padded_tokens, "attention_mask": padded_masks}, padded_tags

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


batch_size = 32
train_dataset = NERDataset(
    ner_train_ds, classify_train_ds, maxlen=128, batch_size=batch_size, shuffle=False
)

# Number of batches
print(len(train_dataset))

# Verify that we have a total of 13 labels
print(train_dataset.all_labels)

# The index of the ignore_label
print(train_dataset.ignore_label)

"""
## Load the BERT model
"""

model = TFBertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(train_dataset.all_labels),
    output_attentions=False,
)

"""
Custom loss function which masks out loss values corresponding to labels that need to be
ignored. For the sentence classification dataset, this means masking out all tokens
except for the [CLS] token, while for the NER dataset, this means the opposite.
"""


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true != train_dataset.ignore_label), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


loss = CustomLoss()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=loss)

"""
## Train the model using a loop
"""

model.fit(train_dataset, shuffle=True, epochs=1)

text = "EU rejects German call to boycott British lamb."
text = "wow that movie sucked so bad! Dont watch this one"
text = "One of the best movies ever!"
tokens = tokenizer.encode(text)

output = model({"input_ids": np.array([tokens])})

predicted_tag_ids = np.argmax(output.logits, axis=-1)

tags = [mapping[tag] for tag in predicted_tag_ids[0]]
print(tags)

test_dataset = NERDataset(
    ner_test_ds, classify_test_ds, maxlen=128, batch_size=1024, shuffle=False
)

len(test_dataset)

"""
## Metrics calculation
"""

"""
Here is a function to calculate the metrics. The function calculates accuracy for the
sentence classification dataset and F1 scores for the NER dataset.
"""


def calculate_metrics(dataset):
    all_true_tags, all_predicted_tags = [], []

    for X, y in dataset:
        output = model(X)
        predictions = np.argmax(output.logits, axis=-1)
        predictions = np.reshape(predictions, [-1])

        true_tag_ids = np.reshape(y, [-1])

        mask = true_tag_ids != train_dataset.ignore_label
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tags.append(true_tag_ids)
        all_predicted_tags.append(predicted_tag_ids)

    all_true_tags = np.concatenate(all_true_tags)
    all_predicted_tags = np.concatenate(all_predicted_tags)

    # For sentence classification only consider tags with values 10 and 11 since those
    # represent the "positive" and "negative" classes
    sentence_tag_mask = (all_true_tags == 10) | (all_true_tags == 11)
    sentence_level_tags = all_true_tags[sentence_tag_mask]
    accuracy_sentence_classification = (
        np.sum(
            all_predicted_tags[sentence_tag_mask] == all_true_tags[sentence_tag_mask]
        )
        / sentence_level_tags.shape[0]
    )

    print("Sentence classification accuracy: %f\n\n" % accuracy_sentence_classification)

    # For NER only consider those tags that correspond to the NER tags
    token_level_mask = (
        (all_true_tags > 0)
        & (all_true_tags < 10)
        & (all_predicted_tags > 0)
        & (all_predicted_tags < 10)
    )
    predicted_tag_ids = all_predicted_tags[token_level_mask]
    real_tag_ids = all_true_tags[token_level_mask]

    predicted_tags = [mapping[tag] for tag in predicted_tag_ids]
    real_tags = [mapping[tag] for tag in real_tag_ids]

    print("Tag level metrics:\n")
    evaluate(real_tags, predicted_tags)


calculate_metrics(test_dataset)
