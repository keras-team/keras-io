"""
Title: Semantic Similarity with KerasNLP
Author: [Anshuman Mishra](https://github.com/shivance/)
Date created: 2023/02/25
Last modified: 2023/02/25
Description: Use pretrained models from KerasNLP for Semantic Similarity Task
Accelerator: GPU
"""

"""
# Introduction

Semantic Similarity is the task of determining how similar two sentences are, in terms of 
what they mean. We already saw in [this](https://keras.io/examples/nlp/semantic_similarity_with_bert/) 
example how to use SNLI (Stanford Natural Language Inference) Corpus to predict sentence 
semantic similarity with HuggingFace Transformers library. In this tutorial we will 
learn how to use [KerasNLP](https://keras.io/keras_nlp/), an extension of the core Keras API, 
for the same task. We'll also learn how KerasNLP reduces the boilerplate code and makes models easy to use.

This guide is broken into three parts:

1. *Setup*, task definition, and establishing a baseline.
2. *Training* a BERT model.
3. *Saving and Reloading* the model.
4. *Performing inference* with the model.

# Setup

To begin, we can import `keras_nlp`, `keras` and `tensorflow`.

"""

"""shell
pip install -q keras-nlp
"""


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import keras_nlp

"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""

"""
Load the SNLI dataset train, validation and test data
There are more than 550k samples in total. To keep this example running quickly, 
we will use 100k samples for this example.

## Overview of SNLI Dataset:

Every sample in the dataset contains three components of form (sentence1, sentence2, 
similarity label).

sentence1 is the initial caption provided to the author of the pair, sentence2 is 
the hypothesis caption created by the author of the pair, and similarity is the 
label assigned by annotators to denote the similarity between the two sentences.

The dataset includes three possible similarity label values: Contradiction, 
Entailment, and Neutral, which respectively represent completely dissimilar sentences,
similar meaning sentences, and neutral sentences where no clear similarity or 
dissimilarity can be established between them.

"""

train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")

# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

# Define labels for classification task
labels = ["contradiction", "entailment", "neutral"]

# Let's look at one sample from the dataset:
print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
print(f"Similarity: {train_df.loc[1, 'similarity']}")


# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)

"""
### Preprocessing
"""

print("Train Target Distribution")
print(train_df.similarity.value_counts())

print("Validation Target Distribution")
print(valid_df.similarity.value_counts())

# Filter and shuffle train and validation data
train_df = (
    train_df[train_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)
valid_df = (
    valid_df[valid_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

# Convert label column to numerical values for train, validation and test data
train_df["label"] = train_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_train = train_df.label

valid_df["label"] = valid_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_val = valid_df.label

test_df["label"] = test_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_test = test_df.label

"""
Initialize a BertPreprocessor object from KerasNLP library with preset 
configuration "bert_tiny_en_uncased". The preprocessor can preprocess 
sentence pairs for BERT-based models by converting tokens to corresponding IDs in 
BERT vocabulary and adding special tokens.
"""
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")

"""
Preprocessing the sentence pairs in the training, validation, and test data using 
a preprocessor object. BertPreprocessor automatically takes care of sentence packing,
and seperates them with [SEP] token
"""
x_train = preprocessor((train_df["sentence1"], train_df["sentence2"]))
x_val = preprocessor((valid_df["sentence1"], valid_df["sentence2"]))
x_test = preprocessor((test_df["sentence1"], test_df["sentence2"]))

"""
### Train the Model End to End
 
We'll use BertClassifier from KerasNLP. This model attaches a classification head to 
a keras_nlp.model.BertBackbone backbone, mapping from the backbone outputs to logit 
output suitable for a classification task. This makes the job a lot easier !

Here we'll use this model with pre-trained weights. `from_preset()` method allows you
to use your own preprocessor. Here we'll set the `num_classes` as 3 for SNLI dataset
"""

model = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=["accuracy"],
)

model.summary()

"""
Training the model we just compiled using the fit() method with the following arguments:

    x_train: The input data for the training set, preprocessed using BertPreprocessor
    y_train: The target labels for the training set
    validation_data: A tuple containing the input and target data for the validation set
    epochs: The number of epochs to train the model
    batch_size: The number of samples per gradient update
"""
model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=4, batch_size=64)

"""
### Evaluate the performance of the trained model on test data.
"""
model.evaluate(x=x_test, y=y_test)


"""
# Save and Reload the model
"""
model.save("bert_classifier.pb", compile=False)
restored_model = keras.models.load_model("bert_classifier.pb")
restored_model.evaluate(x=x_test, y=y_test)

"""
# Inference

Randomly sample 4 sentence pairs from the test set
"""
infer_df = test_df.sample(n=4)
infer_df.head()

# Making predictions
preprocessed = preprocessor((infer_df["sentence1"], infer_df["sentence2"]))
predictions = restored_model.predict(preprocessed)
print(tf.math.argmax(predictions, axis=1).numpy())

"""
KerasNLP is a toolbox of modular building blocks ranging from pretrained state-of-the-art 
models, to low-level Transformer Encoder layers. We have shown one approach to use a pretrained
BERT here, but KerasNLP supports an ever growing array of components for preprocessing text and 
building models. We hope it makes it easier to experiment on solutions to your natural language 
problems.
"""
