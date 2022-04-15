"""
Title: Text Classification using TFDF and Pre-trained embeddings
Author: Gitesh Chawda
Date created: 15/04/2022
Last modified: 15/04/2022
Description: Using Tensorflow Decision Forest for text classification
"""
"""
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF)
provide powerful models, especially with structured data,
It is a collection of state-of-the-art algorithms of Decision Forest models that are
compatible with Keras APIs. The module includes Random Forests, Gradient Boosted Trees,
and CART, and can be used for regression, classification, and ranking tasks.

Alternatively for getting started
In this example we will use Gradient Boosted Trees with pre-trained embeddings to
classify disaster tweets.
"""

"""
Install Tensorflow Decision Forest using following command : 
`!pip3 install -q tensorflow_decision_forests`
"""

"""
## Imports
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt

"""
## Get the data

Dataset is avaliable at [kaggle](https://www.kaggle.com/c/nlp-getting-started)

dataset description :

1. Files

    a. train.csv - the training set

2. Columns

    a. id - a unique identifier for each tweet

    b. text - the text of the tweet

    c. location - the location the tweet was sent from (may be blank)

    d. keyword - a particular keyword from the tweet (may be blank)

e. target - in train.csv only, this denotes whether a tweet is about a real disaster (1)
or not (0)
"""

# Turn .csv files into pandas DataFrame's
train_df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
print(train_df.head())

# Printing shape of the training dataset
print(train_df.shape)

# Shuffling training data
train_df_shuffled = train_df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
train_df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
train_df_shuffled.reset_index(inplace=True, drop=True)
print(train_df_shuffled.head())

print(train_df_shuffled.info())

# Printing total number of Disaster and non-Disaster tweets
print(train_df_shuffled.target.value_counts())

# Viewing 5 records from training data
for ind, counter in enumerate(train_df_shuffled.index):
    print(f"Target : {train_df_shuffled['target'][ind]}")
    print(f"Text : {train_df_shuffled['text'][ind]}")
    if counter == 5:
        break

# Splitting dataset into train and test
test_df = train_df_shuffled.sample(frac=0.2, random_state=42)
train_df = train_df_shuffled.drop(test_df.index)

print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")

"""
## Convert data to tf.Dataset
"""


def create_dataset(dataframe, training=False):
    df = dataframe.copy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["text"].to_numpy(), df["target"].to_numpy())
    )

    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_df, training=True)
test_ds = create_dataset(test_df)

"""
## Downloading Pre-trained Embeddings

The Universal Sentence Encoder encodes text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks.It is trained on a variety of data sources and a variety of tasks. The input is
variable length English text and the output is a 512 dimensional vector.

To understand better about pre-trained embeddings : [Universal Sentence
Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)


"""

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False,
    name="USE",
)

"""
## Build a model

First, we develop a preprocessor model that takes the inputs and passes it through
sentence encoder layer to generate embedding, which produces a 512 dimensional vector for
each sentence.

The Preprocessor is then passed to the GradientBoostedTreesModel which will be used for
predicting disaster tweets.


"""

inputs = layers.Input(shape=(), dtype=tf.string)
# Passing sentences to sentence_encoder_layer
x = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=x)

model = tfdf.keras.GradientBoostedTreesModel(
    preprocessing=preprocessor,
    num_trees=30,
    max_vocab_count=1000,
    max_depth=100,
    min_examples=10,
)

"""
## Train the model
"""

# Compiling model
model.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Training the model
model.fit(x=train_ds, steps_per_epoch=200)

# Prints training logs
model.make_inspector().training_logs()

"""
The `model.summary()` function returns a variety of information about your decision trees
model, including model type, task, input features, and feature importance.

In our the inputs to the GradientBoostedTreesModel are 512 dimensional vectors so, it
prints all information of those vectors.
"""

print(model.summary())

logs = model.make_inspector().training_logs()
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Loss")

plt.show()

"""
## Evaluating on test data
"""

results = model.evaluate(test_ds, return_dict=True, verbose=0)

for name, value in results.items():
    print(f"{name}: {value:.4f}")
