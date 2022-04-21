"""
Title: Text Classification using TFDF and Pre-trained embeddings
Author: Gitesh Chawda
Date created: 22/04/2022
Last modified: 22/04/2022
Description: Using Tensorflow Decision Forest for text classification
"""

"""
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF)
provide powerful models, especially with structured data,
It is a collection of state-of-the-art algorithms of Decision Forest models that are
compatible with Keras APIs. The module includes Random Forests, Gradient Boosted Trees,
and CART, and can be used for regression, classification, and ranking tasks.

Alternatively for getting started you go through official tutorial 
[Beginner Colab](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) 
also text classification using pre-trained embeddings
[Notebook](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).

In this example we will use Gradient Boosted Trees with pre-trained embeddings to
classify disaster tweets.
"""

"""
Install Tensorflow Decision Forest using following command : 
`!pip3 install -U tensorflow_decision_forests`
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
df = pd.read_csv(
"https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
print(df.head())

# Printing shape of the training dataset
print(f"Training dataset shape: {df.shape}")

# Shuffling training data
df_shuffled = df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())

print(df_shuffled.info())

# Printing total number of Disaster and non-Disaster tweets
print(
f"Total Number of disaster and non-disaster tweets\n{df_shuffled.target.value_counts()}"
)

# Viewing 5 records from training data
for ind, counter in enumerate(df_shuffled.index):
    print(f"Target : {df_shuffled['target'][ind]}")
    print(f"Text : {df_shuffled['text'][ind]}")
    if counter == 5:
        break

# Splitting dataset into train and test
test_df = df_shuffled.sample(frac=0.2, random_state=42)
train_df = df_shuffled.drop(test_df.index)

print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")

"""
## Convert data to tf.Dataset
"""


def create_dataset(dataframe):
    df = dataframe.copy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["text"].to_numpy(), df["target"].to_numpy())
    )
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

"""
## Downloading Pre-trained Embeddings

The Universal Sentence Encoder encodes text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks.It is trained on a variety of data sources and a variety of tasks. The input is
variable length English text and the output is a 512 dimensional vector.

To understand better about pre-trained embeddings : 
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)


"""

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
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
outputs = sentence_encoder_layer(inputs)

preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

"""
## Train the model
"""

# Compiling model
model.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model.fit(x=train_ds)

# Prints training logs
print(model.make_inspector().training_logs())

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
