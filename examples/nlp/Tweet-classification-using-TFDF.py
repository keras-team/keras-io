"""
Title: Text Classification using TFDF and Pre-trained embeddings
Author: Gitesh Chawda
Date created: 30/04/2022
Last modified: 30/04/2022
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
[beginner Colab](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) 
also text classification using pre-trained embeddings
[Notebook](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).

In this example we will use Gradient Boosted Trees with pre-trained embeddings to
classify disaster tweets.
"""

"""
Install Tensorflow Decision Forest using following command : 
`!pip install tensorflow_decision_forests`
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
    - train.csv - the training set

2. Columns
    - id - a unique identifier for each tweet
    - text - the text of the tweet
    - location - the location the tweet was sent from (may be blank)
    - keyword - a particular keyword from the tweet (may be blank)
- target - in train.csv only, this denotes whether a tweet is about a real disaster (1)
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
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)

print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")

print(train_df["target"].value_counts())
print(test_df["target"].value_counts())

"""
## Convert data to tf.Dataset
"""


def create_dataset(dataframe):
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

We will create 2 models In first model raw text will be directly passed to the Gradient
Boosted Trees algorithm and In second model raw text will be first processed by
pre-trained embeddings and then passed to Gradient Boosted Trees algorithm for
classification.


"""

inputs = layers.Input(shape=(), dtype=tf.string)
# Passing sentences to sentence_encoder_layer
outputs = sentence_encoder_layer(inputs)

preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

model_2 = tfdf.keras.GradientBoostedTreesModel()

"""
## Train the models

We will compile our model by passing metrics as `Accuracy`, `Recall`, `Precision` and
`AUC`, and for loss TF-DF, automatically detects it from the task (Classification or
regression) which is printed in the model summary.

Also, TF-DF models do not need a validation dataset to monitor overfitting, or to stop
training early. some algorithms do not use a validation dataset (e.g. Random Forest)
while some others do (e.g. Gradient Boosted Trees).Therefore, if a validation dataset is
needed, it will be extracted automatically from the training dataset.
"""

# Compiling model
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_1.fit(train_ds)

# Compiling model
model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_2.fit(train_ds)

# Prints training logs of model_1 and model_2
logs_1 = model_1.make_inspector().training_logs()
logs_2 = model_2.make_inspector().training_logs()

print(logs_1)
print(logs_2)

"""
The `model.summary()` function returns a variety of information about your decision trees
model, including model type, task, input features, and feature importance.

In our the inputs to the GradientBoostedTreesModel are 512 dimensional vectors so, it
prints all information of those vectors.
"""

print(model_1.summary())

print(model_2.summary())

logs = model_1.make_inspector().training_logs()
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

logs = model_2.make_inspector().training_logs()
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

results = model_1.evaluate(test_ds, return_dict=True, verbose=0)

for name, value in results.items():
    print(f"{name}: {value:.4f}")

results = model_2.evaluate(test_ds, return_dict=True, verbose=0)

for name, value in results.items():
    print(f"{name}: {value:.4f}")

"""
# Predicting on validation data
"""

counter = 0
test_df = test_df.sample(frac=0.1)
for index, row in test_df.iterrows():
    text = tf.expand_dims(row["text"], axis=0)
    preds = model_1.predict_step(text)
    preds = tf.squeeze(tf.round(preds))
    print(f"Text: {row['text']}")
    print(f"Prediction: {preds}")
    print(f"Ground Truth : {row['target']}")
    if counter == 5:
        break
    else:
        counter += 1

"""
## Concluding remarks

TensorFlow Decision Forests provide powerful models, especially with structured data. In
our experiments, the Gradient Boosted Tree model with pre-trained embedding achieved 94%
test accuracy while simple Gradient Boosted Tree model had 57.31% accuracy.

In this example we learned how we can process text through pre-trained embeddings and
then pass these learned embeddings to Gradient Boosted Tree algorithm.
"""
