"""
Title: Text classification using Decision Forests and pretrained embeddings
Author: Gitesh Chawda
Date created: 09/05/2022
Last modified: 09/05/2022
Description: Using Tensorflow Decision Forests for text classification.
"""

"""
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF)
is a collection of state-of-the-art algorithms for Decision Forest models that are
compatible with Keras APIs. The module includes Random Forests, Gradient Boosted Trees,
and CART, and can be used for regression, classification, and ranking tasks.

In this example we will use Gradient Boosted Trees with pretrained embeddings to
classify disaster-related tweets.

### See also:

- [TF-DF beginner tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab)
- [TF-DF intermediate tutorial](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).
"""

"""
Install Tensorflow Decision Forest using following command :
`pip install tensorflow_decision_forests`
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

The Dataset is avalaible on [Kaggle](https://www.kaggle.com/c/nlp-getting-started)

Dataset description:

**Files:**

- train.csv: the training set

**Columns:**

- id: a unique identifier for each tweet
- text: the text of the tweet
- location: the location the tweet was sent from (may be blank)
- keyword: a particular keyword from the tweet (may be blank)
- target: in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
"""

# Turn .csv files into pandas DataFrame's
df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
print(df.head())

"""
The dataset includes 7613 samples with 5 columns:
"""

print(f"Training dataset shape: {df.shape}")

"""
Shuffling and dropping unnecessary columns:
"""

df_shuffled = df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())

"""
Printing information about the shuffled dataframe:
"""

print(df_shuffled.info())

"""
Total number of "disaster" and "non-disaster" tweets:
"""

print(
    "Total Number of disaster and non-disaster tweets: "
    f"{df_shuffled.target.value_counts()}"
)

"""
Let's preview a few samples:
"""

for index, example in df_shuffled[:5].iterrows():
    print(f"Example #{index}")
    print(f"\tTarget : {example['target']}")
    print(f"\tText : {example['text']}")

"""
Splitting dataset into training and test sets:
"""
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)
print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")

"""
Total number of "disaster" and "non-disaster" tweets in the training data:
"""
print(train_df["target"].value_counts())

"""
Total number of "disaster" and "non-disaster" tweets in the test data:
"""

print(test_df["target"].value_counts())

"""
## Convert data to a `tf.data.Dataset`
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
## Downloading pretrained embeddings

The Universal Sentence Encoder embeddings encode text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks. They're trained on a variety of data sources and a variety of tasks. Their input is
variable-length English text and their output is a 512 dimensional vector.

To learn more about these pretrained embeddings, see
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4).

"""

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)

"""
## Creating our models

We create two models. In the first model (model_1) raw text will be first encoded via
pretrained embeddings and then passed to a Gradient Boosted Tree model for
classification. In the second model (model_2) raw text will be directly passed to
the Gradient Boosted Trees model.
"""

"""
Building model_1
"""

inputs = layers.Input(shape=(), dtype=tf.string)
outputs = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

"""
Building model_2
"""

model_2 = tfdf.keras.GradientBoostedTreesModel()

"""
## Train the models

We compile our model by passing the metrics `Accuracy`, `Recall`, `Precision` and
`AUC`. When it comes to the loss, TF-DF automatically detects the best loss for the task
(Classification or regression). It is printed in the model summary.

Also, because they're batch-training models rather than mini-batch gradient descent models,
TF-DF models do not need a validation dataset to monitor overfitting, or to stop
training early. Some algorithms do not use a validation dataset (e.g. Random Forest)
while some others do (e.g. Gradient Boosted Trees). If a validation dataset is
needed, it will be extracted automatically from the training dataset.
"""

# Compiling model_1
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_1.fit(train_ds)

# Compiling model_2
model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_2.fit(train_ds)


"""
## Plotting training metrics
"""


def plot_curve(logs):
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


plot_curve(logs_1)
plot_curve(logs_2)

"""
## Evaluating on test data
"""

results = model_1.evaluate(test_ds, return_dict=True, verbose=0)
print("model_1 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

results = model_2.evaluate(test_ds, return_dict=True, verbose=0)
print("model_2 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

"""
## Predicting on validation data
"""

test_df.reset_index(inplace=True, drop=True)
for index, row in test_df.iterrows():
    text = tf.expand_dims(row["text"], axis=0)
    preds = model_1.predict_step(text)
    preds = tf.squeeze(tf.round(preds))
    print(f"Text: {row['text']}")
    print(f"Prediction: {int(preds)}")
    print(f"Ground Truth : {row['target']}")
    if index == 10:
        break

"""
## Concluding remarks

The TensorFlow Decision Forests package provides powerful models
that work especially well with structured data. In our experiments,
the Gradient Boosted Tree model with pretrained embeddings achieved 94%
test accuracy while the plain Gradient Boosted Tree model had 57.31% accuracy.
"""
