"""
Title: FILLME
Author: FILLME
Date created: FILLME
Last modified: FILLME
Description: FILLME
"""
"""shell
!pip3 install -q tensorflow_decision_forests
"""

"""shell
! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
"""

"""shell
!kaggle competitions download -c nlp-getting-started
"""

import zipfile

with zipfile.ZipFile("/content/nlp-getting-started.zip", "r") as zip_ref:
    zip_ref.extractall("/content/")

import pandas as pd
import numpy as np
import re

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.regularizers import l2, l1_l2
import tensorflow_hub as hub

# Turn .csv files into pandas DataFrame's
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()

train_df.shape

train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
train_df_shuffled.head()

train_df_shuffled.info()

train_df_shuffled.target.value_counts()

# How many samples total?
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")

import random

for i in range(5):
    random_index = random.randint(0, len(train_df))
    target = train_df_shuffled.iloc[random_index][1]
    text = train_df_shuffled.iloc[random_index][0]

    if target == 1:
        print(f"Target: {target} (real disaster)")
    else:
        print(f"Target: {target} (not real disaster)")
    print(f"Text: {text}\n")
    print("-" * 10)

from sklearn.model_selection import train_test_split

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1,  # dedicate 10% of samples to validation set
    random_state=42,
)  # random state for reproducibility

train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels))

train_dataset

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False,
    name="USE",
)

inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
x = sentence_encoder_layer(inputs)
preprocessor = tf.keras.Model(inputs=inputs, outputs=x)

model = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor, num_trees=1000)
model.compile(metrics=["accuracy"])

model.fit(x=train_dataset)

print("Evaluation:", model.evaluate(valid_dataset))

print("Evaluation :", model.evaluate(valid_dataset))


def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1,
    }
    return model_results


preds = model.predict(val_sentences)
preds = tf.squeeze(tf.round(preds))
preds.shape, preds[:10]
result = calculate_results(y_true=val_labels, y_pred=preds)
result
