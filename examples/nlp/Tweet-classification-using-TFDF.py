"""
Title: FILLME
Author: FILLME
Date created: FILLME
Last modified: FILLME
Description: FILLME
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision, Accuracy, AUC

# Turn .csv files into pandas DataFrame's
train_df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
test_df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/test.csv"
)
train_df.head()

train_df.shape

train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
train_df_shuffled.reset_index(inplace=True, drop=True)
train_df_shuffled.head()

train_df_shuffled.info()

train_df_shuffled.target.value_counts()

# How many samples total?
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")

for ind, counter in enumerate(train_df_shuffled.index):
    print(f"Target : {train_df_shuffled['target'][ind]}")
    print(f"Text : {train_df_shuffled['text'][ind]}")
    if counter == 5:
        break

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.2,  # dedicate 10% of samples to validation set
    random_state=42,
)  # random state for reproducibility

print(f"train_sentences shape: {train_sentences.shape}")
print(f"train_labels shape: {train_labels.shape}")
print(f"val_sentences shape: {val_sentences.shape}")
print(f"val_labels shape: {val_labels.shape}")


def create_dataset(sentences, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))
    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_sentences, train_labels, training=True)
val_ds = train_ds = create_dataset(val_sentences, val_labels)

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False,
    name="USE",
)

inputs = layers.Input(shape=(32,), dtype=tf.string)
x = sentence_encoder_layer(inputs)
preprocessor = tf.keras.Model(inputs=inputs, outputs=x)

model = tfdf.keras.GradientBoostedTreesModel(
    preprocessing=preprocessor,
    num_trees=1000,
    max_vocab_count=1000,
    max_depth=100,
    min_examples=10,
)
model.compile(metrics=["accuracy"])

model.fit(x=train_ds)

print("Evaluation :", model.evaluate(val_ds))


def calc_metrics(y_true, y_pred):
    recall, precision, accuracy, auc = Recall(), Precision(), Accuracy(), AUC()
    recall.update_state(y_true, y_pred), precision.update_state(y_true, y_pred)
    accuracy.update_state(y_true, y_pred), auc.update_state(y_true, y_pred)
    results = {
        "recall": recall.result().numpy(),
        "precision": precision.result().numpy(),
        "Accuracy": accuracy.result().numpy(),
        "AUC": auc.result().numpy(),
    }

    return results


preds = model.predict(val_sentences)
preds = tf.squeeze(tf.round(preds))
print(calc_metrics(val_labels, preds))

"""
## Test Set
"""

test_df.drop(["id", "keyword", "location"], axis=1, inplace=True)
test_df.reset_index(inplace=True, drop=True)
test_df.head()
test_df.head()

preds = model.predict(test_df)
preds = tf.squeeze(tf.round(preds))
print(preds[:5])
