"""
Semantic Similarity with KerasNLP
Author: [Anshuman Mishra](https://github.com/shivance/)
Date created: 2022/04/18
Last modified: 2022/04/18
Description: Use KerasNLP to train a Transformer model from scratch.
Accelerator: GPU
"""



"""
Introduction

Semantic Similarity is the task of determining how similar two sentences are, in terms of what they mean. This example demonstrates the use of SNLI (Stanford Natural Language Inference) Corpus to predict sentence semantic similarity with Transformers. We will fine-tune a BERT model that takes two sentences as inputs and that outputs a similarity score for these two sentences. This notebook reproduces [this](https://keras.io/examples/nlp/semantic_similarity_with_bert/) notebook but with KerasNLP package.

References
1. [BERT](https://arxiv.org/pdf/1810.04805.pdf)
2. [SNLI](https://nlp.stanford.edu/projects/snli/)

# Setup

To begin, we can import `keras_nlp`, `keras` and `tensorflow`.

A simple thing we can do right off the bat is to enable
[mixed precision](https://keras.io/api/mixed_precision/), which will speed up training by
running most of our computations with 16 bit (instead of 32 bit) floating point numbers.
Training a Transformer can take a while, so it is important to pull out all the stops for
faster training!
"""

"""shell
pip install -q --upgrade keras-nlp tensorflow
"""


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import keras_nlp

# Import classes corresponding to BERT model from KerasNLP
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_classifier import BertClassifier
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer


"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""

labels = ["contradiction", "entailment", "neutral"]


# In[8]:


# There are more than 550k samples in total; we will use 100k for this example.
train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")

# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

"""
Dataset Overview:

   * sentence1: The premise caption that was supplied to the author of the pair.
   * sentence2: The hypothesis caption that was written by the author of the pair.
   * similarity: This is the label chosen by the majority of annotators. Where no majority exists, the label "-" is used (we will skip such samples here).

Here are the "similarity" label values in our dataset:

   * Contradiction: The sentences share no similarity.
   * Entailment: The sentences have similar meaning.
   * Neutral: The sentences are neutral.

Let's look at one sample from the dataset:

"""

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

preprocessor = BertPreprocessor.from_preset("bert_tiny_en_uncased")

x_train = preprocessor((tf.constant(train_df["sentence1"]), tf.constant(train_df["sentence2"])))
x_val   = preprocessor((tf.constant(valid_df["sentence1"]), tf.constant(valid_df["sentence2"])))
x_test = preprocessor((tf.constant(test_df["sentence1"]), tf.constant(test_df["sentence2"])))

"""
### Train the Model End to End
 
KerasNLP offers pretrained models, we'll use the same here
"""

model = BertClassifier.from_preset(
    "bert_tiny_en_uncased", 
    num_classes=3, 
    preprocessor=None)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-5),
              metrics=["accuracy"])

model.summary()


history = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs=4, batch_size=64)


# In just four epochs we achieved 71% of accuracy.

"""
### Evaluate model on the test set
"""

model.evaluate(x = x_test, y = y_test)
