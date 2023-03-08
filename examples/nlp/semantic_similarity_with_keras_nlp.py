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

This guide is broken into following parts:

1. *Setup*, task definition, and establishing a baseline.
2. *Establishing baseline* with BERT.
3. *Saving and Reloading* the model.
4. *Performing inference* with the model.
5  *Enhancing Performance* with RoBERTa

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
import tensorflow_datasets as tfds

"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""

"""
We'll Load the SNLI dataset using tensorflow-datasets library.
There are more than 550k samples in total. To keep this example running quickly, 
we will use 20% of trainining samples for this example.

## Overview of SNLI Dataset:

Every sample in the dataset contains three components of form (hypothesis, premise, 
label).

`premise` is the initial caption provided to the author of the pair, `hypothesis` is 
the hypothesis caption created by the author of the pair, and `label` is assigned by 
annotators to denote the similarity between the two sentences.

The dataset includes three possible similarity label values: Contradiction, 
Entailment, and Neutral, which respectively represent completely dissimilar sentences,
similar meaning sentences, and neutral sentences where no clear similarity or 
dissimilarity can be established between them.

"""

snli_train = tfds.load("snli", split="train[:20%]")
snli_val = tfds.load("snli", split="validation")
snli_test = tfds.load("snli", split="test")

# Define labels for classification task
labels = ["neutral", "entailment", "contradiction"]

"""
### Preprocessing

Tensorflow datasets come as prefetch datasets, with strings converted into bytes. To
make the tutorial easy to grasp, we convert our dataset to dataframe.
"""


def to_dataframe(dataset):
    dataset = tfds.as_dataframe(dataset)
    # decoding bytes to strings
    dataset["hypothesis"] = dataset["hypothesis"].str.decode("utf-8")
    dataset["premise"] = dataset["premise"].str.decode("utf-8")
    return dataset


"""
Let's take a look at how our dataset looks like
"""
snli_val_df = to_dataframe(snli_val)
snli_val_df.head()

"""
This utility function tokenizes our string inputs according to model's requirements.
We return tuples of data and labels.
"""


def preprocess_dataset(dataset, preprocessor):
    dataset = to_dataframe(dataset)
    # some sample have -1 as label entries. We'll simply drop those here
    dataset.drop(dataset[dataset["label"] == -1].index, inplace=True)

    x = preprocessor((dataset["hypothesis"], dataset["premise"]))
    y = np.array(dataset.label)
    return x, y


"""
## BERT Baseline

We'll use BERT from KerasNLP to establish a baseline.

Initialize a BertPreprocessor object from KerasNLP library with preset 
configuration "bert_tiny_en_uncased". The preprocessor can preprocess 
sentence pairs for BERT-based models by converting tokens to corresponding IDs in 
BERT vocabulary and adding special tokens.
"""
bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased"
)

"""
Preprocessing the sentence pairs in the training, validation, and test data using 
a preprocessor object. BertPreprocessor automatically takes care of sentence packing,
and seperates them with [SEP] token
"""
x_train, y_train = preprocess_dataset(snli_train, bert_preprocessor)
x_test, y_test = preprocess_dataset(snli_test, bert_preprocessor)
x_val, y_val = preprocess_dataset(snli_val, bert_preprocessor)

"""
### Train the Model End to End
 
BertClassifier class attaches classification head to the BertBackbone backbone, mapping 
the backbone outputs to logit output suitable for a classification task. This significantly
reduces need of custom code.

Here we'll use this model with pre-trained weights. `from_preset()` method allows you
to use your own preprocessor. Here we'll set the `num_classes` as 3 for SNLI dataset
"""

bert_clf = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

bert_clf.summary()

"""
KerasNLP models compile automatically. Let's train the model we just instantiated, using 
the fit() method with the following arguments:

    x_train: The input data for the training set, preprocessed using BertPreprocessor
    y_train: The target labels for the training set
    validation_data: A tuple containing the input and target data for the validation set
    epochs: The number of epochs to train the model
    batch_size: The number of samples per gradient update
"""
bert_clf.fit(
    x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1, batch_size=512
)

"""

Our BERT classifier gave us ~68% accuracy on validation split, let's see how it performs
on Test split

### Evaluate the performance of the trained model on test data.
"""
bert_clf.evaluate(x=x_test, y=y_test)

"""
Our baseline bert gave almost similar (~68%) accuracy on the test split. Let's see if we can
improve it.

Let's recompile our model with a different learning rate and see performance
"""
bert_clf_1 = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

bert_clf_1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)

bert_clf_1.fit(
    x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1, batch_size=512
)

"""
This time we got 72% accuracy on val and test split. Not bad for 1 epoch ! Let's save our model
for now and learn how to perform inference with it. We took batch size of 512 to utilize our GPUs fully.

# Save and Reload the model
"""
bert_clf.save("bert_classifier.pb", compile=False)
restored_model = keras.models.load_model("bert_classifier.pb")
restored_model.evaluate(x=x_test, y=y_test)

"""
# Inference

Randomly sample 4 sentence pairs from the test set
"""
infer_df = to_dataframe(snli_test).sample(n=4)
infer_df.head()

# Making predictions
preprocessed = bert_preprocessor((infer_df["hypothesis"], infer_df["premise"]))
predictions = bert_clf.predict(preprocessed)
print(tf.math.argmax(predictions, axis=1).numpy())

"""
# Enhancing accuracy with RoBERTA

Now that we have established a baseline, we'll attempt to get better results by trying different
models. KerasNLP makes experimentation easy for us, with just few lines of code, we can train 
a roberta on the same dataset.
"""

# Roberta has it's own data preprocessing methods
roberta_preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(
    "roberta_base_en"
)

# we preprocess original snli dataset with roberta preprocessor
x_train, y_train = preprocess_dataset(snli_train, roberta_preprocessor)
x_test, y_test = preprocess_dataset(snli_test, roberta_preprocessor)
x_val, y_val = preprocess_dataset(snli_val, roberta_preprocessor)

# inittializing a roberta from preset
roberta_clf = keras_nlp.models.RobertaClassifier.from_preset(
    "roberta_base_en", num_classes=3, preprocessor=None
)

roberta_clf.summary()

roberta_clf.fit(
    x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1, batch_size=16
)

roberta_clf.evaluate(x=x_test, y=y_test)

"""
"robeta_base_en" is slightly bigger model than bert_tiny, it took almost 1.5 hrs to train on
Kaggle P100 GPU. 

We achieved a significant performance improvement with roberta. Our accuracy hiked to 88% on
validation and test split. 16 was the biggest batch size that we could fit on Kaggle P100 GPU
with roberta as our model.

The steps to perform inference with roberta remain same as our bert !
"""

preprocessed = roberta_preprocessor((infer_df["hypothesis"], infer_df["premise"]))
predictions = roberta_clf.predict(preprocessed)
print(tf.math.argmax(predictions, axis=1).numpy())

"""
KerasNLP is a toolbox of modular building blocks ranging from pretrained state-of-the-art 
models, to low-level Transformer Encoder layers. We have shown one approach to use a pretrained
BERT here to establish a baseline and improved our performance by training a bigger roberta model,
in just ~5 lines of code. KerasNLP supports an ever growing array of components for preprocessing 
text and building models. We hope it makes it easier to experiment on solutions to your natural 
language problems.
"""
