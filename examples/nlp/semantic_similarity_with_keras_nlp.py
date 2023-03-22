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
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import tensorflow_datasets as tfds

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

# Let's take a look at how our training samples look like, randomly picking up 4 samples
sample = snli_test.batch(4).take(1).get_single_element()
sample

"""
### Preprocessing

In our dataset, correct labels for some of the samples have missing and labelled as -1,
we'll simply filter those out
"""


def filter_labels(sample):
    return sample["label"] >= 0


"""
Utility function to split the example into an `(x, y)` tuple suitable for `model.fit()`. 
By default, `keras_nlp.models.BertClassifier` will tokenize and pack together raw strings 
with a `"[SEP]"` token during training. So this label splitting is all the data preparation 
we need to do!
"""


def split_labels(sample):
    return (sample["hypothesis"], sample["premise"]), sample["label"]


train_ds = (
    snli_train.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
)
val_ds = (
    snli_train.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
)
test_ds = (
    snli_train.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
)

"""
KerasNLP models automatically tokenizes the inputs as per the model used, but user can use custom preprocessing
as per need.
"""

"""
## BERT Baseline

We'll use BERT model from KerasNLP to establish a baseline. 

KerasNLP models take care of tokenization by default. If we pass a tuple as input,
it'll tokenize all strings and concatenates them with a `"[SEP]"` seperator.

`keras_nlp.models.BertClassifier` class attaches classification head to the BERT Backbone, 
mapping  the backbone outputs to logit output suitable for a classification task. This 
significantly reduces need of custom code.

Here we'll use this model with pre-trained weights. `from_preset()` method allows you
to use your own preprocessor. Here we'll set the `num_classes` as 3 for SNLI dataset
"""

bert_classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

"""
Take a note that BERT tiny has 4,386,307 trainable parameters. KerasNLP task models come with
compilation defaults. Let's train the model we just instantiated, by calling the `fit()` method with 
"""

bert_classifier.fit(train_ds, validation_data=val_ds, epochs=1)

"""

Our BERT classifier gave us ~65% accuracy on validation split, let's see how it performs
on Test split

### Evaluate the performance of the trained model on test data.
"""
bert_classifier.evaluate(test_ds)

"""
Our baseline bert gave almost similar (~68%) accuracy on the test split. Let's see if we can
improve it.

Let's recompile our model with a different learning rate and see performance
"""
bert_classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

bert_classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)

bert_classifier.fit(train_ds, validation_data=val_ds, epochs=1)

bert_classifier.evaluate(test_ds)
"""
This time we got ~72% validation accuracy on val and test split. Not bad for 1 epoch ! Let's save our model
for now and learn how to perform inference with it. We took batch size of 512 to utilize our GPUs fully.

Let's see if we can improve it further. Let's use a learning rate scheduler this time.
"""


class TriangularSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear ramp up for `warmup` steps, then linear decay to zero at `total` steps."""

    def __init__(self, rate, warmup, total):
        self.rate = tf.cast(rate, dtype="float32")
        self.warmup = tf.cast(warmup, dtype="float32")
        self.total = tf.cast(total, dtype="float32")

    def __call__(self, step):
        step = tf.cast(step, dtype="float32")
        multiplier = tf.cond(
            step < self.warmup,
            lambda: step / self.warmup,
            lambda: (self.total - step) / (self.total - self.warmup),
        )
        return tf.maximum(self.rate * multiplier, 0.0)


bert_classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", num_classes=3
)

# Get the total count of training batches.
# This requires walking the dataset to filter all -1 labels.
total_steps = sum(1 for _ in train_ds.as_numpy_iterator())
warmup_steps = int(total_steps * 0.2)

bert_classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.experimental.AdamW(
        TriangularSchedule(0.001, warmup_steps, total_steps)
    ),
    metrics=["accuracy"],
)

bert_classifier.fit(train_ds, validation_data=val_ds, epochs=3)

"""
With LR scheduler and `AdamW` optimizer we see that our validation accuracy hikes upto ~79%

Let's evaluate our model on test set !
"""

bert_classifier.evaluate(test_ds)

"""
Our Tiny BERT achieved around ~79% of accuracy on test set with learning rate scheduler

# Save and Reload the model
"""
bert_classifier.save("bert_classifier")
restored_model = keras.models.load_model("bert_classifier")
restored_model.evaluate(test_ds)

"""
# Inference

Let's see how to perform inference with KerasNLP models
"""

# Convert to Hypothesis-Premise pair, for forward pass through model
sample = (sample["hypothesis"], sample["premise"])
sample

"""
Again, we don't need to tokenize the inputs explicitly as we are using the default preprocessor
here
"""
predictions = bert_classifier.predict(sample)


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)


# Get the class predictions with maximum probabilities
predictions = softmax(predictions)

"""
# Enhancing accuracy with RoBERTa

Now that we have established a baseline, we'll attempt to get better results by trying different
models. KerasNLP makes experimentation easy for us, with just few lines of code, we can fine-tune 
a roberta checkpoint on the same dataset.
"""

# Inittializing a RoBERTa from preset
roberta_classifier = keras_nlp.models.RobertaClassifier.from_preset(
    "roberta_base_en", num_classes=3
)

roberta_classifier.fit(train_ds, validation_data=val_ds, epochs=1)

roberta_classifier.evaluate(test_ds)

"""
`robeta_base_en` is bigger model with 124,645,635 trainable parameters (almost 30x of bert tiny 
parameters), it took almost 1.5 hrs to train on Kaggle P100 GPU. 

We achieved a significant performance improvement with roberta. Our accuracy hiked to 88% on
validation and test split. 16 was the biggest batch size that we could fit on Kaggle P100 GPU
with RoBERTa as our model.

The steps to perform inference with the RoBERTa model remain same as with BERT!
"""

predictions = roberta_classifier.predict(sample)
print(tf.math.argmax(predictions, axis=1).numpy())

"""
KerasNLP is a toolbox of modular building blocks ranging from pretrained state-of-the-art 
models, to low-level Transformer Encoder layers. We have shown one approach to use a pretrained
BERT here to establish a baseline and improved our performance by training a bigger roberta model,
in just ~5 lines of code. KerasNLP supports an ever growing array of components for preprocessing 
text and building models. We hope it makes it easier to experiment on solutions to your natural 
language problems.
"""
