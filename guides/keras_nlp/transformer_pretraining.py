"""
Title: Pretraining a Transformer from scratch with KerasNLP
Author: [Matthew Watson](https://github.com/mattdangerw/)
Date created: 2022/04/18
Last modified: 2022/04/18
Description: Use KerasNLP to train a Transformer model from scratch.
"""

"""
KerasNLP aims to make it easy to build state-of-the-art text processing models. In this
guide, we will show how library components simplify pretraining and fine-tuning a
Transformer model from scratch.

This guide is broken into three parts:

1. *Setup*, task definition, and establishing a baseline.
2. *Pretraining* a Transformer model.
3. *Fine-tuning* the Transformer model on our classification task.
"""

"""
## Setup

To begin, we can import `keras_nlp`, `keras` and `tensorflow`.

A simple thing we can do right off the bat is to enable
[mixed precision](https://keras.io/api/mixed_precision/), which will speed up training by
running most of our computations with 16 bit (instead of 32 bit) floating point numbers.
Training a Transformer can take a while, so it is important to pull out all the stops for
faster training!
"""

"""shell
pip install -q keras-nlp
"""

import os

import keras_nlp
import tensorflow as tf
from tensorflow import keras

policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)

"""
Next up, can download two datasets.

- [SST-2](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) a text
classification dataset and our "end goal". This dataset is often used to benchmark
language models.
- [WikiText-103](https://paperswithcode.com/dataset/wikitext-103): A medium sized
collection of featured articles from English wikipedia, which we will use for
pretraining.

Finally, we will download a WordPiece vocabulary, to do sub-word tokenization later on in
this guide.
"""

# Download pretraining data.
keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

# Download finetuning data.
keras.utils.get_file(
    origin="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip", extract=True,
)
sst_dir = os.path.expanduser("~/.keras/datasets/SST-2/")

# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

"""
Next, we define some hyperparameters we will use during training.
"""

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# Training params.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3

"""
### Load data

We load our data with [tf.data](https://www.tensorflow.org/guide/data), which will allow
us to define input pipelines for tokenizing and preprocessing text.
"""

# Load SST-2.
sst_train_ds = tf.data.experimental.CsvDataset(
    sst_dir + "train.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)

# Take a peak at the sst-2 dataset.
print(sst_train_ds.unbatch().batch(4).take(1).get_single_element())

"""
You can see that our `SST-2` dataset contains relatively short snippets of movie review
text. Our goal is to predict the sentiment of the snippet. A label of 1 indicates
positive sentiment, and a label of 0 negative sentiment.
"""

"""
### Establish a baseline

As a first step, we will establish a baseline of good performance. We don't actually need
KerasNLP for this, we can just use core Keras layers.

We will train a simple bag-of-words model, where we learn a positive or negative weight
for each word in our vocabulary. A sample's score is simply the sum of the weights of all
words that are present in the sample.
"""

# This layer will turn our input sentence into a list of 1s and 0s the same size
# our vocabulary, indicating whether a word is present in absent.
multi_hot_layer = keras.layers.TextVectorization(
    max_tokens=4000, output_mode="multi_hot"
)
multi_hot_layer.adapt(sst_train_ds.map(lambda x, y: x))
# We then learn a linear regression over that layer, and that's our entire
# baseline model!
regression_layer = keras.layers.Dense(1, activation="sigmoid")

inputs = keras.Input(shape=(), dtype="string")
outputs = regression_layer(multi_hot_layer(inputs))
baseline_model = keras.Model(inputs, outputs)
baseline_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
baseline_model.fit(sst_train_ds, validation_data=sst_val_ds, epochs=5)

"""
A bag-of-words approach can be a fast and suprisingly powerful, especially when input
examples contain a large number of words. With shorter sequences, it can hit a
performance ceiling.

To do better, we would like to build a model that can evaluate words *in context*. Instead
of evaluating each word in a void, we need to use the information contained in the
*entire ordered sequence* of our input.

This runs us into a problem. `SST-2` is very small dataset, and there's simply not enough
example text to attempt to build a larger, more parameterized model that can learn on a
sequence. We would quickly start to overfit and memorize our training set, without any
increase in our ability to generalize to unseen examples.

Enter **pretraining**, which will allow us to learn on a larger corpus, and transfer our
knowledge to the `SST-2` task. And enter **KerasNLP**, which will allow us to pretrain a
particularly powerful model, the Transformer, with ease.
"""

"""
## Pretraining

To beat our baseline, we will leverage the `WikiText103` dataset, an unlabeled
collection of wikipedia articles that is much bigger than than `SST-2`.

We are going to train a *transformer*, a highly expressive model which will learn
to embed each word in our input as a low dimentional vector. Our wikipedia dataset has no
labels, so we will use an unsupervised training objective called the *Masked Language
Modeling* (MLM) ojective.

Essentially, we will be playing a big game of "guess the missing word". For each input
sample we will obscure 25% of our input data, and train our model to predict the parts we
covered up.
"""

"""
### Preprocess data for the MLM task

Our text preprocessing for the MLM task will occur in two stages.

1. Tokenize input text into integer sequences of token ids.
2. Mask certain positions in our input to predict on.

To tokenize, we can use a `keras_nlp.tokenizers.Tokenizer` -- the KerasNLP building block
for transforming text into sequences of integer token ids.

In particular, we will use `keras_nlp.tokenizers.WordPieceTokenizer` which does
*sub-word* tokenization. Sub-word tokenization is popular when training models on large
text corpora. Essentially, it allows our model to learn from uncommon words, while not
requireing a massive vocabulary of every word in our training set.

The second thing we need to do is mask our input for the MLM task. To do this, we can use
`keras_nlp.layers.MLMMaskGenerator`, which will randomly select a set of tokens in each
input and mask them out.

The tokenizer and the masking layer can both be used inside a call to
[tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map).
We can use `tf.data` to efficiently pre-compute each batch on the CPU, while our GPU or TPU
works on training with the batch that came before. Because our masking layer will
choose new words to mask each time, each epoch over our dataset will give us a totally
new set of labels to train on.
"""

# Setting sequence_length will trim or pad the token outputs to shape
# (batch_size, SEQ_LENGTH).
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file, sequence_length=SEQ_LENGTH,
)
# Setting mask_selection_length will trim or pad the mask outputs to shape
# (batch_size, PREDICTIONS_PER_SEQ).
masker = keras_nlp.layers.MLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=MASK_RATE,
    mask_selection_length=PREDICTIONS_PER_SEQ,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
)


def preprocess(inputs):
    inputs = tokenizer(inputs)
    outputs = masker(inputs)
    # Split the masking layer outputs into a (features, labels, and weights)
    # tuple that we can use with keras.Model.fit().
    features = {
        "tokens": outputs["tokens"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    weights = outputs["mask_weights"]
    return features, labels, weights


# We use prefetch() to pre-compute preprocessed batches on the fly on the CPU.
pretrain_ds = wiki_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pretrain_val_ds = wiki_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
# The masks will change each time you run the cell.
print(pretrain_val_ds.take(1).get_single_element())

"""
The above block sorts our dataset into a `(features, labels, weights)` tuple, which can be
passed directly to `keras.Model.fit()`.

We have two features:

1. `"tokens"`, where some tokens have been replaced with our mask token id.
2. `"mask_positions"`, which keeps track of which tokens we masked out.

Our labels are simply the ids we masked out.

Because not all sequences will have the same number of masks, we also keep a
`sample_weight` tensor, which removes padded labels from our loss function by giving them
zero weight.
"""

"""
### Create the Transformer encoder

KerasNLP provides all the building blocks to quickly build a Transformer encoder.

We use `keras_nlp.layers.TokenAndPositionEmbedding` to first embed our input token ids.
This layer simultaneously learns two embeddings -- one for words in a sentence and another
for integer positions in a sentence. The output embedding is simply the sum of the two.

Then we can add a series of `keras_nlp.layers.TransformerEncoder` layers. These are the
bread and butter of the Transformer model, using an attention mechanism to attend to
different parts of the input sentence, followed by a multi-layer perceptron block.

The output of this model will be a encoded vector per input token id. Unlike the
bag-of-words model we used as a baseline, this model will embed each token accounting for
the context in which it appeared.
"""

inputs = keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32)

# Embed our tokens with a positional embedding.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM,
)
outputs = embedding_layer(inputs)

# Apply layer normalization and dropout to the embedding.
outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(outputs)
outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

# Add a number of encoder blocks
for i in range(NUM_LAYERS):
    outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        layer_norm_epsilon=NORM_EPSILON,
    )(outputs)

encoder_model = keras.Model(inputs, outputs)
encoder_model.summary()

"""
### Pretrain the Transformer

You can think of the `encoder_model` as it's own modular unit, it is the piece of our
model that we are really interested in for our downstream task. However we still need to
set up the encoder to train on the MLM task; to do that we attach a
`keras_nlp.layers.MLMHead`.

This layer will take as one input the token encodings, and as another the positions we
masked out in the original input. It will gather the token encodings we masked, and
transform them back in predictions over our entire vocabulary.

With that, we are ready to compile and run pretraining. If you are running this in a
colab, note that this will take about an hour. Training Transformer is famously compute
intesive, so even this relatively small Transformer will take some time.
"""

# Create the pretraining model by attaching a masked language model head.
inputs = {
    "tokens": keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32),
    "mask_positions": keras.Input(shape=(PREDICTIONS_PER_SEQ,), dtype=tf.int32),
}

# Encode the tokens.
encoded_tokens = encoder_model(inputs["tokens"])

# Predict an output word for each masked input token.
# We use the input token embedding to project from our encoded vectors to
# vocabulary logits, which has been shown to improve training efficiency.
outputs = keras_nlp.layers.MLMHead(
    embedding_weights=embedding_layer.token_embedding.embeddings, activation="softmax",
)(encoded_tokens, mask_positions=inputs["mask_positions"])

# Define and compile our pretraining model.
pretraining_model = keras.Model(inputs, outputs)
pretraining_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.experimental.AdamW(PRETRAINING_LEARNING_RATE),
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)

# Pretrain the model on our wiki text dataset.
pretraining_model.fit(
    pretrain_ds, validation_data=pretrain_val_ds, epochs=PRETRAINING_EPOCHS,
)

# Save this base model for further finetuning.
encoder_model.save("encoder_model")

"""
## Fine-tuning

After pretraining, we can now fine-tune our model on the `SST-2` dataset. We can
leverage the ability of the encoder we build to predict on words in context to boost our
our performance on the downstream task.
"""

"""
### Preprocess data for classification

Preprocessing for fine-tuning is much simpler than for our pretraining MLM task. We just
tokenize our input sentences and we are ready for training!
"""


def preprocess(sentences, labels):
    return tokenizer(sentences), labels


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
finetune_ds = sst_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
finetune_val_ds = sst_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
print(finetune_val_ds.take(1).get_single_element())

"""
### Fine-tune the Transformer

To go from our encoded token output to a classification prediction, we need to attach
another "head" to our Transformer model. We can afford to be simple here. We pool
the encoded tokens together, and use a single dense layer to make a prediction.
"""

# Reload the encoder model from disk so we can restart fine-tuning from scratch.
encoder_model = keras.models.load_model("encoder_model", compile=False)

# Take as input the tokenized input.
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32)

# Encode and pool the tokens.
encoded_tokens = encoder_model(inputs)
pooled_tokens = keras.layers.GlobalAveragePooling1D()(encoded_tokens)

# Predict an output label.
outputs = keras.layers.Dense(1, activation="sigmoid")(pooled_tokens)

# Define and compile our finetuning model.
finetuning_model = keras.Model(inputs, outputs)
finetuning_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.experimental.AdamW(FINETUNING_LEARNING_RATE),
    metrics=["accuracy"],
)

# Finetune the model for the SST-2 task.
finetuning_model.fit(
    finetune_ds, validation_data=finetune_val_ds, epochs=FINETUNING_EPOCHS,
)

"""
Pretraining was enough to boost our performance to 84%, and this is hardly the ceiling
for Transformer models. You may have noticed during pretraining that our validation
performance was still steadily increasing. Our model is still significantly undertrained.
Training for more epochs, training a large Transformer, and training on more unlabeled
text would all continue to boost performance significantly.
"""

"""
### Save a model that accepts raw text

The last thing we can do with our fine-tuned model is saveing including our tokenization
layer. One of the key advantages of KerasNLP is all preprocessing is done inside the
[TensorFlow graph](https://www.tensorflow.org/guide/intro_to_graphs), making it possible
to save and restore a model that can directly run inference on raw text!
"""

# Add our tokenization into our final model.
inputs = keras.Input(shape=(), dtype=tf.string)
tokens = tokenizer(inputs)
outputs = finetuning_model(tokens)
final_model = keras.Model(inputs, outputs)
final_model.save("final_model")

# This model can predict directly on raw text.
restored_model = keras.models.load_model("final_model", compile=False)
inference_data = tf.constant(["Terrible, no good, trash.", "So great; I loved it!"])
print(restored_model(inference_data))

"""
One of the key goals of KerasNLP is to provide a modular approach to NLP model building.
We have shown one approach to building a Transformer here, but KerasNLP supports an ever
growing array of components for preprocessing text and building models. We hope it makes
it easier to experiment on solutions to your natural language problems.
"""
