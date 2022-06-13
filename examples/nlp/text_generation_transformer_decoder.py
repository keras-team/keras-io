"""
Title: Text Generation with Keras NLP TransformerDecoder
Author: Jesse Chan
Date created: 2022/06/13
Last modified: 2022/06/13
Description: Implementation of a small GPT-like model using the TransformerDecoder class.
"""
"""
# Download Library
"""

"""shell
!pip install -q git+https://github.com/jessechancy/keras-nlp.git@jesse-issue182
"""

"""
# Example using keras-nlp TransformerDecoder for text generation
"""

"""
## Imports
"""

import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras_nlp.layers.transformer_decoder import TransformerDecoder

"""
## Settings / Hyperparameters
"""

# Hyperparameters
BATCH_SIZE = 32
SEQ_LEN = 128
NUM_TOKENS_GENERATED = 40
VOCAB_SIZE = 5000
MIN_TRAINING_SEQ_LEN = 100
NUM_TRAINING_BATCHES = 500
# Training
LEARNING_RATE = 5e-4
EPOCHS = 8
# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 2
NUM_LAYERS = 1

# Mixed precision policy for faster training
policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)

"""
## Load data

We are going to be using a wikipedia text dataset. The dataset is filtered by character
length and parsed.
"""

# Download training data.
keras.utils.get_file(
  origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
  extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.raw")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

"""
## Load Tokenizer
Download vocabulary data to initialize WordPieceTokenizer. Preprocess training data with
tokenizer.

WordPieceTokenizer is an efficient implementation of the WordPiece algorithm used by BERT
and other models. It will strip, lower-case and do other un-reversable preprocessing
operations.
"""

# Download vocabulary data.
vocab_file = keras.utils.get_file(
  origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

# Get most frequent VOCAB_SIZE vocab tokens for faster/efficient training
vocab_list = []
with open(vocab_file, "r") as f:
  for line in f:
    vocab_list.append(line.strip())
vocab_sublist = vocab_list[:VOCAB_SIZE]

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_sublist,
    sequence_length=SEQ_LEN+1, # +1 for slicing train and label sequences
)

def preprocess(inputs):
  outputs = tokenizer(inputs)
  data = outputs[:, :-1]
  labels = outputs[:, 1:]
  return data, labels

# Tokenize and split into train and label sequences
ds = wiki_train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

"""
## Model
The model consists of the following layers:
1. A TokenAndPositionEmbedding layer, which combines the embedding for the token and its
position.
2. Multiple TransformerDecoder layers, with the default causal masking. The layer has no
cross-attention when run with decoder sequence only.
3. Output linear layer.
"""

def create_model():
  inputs = keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
  # embedding
  embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
      vocabulary_size=VOCAB_SIZE,
      sequence_length=SEQ_LEN,
      embedding_dim=EMBED_DIM,
      mask_zero=True
  )
  x = embedding_layer(inputs)
  # decoder layers
  for _ in range(NUM_LAYERS):
    transformer_block = TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = transformer_block(x) # no cross-attention
  # output
  outputs = keras.layers.Dense(VOCAB_SIZE)(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(
      "adam", loss=loss_fn,
  )
  return model

"""
## Training

We use a callback to test out the text generation at every epoch. This callback uses
top-k decoding.
We also take a subset of the training data for quicker training since this is used as an
example.
"""

class TopKTextGenerator(keras.callbacks.Callback):
  """A callback to generate text from a trained model using top-k.
  1. Feed some starting prompt to the model
  2. Predict probabilities for the next token
  3. Sample the next token and add it to the next input

  Arguments:
    max_tokens: Integer, the number of tokens to be generated after prompt.
    start_tokens: List of integers, the token indices for the starting prompt.
    tokenizer: WordPieceTokenizer object for de-tokenizing output.
    top_k: Integer, sample from the top_k token predictions.
    print_every: Integer, print after this many epochs.
  """

  def __init__(
      self, max_tokens, start_tokens, tokenizer, top_k=10, print_every=1
  ):
    self.max_tokens = max_tokens
    self.start_tokens = start_tokens
    self.tokenizer = tokenizer
    self.print_every = print_every
    self.k = top_k

  # top k sampling
  def sample_from(self, logits):
    logits, indices = tf.math.top_k(logits, k=self.k)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

  def on_epoch_end(self, epoch, logs=None):
    start_tokens = [_ for _ in self.start_tokens]
    if (epoch + 1) % self.print_every != 0:
      return
    num_tokens_generated = 0
    while num_tokens_generated <= self.max_tokens:
      # pad sequences
      pad_len = SEQ_LEN - len(start_tokens)
      sample_index = len(start_tokens) - 1
      if pad_len < 0:
        x = start_tokens[:SEQ_LEN]
        sample_index = SEQ_LEN - 1
      elif pad_len > 0:
        x = start_tokens + [0] * pad_len
      else:
        x = start_tokens
      x = np.array([x])

      y = self.model.predict(x)
      sample_token = self.sample_from(y[0][sample_index])
      start_tokens.append(sample_token)
      num_tokens_generated += 1
    txt = self.tokenizer.detokenize(start_tokens)
    print(f"generated text: \n{txt}\n")

start_prompt = "the"
# Unpadded token sequence
start_tokens = [tokenizer.token_to_id(token) for token in start_prompt.split()]
text_gen_callback = TopKTextGenerator(
    NUM_TOKENS_GENERATED, start_tokens, tokenizer
)

model = create_model()

model.fit(
    ds.take(NUM_TRAINING_BATCHES),
    verbose=2,
    epochs=EPOCHS,
    callbacks=[text_gen_callback]
)

