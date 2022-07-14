"""
Title: Simple GPT Text Generation with KerasNLP transformers
Author: [Jesse Chan](https://github.com/jessechancy)
Date created: 2022/07/07
Last modified: 2022/07/07
Description: Use KerasNLP transformers to train a mini-GPT model for text generation.
"""

"""
## Introduction

In this example, we will use KerasNLP layers to build a scaled down Generative
Pre-trained (GPT) model. GPT is a transformer based model that allows you to generate
sophisticated text from a small input.

We will train the model on the [simplebooks-92](https://arxiv.org/abs/1911.12391) corpus,
which is a dataset made from several novels. It is a good dataset for this example since
it has a small vocabulary and high word frequency, which is beneficial when training a
model with few parameters.

This example combines concepts from [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/) 
with KerasNLP abstractions. We will demonstrate how KerasNLP tokenization, model, metrics, and
inference libraries can be used to simplify the development process. 

The main KerasNLP features shown in this example are:

- [WordPieceTokenizer](https://keras.io/api/keras_nlp/tokenizers/word_piece_tokenizer/):
An efficient sub-word tokenizer used by BERT and other models.
- [TokenAndPositionEmbedding](https://keras.io/api/keras_nlp/layers/token_and_position_embedding/): 
A layer that learns an embedding based on the token and it's position in the
input.
- [TransformerDecoder](https://keras.io/api/keras_nlp/layers/transformer_decoder/): A
decoder transformer layer with causal masking, as described in the transformers paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs).
- [Text Generation Utilities](https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/utils/text_generation.py): 
Utilities for different algorithms for text generation, including Beam Search,
Top-K and Top-P.
"""

"""
# Mini-GPT Text Generation with KerasNLP transformers

This example requires KerasNLP. You can install it via the following command: `pip
install keras-nlp`
"""

"""
## Imports
"""

import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras

"""
## Settings / Hyperparameters
"""

# Data
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Inference
NUM_TOKENS_TO_GENERATE = 80

# Training
LEARNING_RATE = 5e-4
EPOCHS = 12
NUM_TRAINING_BATCHES = 1000

# Model
BATCH_SIZE = 64
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

"""
## Load Tokenizer

Here we download the vocabulary data and initialize
`keras_nlp.tokenizers.WordPieceTokenizer`. WordPieceTokenizer is an efficient
implementation of the WordPiece algorithm used by BERT and other models. It will strip,
lower-case and do other un-reversable preprocessing operations.

The vocabulary data we download is a list of sub-words sorted in descending order by
frequency. We want to limit the vocabulary as much as possible, as we will see later on
that it has a large affect on the number of model parameters. We also don't want to take
too few vocabulary, or there would be too many OOV sub-words. 

We strip the first 1000 vocabulary that are all of the form `"[unusedXX]"`, since they
are all not useful for this text corpus and unneccesarily increases vocabulary size. Then
we take the `VOCAB_SIZE` most frequent vocabularies after that. There are also some
reserved tokens, so we want to make sure that `"[PAD]"` is at index 0 and `"[UNK]"` is in
the vocabulary, since it is used for padding and unknown words respectively.
"""

# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)
# Get most frequent VOCAB_SIZE vocab tokens for faster/efficient training.
vocab_list = []
with open(vocab_file, "r") as f:
    for line in f:
        vocab_list.append(line.strip())

# The first 1000 tokens are unused tokens that unneccesarily increases vocab size.
vocab_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocab_list[1000:]
vocab_list = vocab_list[:VOCAB_SIZE]

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_list,
    sequence_length=SEQ_LEN + 1,  # +1 for slicing feature and label.
)

"""
## Load Data

Now, let's download the dataset! The SimpleBooks dataset has a small vocabulary, which
makes it easier to fit a simple model on.
"""

keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# Load simplebooks-92 train set and filter out short lines.
train_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load simplebooks-92 validation set and filter out short lines.
valid_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)

"""
## Tokenize Data

We pre-process the dataset by tokenizing and splitting it into `features` and `labels`.
"""


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = outputs[:, :-1]
    labels = outputs[:, 1:]
    return features, labels


# Tokenize and split into train and label sequences.
ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
ds_valid = valid_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

"""
## Model

We create our scaled down GPT model with the following layers:

- One `keras_nlp.layers.TokenAndPositionEmbedding` layer, which combines the embedding
for the token and its position. 
- Multiple `keras_nlp.layers.TransformerDecoder` layers, with the default causal masking.
The layer has no cross-attention when run with decoder sequence only.
- One output dense linear layer
"""


def create_model():
    inputs = keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    # Embedding.
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )
    x = embedding_layer(inputs)
    # Transformer decoders.
    for _ in range(NUM_LAYERS):
        transformer_block = keras_nlp.layers.TransformerDecoder(
            num_heads=NUM_HEADS,
            intermediate_dim=FEED_FORWARD_DIM,
        )
        x = transformer_block(x)  # Giving one argument only skips cross-attention.
    # Output.
    outputs = keras.layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model


model = create_model()

"""
Let's take a look at our model summary here! We can see that a large majority of the
parameters are in the token and position embedding and the output dense layer! This means
that the vocabulary size (`VOCAB_SIZE`) has a large affect on the size of the model,
while the number of transformer decoder layers (`NUM_LAYERS`) doesn't affect it much.
"""

model.summary()

"""
## Training

Now that we have our model, let's train it. We use a subset of the training data to save
on training time. It would also be beneficial to use a GPU to speed up the training
process. Take a break and grab a drink while you wait!
"""

model.fit(ds.take(1), validation_data=ds_valid.take(1), verbose=2, epochs=EPOCHS)

"""
## Inference

Welcome back! With our trained model, we can test it out to gauge it's performance. Since
this is a dataset of mostly fictional books, there is bound to be a hero, so let's use
"The hero" as our starting string! We run it through the tokenizer to get the input for
the model. 
"""

MAX_PREDICT_LEN = 80
start_prompt = "The hero"
# Unpadded token sequence.
start_tokens = [tokenizer.token_to_id(_) for _ in start_prompt.split()]

"""
We will use the `keras_nlp.utils` library for inference. Every text generation
utility would require a `token_logits_fn()` wrapper around the model. This wrapper takes
in an unpadded token sequence, and requires the logits of the next token as the output.
For this transformer model, we will need to pad the input before passing it through the
model.
"""


def token_logits_fn(inputs):
    cur_len = inputs.shape[1]
    padded = tf.pad(inputs, tf.constant([[0, 0], [0, SEQ_LEN - cur_len]]))
    output = model(padded)
    return output[:, cur_len - 1, :]


"""
Creating the wrapper function is the most complex part of using these functions. Now that
it's done, let's test out the different utilties, starting with greedy search.
"""

"""
### Greedy Search

We greedily pick the most probable token at each timestep. In other words, we get the
argmax of the model output.
"""

output_tokens = keras_nlp.utils.greedy_search(
    token_logits_fn,
    tf.convert_to_tensor(start_tokens),
    max_length=NUM_TOKENS_TO_GENERATE,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

"""
As we can see, greedy search starts out making some sense, but quickly starts repeating
itself. This is a common problem with text generation that can be fixed by some of the
probabilistic text generation utilities shown later on!
"""

"""
### Beam Search

At a high-level, beam search keeps track of the `num_beams` most probable sequences at
each timestep, and predicts the best next token from all sequences. It is an improvement
over greedy search since it stores more possibilities. However, it is less efficient than
greedy search since it has to compute and store multiple potential sequences.

Note: beam search with `num_beams=1` is identical to greedy search
"""

output_tokens = keras_nlp.utils.beam_search(
    token_logits_fn,
    tf.convert_to_tensor(start_tokens),
    max_length=NUM_TOKENS_TO_GENERATE,
    num_beams=10,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")

"""
Similar to greedy search, beam search quickly starts repeating itself, since it is still
a deterministic method.
"""

"""
### Random Search

Random search is our first probabilistic method. At each time step, it samples the next
token using the softmax probabilities provided by the model.
"""

output_tokens = keras_nlp.utils.random_search(
    token_logits_fn,
    tf.convert_to_tensor(start_tokens),
    max_length=NUM_TOKENS_TO_GENERATE,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")

"""
Voila, no repetitions! However, with random search, we may see some nonsensical words
appearing since any word in the vocabulary has a chance of appearing with this sampling
method. This is fixed by our next search utility, top-k search.
"""

"""
### Top-K Search

Similar to random search, we sample the next token from the probability distribution
provided by the model. The only difference is that here, we select out the top `k` most
probable tokens, and distribute the probabiltiy mass over them before sampling. This way,
we won't be sampling from low probability tokens, and hence we would have less
nonsensical words!
"""

output_tokens = keras_nlp.utils.top_k_search(
    token_logits_fn,
    tf.convert_to_tensor(start_tokens),
    max_length=NUM_TOKENS_TO_GENERATE,
    k=10,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")

"""
### Top-P Search

Even with the top-k search, there is something to improve upon. With top-k search, the
number `k` is fixed, which means it selects the same number of tokens for any probability
distribution. Consider two scenarios, one where the probability mass is concentrated over
2 words and another where the probability mass is evenly concentrated across 10. Should
we choose `k=2` or `k=10`? There is not a one size fits all `k` here.

This is where top-p search comes in! Instead of choosing a `k`, we choose a probability
`p` that we want the probabilities of the top tokens to sum up to. This way, we can
dynamically adjust the `k` based on the probability distribution. By setting `p=0.9`, if
90% of the probability mass is concentrated on the top 2 tokens, we can filter out the
top 2 tokens to sample from. If instead the 90% is distributed over 10 tokens, it will
similarly filter out the top 10 tokens to sample from.
"""

output_tokens = keras_nlp.utils.top_p_search(
    token_logits_fn,
    tf.convert_to_tensor(start_tokens),
    max_length=NUM_TOKENS_TO_GENERATE,
    p=0.5,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")

"""
### Using Callbacks

We can also wrap the utilities in a callback, which allows you to print out a prediction
sequence for every epoch of the model! Here is an example of a callback for top-k search:
"""


class TopKTextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model using top-k."""

    def __init__(self, k):
        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = keras_nlp.utils.top_k_search(
            token_logits_fn,
            tf.convert_to_tensor(start_tokens),
            max_length=NUM_TOKENS_TO_GENERATE,
            k=self.k,
            from_logits=True,
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# Dummy training loop to demonstrate callback.
model.fit(ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])

"""
## Conclusion

Congrats, you made it through the example! To recap, in this example, we use
WordPieceTokenizer with the pre-trained BERT vocabulary to create sub-word tokens from our data.
We then create a simple GPT-like model using Keras NLP's TokenAndPositionEmbedding and
TransformerDecoder layers. Finally, we demonstrate inference with the text generation
utility library and show how the different algorithms perform.
"""
