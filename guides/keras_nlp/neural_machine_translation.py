"""
Title: English-to-Spanish translation with KerasNLP
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2022/05/26
Last modified: 2022/05/26
Description: Use KerasNLP to train a sequence-to-sequence Transformer model on the machine translation task.
"""

"""
## Introduction

KerasNLP provides building blocks for NLP (model layers, tokenizers, metrics, etc.) and
makes it convenient to construct NLP pipelines.

In this example, we'll use KerasNLP layers to build an encoder-decoder Transformer
model, and train it on the English-to-Spanish machine translation task.

You'll learn how to:

- Tokenize text using the `WordPieceTokenizer` from KerasNLP.
- Implement a sequence-to-sequence Transformer model using KerasNLP's `TransformerEncoder`
, `TransformerDecoder` and `PositionalEmbedding` layers, and train it.
- Use the trained model to generate translations of unseen input sentences!
- Vectorize text using the Keras `TextVectorization` layer.

Don't worry if you aren't familiar with KerasNLP. This tutorial shows exactly
how simple and easy it is to use it! So, what are you waiting for? Dive right in!
"""

"""
## Setup
Before we start implementing the pipeline, let's import all the libraries we need.
"""

"""shell
pip install -q keras-nlp
"""

import keras_nlp
import numpy as np
import pathlib
import random
import tensorflow as tf
from tensorflow import keras

"""
Let's also define our hyperparameters.
"""

BATCH_SIZE = 64
EPOCHS = 1  # Should be 20 to get decent results.
MAX_SEQUENCE_LENGTH = 40

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

"""
## Downloading the data

We'll be working with an English-to-Spanish translation dataset
provided by [Anki](https://www.manythings.org/anki/). Let's download it:
"""

text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

"""
## Parsing the data

Each line contains an English sentence and its corresponding Spanish sentence.
The English sentence is the *source sequence* and Spanish one is the *target sequence*.
We prepend the token `"[start]"` and we append the token `"[end]"` to the Spanish sentence.
Additionally, let's convert the text to lowercase.
"""

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    eng = eng.lower()
    # eng = "[CLS] " + eng + " [SEP]"
    spa = "[CLS] " + spa.lower() + " [SEP]"
    text_pairs.append((eng, spa))

"""
Here's what our sentence pairs look like:
"""

for _ in range(5):
    print(random.choice(text_pairs))

"""
Now, let's split the sentence pairs into a training set, a validation set,
and a test set.
"""

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

"""
## Tokenizing the data
We'll define two tokenizers - one for the source language (English), and the other
for the target language (Spanish). We'll be using the `WordPieceTokenizer` from
KerasNLP to tokenize the text.

Before we define the tokenizer, let's first download the vocabulary files we'll be using.
These vocabularies were obtained while training the WordPieceTokenizer on a large
corpus of text.
"""

"""shell
wget https://huggingface.co/bert-base-uncased/raw/main/vocab.txt
mv vocab.txt eng_vocab.txt

wget https://raw.githubusercontent.com/dccuchile/beto/master/config/uncased_2M/vocab.txt
mv vocab.txt spa_vocab.txt
"""

"""
Let's now find the vocabulary sizes for both English and Spanish.
"""

with open("./eng_vocab.txt", "r") as eng_f:
    ENG_VOCAB_SIZE = len(eng_f.readlines())

with open("./spa_vocab.txt", "r") as spa_f:
    SPA_VOCAB_SIZE = len(spa_f.readlines())

"""
Now, let's define the tokenizers. We will use the download vocabulary files as
input to the tokenizer. The vocabulary files are such that every line contains
a token. We will also define a maximum sequence length so that all sequences
are padded to the same length, if the length of the sequence is less than the
specified sequence length. Otherwise, the sequence is truncated.
"""

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary="eng_vocab.txt",
    lowercase=False,
    split_pattern=" ",
    sequence_length=MAX_SEQUENCE_LENGTH,
)
spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary="spa_vocab.txt",
    lowercase=False,
    split_pattern=" ",
    sequence_length=MAX_SEQUENCE_LENGTH + 1,
)

"""
Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.
"""

eng_input_ex = text_pairs[0][0]
eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
print("English sentence: ", eng_input_ex)
print("Tokens: ", eng_tokens_ex)
print("Recovered text after detokenizing: ", eng_tokenizer.detokenize(eng_tokens_ex))

spa_input_ex = text_pairs[0][1]
spa_tokens_ex = spa_tokenizer.tokenize(spa_input_ex)
print("Spanish sentence: ", spa_input_ex)
print("Tokens: ", spa_tokens_ex)
print("Recovered text after detokenizing: ", spa_tokenizer.detokenize(spa_tokens_ex))

"""
Next, we'll format our datasets.

At each training step, the model will seek to predict target words N+1 (and beyond)
using the source sentence and the target words 0 to N.

As such, the training dataset will yield a tuple `(inputs, targets)`, where:

- `inputs` is a dictionary with the keys `encoder_inputs` and `decoder_inputs`.
`encoder_inputs` is the vectorized source sentence and `decoder_inputs` is the target sentence "so far",
that is to say, the words 0 to N used to predict word N+1 (and beyond) in the target sentence.
- `target` is the target sentence offset by one step:
it provides the next words in the target sentence -- what the model will try to predict.

Additionally, we will also mask the padding tokens.
"""


def format_dataset(eng, spa):
    eng = eng_tokenizer(eng)
    spa = spa_tokenizer(spa)
    return (
        {
            "encoder_inputs": eng,
            "encoder_padding_mask": tf.not_equal(eng, 0),
            "decoder_inputs": spa[:, :-1],
            "decoder_padding_mask": tf.not_equal(spa[:, :-1], 0),
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

"""
Let's take a quick look at the sequence shapes
(we have batches of 64 pairs, and all sequences are 40 steps long):
"""

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

"""
## Building the model

Now, let's move on to the exciting part - defining our model!
We first need an Embedding layer, i.e., a vector for every token in our input sequence.
This Embedding layer can be initialised randomly.
We also need a Positional Embedding layer which encodes the word order in the
sequence.
The convention is to add these two embeddings.
KerasNLP has a `TokenAndPositionEmbedding ` layer which does all of the above
steps for us.

Our sequence-to-sequence Transformer consists of a `TransformerEncoder`
and a `TransformerDecoder` chained together. Earlier, we would have had to define
these classes. But now, all these layers can be used off-the-shelf from KerasNLP!

The source sequence will be passed to the `TransformerEncoder`,
which will produce a new representation of it.
This new representation will then be passed
to the `TransformerDecoder`, together with the target sequence so far (target words 0 to N).
The `TransformerDecoder` will then seek to predict the next words in the target sequence (N+1 and beyond).

A key detail that makes this possible is causal masking.
The `TransformerDecoder` sees the entire sequences at once, and thus we must make
sure that it only uses information from target tokens 0 to N when predicting token N+1
(otherwise, it could use information from the future, which would
result in a model that cannot be used at inference time).
In order to enable causal masking, all we have to do is set the `use_causal_mask`
argument to True.
"""

# Encoder
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
encoder_padding_mask = keras.Input(
    shape=(None,), dtype="bool", name="encoder_padding_mask"
)

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=ENG_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x, padding_mask=encoder_padding_mask)
encoder = keras.Model([encoder_inputs, encoder_padding_mask], encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

decoder_padding_mask = keras.Input(
    shape=(None,), dtype="bool", name="decoder_padding_mask"
)
encoded_seq_padding_mask = keras.Input(
    shape=(None,), dtype="bool", name="encoded_seq_padding_mask"
)

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=SPA_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(
    decoder_sequence=x,
    encoder_sequence=encoded_seq_inputs,
    decoder_padding_mask=decoder_padding_mask,
    encoder_padding_mask=encoded_seq_padding_mask,
    use_causal_mask=True,
)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(SPA_VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
        decoder_padding_mask,
        encoded_seq_padding_mask,
    ],
    decoder_outputs,
)
decoder_outputs = decoder(
    [decoder_inputs, encoder_outputs, decoder_padding_mask, encoder_padding_mask]
)

transformer = keras.Model(
    [encoder_inputs, decoder_inputs, encoder_padding_mask, decoder_padding_mask],
    decoder_outputs,
    name="transformer",
)

"""
## Training our model

We'll use accuracy as a quick way to monitor training progress on the validation data.
Note that machine translation typically uses BLEU scores as well as other metrics, rather than accuracy.

Here we only train for 1 epoch, but to get the model to actually converge
you should train for at least 20 epochs.
"""

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

"""
## Decoding test sentences

Finally, let's demonstrate how to translate brand new English sentences.
We simply feed into the model the tokenized English sentence
as well as the target token `"[CLS]"`, then we repeatedly generated the next token, until
we hit the token `"[SEP]"`.
"""


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_tokenizer([input_sentence])
    input_sentence_padding_mask = tf.not_equal(tokenized_input_sentence, 0)
    tokenized_target_sentence = tf.constant(
        [[int(spa_tokenizer("[CLS]")[0])]]
    )  # 4 is the token ID of the "[CLS]" token
    for i in range(MAX_SEQUENCE_LENGTH):
        target_sentence_padding_mask = tf.not_equal(tokenized_target_sentence, 0)
        predictions = transformer(
            [
                tokenized_input_sentence,
                tokenized_target_sentence,
                input_sentence_padding_mask,
                target_sentence_padding_mask,
            ]
        )

        sampled_token_index = np.argmax(predictions[0, i, :])
        tokenized_target_sentence = tf.concat(
            (tokenized_target_sentence, tf.constant([[sampled_token_index]])), axis=1
        )

        if sampled_token_index == int(spa_tokenizer("[SEP]")[0]):
            break
    decoded_sentence = spa_tokenizer.detokenize(tokenized_target_sentence)
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for i in range(10):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
    print(f"*** Example {i} ***")
    print(input_sentence)
    print(translated.numpy()[0].decode("utf-8"))

"""
After 20 epochs, we get results such as:

*** Example 0 ***
You are smarter than that.
[CLS] [UNK] son mas listo que eso. [SEP]

*** Example 1 ***
Tom is a great motocross rider.
[CLS] [UNK] es un gran camino de la que se te de la [UNK] [SEP]

*** Example 2 ***
Why don't you just find another place to live?
[CLS] [UNK] que no te quiero encontrar otro sitio para vivir? [SEP]

*** Example 3 ***
Tom wants his money today.
[CLS] [UNK] su dinero hoy. [SEP]

*** Example 4 ***
I can't believe that you actually got into Harvard.
[CLS] [UNK] puedo creer que te lo esta en [UNK] [SEP]

*** Example 5 ***
Elephants are the largest land animals alive today.
[CLS] [UNK] son los animales mas de las animales de hoy. [SEP]

*** Example 6 ***
He loves singing.
[CLS] [UNK] [UNK] le encanta cantar. [SEP]

*** Example 7 ***
Tom enjoys working here, I think.
[CLS] [UNK] [UNK] le gusta trabajar aqui, ¿no. [SEP]

*** Example 8 ***
Tom wants to be famous.
[CLS] [UNK] quiere ser famoso. [SEP]

*** Example 9 ***
We saw it.
[CLS] [UNK] lo vio. [SEP]
"""
