"""
Title: Character and byte level text generation with KerasNLP
Author: [Mohammad Aflah Khan](https://github.com/aflah02)
Date created: 2022/07/01
Last modified: 2022/07/01
Description: Use KerasNLP to train an open ended text generation LSTM model.
"""

"""
KerasNLP provides lots of building blocks for NLP (model layers, tokenizers, metrics, etc.) and
makes it convenient to construct NLP pipelines on the fly.

In this tutorial we'll use KerasNLP's `UnicodeCharacterTokenizer` and `ByteTokenizer` to train a 
character level LSTM model for open ended text generation. This example draws inspiration from 
[Character-level recurrent sequence-to-sequence model example](https://keras.io/examples/generative/lstm_character_level_text_generation/) 
by [fchollet](https://twitter.com/fchollet) for it's model architecture and uses 
[Don Quijote by Miguel de Cervantes Saavedra (Spanish Version)](https://www.gutenberg.org/ebooks/2000) as a data source

This tutorial broadly covers the following:
- Tokenization using `keras_nlp.tokenizers.UnicodeCharacterTokenizer` and `keras_nlp.tokenizers.ByteTokenizer` to obtain character level tokens.
- A LSTM model to generate text character-by-character 
- Decoding using `keras_nlp.utils.greedy_search` utility

This tutorial will be pretty useful and will be a good starting point for learning about KerasNLP and how to 
incorporate it into your own NLP pipelines.
"""

"""
## Setup

Let's do our imports first
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_nlp
import numpy as np
import random
import io

"""
Let's get our dataset and do some preprocessing to remove new line characters and the meta data present in the text
"""

path = keras.utils.get_file(
    "spanish_version.txt", origin="https://www.gutenberg.org/files/2000/2000-0.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read()
text = text.replace("\n", " ")

starting_text = "*** start of the project gutenberg ebook don quijote ***"
ind_start = text.find(starting_text)
ending_text = "*** end of the project gutenberg ebook don quijote ***"
ind_end = text.find(ending_text)
text = text[ind_start + len(starting_text):ind_end]
text = ' '.join(text.split())

"""
Now we'll create out tokenizers and tokenize our text, we limit the vocabulary for the UnicodeCharacterTokenizer to 
the max value in the Latin Extended-B block of Unicode version 1.1 as any thing beyond that is not Spanish text and 
probably should not be in our dataset
"""

UnicodeCharacterTokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(vocabulary_size= 591)
ByteTokenizer = keras_nlp.tokenizers.ByteTokenizer()

unicode_tokenized_text = UnicodeCharacterTokenizer(text)
byte_tokenized_text = ByteTokenizer(text)

"""
Here's what our tokens look like
"""

print(unicode_tokenized_text[:50])
print(byte_tokenized_text[:50])

"""
Let's see how we can detokenize the text!
"""

print(UnicodeCharacterTokenizer.detokenize(unicode_tokenized_text[:50]))
print(ByteTokenizer.detokenize(byte_tokenized_text[:50]))

"""
We can see post tokenization the datasets now differ which is to be expected as byte tokenization and unicode 
tokenization use different ways to tokenize characters, for plain English characters which lie in the ASCII range 
they don't differ however for more complex characters present in the Spanish language they may tokenize them differently. 
If we compare the tokens we can see they are not equal and even their lengths are not the same! Infact the byte tokenizer 
produces 57583 extra tokens
"""

print(unicode_tokenized_text == byte_tokenized_text)
print(f"Extra Tokens: {byte_tokenized_text.shape[0] - unicode_tokenized_text.shape[0]}")


"""
Let's look at a sample where they differ as well!
"""

print(unicode_tokenized_text[170:200], byte_tokenized_text[170:200])

"""
Now we'll define some parameters/hyperparameters.
"""

unicode_vocab_size = UnicodeCharacterTokenizer.vocabulary_size()
byte_vocab_size = ByteTokenizer.vocabulary_size()
seq_length = 100
epochs = 6

"""
Let's now convert our data to a `tf.data.Dataset` object, break it into chunks and shape it in a way where we use the last
seq_length number of tokens to predict the next character at each step
"""

unicode_encoded_dataset = tf.data.Dataset.from_tensor_slices(unicode_tokenized_text)
byte_encoded_dataset = tf.data.Dataset.from_tensor_slices(byte_tokenized_text)

unicode_encoded_sequences = unicode_encoded_dataset.batch(seq_length+1, drop_remainder=True)
byte_encoded_sequences = byte_encoded_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[-1]
    return input_text, target_text

unicode_encoded_dataset = unicode_encoded_sequences.map(split_input_target)
byte_encoded_dataset = byte_encoded_sequences.map(split_input_target)

"""
Let's have a look at our dataset
"""

for input_example, target_example in unicode_encoded_dataset.take(1):
    print("Input :", input_example)
    print("Target:", target_example)

for input_example, target_example in byte_encoded_dataset.take(1):
    print("Input :", input_example)
    print("Target:", target_example)

"""
Now we batch and shuffle out dataset before feeding it to the model
"""

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

unicode_encoded_dataset = (
    unicode_encoded_dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

byte_encoded_dataset = (
    byte_encoded_dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

"""
We create a simple LSTM model with an embedding layer and a dense layer for both our datasets.
The model will learn to predict the next character in the sequence based on the previous characters 
fed to it.
"""

x_unicode = keras.Input(shape=(None,))
e_unicode = tf.keras.layers.Embedding(input_dim = 591, output_dim = 128)(x_unicode)
y_unicode = keras.layers.LSTM(128)(e_unicode)
z_unicode = keras.layers.Dense(unicode_vocab_size, activation="softmax")(y_unicode)
model_unicode = keras.Model(x_unicode, z_unicode)

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model_unicode.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

x_byte = keras.Input(shape=(None,))
e_byte = tf.keras.layers.Embedding(input_dim = byte_vocab_size, output_dim = 128)(x_byte)
y_byte = keras.layers.LSTM(128)(e_byte)
z_byte = keras.layers.Dense(byte_vocab_size, activation="softmax")(y_byte)
model_byte = keras.Model(x_byte, z_byte)

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model_byte.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

"""
Now we'll train both our models for 6 Epochs post which the loss starts increasing once again
"""

model_unicode.fit(unicode_encoded_dataset, epochs=6)
model_byte.fit(byte_encoded_dataset, epochs=6)

"""
We'll now showcase the `keras_nlp.utils.greedy_search` utility offered by KerasNLP which essentially outputs the 
most likely token at each time step!

We'll generate outputs for the same prompt from both the models!

You might notice the .decode() call on some strings this is because the tensorflow tensors uses byte strings internally 
which can't directly be interpreted as things such as accents are replaced by special symbols and need to be decoded 
to a utf-8 format to be interpreted well
"""

def decode_sequences(input_sentences, model, generation_length = 50):
    tokenized = UnicodeCharacterTokenizer(input_sentences)
    prompt = tokenized.to_tensor(shape=tokenized.shape.as_list()[:-1] + [tf.strings.length(input_sentences[0])])
    def token_probability_fn(input):
        if (model == 'model_unicode'):
          x = model_unicode(input)[-1, :]
        elif (model == 'model_byte'):
          x = model_byte(input)[-1, :]
        return tf.expand_dims(x, axis=0)
    generated_tokens = keras_nlp.utils.greedy_search(
        token_probability_fn,
        prompt = prompt,
        max_length=100+generation_length,
        )
    generated_sentences = UnicodeCharacterTokenizer.detokenize(generated_tokens)
    return generated_sentences[0]

for i in range(5):
  start_index = random.randint(0, len(text) - 100 - 1)
  sentence = text[start_index: start_index + 100]
  models = ['model_unicode', 'model_byte']
  print(f"EXAMPLE {i+1}")
  print("PROMPT: ", sentence)
  print()
  for model in models:
    translated = decode_sequences(input_sentences = tf.constant([sentence]), model = model, generation_length = 50)
    print(f"Model: {model}")
    print("OUTPUT: ",translated.numpy().decode())
    print()

"""
EXAMPLE 1
PROMPT:   que ya su dulce esposo no vivía, rompió los aires con suspiros, hirió los cielos con quejas, maltra

Model: model_unicode
OUTPUT:   que ya su dulce esposo no vivía, rompió los aires con suspiros, hirió los cielos con quejas, maltrae. Ɏ y el como son prés como sancho de la carón

Model: model_byte
OUTPUT:   que ya su dulce esposo no vivía, rompió los aires con suspiros, hirió los cielos con quejas, maltra de su don quijoteâ, que su perreso de su cab

EXAMPLE 2
PROMPT:   pensar, porque vuestra excelencia la viera en él toda retratada; pero, ¿para qué es ponerme yo ahor

Model: model_unicode
OUTPUT:   pensar, porque vuestra excelencia la viera en él toda retratada; pero, ¿para qué es ponerme yo ahoro de la cama, sin puro por el como son por eso 

Model: model_byte
OUTPUT:   pensar, porque vuestra excelencia la viera en él toda retratada; pero, ¿para qué es ponerme yo ahor y en estar de su parecer mÃ­ en ella de su par

EXAMPLE 3
PROMPT:  scrito, el cual contenía estas razones: un necio e impertinente deseo me quitó la vida. si las nueva

Model: model_unicode
OUTPUT:  scrito, el cual contenía estas razones: un necio e impertinente deseo me quitó la vida. si las nueva a la carónes. pero con ella carónedo de la cart

Model: model_byte
OUTPUT:  scrito, el cual contenía estas razones: un necio e impertinente deseo me quitó la vida. si las nuevae de su don quijoteâ, que su perreso de su cab

EXAMPLE 4
PROMPT:  o. mandó la señora regenta a un criado suyo diese luego los ochenta escudos que le habían repartido,

Model: model_unicode
OUTPUT:  o. mandó la señora regenta a un criado suyo diese luego los ochenta escudos que le habían repartido,o esta can la mano hallación, y el como son por

Model: model_byte
OUTPUT:  o. mandó la señora regenta a un criado suyo diese luego los ochenta escudos que le habían repartido, y en estad que su cuerpo de su parecer mÃ­ en 

EXAMPLE 5
PROMPT:  odré decir con segura conciencia, que no es poco: "desnudo nací, desnudo me hallo: ni pierdo ni gano

Model: model_unicode
OUTPUT:  odré decir con segura conciencia, que no es poco: "desnudo nací, desnudo me hallo: ni pierdo ni ganoa de la carteso que sancho, sin puro por el como

Model: model_byte
OUTPUT:  odré decir con segura conciencia, que no es poco: "desnudo nací, desnudo me hallo: ni pierdo ni ganoa y en estar de su parecer mÃ­ en ella de su par"""