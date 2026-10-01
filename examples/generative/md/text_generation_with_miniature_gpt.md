# Text generation with a miniature GPT

**Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
**Date created:** 2020/05/29<br>
**Last modified:** 2026/08/21<br>
**Description:** Implement a miniature version of GPT and train it to generate text.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/text_generation_with_miniature_gpt.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/text_generation_with_miniature_gpt.py)



---
## Introduction

This example demonstrates how to implement an autoregressive language model
using a miniature version of the GPT model.
The model consists of a single Transformer block with causal masking
in its attention layer.
We use the text from the IMDB sentiment classification dataset for training
and generate new movie reviews for a given prompt.
When using this script with your own dataset, make sure it has at least
1 million words.

**References:**

- [GPT](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
- [GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
- [GPT-3](https://arxiv.org/abs/2005.14165)

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
```

---
## Implement a Transformer block as a layer


```python

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.att(inputs, inputs, use_causal_mask=True)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

```

---
## Implement an embedding layer

Create two separate embedding layers: one for tokens and one for token index
(positions).


```python

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

```

---
## Implement the miniature GPT model


```python
vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model

```

---
## Prepare the data for word-level language modelling

Download the IMDB dataset and combine training and validation sets for a text
generation task.


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
```
</div>

  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

    
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

    
  0 80.2M    0 98304    0     0  62202      0  0:22:32  0:00:01  0:22:31 62178

    
  0 80.2M    0  304k    0     0   125k      0  0:10:55  0:00:02  0:10:53  125k

    
  0 80.2M    0  768k    0     0   223k      0  0:06:08  0:00:03  0:06:05  223k

    
  1 80.2M    1 1584k    0     0   355k      0  0:03:51  0:00:04  0:03:47  355k

    
  3 80.2M    3 3056k    0     0   558k      0  0:02:27  0:00:05  0:02:22  622k

    
  6 80.2M    6 5584k    0     0   860k      0  0:01:35  0:00:06  0:01:29 1118k

    
 11 80.2M   11 9728k    0     0  1296k      0  0:01:03  0:00:07  0:00:56 1855k

    
 19 80.2M   19 15.7M    0     0  1892k      0  0:00:43  0:00:08  0:00:35 3023k

    
 28 80.2M   28 22.7M    0     0  2446k      0  0:00:33  0:00:09  0:00:24 4279k

    
 37 80.2M   37 29.7M    0     0  2891k      0  0:00:28  0:00:10  0:00:18 5402k

    
 46 80.2M   46 37.4M    0     0  3313k      0  0:00:24  0:00:11  0:00:13 6443k

    
 55 80.2M   55 44.1M    0     0  3641k      0  0:00:22  0:00:12  0:00:10 7226k

    
 64 80.2M   64 51.6M    0     0  3934k      0  0:00:20  0:00:13  0:00:07 7475k

    
 72 80.2M   72 57.8M    0     0  4098k      0  0:00:20  0:00:14  0:00:06 7307k

    
 80 80.2M   80 64.4M    0     0  4266k      0  0:00:19  0:00:15  0:00:04 7221k

    
 88 80.2M   88 71.2M    0     0  4429k      0  0:00:18  0:00:16  0:00:02 7059k

    
 97 80.2M   97 78.2M    0     0  4583k      0  0:00:17  0:00:17 --:--:-- 6885k

    
100 80.2M  100 80.2M    0     0  4650k      0  0:00:17  0:00:17 --:--:-- 6924k



```python

batch_size = 128

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# Create a dataset from text files
random.shuffle(filenames)
text_ds = tf_data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tensorflow.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
text_ds = text_ds.prefetch(tf_data.AUTOTUNE)

```

<div class="k-default-codeblock">
```
50000 files
```
</div>

---
## Implement a Keras callback for generating text


```python

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[-maxlen:]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

```

---
## Train the model

Note: This code should preferably be run on GPU.


```python
model = create_model()

model.fit(text_ds, verbose=2, epochs=25, callbacks=[text_gen_callback])
```

<div class="k-default-codeblock">
```
Epoch 1/25

/usr/local/lib/python3.13/dist-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

generated text:
this movie is a good horror movie , and it 's not a good movie about an excellent family . it has never a few years ago , but the story and all over the movie is a good movie . it is the

391/391 - 60s - 154ms/step - loss: 5.6142

Epoch 2/25

generated text:
this movie is bad , it isn 't the best film . if you were going to get a copy of the [UNK] of the movie . but it is not the only a bad movie that is a great movie that makes it

391/391 - 48s - 123ms/step - loss: 4.7214

Epoch 3/25

generated text:
this movie is not the worst movie ! ! ! [UNK] and you 'll know what is about a good thing about this movie that makes me look like [UNK] " i can see it . i am not a big fan of the

391/391 - 49s - 126ms/step - loss: 4.4726

Epoch 4/25

generated text:
this movie is a little gem . it is a movie about a group of teenagers , who enjoy themselves , who are on a journey , the [UNK] , the movie is [UNK] of an ancient [UNK] " and the film is a

391/391 - 49s - 126ms/step - loss: 4.3193

Epoch 5/25

generated text:
this movie is very very entertaining . it is not worth a look at it . if it is a true story or the plot is a very funny . it has to be funny . it doesn 't take the word from that

391/391 - 49s - 126ms/step - loss: 4.2022

Epoch 6/25

generated text:
this movie is one of the worst movies i have ever watched . i can 't remember a little movie , but i don 't really care what i have ever seen on the first episode in the original series , but i can

391/391 - 50s - 127ms/step - loss: 4.1067

Epoch 7/25

generated text:
this movie is not the best film of all time . it was made on a saturday night television and i was hooked , when it first came out on tv . . . . . i was so , i went to a

391/391 - 50s - 128ms/step - loss: 4.0243

Epoch 8/25

generated text:
this movie is just terrible . the worst thing i 've seen it many films , and that you are looking for the dvd release in this movie . this is so awful that it is not a [UNK] of a horror movie .

391/391 - 50s - 127ms/step - loss: 3.9527

Epoch 9/25

generated text:
this movie is a great movie . the story is about as simple as a family employing the old [UNK] of the old age of northern afghanistan and the other members of [UNK] , the family are very well chosen , and it is

391/391 - 50s - 127ms/step - loss: 3.8885

Epoch 10/25

generated text:
this movie is so bad that it 's a good movie . i was expecting to see how bad it was , but it is not worth it . it is very good and i have seen to be one of them . the

391/391 - 50s - 127ms/step - loss: 3.8317

Epoch 11/25

generated text:
this movie is really bad . it 's very well paced , the ending is very well made . i am so sure , but if the script had no sense of humor or intelligence to be seen by [UNK] , or not in

391/391 - 82s - 210ms/step - loss: 3.7802

Epoch 12/25

generated text:
this movie is really bad . the plot is very simple . the acting is terrible . the plot [UNK] , but the movie is so bad they could have a good thing but it 's not for everyone , but the story .

391/391 - 51s - 130ms/step - loss: 3.7335

Epoch 13/25

generated text:
this movie is so funny , and it has a funny little . . the [UNK] of the original story : a [UNK] of a bunch of [UNK] [UNK] [UNK] ) who can 't act . the whole movie makes it so good it

391/391 - 49s - 126ms/step - loss: 3.6900

Epoch 14/25

generated text:
this movie is so bad i have ever seen . . bad . but i 'm a big fan of the horror genre is usually not even a bad movie . i have no idea , but i can say that i really liked

391/391 - 50s - 128ms/step - loss: 3.6512

Epoch 15/25

generated text:
this movie is very good . the story line in the first film . it is very slow , plodding , dull , plodding and surprisingly tedious . it 's not really bad , dull , dull , plodding pace , but i think

391/391 - 50s - 127ms/step - loss: 3.6147

Epoch 16/25

generated text:
this movie is not really a great idea for a film . it 's a classic that i can 't believe the main actor is in this movie that you could get better . i am still wondering who is a real movie but

391/391 - 50s - 128ms/step - loss: 3.5816

Epoch 17/25

generated text:
this movie is very bad . the first , a [UNK] , a [UNK] [UNK] " movie with [UNK] . i 'm glad to report the executives involved are at a [UNK] table , and their shows that you should have to be entertained

391/391 - 50s - 127ms/step - loss: 3.5503

Epoch 18/25

generated text:
this movie is not the worst . the script is a waste of time ! ! it has the script or plot of the movie is so bad that it 's not funny . the first of all , it has all , so

391/391 - 50s - 127ms/step - loss: 3.5218

Epoch 19/25

generated text:
this movie is so bad that i think it 's not a bad movie but it is so bad it 's a bad idea , the plot is simple but it 's not . the characters are all very likable and just a good

391/391 - 50s - 127ms/step - loss: 3.4949

Epoch 20/25

generated text:
this movie is a bad , and it doesn 't work . it 's a shame that it doesn 't make much seem to be a great movie . . . . a good thing that makes it a nice to see the story

391/391 - 50s - 127ms/step - loss: 3.4693

Epoch 21/25

generated text:
this movie is not a comedy about a girl who falls in love with her love . she is also friendly , sweet , and sweet , charming . . . . but the story line with a young [UNK] woman who is now

391/391 - 50s - 127ms/step - loss: 3.4459

Epoch 22/25

generated text:
this movie is really terrible ! i really liked [UNK] , it does not get the michigan , but it 's not funny . . [UNK] . and [UNK] ) [UNK] [UNK] is so sweet , the story of a young boy named dexter

391/391 - 50s - 127ms/step - loss: 3.4238

Epoch 23/25

generated text:
this movie is a bad movie , because of the bad acting , bad dialogue , and bad dialogue . it 's almost too cheesy sounding like some of them are not even [UNK] . [UNK] 's character was a good choice but i

391/391 - 50s - 128ms/step - loss: 3.4028

Epoch 24/25

generated text:
this movie is not bad , but that bad enough . the acting is good , the special effects are so awful that you can 't see . i am very sure that this isn 't the worst film ever . this is a

391/391 - 50s - 128ms/step - loss: 3.3821

Epoch 25/25

generated text:
this movie is the only one i 've ever seen a lot in . it seems like the movie , and i can only remember it , as a child , it 's the most irritating , tender , [UNK] , and my wife

391/391 - 50s - 127ms/step - loss: 3.3640

<keras.src.callbacks.history.History at 0x781cac40dd30>
```
</div>

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
- [Chapter 16: Text generation](https://deeplearningwithpython.io/chapters/chapter16_text-generation)
