# Text generation with a miniature GPT

**Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
**Date created:** 2020/05/29<br>
**Last modified:** 2020/05/29<br>
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

This example should be run with `tf-nightly>=2.3.0-dev20200531` or
with TensorFlow 2.3 or higher.

**References:**

- [GPT](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
- [GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
- [GPT-3](https://arxiv.org/abs/2005.14165)

---
## Setup


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re
import string
import random

```

---
## Implement a Transformer block as a layer


```python

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

```

---
## Implement an embedding layer

Create two seperate embedding layers: one for tokens and one for token index
(positions).


```python

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
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
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
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
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


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
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  24.2M      0  0:00:03  0:00:03 --:--:-- 24.2M

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
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
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
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
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
391/391 - 135s - loss: 5.5949 - dense_2_loss: 5.5949
generated text:
this movie is a great movie . the film is so many other comments . the plot and some people were [UNK] to do . i think the story is about that it is not a good movie . there are very good actors
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 2/25
391/391 - 135s - loss: 4.7108 - dense_2_loss: 4.7108
generated text:
this movie is one of the worst movies i have ever seen . i have no doubt the better movies of this one 's worst movies i have ever seen . i don 't know what the hell , and i 'm not going
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 3/25
391/391 - 135s - loss: 4.4620 - dense_2_loss: 4.4620
generated text:
this movie is a very good movie , i think i am not a kid . the story is a great movie . the director who is a great director who likes the director 's film . this was not funny and the director
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 4/25
391/391 - 136s - loss: 4.3047 - dense_2_loss: 4.3047
generated text:
this movie is a very good story and very well . this movie is one of the worst movies i have ever seen , and there are some good actors and actresses in the movie , it is not the worst . the script
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 5/25
391/391 - 135s - loss: 4.1840 - dense_2_loss: 4.1840
generated text:
this movie is a very good movie . it is the best thing about it 's a very good movie . it 's not funny , very , it 's so bad that it 's so funny , it 's like most romantic movie
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 6/25
391/391 - 135s - loss: 4.0834 - dense_2_loss: 4.0834
generated text:
this movie is the worst . the acting is awful . i have to admit that you 're just watching this film as i have to say that it is a [UNK] with [UNK] [UNK] " in the last ten years . i think
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 7/25
391/391 - 135s - loss: 3.9987 - dense_2_loss: 3.9987
generated text:
this movie is really about the acting is good and the script . i don 't think this is just a waste of movie . it was so terrible that it wasn 't funny , but that 's what it was made in movies
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 8/25
391/391 - 134s - loss: 3.9242 - dense_2_loss: 3.9242
generated text:
this movie is so bad . the story itself is about a family guy named jack , who is told by a father , who is trying to get to help him to commit . he has the same problem and the [UNK] .
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 9/25
391/391 - 135s - loss: 3.8579 - dense_2_loss: 3.8579
generated text:
this movie is not bad , it does not deserve one . i can say that i was able to sit at , relax [UNK] . i was wrong , and i think i was able to buy the dvd , i would say
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 10/25
391/391 - 134s - loss: 3.7989 - dense_2_loss: 3.7989
generated text:
this movie is very funny ! its very funny . a touching movie about three women who don 't know who is not to go on with a movie that has a lot of fun to watch . it is funny . the main
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 11/25
391/391 - 134s - loss: 3.7459 - dense_2_loss: 3.7459
generated text:
this movie is not the best movie i 've seen in a long time . this movie was just about a guy who gets killed for one . . i saw this movie at a time when i first saw it in the movie
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 12/25
391/391 - 134s - loss: 3.6974 - dense_2_loss: 3.6974
generated text:
this movie is a good example of how many films have seen and many films , that are often overlooked , in the seventies , in fact it is more enjoyable than the average viewer has some interesting parallels . this movie is based
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 13/25
391/391 - 134s - loss: 3.6534 - dense_2_loss: 3.6534
generated text:
this movie is so bad ! i think this is one . i really didn 't think anybody who gets the impression that the people who is trying to find themselves to be funny . . there 's the humor is no punchline ?
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 14/25
391/391 - 134s - loss: 3.6123 - dense_2_loss: 3.6123
generated text:
this movie is really bad . the actors are good ,the acting is great . a must see [UNK] the worst in history of all time . the plot is so bad that you can 't even make a bad movie about the bad
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 15/25
391/391 - 134s - loss: 3.5745 - dense_2_loss: 3.5745
generated text:
this movie is one of the worst movies i 've ever had . the acting and direction are terrible . what i 've seen , i 've watched it several times , and i can 't really believe how to make a movie about
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 16/25
391/391 - 134s - loss: 3.5404 - dense_2_loss: 3.5404
generated text:
this movie is so bad it is . that it is supposed to be a comedy . the script , which is just as bad as some movies are bad . if you 're looking for it , if you 're in the mood
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 17/25
391/391 - 134s - loss: 3.5083 - dense_2_loss: 3.5083
generated text:
this movie is one of all bad movies i have a fan ever seen . i have seen a good movies , this isn 't the worst . i 've seen in a long time . the story involves twins , a priest and
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 18/25
391/391 - 134s - loss: 3.4789 - dense_2_loss: 3.4789
generated text:
this movie is a great movie . it 's a shame that it was hard to see that it was . this movie is a good movie . the movie itself is a complete waste of time and time you have a bad rant
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 19/25
391/391 - 134s - loss: 3.4513 - dense_2_loss: 3.4513
generated text:
this movie is not one of the most moving movies i have ever seen . the story is about the plot is just so ridiculous that i could have done it with the actors . the actors are great and the acting is great
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 20/25
391/391 - 134s - loss: 3.4251 - dense_2_loss: 3.4251
generated text:
this movie is about a man named todd . it is a funny movie that has a lot of nerve on screen . it is not just the right ingredients and a movie . it is a great film , and it is a
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 21/25
391/391 - 134s - loss: 3.4011 - dense_2_loss: 3.4011
generated text:
this movie is not only funny , but i have never seen it before . the other comments i am not kidding or have been [UNK] and the worst movie i have to be . . there is something that is no where else
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 22/25
391/391 - 134s - loss: 3.3787 - dense_2_loss: 3.3787
generated text:
this movie is a very entertaining , very funny , and very funny , very well written and very nicely directed movie . this was done , very well done , with very good acting and a wonderful script , a very good movie
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 23/25
391/391 - 133s - loss: 3.3575 - dense_2_loss: 3.3575
generated text:
this movie is the kind of movie you will not be disappointed . it 's like an [UNK] [UNK] , who is a movie . it 's a great story and the characters are great , the actors are good , their [UNK] ,
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 24/25
391/391 - 134s - loss: 3.3372 - dense_2_loss: 3.3372
generated text:
this movie is a classic 80s horror movie . this has a great premise and the characters is a bit too typical [UNK] and [UNK] " with the [UNK] " . it 's all that makes sense . the characters were shallow and unrealistic
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 25/25
391/391 - 134s - loss: 3.3182 - dense_2_loss: 3.3182
generated text:
this movie is not the worst movie i have ever seen . it 's a movie where i 've never seen it before and i 've seen it again and again , again , i can 't believe it was made in a theatre
```
</div>
    





<div class="k-default-codeblock">
```
<tensorflow.python.keras.callbacks.History at 0x7f327449c780>

```
</div>