# Text Generation with miniature GPT

**Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
**Date created:** 2020/05/29<br>
**Last modified:** 2020/05/29<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/text_generation_with_miniature_gpt.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/text_generation_with_miniature_gpt.py)


**Description:** Implement miniature version of GPT and learn to generate text.

---
## Introduction

This example demonstrates autoregressive language modelling using a
a miniature version of GPT model.
The model consists of a single transformer block with causal masking
in the its attention layer.
We use the text from IMDB sentiment classification dataset for training
and generate new movie reviews for a given prompt.
When using this script with your own data, make sure it has atleast
1M words.

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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import os
import re
import string
import random

```

---
## Self-attention with causal masking

We compute self-attention as usual, but prevent any information to flow
from future tokens by masking the upper half of the scaled dot product matrix.



```python

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    @staticmethod
    def causal_attention_mask(n_dest, n_src, dtype):
        """
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        return tf.cast(m, dtype)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # prevent information flow from future tokens
        shape = tf.shape(scaled_score)
        dim_dest, dim_src = shape[2], shape[3]
        attention_mask = self.causal_attention_mask(
            dim_dest, dim_src, scaled_score.dtype
        )
        attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
        scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


```

---
## Implement a Transformer block as a layer



```python

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.att(inputs)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


```

---
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).



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
## Implement miniature GPT model



```python
vocab_size = 20000  # Only consider the top 20k words
maxlen = 100  # Max sequence size
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
## Prepare data for word level language modelling

We will download IMDB data, and combine training and validation sets for
text generation task.



```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz

```

```python

batch_size = 32

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

# Create dataset from text files
random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create vectcorization layer and adapt it to the text
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
text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)


```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  10.6M      0  0:00:07  0:00:07 --:--:-- 17.0M

50000 files

```
</div>
---
## Callback for generating text



```python

class TextGenerator(keras.callbacks.Callback):
    """Callback to generate text from trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for next token
    3. Sample next token and add it to the next input

    # Arguments
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from TextVectorization layer.
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
## Train

Note: This code should preferably be run on GPU.



```python
model = create_model()

model.fit(text_ds, verbose=2, epochs=30, callbacks=[text_gen_callback])

```

<div class="k-default-codeblock">
```
Epoch 1/30
generated text:
this movie is the best of the funniest and i have seen , and have ever seen in this movie . i don 't know it just to watch the show . but i don 't like this movie for those movies that they
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 5.0624 - dense_6_loss: 5.0624
Epoch 2/30
generated text:
this movie is not a good drama . it is not the only thing about the way . the story is very basic but is not just so much as a kid i think it was a bit more than i have the chance
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 4.4791 - dense_6_loss: 4.4791
Epoch 3/30
generated text:
this movie is the first movie it makes you wonder if you were going to watch and again . i can 't imagine how bad i felt like this , this movie wasn 't bad . i was expecting it a lot , but
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 4.2813 - dense_6_loss: 4.2813
Epoch 4/30
generated text:
this movie is the first time capsule of all time . i think i would like to say this is a good movie . it was very entertaining because it was a lot more interesting . it was not a good movie , and
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 4.1470 - dense_6_loss: 4.1470
Epoch 5/30
generated text:
this movie is a wonderful family film . it is a beautiful movie . it is an example of how a man who is a good movie that i have to do . but it is really the best way to watch . it
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 4.0431 - dense_6_loss: 4.0431
Epoch 6/30
generated text:
this movie is very different than that it is very entertaining in it . but it is also an interesting plot , it 's very interesting to the end of wwii and the war . the story is also [UNK] " is a true
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.9584 - dense_6_loss: 3.9584
Epoch 7/30
generated text:
this movie is a good movie . it isn 't scary , but that is a movie with lots of blood . it 's a very good horror movie but it has a good time , but the plot of the movie is good
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.8868 - dense_6_loss: 3.8868
Epoch 8/30
generated text:
this movie is a great movie that has the plot . the acting is good . i have seen some bad movies from my list ! but this one is so bad that it has to be so bad i can watch it over
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.8246 - dense_6_loss: 3.8246
Epoch 9/30
generated text:
this movie is a good one and the most amazing thing i 've ever seen . the movie was good , and i was so excited that it wasn 't . i didn 't get much . there were some good actors in this
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.7712 - dense_6_loss: 3.7712
Epoch 10/30
generated text:
this movie is a bit boring in a long way . i 've never heard of seeing such bad acting in this movie . i was just too long and boring . the music is awful , and the story , the story is
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.7230 - dense_6_loss: 3.7230
Epoch 11/30
generated text:
this movie is a terrible movie , but it has its elements of [UNK] " and is [UNK] . . i 'm a big fan of this movie , i am a big fan of horror movies , but it is not scary at
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.6805 - dense_6_loss: 3.6805
Epoch 12/30
generated text:
this movie is an absolute abomination . i don 't understand why it was a comedy , the characters , and i can 't believe it was meant to be , and it was a great comedy . there is something good . .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.6424 - dense_6_loss: 3.6424
Epoch 13/30
generated text:
this movie is a lot of fun watching the movie in its [UNK] -ness " . i was surprised to see a very good movie . it was a bit long , but it didn 't come across as good as the film .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.6081 - dense_6_loss: 3.6081
Epoch 14/30
generated text:
this movie is about a man who is a [UNK] , and he does in an old house with the spirits of [UNK] and the dead . i 'm not kidding , the plot has a [UNK] of old man , the woman is
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.5760 - dense_6_loss: 3.5760
Epoch 15/30
generated text:
this movie is one of the most beautiful films of all time . it is one of the greatest movies ever created , and in the wilderness . this movie was made by a beautiful young woman who finds her way out her .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.5473 - dense_6_loss: 3.5473
Epoch 16/30
generated text:
this movie is really bad , and it is so terrible . it 's hard to make a movie and a whole lot better than i watched , but this wasn 't a bad movie , but it 's pretty damn good . i
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.5210 - dense_6_loss: 3.5210
Epoch 17/30
generated text:
this movie is great ! ! i am a big fan of [UNK] . i don 't like this movie because i don 't think i can 't think of it , but it 's got me wrong and i 'm not sure why
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.4966 - dense_6_loss: 3.4966
Epoch 18/30
generated text:
this movie is one of those awful movies that is one of the few [UNK] films that were the [UNK] . there is no real story , a young girl with a young girl , who 's been kidnapped by an evil witch .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.4731 - dense_6_loss: 3.4731
Epoch 19/30
generated text:
this movie is one of the greatest movies i have ever watched . i watched it . the movie is so funny , it is very good . it is a great movie with a great acting in this movie , and i loved
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.4519 - dense_6_loss: 3.4519
Epoch 20/30
generated text:
this movie is a pretty interesting , the characters are very good . the story line is so good and the plot is good as the main character of a good [UNK] [UNK] ) is a bit of [UNK] [UNK] for the first time
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.4321 - dense_6_loss: 3.4321
Epoch 21/30
generated text:
this movie is not the worst movie ever . there is no emotion . this movie starts out on one of the most boring movies ever made . it doesn 't even come close to the conclusion of the director 's character [UNK] .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.4133 - dense_6_loss: 3.4133
Epoch 22/30
generated text:
this movie is just plain awful . it 's just awful . it doesn 't have a shred of good dialogue . there isn 't enough action and the actors are good . the movie is just plain awful . the acting is bad
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3956 - dense_6_loss: 3.3956
Epoch 23/30
generated text:
this movie is very bad , if i 'm not sure what the makers hoped it was going to be . i was expecting it to be a bad movie . it was terrible and i was expecting that the actors had a great
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3791 - dense_6_loss: 3.3791
Epoch 24/30
generated text:
this movie is very entertaining to see . i am a big fan of the [UNK] " . it was also a very good movie for the first time , and for the [UNK] " [UNK] ' , and i 'm not really sure
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3636 - dense_6_loss: 3.3636
Epoch 25/30
generated text:
this movie is a great movie and its a good one . the only thing in those movies that are really great , it is not a great movie , but this is a great movie . . it is not that bad .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3490 - dense_6_loss: 3.3490
Epoch 26/30
generated text:
this movie is great to see , but it is a great movie . this is the story of juliette lewis character and is a woman . her acting is so good . she doesn 't want to see how many times she gets
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3342 - dense_6_loss: 3.3342
Epoch 27/30
generated text:
this movie is very good , i have read a review for the fact that it was very good ! i am a christian [UNK] fan , and i must say that i am not a huge fan of the bible code of the
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3215 - dense_6_loss: 3.3215
Epoch 28/30
generated text:
this movie is really a great film , and it was really good . the story is about a girl named gerda and kai falling asleep . this one is a very well done and the rest of the cast is well written .
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.3086 - dense_6_loss: 3.3086
Epoch 29/30
generated text:
this movie is one of the best movies ever to win best movie ever and it is the first movie i ever saw . it was a very good movie . it 's really a lot of laughs and it 's funny . it
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.2968 - dense_6_loss: 3.2968
Epoch 30/30
generated text:
this movie is very interesting . i have no idea how the movie is . it 's just a little boring , confusing and pointless characters . it is also a good movie . i like the characters in the movie . i am
```
</div>

<div class="k-default-codeblock">
```
1563/1563 - 146s - loss: 3.2849 - dense_6_loss: 3.2849

<tensorflow.python.keras.callbacks.History at 0x7f0da81a3e10>

```
</div>
