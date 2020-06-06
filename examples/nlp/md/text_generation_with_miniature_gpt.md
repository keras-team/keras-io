
# Text Generation with miniature GPT

**Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
**Date created:** 2020/05/29<br>
**Last modified:** 2020/05/29<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_generation_with_miniature_gpt.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_generation_with_miniature_gpt.py)


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
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

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
maxlen = 20  # Max sequence size
embed_dim = 64  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 64  # Hidden layer size in feed forward network inside transformer


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
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.batch(batch_size)

# Create vectcorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    max_tokens=vocab_size - 1, output_mode="int", output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

# Add PAD and UNK tokens if not already present
if len(vocab) == vocab_size - 2:
    vocab.insert(0, "[UNK]")
    vocab.insert(0, "[PAD]")


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
100 80.2M  100 80.2M    0     0   9.7M      0  0:00:08  0:00:08 --:--:-- 18.0M

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
num_tokens_generated = 10
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
this movie is one thing it is a movie about a [UNK] [UNK] [UNK]
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 5.7165 - dense_6_loss: 5.7165
Epoch 2/30
generated text:
this movie is so boring that it is one of the worst movie ever
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 16s - loss: 4.9898 - dense_6_loss: 4.9898
Epoch 3/30
generated text:
this movie is terrible it is the worst movie of the actors are so
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.7290 - dense_6_loss: 4.7290
Epoch 4/30
generated text:
this movie is so stupid i love this movie i cant even say that
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.5652 - dense_6_loss: 4.5652
Epoch 5/30
generated text:
this movie is so horrible the movie made and that makes the worst film
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.4481 - dense_6_loss: 4.4481
Epoch 6/30
generated text:
this movie is not really a terrible plot for the acting but nothing is
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.3589 - dense_6_loss: 4.3589
Epoch 7/30
generated text:
this movie is a [UNK] of the same film which is not only a
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.2889 - dense_6_loss: 4.2889
Epoch 8/30
generated text:
this movie is a complete waste of time the celluloid it was made it
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.2309 - dense_6_loss: 4.2309
Epoch 9/30
generated text:
this movie is about the most stupid i have ever seen it was on
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.1817 - dense_6_loss: 4.1817
Epoch 10/30
generated text:
this movie is really bad i have the misfortune to watch this movie but
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 18s - loss: 4.1395 - dense_6_loss: 4.1395
Epoch 11/30
generated text:
this movie is so bad that is the worst movie ever made it has
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 18s - loss: 4.1014 - dense_6_loss: 4.1014
Epoch 12/30
generated text:
this movie is so bad it really is terrible its not the story is
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.0677 - dense_6_loss: 4.0677
Epoch 13/30
generated text:
this movie is so stupid with a [UNK] of a movie that is not
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.0376 - dense_6_loss: 4.0376
Epoch 14/30
generated text:
this movie is a complete waste of time and money on the plot is
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 4.0097 - dense_6_loss: 4.0097
Epoch 15/30
generated text:
this movie is really a film with me that it has no idea how
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.9853 - dense_6_loss: 3.9853
Epoch 16/30
generated text:
this movie is really bad i cant believe it i just finished watching it
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.9622 - dense_6_loss: 3.9622
Epoch 17/30
generated text:
this movie is the best movie i have ever watched all it is a
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.9410 - dense_6_loss: 3.9410
Epoch 18/30
generated text:
this movie is a bad movie that is so awful it was a terrible
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.9201 - dense_6_loss: 3.9201
Epoch 19/30
generated text:
this movie is bad the acting of the main character in history for this
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 18s - loss: 3.9026 - dense_6_loss: 3.9026
Epoch 20/30
generated text:
this movie is terrible for me in the 80s and i think the story
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 16s - loss: 3.8842 - dense_6_loss: 3.8842
Epoch 21/30
generated text:
this movie is a horrible waste of your talent it is a [UNK] of
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.8681 - dense_6_loss: 3.8681
Epoch 22/30
generated text:
this movie is not funny and it is not the plot or the film
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.8528 - dense_6_loss: 3.8528
Epoch 23/30
generated text:
this movie is really a terrible piece of junk and the acting was bad
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 18s - loss: 3.8379 - dense_6_loss: 3.8379
Epoch 24/30
generated text:
this movie is one of the worst movies i have ever seen it is
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 16s - loss: 3.8236 - dense_6_loss: 3.8236
Epoch 25/30
generated text:
this movie is a waste of the time i dont know how it is
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.8097 - dense_6_loss: 3.8097
Epoch 26/30
generated text:
this movie is not a film in a way too [UNK] of the movie
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.7975 - dense_6_loss: 3.7975
Epoch 27/30
generated text:
this movie is not really bad but the movie is not funny in the
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.7852 - dense_6_loss: 3.7852
Epoch 28/30
generated text:
this movie is not a [UNK] but it is really good i am a
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.7744 - dense_6_loss: 3.7744
Epoch 29/30
generated text:
this movie is really bad i know what i do with its a movie
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 17s - loss: 3.7633 - dense_6_loss: 3.7633
Epoch 30/30
generated text:
this movie is very interesting the first movie that is about a man whos
```
</div>
    
<div class="k-default-codeblock">
```
1563/1563 - 16s - loss: 3.7526 - dense_6_loss: 3.7526

<tensorflow.python.keras.callbacks.History at 0x7f76ee4172e8>

```
</div>