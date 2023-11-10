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
# We set the backend to TensorFlow. The code works with
# both `tensorflow` and `torch`. It does not work with JAX
# due to the behavior of `jax.numpy.tile` in a jit scope
# (used in `causal_attention_mask()`: `tile` in JAX does
# not support a dynamic `reps` argument.
# You can make the code work in JAX by wrapping the
# inside of the `causal_attention_mask` function in
# a decorator to prevent jit compilation:
# `with jax.ensure_compile_time_eval():`.
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

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
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


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
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
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
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  7926k      0  0:00:10  0:00:10 --:--:-- 7661k

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
                x = start_tokens[:maxlen]
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

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1699499022.078758  633491 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
/home/mattdangerw/miniconda3/envs/keras-tensorflow/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self.gen.throw(typ, value, traceback)

generated text:
this movie is a good example of the [UNK] " movies , and the movie was pretty well written , i had to say that the movie made me of the [UNK] " and was very well done . i 've seen a few
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 33s - 84ms/step - loss: 5.4696
Epoch 2/25
generated text:
this movie is so far the worst movies i have ever seen . it is that it just a bit of a movie but i really don 't think it is a very bad movie . it is a lot and the characters in
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 16s - 42ms/step - loss: 4.7016
Epoch 3/25
generated text:
this movie is a classic and has a good cast in a good story . the movie itself is good at best . the acting is superb , but the story is a little bit slow , the music hall , and music is
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 16s - 42ms/step - loss: 4.4533
Epoch 4/25
generated text:
this movie is a good , and is not the greatest movie ever since , the director has a lot of [UNK] , but it 's just a bit of the original and the plot has some decent acting , it has a bit
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 16s - 42ms/step - loss: 4.2985
Epoch 5/25
generated text:
this movie is really bad , the acting in this movie is bad and bad . it 's not bad it . it 's a bad story about a bad film but i can 't imagine if it 's a bad ending . the
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 4.1787
Epoch 6/25
generated text:
this movie is so bad , the bad acting , everything is awful , the script is bad , and the only one that i just saw in the original [UNK] . i was hoping it could make up the sequel . it wasn
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 4.0807
Epoch 7/25
generated text:
this movie is one of the best kung fu movies i 've ever seen , i have seen in my life that 's not for anyone who has to think of it , or maybe even though , i can 't find it funny
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 16s - 42ms/step - loss: 3.9978
Epoch 8/25
generated text:
this movie is just plain boring . . . . . . . . . . . . . . . . . [UNK] , the movie [UNK] . . . [UNK] . . . . . . [UNK] is a bad , it
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.9236
Epoch 9/25
generated text:
this movie is the only good movie i think i 've never seen it again . but it 's the only thing i feel about it . the story was about the fact that it was a very good movie . the movie has
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.8586
Epoch 10/25
generated text:
this movie is very well written and directed . it contains some of the best [UNK] in the genre . the story is about a group of actors , especially jamie harris and danny glover who are the only good guys that is really
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.8002
Epoch 11/25
generated text:
this movie is so terrible . i think that the movie isn 't as bad as you should go and watch it again . there were so many clichés that it 's a very bad movie in itself . there is no story line
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.7478
Epoch 12/25
generated text:
this movie is a total waste of money and money . i am surprised to find it very funny , very enjoyable . the plot is totally unbelievable , the acting is horrible . the story is awful , it 's not scary at
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.6993
Epoch 13/25
generated text:
this movie is so bad and not very good as it goes . it 's a nice movie and it 's so bad that it takes you back on your tv . i don 't really know how bad this movie is . you
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.6546
Epoch 14/25
generated text:
this movie is a great fun story , with lots of action , and romance . if you like the action and the story is really bad . it doesn 't get the idea , but you have your heart of darkness . the
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.6147
Epoch 15/25
generated text:
this movie is a little more than a horror film . it 's not really a great deal , i can honestly say , a story about a group of teens that are all over the place . but this is still a fun
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.5769
Epoch 16/25
generated text:
this movie is just about a guy who is supposed to be a girl in the [UNK] of a movie that doesn 't make sense . the humor is not to watch it all the way the movie is . you can 't know
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.5425
Epoch 17/25
generated text:
this movie is one of the best movies i 've ever seen . i was really surprised when renting it and it wasn 't even better in it , it was not even funny and i really don 't really know what i was
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.5099
Epoch 18/25
generated text:
this movie is so bad . i think it 's a bit overrated . i have a lot of bad movies . i have to say that this movie was just bad . i was hoping the [UNK] . the [UNK] is good "
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 43ms/step - loss: 3.4800
Epoch 19/25
generated text:
this movie is one of the best kung fu movies i 've ever seen . it was a great movie , and for the music . the graphics are really cool . it 's like a lot more than the action scenes and action
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.4520
Epoch 20/25
generated text:
this movie is just plain awful and stupid .i cant get the movie . i cant believe people have ever spent money and money on the [UNK] . i swear i was so embarrassed to say that i had a few lines that are
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.4260
Epoch 21/25
generated text:
this movie is one of those movies that i 've ever seen , and you must know that i can say that i was not impressed with this one . i found it to be an interesting one . the story of the first
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.4014
Epoch 22/25
generated text:
this movie is about a man 's life and it is a very good film and it takes a look at some sort of movie . this movie is one of the most amazing movie you have ever had to go in , so
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.3783
Epoch 23/25
generated text:
this movie is a great , good thing about this movie , even the worst i 've ever seen ! it doesn 't mean anything terribly , the acting and the directing is terrible . the script is bad , the plot and the
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.3564
Epoch 24/25
generated text:
this movie is one of the best movies ever . [UNK] [UNK] ' is about the main character and a nobleman named fallon ; is stranded on an eccentric , falls in love when her island escaped . when , meanwhile , the escaped
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.3362
Epoch 25/25
generated text:
this movie is very good . the acting , especially the whole movie itself - a total of the worst . this movie had a lot to recommend it to anyone . it is not funny . the story is so lame ! the
```
</div>
    
<div class="k-default-codeblock">
```
391/391 - 17s - 42ms/step - loss: 3.3170

<keras.src.callbacks.history.History at 0x7f2166975f90>

```
</div>