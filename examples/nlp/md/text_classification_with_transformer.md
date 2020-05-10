# Text classification with Transformer

**Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
**Date created:** 2020/05/10<br>
**Last modified:** 2020/05/10<br>
**Description:** Implement transformer block as a Keras layer and use it for text classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_with_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py)



---
## Setup



```python
import tensorflow as tf
from tensorflow import keras

```

---
## Implement multi head self attention as a Keras layer



```python

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), "embedding dimension not divisible by num heads"
        self.projection_dim = embed_dim // num_heads
        self.wq = keras.layers.Dense(embed_dim)
        self.wk = keras.layers.Dense(embed_dim)
        self.wv = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(x)[0]
        q = self.wq(x)  # (batch_size, seq_len, embed_dim)
        k = self.wk(x)  # (batch_size, seq_len, embed_dim)
        v = self.wv(x)  # (batch_size, seq_len, embed_dim)
        q = self.separate_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        k = self.separate_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        v = self.separate_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(q, k, v)
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
## Implement transformer block as a layer



```python

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerLayer, self).__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.att(x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


```

---
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).



```python

class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(EmbeddingLayer, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=emded_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


```

---
## Create classifier model using transformer layer

Transformer layer outputs one vector for each time step of your input sequence. Here, we take the mean across all time steps and build a two layered feed forward network on top of it.



```python

class TransformerClassifier(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embed_dim, ff_dim, num_heads):
        super(TransformerClassifier, self).__init__()
        self.emb = EmbeddingLayer(maxlen, vocab_size, embed_dim)
        self.transformer = TransformerLayer(embed_dim, num_heads, ff_dim)
        self.prehead = keras.layers.Dense(20, activation="relu")
        self.dropout1 = keras.layers.Dropout(0.05)
        self.dropout2 = keras.layers.Dropout(0.05)
        self.head = keras.layers.Dense(2, activation="softmax")

    def call(self, x, training):
        x = self.emb(x)
        x = self.transformer(x, training)
        x = tf.math.reduce_mean(x, axis=1)
        x = self.dropout1(x, training=training)
        x = self.prehead(x)
        x = self.dropout2(x, training=training)
        x = self.head(x)
        return x


```

---
## Download and prepare dataset



```python
max_features = 6000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17465344/17464789 [==============================] - 2s 0us/step
25000 Training sequences
25000 Validation sequences

```
</div>
---
## Train and Evaluate



```python
model = TransformerClassifier(
    maxlen=maxlen, vocab_size=max_features, embed_dim=32, ff_dim=32, num_heads=1
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
)

```

<div class="k-default-codeblock">
```
Epoch 1/2
782/782 [==============================] - 29s 37ms/step - loss: 0.3767 - accuracy: 0.8248 - val_loss: 0.3363 - val_accuracy: 0.8561
Epoch 2/2
782/782 [==============================] - 30s 39ms/step - loss: 0.2497 - accuracy: 0.8992 - val_loss: 0.2959 - val_accuracy: 0.8716

```
</div>