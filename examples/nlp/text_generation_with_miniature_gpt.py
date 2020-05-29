"""
Title: Text Generation with miniature GPT
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/29
Last modified: 2020/05/29
Description: Implement miniature version of GPT and learn to generate text.
"""
"""
## Setup
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
## Self attention with causal masking

We compute self attention as usual, but prevent any information to flow
from future tokens by masking the upper half of scaled dot product matrix.
"""


def shape_list(x):
    """
    Returns shape of tensor as a List
    Deals with dynamic shape in tensorflow cleanly.
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


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
        _, _, dim_dest, dim_src = shape_list(scaled_score)
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


"""
## Implement a Transformer block as a layer
"""


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

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).
"""


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


"""
## Implement miniature GPT model
"""
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Max sequence size
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    model.compile(
        "adam", loss=["sparse_categorical_crossentropy", None],
    )
    return model


"""
## Prepare data for word level language modelling

We will download IMDB data, and combine training and validation sets for
text generation task.
"""


(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

x_train = np.hstack([x_train, x_val])  # Combine training and validation reviews

print(len(x_train), "Total sequences")

# Dictionary with words mapped to their vocabulary index
word_index = keras.datasets.imdb.get_word_index()

# Dictionary with token index mapped to words
# Special tokens:
# 0 : padding
# 1 : start of review
# 2 : Out of vocabulary
index_word = {}
for word in word_index:
    index = word_index[word]
    index_word[index + 3] = word

index_word[0] = "<pad>"
index_word[1] = "<s>"
index_word[2] = "<oov>"


def prepare_lm_inputs_labels(tokenized_sentences, max_len):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    x_lm, y_lm = [], []
    for sentence in tokenized_sentences:
        x = sentence[:-1]
        y = sentence[1:]
        pad_len = max_len - len(x)
        if pad_len > 0:
            x = x + [0] * pad_len
            y = y + [0] * pad_len
        elif pad_len < 0:
            x = x[:max_len]
            y = y[:max_len]
        x_lm.append(x)
        y_lm.append(y)
    return np.array(x_lm), np.array(y_lm)


x_lm, y_lm = prepare_lm_inputs_labels(x_train, max_len=maxlen)
x_lm.shape


"""
## Callback for generating text
"""


class TextGenerator(keras.callbacks.Callback):
    """
    Feed some starting prompt to the model
    Predict probabilities for next token
    Sample next token and add it to the next input
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, temp=1.0, print_every=5
    ):
        """
        :param max_tokens: number of tokens to be generated
        :param start_tokens: starting prompt
        :param index_to_word: Dict mapping token index to word
        :param temp: temperature dictates the variation during sampling
                     high temp = less probable tokens are also likely to be sampled
                     low temp = highly probable tokens get picked up
        :param print_every: print after this many epochs
        """
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.temp = temp

    def sample_from(self, preds):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / self.temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        print(f"\n\nepoch = {epoch+1}")
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated < self.max_tokens:
            pad_len = maxlen - len(self.start_tokens)
            if pad_len < 0:  # Truncate start token sequence
                x = self.start_tokens[:maxlen]
            elif pad_len > 0:  # Pad start token sequence
                x = self.start_tokens + [0] * pad_len
            else:
                x = self.start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][len(self.start_tokens) - 1])
            tokens_generated.append(sample_token)
            self.start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([index_word[_] for _ in self.start_tokens + tokens_generated])
        print(f"{txt}\n\n")


start_prompt = "this film"
start_tokens = [1] + [word_index[_] + 3 for _ in start_prompt.split()]
num_tokens_generated = 20
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, index_word)


"""
## Create model
"""
use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()


"""
## Train
"""

model.fit(
    x_lm, y_lm, batch_size=64, verbose=2, epochs=50, callbacks=[text_gen_callback]
)
