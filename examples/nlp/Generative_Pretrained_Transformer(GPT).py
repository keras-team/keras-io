"""
Title: Generative Pretrained Transformer(GPT)
Author: [Sadeq](https://github.com/Sadeqk94) [Kord](https://www.linkedin.com/in/sadeq-kord)
Date created: 2024/05/25
Last modified: 2024/05/25
Description: Generative Transformer for Tiny Shakespeare dataset.
Accelerator: GPU
"""
"""
# Introduction
## Let's build GPT using Tensorflow and Keras!

For those interested in learning about GPT models, there is an excellent video by [Andrej
Karpathy](https://karpathy.ai/) titled ["Let's build GPT: from scratch, in code, spelled
out."](https://youtu.be/kCc8FmEb1nY?si=_l3tBiaZgq1NXwWW). In the video, Karpathy provides
a detailed, step-by-step guide to building a GPT model using PyTorch.

In this notebook, I present a TensorFlow/Keras version of Karpathy's implementation. This
notebook follows the same principles and steps outlined in the video, allowing you to
gain the same understanding and insights using TensorFlow/Keras. You can follow along
with the video and refer to this notebook for the equivalent TensorFlow/Keras code,
making it a valuable resource for anyone familiar with TensorFlow or looking to learn it.
"""

"""
We begin by importing essential libraries for building and training our neural network
model using TensorFlow and Keras. TensorFlow is a powerful deep learning library widely
used for building various machine learning models, while Keras provides a high-level API
simplifying the process of building neural networks. Additionally, we import NumPy for
numerical computations and matplotlib for data visualization purposes.

Furthermore, we define several hyperparameters essential for training our language model.
These parameters include batch size, determining the number of independent sequences
processed simultaneously during training, and block size, representing the maximum
context length for predictions. Other parameters such as the number of iterations,
learning rate, embedding size, number of heads, number of layers, and dropout rate are
also specified. Setting seeds for NumPy and TensorFlow ensures reproducibility of results
across different runs, a crucial aspect in machine learning experimentation.
"""

# Importing Libraries

import tensorflow as tf
import keras
from keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100

Epochs = max_iters // eval_interval

learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------


"""
# Dataset prepration
"""

"""
In next step we should prepare data, therefore we download the `Tiny Shakespeare`
dataset, a popular choice for language modeling tasks, to preprocess it for training a
language model. The dataset is read from a text file, and we extract unique characters
from it to construct the vocabulary. Using these characters, we create mappings from
characters to integers and vice versa, facilitating the encoding and decoding of text
data into numerical form, which is essential for training neural networks. Subsequently,
we split the dataset into training and validation sets, with 90% of the data allocated
for training and the remaining 10% for validation.

Following data preparation, we define functions to generate batches of data for training
and validation. These functions enable the creation of input-output pairs, where input
sequences serve as context for predicting subsequent characters. Utilizing TensorFlow's
data processing capabilities, we convert these batch generation functions into TensorFlow
datasets, ensuring seamless integration with TensorFlow's training pipeline. These
datasets are structured to produce batches of sequences, each consisting of a fixed
number of characters, which will be used to train and validate our language model
efficiently.
"""
"""
### Download and prepare dataset
"""
"""shell
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

# Read the text file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = np.array(encode(text), dtype=np.int64)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Prepare train/val dataset
def train_data_generator():
    while True:
        yield get_batch("train")


def val_data_generator():
    while True:
        yield get_batch("val")


train_data_generator = tf.data.Dataset.from_generator(
    train_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
    ),
)

val_data_generator = tf.data.Dataset.from_generator(
    val_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
    ),
)

"""
# Language model Architucture
"""

"""
Our languge model is constructed based on 3 main parts/class. The first class,
`FeedForward`, represents a simple feedforward neural network layer, a fundamental
component of many deep learning architectures. In its `__init__` method, it initializes a
sequential model consisting of two dense layers with ReLU activation functions and a
dropout layer. The `call` method executes a forward pass through this network, taking an
input tensor `x` and passing it through the sequential model, returning the output
tensor.

The `Block` class represents a single block of a transformer architecture. In its
`__init__` method, it initializes the block with a multi-head self-attention layer
(`sa`), a feedforward neural network layer (`ffwd`), and layer normalization layers
(`ln1` and `ln2`). The `call` method executes a forward pass through this block. It first
applies self-attention to the input tensor `x`, then adds the output to the input tensor,
and passes it through the feedforward neural network layer. Finally, it returns the
output tensor.
It should be noted that, for language modeling we use `decoder attention` block and it
has triangular masking that provides autoregressive settings and allows tokens to
cominucate only with pervious tokens, you can see the explination
[here](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4454s). when calling Keras
MultiHeadAttention layer in `Block` class, we set `use_causal_mask=True` for this reason.

The `BigramLanguageModel` class represents the entire language model architecture. It
consists of embedding layers for token and positional embeddings, multiple transformer
blocks, layer normalization, and a dense layer for output logits. The `call` method
executes a forward pass through the model, applying token and positional embeddings,
passing the input through the transformer blocks, and generating logits for the next
token. It also computes the loss if targets are provided. Additionally, it includes
methods `train_step` and `test_step` for training and evaluation steps, respectively, and
a `generate` method for generating text given an input sequence.
"""


# %% Model components
class FeedForward(layers.Layer):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = models.Sequential(
            [
                layers.Dense(4 * n_embd, activation="relu"),
                layers.Dense(n_embd),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        return self.net(x)


class Block(layers.Layer):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        self.sa = layers.MultiHeadAttention(
            num_heads=n_head, key_dim=n_embd // n_head, dropout=dropout
        )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.sa(
            self.ln1(x), self.ln1(x), use_causal_mask=True
        )  # use causal mask to ensure each token can only see previous tokens
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram Language Model
class BigramLanguageModel(keras.Model):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            tf.range(T)[tf.newaxis, :]
        )  # initially (T,C) adding new axis and get # (1,T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        for block in self.blocks:  # (B,T,C)
            x = block(x)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            return logits, None

        logits_flat = tf.reshape(logits, [-1, logits.shape[-1]])
        targets_flat = tf.reshape(targets, [-1])
        loss = keras.losses.sparse_categorical_crossentropy(
            targets_flat, logits_flat, from_logits=True
        )
        return logits, tf.reduce_mean(loss)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, loss = self(x, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        logits, loss = self(x, y)
        return {"loss": loss}

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # sample from the distribution
            idx_next = tf.random.categorical(logits, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = tf.concat([idx, idx_next], axis=1)  # (B, T+1)
        return idx


"""
# Model training
"""

"""
Now its time to initialize the language model (BigramLanguageModel) and see the number of
trainable parameters in the model using the count_params() method. This gives us insight
into the complexity of the model and the memory requirements for training.

After initializing the model, we compile it using the Adam optimizer with a specified
learning rate. Compilation involves setting up the model for training, including
specifying the loss function and optimization algorithm.

Next, we train the model using the fit method. We specify the training data generator
(train_data_generator), the number of epochs (Epochs), the evaluation interval
(eval_interval), the validation data generator (val_data_generator), and the number of
validation steps (eval_iters). During training, the model learns to minimize the loss
function on the training data while monitoring its performance on the validation data.

Finally, we plot the learning curves using Matplotlib. The learning curves show the
training and validation loss as a function of the number of epochs. This visualization
helps us understand how well the model is learning over time and whether it is
overfitting or underfitting. By observing the trends in the loss curves, we can make
informed decisions about model training and optimization.
"""

# Initialize the model train and plotting loss curves
model = BigramLanguageModel()
# print the number of parameters in the model
model.build((batch_size, block_size))
print("Number of trainable parameters:", model.count_params())

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate))

# Train the model
Hist = model.fit(
    train_data_generator,
    epochs=Epochs,
    steps_per_epoch=eval_interval,
    validation_data=val_data_generator,
    validation_steps=eval_iters,
)

# Plot learning curve
plt.figure()
plt.plot(
    np.arange(1, Epochs + 1),
    np.vstack((Hist.history["loss"], Hist.history["val_loss"])).T,
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train_loss", "val_loss"])

"""
# Geneating Shakespeare-like text!
"""

"""
And finally, we generate text using the trained language model (BigramLanguageModel). We
initialize the generation process by providing an initial context, which is represented
as an array of zeros with shape (1, 1). This context serves as the starting point for
text generation.

We then use the generate method of the model to generate a sequence of tokens. We specify
the maximum number of new tokens to generate (max_new_tokens) as 2000. The model
iteratively predicts the next token based on the provided context and appends it to the
sequence.

Once the text generation process is complete, we decode the generated sequence of tokens
into human-readable text using the decode function. This function maps each token index
back to its corresponding character in the vocabulary.

Finally, we print the generated text to the console, allowing us to inspect the output of
the language model and assess its quality. This text generation process demonstrates the
model's ability to generate coherent and contextually relevant text based on the patterns
learned during training.
"""

# Generate text
context = np.zeros((1, 1), dtype=np.int64)
generated = model.generate(context, max_new_tokens=2000)
print(decode(generated[0].numpy().tolist()))
