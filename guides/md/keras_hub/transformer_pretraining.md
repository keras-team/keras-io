# Pretraining a Transformer from scratch with KerasHub

**Author:** [Matthew Watson](https://github.com/mattdangerw/)<br>
**Date created:** 2022/04/18<br>
**Last modified:** 2023/07/15<br>
**Description:** Use KerasHub to train a Transformer model from scratch.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/transformer_pretraining.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/transformer_pretraining.py)



KerasHub aims to make it easy to build state-of-the-art text processing models. In this
guide, we will show how library components simplify pretraining and fine-tuning a
Transformer model from scratch.

This guide is broken into three parts:

1. *Setup*, task definition, and establishing a baseline.
2. *Pretraining* a Transformer model.
3. *Fine-tuning* the Transformer model on our classification task.

---
## Setup

The following guide uses Keras 3 to work in any of `tensorflow`, `jax` or
`torch`. We select the `jax` backend below, which will give us a particularly
fast train step below, but feel free to mix it up.


```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"


import keras_hub
import tensorflow as tf
import keras
```
<div class="k-default-codeblock">
```

```
</div>
Next up, we can download two datasets.

- [SST-2](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) a text
classification dataset and our "end goal". This dataset is often used to benchmark
language models.
- [WikiText-103](https://paperswithcode.com/dataset/wikitext-103): A medium sized
collection of featured articles from English Wikipedia, which we will use for
pretraining.

Finally, we will download a WordPiece vocabulary, to do sub-word tokenization later on in
this guide.


```python
# Download pretraining data.
keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

# Download finetuning data.
keras.utils.get_file(
    origin="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    extract=True,
)
sst_dir = os.path.expanduser("~/.keras/datasets/SST-2/")

# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-hub/examples/bert/bert_vocab_uncased.txt",
)
```

Next, we define some hyperparameters we will use during training.


```python
# Preprocessing params.
PRETRAINING_BATCH_SIZE = 128
FINETUNING_BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# Training params.
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3
```

### Load data

We load our data with [tf.data](https://www.tensorflow.org/guide/data), which will allow
us to define input pipelines for tokenizing and preprocessing text.


```python
# Load SST-2.
sst_train_ds = tf.data.experimental.CsvDataset(
    sst_dir + "train.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
).batch(FINETUNING_BATCH_SIZE)

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.raw")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)

# Take a peak at the sst-2 dataset.
print(sst_train_ds.unbatch().batch(4).take(1).get_single_element())
```

<div class="k-default-codeblock">
```
(<tf.Tensor: shape=(4,), dtype=string, numpy=
array([b'hide new secretions from the parental units ',
       b'contains no wit , only labored gags ',
       b'that loves its characters and communicates something rather beautiful about human nature ',
       b'remains utterly satisfied to remain the same throughout '],
      dtype=object)>, <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 0, 1, 0], dtype=int32)>)

```
</div>
You can see that our `SST-2` dataset contains relatively short snippets of movie review
text. Our goal is to predict the sentiment of the snippet. A label of 1 indicates
positive sentiment, and a label of 0 negative sentiment.

### Establish a baseline

As a first step, we will establish a baseline of good performance. We don't actually need
KerasHub for this, we can just use core Keras layers.

We will train a simple bag-of-words model, where we learn a positive or negative weight
for each word in our vocabulary. A sample's score is simply the sum of the weights of all
words that are present in the sample.


```python
# This layer will turn our input sentence into a list of 1s and 0s the same size
# our vocabulary, indicating whether a word is present in absent.
multi_hot_layer = keras.layers.TextVectorization(
    max_tokens=4000, output_mode="multi_hot"
)
multi_hot_layer.adapt(sst_train_ds.map(lambda x, y: x))
multi_hot_ds = sst_train_ds.map(lambda x, y: (multi_hot_layer(x), y))
multi_hot_val_ds = sst_val_ds.map(lambda x, y: (multi_hot_layer(x), y))

# We then learn a linear regression over that layer, and that's our entire
# baseline model!

inputs = keras.Input(shape=(4000,), dtype="int32")
outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
baseline_model = keras.Model(inputs, outputs)
baseline_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
baseline_model.fit(multi_hot_ds, validation_data=multi_hot_val_ds, epochs=5)
```

<div class="k-default-codeblock">
```
Epoch 1/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 2s 698us/step - accuracy: 0.6421 - loss: 0.6469 - val_accuracy: 0.7567 - val_loss: 0.5391
Epoch 2/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 493us/step - accuracy: 0.7524 - loss: 0.5392 - val_accuracy: 0.7868 - val_loss: 0.4891
Epoch 3/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 513us/step - accuracy: 0.7832 - loss: 0.4871 - val_accuracy: 0.7991 - val_loss: 0.4671
Epoch 4/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 475us/step - accuracy: 0.7991 - loss: 0.4543 - val_accuracy: 0.8069 - val_loss: 0.4569
Epoch 5/5
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 1s 476us/step - accuracy: 0.8100 - loss: 0.4313 - val_accuracy: 0.8036 - val_loss: 0.4530

<keras.src.callbacks.history.History at 0x7f13902967a0>

```
</div>
A bag-of-words approach can be a fast and surprisingly powerful, especially when input
examples contain a large number of words. With shorter sequences, it can hit a
performance ceiling.

To do better, we would like to build a model that can evaluate words *in context*. Instead
of evaluating each word in a void, we need to use the information contained in the
*entire ordered sequence* of our input.

This runs us into a problem. `SST-2` is very small dataset, and there's simply not enough
example text to attempt to build a larger, more parameterized model that can learn on a
sequence. We would quickly start to overfit and memorize our training set, without any
increase in our ability to generalize to unseen examples.

Enter **pretraining**, which will allow us to learn on a larger corpus, and transfer our
knowledge to the `SST-2` task. And enter **KerasHub**, which will allow us to pretrain a
particularly powerful model, the Transformer, with ease.

---
## Pretraining

To beat our baseline, we will leverage the `WikiText103` dataset, an unlabeled
collection of Wikipedia articles that is much bigger than `SST-2`.

We are going to train a *transformer*, a highly expressive model which will learn
to embed each word in our input as a low dimensional vector. Our wikipedia dataset has no
labels, so we will use an unsupervised training objective called the *Masked Language
Modeling* (MaskedLM) objective.

Essentially, we will be playing a big game of "guess the missing word". For each input
sample we will obscure 25% of our input data, and train our model to predict the parts we
covered up.

### Preprocess data for the MaskedLM task

Our text preprocessing for the MaskedLM task will occur in two stages.

1. Tokenize input text into integer sequences of token ids.
2. Mask certain positions in our input to predict on.

To tokenize, we can use a `keras_hub.tokenizers.Tokenizer` -- the KerasHub building block
for transforming text into sequences of integer token ids.

In particular, we will use `keras_hub.tokenizers.WordPieceTokenizer` which does
*sub-word* tokenization. Sub-word tokenization is popular when training models on large
text corpora. Essentially, it allows our model to learn from uncommon words, while not
requiring a massive vocabulary of every word in our training set.

The second thing we need to do is mask our input for the MaskedLM task. To do this, we can use
`keras_hub.layers.MaskedLMMaskGenerator`, which will randomly select a set of tokens in each
input and mask them out.

The tokenizer and the masking layer can both be used inside a call to
[tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map).
We can use `tf.data` to efficiently pre-compute each batch on the CPU, while our GPU or TPU
works on training with the batch that came before. Because our masking layer will
choose new words to mask each time, each epoch over our dataset will give us a totally
new set of labels to train on.


```python
# Setting sequence_length will trim or pad the token outputs to shape
# (batch_size, SEQ_LENGTH).
tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)
# Setting mask_selection_length will trim or pad the mask outputs to shape
# (batch_size, PREDICTIONS_PER_SEQ).
masker = keras_hub.layers.MaskedLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=MASK_RATE,
    mask_selection_length=PREDICTIONS_PER_SEQ,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
)


def preprocess(inputs):
    inputs = tokenizer(inputs)
    outputs = masker(inputs)
    # Split the masking layer outputs into a (features, labels, and weights)
    # tuple that we can use with keras.Model.fit().
    features = {
        "token_ids": outputs["token_ids"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    weights = outputs["mask_weights"]
    return features, labels, weights


# We use prefetch() to pre-compute preprocessed batches on the fly on the CPU.
pretrain_ds = wiki_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pretrain_val_ds = wiki_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
# The masks will change each time you run the cell.
print(pretrain_val_ds.take(1).get_single_element())
```

<div class="k-default-codeblock">
```
({'token_ids': <tf.Tensor: shape=(128, 128), dtype=int32, numpy=
array([[7570, 7849, 2271, ..., 9673,  103, 7570],
       [7570, 7849,  103, ..., 1007, 1012, 2023],
       [1996, 2034, 3940, ...,    0,    0,    0],
       ...,
       [2076, 1996, 2307, ...,    0,    0,    0],
       [3216,  103, 2083, ...,    0,    0,    0],
       [ 103, 2007, 1045, ...,    0,    0,    0]], dtype=int32)>, 'mask_positions': <tf.Tensor: shape=(128, 32), dtype=int64, numpy=
array([[  5,   6,   7, ..., 118, 120, 126],
       [  2,   3,  14, ..., 105, 106, 113],
       [  4,   9,  10, ...,   0,   0,   0],
       ...,
       [  4,  11,  19, ..., 117, 118,   0],
       [  1,  14,  17, ...,   0,   0,   0],
       [  0,   3,   6, ...,   0,   0,   0]])>}, <tf.Tensor: shape=(128, 32), dtype=int32, numpy=
array([[ 1010,  2124,  2004, ...,  2095, 11300,  1012],
       [ 2271, 13091,  2303, ...,  2029,  2027,  1010],
       [23976,  2007,  1037, ...,     0,     0,     0],
       ...,
       [ 1010,  1996,  1010, ...,  1999,  7511,     0],
       [ 2225,  1998, 10722, ...,     0,     0,     0],
       [ 9794,  1030,  2322, ...,     0,     0,     0]], dtype=int32)>, <tf.Tensor: shape=(128, 32), dtype=float32, numpy=
array([[1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 1., ..., 1., 1., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.]], dtype=float32)>)

```
</div>
The above block sorts our dataset into a `(features, labels, weights)` tuple, which can be
passed directly to `keras.Model.fit()`.

We have two features:

1. `"token_ids"`, where some tokens have been replaced with our mask token id.
2. `"mask_positions"`, which keeps track of which tokens we masked out.

Our labels are simply the ids we masked out.

Because not all sequences will have the same number of masks, we also keep a
`sample_weight` tensor, which removes padded labels from our loss function by giving them
zero weight.

### Create the Transformer encoder

KerasHub provides all the building blocks to quickly build a Transformer encoder.

We use `keras_hub.layers.TokenAndPositionEmbedding` to first embed our input token ids.
This layer simultaneously learns two embeddings -- one for words in a sentence and another
for integer positions in a sentence. The output embedding is simply the sum of the two.

Then we can add a series of `keras_hub.layers.TransformerEncoder` layers. These are the
bread and butter of the Transformer model, using an attention mechanism to attend to
different parts of the input sentence, followed by a multi-layer perceptron block.

The output of this model will be a encoded vector per input token id. Unlike the
bag-of-words model we used as a baseline, this model will embed each token accounting for
the context in which it appeared.


```python
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# Embed our tokens with a positional embedding.
embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM,
)
outputs = embedding_layer(inputs)

# Apply layer normalization and dropout to the embedding.
outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(outputs)
outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

# Add a number of encoder blocks
for i in range(NUM_LAYERS):
    outputs = keras_hub.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        layer_norm_epsilon=NORM_EPSILON,
    )(outputs)

encoder_model = keras.Model(inputs, outputs)
encoder_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_3"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ token_and_position_embedding    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │  <span style="color: #00af00; text-decoration-color: #00af00">7,846,400</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbedding</span>)     │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ layer_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalization</span>)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │    <span style="color: #00af00; text-decoration-color: #00af00">527,104</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncoder</span>)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │    <span style="color: #00af00; text-decoration-color: #00af00">527,104</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncoder</span>)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)          │    <span style="color: #00af00; text-decoration-color: #00af00">527,104</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncoder</span>)            │                           │            │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,428,224</span> (287.73 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,428,224</span> (287.73 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



### Pretrain the Transformer

You can think of the `encoder_model` as it's own modular unit, it is the piece of our
model that we are really interested in for our downstream task. However we still need to
set up the encoder to train on the MaskedLM task; to do that we attach a
`keras_hub.layers.MaskedLMHead`.

This layer will take as one input the token encodings, and as another the positions we
masked out in the original input. It will gather the token encodings we masked, and
transform them back in predictions over our entire vocabulary.

With that, we are ready to compile and run pretraining. If you are running this in a
Colab, note that this will take about an hour. Training Transformer is famously compute
intensive, so even this relatively small Transformer will take some time.


```python
# Create the pretraining model by attaching a masked language model head.
inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "mask_positions": keras.Input(
        shape=(PREDICTIONS_PER_SEQ,), dtype="int32", name="mask_positions"
    ),
}

# Encode the tokens.
encoded_tokens = encoder_model(inputs["token_ids"])

# Predict an output word for each masked input token.
# We use the input token embedding to project from our encoded vectors to
# vocabulary logits, which has been shown to improve training efficiency.
outputs = keras_hub.layers.MaskedLMHead(
    token_embedding=embedding_layer.token_embedding,
    activation="softmax",
)(encoded_tokens, mask_positions=inputs["mask_positions"])

# Define and compile our pretraining model.
pretraining_model = keras.Model(inputs, outputs)
pretraining_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(PRETRAINING_LEARNING_RATE),
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)

# Pretrain the model on our wiki text dataset.
pretraining_model.fit(
    pretrain_ds,
    validation_data=pretrain_val_ds,
    epochs=PRETRAINING_EPOCHS,
)

# Save this base model for further finetuning.
encoder_model.save("encoder_model.keras")
```

<div class="k-default-codeblock">
```
Epoch 1/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 242s 41ms/step - loss: 5.4679 - sparse_categorical_accuracy: 0.1353 - val_loss: 3.4570 - val_sparse_categorical_accuracy: 0.3522
Epoch 2/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 234s 40ms/step - loss: 3.6031 - sparse_categorical_accuracy: 0.3396 - val_loss: 3.0514 - val_sparse_categorical_accuracy: 0.4032
Epoch 3/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 232s 40ms/step - loss: 3.2609 - sparse_categorical_accuracy: 0.3802 - val_loss: 2.8858 - val_sparse_categorical_accuracy: 0.4240
Epoch 4/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 233s 40ms/step - loss: 3.1099 - sparse_categorical_accuracy: 0.3978 - val_loss: 2.7897 - val_sparse_categorical_accuracy: 0.4375
Epoch 5/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 235s 40ms/step - loss: 3.0145 - sparse_categorical_accuracy: 0.4090 - val_loss: 2.7504 - val_sparse_categorical_accuracy: 0.4419
Epoch 6/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 252s 43ms/step - loss: 2.9530 - sparse_categorical_accuracy: 0.4157 - val_loss: 2.6925 - val_sparse_categorical_accuracy: 0.4474
Epoch 7/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 232s 40ms/step - loss: 2.9088 - sparse_categorical_accuracy: 0.4210 - val_loss: 2.6554 - val_sparse_categorical_accuracy: 0.4513
Epoch 8/8
 5857/5857 ━━━━━━━━━━━━━━━━━━━━ 236s 40ms/step - loss: 2.8721 - sparse_categorical_accuracy: 0.4250 - val_loss: 2.6389 - val_sparse_categorical_accuracy: 0.4548

```
</div>
---
## Fine-tuning

After pretraining, we can now fine-tune our model on the `SST-2` dataset. We can
leverage the ability of the encoder we build to predict on words in context to boost 
our performance on the downstream task.

### Preprocess data for classification

Preprocessing for fine-tuning is much simpler than for our pretraining MaskedLM task. We just
tokenize our input sentences and we are ready for training!


```python

def preprocess(sentences, labels):
    return tokenizer(sentences), labels


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
finetune_ds = sst_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
finetune_val_ds = sst_val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Preview a single input example.
print(finetune_val_ds.take(1).get_single_element())
```

<div class="k-default-codeblock">
```
(<tf.Tensor: shape=(32, 128), dtype=int32, numpy=
array([[ 2009,  1005,  1055, ...,     0,     0,     0],
       [ 4895, 10258,  2378, ...,     0,     0,     0],
       [ 4473,  2149,  2000, ...,     0,     0,     0],
       ...,
       [ 1045,  2018,  2000, ...,     0,     0,     0],
       [ 4283,  2000,  3660, ...,     0,     0,     0],
       [ 1012,  1012,  1012, ...,     0,     0,     0]], dtype=int32)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 1, 0], dtype=int32)>)

```
</div>
### Fine-tune the Transformer

To go from our encoded token output to a classification prediction, we need to attach
another "head" to our Transformer model. We can afford to be simple here. We pool
the encoded tokens together, and use a single dense layer to make a prediction.


```python
# Reload the encoder model from disk so we can restart fine-tuning from scratch.
encoder_model = keras.models.load_model("encoder_model.keras", compile=False)

# Take as input the tokenized input.
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype="int32")

# Encode and pool the tokens.
encoded_tokens = encoder_model(inputs)
pooled_tokens = keras.layers.GlobalAveragePooling1D()(encoded_tokens[0])

# Predict an output label.
outputs = keras.layers.Dense(1, activation="sigmoid")(pooled_tokens)

# Define and compile our fine-tuning model.
finetuning_model = keras.Model(inputs, outputs)
finetuning_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(FINETUNING_LEARNING_RATE),
    metrics=["accuracy"],
)

# Finetune the model for the SST-2 task.
finetuning_model.fit(
    finetune_ds,
    validation_data=finetune_val_ds,
    epochs=FINETUNING_EPOCHS,
)
```

<div class="k-default-codeblock">
```
Epoch 1/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 21s 9ms/step - accuracy: 0.7500 - loss: 0.4891 - val_accuracy: 0.8036 - val_loss: 0.4099
Epoch 2/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 16s 8ms/step - accuracy: 0.8826 - loss: 0.2779 - val_accuracy: 0.8482 - val_loss: 0.3964
Epoch 3/3
 2105/2105 ━━━━━━━━━━━━━━━━━━━━ 16s 8ms/step - accuracy: 0.9176 - loss: 0.2066 - val_accuracy: 0.8549 - val_loss: 0.4142

<keras.src.callbacks.history.History at 0x7f12d85c21a0>

```
</div>
Pretraining was enough to boost our performance to 84%, and this is hardly the ceiling
for Transformer models. You may have noticed during pretraining that our validation
performance was still steadily increasing. Our model is still significantly undertrained.
Training for more epochs, training a large Transformer, and training on more unlabeled
text would all continue to boost performance significantly.

One of the key goals of KerasHub is to provide a modular approach to NLP model building.
We have shown one approach to building a Transformer here, but KerasHub supports an ever
growing array of components for preprocessing text and building models. We hope it makes
it easier to experiment on solutions to your natural language problems.
