# Float8 training and inference with a simple Transformer model

**Author:** [Hongyu Chiu](https://github.com/james77777778)<br>
**Date created:** 2024/05/14<br>
**Last modified:** 2024/05/14<br>
**Description:** Train a simple Transformer model with the float8 quantization.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/float8_training_and_inference_with_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/float8_training_and_inference_with_transformer.py)



---
## Introduction

As the number of parameters in Transformer models continues to grow, training
and inference become highly memory and compute-intensive. Therefore, 8-bit
floating point (FP8) was introduced, offering improved performance over 16-bit
floating point with nearly no degradation in accuracy.

In detail, there are two distinct types of FP8: E4M3 and E5M2, useful in
different parts of training.
- E4M3: It consists of 1 sign bit, 4 exponent bits and 3 bits of mantissa. It
    can store values up to +/-448 and nan.
- E5M2: It consists of 1 sign bit, 5 exponent bits and 2 bits of mantissa. It
    can store values up to +/-57344, +/-inf and nan. The tradeoff of the
    increased dynamic range is lower precision of the stored values.
Typically, E4M3 is best used during the forward pass because activations and
weights require more precision. In the backward pass, however, E5M2 is utilized
because gradients are less susceptible to the loss of precision but require
higher dynamic range.

It is worth noting that FP8 inference deployment is greatly simplified, as
inference and training use the same datatype. This is in contrast to INT8
inference with networks trained in 32- or 16-bit floating point, which require
post-training quantization (PTQ) calibration and even quantization-aware
training (QAT) in order to maintain model accuracy.

In this example, we will build a simple Transformer model and train it with
both FP16 and FP8 precision. You will observe that the accuracy doesn't decrease
with lower precision.

Note: You will need a decent GPU with FP8 Tensor Cores support for the expected
performance improvement.

---
## Setup

We will use KerasNLP library to simplify the model implementation. Additionally,
use mixed precision training to reduce the training time.

Note: The dependency on TensorFlow is only required for data processing.


```python
!pip install -q --upgrade git+https://github.com/keras-team/keras-nlp.git  # Get the latest version of KerasNLP
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import keras
import keras_nlp
import tensorflow as tf

keras.config.set_dtype_policy("mixed_bfloat16")
```

Define some hyperparameters.


```python
EPOCHS = 3
BATCH_SIZE = 32
VOCABULARY_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
MODEL_KWARGS = dict(
    vocabulary_size=VOCABULARY_SIZE,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    hidden_dim=32,  # Hidden size for each token
    num_heads=2,  # Number of attention heads
    intermediate_dim=32,  # Intermediate size in feedforward network
    dropout=0.1,  # Dropout rate
)
```

---
## Dataset

First, let's download the IMDB dataset and extract it.


```python
!mkdir -p datasets
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O datasets/aclImdb_v1.tar.gz
!mkdir -p datasets/aclImdb
!tar -xzf datasets/aclImdb_v1.tar.gz -C datasets
!rm -rf datasets/aclImdb/train/unsup
```

<div class="k-default-codeblock">
```
--2024-05-20 10:21:19--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... 

connected.
HTTP request sent, awaiting response... 

200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘datasets/aclImdb_v1.tar.gz’
2024-05-20 10:38:01 (82.0 KB/s) - ‘datasets/aclImdb_v1.tar.gz’ saved [84125825/84125825]
```
</div>

We'll use the `keras.utils.text_dataset_from_directory` utility to generate our
labelled `tf.data.Dataset` dataset from text files.


```python
train_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/test", batch_size=BATCH_SIZE
)
```

<div class="k-default-codeblock">
```
Found 25000 files belonging to 2 classes.

Using 20000 files for training.

Found 25000 files belonging to 2 classes.

Using 5000 files for validation.

Found 25000 files belonging to 2 classes.

```
</div>
We will now convert the text to lowercase.


```python
train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))
```

Let's print a few samples.


```python
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f"Text: {text_batch.numpy()[i]}")
        print(f"Label: {label_batch.numpy()[i]}")
```

<div class="k-default-codeblock">
```
Text: b'"pandemonium" is a horror movie spoof that comes off more stupid than funny. believe me when i tell you, i love comedies. especially comedy spoofs. "airplane", "the naked gun" trilogy, "blazing saddles", "high anxiety", and "spaceballs" are some of my favorite comedies that spoof a particular genre. "pandemonium" is not up there with those films. most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\'t all that funny. there are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\'s all this film has going for it. geez, "scream" had more laughs than this film and that was more of a horror film. how bizarre is that?<br /><br />*1/2 (out of four)'
Label: 0
Text: b"david mamet is a very interesting and a very un-equal director. his first movie 'house of games' was the one i liked best, and it set a series of films with characters whose perspective of life changes as they get into complicated situations, and so does the perspective of the viewer.<br /><br />so is 'homicide' which from the title tries to set the mind of the viewer to the usual crime drama. the principal characters are two cops, one jewish and one irish who deal with a racially charged area. the murder of an old jewish shop owner who proves to be an ancient veteran of the israeli independence war triggers the jewish identity in the mind and heart of the jewish detective.<br /><br />this is were the flaws of the film are the more obvious. the process of awakening is theatrical and hard to believe, the group of jewish militants is operatic, and the way the detective eventually walks to the final violent confrontation is pathetic. the end of the film itself is mamet-like smart, but disappoints from a human emotional perspective.<br /><br />joe mantegna and william macy give strong performances, but the flaws of the story are too evident to be easily compensated."
Label: 0
Text: b'great documentary about the lives of ny firefighters during the worst terrorist attack of all time.. that reason alone is why this should be a must see collectors item.. what shocked me was not only the attacks, but the"high fat diet" and physical appearance of some of these firefighters. i think a lot of doctors would agree with me that,in the physical shape they were in, some of these firefighters would not of made it to the 79th floor carrying over 60 lbs of gear. having said that i now have a greater respect for firefighters and i realize becoming a firefighter is a life altering job. the french have a history of making great documentary\'s and that is what this is, a great documentary.....'
Label: 1

```
</div>
### Tokenizing the data

We'll be using the `keras_nlp.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_nlp.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization
algorithm; training it on a corpus gives us a vocabulary of subwords. A subword
tokenizer is a compromise between word tokenizers (word tokenizers need very
large vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, KerasNLP
makes it very simple to train WordPiece on a corpus with the
`keras_nlp.tokenizers.compute_word_piece_vocabulary` utility.


```python

def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

```

Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence
length when the input sequence length is shorter than the maximum sequence
length.
- `"[UNK]"` - Unknown token.


```python
reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
vocab = train_word_piece(train_ds, VOCABULARY_SIZE, reserved_tokens)
```

Let's see some tokens!


```python
print("Tokens: ", vocab[100:110])
```

<div class="k-default-codeblock">
```
Tokens:  ['à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é']

```
</div>
Now, let's define the tokenizer. We will configure the tokenizer with the
the vocabularies trained above. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.


```python
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)
```

Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.


```python
input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))
```

<div class="k-default-codeblock">
```
Sentence:  tf.Tensor(b'great movie - especially the music - etta james - "at last". this speaks volumes when you have finally found that special someone.', shape=(), dtype=string)
Tokens:  

[  218   150    14   393   137   356    14  4917  2941   719    14     3
   164   370     3    15   145  2705 11670   186   155   160   557   391
   146   452   416    15     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0]
Recovered text after detokenizing:  tf.Tensor(b'great movie - especially the music - etta james - " at last " . this speaks volumes when you have finally found that special someone . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]', shape=(), dtype=string)

```
</div>
---
## Formatting the dataset

Next, we'll format our datasets in the form that will be fed to the models. We
need to tokenize the text.


```python

def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(tf.data.AUTOTUNE).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)

```

---
## Model

Let's build a simple Transformer model. We will use `TokenAndPositionEmbedding`
and `TransformerDecoder` from KerasNLP library. `TokenAndPositionEmbedding`
represents words and their order in a sentence, while `TransformerDecoder`
outputs one vector for each time step of our input sequence. Here, we take the
mean across all time steps and use a feedforward network on top of it to
classify text.


```python

def build_model(
    vocabulary_size=20000,
    max_sequence_length=200,
    hidden_dim=32,
    num_heads=2,
    intermediate_dim=32,
    dropout=0.1,
):
    token_id_input = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        embedding_dim=hidden_dim,
    )(token_id_input)
    x = keras.layers.Dropout(rate=dropout)(x)
    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=dropout,
    )(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(intermediate_dim, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=token_id_input, outputs=outputs)

```

---
## Training and evaluating our model

First, we train and evaluate the model with mixed precision
(`"mixed_bfloat16"`). Afterward, we compare the results with FP8
training/inference.


```python
model = build_model(**MODEL_KWARGS)
model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (mixed_bfloat16): {result[1]:.2%}")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ token_and_position_embedding    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">646,400</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbedding</span>)     │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ transformer_decoder             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">6,464</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerDecoder</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">653,953</span> (2.49 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">653,953</span> (2.49 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  26:04 3s/step - accuracy: 0.5938 - loss: 0.7035

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  11:06 1s/step - accuracy: 0.5078 - loss: 0.7196

<div class="k-default-codeblock">
```

```
</div>
  56/625 ━[37m━━━━━━━━━━━━━━━━━━━  11s 20ms/step - accuracy: 0.5148 - loss: 0.6960

<div class="k-default-codeblock">
```

```
</div>
 113/625 ━━━[37m━━━━━━━━━━━━━━━━━  5s 10ms/step - accuracy: 0.5447 - loss: 0.6829 

<div class="k-default-codeblock">
```

```
</div>
 172/625 ━━━━━[37m━━━━━━━━━━━━━━━  3s 7ms/step - accuracy: 0.5725 - loss: 0.6647 

<div class="k-default-codeblock">
```

```
</div>
 232/625 ━━━━━━━[37m━━━━━━━━━━━━━  2s 6ms/step - accuracy: 0.5983 - loss: 0.6437

<div class="k-default-codeblock">
```

```
</div>
 296/625 ━━━━━━━━━[37m━━━━━━━━━━━  1s 4ms/step - accuracy: 0.6207 - loss: 0.6231

<div class="k-default-codeblock">
```

```
</div>
 361/625 ━━━━━━━━━━━[37m━━━━━━━━━  1s 4ms/step - accuracy: 0.6386 - loss: 0.6056

<div class="k-default-codeblock">
```

```
</div>
 426/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 3ms/step - accuracy: 0.6533 - loss: 0.5908

<div class="k-default-codeblock">
```

```
</div>
 490/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - accuracy: 0.6655 - loss: 0.5780

<div class="k-default-codeblock">
```

```
</div>
 552/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 3ms/step - accuracy: 0.6759 - loss: 0.5668

<div class="k-default-codeblock">
```

```
</div>
 618/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - accuracy: 0.6855 - loss: 0.5562

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - accuracy: 0.6866 - loss: 0.5550 - val_accuracy: 0.8204 - val_loss: 0.3901


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  11:15 1s/step - accuracy: 0.8750 - loss: 0.2920

<div class="k-default-codeblock">
```

```
</div>
  59/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 870us/step - accuracy: 0.8755 - loss: 0.3043

<div class="k-default-codeblock">
```

```
</div>
 119/625 ━━━[37m━━━━━━━━━━━━━━━━━  0s 857us/step - accuracy: 0.8755 - loss: 0.3023

<div class="k-default-codeblock">
```

```
</div>
 180/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 848us/step - accuracy: 0.8731 - loss: 0.3042

<div class="k-default-codeblock">
```

```
</div>
 241/625 ━━━━━━━[37m━━━━━━━━━━━━━  0s 842us/step - accuracy: 0.8739 - loss: 0.3013

<div class="k-default-codeblock">
```

```
</div>
 302/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 840us/step - accuracy: 0.8760 - loss: 0.2970

<div class="k-default-codeblock">
```

```
</div>
 367/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 829us/step - accuracy: 0.8781 - loss: 0.2932

<div class="k-default-codeblock">
```

```
</div>
 432/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 820us/step - accuracy: 0.8800 - loss: 0.2899

<div class="k-default-codeblock">
```

```
</div>
 496/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 815us/step - accuracy: 0.8815 - loss: 0.2875

<div class="k-default-codeblock">
```

```
</div>
 561/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 810us/step - accuracy: 0.8828 - loss: 0.2854

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 923us/step - accuracy: 0.8840 - loss: 0.2835 - val_accuracy: 0.8078 - val_loss: 0.4478


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 0.9062 - loss: 0.1977

<div class="k-default-codeblock">
```

```
</div>
  66/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 779us/step - accuracy: 0.9125 - loss: 0.2358

<div class="k-default-codeblock">
```

```
</div>
 130/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 783us/step - accuracy: 0.9128 - loss: 0.2306

<div class="k-default-codeblock">
```

```
</div>
 194/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 784us/step - accuracy: 0.9094 - loss: 0.2338

<div class="k-default-codeblock">
```

```
</div>
 258/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 785us/step - accuracy: 0.9096 - loss: 0.2315

<div class="k-default-codeblock">
```

```
</div>
 322/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 785us/step - accuracy: 0.9108 - loss: 0.2284

<div class="k-default-codeblock">
```

```
</div>
 386/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 785us/step - accuracy: 0.9124 - loss: 0.2253

<div class="k-default-codeblock">
```

```
</div>
 452/625 ━━━━━━━━━━━━━━[37m━━━━━━  0s 783us/step - accuracy: 0.9140 - loss: 0.2223

<div class="k-default-codeblock">
```

```
</div>
 517/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 781us/step - accuracy: 0.9151 - loss: 0.2203

<div class="k-default-codeblock">
```

```
</div>
 583/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 780us/step - accuracy: 0.9159 - loss: 0.2188

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 890us/step - accuracy: 0.9163 - loss: 0.2179 - val_accuracy: 0.8388 - val_loss: 0.4165


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  5:01 386ms/step - accuracy: 0.8750 - loss: 0.4444

<div class="k-default-codeblock">
```

```
</div>
  85/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 597us/step - accuracy: 0.8026 - loss: 0.5145  

<div class="k-default-codeblock">
```

```
</div>
 171/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 592us/step - accuracy: 0.8048 - loss: 0.5078

<div class="k-default-codeblock">
```

```
</div>
 253/782 ━━━━━━[37m━━━━━━━━━━━━━━  0s 598us/step - accuracy: 0.8066 - loss: 0.5026

<div class="k-default-codeblock">
```

```
</div>
 370/782 ━━━━━━━━━[37m━━━━━━━━━━━  0s 544us/step - accuracy: 0.8082 - loss: 0.4982

<div class="k-default-codeblock">
```

```
</div>
 483/782 ━━━━━━━━━━━━[37m━━━━━━━━  0s 520us/step - accuracy: 0.8094 - loss: 0.4955

<div class="k-default-codeblock">
```

```
</div>
 600/782 ━━━━━━━━━━━━━━━[37m━━━━━  0s 503us/step - accuracy: 0.8101 - loss: 0.4936

<div class="k-default-codeblock">
```

```
</div>
 712/782 ━━━━━━━━━━━━━━━━━━[37m━━  0s 494us/step - accuracy: 0.8104 - loss: 0.4934

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 1s 491us/step - accuracy: 0.8105 - loss: 0.4934


<div class="k-default-codeblock">
```
Accuracy (mixed_bfloat16): 81.16%

```
</div>
We can enable FP8 training/inference with a one-line API:
`model.quantize("float8")`.


```python
model = build_model(**MODEL_KWARGS)
model.quantize("float8")
```

<div class="k-default-codeblock">
```
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer InputLayer does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer PositionEmbedding does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer ReversibleEmbedding does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer Dropout does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer LayerNormalization does not have a `quantize()` method implemented.
  warnings.warn(str(e))

/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer Softmax does not have a `quantize()` method implemented.
  warnings.warn(str(e))

/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer GlobalAveragePooling1D does not have a `quantize()` method implemented.
  warnings.warn(str(e))

```
</div>
To inspect that FP8 training takes place, we can print out some variables
related to FP8 training:

- `*_scale`: The scaling factor that shift the distribution of inputs, weights
    and gradients into the representable range of FP8. Defaults to `1.0`
- `*_amax_history`: The amax history window used for scaling factor computation.
    Defaults to `0.0` with the length of 1024.


```python
pattern = r"(transformer).+(multi_head).+(query).+(scale|amax_history)"
for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))
```

The dtype policies of FP8 layers have also been modified.


```python
for layer in model._flatten_layers(recursive=True):
    if "float8" in str(layer.dtype_policy):
        print(f"{layer.name}: {layer.dtype_policy}")
```

<div class="k-default-codeblock">
```
feedforward_output_dense: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
feedforward_intermediate_dense: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
attention_output: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
value: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
key: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
query: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
dense_2: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
dense_3: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">

```
</div>
Let's train the model and see the results. We can verify that the accuracy
doesn't decrease with FP8 training that the variables containing FP8 information
change after fitting.


```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (float8): {result[1]:.2%}")

for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))
```

<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  25:46 2s/step - accuracy: 0.4062 - loss: 0.7810

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  16:26 2s/step - accuracy: 0.4844 - loss: 0.7454

<div class="k-default-codeblock">
```

```
</div>
  43/625 ━[37m━━━━━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.5172 - loss: 0.7026

<div class="k-default-codeblock">
```

```
</div>
  85/625 ━━[37m━━━━━━━━━━━━━━━━━━  10s 20ms/step - accuracy: 0.5261 - loss: 0.6979

<div class="k-default-codeblock">
```

```
</div>
 127/625 ━━━━[37m━━━━━━━━━━━━━━━━  6s 14ms/step - accuracy: 0.5379 - loss: 0.6923 

<div class="k-default-codeblock">
```

```
</div>
 169/625 ━━━━━[37m━━━━━━━━━━━━━━━  4s 11ms/step - accuracy: 0.5548 - loss: 0.6818

<div class="k-default-codeblock">
```

```
</div>
 211/625 ━━━━━━[37m━━━━━━━━━━━━━━  3s 9ms/step - accuracy: 0.5726 - loss: 0.6683 

<div class="k-default-codeblock">
```

```
</div>
 253/625 ━━━━━━━━[37m━━━━━━━━━━━━  2s 7ms/step - accuracy: 0.5894 - loss: 0.6536

<div class="k-default-codeblock">
```

```
</div>
 297/625 ━━━━━━━━━[37m━━━━━━━━━━━  2s 7ms/step - accuracy: 0.6045 - loss: 0.6398

<div class="k-default-codeblock">
```

```
</div>
 342/625 ━━━━━━━━━━[37m━━━━━━━━━━  1s 6ms/step - accuracy: 0.6179 - loss: 0.6270

<div class="k-default-codeblock">
```

```
</div>
 386/625 ━━━━━━━━━━━━[37m━━━━━━━━  1s 5ms/step - accuracy: 0.6294 - loss: 0.6157

<div class="k-default-codeblock">
```

```
</div>
 431/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 5ms/step - accuracy: 0.6398 - loss: 0.6052

<div class="k-default-codeblock">
```

```
</div>
 475/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - accuracy: 0.6489 - loss: 0.5958

<div class="k-default-codeblock">
```

```
</div>
 520/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 4ms/step - accuracy: 0.6574 - loss: 0.5869

<div class="k-default-codeblock">
```

```
</div>
 565/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 0.6651 - loss: 0.5786

<div class="k-default-codeblock">
```

```
</div>
 610/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 4ms/step - accuracy: 0.6722 - loss: 0.5709

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 6s 5ms/step - accuracy: 0.6745 - loss: 0.5684 - val_accuracy: 0.8140 - val_loss: 0.4036


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  16:12 2s/step - accuracy: 0.7812 - loss: 0.3850

<div class="k-default-codeblock">
```

```
</div>
  41/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8619 - loss: 0.3272  

<div class="k-default-codeblock">
```

```
</div>
  83/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8637 - loss: 0.3236

<div class="k-default-codeblock">
```

```
</div>
 125/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8636 - loss: 0.3212

<div class="k-default-codeblock">
```

```
</div>
 167/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8641 - loss: 0.3186

<div class="k-default-codeblock">
```

```
</div>
 209/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8659 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 251/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8686 - loss: 0.3091

<div class="k-default-codeblock">
```

```
</div>
 295/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8710 - loss: 0.3045

<div class="k-default-codeblock">
```

```
</div>
 340/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 1ms/step - accuracy: 0.8732 - loss: 0.3009

<div class="k-default-codeblock">
```

```
</div>
 385/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.8752 - loss: 0.2976

<div class="k-default-codeblock">
```

```
</div>
 430/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.8769 - loss: 0.2948

<div class="k-default-codeblock">
```

```
</div>
 475/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.8784 - loss: 0.2924

<div class="k-default-codeblock">
```

```
</div>
 518/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.8795 - loss: 0.2906

<div class="k-default-codeblock">
```

```
</div>
 562/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.8805 - loss: 0.2890

<div class="k-default-codeblock">
```

```
</div>
 607/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.8814 - loss: 0.2874

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.8818 - loss: 0.2869 - val_accuracy: 0.8300 - val_loss: 0.3832


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  2s 3ms/step - accuracy: 0.9062 - loss: 0.2026

<div class="k-default-codeblock">
```

```
</div>
  45/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9078 - loss: 0.2419

<div class="k-default-codeblock">
```

```
</div>
  89/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9122 - loss: 0.2290

<div class="k-default-codeblock">
```

```
</div>
 133/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9123 - loss: 0.2268

<div class="k-default-codeblock">
```

```
</div>
 177/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9115 - loss: 0.2264

<div class="k-default-codeblock">
```

```
</div>
 222/625 ━━━━━━━[37m━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9110 - loss: 0.2257

<div class="k-default-codeblock">
```

```
</div>
 267/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9117 - loss: 0.2227

<div class="k-default-codeblock">
```

```
</div>
 312/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9126 - loss: 0.2203

<div class="k-default-codeblock">
```

```
</div>
 357/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 1ms/step - accuracy: 0.9136 - loss: 0.2181

<div class="k-default-codeblock">
```

```
</div>
 402/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.9148 - loss: 0.2157

<div class="k-default-codeblock">
```

```
</div>
 447/625 ━━━━━━━━━━━━━━[37m━━━━━━  0s 1ms/step - accuracy: 0.9159 - loss: 0.2135

<div class="k-default-codeblock">
```

```
</div>
 492/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.9168 - loss: 0.2119

<div class="k-default-codeblock">
```

```
</div>
 537/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.9173 - loss: 0.2109

<div class="k-default-codeblock">
```

```
</div>
 582/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.9177 - loss: 0.2101

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9181 - loss: 0.2094 - val_accuracy: 0.8400 - val_loss: 0.4063


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 0.9062 - loss: 0.4228

<div class="k-default-codeblock">
```

```
</div>
  80/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 638us/step - accuracy: 0.8027 - loss: 0.5005

<div class="k-default-codeblock">
```

```
</div>
 161/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 630us/step - accuracy: 0.8043 - loss: 0.4972

<div class="k-default-codeblock">
```

```
</div>
 240/782 ━━━━━━[37m━━━━━━━━━━━━━━  0s 633us/step - accuracy: 0.8066 - loss: 0.4921

<div class="k-default-codeblock">
```

```
</div>
 321/782 ━━━━━━━━[37m━━━━━━━━━━━━  0s 631us/step - accuracy: 0.8084 - loss: 0.4878

<div class="k-default-codeblock">
```

```
</div>
 401/782 ━━━━━━━━━━[37m━━━━━━━━━━  0s 630us/step - accuracy: 0.8093 - loss: 0.4852

<div class="k-default-codeblock">
```

```
</div>
 482/782 ━━━━━━━━━━━━[37m━━━━━━━━  0s 629us/step - accuracy: 0.8100 - loss: 0.4832

<div class="k-default-codeblock">
```

```
</div>
 562/782 ━━━━━━━━━━━━━━[37m━━━━━━  0s 629us/step - accuracy: 0.8105 - loss: 0.4815

<div class="k-default-codeblock">
```

```
</div>
 637/782 ━━━━━━━━━━━━━━━━[37m━━━━  0s 634us/step - accuracy: 0.8106 - loss: 0.4809

<div class="k-default-codeblock">
```

```
</div>
 709/782 ━━━━━━━━━━━━━━━━━━[37m━━  0s 641us/step - accuracy: 0.8106 - loss: 0.4807

<div class="k-default-codeblock">
```

```
</div>
 780/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 647us/step - accuracy: 0.8105 - loss: 0.4807

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 1s 650us/step - accuracy: 0.8105 - loss: 0.4807


<div class="k-default-codeblock">
```
Accuracy (float8): 80.98%

```
</div>
---
## Recipes

- The improvements in training speed are relatively small if the model is not
sufficiently large. The recommendation is to train with a model containing
parameters >5B.
- You will need hardware such as NVIDIA H100 that supports FP8 Tensor Cores to
gain the speedups.

---
## References
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Flax - fp8_ops.py](https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py)
