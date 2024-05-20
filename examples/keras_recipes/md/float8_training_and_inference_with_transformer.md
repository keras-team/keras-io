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
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -q -O datasets/aclImdb_v1.tar.gz
!mkdir -p datasets/aclImdb
!tar -xzf datasets/aclImdb_v1.tar.gz -C datasets
!rm -rf datasets/aclImdb/train/unsup
```

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
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  25:20 2s/step - accuracy: 0.6875 - loss: 0.6531

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  10:31 1s/step - accuracy: 0.6484 - loss: 0.6760

<div class="k-default-codeblock">
```

```
</div>
  57/625 ━[37m━━━━━━━━━━━━━━━━━━━  10s 19ms/step - accuracy: 0.5475 - loss: 0.6941

<div class="k-default-codeblock">
```

```
</div>
 116/625 ━━━[37m━━━━━━━━━━━━━━━━━  4s 10ms/step - accuracy: 0.5659 - loss: 0.6822 

<div class="k-default-codeblock">
```

```
</div>
 177/625 ━━━━━[37m━━━━━━━━━━━━━━━  2s 7ms/step - accuracy: 0.5911 - loss: 0.6633 

<div class="k-default-codeblock">
```

```
</div>
 239/625 ━━━━━━━[37m━━━━━━━━━━━━━  1s 5ms/step - accuracy: 0.6143 - loss: 0.6419

<div class="k-default-codeblock">
```

```
</div>
 300/625 ━━━━━━━━━[37m━━━━━━━━━━━  1s 4ms/step - accuracy: 0.6334 - loss: 0.6228

<div class="k-default-codeblock">
```

```
</div>
 365/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 4ms/step - accuracy: 0.6500 - loss: 0.6054

<div class="k-default-codeblock">
```

```
</div>
 430/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 3ms/step - accuracy: 0.6640 - loss: 0.5900

<div class="k-default-codeblock">
```

```
</div>
 495/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - accuracy: 0.6759 - loss: 0.5765

<div class="k-default-codeblock">
```

```
</div>
 561/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 3ms/step - accuracy: 0.6861 - loss: 0.5646

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6946 - loss: 0.5545

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.6948 - loss: 0.5544 - val_accuracy: 0.8182 - val_loss: 0.4013


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  10:32 1s/step - accuracy: 0.9062 - loss: 0.2136

<div class="k-default-codeblock">
```

```
</div>
  59/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 862us/step - accuracy: 0.8786 - loss: 0.2980

<div class="k-default-codeblock">
```

```
</div>
 120/625 ━━━[37m━━━━━━━━━━━━━━━━━  0s 841us/step - accuracy: 0.8775 - loss: 0.2995

<div class="k-default-codeblock">
```

```
</div>
 182/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 832us/step - accuracy: 0.8789 - loss: 0.2960

<div class="k-default-codeblock">
```

```
</div>
 244/625 ━━━━━━━[37m━━━━━━━━━━━━━  0s 828us/step - accuracy: 0.8812 - loss: 0.2909

<div class="k-default-codeblock">
```

```
</div>
 304/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 830us/step - accuracy: 0.8835 - loss: 0.2864

<div class="k-default-codeblock">
```

```
</div>
 364/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 832us/step - accuracy: 0.8853 - loss: 0.2828

<div class="k-default-codeblock">
```

```
</div>
 427/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 827us/step - accuracy: 0.8869 - loss: 0.2795

<div class="k-default-codeblock">
```

```
</div>
 491/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 823us/step - accuracy: 0.8883 - loss: 0.2766

<div class="k-default-codeblock">
```

```
</div>
 557/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 816us/step - accuracy: 0.8895 - loss: 0.2741

<div class="k-default-codeblock">
```

```
</div>
 623/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 811us/step - accuracy: 0.8903 - loss: 0.2724

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 922us/step - accuracy: 0.8904 - loss: 0.2723 - val_accuracy: 0.8206 - val_loss: 0.4375


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 1.0000 - loss: 0.0817

<div class="k-default-codeblock">
```

```
</div>
  67/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 762us/step - accuracy: 0.9290 - loss: 0.1918

<div class="k-default-codeblock">
```

```
</div>
 133/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 761us/step - accuracy: 0.9257 - loss: 0.2013

<div class="k-default-codeblock">
```

```
</div>
 199/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 761us/step - accuracy: 0.9257 - loss: 0.2008

<div class="k-default-codeblock">
```

```
</div>
 264/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 763us/step - accuracy: 0.9244 - loss: 0.2035

<div class="k-default-codeblock">
```

```
</div>
 330/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 764us/step - accuracy: 0.9235 - loss: 0.2046

<div class="k-default-codeblock">
```

```
</div>
 392/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 772us/step - accuracy: 0.9233 - loss: 0.2049

<div class="k-default-codeblock">
```

```
</div>
 455/625 ━━━━━━━━━━━━━━[37m━━━━━━  0s 776us/step - accuracy: 0.9233 - loss: 0.2048

<div class="k-default-codeblock">
```

```
</div>
 518/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 779us/step - accuracy: 0.9235 - loss: 0.2045

<div class="k-default-codeblock">
```

```
</div>
 580/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 782us/step - accuracy: 0.9238 - loss: 0.2039

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 901us/step - accuracy: 0.9240 - loss: 0.2036 - val_accuracy: 0.7998 - val_loss: 0.4888


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  4:45 366ms/step - accuracy: 0.8125 - loss: 0.4856

<div class="k-default-codeblock">
```

```
</div>
  97/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 521us/step - accuracy: 0.7714 - loss: 0.5504  

<div class="k-default-codeblock">
```

```
</div>
 183/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 552us/step - accuracy: 0.7668 - loss: 0.5699

<div class="k-default-codeblock">
```

```
</div>
 286/782 ━━━━━━━[37m━━━━━━━━━━━━━  0s 528us/step - accuracy: 0.7645 - loss: 0.5787

<div class="k-default-codeblock">
```

```
</div>
 399/782 ━━━━━━━━━━[37m━━━━━━━━━━  0s 504us/step - accuracy: 0.7631 - loss: 0.5844

<div class="k-default-codeblock">
```

```
</div>
 512/782 ━━━━━━━━━━━━━[37m━━━━━━━  0s 491us/step - accuracy: 0.7618 - loss: 0.5884

<div class="k-default-codeblock">
```

```
</div>
 629/782 ━━━━━━━━━━━━━━━━[37m━━━━  0s 479us/step - accuracy: 0.7610 - loss: 0.5913

<div class="k-default-codeblock">
```

```
</div>
 748/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 470us/step - accuracy: 0.7603 - loss: 0.5933

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 1s 471us/step - accuracy: 0.7601 - loss: 0.5939


<div class="k-default-codeblock">
```
Accuracy (mixed_bfloat16): 75.56%

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
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  24:44 2s/step - accuracy: 0.6250 - loss: 0.6934

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  16:09 2s/step - accuracy: 0.6016 - loss: 0.6952

<div class="k-default-codeblock">
```

```
</div>
  44/625 ━[37m━━━━━━━━━━━━━━━━━━━  21s 37ms/step - accuracy: 0.5326 - loss: 0.6933

<div class="k-default-codeblock">
```

```
</div>
  86/625 ━━[37m━━━━━━━━━━━━━━━━━━  10s 19ms/step - accuracy: 0.5433 - loss: 0.6878

<div class="k-default-codeblock">
```

```
</div>
 129/625 ━━━━[37m━━━━━━━━━━━━━━━━  6s 13ms/step - accuracy: 0.5625 - loss: 0.6765 

<div class="k-default-codeblock">
```

```
</div>
 171/625 ━━━━━[37m━━━━━━━━━━━━━━━  4s 10ms/step - accuracy: 0.5826 - loss: 0.6606

<div class="k-default-codeblock">
```

```
</div>
 212/625 ━━━━━━[37m━━━━━━━━━━━━━━  3s 9ms/step - accuracy: 0.6002 - loss: 0.6447 

<div class="k-default-codeblock">
```

```
</div>
 254/625 ━━━━━━━━[37m━━━━━━━━━━━━  2s 7ms/step - accuracy: 0.6160 - loss: 0.6295

<div class="k-default-codeblock">
```

```
</div>
 298/625 ━━━━━━━━━[37m━━━━━━━━━━━  2s 6ms/step - accuracy: 0.6307 - loss: 0.6149

<div class="k-default-codeblock">
```

```
</div>
 343/625 ━━━━━━━━━━[37m━━━━━━━━━━  1s 6ms/step - accuracy: 0.6434 - loss: 0.6021

<div class="k-default-codeblock">
```

```
</div>
 389/625 ━━━━━━━━━━━━[37m━━━━━━━━  1s 5ms/step - accuracy: 0.6548 - loss: 0.5904

<div class="k-default-codeblock">
```

```
</div>
 434/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 5ms/step - accuracy: 0.6646 - loss: 0.5800

<div class="k-default-codeblock">
```

```
</div>
 479/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 4ms/step - accuracy: 0.6733 - loss: 0.5707

<div class="k-default-codeblock">
```

```
</div>
 524/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 4ms/step - accuracy: 0.6811 - loss: 0.5623

<div class="k-default-codeblock">
```

```
</div>
 569/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 0.6880 - loss: 0.5547

<div class="k-default-codeblock">
```

```
</div>
 614/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 4ms/step - accuracy: 0.6942 - loss: 0.5478

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 6s 5ms/step - accuracy: 0.6957 - loss: 0.5460 - val_accuracy: 0.8326 - val_loss: 0.3718


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  15:48 2s/step - accuracy: 0.9062 - loss: 0.2280

<div class="k-default-codeblock">
```

```
</div>
  41/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8789 - loss: 0.2997  

<div class="k-default-codeblock">
```

```
</div>
  82/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8823 - loss: 0.3006

<div class="k-default-codeblock">
```

```
</div>
 123/625 ━━━[37m━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8838 - loss: 0.2984

<div class="k-default-codeblock">
```

```
</div>
 165/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8853 - loss: 0.2943

<div class="k-default-codeblock">
```

```
</div>
 207/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8874 - loss: 0.2895

<div class="k-default-codeblock">
```

```
</div>
 252/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8896 - loss: 0.2847

<div class="k-default-codeblock">
```

```
</div>
 297/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8912 - loss: 0.2807

<div class="k-default-codeblock">
```

```
</div>
 342/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 1ms/step - accuracy: 0.8924 - loss: 0.2779

<div class="k-default-codeblock">
```

```
</div>
 387/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.8934 - loss: 0.2755

<div class="k-default-codeblock">
```

```
</div>
 432/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.8943 - loss: 0.2734

<div class="k-default-codeblock">
```

```
</div>
 477/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.8950 - loss: 0.2715

<div class="k-default-codeblock">
```

```
</div>
 522/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.8957 - loss: 0.2698

<div class="k-default-codeblock">
```

```
</div>
 567/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.8963 - loss: 0.2685

<div class="k-default-codeblock">
```

```
</div>
 612/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.8967 - loss: 0.2675

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.8968 - loss: 0.2673 - val_accuracy: 0.8366 - val_loss: 0.3944


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  1s 3ms/step - accuracy: 0.9375 - loss: 0.1151

<div class="k-default-codeblock">
```

```
</div>
  46/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9290 - loss: 0.1854

<div class="k-default-codeblock">
```

```
</div>
  88/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9287 - loss: 0.1937

<div class="k-default-codeblock">
```

```
</div>
 134/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9281 - loss: 0.1960

<div class="k-default-codeblock">
```

```
</div>
 174/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9288 - loss: 0.1944

<div class="k-default-codeblock">
```

```
</div>
 206/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9294 - loss: 0.1940

<div class="k-default-codeblock">
```

```
</div>
 250/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9289 - loss: 0.1958

<div class="k-default-codeblock">
```

```
</div>
 296/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9286 - loss: 0.1967

<div class="k-default-codeblock">
```

```
</div>
 341/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 1ms/step - accuracy: 0.9285 - loss: 0.1968

<div class="k-default-codeblock">
```

```
</div>
 387/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.9286 - loss: 0.1965

<div class="k-default-codeblock">
```

```
</div>
 433/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.9289 - loss: 0.1959

<div class="k-default-codeblock">
```

```
</div>
 478/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.9291 - loss: 0.1952

<div class="k-default-codeblock">
```

```
</div>
 523/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.9294 - loss: 0.1945

<div class="k-default-codeblock">
```

```
</div>
 568/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.9297 - loss: 0.1938

<div class="k-default-codeblock">
```

```
</div>
 613/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.9298 - loss: 0.1936

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9298 - loss: 0.1936 - val_accuracy: 0.7870 - val_loss: 0.5621


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 0.8125 - loss: 0.5110

<div class="k-default-codeblock">
```

```
</div>
  80/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 638us/step - accuracy: 0.7598 - loss: 0.6264

<div class="k-default-codeblock">
```

```
</div>
 159/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 636us/step - accuracy: 0.7546 - loss: 0.6516

<div class="k-default-codeblock">
```

```
</div>
 240/782 ━━━━━━[37m━━━━━━━━━━━━━━  0s 632us/step - accuracy: 0.7520 - loss: 0.6638

<div class="k-default-codeblock">
```

```
</div>
 321/782 ━━━━━━━━[37m━━━━━━━━━━━━  0s 630us/step - accuracy: 0.7506 - loss: 0.6696

<div class="k-default-codeblock">
```

```
</div>
 402/782 ━━━━━━━━━━[37m━━━━━━━━━━  0s 629us/step - accuracy: 0.7495 - loss: 0.6747

<div class="k-default-codeblock">
```

```
</div>
 482/782 ━━━━━━━━━━━━[37m━━━━━━━━  0s 628us/step - accuracy: 0.7485 - loss: 0.6787

<div class="k-default-codeblock">
```

```
</div>
 563/782 ━━━━━━━━━━━━━━[37m━━━━━━  0s 628us/step - accuracy: 0.7478 - loss: 0.6817

<div class="k-default-codeblock">
```

```
</div>
 644/782 ━━━━━━━━━━━━━━━━[37m━━━━  0s 627us/step - accuracy: 0.7473 - loss: 0.6841

<div class="k-default-codeblock">
```

```
</div>
 724/782 ━━━━━━━━━━━━━━━━━━[37m━━  0s 627us/step - accuracy: 0.7469 - loss: 0.6861

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 0s 629us/step - accuracy: 0.7465 - loss: 0.6876


<div class="k-default-codeblock">
```
Accuracy (float8): 74.16%

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
