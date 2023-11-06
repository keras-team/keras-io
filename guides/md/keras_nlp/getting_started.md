# Getting Started with KerasNLP

**Author:** [Jonathan Bischof](https://github.com/jbischof)<br>
**Date created:** 2022/12/15<br>
**Last modified:** 2023/07/01<br>
**Description:** An introduction to the KerasNLP API.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/getting_started.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/getting_started.py)



---
## Introduction

KerasNLP is a natural language processing library that supports users through
their entire development cycle. Our workflows are built from modular components
that have state-of-the-art preset weights and architectures when used
out-of-the-box and are easily customizable when more control is needed.

This library is an extension of the core Keras API; all high-level modules are
[`Layers`](/api/layers/) or [`Models`](/api/models/). If you are familiar with Keras,
congratulations! You already understand most of KerasNLP.

KerasNLP uses the [Keras Core](https://keras.io/keras_core/) library to work
with any of TensorFlow, Pytorch and Jax. In the guide below, we will use the
`jax` backend for training our models, and [tf.data](https://www.tensorflow.org/guide/data)
for efficiently running our input preprocessing. But feel free to mix things up!
This guide runs in TensorFlow or PyTorch backends with zero changes, simply update
the `KERAS_BACKEND` below.

This guide demonstrates our modular approach using a sentiment analysis example at six
levels of complexity:

* Inference with a pretrained classifier
* Fine tuning a pretrained backbone
* Fine tuning with user-controlled preprocessing
* Fine tuning a custom model
* Pretraining a backbone model
* Build and train your own transformer from scratch

Throughout our guide, we use Professor Keras, the official Keras mascot, as a visual
reference for the complexity of the material:

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_evolution.png" alt="drawing" height="250"/>


```python
!pip install -q --upgrade keras-nlp
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras_nlp
import keras_core as keras
```

<div class="k-default-codeblock">
```
Using JAX backend.

```
</div>
---
## API quickstart

Our highest level API is `keras_nlp.models`. These symbols cover the complete user
journey of converting strings to tokens, tokens to dense features, and dense features to
task-specific output. For each `XX` architecture (e.g., `Bert`), we offer the following
modules:

* **Tokenizer**: `keras_nlp.models.XXTokenizer`
  * **What it does**: Converts strings to sequences of token ids.
  * **Why it's important**: The raw bytes of a string are too high dimensional to be useful
    features so we first map them to a small number of tokens, for example `"The quick brown
    fox"` to `["the", "qu", "##ick", "br", "##own", "fox"]`.
  * **Inherits from**: `keras.layers.Layer`.
* **Preprocessor**: `keras_nlp.models.XXPreprocessor`
  * **What it does**: Converts strings to a dictionary of preprocessed tensors consumed by
    the backbone, starting with tokenization.
  * **Why it's important**: Each model uses special tokens and extra tensors to understand
    the input such as delimiting input segments and identifying padding tokens. Padding each
    sequence to the same length improves computational efficiency.
  * **Has a**: `XXTokenizer`.
  * **Inherits from**: `keras.layers.Layer`.
* **Backbone**: `keras_nlp.models.XXBackbone`
  * **What it does**: Converts preprocessed tensors to dense features. *Does not handle
    strings; call the preprocessor first.*
  * **Why it's important**: The backbone distills the input tokens into dense features that
    can be used in downstream tasks. It is generally pretrained on a language modeling task
    using massive amounts of unlabeled data. Transferring this information to a new task is a
    major breakthrough in modern NLP.
  * **Inherits from**: `keras.Model`.
* **Task**: e.g., `keras_nlp.models.XXClassifier`
  * **What it does**: Converts strings to task-specific output (e.g., classification
    probabilities).
  * **Why it's important**: Task models combine string preprocessing and the backbone model
    with task-specific `Layers` to solve a problem such as sentence classification, token
    classification, or text generation. The additional `Layers` must be fine-tuned on labeled
    data.
  * **Has a**: `XXBackbone` and `XXPreprocessor`.
  * **Inherits from**: `keras.Model`.

Here is the modular hierarchy for `BertClassifier` (all relationships are compositional):

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/class_diagram.png" alt="drawing" height="300"/>

All modules can be used independently and have a `from_preset()` method in addition to
the standard constructor that instantiates the class with **preset** architecture and
weights (see examples below).

---
## Data

We will use a running example of sentiment analysis of IMDB movie reviews. In this task,
we use the text to predict whether the review was positive (`label = 1`) or negative
(`label = 0`).

We load the data using `keras.utils.text_dataset_from_directory`, which utilizes the
powerful `tf.data.Dataset` format for examples.


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!# Remove unsupervised examples
!rm -r aclImdb/train/unsup
```

```python
BATCH_SIZE = 16
imdb_train = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
)
imdb_test = keras.utils.text_dataset_from_directory(
    "aclImdb/test",
    batch_size=BATCH_SIZE,
)

# Inspect first review
# Format is (review text tensor, label tensor)
print(imdb_train.unbatch().take(1).get_single_element())

```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  3709k      0  0:00:22  0:00:22 --:--:-- 4677k

Found 25000 files belonging to 2 classes.
Found 25000 files belonging to 2 classes.
(<tf.Tensor: shape=(), dtype=string, numpy=b'John Thaw, of Inspector Morse fame, plays old Tom Oakley in this movie. Tom lives in a tiny English village during 1939 and the start of the Second World War. A bit of a recluse, Tom has not yet recovered from the death of his wife and son while he was serving during the First World War. If you can imagine Inspector Morse old and retired, twice as crochety as when he was a policeman, then you\'ve got Tom Oakley\'s character.<br /><br />Yet this heart of flint is about to melt. London children are evacuated in advance of the blitz. Young William (Willie) Beech is billeted with the protesting Tom. Willie is played to good effect by Nick Robinson.<br /><br />This boy is in need of care with a capital C. Behind in school, still wetting the bed, and unable to read are the smallest of his problems. He comes from a horrific background in London, with a mother who cannot cope, to put it mildly.<br /><br />Slowly, yet steadily, man and boy warm to each other. Tom discovers again his ability to love and care. And the boy learns to accept this love and caring. See Tom and Willie building a bomb shelter at the end of their garden. See Willie\'s joy at what is probably his first ever birthday party thrown by Tom.<br /><br />Not to give away the ending, but Willie is adopted by Tom after much struggle, and the pair begin a new life much richer for their mutual love.<br /><br />In this movie, Thaw and Robinson are following in a long line of movies where man meets boy and develop a mutual love. See the late Dirk Bogarde and Jon Whiteley in "Spanish Gardener". Or Clark Gable and Carlo Angeletti in "It Started in Naples". Or Robert Ulrich and Kenny Vadas in "Captains Courageous". Or Mel Gibson and Nick Stahl in "Man Without a Face".<br /><br />Two points of interest. This is the only appearance of Thaw that I know of where he sings. Only a verse of a hymn, New Jerusalem, but he does sing.<br /><br />Second, young Robinson also starred in a second movie featuring "Tom" in the title, "Tom\'s Midnight Garden", which is based on a classic children\'s novel.'>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)

```
</div>
---
## Inference with a pretrained classifier

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png" alt="drawing" height="250"/>

The highest level module in KerasNLP is a **task**. A **task** is a `keras.Model`
consisting of a (generally pretrained) **backbone** model and task-specific layers.
Here's an example using `keras_nlp.models.BertClassifier`.

**Note**: Outputs are the logits per class (e.g., `[0, 0]` is 50% chance of positive). The output is
[negative, positive] for binary classification.


```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
# Note: batched inputs expected so must wrap string in iterable
classifier.predict(["I love modular workflows in keras-nlp!"])
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 882ms/step

array([[-1.5376465,  1.5407037]], dtype=float32)

```
</div>
All **tasks** have a `from_preset` method that constructs a `keras.Model` instance with
preset preprocessing, architecture and weights. This means that we can pass raw strings
in any format accepted by a `keras.Model` and get output specific to our task.

This particular **preset** is a `"bert_tiny_uncased_en"` **backbone** fine-tuned on
`sst2`, another movie review sentiment analysis (this time from Rotten Tomatoes). We use
the `tiny` architecture for demo purposes, but larger models are recommended for SoTA
performance. For all the task-specific presets available for `BertClassifier`, see
our keras.io [models page](https://keras.io/api/keras_nlp/models/).

Let's evaluate our classifier on the IMDB dataset. You will note we don't need to
call `keras.Model.compile` here. All **task** models like `BertClassifier` ship with
compilation defaults, meaning we can just call `keras.Model.evaluate` directly. You
can always call compile as normal to override these defaults (e.g. to add new metrics).

The output below is [loss, accuracy],


```python
classifier.evaluate(imdb_test)
```

<div class="k-default-codeblock">
```
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.4566 - sparse_categorical_accuracy: 0.7885

[0.46291637420654297, 0.7834799885749817]

```
</div>
Our result is 78% accuracy without training anything. Not bad!

---
## Fine tuning a pretrained BERT backbone

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png" alt="drawing" height="250"/>

When labeled text specific to our task is available, fine-tuning a custom classifier can
improve performance. If we want to predict IMDB review sentiment, using IMDB data should
perform better than Rotten Tomatoes data! And for many tasks, no relevant pretrained model
will be available (e.g., categorizing customer reviews).

The workflow for fine-tuning is almost identical to above, except that we request a
**preset** for the **backbone**-only model rather than the entire classifier. When passed
a **backbone** **preset**, a **task** `Model` will randomly initialize all task-specific
layers in preparation for training. For all the **backbone** presets available for
`BertClassifier`, see our keras.io [models page](https://keras.io/api/keras_nlp/models/).

To train your classifier, use `keras.Model.fit` as with any other
`keras.Model`. As with our inference example, we can rely on the compilation
defaults for the **task** and skip `keras.Model.compile`. As preprocessing is
included, we again pass the raw data.


```python
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
classifier.fit(
    imdb_train,
    validation_data=imdb_test,
    epochs=1,
)
```

<div class="k-default-codeblock">
```
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 19s 11ms/step - loss: 0.5128 - sparse_categorical_accuracy: 0.7350 - val_loss: 0.2974 - val_sparse_categorical_accuracy: 0.8746

<keras_core.src.callbacks.history.History at 0x7f86a0649db0>

```
</div>
Here we see a significant lift in validation accuracy (0.78 -> 0.87) with a single epoch of
training even though the IMDB dataset is much smaller than `sst2`.

---
## Fine tuning with user-controlled preprocessing
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png" alt="drawing" height="250"/>

For some advanced training scenarios, users might prefer direct control over
preprocessing. For large datasets, examples can be preprocessed in advance and saved to
disk or preprocessed by a separate worker pool using `tf.data.experimental.service`. In
other cases, custom preprocessing is needed to handle the inputs.

Pass `preprocessor=None` to the constructor of a **task** `Model` to skip automatic
preprocessing or pass a custom `BertPreprocessor` instead.

### Separate preprocessing from the same preset

Each model architecture has a parallel **preprocessor** `Layer` with its own
`from_preset` constructor. Using the same **preset** for this `Layer` will return the
matching **preprocessor** as the **task**.

In this workflow we train the model over three epochs using `tf.data.Dataset.cache()`,
which computes the preprocessing once and caches the result before fitting begins.

**Note:** we can use `tf.data` for preprocessing while running on the
Jax or PyTorch backend. The input dataset will automatically be converted to
backend native tensor types during fit. In fact, given the efficiency of `tf.data`
for running preprocessing, this is good practice on all backends.


```python
import tensorflow as tf

preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased",
    sequence_length=512,
)

# Apply the preprocessor to every sample of train and test data using `map()`.
# `tf.data.AUTOTUNE` and `prefetch()` are options to tune performance, see
# https://www.tensorflow.org/guide/data_performance for details.

# Note: only call `cache()` if you training data fits in CPU memory!
imdb_train_cached = (
    imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)
imdb_test_cached = (
    imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased", preprocessor=None, num_classes=2
)
classifier.fit(
    imdb_train_cached,
    validation_data=imdb_test_cached,
    epochs=3,
)
```

<div class="k-default-codeblock">
```
Epoch 1/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 18s 11ms/step - loss: 0.5338 - sparse_categorical_accuracy: 0.7117 - val_loss: 0.3015 - val_sparse_categorical_accuracy: 0.8737
Epoch 2/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 15s 9ms/step - loss: 0.2855 - sparse_categorical_accuracy: 0.8829 - val_loss: 0.3053 - val_sparse_categorical_accuracy: 0.8771
Epoch 3/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 15s 9ms/step - loss: 0.2094 - sparse_categorical_accuracy: 0.9215 - val_loss: 0.3238 - val_sparse_categorical_accuracy: 0.8756

<keras_core.src.callbacks.history.History at 0x7f864c4a3b80>

```
</div>
After three epochs, our validation accuracy has only increased to 0.88. This is both a
function of the small size of our dataset and our model. To exceed 90% accuracy, try
larger **presets** such as  `"bert_base_en_uncased"`. For all the **backbone** presets
available for `BertClassifier`, see our keras.io [models page](https://keras.io/api/keras_nlp/models/).

### Custom preprocessing

In cases where custom preprocessing is required, we offer direct access to the
`Tokenizer` class that maps raw strings to tokens. It also has a `from_preset()`
constructor to get the vocabulary matching pretraining.

**Note:** `BertTokenizer` does not pad sequences by default, so the output is
ragged (each sequence has varying length). The `MultiSegmentPacker` below
handles padding these ragged sequences to dense tensor types (e.g. `tf.Tensor`
or `torch.Tensor`).


```python
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
tokenizer(["I love modular workflows!", "Libraries over frameworks!"])

# Write your own packer or use one of our `Layers`
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    # Note: This cannot be longer than the preset's `sequence_length`, and there
    # is no check for a custom preprocessor!
    sequence_length=64,
)


# This function that takes a text sample `x` and its
# corresponding label `y` as input and converts the
# text into a format suitable for input into a BERT model.
def preprocessor(x, y):
    token_ids, segment_ids = packer(tokenizer(x))
    x = {
        "token_ids": token_ids,
        "segment_ids": segment_ids,
        "padding_mask": token_ids != 0,
    }
    return x, y


imdb_train_preprocessed = imdb_train.map(preprocessor, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
imdb_test_preprocessed = imdb_test.map(preprocessor, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# Preprocessed example
print(imdb_train_preprocessed.unbatch().take(1).get_single_element())
```

<div class="k-default-codeblock">
```
({'token_ids': <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([  101,  1996,  2466,  1997, 12311,  5163,  2038,  2042,  4372,
        4095, 21332,  2098,  1999, 10661,  1998,  4654, 27609,  3370,
        2005,  2051,  2041,  1997,  2192,  1010,  1998,  2023,  2143,
        2003,  2053,  6453,  1012,  2054, 21312, 12311,  5163,  2038,
        1037,  4568,  2173,  1999,  2381,  2003,  1996,  3947,  2002,
        2253,  2000,  1999,  2344,  2000,  2130,  1996, 10238,  2114,
        1996, 19809,  5933,  2032,  1012,  2076,  2195,  7465,  1010,
         102], dtype=int32)>, 'segment_ids': <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32)>, 'padding_mask': <tf.Tensor: shape=(64,), dtype=bool, numpy=
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])>}, <tf.Tensor: shape=(), dtype=int32, numpy=1>)

```
</div>
---
## Fine tuning with a custom model
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png" alt="drawing" height="250"/>

For more advanced applications, an appropriate **task** `Model` may not be available. In
this case, we provide direct access to the **backbone** `Model`, which has its own
`from_preset` constructor and can be composed with custom `Layer`s. Detailed examples can
be found at our [transfer learning guide](https://keras.io/guides/transfer_learning/).

A **backbone** `Model` does not include automatic preprocessing but can be paired with a
matching **preprocessor** using the same **preset** as shown in the previous workflow.

In this workflow, we experiment with freezing our backbone model and adding two trainable
transformer layers to adapt to the new input.

**Note**: We can ignore the warning about gradients for the `pooled_dense` layer because
we are using BERT's sequence output.


```python
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")

imdb_train_preprocessed = (
    imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)
imdb_test_preprocessed = (
    imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)

backbone.trainable = False
inputs = backbone.input
sequence = backbone(inputs)["sequence_output"]
for _ in range(2):
    sequence = keras_nlp.layers.TransformerEncoder(
        num_heads=2,
        intermediate_dim=512,
        dropout=0.1,
    )(sequence)
# Use [CLS] token output to classify
outputs = keras.layers.Dense(2)(sequence[:, backbone.cls_token_index, :])

model = keras.Model(inputs, outputs)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(5e-5),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,
)
model.summary()
model.fit(
    imdb_train_preprocessed,
    validation_data=imdb_test_preprocessed,
    epochs=3,
)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold"> Param # </span>┃<span style="font-weight: bold"> Connected to         </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ segment_ids         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ bert_backbone_3     │ [(<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>),     │ <span style="color: #00af00; text-decoration-color: #00af00">4,385,…</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BertBackbone</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>,      │         │ segment_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)]             │         │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ transformer_encoder │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │ <span style="color: #00af00; text-decoration-color: #00af00">198,272</span> │ bert_backbone_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncode…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ transformer_encode… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │ <span style="color: #00af00; text-decoration-color: #00af00">198,272</span> │ transformer_encoder… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncode…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ get_item_4          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ transformer_encoder… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)         │     <span style="color: #00af00; text-decoration-color: #00af00">258</span> │ get_item_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,782,722</span> (145.96 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">396,802</span> (12.11 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,385,920</span> (133.85 MB)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 23s 14ms/step - loss: 0.6078 - sparse_categorical_accuracy: 0.6726 - val_loss: 0.5193 - val_sparse_categorical_accuracy: 0.7432
Epoch 2/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 19s 12ms/step - loss: 0.5087 - sparse_categorical_accuracy: 0.7498 - val_loss: 0.4267 - val_sparse_categorical_accuracy: 0.8032
Epoch 3/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 19s 12ms/step - loss: 0.4424 - sparse_categorical_accuracy: 0.7942 - val_loss: 0.3937 - val_sparse_categorical_accuracy: 0.8229

<keras_core.src.callbacks.history.History at 0x7f860c194ac0>

```
</div>
This model achieves reasonable accuracy despite having only 10% of the trainable parameters
of our `BertClassifier` model. Each training step takes about 1/3 of the time---even
accounting for cached preprocessing.

---
## Pretraining a backbone model
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_expert.png" alt="drawing" height="250"/>

Do you have access to large unlabeled datasets in your domain? Are they around the
same size as used to train popular backbones such as BERT, RoBERTa, or GPT2 (XX+ GiB)? If
so, you might benefit from domain-specific pretraining of your own backbone models.

NLP models are generally pretrained on a language modeling task, predicting masked words
given the visible words in an input sentence. For example, given the input
`"The fox [MASK] over the [MASK] dog"`, the model might be asked to predict `["jumped", "lazy"]`.
The lower layers of this model are then packaged as a **backbone** to be combined with
layers relating to a new task.

The KerasNLP library offers SoTA **backbones** and **tokenizers** to be trained from
scratch without presets.

In this workflow, we pretrain a BERT **backbone** using our IMDB review text. We skip the
"next sentence prediction" (NSP) loss because it adds significant complexity to the data
processing and was dropped by later models like RoBERTa. See our e2e
[Transformer pretraining](https://keras.io/guides/keras_nlp/transformer_pretraining/#pretraining)
for step-by-step details on how to replicate the original paper.

### Preprocessing


```python
# All BERT `en` models have the same vocabulary, so reuse preprocessor from
# "bert_tiny_en_uncased"
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased",
    sequence_length=256,
)
packer = preprocessor.packer
tokenizer = preprocessor.tokenizer

# keras.Layer to replace some input tokens with the "[MASK]" token
masker = keras_nlp.layers.MaskedLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=0.25,
    mask_selection_length=64,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
    unselectable_token_ids=[
        tokenizer.token_to_id(x) for x in ["[CLS]", "[PAD]", "[SEP]"]
    ],
)


def preprocess(inputs, label):
    inputs = preprocessor(inputs)
    masked_inputs = masker(inputs["token_ids"])
    # Split the masking layer outputs into a (features, labels, and weights)
    # tuple that we can use with keras.Model.fit().
    features = {
        "token_ids": masked_inputs["token_ids"],
        "segment_ids": inputs["segment_ids"],
        "padding_mask": inputs["padding_mask"],
        "mask_positions": masked_inputs["mask_positions"],
    }
    labels = masked_inputs["mask_ids"]
    weights = masked_inputs["mask_weights"]
    return features, labels, weights


pretrain_ds = imdb_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
pretrain_val_ds = imdb_test.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Tokens with ID 103 are "masked"
print(pretrain_ds.unbatch().take(1).get_single_element())
```

<div class="k-default-codeblock">
```
({'token_ids': <tf.Tensor: shape=(256,), dtype=int32, numpy=
array([  101,  1996,   103,  5236,  5195,  1012,  1045,   103,  1996,
        4364,  5613,  2012,  1996,  2927,   103,  2028,   103,  7112,
       16562,  2140,  1005,  1055,  5691,  2001,   103,  2098,  2000,
       12934,  5076,   103,  2010,  3596,  2000,  3153,  2189,  2012,
        1996,   103,  1997,  2023,  5236,  3185,  1012,  1996,  5436,
         103, 21425,  1010,  1996,   103,  2020,  4189,  1998,  5076,
        4490,  2055,   103,  2092,   103,  1996, 10682,  2002, 10299,
         103,  2070,  4066,   103,   103,  1999,  2028,   103,  1012,
         102,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0], dtype=int32)>, 'segment_ids': <tf.Tensor: shape=(256,), dtype=int32, numpy=
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)>, 'padding_mask': <tf.Tensor: shape=(256,), dtype=bool, numpy=
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False])>, 'mask_positions': <tf.Tensor: shape=(64,), dtype=int64, numpy=
array([ 2,  7, 12, 14, 16, 24, 29, 37, 45, 49, 50, 56, 58, 59, 63, 66, 67,
       70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])>}, <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([6669, 2228, 1996, 1997, 1997, 7848, 2725, 2927, 2003, 9590, 2020,
       2004, 2004, 1996, 2007, 1997, 5195, 3496,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0], dtype=int32)>, <tf.Tensor: shape=(64,), dtype=float32, numpy=
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>)

```
</div>
### Pretraining model


```python
# BERT backbone
backbone = keras_nlp.models.BertBackbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    num_layers=2,
    num_heads=2,
    hidden_dim=128,
    intermediate_dim=512,
)

# Language modeling head
mlm_head = keras_nlp.layers.MaskedLMHead(
    embedding_weights=backbone.token_embedding.embeddings,
)

inputs = {
    "token_ids": keras.Input(shape=(None,), dtype=tf.int32, name="token_ids"),
    "segment_ids": keras.Input(shape=(None,), dtype=tf.int32, name="segment_ids"),
    "padding_mask": keras.Input(shape=(None,), dtype=tf.int32, name="padding_mask"),
    "mask_positions": keras.Input(shape=(None,), dtype=tf.int32, name="mask_positions"),
}

# Encoded token sequence
sequence = backbone(inputs)["sequence_output"]

# Predict an output word for each masked input token.
# We use the input token embedding to project from our encoded vectors to
# vocabulary logits, which has been shown to improve training efficiency.
outputs = mlm_head(sequence, mask_positions=inputs["mask_positions"])

# Define and compile our pretraining model.
pretraining_model = keras.Model(inputs, outputs)
pretraining_model.summary()
pretraining_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(learning_rate=5e-4),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,
)

# Pretrain on IMDB dataset
pretraining_model.fit(
    pretrain_ds,
    validation_data=pretrain_val_ds,
    epochs=3,  # Increase to 6 for higher accuracy
)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_3"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold"> Param # </span>┃<span style="font-weight: bold"> Connected to         </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ mask_positions      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ padding_mask        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ segment_ids         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ bert_backbone_4     │ [(<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>),     │ <span style="color: #00af00; text-decoration-color: #00af00">4,385,…</span> │ mask_positions[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BertBackbone</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>,      │         │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],  │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)]             │         │ segment_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│                     │                   │         │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ masked_lm_head      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30522</span>)     │ <span style="color: #00af00; text-decoration-color: #00af00">3,954,…</span> │ bert_backbone_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaskedLMHead</span>)      │                   │         │ mask_positions[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,433,210</span> (135.29 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,433,210</span> (135.29 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 21s 12ms/step - loss: 5.6220 - sparse_categorical_accuracy: 0.0615 - val_loss: 4.9762 - val_sparse_categorical_accuracy: 0.1155
Epoch 2/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 16s 10ms/step - loss: 4.9844 - sparse_categorical_accuracy: 0.1214 - val_loss: 4.8706 - val_sparse_categorical_accuracy: 0.1321
Epoch 3/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 16s 10ms/step - loss: 4.8614 - sparse_categorical_accuracy: 0.1385 - val_loss: 4.4897 - val_sparse_categorical_accuracy: 0.2069

<keras_core.src.callbacks.history.History at 0x7f862c356e30>

```
</div>
After pretraining save your `backbone` submodel to use in a new task!

---
## Build and train your own transformer from scratch
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_expert.png" alt="drawing" height="250"/>

Want to implement a novel transformer architecture? The KerasNLP library offers all the
low-level modules used to build SoTA architectures in our `models` API. This includes the
`keras_nlp.tokenizers` API which allows you to train your own subword tokenizer using
`WordPieceTokenizer`, `BytePairTokenizer`, or `SentencePieceTokenizer`.

In this workflow, we train a custom tokenizer on the IMDB data and design a backbone with
custom transformer architecture. For simplicity, we then train directly on the
classification task. Interested in more details? We wrote an entire guide to pretraining
and finetuning a custom transformer on
[keras.io](https://keras.io/guides/keras_nlp/transformer_pretraining/),

### Train custom vocabulary from IMDB data


```python
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    imdb_train.map(lambda x, y: x),
    vocabulary_size=20_000,
    lowercase=True,
    strip_accents=True,
    reserved_tokens=["[PAD]", "[START]", "[END]", "[MASK]", "[UNK]"],
)
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=True,
    strip_accents=True,
    oov_token="[UNK]",
)
```

### Preprocess data with a custom tokenizer


```python
packer = keras_nlp.layers.StartEndPacker(
    start_value=tokenizer.token_to_id("[START]"),
    end_value=tokenizer.token_to_id("[END]"),
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=512,
)


def preprocess(x, y):
    token_ids = packer(tokenizer(x))
    return token_ids, y


imdb_preproc_train_ds = imdb_train.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
imdb_preproc_val_ds = imdb_test.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

print(imdb_preproc_train_ds.unbatch().take(1).get_single_element())
```

<div class="k-default-codeblock">
```
(<tf.Tensor: shape=(512,), dtype=int32, numpy=
array([    1,    51,   549,   104,   110,    18,   103,   285,    51,
         549,   203,   126,   611,   103,   104,   110,    18,    51,
        6195,   136,  2743,   107,    43,  2943,  2467,   103,    96,
         429,   416,    98,    96,   110,    18,   113,   294,   472,
         163,   144,   790,   103,    96, 11386,   226,   146,    96,
        2090,   106, 10633,   408,   114,   112,    18,    51,   106,
          96,   757,   103,    96,  1107,  3703,   109,   152,  1051,
       10275,   114,   152,   487,   103,  2246,    99,   140,   161,
         162,   240,   114,    96,   110, 16526,    18,   103,   285,
         124,  1520,   657,   163,    43,   264,   304,   128,   102,
          11,    61,   347,    99,   805,   105,  1433,    18,  3627,
         148,    99,   461,  1944,   407,    18,   746,   102,   308,
          99,  2027,   609,    18, 13687,  8042,  6969,  3929,   853,
       17549,    16, 15274,    51,   549,   104,   110,    18,   103,
         285,    51,   549,   203,   126,   611,   103,   104,   110,
          18,    51,  6195,   136,  2743,   107,    43,  2943,  2467,
         103,    96,   429,   416,    98,    96,   110,    18,   113,
         294,   472,   163,   144,   790,   103,    96, 11386,   226,
         146,    96,  2090,   106, 10633,   408,   114,   112,    18,
          51,   106,    96,   757,   103,    96,  1107,  3703,   109,
         152,  1051, 10275,   114,   152,   487,   103,  2246,    99,
         140,   161,   162,   240,   114,    96,   110, 16526,    18,
         103,   285,   124,  1520,   657,   163,    43,   264,   304,
         128,   102,    11,    61,   347,    99,   805,   105,  1433,
          18,  3627,   148,    99,   461,  1944,   407,    18,   746,
         102,   308,    99,  2027,   609,    18, 13687,  8042,  6969,
        3929,    32,   101,    19,    34,    32,   101,    19,    34,
         853, 17549,    16, 15274,     2,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0],
      dtype=int32)>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)

```
</div>
### Design a tiny transformer


```python
token_id_input = keras.Input(
    shape=(None,),
    dtype="int32",
    name="token_ids",
)
outputs = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=len(vocab),
    sequence_length=packer.sequence_length,
    embedding_dim=64,
)(token_id_input)
outputs = keras_nlp.layers.TransformerEncoder(
    num_heads=2,
    intermediate_dim=128,
    dropout=0.1,
)(outputs)
# Use "[START]" token to classify
outputs = keras.layers.Dense(2)(outputs[:, 0, :])
model = keras.Model(
    inputs=token_id_input,
    outputs=outputs,
)

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_5"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ token_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ token_and_position_embedding    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │  <span style="color: #00af00; text-decoration-color: #00af00">1,259,648</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbedding</span>)     │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ transformer_encoder_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │     <span style="color: #00af00; text-decoration-color: #00af00">33,472</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncoder</span>)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ get_item_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_28 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                 │        <span style="color: #00af00; text-decoration-color: #00af00">130</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,293,250</span> (39.47 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,293,250</span> (39.47 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



### Train the transformer directly on the classification objective


```python
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(5e-5),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,
)
model.fit(
    imdb_preproc_train_ds,
    validation_data=imdb_preproc_val_ds,
    epochs=3,
)
```

<div class="k-default-codeblock">
```
Epoch 1/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.6688 - sparse_categorical_accuracy: 0.5758 - val_loss: 0.3674 - val_sparse_categorical_accuracy: 0.8507
Epoch 2/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.3126 - sparse_categorical_accuracy: 0.8725 - val_loss: 0.3138 - val_sparse_categorical_accuracy: 0.8729
Epoch 3/3
 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.2226 - sparse_categorical_accuracy: 0.9151 - val_loss: 0.4513 - val_sparse_categorical_accuracy: 0.8125

<keras_core.src.callbacks.history.History at 0x7f8520133970>

```
</div>
Excitingly, our custom classifier is similar to the performance of fine-tuning
`"bert_tiny_en_uncased"`! To see the advantages of pretraining and exceed 90% accuracy we
would need to use larger **presets** such as `"bert_base_en_uncased"`.
