# Getting Started with KerasNLP

**Author:** [Jonathan Bischof](https://github.com/jbischof)<br>
**Date created:** 2022/12/15<br>
**Last modified:** 2022/12/15<br>
**Description:** An introduction to the KerasNLP API.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/getting_started.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/getting_started.py)



---
## Introduction

KerasNLP is a natural language processing library that supports users through
their entire development cycle. Our workflows are built from modular components
that have state-of-the-art preset weights and architectures when used
out-of-the-box and are easily customizable when more control is needed. We
emphasize in-graph computation for all workflows so that developers can expect
easy productionization using the TensorFlow ecosystem.

This library is an extension of the core Keras API; all high-level modules are
[`Layers`](/api/layers/) or [`Models`](/api/models/). If you are familiar with Keras,
congratulations! You already understand most of KerasNLP.

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
!pip install -q --upgrade keras-nlp tensorflow
```

```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Use mixed precision for optimal performance
keras.mixed_precision.set_global_policy("mixed_float16")
```
<div class="k-default-codeblock">
```
/bin/bash: /home/haifengj/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)

INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
Your GPU will likely run quickly with dtype policy mixed_float16 as it has compute capability of at least 7.0. Your GPU: Tesla V100-SXM2-16GB, compute capability 7.0

```
</div>
---
## API quickstart

Our highest level API is `keras_nlp.models`. These symbols cover the complete user
journey of converting strings to tokens, tokens to dense features, and dense features to
task-specific output. For each `XX` architecture (e.g., `Bert`), we offer the following
modules:

* **Tokenizer**: `keras_nlp.models.XXTokenizer`
  * **What it does**: Converts strings to `tf.RaggedTensor`s of token ids.
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
imdb_train = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
)
imdb_test = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/test",
    batch_size=BATCH_SIZE,
)

# Inspect first review
# Format is (review text tensor, label tensor)
print(imdb_train.unbatch().take(1).get_single_element())
```
<div class="k-default-codeblock">
```
/bin/bash: /home/haifengj/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  56.7M      0  0:00:01  0:00:01 --:--:-- 56.7M
/bin/bash: /home/haifengj/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)
/bin/bash: /home/haifengj/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)
/bin/bash: /home/haifengj/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)

Found 25000 files belonging to 2 classes.
Found 25000 files belonging to 2 classes.
(<tf.Tensor: shape=(), dtype=string, numpy=b'This animation TV series is simply the best way for children to learn how the human body works. Yes, this is biology but they will never tell it is.<br /><br />I truly think this is the best part of this stream of "educational cartoons". I do remember you can find little books and a plastic body in several parts: skin, skeleton, and of course: organs.<br /><br /> In the same stream, you\'ll find: "Il \xc3\xa9tait une fois l\'homme" which relate the human History from the big bang to the 20th century. There is: "Il \xc3\xa9tait une fois l\'espace" as well (about the space and its exploration) but that one is more a fiction than a description of the reality since it takes place in the future.'>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)

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
WARNING:tensorflow:From /home/haifengj/miniconda3/lib/python3.10/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
1/1 [==============================] - 3s 3s/step

array([[-1.539,  1.542]], dtype=float16)

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
1563/1563 [==============================] - 42s 25ms/step - loss: 0.4630 - sparse_categorical_accuracy: 0.7835

[0.4629528820514679, 0.7834799885749817]

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
1563/1563 [==============================] - 294s 179ms/step - loss: 0.4203 - sparse_categorical_accuracy: 0.8024 - val_loss: 0.3077 - val_sparse_categorical_accuracy: 0.8700

<keras.callbacks.History at 0x7fbed01424d0>

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

**Note:** this code only works if your data fits in memory. If not, pass a `filename` to
`cache()`.


```python
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased",
    sequence_length=512,
)
# Apply the preprocessor to every sample of train and test data using `map()`.
# `tf.data.AUTOTUNE` and `prefetch()` are options to tune performance, see
# https://www.tensorflow.org/guide/data_performance for details.
imdb_train_cached = (
    imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)
imdb_test_cached = (
    imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    preprocessor=None,
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
1563/1563 [==============================] - 262s 159ms/step - loss: 0.4221 - sparse_categorical_accuracy: 0.8002 - val_loss: 0.3077 - val_sparse_categorical_accuracy: 0.8699
Epoch 2/3
1563/1563 [==============================] - 225s 144ms/step - loss: 0.2673 - sparse_categorical_accuracy: 0.8923 - val_loss: 0.2935 - val_sparse_categorical_accuracy: 0.8783
Epoch 3/3
1563/1563 [==============================] - 225s 144ms/step - loss: 0.1974 - sparse_categorical_accuracy: 0.9271 - val_loss: 0.3418 - val_sparse_categorical_accuracy: 0.8686

<keras.callbacks.History at 0x7fbe99bc5960>

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
a `tf.RaggedTensor`.


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
array([  101, 11271,  9261,  2003,  2028,  1997,  2216,  5889,  2008,
        1045,  1005,  2310,  2763,  2464,  1999,  1037,  6474,  3152,
        1010,  2021,  2040,  2038,  2196,  2428,  5068,  2005,  2033,
        1012,  2411,  2358, 10893,  2094,  1010, 11937, 26243, 14287,
        1010,  2652,  1996,  2168,  7957,  1997,  4395,  1998,  2559,
        5399,  2066,  1996,  6660,  2104,  9250,  9465,  4811,  1010,
        2002,  1005,  1055,  2019,  3364,  2008,  3138,  2070,  3947,
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
    optimizer=keras.optimizers.experimental.AdamW(5e-5),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
)
model.summary()
model.fit(
    imdb_train_preprocessed,
    validation_data=imdb_test_preprocessed,
    epochs=3,
)
```

<div class="k-default-codeblock">
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 padding_mask (InputLayer)      [(None, None)]       0           []                               
                                                                                                  
 segment_ids (InputLayer)       [(None, None)]       0           []                               
                                                                                                  
 token_ids (InputLayer)         [(None, None)]       0           []                               
                                                                                                  
 bert_backbone_3 (BertBackbone)  {'sequence_output':  4385920    ['padding_mask[0][0]',           
                                 (None, None, 128),               'segment_ids[0][0]',            
                                 'pooled_output': (               'token_ids[0][0]']              
                                None, 128)}                                                       
                                                                                                  
 transformer_encoder (Transform  (None, None, 128)   198272      ['bert_backbone_3[0][1]']        
 erEncoder)                                                                                       
                                                                                                  
 transformer_encoder_1 (Transfo  (None, None, 128)   198272      ['transformer_encoder[0][0]']    
 rmerEncoder)                                                                                     
                                                                                                  
 tf.__operators__.getitem_4 (Sl  (None, 128)         0           ['transformer_encoder_1[0][0]']  
 icingOpLambda)                                                                                   
                                                                                                  
 dense (Dense)                  (None, 2)            258         ['tf.__operators__.getitem_4[0][0
                                                                 ]']                              
                                                                                                  
==================================================================================================
Total params: 4,782,722
Trainable params: 396,802
Non-trainable params: 4,385,920
__________________________________________________________________________________________________
Epoch 1/3
1563/1563 [==============================] - 50s 23ms/step - loss: 0.5825 - sparse_categorical_accuracy: 0.6916 - val_loss: 0.5144 - val_sparse_categorical_accuracy: 0.7460
Epoch 2/3
1563/1563 [==============================] - 15s 10ms/step - loss: 0.4842 - sparse_categorical_accuracy: 0.7655 - val_loss: 0.4286 - val_sparse_categorical_accuracy: 0.8025
Epoch 3/3
1563/1563 [==============================] - 15s 10ms/step - loss: 0.4409 - sparse_categorical_accuracy: 0.7968 - val_loss: 0.4084 - val_sparse_categorical_accuracy: 0.8145

<keras.callbacks.History at 0x7fbe20713af0>

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
array([  101,  2064,  2151,  3185,  2468,  2062, 15743,  2084,  2023,
        1029,   103,  2064,  2102,  2903,  7631,   103,   103,  2023,
        5896,   103,   103,  2049,  7020,  9541,  9541,  2080, 21425,
        2008,  2017,  2064,  2425,   103,  5436,  3574,  1996,   103,
        2013,  1996,   103,  2184,   103,  1012,  1996,  2877,  3883,
        3849,  2066,  2016,  4122,  2000,  2022,   103,  1006,  2021,
        2016,  2987,   103,  1056,  2191,  2009,  1010,   103, 10658,
        2038,  2062, 23041,  4813,  1007,  1012,   103,  7987,  1013,
        1028,   103,  7987,  1013,  1028,  1996,   103,   103,   103,
         103,   103,  2839,  5235,  1998,  3464,  1999,  1037,  1037,
        2189,  2082,  3084,  1996, 11588,   103,  1996,  3850,  3117,
        4025,   103,  1037,   103,  8308,  1012,   103,   103,  2102,
         103,   103,  2006,  8114,   684,  2791,  1997,   103,  3494,
        2021,  1996,  2028,  2204,  2518,  1997,  1996,   103,  2003,
        5506,   103,   103, 26402,  2836,  2029,  9020,  2000,   103,
        2166,  2000,  1037, 11463,  2080,   103,  2066,  2028,  1011,
        8789,  2839, 28350,  1026,  7987,  1013,  1028,  1026,  7987,
        1013, 15799,  1996,  3185,  2003,  2061, 18178,   103,  2100,
         103,  2009, 12668,  2000,   103,  4091,   103,  1045,  2064,
        2228,  2070,  2410,   103,  2214, 29168,  1011, 15896,  3057,
       11273,  1000,  1051,  1010,  2079,  2507,  2149,  1037,  3338,
         999,  2065,  2057,   103,  8867,  7122,  2045,  2003,  2467,
        1996,   103, 24287,  2338,  5023,  4873,  1999,  1996, 14832,
        1000,  1012,  1045,  2435,   103,  1016,  2612,   103,  2028,
        2069,  7286,  3448,  5506,   103,   103,   102,     0,     0,
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
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False])>, 'mask_positions': <tf.Tensor: shape=(64,), dtype=int64, numpy=
array([ 10,  14,  15,  16,  19,  20,  31,  33,  35,  38,  40,  51,  56,
        58,  61,  65,  69,  73,  78,  79,  80,  81,  82,  95, 100, 102,
       105, 106, 108, 109, 111, 112, 115, 121, 124, 126, 127, 128, 129,
       134, 140, 146, 154, 160, 162, 166, 168, 174, 192, 199, 208, 211,
       214, 217, 220, 221,   0,   0,   0,   0,   0,   0,   0,   0])>}, <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([ 2017,  1037,  3538,  1997,  1012,  1998,  1996,  1998,  4566,
        2034,  2781, 22635,  1005,  2191,  1996,  3772,  1026,  1026,
       19413, 11493,  7971,  2008,  1996,  1997,  2066,  3439,  1045,
        2180,  2130,  7615,  1996,  8467,  1996,  2518,  2143,  5506,
        5054,  1005,  1055,  3288,  1011,  1012,  1028,  2229,  2008,
        2115,  1012,  2095,  2215,  3428,  1012,  2009,  1997,  2005,
        5054,  1012,     0,     0,     0,     0,     0,     0,     0,
           0], dtype=int32)>, <tf.Tensor: shape=(64,), dtype=float16, numpy=
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float16)>)

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
    "token_ids": keras.Input(shape=(None,), dtype=tf.int32),
    "segment_ids": keras.Input(shape=(None,), dtype=tf.int32),
    "padding_mask": keras.Input(shape=(None,), dtype=tf.int32),
    "mask_positions": keras.Input(shape=(None,), dtype=tf.int32),
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
    optimizer=keras.optimizers.experimental.AdamW(learning_rate=5e-4),
    weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
)

# Pretrain on IMDB dataset
pretraining_model.fit(
    pretrain_ds,
    validation_data=pretrain_val_ds,
    epochs=3,  # Increase to 6 for higher accuracy
)
```

<div class="k-default-codeblock">
```
/home/haifengj/miniconda3/lib/python3.10/site-packages/keras/engine/functional.py:638: UserWarning: Input dict contained keys ['mask_positions'] which did not match any model input. They will be ignored by the model.
  inputs = self._flatten_to_reference_inputs(inputs)

Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_4 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 input_3 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 input_1 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 bert_backbone_4 (BertBackbone)  {'sequence_output':  4385920    ['input_4[0][0]',                
                                 (None, None, 128),               'input_3[0][0]',                
                                 'pooled_output': (               'input_2[0][0]',                
                                None, 128)}                       'input_1[0][0]']                
                                                                                                  
 masked_lm_head (MaskedLMHead)  (None, None, 30522)  3954106     ['bert_backbone_4[0][1]',        
                                                                  'input_4[0][0]']                
                                                                                                  
==================================================================================================
Total params: 4,433,210
Trainable params: 4,433,210
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/3
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
1563/1563 [==============================] - 103s 57ms/step - loss: 5.2620 - sparse_categorical_accuracy: 0.0866 - val_loss: 4.9799 - val_sparse_categorical_accuracy: 0.1172
Epoch 2/3
1563/1563 [==============================] - 77s 49ms/step - loss: 4.9584 - sparse_categorical_accuracy: 0.1241 - val_loss: 4.8639 - val_sparse_categorical_accuracy: 0.1327
Epoch 3/3
1563/1563 [==============================] - 77s 49ms/step - loss: 4.7992 - sparse_categorical_accuracy: 0.1480 - val_loss: 4.5584 - val_sparse_categorical_accuracy: 0.1919

<keras.callbacks.History at 0x7fbe2ca08700>

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
array([    1,    51,    11,    55,  4588,    98,   104,   112,    97,
         102,   230,  1571,   538,   163,   105,   128,   201,     6,
        2116,     6,  4596,   102,  2053,    96,   895,  1733,  2508,
          18,    43,   903,   745,   100,   303, 10439,  1639,    31,
         103,   937,   126,  1225,   112,  4406,   114,    96,  1767,
          11,    61,  1897,    97,   234,   120,  7017,   114,   184,
         298,  1417,    18,  1507,   107,    43, 11524,  1955,    16,
         104,   100,    43, 17039,  3669,    99,  9709,   165,    96,
        7133,   409,    18,   228,   104,   112,   139,   184,  1640,
         102,  2894, 18115,    99,    96,  2603,    16,   354,   128,
         321,   107,    96,   195,   117,  2102,    18,    32,   101,
          19,    34,    32,   101,    19,    34,   103,    96,  1733,
         409,  9709,    11,    61, 12297,   100,  1300,  6172,    30,
         129,    96,   176,   229,   116,   100,  1096,   107,    43,
         909,  1210,    16,  1466,  3025,  2576,    16,  8945,   125,
        2082,   281,   180,    97,   206,  2248,   114,   118,   298,
         198,  2345,    12,   104,   100,  3346,    97, 16468,   272,
         378,    18,    13,   113,   689,   227,  2091,  2640,   104,
        9476, 15084, 15535,  7658,    99,  1790,    96,   121,  1865,
          99,    96,  8485,    98,  5044,    18,  1733,  2781,  1729,
        1518,  8597, 15031, 10060,   136,  9709,    97, 13478, 12297,
          98,   118,  2035,   651,   146,   116,  2219,    43,   452,
       15038,    99,  2392,   180,   129,    96,  1865,  1916,  1986,
        8210,   136,    18,    18,    18,  3731,    11,    61,   409,
        7411,    96,   295,    16,  5700,    43, 12297,   127,  4198,
          96,    65,  1986,  8210,   136,     6,   231,   235,    97,
        1216,   163,   150,   144,     5,     6,  3731,  9150,   136,
       12297,    11,    61,   196,   107,  9709,  3992,   102,    18,
         161,   158,   118,  6712,   107,    96,  3571,    98,   118,
        9969, 13378,    18, 12297,   100,   128,  8058,    99,  1694,
          98,    96,  2852,   113,   689,   227,  1733,   657,   232,
        9474,    12,   116,   218,   128,   109,  2099,   195,    13,
         329,   196,  1034,  8550,  5948,    96,   204,    99,   184,
        3720,  1282,    18,  3731,    11,    61,   112,   220,  6088,
          96, 13699,   293,  4229,  9173,   136,    97, 10650,   917,
        3745,   103,    43,   264, 12994,  6867,   229,   105,   117,
         436,  2042,   103,  1733,    11,    61,   409,    18,  4749,
         161,   158,  3256,  3196, 16112,  3829,  1429,   237,   109,
       13434,   993,  8330,    97,  8158,    16,  1132,   237,    96,
         421,    98, 17038,    18,   108,   152,  4364,    96, 15623,
         100,   144,  1508,   103,    96,  1225,   409,   165,    96,
         511,   297,    18,    32,   101,    19,    34,    32,   101,
          19,    34,  9709,    11,    61,  4975,   100,   126,   766,
       12015,   103,    96,  3077,    97,  3731,   260,  3590,    99,
         104,   103,   121,    98,   152,   607,   229,    30,   146,
       12297,  3381,    96,     6,  8158,   245, 13380,     6,   704,
          16,   325,   254, 10081,   175, 15963,   452,    18,    96,
       16947,  3444,   100,  3055,   272, 14726,  1571,  2946,  9553,
         109,    96,  4075,   850,   317,   146,  1695,   397,    16,
        2667,    99,   118,   459,   111,  5122,    99,    96,    65,
        1986,  8210,   136,    16, 12297,   100,  8366,  4079,    18,
         104,   100,   121,    98,    96,   182,  3391,  9302,    98,
          96,   112,    97,  5853,   163,  2101,    51,   158,   102,
          18,    32,   101,    19,    34,    32,   101,    19,    34,
          96,   164, 10410,   446,    19,   149, 13468,  6236,   104,
         112,    11,    61,   573,  1598,   111,   160,   102,   169,
        6076,   102,   294,   119, 17262,   328,   109,     2],
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

<div class="k-default-codeblock">
```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 token_ids (InputLayer)      [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 64)         1259648   
 g (TokenAndPositionEmbeddin                                     
 g)                                                              
                                                                 
 transformer_encoder_2 (Tran  (None, None, 64)         33472     
 sformerEncoder)                                                 
                                                                 
 tf.__operators__.getitem_6   (None, 64)               0         
 (SlicingOpLambda)                                               
                                                                 
 dense_1 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 1,293,250
Trainable params: 1,293,250
Non-trainable params: 0
_________________________________________________________________

```
</div>
### Train the transformer directly on the classification objective


```python
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.experimental.AdamW(5e-5),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
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
1563/1563 [==============================] - 128s 77ms/step - loss: 0.6113 - sparse_categorical_accuracy: 0.6411 - val_loss: 0.4020 - val_sparse_categorical_accuracy: 0.8279
Epoch 2/3
1563/1563 [==============================] - 117s 75ms/step - loss: 0.3117 - sparse_categorical_accuracy: 0.8729 - val_loss: 0.3062 - val_sparse_categorical_accuracy: 0.8786
Epoch 3/3
1563/1563 [==============================] - 135s 87ms/step - loss: 0.2381 - sparse_categorical_accuracy: 0.9066 - val_loss: 0.3113 - val_sparse_categorical_accuracy: 0.8734

<keras.callbacks.History at 0x7fba26e94490>

```
</div>
Excitingly, our custom classifier is similar to the performance of fine-tuning
`"bert_tiny_en_uncased"`! To see the advantages of pretraining and exceed 90% accuracy we
would need to use larger **presets** such as `"bert_base_en_uncased"`.
