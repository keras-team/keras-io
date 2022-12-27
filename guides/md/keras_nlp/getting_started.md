# Getting Started with KerasNLP

**Author:** [jbischof](https://github.com/jbischof)<br>
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

This library is an extension of the core Keras API; all high level modules are
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

Throughout our guide we use Professor Keras, the official Keras mascot, as a visual
reference for the complexity of the material:

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_evolution.png" alt="drawing" height="250"/>


```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Use mixed precision for optimal performance
keras.mixed_precision.set_global_policy("mixed_float16")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK

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
  * **What it does**: Converts strings to a dictonary of preprocessed tensors consumed by
    the backbone, starting with tokenization.
  * **Why it's important**: Each model uses special tokens and extra tensors to understand
    the input such as deliminting input segments and identifying padding tokens. Padding each
    sequence to the same length improves computational efficiency.
  * **Has a**: `XXTokenizer`.
  * **Inherits from**: `keras.layers.Layer`.
* **Backbone**: `keras_nlp.models.XXBackbone`
  * **What it does**: Converts preprocessed tensors to dense features. *Does not handle
    strings; call the preprocessor first.*
  * **Why it's important**: The backbone distills the input tokens into dense features that
    can be used in downstream tasks. It is generally pretrained on a language modeling task
    using massive amounts of unlabeled data. Transfering this information to a new task is a
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
imdb_train = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
)
imdb_test = tf.keras.preprocessing.text_dataset_from_directory(
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
100 80.2M  100 80.2M    0     0  91.2M      0 --:--:-- --:--:-- --:--:-- 91.2M

Found 25000 files belonging to 2 classes.
Found 25000 files belonging to 2 classes.
(<tf.Tensor: shape=(), dtype=string, numpy=b"Although Kurt Russell was and is probably the closest person to look like Elvis in show-business, so many things were false in this film. First of all, the makers claimed Elvis opened his famous live shows in '69 after a 9 year hault for films by wearing a white jump-suit made in 1972. Also they claimed he sang 'burning love' which he first sung in 1972 and 'the wonder of you' which he first recorded in 1970. They also claim that he got his first guitar for christmas when all Elvis fans know he got it for his birthday. I know all movies based on past have something false but these things are so obvious to people who like Elvis.">, <tf.Tensor: shape=(), dtype=int32, numpy=1>)

```
</div>
---
## Inference with a pretrained classifier

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png" alt="drawing" height="250"/>

The highest level module in KerasNLP is a **task**. A **task** is a `keras.Model`
consisting of a (generally pretrained) **backbone** model and task-specific layers.
Here's an example using `keras_nlp.models.BertClassifier`.

**Note**: Outputs are the logits per class (e.g., `[0, 0]` is 50% chance of positive).


```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
# Note: batched inputs expected so must wrap string in iterable
classifier.predict(["I love modular workflows in keras-nlp!"])
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 1s 1s/step

array([[-1.54 ,  1.543]], dtype=float16)

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

Let's evaluate our classifier on the IMDB dataset. We first need to compile the
`keras.Model`. The output is `[loss, accuracy]`,

**Note**: We don't need an optimizer since we're not training the model.


```python
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
)

classifier.evaluate(imdb_test)
```

<div class="k-default-codeblock">
```
1563/1563 [==============================] - 16s 10ms/step - loss: 0.4630 - sparse_categorical_accuracy: 0.7837

[0.4629555940628052, 0.7836800217628479]

```
</div>
Our result is 78% accuracy without training anything. Not bad!

---
## Fine tuning a pretrained BERT backbone

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png" alt="drawing" height="250"/>

When labeled text specific to our task is available, fine-tuning a custom classifier can
improve performance. If we want to predict IMDB review sentiment, using IMDB data should
perform better than Rotten Tomatoes data! And for many tasks no relevant pretrained model
will be available (e.g., categorizing customer reviews).

The workflow for fine-tuning is almost identical to above, except that we request a
**preset** for the **backbone**-only model rather than the entire classifier. When passed
a **backone** **preset**, a **task** `Model` will randomly initialize all task-specific
layers in preparation for training. For all the **backbone** presets available for
`BertClassifier`, see our keras.io [models page](https://keras.io/api/keras_nlp/models/).

To train your classifier, use `Model.compile()` and `Model.fit()` as with any other
`keras.Model`. Since preprocessing is included in all **tasks** by default, we again pass
the raw data.


```python
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.experimental.AdamW(5e-5),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
)
classifier.fit(
    imdb_train,
    validation_data=imdb_test,
    epochs=1,
)
```

<div class="k-default-codeblock">
```
1563/1563 [==============================] - 183s 113ms/step - loss: 0.4156 - sparse_categorical_accuracy: 0.8085 - val_loss: 0.3088 - val_sparse_categorical_accuracy: 0.8687

<keras.callbacks.History at 0x7f806e165580>

```
</div>
Here we see significant lift in validation accuracy (0.78 -> 0.87) with a single epoch of
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
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.experimental.AdamW(5e-5),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
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
1563/1563 [==============================] - 172s 106ms/step - loss: 0.4133 - sparse_categorical_accuracy: 0.8096 - val_loss: 0.3109 - val_sparse_categorical_accuracy: 0.8690
Epoch 2/3
1563/1563 [==============================] - 153s 98ms/step - loss: 0.2666 - sparse_categorical_accuracy: 0.8948 - val_loss: 0.2932 - val_sparse_categorical_accuracy: 0.8792
Epoch 3/3
1563/1563 [==============================] - 154s 99ms/step - loss: 0.1961 - sparse_categorical_accuracy: 0.9255 - val_loss: 0.3505 - val_sparse_categorical_accuracy: 0.8664

<keras.callbacks.History at 0x7f806d7fe340>

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

# Write your own packer or use one our `Layers`
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    # Note: This cannot be longer than the preset's `sequence_length`, and there
    # is no check for a custom preprocessor!
    sequence_length=64,
)


def preprocessor(x, y):
    token_ids, segment_ids = packer(tokenizer(x))
    x = {
        "token_ids": token_ids,
        "segment_ids": segment_ids,
        "padding_mask": token_ids != 0,
    }
    return x, y


imbd_train_preprocessed = imdb_train.map(preprocessor, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
imdb_test_preprocessed = imdb_test.map(preprocessor, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# Preprocessed example
print(imbd_train_preprocessed.unbatch().take(1).get_single_element())
```

<div class="k-default-codeblock">
```
({'token_ids': <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([  101,  2074,  2387,  2023,  2012,  1996,  2494,  6930,  2248,
        2143,  2782,  1998,  2009,  2001,  6057,  2004,  3109,  1998,
        1037,  2978, 16524,  1012,  3138,  2173,  1999,  4361,  1010,
        2073,  2122,  2048, 23160,  2444,  1999,  2023,  2448,  1011,
        2091,  2160,  1999,  1996,  2690,  1997,  1037, 10846,  2291,
        1012,  2585,  2002, 13668,  6582,  1006,  9231,  1010, 14291,
        1010, 22330, 27921,  1007,  1998,  4080,  4679,  1006, 14291,
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
this case we provide direct access to the **backbone** `Model`, which has its own
`from_preset` constructor and can be composed with custom `Layer`s. Detailed examples can
be found at our [transfer learning guide](https://keras.io/guides/transfer_learning/).

A **backbone** `Model` does not include automatic preprocessing but can be paired with a
matching **preprocessor** using the same **preset** as shown in the previous workflow.

In this workflow we experiment with freezing our backbone model and adding two trainable
transfomer layers to adapt to the new input.

**Note**: We can igonore the warning about gradients for the `pooled_dense` layer because
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
1563/1563 [==============================] - 34s 17ms/step - loss: 0.5811 - sparse_categorical_accuracy: 0.6941 - val_loss: 0.5046 - val_sparse_categorical_accuracy: 0.7554
Epoch 2/3
1563/1563 [==============================] - 18s 11ms/step - loss: 0.4859 - sparse_categorical_accuracy: 0.7686 - val_loss: 0.4235 - val_sparse_categorical_accuracy: 0.8056
Epoch 3/3
1563/1563 [==============================] - 17s 11ms/step - loss: 0.4405 - sparse_categorical_accuracy: 0.7947 - val_loss: 0.3994 - val_sparse_categorical_accuracy: 0.8198

<keras.callbacks.History at 0x7f7fd8458b50>

```
</div>
This model achieves reasonable accuracy despite having only 10% the trainable parameters
of our `BertClassifier` model. Each training step takes about 1/3 of the time---even
accounting for cached preprocessing.

---
## Pretraining a backbone model
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_expert.png" alt="drawing" height="250"/>

Do you have access to large unlabeled datasets in your domain? Are they are around the
same size as used to train popular backbones such as BERT, RoBERTa, or GPT2 (XX+ GiB)? If
so, you might benefit from domain-specific pretraining of your own backbone models.

NLP models are generally pretrained on a language modeling task, predicting masked words
given the visible words in an input sentence. For example, given the input
`"The fox [MASK] over the [MASK] dog"`, the model might be asked to predict `["jumped", "lazy"]`.
The lower layers of this model are then packaged as a **backbone** to be combined with
layers relating to a new task.

The KerasNLP library offers SoTA **backbones** and **tokenizers** to be trained from
scratch without presets.

In this workflow we pretrain a BERT **backbone** using our IMDB review text. We skip the
"next sentence prediction" (NSP) loss because it adds significant complexity to the data
processing and was dropped by later models like RoBERTa. See our e2e [BERT pretraining
example](https://github.com/keras-team/keras-nlp/tree/master/examples/bert) for
step-by-step details on how to replicate the original paper.

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
array([  101,  9643,  1010,  9643,  1010,  9643,  1012,  1026,  7987,
        1013,  1028,  1026,  7987,  1013,  1028,  1037,   103, 11020,
       18537,  4844,  2012,  1996,  2707,  1998,  1037,  2261, 11808,
       24534,  7760,   103,  2025,   103,  2204,  3185,   103,   103,
        2320,  2153,  1045,  1005,   103, 15261,  2012,  1996,   103,
        1998,   103,   103,  2070,   103,   103,  1999, 10910,  1037,
        3185,  2537,  1998,   103,  2027,  2123,  1005,  1056,  2031,
        1996,  6620,  2000, 13776,  2008,  2054,  2027,  2031,  2081,
        2003,  2019,   103,  8632,  1997, 10231,  1012,  1026,  7987,
        1013,   103,  1026,  7987,  1013,  1028, 20323,  2970,  1998,
        2699,  2000,  2228,   103,   103,  2417, 21564,  2075,  3444,
        2061,  2008,  1045,  2071,   103,  2560,  5703,  2026,  6256,
         103,  1996,  2069,  2028,   103,  2071,  2228,  1997,  2001,
        2008,   103,  3098,   103,  2011,  5061,   103,  3492,  2204,
        1012,   103,   103,  1012,  1045,  4687,   103,  2016,  2038,
        2464,  2023,  1029,  1026,  7987,   103,  1028, 26907,   103,
        1013,  1028,  3422,  2023,  2012,   103,  2566,  4014,   103,
         103, 29556,  2089,  3102,  2017,  1012,   102,     0,     0,
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
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False])>, 'mask_positions': <tf.Tensor: shape=(64,), dtype=int64, numpy=
array([ 15,  16,  19,  29,  31,  34,  35,  40,  44,  46,  47,  49,  50,
        57,  66,  74,  82,  86,  87,  88,  93,  94, 103, 108, 112, 118,
       120, 122, 123, 126, 127, 128, 132, 140, 142, 143, 146, 149, 152,
       153,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])>}, <tf.Tensor: shape=(64,), dtype=int32, numpy=
array([ 1037, 24707, 17674,  2515,  1037,  2191,  1012,  1049,  9128,
        8066,  2008,  2111,  2031,  2664, 19148, 14395,  1028,  1028,
        1045,  2938,  1997,  1037,  2012,  2021,  1045,  1996,  2650,
        5061,  2001,  1012,  1012,  1012,  2065,  1013,  1026,  7987,
        3422,  2115,  1010,  1996,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0], dtype=int32)>, <tf.Tensor: shape=(64,), dtype=float16, numpy=
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float16)>)

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

/home/matt/miniconda3/envs/keras-io/lib/python3.9/site-packages/keras/engine/functional.py:638: UserWarning: Input dict contained keys ['mask_positions'] which did not match any model input. They will be ignored by the model.
  inputs = self._flatten_to_reference_inputs(inputs)

Epoch 1/3
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
1563/1563 [==============================] - 68s 39ms/step - loss: 5.2595 - sparse_categorical_accuracy: 0.0866 - val_loss: 4.9751 - val_sparse_categorical_accuracy: 0.1144
Epoch 2/3
1563/1563 [==============================] - 58s 37ms/step - loss: 4.9573 - sparse_categorical_accuracy: 0.1244 - val_loss: 4.8743 - val_sparse_categorical_accuracy: 0.1310
Epoch 3/3
1563/1563 [==============================] - 57s 37ms/step - loss: 4.8230 - sparse_categorical_accuracy: 0.1440 - val_loss: 4.6139 - val_sparse_categorical_accuracy: 0.1837

<keras.callbacks.History at 0x7f806dfd3bb0>

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

In this workflow we train a custom tokenizer on the IMDB data and design a backbone with
custom transformer architecture. For simplicity we then train directly on the
classification task. Interested in more details? We wrote an entire guide to pretraining
and finetuning a custom transformer on
[keras.io](https://keras.io/guides/keras_nlp/transformer_pretraining/),

### Train custom vocabulary from IMBD data


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

### Preprocess data with custom tokenizer


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
array([    1,   715,   137,    98,   199,    99,   943,    16,    51,
        1776,   148,   971, 11048,    18,    51, 11564,   236,    96,
         176,   932,    98,    96,   110,  1661,   138,    96,   457,
         153,   120,    43,   536,    99,   158,    18,  3213,    16,
         102,   157, 16673,   148,    16,    97,    51,   549,    96,
         110,   274,    43,   318,    18,    96,   429,   416,    98,
          96,   110,   160,   163, 14683,    97,  1203,    96,   535,
         149,    18,  2895,    16,   353,   131,   162,  1952,    98,
        1796,  2085,    16,   124,   162,   115,  4000,   133, 10280,
          18,    32,   101,    19,    34,    32,   101,    19,    34,
          51,   153,   119,    99,   225,    16,   242,    17,    96,
         248,   122,   119,  1273,  2007,  2630,    16,   128,   120,
        3010,   138,   113,   119,  1216,  2062,   239,  2495,   133,
         159,  2857,   580,    18,    32,   101,    19,    34,    32,
         101,    19,    34,    51,   157,   549,   104,   110,    16,
          97,   171,   119,    99,   752,   641,    43,  1131,   108,
         152,   298,  1722,    18,     2,     0,     0,     0,     0,
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
1563/1563 [==============================] - 81s 50ms/step - loss: 0.6350 - sparse_categorical_accuracy: 0.6133 - val_loss: 0.4307 - val_sparse_categorical_accuracy: 0.8206
Epoch 2/3
1563/1563 [==============================] - 77s 49ms/step - loss: 0.3256 - sparse_categorical_accuracy: 0.8674 - val_loss: 0.3166 - val_sparse_categorical_accuracy: 0.8699
Epoch 3/3
1563/1563 [==============================] - 77s 49ms/step - loss: 0.2482 - sparse_categorical_accuracy: 0.9009 - val_loss: 0.2934 - val_sparse_categorical_accuracy: 0.8816

<keras.callbacks.History at 0x7f7fc0fee760>

```
</div>
Excitingly, our custom classifier is similar to the performance of fine-tuning
`"bert_tiny_en_uncased"`! To see the advantages of pretraining and exceed 90% accuracy we
would need to use larger **presets** such as `"bert_base_en_uncased"`.
