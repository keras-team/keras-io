"""
Title: Getting Started with KerasNLP
Author: [Jonathan Bischof](https://github.com/jbischof)
Date created: 2022/12/15
Last modified: 2022/12/15
Description: An introduction to the KerasNLP API.
Accelerator: GPU
"""
"""
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
"""

"""shell
pip install -q --upgrade keras-nlp tensorflow
"""

import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Use mixed precision for optimal performance
keras.mixed_precision.set_global_policy("mixed_float16")

"""
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
"""

"""
## Data

We will use a running example of sentiment analysis of IMDB movie reviews. In this task,
we use the text to predict whether the review was positive (`label = 1`) or negative
(`label = 0`).

We load the data using `keras.utils.text_dataset_from_directory`, which utilizes the
powerful `tf.data.Dataset` format for examples.
"""

"""shell
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
# Remove unsupervised examples
rm -r aclImdb/train/unsup
"""

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


"""
## Inference with a pretrained classifier

<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_beginner.png" alt="drawing" height="250"/>

The highest level module in KerasNLP is a **task**. A **task** is a `keras.Model`
consisting of a (generally pretrained) **backbone** model and task-specific layers.
Here's an example using `keras_nlp.models.BertClassifier`.

**Note**: Outputs are the logits per class (e.g., `[0, 0]` is 50% chance of positive). The output is
[negative, positive] for binary classification.
"""

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
# Note: batched inputs expected so must wrap string in iterable
classifier.predict(["I love modular workflows in keras-nlp!"])

"""
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
"""

classifier.evaluate(imdb_test)

"""
Our result is 78% accuracy without training anything. Not bad!
"""

"""
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
"""

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
classifier.fit(
    imdb_train,
    validation_data=imdb_test,
    epochs=1,
)

"""
Here we see a significant lift in validation accuracy (0.78 -> 0.87) with a single epoch of
training even though the IMDB dataset is much smaller than `sst2`.
"""

"""
## Fine tuning with user-controlled preprocessing
<img src="https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png" alt="drawing" height="250"/>

For some advanced training scenarios, users might prefer direct control over
preprocessing. For large datasets, examples can be preprocessed in advance and saved to
disk or preprocessed by a separate worker pool using `tf.data.experimental.service`. In
other cases, custom preprocessing is needed to handle the inputs.

Pass `preprocessor=None` to the constructor of a **task** `Model` to skip automatic
preprocessing or pass a custom `BertPreprocessor` instead.
"""

"""
### Separate preprocessing from the same preset

Each model architecture has a parallel **preprocessor** `Layer` with its own
`from_preset` constructor. Using the same **preset** for this `Layer` will return the
matching **preprocessor** as the **task**.

In this workflow we train the model over three epochs using `tf.data.Dataset.cache()`,
which computes the preprocessing once and caches the result before fitting begins.

**Note:** this code only works if your data fits in memory. If not, pass a `filename` to
`cache()`.
"""

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

"""
After three epochs, our validation accuracy has only increased to 0.88. This is both a
function of the small size of our dataset and our model. To exceed 90% accuracy, try
larger **presets** such as  `"bert_base_en_uncased"`. For all the **backbone** presets
available for `BertClassifier`, see our keras.io [models page](https://keras.io/api/keras_nlp/models/).
"""

"""
### Custom preprocessing

In cases where custom preprocessing is required, we offer direct access to the
`Tokenizer` class that maps raw strings to tokens. It also has a `from_preset()`
constructor to get the vocabulary matching pretraining.

**Note:** `BertTokenizer` does not pad sequences by default, so the output is
a `tf.RaggedTensor`.
"""

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

"""
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
"""

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

"""
This model achieves reasonable accuracy despite having only 10% of the trainable parameters
of our `BertClassifier` model. Each training step takes about 1/3 of the time---even
accounting for cached preprocessing.
"""

"""
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
"""

"""
### Preprocessing
"""

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

"""
### Pretraining model
"""

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

"""
After pretraining save your `backbone` submodel to use in a new task!
"""

"""
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
"""

"""
### Train custom vocabulary from IMDB data
"""

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

"""
### Preprocess data with a custom tokenizer
"""

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

"""

### Design a tiny transformer
"""

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

"""
### Train the transformer directly on the classification objective
"""

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

"""
Excitingly, our custom classifier is similar to the performance of fine-tuning
`"bert_tiny_en_uncased"`! To see the advantages of pretraining and exceed 90% accuracy we
would need to use larger **presets** such as `"bert_base_en_uncased"`.
"""
