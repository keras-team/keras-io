"""
Title: KerasNLP quick tour
Author: [jbischof](https://github.com/jbischof)
Date created: 2022-12-15
Last modified: 2022-12-15
Description: An introduction to the KerasNLP API.
"""
"""
<a
href="https://colab.research.google.com/github/jbischof/keras-io/blob/quickstart/guides/ke
ras_nlp/keras_nlp_quick_tour.ipynb" target="_parent"><img
src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import keras_nlp
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Use mixed precision for optimal performance
keras.mixed_precision.set_global_policy("mixed_float16")

"""
# KerasNLP: Modular NLP Workflows for Keras


`keras-nlp` is a natural language processing library that supports users through their
entire development cycle. Our workflows are built from modular components that have SoTA
preset weights and architectures when used out-of-the-box and are easily customizable
when more control is needed.

This library is an extension of the core `keras` API; all high level modules are `Layers`
or `Models`. If you are familiar with `keras`, congratulations! You already understand
most of `keras-nlp`.

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

![picture](https://drive.google.com/uc?id=1d14Qpmfgjf6zu4z30HBaonH8PYDHgVoU)


"""

"""
# API quickstart

Our highest level API is `keras_nlp.models`. For each `XX` architecture (e.g., `Bert`),
we offer the following modules:
* **Tokenizer**: `keras_nlp.models.XXTokenizer`
    * Maps raw text to `tf.RaggedTensor`s of token ids.
    * Inherits from `keras.Layer`.
* **Preprocessor**: `keras_nlp.models.XXPreprocessor`
    * Maps raw text to a dictonary of dense tensors consumed by the model.
    * Has a `XXTokenizer`.
    * Inherits from `keras.Layer`.
* **Backbone**: `keras_nlp.models.XXBackbone`
    * Maps preprocessed tensors to dense representation. *Does not handle raw text*.
    * Inherits from `keras.Model`.
* **Task**: e.g., `keras_nlp.models.XXClassifier`
    * Maps raw text to task-specific output (e.g., classification probabilities).
    * Has a `XXBackbone` and `XXPreprocessor`.
    * Inherits from `keras.Model`.

Here is the modular hierarchy for `BertClassifier` (all relationships are compositional):

![picture](https://drive.google.com/uc?id=1vHBQ1oFbto8ItfhsLcxKhIwOIdJE1X9n)

All modules can be used independently and have a `from_preset()` method in addition to
the standard constructor that instantiates the class with **preset** architecture and
weights (see examples below).
"""

"""
# Data

We will use a running example of sentiment analysis of IMDB movie reviews. In this task,
we use the text to predict whether the review was positive (`label = 1`) or negative
(`label = 0`).

We load the data from `tensorflow_datasets`, a collection of machine learning benchmarks
that uses the powerful `tf.data.Dataset` format for examples.
"""

BATCH_SIZE = 16
imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=BATCH_SIZE,
)

# Inspect first review
# Format is (review text tensor, label tensor)
imdb_train.unbatch().take(1).get_single_element()

"""
# Inference with a pretrained classifier

![picture](https://drive.google.com/uc?id=1xeMHVCxYhm3_oC37Gg7k0bG-yhsVr0Dv)

The highest level module in `keras-nlp` is a **task**. A **task** is a `keras.Model`
consisting of a (generally pretrained) **backbone** model and task-specific layers.
Here's an example using `keras_nlp.models.BertClassifier`.

**Note**: Outputs are the logits per class (`[0, 0]` is 50% chance of positive).


"""

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
# Note: batched inputs expected so must wrap string in iterable
classifier.predict(["I love modular workflows in keras-nlp!"])

"""

"""

"""
All **tasks** have a `from_preset` method that constructs a `keras.Model` instance with
preset preprocessing, architecture and weights. This means that we can pass raw strings
in any format accepted by a `keras.Model` and get output specific to our task.

This particular **preset** is a `bert_tiny_uncased_en` **backbone** fine-tuned on `sst2`,
another movie review sentiment analysis (this time from Rotten Tomatoes). We use the
`tiny` architecture for demo purposes, but larger models are recommended for SoTA
performance. For all the task-specific presets available for `BertClassifier`, see
[keras.io](https://resilient-dango-43f7b8.netlify.app/api/keras_nlp/models/).

Let's evaluate our classifier on the IMDB dataset. We first need to compile the
`keras.Model`. Since we are not training, we do not need a `loss` argument.
"""

classifier.compile(
    metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)

classifier.evaluate(imdb_test)

"""
# Fine tuning a pretrained BERT backbone

![picture](https://drive.google.com/uc?id=1YytOYRSqsrhJ4NLatVOSuVMbLPa9iXrw)

When labeled text specific to our task is available, fine-tuning a custom classifier can
improve performance. If we want to predict IMDB review sentiment, using IMDB data should
perform better than Rotten Tomatoes data! And for many tasks no relevant pretrained model
will be available (e.g., categorizing customer reviews).

The workflow for fine-tuning is almost identical to above, except that we request a
**preset** for the **backbone**-only model rather than the entire classifier. When passed
a **backone** **preset**, a **task** `Model` will randomly initialize all task-specific
layers in preparation for training. For all the **backbone** presets available for
`BertClassifier`, see
[keras.io](https://resilient-dango-43f7b8.netlify.app/api/keras_nlp/models/).

To train your classifier, use `Model.compile()` and `Model.fit()` as with any other
`keras.Model`. Since preprocessing is included in all **tasks** by default, we again pass
the raw data.

"""

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

"""
Here we see significant lift in validation accuracy (0.78 -> 0.87) with a single epoch of
training even though the IMDB dataset is much smaller than `sst2`.

"""

"""
# Fine tuning with user-controlled preprocessing
![picture](https://drive.google.com/uc?id=1T_40vtl8daihS-kKYTFWejFd19KJAyDK)

For some advanced training scenarios, users might prefer direct control over
preprocessing. For large datasets, examples can be preprocessed in advance and saved to
disk or preprocessed by a separate worker pool using `tf.data.experimental.service`. In
other cases, custom preprocessing is needed to handle the inputs.

Pass `preprocessor=None` to the constructor of a **task** `Model` to skip automatic
preprocessing or supply your own `keras.Layer` to perform a custom operation instead.


"""

"""
## Separate preprocessing from the same preset

Each model architecture has a parallel **preprocessor** `Layer` with its own
`from_preset` constructor. Using the same **preset** for this `Layer` will return the
matching **preprocessor** as the **task**.

In this workflow we train the model over three epochs using `tf.data.Dataset.cache()`,
which computes the preprocessing once and caches the result before fitting begins.

**Note:** this code only works if your data fits in memory. If not, pass a `filename` to
`cache()`.
"""

preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")

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

"""
After three epochs, our validation accuracy has only increased to 0.88. This is mainly a
function of the small size of our dataset; even with the `bert_tiny` architecture we've
already learned most generalizable patterns in the first pass.
"""

"""
## Custom preprocessing

In cases where custom preprocessing is required, we offer direct access to the
`Tokenizer` class that maps raw strings to tokens. It also has a `from_preset`
constructor to get the vocabulary matching pretraining.

**Note:** `Tokenizer` does not pad sequences, so output is `tf.RaggedTensor`.


"""

tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
tokenizer(["I love modular workflows!", "Libraries over frameworks!"])

# Write your own packer or use one our `Layers`
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    sequence_length=64,
)


def preprocess(x, y):
    token_ids, segment_ids = packer(tokenizer(x))
    x = {
        "token_ids": token_ids,
        "segment_ids": segment_ids,
        "padding_mask": token_ids != 0,
    }
    return x, y


imbd_train_preprocessed = imdb_train.map(preprocess, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
imdb_test_preprocessed = imdb_test.map(preprocess, tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# Preprocessed example
imbd_train_preprocessed.unbatch().take(1).get_single_element()

"""
# Fine tuning with a custom model
![picture](https://drive.google.com/uc?id=1T_40vtl8daihS-kKYTFWejFd19KJAyDK)

For more advanced applications, an appropriate **task** `Model` may not be available. In
this case we provide direct access to the **backbone** `Model`, which has its own
`from_preset` constructor and can be composed with custom `Layer`s. Detailed examples can
be found at https://keras.io/guides/transfer_learning/.

A **backbone** `Model` does not include automatic preprocessing but can be paired with a
matching **preprocessor** using the same **preset** as shown in the previous workflow.

In this workflow we experiment with freezing our backbone model and adding two trainable
transfomer layers to adapt to the new input.

**Note**: We can igonore the warning about gradients for the `pooled_dense` layer because
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
    )(sequence, padding_mask=inputs["padding_mask"])
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
This model achieves reasonable accuracy despite having only 10% the trainable parameters
of our `BertClassifier` model. Each training step takes about 1/3 of the time---even
accounting for cached preprocessing.
"""

"""
# Pretraining a backbone model
![picture](https://drive.google.com/uc?id=1pzwLPCtvzmHY3DKzH-MBzmjWFJ3pKVB5)

Do you have access to large unlabeled datasets in your domain? Are they are around the
same size as used to train popular backbones such as BERT, RoBERTa, or GPT2 (XX+ GiB)? If
so, you might benefit from domain-specific pretraining of your own backbone models.

NLP models are generally pretrained on a language modeling task, predicting masked words
given the visible words in an input sentence. For example, given the input `"The fox
[MASK] over the [MASK] dog"`, the model might be asked to predict `["jumped", "lazy"]`.
The lower layers of this model are then packaged as a **backbone** to be combined with
layers relating to a new task.

The `keras-nlp` library offers SoTA **backbones** and **tokenizers** to be trained from
scratch without presets.

In this workflow we pretrain a BERT **backbone** using our IMDB review text. We skip the
"next sentence prediction" (NSP) loss because it adds significant complexity to the data
processing and was dropped by later models like RoBERTa. See our e2e [BERT pretraining
example](https://github.com/keras-team/keras-nlp/tree/4f9ebefa82af22b4f4267dfa80fa525f7a03
bd5d/examples/bert) for step-by-step details on how to replicate the original paper.
"""

"""
## Preprocessing
"""

# All BERT `en` models have the same vocabulary, so reuse preprocessor from
# "bert_tiny_en_uncased"
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased",
    sequence_length=128,
)
packer = preprocessor.packer
tokenizer = preprocessor.tokenizer

# keras.Layer to replace some input tokens with the "[MASK]" token
masker = keras_nlp.layers.MLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=0.25,
    mask_selection_length=32,
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
pretrain_ds.unbatch().take(1).get_single_element()

"""
## Pretraining model
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
mlm_head = keras_nlp.layers.MLMHead(
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
# Build and train your own transformer from scratch
![picture](https://drive.google.com/uc?id=1pzwLPCtvzmHY3DKzH-MBzmjWFJ3pKVB5)

Want to implement a novel transformer architecture? The `keras-nlp` library offers all
the low-level modules used to build SoTA architectures in our `models` API. This includes
training your own subword tokenizer using `WordPiece`, `BytePairEncoder`, or
`SentencePiece`.

In this workflow we train a custom tokenizer on the IMDB data and design a backbone with
custom transformer architecture. For simplicity we then train directly on the
classification task. Interested in more details? We wrote an entire guide to pretraining
and finetuning a custom transformer:
https://keras.io/guides/keras_nlp/transformer_pretraining/
"""

"""
## Train custom vocabulary from IMBD data
"""

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    imdb_train.map(lambda x, y: x),
    vocabulary_size=10_000,  # Increase to 20_000 for better performance
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

## Preprocess data with custom tokenizer
"""

packer = keras_nlp.layers.StartEndPacker(
    start_value=tokenizer.token_to_id("[START]"),
    end_value=tokenizer.token_to_id("[END]"),
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=64,
)


def preprocess(x, y):
    token_ids = packer(tokenizer(x))
    x = {
        "token_ids": token_ids,
        "padding_mask": token_ids != tokenizer.token_to_id("[PAD]"),
    }
    return x, y


imdb_preproc_train_ds = imdb_train.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
imdb_preproc_val_ds = imdb_test.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

imdb_preproc_train_ds.unbatch().take(1).get_single_element()

"""

## Design a tiny transformer
"""

token_id_input = keras.Input(
    shape=(None,),
    dtype="int32",
    name="token_ids",
)
padding_mask = keras.Input(
    shape=(None,),
    dtype="int32",
    name="padding_mask",
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
)(outputs, padding_mask=padding_mask)
# Use "[START]" token to classify
outputs = keras.layers.Dense(2)(outputs[:, 0, :])
model = keras.Model(
    inputs={
        "token_ids": token_id_input,
        "padding_mask": padding_mask,
    },
    outputs=outputs,
)

model.summary()

"""
## Train the transformer directly on the classification objective
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
While our classification accuracy is a fairly poor 0.76, the transformer architecture is
too complicated to learn from scratch on a small dataset. The large performance gap with
our earlier models shows the power of pretraining and transfer learning in modern NLP.
"""
