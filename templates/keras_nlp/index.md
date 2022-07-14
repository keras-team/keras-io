# KerasNLP

KerasNLP is a simple and powerful API for building Natural Language Processing
(NLP) models within the Keras ecosystem.

KerasNLP provides modular building blocks following standard Keras interfaces
(layers, metrics) that allow you to quickly and flexibly iterate on your task.
Engineers working in applied NLP can leverage the library to assemble training
and inference pipelines that are both state-of-the-art and production-grade.

KerasNLP can be understood as a horizontal extension of the Keras API:
components are first-party Keras objects that are too specialized to be
added to core Keras, but that receive the same level of polish as the rest of
the Keras API.

KerasNLP is also new and growing! If you are interested in contributing, please
check out our
[contributing guide](https://github.com/keras-team/keras-nlp/blob/master/CONTRIBUTING.md).

---
## Quick links

* [KerasNLP API reference](/api/keras_nlp/)
* [KerasNLP on GitHub](https://github.com/keras-team/keras-nlp)

---
## Guides

* [Pretraining a Transformer from scratch](/guides/keras_nlp/transformer_pretraining/)

---
## Examples

* [English-to-Spanish translation](/examples/nlp/neural_machine_translation_with_keras_nlp/)
* [Text Classification using FNet](/examples/nlp/fnet_classification_with_keras_nlp/)

---
## Installation

KerasNLP requires **Python 3.7+** and **TensorFlow 2.9+**.

Install the latest release:

```
pip install keras-nlp --upgrade
```

You can check out release notes and versions on our
[releases page](https://github.com/keras-team/keras-nlp/releases).

KerasNLP is currently in pre-release (0.y.z) development. Until version 1.0, we
may break compatibility at any time and APIs should not be considered stable.

---
## Quick introduction

The following snippet will tokenize some text, build a tiny transformer, and
train a single batch.

```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Tokenize some inputs with a binary label.
vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
sentences = ["The quick brown fox jumped.", "The fox slept."]
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=10,
)
x, y = tokenizer(sentences), tf.constant([1, 0])

# Create a tiny transformer.
inputs = keras.Input(shape=(None,), dtype="int32")
outputs = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=len(vocab),
    sequence_length=10,
    embedding_dim=16,
)(inputs)
outputs = keras_nlp.layers.TransformerEncoder(
    num_heads=4,
    intermediate_dim=32,
)(outputs)
outputs = keras.layers.GlobalAveragePooling1D()(outputs)
outputs = keras.layers.Dense(1, activation="sigmoid")(outputs)
model = keras.Model(inputs, outputs)

# Run a single batch of gradient descent.
model.compile(optimizer="rmsprop", loss="binary_crossentropy", jit_compile=True)
model.train_on_batch(x, y)
```

To see an end-to-end example using KerasNLP, check out our guide on
[pre-training a transfomer from scratch](/guides/keras_nlp/transformer_pretraining/).

---
## Citing KerasNLP

If KerasNLP helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasnlp2022,
  title={KerasNLP},
  author={Watson, Matthew, and Qian, Chen, and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```
