# KerasNLP

---
## Quick links

* [Developer guides](/guides/keras_nlp/)
* [API reference](/api/keras_nlp/)

---
## Installation

KerasNLP requires **Python 3.7+** and **TensorFlow 2.9+**.

Install the latest release:

```
pip install keras-nlp --upgrade
```

You can also check out release notes and other releases on our
[GitHub releases page](https://github.com/keras-team/keras-nlp/releases).

We follow Semantic Versioning, and will provide backwards compatibility for both
code and saved models. While we continue with pre-release 0.y.z development, we
may break compatibility at any time and APIs should not be consider stable.

---
## Quick introduction

Import KerasNLP and Keras:

```python
import keras_nlp
from tensorflow import keras
```

Sub-word tokenize a string:

```python
vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
inputs = "The quick brown fox."

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab)
tokenizer(inputs)
```

Build a tiny transformer:

```python
sequence_length = 100
vocab_size = 10000
model_dim = 64
intermediate_dim = 128
num_heads = 4

inputs = keras.Input(shape=(sequence_length,), dtype="int32")
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=vocab_size,
    max_length=sequence_length,
    embedding_dim=model_dim,
)(inputs)
x = keras_nlp.layers.TransformerEncoder(
    num_heads=num_heads,
    intermediate_dim=intermediate_dim,
)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.summary()
```

To learn more about KerasNLP, check out the
[transformer pretraining guide](/guides/keras_nlp/transformer_pretraining/).

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
