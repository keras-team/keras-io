# KerasNLP

<a class="github-button" href="https://github.com/keras-team/keras-nlp" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-nlp on GitHub">Star</a>

KerasNLP is a natural language processing library that supports users through
their entire development cycle. Our workflows are built from modular components
that have state-of-the-art preset weights and architectures when used
out-of-the-box and are easily customizable when more control is needed. We
emphasize in-graph computation for all workflows so that developers can expect
easy productionization using the TensorFlow ecosystem.

This library is an extension of the core Keras API; all high-level modules are
[`Layers`](/api/layers/) or [`Models`](/api/models/) that recieve that same
level of polish as core Keras. If you are familiar with Keras, congratulations!
You already understand most of KerasNLP.

See our [Getting Started guide](/guides/keras_nlp/getting_started)
for example usage of our modular API starting with evaluating pretrained models
and building up to designing a novel transformer architecture and training a
tokenizer from scratch.

KerasNLP is new and growing! If you are interested in contributing, please
check out our
[contributing guide](https://github.com/keras-team/keras-nlp/blob/master/CONTRIBUTING.md).

---
## Quick links

* [KerasNLP API reference](/api/keras_nlp/)
* [KerasNLP on GitHub](https://github.com/keras-team/keras-nlp)
* [List of available models and presets](/api/keras_nlp/models/)
---
## Guides

* [Getting Started with KerasNLP](/guides/keras_nlp/getting_started/)
* [Pretraining a Transformer from scratch](/guides/keras_nlp/transformer_pretraining/)


---
## Examples

* [English-to-Spanish translation](/examples/nlp/neural_machine_translation_with_keras_nlp/)
* [GPT2 text generation](/examples/generative/gpt2_text_generation_with_kerasnlp/)
* [GPT text generation from scratch](/examples/generative/text_generation_gpt/)
* [Text Classification using FNet](/examples/nlp/fnet_classification_with_keras_nlp/)

---
## Installation

To install the latest official release:

```
pip install keras-nlp --upgrade
```

To install the latest unreleased changes to the library, we recommend using
pip to install directly from the master branch on github:

```
pip install git+https://github.com/keras-team/keras-nlp.git --upgrade
```

## Quickstart

Fine-tune BERT on a small sentiment analysis task using the
[`keras_nlp.models`](/api/keras_nlp/models/) API:

```python
import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased")
# Fine-tune on IMDb movie reviews.
classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
classifier.predict(["What an amazing movie!", "A total waste of my time."])
```

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to
provide backwards compatibility guarantees both for code and saved models built
with our components. While we continue with pre-release `0.y.z` development, we
may break compatibility at any time and APIs should not be consider stable.

## Disclaimer

KerasNLP provides access to pre-trained models via the `keras_nlp.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind. The following underlying models are provided by third
parties, and subject to separate licenses:
DistilBERT, RoBERTa, XLM-RoBERTa, DeBERTa, and GPT-2.

## Citing KerasNLP

If KerasNLP helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasnlp2022,
  title={KerasNLP},
  author={Watson, Matthew, and Qian, Chen, and Bischof, Jonathan and Chollet, 
  Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```
