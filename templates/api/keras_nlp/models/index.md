# KerasNLP Models

KerasNLP contains end-to-end implementations of popular model architectures.
These models can be created in two ways:

- Through the `from_preset()` constructor, which instantiates an object with
  a pre-trained configurations, vocabularies, and (optionally) weights.
- Through custom configuration controlled by the user.

Below, we list all presets available in the library. For more detailed usage,
browse the docstring for a particular class. For an in depth introduction
to our API, see the [getting started guide](guides/keras_nlp/getting_started/).

## Backbone presets

The following preset names correspond to a configuration, weights and vocabulary
for a model **backbone**. These presets are not inference-ready, and must be
fine-tuned for a given task!

The names below can be used with any `from_preset()` constructor for a given model.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased")
backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
```

{{backbone_presets_table}}

## Classification presets

The following preset names correspond to a configuration, weights and vocabulary
for a model **classifier**. These models are inference ready, but can be further
fine-tuned if desired.

The names below can be used with the `from_preset()` constructor for classifier models
and preprocessing layers.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased_sst2")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased_sst2")
```

{{classifier_presets_table}}

## API Documentation

{{toc}}
