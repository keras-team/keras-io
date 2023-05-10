# KerasNLP Models

KerasNLP contains end-to-end implementations of popular model
architectures. These models can be created in two ways:

- Through the `from_preset()` constructor, which instantiates an object with
  a pre-trained configurations, vocabularies, and (optionally) weights.
  Available preset names are listed on this page.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased")
```

- Through custom configuration controlled by the user. To do this, simply
  pass the desired configuration parameters to the default constructors of the
  symbols documented below.

```python
tokenizer = keras_nlp.models.BertTokenizer(
    vocabulary="./vocab.txt",
)
preprocessor = keras_nlp.models.BertPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_nlp.models.BertBackbone(
    vocabulary_size=30552,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=3072,
    max_sequence_length=128,
)
classifier = keras_nlp.models.BertClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=4,
)
```

For a more in depth introduction to how our API fits together, see the
[getting started guide](guides/keras_nlp/getting_started/).

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
