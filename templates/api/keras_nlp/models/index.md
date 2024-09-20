# KerasHub Models

KerasHub contains end-to-end implementations of popular model architectures.
These models can be created in two ways:

- Through the `from_preset()` constructor, which instantiates an object with
  a pre-trained configurations, vocabularies, and (optionally) weights.
- Through custom configuration controlled by the user.

Below, we list all presets available in the library. For more detailed usage,
browse the docstring for a particular class. For an in depth introduction
to our API, see the [getting started guide](/guides/keras_hub/getting_started/).

## Presets

The following preset names correspond to a config and weights for a pretrained
model. Any task, preprocessor, backbone or tokenizer `from_preset()` can be used
to create a model from the saved preset.

```python
backbone = keras_hub.models.Backbone.from_preset("bert_base_en")
tokenizer = keras_hub.models.Tokenizer.from_preset("bert_base_en")
classifier = keras_hub.models.TextClassifier.from_preset("bert_base_en", num_classes=2)
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset("bert_base_en")
```

{{backbone_presets_table}}

**Note**: The links provided will lead to the model card or to the official README,
if no model card has been provided by the author.

## API Documentation

{{toc}}
