# KerasHub pretrained models

Below, we list all presets available in the KerasHub library. For more detailed
usage, browse the docstring for a particular class. For an in depth introduction
to our API, see the [getting started guide](/keras_hub/getting_started/).

The following preset names correspond to a config and weights for a pretrained
model. Any task, preprocessor, backbone, or tokenizer `from_preset()` can be used
to create a model from the saved preset.

```python
backbone = keras_hub.models.Backbone.from_preset("bert_base_en")
tokenizer = keras_hub.models.Tokenizer.from_preset("bert_base_en")
classifier = keras_hub.models.TextClassifier.from_preset("bert_base_en", num_classes=2)
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset("bert_base_en")
```

{{presets_table}}
