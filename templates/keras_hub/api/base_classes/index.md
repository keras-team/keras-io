# KerasHub Modeling API

The following base classes form the API for working with pretrained models
through KerasHub. These base classes can be used with the `from_preset()`
constructor to automatically instantiate a subclass with the correct model
architecture, e.g.
`keras_hub.models.TextClassifier.from_preset("bert_base_en", num_classes=2)`.

For the full list of available pretrained model presets shipped directly by the
Keras team, see the [Pretrained Models](/keras_hub/presets/) page.

{{toc}}
