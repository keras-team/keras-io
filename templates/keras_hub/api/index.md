# KerasHub API documentation

KerasHub is a toolbox of modular building blocks ranging from pretrained
state-of-the-art models, to low-level Transformer Encoder layers.

- **Modeling API**: Base classes that can be used for most high-level tasks
  using pretrained models. Note that you can use the `from_preset()`
  constructor on a base class to instantiate a model of the correct subclass.
- **Model Architectures**: Implementations of all pretrained model architectures
  shipped with KerasHub.
- **Tokenizers**: Layer implementations of tokenization routines for text-based
  models.
- **Preprocessing Layers**: Layers for building preprocessing pipelines
  that handle audio, text, and image input.
- **Modeling Layers**: Common modeling layers used by pretrained model
  architectures.
- **Samplers**: An API for controlling generative text sampling.
- **Metrics**: Metrics useful for audio, text, and image workflows.

{{toc}}
