# KerasNLP Preprocessing Layers

KerasNLP preprocessing layers are `keras.Layer` that help with data
preparation: static transformations of your data that happen before the
learnable part of your model.

Preprocessing layers for NLP tasks should generally be run inside a
[tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
during training.

{{toc}}
