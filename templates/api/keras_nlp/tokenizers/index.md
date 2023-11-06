# KerasNLP Tokenizers

Tokenizers convert raw string input into integer input suitable for a Keras `Embedding` layer.
They can also convert back from predicted integer sequences to raw string output.

All tokenizers subclass `keras_nlp.tokenizers.Tokenizer`, which in turn
subclasses `keras.layers.Layer`. Tokenizers should generally be applied inside a
[tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
for training, and can be included inside a `keras.Model` for inference.

{{toc}}
