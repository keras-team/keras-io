# Call for code example contributions

This is a constantly-updated list of code examples that we're currently interested in.


## Transformer model for MIDI music generation

[Reference TF/Keras implementation](https://github.com/jason9693/MusicTransformer-tensorflow2.0)


## StyleGAN / StyleGAN2

[Reference paper 1](https://arxiv.org/abs/1812.04948) [2](https://arxiv.org/abs/1912.04958)
[Reference TF implementation](https://github.com/NVlabs/stylegan2)

 - Currently being worked on by @parthivc and @Korusuke as of 07/14/20. See [issue #144](https://github.com/keras-team/keras-io/issues/144)

## Improve next-frame prediction with ConvLSTM

[Current example](https://keras.io/examples/vision/conv_lstm/)

Recommendations:

- Find a better, more interesting dataset
- Make it work better
- Add detailed explanations
- As of 6/23/20, this is being worked on. See [issue #112](https://github.com/keras-team/keras-io/issues/112)


## Text-to-speech

[Example TF2/Keras implementation](https://github.com/dathudeptrai/TensorflowTTS)


## Speech recognition

[Example TF2/Keras implementation](https://github.com/rolczynski/Automatic-Speech-Recognition)


## Learning to rank

[Reference Kaggle competition](https://www.kaggle.com/c/wm-2017-learning-to-rank)


## Large-scale multi-label text classification

Using word bi-grams + TF-IDF + a small MLP, from raw text strings.
The tokenization and extraction of TF-IDF ngrams should be done with the `TextVectorization` layer.

The dataset should have at least 50k samples and there should be at least a dozen of labels.


## DETR: End-to-End Object Detection with Transformers

[Reference implementation](https://github.com/facebookresearch/detr)
[TF/Keras implementation](https://github.com/auvisusAI/detr-tensorflow)


## 3D image classification

Using a dataset of CT scans (a few are available on Kaggle).

The model should use `Conv3D` layers.


## Image similarity search engine using triplet loss

A convnet trained with a triplet loss to estimate image similarity.
It could be trained on something like [this dataset](https://sites.google.com/view/totally-looks-like-dataset)
(other datasets are also possible).



