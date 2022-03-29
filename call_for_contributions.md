# Call for code example contributions

This is a constantly-updated list of code examples that we're currently interested in.

If you're not sure whether your idea would make a good code example, please ask us first!

---

## Structured data examples featuring Keras Preprocessing Layers (KPL)

E.g. feature hashing, feature indexing with handling of missing values,
mixing numerical, categorical, and text features, doing feature engineering with KPL, etc.

---

## Transformer model for MIDI music generation

[Reference TF/Keras implementation](https://github.com/jason9693/MusicTransformer-tensorflow2.0)

---

## StyleGAN2

- [Paper](https://arxiv.org/abs/1912.04958)
- [Reference TF implementation](https://github.com/NVlabs/stylegan2)


---

## Text-to-speech

[Example TF2/Keras implementation](https://github.com/dathudeptrai/TensorflowTTS)

---

## Learning to rank

[Reference Kaggle competition](https://www.kaggle.com/c/wm-2017-learning-to-rank)


---

## DETR: End-to-End Object Detection with Transformers

- [Reference implementation](https://github.com/facebookresearch/detr)
- [TF/Keras implementation](https://github.com/Visual-Behavior/detr-tensorflow)


---

## 3D image segmentation

---

## Question answering from structured knowledge base and freeform documents

---

## Instance segmentation

- [Tensorflow-YOLACT](https://github.com/leohsuofnthu/Tensorflow-YOLACT)
- [Additional references](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/278883#1546104)

---

## EEG & MEG signal classification

---

## Text summarization

---

## Audio track separation

---

## Audio style transfer

---

## Timeseries imputation

---

## Customer lifetime value prediction

---

## Keras reproducibility recipes

---

## Standalone Mixture-of-Experts (MoE) layer

MoE layers provide a flexible way to scale deep models to train on larger datasets. The aim of this example should be to show 
how replace the regular layers (such as `Dense`, `Conv2D`) with compatible MoE layers. 

References:

* A relevant paper on MoE: https://arxiv.org/abs/1701.06538
* [Switch Transformers on keras.io](https://keras.io/examples/nlp/text_classification_with_switch_transformer/)
* [Keras implementation of Dense and Conv2D MoE layers](https://github.com/eminorhan/mixture-of-experts)

---

## Guide to report the efficiency of a Keras model

It's often important to report the efficiency of a model. But what factors should be included when reporting the efficiency
of a deep learning model? [The Efficiency Misnomer](https://openreview.net/forum?id=iulEMLYh1uR) paper discusses this thoroughly and provides guidelines for practitioners on how to properly report model efficiency. 

The objectives of this guide will include the following:

* What factors to consider when reporting model efficiency?
* How to calculate certain metrics like FLOPS, number of examples a model can process per second (both in training and inference mode), etc?
