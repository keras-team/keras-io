# The Keras ecosystem

The Keras project isn't limited to the core Keras API for building and training neural networks.
It spans a wide range of related initiatives that cover every step of the machine learning workflow.

---

## KerasTuner

[KerasTuner Documentation](/keras_tuner/) - [KerasTuner GitHub repository](https://github.com/keras-team/keras-tuner)

KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. Easily configure your search space with a define-by-run syntax, then leverage one of the available search algorithms to find the best hyperparameter values for your models. KerasTuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms.

---

## KerasNLP

[KerasNLP Documentation](/keras_nlp/) - [KerasNLP GitHub repository](https://github.com/keras-team/keras-nlp)

KerasNLP is a simple and powerful API for building Natural Language
Processing (NLP) models. KerasNLP provides modular building blocks following
standard Keras interfaces (layers, metrics) that allow you to quickly and
flexibly iterate on your task. Engineers working in applied NLP can leverage the
library to assemble training and inference pipelines that are both
state-of-the-art and production-grade. KerasNLP is maintained directly by the
Keras team.

---

## AutoKeras

[AutoKeras Documentation](https://autokeras.com/) - [AutoKeras GitHub repository](https://github.com/keras-team/autokeras)

AutoKeras is an AutoML system based on Keras. It is developed by [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html) at Texas A&M University.
The goal of AutoKeras is to make machine learning accessible for everyone. It provides high-level end-to-end APIs
such as [`ImageClassifier`](https://autokeras.com/tutorial/image_classification/) or
[`TextClassifier`](https://autokeras.com/tutorial/text_classification/) to solve machine learning problems in a few lines,
as well as [flexible building blocks](https://autokeras.com/tutorial/customized/) to perform architecture search.

---

## KerasCV

[KerasCV Documentation](/keras_cv/) - [KerasCV GitHub repository](https://github.com/keras-team/keras-cv)

KerasCV is a repository of modular building blocks (layers, metrics, losses, data-augmentation) that applied computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party Keras objects (layers, metrics, etc) that are too specialized to be added to core Keras, but that receive the same level of polish and backwards compatibility guarantees as the rest of the Keras API and that are maintained by the Keras team itself (unlike TFAddons).

---

## TensorFlow Cloud

Managed by the Keras team at Google, [TensorFlow Cloud](https://github.com/tensorflow/cloud) is a set of utilities to help you run large-scale
Keras training jobs on GCP with very little configuration effort. Running your experiments on 8 or more GPUs in the cloud
should be as easy as calling `model.fit()`.

---

## TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js) is TensorFlow's JavaScript runtime, capable of running TensorFlow models in the browser or on a [Node.js](https://nodejs.org/en/) server,
both for training and inference. It natively supports loading Keras models, including the ability to fine-tune or retrain your Keras models directly in the browser.


---

## TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite) is a runtime for efficient on-device inference that has native support for Keras models.
Deploy your models on Android, iOS, or on embedded devices.


---

## Model optimization toolkit

The [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) is a set of utilities to make your inference models faster, more memory-efficient,
and more power-efficient, by performing *post-training weight quantization* and *pruning-aware training*.
It has native support for Keras models, and its pruning API is built directly on top on the Keras API.


---

## TFX integration

TFX is an end-to-end platform for deploying and maintaining production machine learning pipelines.
TFX has [native support for Keras models](https://www.tensorflow.org/tfx/guide/keras).


---
