# The Keras ecosystem

The Keras project isn't limited to the core Keras API for building and training neural networks.
It spans a wide range of related initiatives that cover every step of the machine learning workflow.

---

## KerasTuner

[KerasTuner Documentation](/keras_tuner/) - [KerasTuner GitHub repository](https://github.com/keras-team/keras-tuner)

KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. Easily configure your search space with a define-by-run syntax, then leverage one of the available search algorithms to find the best hyperparameter values for your models. KerasTuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms.

---

## KerasHub

[KerasHub Documentation](/keras_hub/) - [KerasHub GitHub repository](https://github.com/keras-team/keras-hub)

KerasHub is a natural language processing library that supports users through
their entire development cycle. Our workflows are built from modular components 
that have state-of-the-art preset weights and architectures when used 
out-of-the-box and are easily customizable when more control is needed.

---

## KerasCV

[KerasCV Documentation](/keras_cv/) - [KerasCV GitHub repository](https://github.com/keras-team/keras-cv)

KerasCV is a repository of modular building blocks (layers, metrics, losses, data-augmentation) that applied computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party Keras objects (layers, metrics, etc) that are too specialized to be added to core Keras, but that receive the same level of polish and backwards compatibility guarantees as the rest of the Keras API.

---

## AutoKeras

[AutoKeras Documentation](https://autokeras.com/) - [AutoKeras GitHub repository](https://github.com/keras-team/autokeras)

AutoKeras is an AutoML system based on Keras. It is developed by [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html) at Texas A&M University.
The goal of AutoKeras is to make machine learning accessible for everyone. It provides high-level end-to-end APIs
such as [`ImageClassifier`](https://autokeras.com/tutorial/image_classification/) or
[`TextClassifier`](https://autokeras.com/tutorial/text_classification/) to solve machine learning problems in a few lines,
as well as [flexible building blocks](https://autokeras.com/tutorial/customized/) to perform architecture search.

```python
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```
