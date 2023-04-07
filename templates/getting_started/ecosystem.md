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

KerasNLP is a natural language processing library that supports users through
their entire development cycle. Our workflows are built from modular components 
that have state-of-the-art preset weights and architectures when used 
out-of-the-box and are easily customizable when more control is needed. We 
emphasize in-graph computation for all workflows so that developers can expect 
easy productionization using the TensorFlow ecosystem.

---

## KerasCV

[KerasCV Documentation](/keras_cv/) - [KerasCV GitHub repository](https://github.com/keras-team/keras-cv)

KerasCV is a repository of modular building blocks (layers, metrics, losses, data-augmentation) that applied computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party Keras objects (layers, metrics, etc) that are too specialized to be added to core Keras, but that receive the same level of polish and backwards compatibility guarantees as the rest of the Keras API and that are maintained by the Keras team itself (unlike TFAddons).

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

---

## TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js) is TensorFlow's JavaScript runtime, capable of running TensorFlow models in the browser or on a [Node.js](https://nodejs.org/en/) server,
both for training and inference. It natively supports loading Keras models, including the ability to fine-tune or retrain your Keras models directly in the browser.


---

## TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite) is a runtime for efficient on-device inference that has native support for Keras models.
Deploy your models on Android, iOS, or on embedded devices.


---

## Model Optimization Toolkit

The [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) is a set of utilities to make your inference models faster, more memory-efficient,
and more power-efficient, by performing *post-training weight quantization* and *pruning-aware training*.
It has native support for Keras models, and its pruning API is built directly on top on the Keras API.

```python
import tensorflow_model_optimization as tfmot

# Define a Keras model.
model = tf.keras.Sequential([...])

# Define a training-time pruning schedule.
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                      initial_sparsity=0.0, final_sparsity=0.5,
                      begin_step=2000, end_step=4000)

# Convert your Keras model to a pruning-optimized model.
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)

# Fit the optimized model.
model_for_pruning.fit(...)
```

---

## TFX integration

TFX is an end-to-end platform for deploying and maintaining production machine learning pipelines.
TFX has [native support for Keras models](https://www.tensorflow.org/tfx/guide/keras).

---

## TensorFlow Recommenders

[TensorFlow Recommenders](https://www.tensorflow.org/recommenders) is a library for building recommender system models, built on Keras.
It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.

```python
import tensorflow_recommenders as tfrs

# Build flexible representation models with Keras.
user_model = keras.Sequential([...])
movie_model = keras.Sequential([...])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)

# Create a retrieval model.
model = MovielensModel(user_model, movie_model, task)
model.compile(optimizer=keras.optimizers.Adagrad(0.5))

# Train.
model.fit(ratings.batch(4096), epochs=3)
```

---

## TensorFlow Decision Forests

TensorFlow Decision Forests is a library to train, run and interpret decision forest models
(e.g., Random Forests, Gradient Boosted Trees) in TensorFlow and Keras.
It supports classification, regression, ranking and uplifting.


```python
import tensorflow_decision_forests as tfdf

# Train a Random Forest model.
model = tfdf.keras.RandomForestModel()
model.fit(training_dataset)

# Summary of the model structure.
model.summary()

# Evaluate the model.
model.evaluate(test_dataset)

# Export the model to a SavedModel.
model.save("project/model")
```

---

## Model Remediation Toolkit

The TensorFlow / Keras [Model Remediation Toolkit](https://www.tensorflow.org/responsible_ai/model_remediation)
is a library of utilities for identifying and addressing fairness and bias issues in deep learning models,
built on top of the Keras API.

---
