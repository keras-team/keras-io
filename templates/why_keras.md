# Why choose Keras?

Here are some of the areas in which Keras compares favorably to existing alternatives.


## Keras is one of the most widely used Machine Learning frameworks

With around 2.5 million developers as of early 2023, Keras is at the center of a massive community and ecosystem.

You are already constantly interacting with features built with Keras -- it is in use at YouTube, Netflix, Uber, Yelp, Instacart, Zocdoc, Twitter, Square/Block, and many others.
Keras is especially popular among startups that place deep learning at the core of their products.
Keras is also used in many well-known companies you might not associate with Machine Learning, such as JP Morgan Chase, Orange, and Comcast,
and by research units at the likes of NASA, the US DOE, and CERN.

In the 2022 survey "State of Data Science and Machine Learning" by Kaggle,
Keras had a 61% adoption rate among Machine Learning developers and Data Scientists [[source](https://www.kaggle.com/kaggle-survey-2022)].

![ML frameworks adoption stats](/img/kaggle-2022-adoption.png)


---

## Keras prioritizes developer experience
    
- Keras is an API designed for human beings, not machines. [Keras follows best practices for reducing cognitive load](https://blog.keras.io/user-experience-design-for-apis.html): it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.
- This makes Keras easy to learn and easy to use. As a Keras user, you are more productive, allowing you to try more ideas than your competition, faster -- which in turn [helps you win machine learning competitions](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).
- This ease of use does not come at the cost of reduced flexibility: because Keras integrates deeply with low-level TensorFlow functionality, it enables you to develop highly hackable workflows where any piece of functionality can be customized.


---


## Keras makes it easy to turn models into products

Your Keras models can be easily deployed across a greater range of platforms than any other deep learning API:

- On server via a Python runtime or a Node.js runtime
- On server via TFX / TF Serving
- In the browser via TF.js
- On Android or iOS via TF Lite or Apple's CoreML
- On Raspberry Pi, on Edge TPU, or another embedded system


---

## Keras has strong multi-GPU & distributed training support

Keras is scalable. Using the [TensorFlow `DistributionStrategy` API](https://www.tensorflow.org/tutorials/distribute/keras), which is supported natively by Keras,
you easily can run your models on large GPU clusters (up to thousands of devices) or an entire TPU pod, representing over one exaFLOPs of computing power.

Keras also has native support for mixed-precision training on the latest NVIDIA GPUs as well as on TPUs, which can offer up to 2x speedup for training and inference.

For more information on data-parallel training, see our [guide to multi-GPU & distributed training](/guides/distributed_training/).

---

## Keras is at the nexus of a large ecosystem

Like you, we know firsthand that building and training a model is only one slice of a machine learning workflow. Keras is built for the real world,
and in the real world, a successful model begins with data collection and ends with production deployment. 

Keras is at the center of a wide ecosystem of tightly-connected projects that together cover every step of the machine learning workflow, in particular:

- Natural Language Processing (NLP) workflows with [KerasNLP](/keras_nlp/)
- Computer Vision (CV) workflows with [KerasCV](/keras_cv/)
- Hyperparameter tuning with [KerasTuner](/keras_tuner/)
- Gradient boosting and decision forests with [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)
- Recommender systems with [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)
- Rapid model prototyping with [AutoKeras](https://autokeras.com/)
- Inference model quantization & pruning with the [TF Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- Model deployment on mobile or on an embedded with [TF Lite](https://www.tensorflow.org/lite)
- Model deployment in the browser via [TF.js](https://www.tensorflow.org/js)
- ...and many more.

Learn more about the Keras ecosystem [here](/getting_started/ecosystem/).
