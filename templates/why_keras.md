# Why choose Keras?

There are many deep learning frameworks available today. Why use Keras rather than any other?

Here are some of the areas in which Keras compares favorably to existing alternatives.

---


## Keras has broad adoption in the industry and the research community


With over one million individual users as of late 2021, Keras has strong adoption across both the industry and the research community. Together with TensorFlow 2, Keras has more adoption than any other deep learning solution -- in every vertical.

You are already constantly interacting with features built with Keras -- it is in use at Netflix, Uber, Yelp, Instacart, Zocdoc, Square, and many others. It is especially popular among startups that place deep learning at the core of their products.

Keras & TensorFlow 2 are also a favorite among researchers, coming in #1 in terms of mentions in scientific papers indexed by Google Scholar. Keras has also been adopted by researchers at large scientific organizations, such as CERN and NASA.


![2021 deep learning frameworks adoption metrics](/img/deep_learning_frameworks_adoption_2021.jpg)


---

## Keras prioritizes developer experience
    
- Keras is an API designed for human beings, not machines. [Keras follows best practices for reducing cognitive load](https://blog.keras.io/user-experience-design-for-apis.html): it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.
- This makes Keras easy to learn and easy to use. As a Keras user, you are more productive, allowing you to try more ideas than your competition, faster -- which in turn [helps you win machine learning competitions](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).
- This ease of use does not come at the cost of reduced flexibility: because Keras integrates deeply with low-level TensorFlow functionality, it enables you to develop highly hackable workflows where any piece of functionality can be customized.

In early 2019, we ran a survey among teams that ended in the top 5 of any Kaggle competition in the two previous years (N=120). We asked them about:

1. The *primary* machine learning framework they used in the competition where they made it to the top 5.
2. All frameworks (primary and auxiliary) they used.

Keras ranked as #1 for deep learning both among primary frameworks and among all frameworks used.

![Primary ML frameworks used by top-5 teams on Kaggle](/img/graph-kaggle-1.jpeg)

![All ML frameworks used by top-5 teams on Kaggle](/img/graph-kaggle-2.jpeg)



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

For more information, see our [guide to multi-GPU & distributed training](/guides/distributed_training/).


---

## Keras is at the nexus of a large ecosystem

Like you, we know firsthand that building and training a model is only one slice of a machine learning workflow. Keras is built for the real world,
and in the real world, a successful model begins with data collection and ends with production deployment. 

Keras is at the center of a wide ecosystem of tightly-connected projects that together cover every step of the machine learning workflow, in particular:

- Rapid model prototyping with [AutoKeras](https://autokeras.com/)
- Scalable model training in on GCP via [TF Cloud](https://github.com/tensorflow/cloud)
- Hyperparameter tuning with [KerasTuner](https://keras.io/keras_tuner/)
- Extra layers, losses, metrics, callbacks... via [TensorFlow Addons](https://www.tensorflow.org/addons/api_docs/python/tfa)
- Inference model quantization & pruning with the [TF Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- Model deployment on mobile or on an embedded with [TF Lite](https://www.tensorflow.org/lite)
- Model deployment in the browser via [TF.js](https://www.tensorflow.org/js)
- ...and many more.

