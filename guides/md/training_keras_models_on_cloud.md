# Training Keras models with TensorFlow Cloud

**Author:** [Jonah Kohn](https://jonahkohn.com)<br>
**Date created:** 2020/08/11<br>
**Last modified:** 2020/08/11<br>
**Description:** In-depth usage guide for TensorFlow Cloud.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/training_keras_models_on_cloud.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/training_keras_models_on_cloud.py)



---
## Introduction

[TensorFlow Cloud](https://github.com/tensorflow/cloud) is a Python package that
provides APIs for a seamless transition from local debugging to distributed training
in Google Cloud. It simplifies the process of training TensorFlow models on the
cloud into a single, simple function call, requiring minimal setup and no changes
to your model. TensorFlow Cloud handles cloud-specific tasks such as creating VM
instances and distribution strategies for your models automatically. This guide
will demonstrate how to interface with Google Cloud through TensorFlow Cloud,
and the wide range of functionality provided within TensorFlow Cloud. We'll start
with the simplest use-case.

---
## Setup

We'll get started by installing TensorFlow Cloud, and importing the packages we
will need in this guide.


```python
!pip install -q tensorflow_cloud
```


```python
import tensorflow as tf
import tensorflow_cloud as tfc

from tensorflow import keras
from tensorflow.keras import layers
```

---
## API overview: a first end-to-end example

Let's begin with a Keras model training script, such as the following CNN:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        # Use a Rescaling layer to make sure input values are in the [0, 1] range.
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        # The original images have shape (28, 28), so we reshape them to (28, 28, 1)
        layers.Reshape(target_shape=(28, 28, 1)),
        # Follow-up with a classic small convnet
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
)

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)
```

To train this model on Google Cloud we just need to add a call to `run()` at
the beginning of the script, before the imports:
```python
tfc.run()
```

You don’t need to worry about cloud-specific tasks such as creating VM instances
and distribution strategies when using TensorFlow Cloud.
The API includes intelligent defaults for all the parameters -- everything is
configurable, but many models can rely on these defaults.

Upon calling `run()`, TensorFlow Cloud will:

- Make your Python script or notebook distribution-ready.
- Convert it into a Docker image with required dependencies.
- Run the training job on a GCP GPU-powered VM.
- Stream relevant logs and job information.

The default VM configuration is 1 chief and 0 workers with 8 CPU cores and
1 Tesla T4 GPU.

---
## Google Cloud configuration

In order to facilitate the proper pathways for Cloud training, you will need to
do some first-time setup. If you're a new Google Cloud user, there are a few
preliminary steps you will need to take:

1. Create a GCP Project;
2. Enable AI Platform Services;
3. Create a Service Account;
4. Download an authorization key;
5. Create a Cloud Storage bucket.

Detailed first-time setup instructions can be found in the
[TensorFlow Cloud README](https://github.com/tensorflow/cloud#setup-instructions),
and an additional setup example is shown on the
[TensorFlow Blog](https://blog.tensorflow.org/2020/08/train-your-tensorflow-model-on-google.html).

---
## Common workflows and Cloud storage

In most cases, you'll want to retrieve your model after training on Google Cloud.
For this, it's crucial to redirect saving and loading to Cloud Storage while
training remotely. We can direct TensorFlow Cloud to our Cloud Storage bucket for
a variety of tasks. The storage bucket can be used to save and load large training
datasets, store callback logs or model weights, and save trained model files.
To begin, let's configure `fit()` to save the model to a Cloud Storage, and set
up TensorBoard monitoring to track training progress.


```python

def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
            layers.Reshape(target_shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.SparseCategoricalAccuracy(),
    )
    return model

```

Let's save the TensorBoard logs and model checkpoints generated during training
in our cloud storage bucket.


```python
import datetime
import os

# Note: Please change the gcp_bucket to your bucket name.
gcp_bucket = "keras-examples"

checkpoint_path = os.path.join("gs://", gcp_bucket, "mnist_example", "save_at_{epoch}")

tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
    "gs://", gcp_bucket, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

callbacks = [
    # TensorBoard will store logs for each epoch and graph performance for us.
    keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    # ModelCheckpoint will save models after each epoch for retrieval later.
    keras.callbacks.ModelCheckpoint(checkpoint_path),
    # EarlyStopping will terminate training when val_loss ceases to improve.
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

model = create_model()
```

Here, we will load our data from Keras directly. In general, it's best practice
to store your dataset in your Cloud Storage bucket, however TensorFlow Cloud can
also accomodate datasets stored locally. That's covered in the Multi-file section
of this guide.


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

The [TensorFlow Cloud](https://github.com/tensorflow/cloud) API provides the
`remote()` function to determine whether code is being executed locally or on
the cloud. This allows for the separate designation of `fit()` parameters for
local and remote execution, and provides means for easy debugging without overloading
your local machine.


```python
if tfc.remote():
    epochs = 100
    callbacks = callbacks
    batch_size = 128
else:
    epochs = 5
    batch_size = 64
    callbacks = None

model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, batch_size=batch_size)
```

<div class="k-default-codeblock">
```
Epoch 1/5
938/938 [==============================] - 6s 7ms/step - loss: 0.2021 - sparse_categorical_accuracy: 0.9383
Epoch 2/5
938/938 [==============================] - 6s 7ms/step - loss: 0.0533 - sparse_categorical_accuracy: 0.9836
Epoch 3/5
938/938 [==============================] - 6s 7ms/step - loss: 0.0385 - sparse_categorical_accuracy: 0.9883
Epoch 4/5
938/938 [==============================] - 6s 7ms/step - loss: 0.0330 - sparse_categorical_accuracy: 0.9895
Epoch 5/5
938/938 [==============================] - 6s 7ms/step - loss: 0.0255 - sparse_categorical_accuracy: 0.9916

<tensorflow.python.keras.callbacks.History at 0x7f9fb82bbf40>

```
</div>
Let's save the model in GCS after the training is complete.


```python
save_path = os.path.join("gs://", gcp_bucket, "mnist_example")

if tfc.remote():
    model.save(save_path)
```

We can also use this storage bucket for Docker image building, instead of your local
Docker instance. For this, just add your bucket to the `docker_image_bucket_name` parameter.


```python
# docs_infra: no_execute
tfc.run(docker_image_bucket_name=gcp_bucket)
```

After training the model, we can load the saved model and view our TensorBoard logs
to monitor performance.


```python
# docs_infra: no_execute
model = keras.models.load_model(save_path)
```


```python
!#docs_infra: no_execute
!tensorboard dev upload --logdir "gs://keras-examples-jonah/logs/fit" --name "Guide MNIST"
```

---
## Large-scale projects

In many cases, your project containing a Keras model may encompass more than one
Python script, or may involve external data or specific dependencies. TensorFlow
Cloud is entirely flexible for large-scale deployment, and provides a number of
intelligent functionalities to aid your projects.

### Entry points: support for Python scripts and Jupyter notebooks

Your call to the `run()` API won't always be contained inside the same Python script
as your model training code. For this purpose, we provide an `entry_point` parameter.
The `entry_point` parameter can be used to specify the Python script or notebook in
which your model training code lives. When calling `run()` from the same script as
your model, use the `entry_point` default of `None`.

### `pip` dependencies

If your project calls on additional `pip` dependencies, it's possible to specify
the additional required libraries by including a `requirements.txt` file. In this
file, simply put a list of all the required dependencies and TensorFlow Cloud will
handle integrating these into your cloud build.

### Python notebooks

TensorFlow Cloud is also runnable from Python notebooks. Additionally, your specified
`entry_point` can be a notebook if needed. There are two key differences to keep
in mind between TensorFlow Cloud on notebooks compared to scripts:

- When calling `run()` from within a notebook, a Cloud Storage bucket must be specified
for building and storing your Docker image.
- GCloud authentication happens entirely through your authentication key, without
project specification. An example workflow using TensorFlow Cloud from a notebook
is provided in the "Putting it all together" section of this guide.

### Multi-file projects

If your model depends on additional files, you only need to ensure that these files
live in the same directory (or subdirectory) of the specified entry point. Every file
that is stored in the same directory as the specified `entry_point` will be included
in the Docker image, as well as any files stored in subdirectories adjacent to the
`entry_point`. This is also true for dependencies you may need which can't be acquired
through `pip`

For an example of a custom entry-point and multi-file project with additional pip
dependencies, take a look at this multi-file example on the
[TensorFlow Cloud Repository](https://github.com/tensorflow/cloud/tree/master/src/python/tensorflow_cloud/core/tests/examples/multi_file_example).
For brevity, we'll just include the example's `run()` call:

```python
tfc.run(
    docker_image_bucket_name=gcp_bucket,
    entry_point="train_model.py",
    requirements="requirements.txt"
)
```

---
## Machine configuration and distributed training

Model training may require a wide range of different resources, depending on the
size of the model or the dataset. When accounting for configurations with multiple
GPUs, it becomes critical to choose a fitting
[distribution strategy](https://www.tensorflow.org/guide/distributed_training).
Here, we outline a few possible configurations:

### Multi-worker distribution
Here, we can use `COMMON_MACHINE_CONFIGS` to designate 1 chief CPU and 4 worker GPUs.

```python
tfc.run(
    docker_image_bucket_name=gcp_bucket,
    chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'],
    worker_count=2,
    worker_config=tfc.COMMON_MACHINE_CONFIGS['T4_4X']
)
```
By default, TensorFlow Cloud chooses the best distribution strategy for your machine
configuration with a simple formula using the `chief_config`, `worker_config` and
`worker_count` parameters provided.

- If the number of GPUs specified is greater than zero, `tf.distribute.MirroredStrategy` will be chosen.
- If the number of workers is greater than zero, `tf.distribute.experimental.MultiWorkerMirroredStrategy` or `tf.distribute.experimental.TPUStrategy` will be chosen based on the accelerator type.
- Otherwise, `tf.distribute.OneDeviceStrategy` will be chosen.

### TPU distribution
Let's train the same model on TPU, as shown:
```python
tfc.run(
    docker_image_bucket_name=gcp_bucket,
    chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
    worker_count=1,
    worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"]
)
```

### Custom distribution strategy
To specify a custom distribution strategy, format your code normally as you would
according to the
[distributed training guide](https://www.tensorflow.org/guide/distributed_training)
and set `distribution_strategy` to `None`. Below, we'll specify our own distribution
strategy for the same MNIST model.
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = create_model()

if tfc.remote():
    epochs = 100
    batch_size = 128
else:
    epochs = 10
    batch_size = 64
    callbacks = None

model.fit(
    x_train, y_train, epochs=epochs, callbacks=callbacks, batch_size=batch_size
)

tfc.run(
    docker_image_bucket_name=gcp_bucket,
    chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'],
    worker_count=2,
    worker_config=tfc.COMMON_MACHINE_CONFIGS['T4_4X'],
    distribution_strategy=None
)
```

---
## Custom Docker images

By default, TensorFlow Cloud uses a
[Docker base image](https://hub.docker.com/r/tensorflow/tensorflow/)
supplied by Google and corresponding to your current TensorFlow version. However,
you can also specify a custom Docker image to fit your build requirements, if necessary.
For this example, we will specify the Docker image from an older version of TensorFlow:
```python
tfc.run(
    docker_image_bucket_name=gcp_bucket,
    base_docker_image="tensorflow/tensorflow:2.1.0-gpu"
)
```

---
## Additional metrics

You may find it useful to tag your Cloud jobs with specific labels, or to stream
your model's logs during Cloud training.
It's good practice to maintain proper labeling on all Cloud jobs, for record-keeping.
For this purpose, `run()` accepts a dictionary of labels up to 64 key-value pairs,
which are visible from the Cloud build logs. Logs such as epoch performance and model
saving internals can be accessed using the link provided by executing `tfc.run` or
printed to your local terminal using the `stream_logs` flag.
```python
job_labels = {"job": "mnist-example", "team": "keras-io", "user": "jonah"}

tfc.run(
    docker_image_bucket_name=gcp_bucket,
    job_labels=job_labels,
    stream_logs=True
)
```

---
## Putting it all together

For an in-depth Colab which uses many of the features described in this guide,
follow along
[this example](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/core/tests/examples/dogs_classification.ipynb)
to train a state-of-the-art model to recognize dog breeds from photos using feature
extraction.
