# Training Keras models with TensorFlow Cloud

**Authors:** [Jonah Kohn](https://jonahkohn.com), [Sina Chavoshi](https://www.linkedin.com/in/sinachavoshi/)<br>
**Date created:** 2020/08/11<br>
**Last modified:** 2021/07/23<br>
**Description:** Usage guide for TensorFlow Cloud.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/training_keras_models_on_cloud.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/training_keras_models_on_cloud.py)



---
## Introduction

TensorFlow Cloud is a library that makes it easier to do training and
hyperparameter tuning of Keras models on Google Cloud.

Using TensorFlow Cloud's `run` API, you can send your model code directly to
your Google Cloud account, and use Google Cloud compute resources without
needing to login and interact with the Cloud UI (once you have set up your
project in the console).

This means that you can use your Google Cloud compute resources from inside
directly a Python notebook: a notebook just like this one! You can also send
models to Google Cloud from a plain `.py` Python script.

---
## Simple example

This is a simple introductory example to demonstrate how to train a model
remotely using [TensorFlow Cloud](https://tensorflow.org/cloud) and Google
Cloud.

You can just read through it to get an idea of how this works, or you can run
the notebook in Google Colab. Running the notebook requires connecting to a
Google Cloud account and entering your credentials and project ID. See
[Setting Up and Connecting To Your Google Cloud Account](https://github.com/tensorflow/cloud/blob/master/g3doc/tutorials/google_cloud_project_setup_instructions.ipynb)
if you don't have an account yet or are not sure how to set up a project in the
console.

---
## Import required modules

This guide requires TensorFlow Cloud, which you can install via:

`pip install tensorflow-cloud`


```python
import os
import sys
import tensorflow as tf
import tensorflow_cloud as tfc
```

---
## Project Configurations

Set project parameters. If you don't know what your `GCP_PROJECT_ID` or
`GCS_BUCKET` should be, see
[Setting Up and Connecting To Your Google Cloud Account](https://github.com/tensorflow/cloud/blob/master/g3doc/tutorials/google_cloud_project_setup_instructions.ipynb).

The `JOB_NAME` is optional, and you can set it to any string. If you are doing
multiple training experiemnts (for example) as part of a larger project, you may
want to give each of them a unique `JOB_NAME`.


```python
# Set Google Cloud Specific parameters

# TODO: Please set GCP_PROJECT_ID to your own Google Cloud project ID.
GCP_PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}

# TODO: set GCS_BUCKET to your own Google Cloud Storage (GCS) bucket.
GCS_BUCKET = "YOUR_GCS_BUCKET_NAME"  # @param {type:"string"}

# DO NOT CHANGE: Currently only the 'us-central1' region is supported.
REGION = "us-central1"

# OPTIONAL: You can change the job name to any string.
JOB_NAME = "mnist"  # @param {type:"string"}

# Setting location were training logs and checkpoints will be stored
GCS_BASE_PATH = f"gs://{GCS_BUCKET}/{JOB_NAME}"
TENSORBOARD_LOGS_DIR = os.path.join(GCS_BASE_PATH, "logs")
MODEL_CHECKPOINT_DIR = os.path.join(GCS_BASE_PATH, "checkpoints")
SAVED_MODEL_DIR = os.path.join(GCS_BASE_PATH, "saved_model")
```

---
## Authenticating the notebook to use your Google Cloud Project

This code authenticates the notebook, checking your valid Google Cloud
credentials and identity. It is inside the `if not tfc.remote()` block to ensure
that it is only run in the notebook, and will not be run when the notebook code
is sent to Google Cloud.

Note: For Kaggle Notebooks click on "Add-ons"->"Google Cloud SDK" before running
the cell below.


```python
# Using tfc.remote() to ensure this code only runs in notebook
if not tfc.remote():

    # Authentication for Kaggle Notebooks
    if "kaggle_secrets" in sys.modules:
        from kaggle_secrets import UserSecretsClient

        UserSecretsClient().set_gcloud_credentials(project=GCP_PROJECT_ID)

    # Authentication for Colab Notebooks
    if "google.colab" in sys.modules:
        from google.colab import auth

        auth.authenticate_user()
        os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT_ID
```

---
## Model and data setup

From here we are following the basic procedure for setting up a simple Keras
model to run classification on the MNIST dataset.

### Load and split data

Read raw data and split to train and test data sets.


```python
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype("float32") / 255
```

### Create a model and prepare for training

Create a simple model and set up a few callbacks for it.


```python
from tensorflow.keras import layers

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
```

<div class="k-default-codeblock">
```

```
</div>
### Quick validation training

We'll train the model for one (1) epoch just to make sure everything is set up
correctly, and we'll wrap that training command in `if not` `tfc.remote`, so
that it only happens here in the runtime environment in which you are reading
this, not when it is sent  to Google Cloud.


```python
if not tfc.remote():
    # Run the training for 1 epoch and a small subset of the data to validate setup
    model.fit(x=x_train[:100], y=y_train[:100], validation_split=0.2, epochs=1)
```

<div class="k-default-codeblock">
```

3/3 [==============================] - 1s 78ms/step - loss: 2.3081 - accuracy: 0.1375 - val_loss: 1.7350 - val_accuracy: 0.5000

```
</div>
---
## Prepare for remote training

The code below will only run when the notebook code is sent to Google Cloud, not
inside the runtime in which you are reading this.

First, we set up callbacks which will:

* Create logs for [TensorBoard](https://www.tensorflow.org/tensorboard).
* Create [checkpoints](https://keras.io/api/callbacks/model_checkpoint/) and save them to the checkpoints
directory specified above.
* Stop model training if loss is not improving sufficiently.

Then we call `model.fit` and `model.save`, which (when this code is running on
Google Cloud) which actually run the full training (100 epochs) and then save
the trained model in the GCS Bucket and directory defined above.


```python
if tfc.remote():
    # Configure Tensorboard logs
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_DIR),
        tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_DIR, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.001, patience=3),
    ]

    model.fit(
        x=x_train, y=y_train, epochs=100, validation_split=0.2, callbacks=callbacks,
    )

    model.save(SAVED_MODEL_DIR)
```

---
## Start the remote training

TensorFlow Cloud takes all the code from its local execution environment (this
notebook), wraps it up, and sends it to Google Cloud for execution. (That's why
the `if` and `if not` `tfc.remote` wrappers are important.)

This step will prepare your code from this notebook for remote execution and
then start a remote training job on Google Cloud Platform to train the model.

First we add the `tensorflow-cloud` Python package to a `requirements.txt` file,
which will be sent along with the code in this notebook. You can add other
packages here as needed.

Then a GPU and a CPU image are specified. You only need to specify one or the
other; the GPU is used in the code that follows.

Finally, the heart of TensorFlow cloud: the call to `tfc.run`. When this is
executed inside this notebook, all the code from this notebook, and the rest of
the files in this directory, will be packaged and sent to Google Cloud for
execution. The parameters on the `run` method specify the details of the  GPU
CPU images are specified. You only need to specify one or the other; the GPU is
used in the code that follows.

Finally, the heart of TensorFlow cloud: the call to `tfc.run`. When this is
executed inside this notebook, all the code from this notebook, and the rest of
the files in this directory, will be packaged and sent to Google Cloud for
execution. The parameters on the `run` method specify the details of the  GPU
and CPU images are specified. You only need to specify one or the other; the GPU
is used in the code that follows.

Finally, the heart of TensorFlow cloud: the call to `tfc.run`. When this is
executed inside this notebook, all the code from this notebook, and the rest of
the files in this directory, will be packaged and sent to Google Cloud for
execution. The parameters on the `run` method specify the details of the
execution environment and the distribution strategy (if any) to be used.

Once the job is submitted you can go to the next step to monitor the jobs
progress via Tensorboard.


```python
# If you are using a custom image you can install modules via requirements
# txt file.
with open("requirements.txt", "w") as f:
    f.write("tensorflow-cloud\n")

# Optional: Some recommended base images. If you provide none the system
# will choose one for you.
TF_GPU_IMAGE = "gcr.io/deeplearning-platform-release/tf2-cpu.2-5"
TF_CPU_IMAGE = "gcr.io/deeplearning-platform-release/tf2-gpu.2-5"

# Submit a single node training job using GPU.
tfc.run(
    distribution_strategy="auto",
    requirements_txt="requirements.txt",
    docker_config=tfc.DockerConfig(
        parent_image=TF_GPU_IMAGE, image_build_bucket=GCS_BUCKET
    ),
    chief_config=tfc.COMMON_MACHINE_CONFIGS["K80_1X"],
    job_labels={"job": JOB_NAME},
)
```

---
## Training Results

### Reconnect your Colab instance

Most remote training jobs are long running. If you are using Colab, it may time
out before the training results are available.

In that case, **rerun the following sections in order** to reconnect and
configure your Colab instance to access the training results.

1.   Import required modules
2.   Project Configurations
3.   Authenticating the notebook to use your Google Cloud Project

**DO NOT** rerun the rest of the code.

### Load Tensorboard

While the training is in progress you can use Tensorboard to view the results.
Note the results will show only after your training has started. This may take a
few minutes.


```
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOGS_DIR
```

---
## Load your trained model

Once training is complete, you can retrieve your model from the GCS Bucket you
specified above.


```python
trained_model = tf.keras.models.load_model(SAVED_MODEL_DIR)
trained_model.summary()
```
