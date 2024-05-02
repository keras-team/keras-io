"""
Title: Uploading Models with KerasNLP
Author: [Samaneh Saadat](https://github.com/SamanehSaadat/), [Matthew Watson](https://github.com/mattdangerw/)
Date created: 2024/04/29
Last modified: 2024/04/29
Description: An introduction on how to upload a fine-tuned KerasNLP model to model hubs.
Accelerator: GPU
"""

"""
# Introduction

Fine-tuning a machine learning model can yield impressive results for specific tasks.
Uploading your fine-tuned model to a model hub allow you to share it with the broader community.
By sharing your models, you'll enhance accessibility for other researchers and developers,
making your contributions an integral part of the machine learning landscape.
This can also streamline the integration of your model into real-world applications.

This guide walks you through how to upload your fine-tuned models to popular model hubs such as
[Kaggle Models](https://www.kaggle.com/models) and [Hugging Face Hub](https://huggingface.co/models).
"""

"""
# Setup

Let's start by installing and importing all the libraries we need. We use KerasNLP for this guide.
"""

"""shell
pip install -q --upgrade keras-nlp huggingface-hub
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp


"""
# Data

We can use the IMDB reviews dataset for this guide. Let's load the dataset from `tensorflow_dataset`.
"""

import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=4,
)

"""
We only use a small subset of the training samples to make the guide run faster.
However, if you need a higher quality model, consider using a larger number of training samples.
"""

imdb_train = imdb_train.take(100)

"""
# Task Upload

A `keras_nlp.models.Task`, wraps a `keras_nlp.models.Backbone` and a `keras_nlp.models.Preprocessor` to create
a model that can be directly used for training, fine-tuning, and prediction for a given text problem.
In this section, we explain how to create a `Task`, fine-tune and upload it to a model hub.
"""

"""
## Load Model

If you want to build a Causal LM based on a base model, simply call `keras_nlp.models.CausalLM.from_preset`
and pass a built-in preset identifier.
"""

causal_lm = keras_nlp.models.CausalLM.from_preset("gpt2_base_en")


"""
## Fine-tune Model

After loading the model, you can call `.fit()` on the model to fine-tune it.
Here, we fine-tune the model on the IMDB reviews which makes the model movie domain-specific.
"""

# Drop labels and keep the review text only for the Causal LM.
imdb_train_reviews = imdb_train.map(lambda x, y: x)

# Fine-tune the Causal LM.
causal_lm.fit(imdb_train_reviews)

"""
## Save the Model Locally

To upload a model, you need to first save the model locally using `save_to_preset`.
"""

preset_dir = "./gpt2_imdb"
causal_lm.save_to_preset(preset_dir)

"""
Let's see what are the files what are the saved files.
"""

os.listdir(preset_dir)

"""
### Load a Locally Saved Model

A model that is saved to a local preset, can be loaded using `from_preset`.
What you save in, is what you get back out.
"""

causal_lm = keras_nlp.models.CausalLM.from_preset(preset_dir)

"""
You can also load the `keras_nlp.models.Backbone` and `keras_nlp.models.Tokenizer` objects from this preset directory.
Note that these objects are equivalent to `causal_lm.backbone` and `causal_lm.preprocessor.tokenizer` above.
"""

backbone = keras_nlp.models.Backbone.from_preset(preset_dir)
tokenizer = keras_nlp.models.Tokenizer.from_preset(preset_dir)

"""
## Upload the Model to a Model Hub

After saving a preset to a directory, this directory can be uploaded to a model hub such as Kaggle or Hugging Face directly from the KerasNLP library.
To upload the model to Kaggle, the URI should start with `kaggle://` and to upload to Hugging Face, it should start with `hf://`.
"""
"""
### Upload to Kaggle
"""

"""
To upload a model to Kaggle, first, we need to authenticate with Kaggle.
This can by one of the followings:
1. Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. Provide a local `~/.kaggle/kaggle.json`.
3. Call `kagglehub.login()`.

Let's make sure we are logged in before coninuing.
"""

import kagglehub

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()


"""

To upload a model we can use `keras_nlp.upload_preset(uri, preset_dir)` API where `uri` has the format of
`kaggle://<KAGGLE_USERNAME>/<MODEL>/Keras/<VARIATION>` for uploading to Kaggle and `preset_dir` is the directory that the model is saved in.

Running the following uploads the model that is saved in `preset_dir` to Kaggle:
"""
kaggle_username = os.getenv("KAGGLE_USERNAME")  # TODO: Assign username.
kaggle_uri = f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
keras_nlp.upload_preset(kaggle_uri, preset_dir)

"""
### Upload to Hugging Face
"""

"""
To upload a model to Hugging Face, first, we need to authenticate with Hugging Face.
This can by one of the followings:
1. Set environment variables `HF_USERNAME` and `HF_TOKEN`.
2. Call `huggingface_hub.notebook_login()`.

Let's make sure we are logged in before coninuing.
"""

import huggingface_hub

if "HF_USERNAME" not in os.environ or "HF_TOKEN" not in os.environ:
    huggingface_hub.notebook_login()

"""

`keras_nlp.upload_preset(uri, preset_dir)` can be used to upload a model to Hugging Face if `uri` has the format of
`kaggle://<HF_USERNAME>/<MODEL>`.

Running the following uploads the model that is saved in `preset_dir` to Hugging Face:
"""

hf_username = huggingface_hub.whoami()["name"]
hf_uri = f"hf://{hf_username}/gpt2_imdb"
keras_nlp.upload_preset(hf_uri, preset_dir)


"""
## Load a User Uploaded Model

After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.
"""

causal_lm = keras_nlp.models.CausalLM.from_preset(
    f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
)

# Load a user uploaded CausalLM from Hugging Face.
causal_lm = keras_nlp.models.CausalLM.from_preset(f"hf://{hf_username}/gpt2_imdb")

"""
# Classifier Upload

Uploading a classifier model is similar to Causal LM upload.
To upload the fine-tuned model, first, the model should be saved to a local directory using `save_to_preset`
API and then it can be uploaded via `keras_nlp.upload_preset`.
"""

# Load the base model.
classifier = keras_nlp.models.Classifier.from_preset(
    "bert_tiny_en_uncased", num_classes=2
)

# Fine-tune the classifier.
classifier.fit(imdb_train)

# Save the model to a local preset directory.
preset_dir = "./bert_tiny_imdb"
classifier.save_to_preset(preset_dir)

# Upload to Kaggle.
keras_nlp.upload_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb", preset_dir
)

# Upload to Hugging Face.
keras_nlp.upload_preset(f"hf://{hf_username}/bert_tiny_imdb", preset_dir)

"""
After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.
"""

classifier = keras_nlp.models.Classifier.from_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb"
)
