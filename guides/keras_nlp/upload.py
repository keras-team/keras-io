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
pip install -q --upgrade keras-nlp
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp

"""
# Task Upload

A KerasNLP `keras_nlp.models.Task`, wraps a `keras_nlp.models.Backbone` and a `keras_nlp.models.Preprocessor` to create
a model that can be directly used for training, fine-tuning, and prediction for a given text problem.
In this section, we explain how to create a `Task`, fine-tune and upload it to a model hub.
"""

"""
## Load Model

If you want to build a Causal LM based on a base model, simply call `keras_nlp.models.Classifier.from_preset`
and pass a built-in preset identifier.
"""

causal_lm = keras_nlp.models.CausalLM.from_preset("gpt2_base_en")


"""
## Fine-tune Model
After loading the model, you can call `.fit()` on the model to fine-tune it.
"""

# A toy dataset.
data = ["The quick brown fox jumped.", "I forgot my homework."]

# Fine-tune the `causal_lm` model.
causal_lm.fit(data, batch_size=2)

"""
## Save the Model Locally

To upload a model, you need to first save the model locally using `save_to_preset`.
"""

preset_dir = "./finetuned_gpt2"
causal_lm.save_to_preset(preset_dir)

"""
Let's see what are the files what are the saved files.
"""

os.listdir(preset_dir)

"""
## Upload the Model to a Model Hub

When the model is saved to a local directory `preset_dir`, this directory can be seamlessly
uploaded to a model hub such as Kaggle or Hugging Face programmatically.
"""
"""
### Upload to Kaggle Models
"""

"""
To upload a model to Kaggle Models, first, we need to authenticate with Kaggle.
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
Run this command to upload the model that is save in `preset_dir` to Kaggle Models:
"""

kaggle_username = "" # TODO: assing username.
kaggle_uri = f"kaggle://{kaggle_username}/gpt2/keras/finetuned_gpt2"
keras_nlp.upload_preset(kaggle_uri, preset_dir)

"""
### Upload to Hugging Face Hub
"""

"""
To upload a model to Hugging Face Hub, first, we need to authenticate with Hugging Face.
This can by one of the followings:
1. Set environment variables `HF_USERNAME` and `HF_TOKEN`.
2. Call `huggingface_hub.notebook_login()`.

Let's make sure we are logged in before coninuing.
"""

import huggingface_hub

if "HF_USERNAME" not in os.environ or "HF_TOKEN" not in os.environ:
    huggingface_hub.notebook_login()

"""
Run this command to upload the model that is save in `preset_dir` to Hugging Face Hub:
"""

hf_username = huggingface_hub.whoami()
hf_uri = f"hf://{hf_username}/finetuned_gpt2"
keras_nlp.upload_preset(hf_uri, preset_dir)

"""
# Classifier Upload

Uploading a classifier model is similar to Causal LM upload. You can follow
[this guide](https://keras.io/guides/keras_nlp/getting_started/#fine-tuning-a-pretrained-bert-backbone)
to fine-tune a classfier.


To upload the fine-tuned model, first, the model is saved to a local directory using `save_to_preset`
API and then it can be uploaded via `keras_nlp.upload_preset`.
"""

# A toy dataset.
data = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 2]

# Load the base model.
classifier = keras_nlp.models.Classifier.from_preset(
    "bert_tiny_en_uncased", num_classes=4
)

# Fine-tune the classifier.
classifier.fit(x=data, y=labels, batch_size=2)

# Save the model to a local preset directory.
preset_dir = "./finetuned_bert"
classifier.save_to_preset(preset_dir)

# Upload to Kaggle Models.
keras_nlp.upload_preset(
    f"kaggle://{kaggle_username}/bert/keras/finetuned_bert", preset_dir
)

# Upload to Hugging Face.
keras_nlp.upload_preset(f"hf://{hf_username}/finetuned_bert", preset_dir)

"""
# Base Upload

If you have fine-tuned a base class that is not related to any specific task or if you need
more flexibility and want to upload the base model only, we recommend saving `Backbone` and
`Tokenizer` into a preset directory and uploading the custom direcotry.

```python
backbone.save_to_preset(preset_dir)
tokenizer.save_to_preset(preset_dir)
keras_nlp.upload_preset(uri, preset_dir)
```
"""
