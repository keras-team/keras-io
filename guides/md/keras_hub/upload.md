# Uploading Models with KerasHub

**Author:** [Samaneh Saadat](https://github.com/SamanehSaadat/), [Matthew Watson](https://github.com/mattdangerw/)<br>
**Date created:** 2024/04/29<br>
**Last modified:** 2024/04/29<br>
**Description:** An introduction on how to upload a fine-tuned KerasHub model to model hubs.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/upload.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/upload.py)



# Introduction

Fine-tuning a machine learning model can yield impressive results for specific tasks.
Uploading your fine-tuned model to a model hub allows you to share it with the broader community.
By sharing your models, you'll enhance accessibility for other researchers and developers,
making your contributions an integral part of the machine learning landscape.
This can also streamline the integration of your model into real-world applications.

This guide walks you through how to upload your fine-tuned models to popular model hubs such as
[Kaggle Models](https://www.kaggle.com/models) and [Hugging Face Hub](https://huggingface.co/models).

# Setup

Let's start by installing and importing all the libraries we need. We use KerasHub for this guide.


```python
!pip install -q --upgrade keras-hub huggingface-hub kagglehub
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_hub

```

# Data

We can use the IMDB reviews dataset for this guide. Let's load the dataset from `tensorflow_dataset`.


```python
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=4,
)
```

We only use a small subset of the training samples to make the guide run faster.
However, if you need a higher quality model, consider using a larger number of training samples.


```python
imdb_train = imdb_train.take(100)
```

# Task Upload

A `keras_hub.models.Task`, wraps a `keras_hub.models.Backbone` and a `keras_hub.models.Preprocessor` to create
a model that can be directly used for training, fine-tuning, and prediction for a given text problem.
In this section, we explain how to create a `Task`, fine-tune and upload it to a model hub.

---
## Load Model

If you want to build a Causal LM based on a base model, simply call `keras_hub.models.CausalLM.from_preset`
and pass a built-in preset identifier.


```python
causal_lm = keras_hub.models.CausalLM.from_preset("gpt2_base_en")

```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/task.json...

Downloading from https://www.kaggle.com/api/v1/models/keras/gpt2/keras/gpt2_base_en/2/download/preprocessor.json...

```
</div>
---
## Fine-tune Model

After loading the model, you can call `.fit()` on the model to fine-tune it.
Here, we fine-tune the model on the IMDB reviews which makes the model movie domain-specific.


```python
# Drop labels and keep the review text only for the Causal LM.
imdb_train_reviews = imdb_train.map(lambda x, y: x)

# Fine-tune the Causal LM.
causal_lm.fit(imdb_train_reviews)
```
 100/100 ━━━━━━━━━━━━━━━━━━━━ 151s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3271

---
## Save the Model Locally

To upload a model, you need to first save the model locally using `save_to_preset`.


```python
preset_dir = "./gpt2_imdb"
causal_lm.save_to_preset(preset_dir)
```

Let's see the saved files.


```python
os.listdir(preset_dir)
```




<div class="k-default-codeblock">
```
['preprocessor.json',
 'tokenizer.json',
 'task.json',
 'model.weights.h5',
 'config.json',
 'metadata.json',
 'assets']

```
</div>
### Load a Locally Saved Model

A model that is saved to a local preset can be loaded using `from_preset`.
What you save in, is what you get back out.


```python
causal_lm = keras_hub.models.CausalLM.from_preset(preset_dir)
```

You can also load the `keras_hub.models.Backbone` and `keras_hub.models.Tokenizer` objects from this preset directory.
Note that these objects are equivalent to `causal_lm.backbone` and `causal_lm.preprocessor.tokenizer` above.


```python
backbone = keras_hub.models.Backbone.from_preset(preset_dir)
tokenizer = keras_hub.models.Tokenizer.from_preset(preset_dir)
```

---
## Upload the Model to a Model Hub

After saving a preset to a directory, this directory can be uploaded to a model hub such as Kaggle or Hugging Face directly from the KerasHub library.
To upload the model to Kaggle, the URI must start with `kaggle://` and to upload to Hugging Face, it should start with `hf://`.

### Upload to Kaggle

To upload a model to Kaggle, first, we need to authenticate with Kaggle.
This can in one of the following ways:
1. Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. Provide a local `~/.kaggle/kaggle.json`.
3. Call `kagglehub.login()`.

Let's make sure we are logged in before continuing.


```python
import kagglehub

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

```

To upload a model we can use `keras_hub.upload_preset(uri, preset_dir)` API where `uri` has the format of
`kaggle://<KAGGLE_USERNAME>/<MODEL>/Keras/<VARIATION>` for uploading to Kaggle and `preset_dir` is the directory that the model is saved in.

Running the following uploads the model that is saved in `preset_dir` to Kaggle:


```python
kaggle_username = kagglehub.whoami()["username"]
kaggle_uri = f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
keras_hub.upload_preset(kaggle_uri, preset_dir)
```

<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (834B)
Upload successful: tokenizer.json (322B)
Upload successful: task.json (2KB)
Upload successful: model.weights.h5 (475MB)
Upload successful: config.json (431B)
Upload successful: metadata.json (142B)
Upload successful: merges.txt (446KB)
Upload successful: vocabulary.json (1018KB)

Your model instance version has been created.

```
</div>
### Upload to Hugging Face

To upload a model to Hugging Face, first, we need to authenticate with Hugging Face.
This can in one of the following ways:
1. Set environment variables `HF_USERNAME` and `HF_TOKEN`.
2. Call `huggingface_hub.notebook_login()`.

Let's make sure we are logged in before coninuing.


```python
import huggingface_hub

if "HF_USERNAME" not in os.environ or "HF_TOKEN" not in os.environ:
    huggingface_hub.notebook_login()
```

`keras_hub.upload_preset(uri, preset_dir)` can be used to upload a model to Hugging Face if `uri` has the format of
`kaggle://<HF_USERNAME>/<MODEL>`.

Running the following uploads the model that is saved in `preset_dir` to Hugging Face:


```python
hf_username = huggingface_hub.whoami()["name"]
hf_uri = f"hf://{hf_username}/gpt2_imdb"
keras_hub.upload_preset(hf_uri, preset_dir)

```

---
## Load a User Uploaded Model

After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.

```python
causal_lm = keras_hub.models.CausalLM.from_preset(
    f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
)
```

We can also load the model uploaded to Hugging Face by calling `from_preset`.

```python
causal_lm = keras_hub.models.CausalLM.from_preset(f"hf://{hf_username}/gpt2_imdb")
```

# Classifier Upload

Uploading a classifier model is similar to Causal LM upload.
To upload the fine-tuned model, first, the model should be saved to a local directory using `save_to_preset`
API and then it can be uploaded via `keras_hub.upload_preset`.


```python
# Load the base model.
classifier = keras_hub.models.Classifier.from_preset(
    "bert_tiny_en_uncased", num_classes=2
)

# Fine-tune the classifier.
classifier.fit(imdb_train)

# Save the model to a local preset directory.
preset_dir = "./bert_tiny_imdb"
classifier.save_to_preset(preset_dir)

# Upload to Kaggle.
keras_hub.upload_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb", preset_dir
)
```
 100/100 ━━━━━━━━━━━━━━━━━━━━ 7s 31ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5164


<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (947B)
Upload successful: tokenizer.json (461B)
Upload successful: task.json (2KB)
Upload successful: task.weights.h5 (50MB)
Upload successful: model.weights.h5 (17MB)
Upload successful: config.json (454B)
Upload successful: metadata.json (140B)
Upload successful: vocabulary.txt (226KB)

Your model instance version has been created.
```
</div>
After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.

```python
classifier = keras_hub.models.Classifier.from_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb"
)
```