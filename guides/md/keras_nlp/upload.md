# Uploading Models with KerasNLP

**Author:** [Samaneh Saadat](https://github.com/SamanehSaadat/), [Matthew Watson](https://github.com/mattdangerw/)<br>
**Date created:** 2024/04/29<br>
**Last modified:** 2024/04/29<br>
**Description:** An introduction on how to upload a fine-tuned KerasNLP model to model hubs.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/upload.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/upload.py)



# Introduction

Fine-tuning a machine learning model can yield impressive results for specific tasks.
Uploading your fine-tuned model to a model hub allow you to share it with the broader community.
By sharing your models, you'll enhance accessibility for other researchers and developers,
making your contributions an integral part of the machine learning landscape.
This can also streamline the integration of your model into real-world applications.

This guide walks you through how to upload your fine-tuned models to popular model hubs such as
[Kaggle Models](https://www.kaggle.com/models) and [Hugging Face Hub](https://huggingface.co/models).

# Setup

Let's start by installing and importing all the libraries we need. We use KerasNLP for this guide.


```python
!pip install -q --upgrade keras-nlp huggingface-hub
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp

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

A `keras_nlp.models.Task`, wraps a `keras_nlp.models.Backbone` and a `keras_nlp.models.Preprocessor` to create
a model that can be directly used for training, fine-tuning, and prediction for a given text problem.
In this section, we explain how to create a `Task`, fine-tune and upload it to a model hub.

---
## Load Model

If you want to build a Causal LM based on a base model, simply call `keras_nlp.models.CausalLM.from_preset`
and pass a built-in preset identifier.


```python
causal_lm = keras_nlp.models.CausalLM.from_preset("gpt2_base_en")

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

    
   1/100 [37m━━━━━━━━━━━━━━━━━━━━  49:11 30s/step - loss: 0.5584 - sparse_categorical_accuracy: 0.2781

<div class="k-default-codeblock">
```

```
</div>
   2/100 [37m━━━━━━━━━━━━━━━━━━━━  14:47 9s/step - loss: 0.7601 - sparse_categorical_accuracy: 0.3022 

<div class="k-default-codeblock">
```

```
</div>
   3/100 [37m━━━━━━━━━━━━━━━━━━━━  8:12 5s/step - loss: 0.8292 - sparse_categorical_accuracy: 0.3068 

<div class="k-default-codeblock">
```

```
</div>
   4/100 [37m━━━━━━━━━━━━━━━━━━━━  6:00 4s/step - loss: 0.8888 - sparse_categorical_accuracy: 0.3105

<div class="k-default-codeblock">
```

```
</div>
   5/100 ━[37m━━━━━━━━━━━━━━━━━━━  4:54 3s/step - loss: 0.9495 - sparse_categorical_accuracy: 0.3137

<div class="k-default-codeblock">
```

```
</div>
   6/100 ━[37m━━━━━━━━━━━━━━━━━━━  4:14 3s/step - loss: 0.9879 - sparse_categorical_accuracy: 0.3159

<div class="k-default-codeblock">
```

```
</div>
   7/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:46 2s/step - loss: 1.0065 - sparse_categorical_accuracy: 0.3170

<div class="k-default-codeblock">
```

```
</div>
   8/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:27 2s/step - loss: 1.0233 - sparse_categorical_accuracy: 0.3183

<div class="k-default-codeblock">
```

```
</div>
   9/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:11 2s/step - loss: 1.0313 - sparse_categorical_accuracy: 0.3193

<div class="k-default-codeblock">
```

```
</div>
  10/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:59 2s/step - loss: 1.0330 - sparse_categorical_accuracy: 0.3201

<div class="k-default-codeblock">
```

```
</div>
  11/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:50 2s/step - loss: 1.0372 - sparse_categorical_accuracy: 0.3208

<div class="k-default-codeblock">
```

```
</div>
  12/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:41 2s/step - loss: 1.0402 - sparse_categorical_accuracy: 0.3214

<div class="k-default-codeblock">
```

```
</div>
  13/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:34 2s/step - loss: 1.0422 - sparse_categorical_accuracy: 0.3219

<div class="k-default-codeblock">
```

```
</div>
  14/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 2s/step - loss: 1.0422 - sparse_categorical_accuracy: 0.3223

<div class="k-default-codeblock">
```

```
</div>
  15/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 2s/step - loss: 1.0447 - sparse_categorical_accuracy: 0.3227

<div class="k-default-codeblock">
```

```
</div>
  16/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 2s/step - loss: 1.0463 - sparse_categorical_accuracy: 0.3230

<div class="k-default-codeblock">
```

```
</div>
  17/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:14 2s/step - loss: 1.0474 - sparse_categorical_accuracy: 0.3231

<div class="k-default-codeblock">
```

```
</div>
  18/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:10 2s/step - loss: 1.0482 - sparse_categorical_accuracy: 0.3233

<div class="k-default-codeblock">
```

```
</div>
  19/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:06 2s/step - loss: 1.0475 - sparse_categorical_accuracy: 0.3235

<div class="k-default-codeblock">
```

```
</div>
  20/100 ━━━━[37m━━━━━━━━━━━━━━━━  2:03 2s/step - loss: 1.0468 - sparse_categorical_accuracy: 0.3236

<div class="k-default-codeblock">
```

```
</div>
  21/100 ━━━━[37m━━━━━━━━━━━━━━━━  2:00 2s/step - loss: 1.0469 - sparse_categorical_accuracy: 0.3235

<div class="k-default-codeblock">
```

```
</div>
  22/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:57 2s/step - loss: 1.0466 - sparse_categorical_accuracy: 0.3235

<div class="k-default-codeblock">
```

```
</div>
  23/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:54 1s/step - loss: 1.0467 - sparse_categorical_accuracy: 0.3236

<div class="k-default-codeblock">
```

```
</div>
  24/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:52 1s/step - loss: 1.0471 - sparse_categorical_accuracy: 0.3237

<div class="k-default-codeblock">
```

```
</div>
  25/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:49 1s/step - loss: 1.0466 - sparse_categorical_accuracy: 0.3238

<div class="k-default-codeblock">
```

```
</div>
  26/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:47 1s/step - loss: 1.0465 - sparse_categorical_accuracy: 0.3239

<div class="k-default-codeblock">
```

```
</div>
  27/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:44 1s/step - loss: 1.0459 - sparse_categorical_accuracy: 0.3240

<div class="k-default-codeblock">
```

```
</div>
  28/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:42 1s/step - loss: 1.0457 - sparse_categorical_accuracy: 0.3242

<div class="k-default-codeblock">
```

```
</div>
  29/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:40 1s/step - loss: 1.0447 - sparse_categorical_accuracy: 0.3243

<div class="k-default-codeblock">
```

```
</div>
  30/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:38 1s/step - loss: 1.0438 - sparse_categorical_accuracy: 0.3245

<div class="k-default-codeblock">
```

```
</div>
  31/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:36 1s/step - loss: 1.0434 - sparse_categorical_accuracy: 0.3246

<div class="k-default-codeblock">
```

```
</div>
  32/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:34 1s/step - loss: 1.0427 - sparse_categorical_accuracy: 0.3248

<div class="k-default-codeblock">
```

```
</div>
  33/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:32 1s/step - loss: 1.0418 - sparse_categorical_accuracy: 0.3249

<div class="k-default-codeblock">
```

```
</div>
  34/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:30 1s/step - loss: 1.0409 - sparse_categorical_accuracy: 0.3250

<div class="k-default-codeblock">
```

```
</div>
  35/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:29 1s/step - loss: 1.0402 - sparse_categorical_accuracy: 0.3251

<div class="k-default-codeblock">
```

```
</div>
  36/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:27 1s/step - loss: 1.0397 - sparse_categorical_accuracy: 0.3252

<div class="k-default-codeblock">
```

```
</div>
  37/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:25 1s/step - loss: 1.0390 - sparse_categorical_accuracy: 0.3253

<div class="k-default-codeblock">
```

```
</div>
  38/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:24 1s/step - loss: 1.0384 - sparse_categorical_accuracy: 0.3255

<div class="k-default-codeblock">
```

```
</div>
  39/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:22 1s/step - loss: 1.0378 - sparse_categorical_accuracy: 0.3256

<div class="k-default-codeblock">
```

```
</div>
  40/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:20 1s/step - loss: 1.0376 - sparse_categorical_accuracy: 0.3257

<div class="k-default-codeblock">
```

```
</div>
  41/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:19 1s/step - loss: 1.0371 - sparse_categorical_accuracy: 0.3259

<div class="k-default-codeblock">
```

```
</div>
  42/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:17 1s/step - loss: 1.0364 - sparse_categorical_accuracy: 0.3260

<div class="k-default-codeblock">
```

```
</div>
  43/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:16 1s/step - loss: 1.0357 - sparse_categorical_accuracy: 0.3261

<div class="k-default-codeblock">
```

```
</div>
  44/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:14 1s/step - loss: 1.0349 - sparse_categorical_accuracy: 0.3262

<div class="k-default-codeblock">
```

```
</div>
  45/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:13 1s/step - loss: 1.0343 - sparse_categorical_accuracy: 0.3264

<div class="k-default-codeblock">
```

```
</div>
  46/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:11 1s/step - loss: 1.0338 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  47/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:10 1s/step - loss: 1.0333 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  48/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:08 1s/step - loss: 1.0329 - sparse_categorical_accuracy: 0.3266

<div class="k-default-codeblock">
```

```
</div>
  49/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:07 1s/step - loss: 1.0325 - sparse_categorical_accuracy: 0.3267

<div class="k-default-codeblock">
```

```
</div>
  50/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:05 1s/step - loss: 1.0323 - sparse_categorical_accuracy: 0.3268

<div class="k-default-codeblock">
```

```
</div>
  51/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:04 1s/step - loss: 1.0320 - sparse_categorical_accuracy: 0.3268

<div class="k-default-codeblock">
```

```
</div>
  52/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:02 1s/step - loss: 1.0318 - sparse_categorical_accuracy: 0.3269

<div class="k-default-codeblock">
```

```
</div>
  53/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:01 1s/step - loss: 1.0316 - sparse_categorical_accuracy: 0.3269

<div class="k-default-codeblock">
```

```
</div>
  54/100 ━━━━━━━━━━[37m━━━━━━━━━━  59s 1s/step - loss: 1.0313 - sparse_categorical_accuracy: 0.3270 

<div class="k-default-codeblock">
```

```
</div>
  55/100 ━━━━━━━━━━━[37m━━━━━━━━━  58s 1s/step - loss: 1.0309 - sparse_categorical_accuracy: 0.3270

<div class="k-default-codeblock">
```

```
</div>
  56/100 ━━━━━━━━━━━[37m━━━━━━━━━  57s 1s/step - loss: 1.0307 - sparse_categorical_accuracy: 0.3271

<div class="k-default-codeblock">
```

```
</div>
  57/100 ━━━━━━━━━━━[37m━━━━━━━━━  55s 1s/step - loss: 1.0303 - sparse_categorical_accuracy: 0.3271

<div class="k-default-codeblock">
```

```
</div>
  58/100 ━━━━━━━━━━━[37m━━━━━━━━━  54s 1s/step - loss: 1.0299 - sparse_categorical_accuracy: 0.3272

<div class="k-default-codeblock">
```

```
</div>
  59/100 ━━━━━━━━━━━[37m━━━━━━━━━  52s 1s/step - loss: 1.0295 - sparse_categorical_accuracy: 0.3272

<div class="k-default-codeblock">
```

```
</div>
  60/100 ━━━━━━━━━━━━[37m━━━━━━━━  51s 1s/step - loss: 1.0291 - sparse_categorical_accuracy: 0.3273

<div class="k-default-codeblock">
```

```
</div>
  61/100 ━━━━━━━━━━━━[37m━━━━━━━━  50s 1s/step - loss: 1.0288 - sparse_categorical_accuracy: 0.3273

<div class="k-default-codeblock">
```

```
</div>
  62/100 ━━━━━━━━━━━━[37m━━━━━━━━  48s 1s/step - loss: 1.0283 - sparse_categorical_accuracy: 0.3273

<div class="k-default-codeblock">
```

```
</div>
  63/100 ━━━━━━━━━━━━[37m━━━━━━━━  47s 1s/step - loss: 1.0278 - sparse_categorical_accuracy: 0.3274

<div class="k-default-codeblock">
```

```
</div>
  64/100 ━━━━━━━━━━━━[37m━━━━━━━━  45s 1s/step - loss: 1.0273 - sparse_categorical_accuracy: 0.3274

<div class="k-default-codeblock">
```

```
</div>
  65/100 ━━━━━━━━━━━━━[37m━━━━━━━  44s 1s/step - loss: 1.0268 - sparse_categorical_accuracy: 0.3274

<div class="k-default-codeblock">
```

```
</div>
  66/100 ━━━━━━━━━━━━━[37m━━━━━━━  43s 1s/step - loss: 1.0263 - sparse_categorical_accuracy: 0.3275

<div class="k-default-codeblock">
```

```
</div>
  67/100 ━━━━━━━━━━━━━[37m━━━━━━━  41s 1s/step - loss: 1.0259 - sparse_categorical_accuracy: 0.3275

<div class="k-default-codeblock">
```

```
</div>
  68/100 ━━━━━━━━━━━━━[37m━━━━━━━  40s 1s/step - loss: 1.0255 - sparse_categorical_accuracy: 0.3276

<div class="k-default-codeblock">
```

```
</div>
  69/100 ━━━━━━━━━━━━━[37m━━━━━━━  39s 1s/step - loss: 1.0250 - sparse_categorical_accuracy: 0.3277

<div class="k-default-codeblock">
```

```
</div>
  70/100 ━━━━━━━━━━━━━━[37m━━━━━━  37s 1s/step - loss: 1.0245 - sparse_categorical_accuracy: 0.3277

<div class="k-default-codeblock">
```

```
</div>
  71/100 ━━━━━━━━━━━━━━[37m━━━━━━  36s 1s/step - loss: 1.0241 - sparse_categorical_accuracy: 0.3278

<div class="k-default-codeblock">
```

```
</div>
  72/100 ━━━━━━━━━━━━━━[37m━━━━━━  35s 1s/step - loss: 1.0238 - sparse_categorical_accuracy: 0.3278

<div class="k-default-codeblock">
```

```
</div>
  73/100 ━━━━━━━━━━━━━━[37m━━━━━━  34s 1s/step - loss: 1.0234 - sparse_categorical_accuracy: 0.3279

<div class="k-default-codeblock">
```

```
</div>
  74/100 ━━━━━━━━━━━━━━[37m━━━━━━  32s 1s/step - loss: 1.0230 - sparse_categorical_accuracy: 0.3279

<div class="k-default-codeblock">
```

```
</div>
  75/100 ━━━━━━━━━━━━━━━[37m━━━━━  31s 1s/step - loss: 1.0228 - sparse_categorical_accuracy: 0.3279

<div class="k-default-codeblock">
```

```
</div>
  76/100 ━━━━━━━━━━━━━━━[37m━━━━━  30s 1s/step - loss: 1.0226 - sparse_categorical_accuracy: 0.3280

<div class="k-default-codeblock">
```

```
</div>
  77/100 ━━━━━━━━━━━━━━━[37m━━━━━  28s 1s/step - loss: 1.0224 - sparse_categorical_accuracy: 0.3280

<div class="k-default-codeblock">
```

```
</div>
  78/100 ━━━━━━━━━━━━━━━[37m━━━━━  27s 1s/step - loss: 1.0222 - sparse_categorical_accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
  79/100 ━━━━━━━━━━━━━━━[37m━━━━━  26s 1s/step - loss: 1.0219 - sparse_categorical_accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
  80/100 ━━━━━━━━━━━━━━━━[37m━━━━  25s 1s/step - loss: 1.0216 - sparse_categorical_accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
  81/100 ━━━━━━━━━━━━━━━━[37m━━━━  23s 1s/step - loss: 1.0214 - sparse_categorical_accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
  82/100 ━━━━━━━━━━━━━━━━[37m━━━━  22s 1s/step - loss: 1.0211 - sparse_categorical_accuracy: 0.3282

<div class="k-default-codeblock">
```

```
</div>
  83/100 ━━━━━━━━━━━━━━━━[37m━━━━  21s 1s/step - loss: 1.0209 - sparse_categorical_accuracy: 0.3282

<div class="k-default-codeblock">
```

```
</div>
  84/100 ━━━━━━━━━━━━━━━━[37m━━━━  19s 1s/step - loss: 1.0206 - sparse_categorical_accuracy: 0.3282

<div class="k-default-codeblock">
```

```
</div>
  85/100 ━━━━━━━━━━━━━━━━━[37m━━━  18s 1s/step - loss: 1.0205 - sparse_categorical_accuracy: 0.3282

<div class="k-default-codeblock">
```

```
</div>
  86/100 ━━━━━━━━━━━━━━━━━[37m━━━  17s 1s/step - loss: 1.0203 - sparse_categorical_accuracy: 0.3283

<div class="k-default-codeblock">
```

```
</div>
  87/100 ━━━━━━━━━━━━━━━━━[37m━━━  16s 1s/step - loss: 1.0201 - sparse_categorical_accuracy: 0.3283

<div class="k-default-codeblock">
```

```
</div>
  88/100 ━━━━━━━━━━━━━━━━━[37m━━━  14s 1s/step - loss: 1.0199 - sparse_categorical_accuracy: 0.3283

<div class="k-default-codeblock">
```

```
</div>
  89/100 ━━━━━━━━━━━━━━━━━[37m━━━  13s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3284

<div class="k-default-codeblock">
```

```
</div>
  90/100 ━━━━━━━━━━━━━━━━━━[37m━━  12s 1s/step - loss: 1.0196 - sparse_categorical_accuracy: 0.3284

<div class="k-default-codeblock">
```

```
</div>
  91/100 ━━━━━━━━━━━━━━━━━━[37m━━  11s 1s/step - loss: 1.0195 - sparse_categorical_accuracy: 0.3284

<div class="k-default-codeblock">
```

```
</div>
  92/100 ━━━━━━━━━━━━━━━━━━[37m━━  9s 1s/step - loss: 1.0194 - sparse_categorical_accuracy: 0.3285 

<div class="k-default-codeblock">
```

```
</div>
  93/100 ━━━━━━━━━━━━━━━━━━[37m━━  8s 1s/step - loss: 1.0193 - sparse_categorical_accuracy: 0.3285

<div class="k-default-codeblock">
```

```
</div>
  94/100 ━━━━━━━━━━━━━━━━━━[37m━━  7s 1s/step - loss: 1.0192 - sparse_categorical_accuracy: 0.3285

<div class="k-default-codeblock">
```

```
</div>
  95/100 ━━━━━━━━━━━━━━━━━━━[37m━  6s 1s/step - loss: 1.0192 - sparse_categorical_accuracy: 0.3286

<div class="k-default-codeblock">
```

```
</div>
  96/100 ━━━━━━━━━━━━━━━━━━━[37m━  4s 1s/step - loss: 1.0191 - sparse_categorical_accuracy: 0.3286

<div class="k-default-codeblock">
```

```
</div>
  97/100 ━━━━━━━━━━━━━━━━━━━[37m━  3s 1s/step - loss: 1.0191 - sparse_categorical_accuracy: 0.3286

<div class="k-default-codeblock">
```

```
</div>
  98/100 ━━━━━━━━━━━━━━━━━━━[37m━  2s 1s/step - loss: 1.0190 - sparse_categorical_accuracy: 0.3287

<div class="k-default-codeblock">
```

```
</div>
  99/100 ━━━━━━━━━━━━━━━━━━━[37m━  1s 1s/step - loss: 1.0190 - sparse_categorical_accuracy: 0.3287

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 1.0190 - sparse_categorical_accuracy: 0.3288

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 152s 1s/step - loss: 1.0190 - sparse_categorical_accuracy: 0.3288





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7fe4a03ee350>

```
</div>
---
## Save the Model Locally

To upload a model, you need to first save the model locally using `save_to_preset`.


```python
preset_dir = "./gpt2_imdb"
causal_lm.save_to_preset(preset_dir)
```

Let's see what are the files what are the saved files.


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

A model that is saved to a local preset, can be loaded using `from_preset`.
What you save in, is what you get back out.


```python
causal_lm = keras_nlp.models.CausalLM.from_preset(preset_dir)
```

You can also load the `keras_nlp.models.Backbone` and `keras_nlp.models.Tokenizer` objects from this preset directory.
Note that these objects are equivalent to `causal_lm.backbone` and `causal_lm.preprocessor.tokenizer` above.


```python
backbone = keras_nlp.models.Backbone.from_preset(preset_dir)
tokenizer = keras_nlp.models.Tokenizer.from_preset(preset_dir)
```

---
## Upload the Model to a Model Hub

After saving a preset to a directory, this directory can be uploaded to a model hub such as Kaggle or Hugging Face directly from the KerasNLP library.
To upload the model to Kaggle, the URI should start with `kaggle://` and to upload to Hugging Face, it should start with `hf://`.

### Upload to Kaggle

To upload a model to Kaggle, first, we need to authenticate with Kaggle.
This can by one of the followings:
1. Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. Provide a local `~/.kaggle/kaggle.json`.
3. Call `kagglehub.login()`.

Let's make sure we are logged in before coninuing.


```python
import kagglehub

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

```

To upload a model we can use `keras_nlp.upload_preset(uri, preset_dir)` API where `uri` has the format of
`kaggle://<KAGGLE_USERNAME>/<MODEL>/Keras/<VARIATION>` for uploading to Kaggle and `preset_dir` is the directory that the model is saved in.

Running the following uploads the model that is saved in `preset_dir` to Kaggle:


```python
kaggle_username = kagglehub.whoami()["username"]
kaggle_uri = f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
keras_nlp.upload_preset(kaggle_uri, preset_dir)
```

<div class="k-default-codeblock">
```
Starting upload for file preprocessor.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/834 [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 834/834 [00:00<00:00, 1.33kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (834B)

Starting upload for file tokenizer.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/322 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 322/322 [00:00<00:00, 528B/s]

    
<div class="k-default-codeblock">
```
Upload successful: tokenizer.json (322B)

Starting upload for file task.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/1.82k [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.82k/1.82k [00:00<00:00, 2.27kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.json (2KB)

Starting upload for file model.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/498M [00:00<?, ?B/s]

    
Uploading:   1%|█▎                                                                                                                                                                                                  | 3.39M/498M [00:00<00:14, 33.9MB/s]

    
Uploading:   3%|██████▏                                                                                                                                                                                             | 15.6M/498M [00:00<00:05, 85.7MB/s]

    
Uploading:   6%|████████████                                                                                                                                                                                        | 30.5M/498M [00:00<00:05, 90.3MB/s]

    
Uploading:  10%|██████████████████▋                                                                                                                                                                                 | 47.5M/498M [00:00<00:04, 99.6MB/s]

    
Uploading:  13%|█████████████████████████▍                                                                                                                                                                           | 64.2M/498M [00:00<00:04, 106MB/s]

    
Uploading:  16%|████████████████████████████████                                                                                                                                                                     | 81.0M/498M [00:00<00:03, 105MB/s]

    
Uploading:  20%|██████████████████████████████████████▋                                                                                                                                                              | 97.9M/498M [00:00<00:03, 104MB/s]

    
Uploading:  23%|█████████████████████████████████████████████▌                                                                                                                                                        | 115M/498M [00:01<00:03, 106MB/s]

    
Uploading:  26%|████████████████████████████████████████████████████▏                                                                                                                                                 | 131M/498M [00:01<00:03, 107MB/s]

    
Uploading:  30%|██████████████████████████████████████████████████████████▉                                                                                                                                           | 148M/498M [00:01<00:03, 113MB/s]

    
Uploading:  33%|█████████████████████████████████████████████████████████████████▋                                                                                                                                    | 165M/498M [00:01<00:03, 109MB/s]

    
Uploading:  37%|████████████████████████████████████████████████████████████████████████▎                                                                                                                             | 182M/498M [00:01<00:02, 109MB/s]

    
Uploading:  40%|██████████████████████████████████████████████████████████████████████████████▉                                                                                                                       | 199M/498M [00:01<00:02, 112MB/s]

    
Uploading:  43%|█████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                | 215M/498M [00:02<00:02, 109MB/s]

    
Uploading:  47%|████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                         | 232M/498M [00:02<00:02, 111MB/s]

    
Uploading:  50%|██████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                   | 249M/498M [00:02<00:02, 111MB/s]

    
Uploading:  53%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                            | 266M/498M [00:02<00:02, 112MB/s]

    
Uploading:  57%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                     | 283M/498M [00:02<00:01, 109MB/s]

    
Uploading:  60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                               | 299M/498M [00:02<00:01, 111MB/s]

    
Uploading:  63%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                        | 316M/498M [00:02<00:01, 109MB/s]

    
Uploading:  67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                 | 333M/498M [00:03<00:01, 109MB/s]

    
Uploading:  70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                           | 350M/498M [00:03<00:01, 111MB/s]

    
Uploading:  74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                    | 366M/498M [00:03<00:01, 113MB/s]

    
Uploading:  77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 383M/498M [00:03<00:01, 109MB/s]

    
Uploading:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                       | 400M/498M [00:03<00:00, 110MB/s]

    
Uploading:  84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                | 417M/498M [00:03<00:00, 112MB/s]

    
Uploading:  87%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                         | 434M/498M [00:04<00:00, 109MB/s]

    
Uploading:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                   | 450M/498M [00:04<00:00, 111MB/s]

    
Uploading:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋            | 467M/498M [00:04<00:00, 108MB/s]

    
Uploading:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎     | 484M/498M [00:04<00:00, 109MB/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 498M/498M [00:05<00:00, 96.0MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: model.weights.h5 (475MB)

Starting upload for file config.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/431 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 431/431 [00:01<00:00, 353B/s]

    
<div class="k-default-codeblock">
```
Upload successful: config.json (431B)

Starting upload for file metadata.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/142 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:00<00:00, 177B/s]

    
<div class="k-default-codeblock">
```
Upload successful: metadata.json (142B)

Starting upload for file merges.txt

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/456k [00:00<?, ?B/s]

    
Uploading: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:01<00:00, 394kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: merges.txt (446KB)

Starting upload for file vocabulary.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/1.04M [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 1.20MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: vocabulary.json (1018KB)

Your model instance version has been created.
Files are being processed...
See at: https://www.kaggle.com/models/smnsdt/gpt2/keras/gpt2_imdb

```
</div>
### Upload to Hugging Face

To upload a model to Hugging Face, first, we need to authenticate with Hugging Face.
This can by one of the followings:
1. Set environment variables `HF_USERNAME` and `HF_TOKEN`.
2. Call `huggingface_hub.notebook_login()`.

Let's make sure we are logged in before coninuing.


```python
import huggingface_hub

if "HF_USERNAME" not in os.environ or "HF_TOKEN" not in os.environ:
    huggingface_hub.notebook_login()
```

`keras_nlp.upload_preset(uri, preset_dir)` can be used to upload a model to Hugging Face if `uri` has the format of
`kaggle://<HF_USERNAME>/<MODEL>`.

Running the following uploads the model that is saved in `preset_dir` to Hugging Face:


```python
hf_username = huggingface_hub.whoami()["name"]
hf_uri = f"hf://{hf_username}/gpt2_imdb"
keras_nlp.upload_preset(hf_uri, preset_dir)

```


<div class="k-default-codeblock">
```
model.weights.h5:   0%|          | 0.00/498M [00:00<?, ?B/s]

```
</div>
---
## Load a User Uploaded Model

After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset(
    f"kaggle://{kaggle_username}/gpt2/keras/gpt2_imdb"
)
```

We can also load the model uploaded to Hugging Face by calling `from_preset`.

```python
causal_lm = keras_nlp.models.CausalLM.from_preset(f"hf://{hf_username}/gpt2_imdb")
```

# Classifier Upload

Uploading a classifier model is similar to Causal LM upload.
To upload the fine-tuned model, first, the model should be saved to a local directory using `save_to_preset`
API and then it can be uploaded via `keras_nlp.upload_preset`.


```python
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
```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/metadata.json...

```
</div>
    
  0%|                                                                                                                                                                                                                         | 0.00/139 [00:00<?, ?B/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 139/139 [00:00<00:00, 233kB/s]

    
<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/task.json...

Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/config.json...

```
</div>
    
  0%|                                                                                                                                                                                                                         | 0.00/507 [00:00<?, ?B/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 507/507 [00:00<00:00, 1.04MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/model.weights.h5...

```
</div>
    
  0%|                                                                                                                                                                                                                       | 0.00/16.8M [00:00<?, ?B/s]

    
  6%|████████████▎                                                                                                                                                                                                 | 1.00M/16.8M [00:00<00:05, 2.79MB/s]

    
 18%|████████████████████████████████████▊                                                                                                                                                                         | 3.00M/16.8M [00:00<00:01, 7.78MB/s]

    
 54%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                               | 9.00M/16.8M [00:00<00:00, 22.4MB/s]

    
 89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                      | 15.0M/16.8M [00:00<00:00, 33.7MB/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16.8M/16.8M [00:00<00:00, 24.4MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/preprocessor.json...

Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/tokenizer.json...

```
</div>
    
  0%|                                                                                                                                                                                                                         | 0.00/547 [00:00<?, ?B/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 547/547 [00:00<00:00, 945kB/s]

    
<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/assets/tokenizer/vocabulary.txt...

```
</div>
    
  0%|                                                                                                                                                                                                                        | 0.00/226k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 226k/226k [00:00<00:00, 1.09MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 226k/226k [00:00<00:00, 1.08MB/s]

    


    
   1/100 [37m━━━━━━━━━━━━━━━━━━━━  6:52 4s/step - loss: 0.7252 - sparse_categorical_accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
   2/100 [37m━━━━━━━━━━━━━━━━━━━━  3:14 2s/step - loss: 0.7127 - sparse_categorical_accuracy: 0.3125

<div class="k-default-codeblock">
```

```
</div>
   6/100 ━[37m━━━━━━━━━━━━━━━━━━━  38s 408ms/step - loss: 0.7043 - sparse_categorical_accuracy: 0.3979

<div class="k-default-codeblock">
```

```
</div>
  10/100 ━━[37m━━━━━━━━━━━━━━━━━━  20s 232ms/step - loss: 0.7028 - sparse_categorical_accuracy: 0.4297

<div class="k-default-codeblock">
```

```
</div>
  14/100 ━━[37m━━━━━━━━━━━━━━━━━━  14s 165ms/step - loss: 0.7018 - sparse_categorical_accuracy: 0.4512

<div class="k-default-codeblock">
```

```
</div>
  19/100 ━━━[37m━━━━━━━━━━━━━━━━━  9s 122ms/step - loss: 0.7023 - sparse_categorical_accuracy: 0.4609 

<div class="k-default-codeblock">
```

```
</div>
  24/100 ━━━━[37m━━━━━━━━━━━━━━━━  7s 98ms/step - loss: 0.7030 - sparse_categorical_accuracy: 0.4662 

<div class="k-default-codeblock">
```

```
</div>
  29/100 ━━━━━[37m━━━━━━━━━━━━━━━  5s 82ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.4704

<div class="k-default-codeblock">
```

```
</div>
  34/100 ━━━━━━[37m━━━━━━━━━━━━━━  4s 71ms/step - loss: 0.7031 - sparse_categorical_accuracy: 0.4754

<div class="k-default-codeblock">
```

```
</div>
  41/100 ━━━━━━━━[37m━━━━━━━━━━━━  3s 60ms/step - loss: 0.7023 - sparse_categorical_accuracy: 0.4813

<div class="k-default-codeblock">
```

```
</div>
  47/100 ━━━━━━━━━[37m━━━━━━━━━━━  2s 53ms/step - loss: 0.7018 - sparse_categorical_accuracy: 0.4841

<div class="k-default-codeblock">
```

```
</div>
  53/100 ━━━━━━━━━━[37m━━━━━━━━━━  2s 48ms/step - loss: 0.7014 - sparse_categorical_accuracy: 0.4846

<div class="k-default-codeblock">
```

```
</div>
  59/100 ━━━━━━━━━━━[37m━━━━━━━━━  1s 44ms/step - loss: 0.7009 - sparse_categorical_accuracy: 0.4864

<div class="k-default-codeblock">
```

```
</div>
  66/100 ━━━━━━━━━━━━━[37m━━━━━━━  1s 40ms/step - loss: 0.7003 - sparse_categorical_accuracy: 0.4886

<div class="k-default-codeblock">
```

```
</div>
  72/100 ━━━━━━━━━━━━━━[37m━━━━━━  1s 37ms/step - loss: 0.6999 - sparse_categorical_accuracy: 0.4895

<div class="k-default-codeblock">
```

```
</div>
  78/100 ━━━━━━━━━━━━━━━[37m━━━━━  0s 35ms/step - loss: 0.6996 - sparse_categorical_accuracy: 0.4898

<div class="k-default-codeblock">
```

```
</div>
  84/100 ━━━━━━━━━━━━━━━━[37m━━━━  0s 33ms/step - loss: 0.6993 - sparse_categorical_accuracy: 0.4906

<div class="k-default-codeblock">
```

```
</div>
  90/100 ━━━━━━━━━━━━━━━━━━[37m━━  0s 32ms/step - loss: 0.6990 - sparse_categorical_accuracy: 0.4907

<div class="k-default-codeblock">
```

```
</div>
  97/100 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - loss: 0.6987 - sparse_categorical_accuracy: 0.4904

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 7s 29ms/step - loss: 0.6986 - sparse_categorical_accuracy: 0.4902


<div class="k-default-codeblock">
```
Starting upload for file preprocessor.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/947 [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 947/947 [00:00<00:00, 1.53kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (947B)

Starting upload for file tokenizer.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/461 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 461/461 [00:00<00:00, 751B/s]

    
<div class="k-default-codeblock">
```
Upload successful: tokenizer.json (461B)

Starting upload for file task.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/2.09k [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.09k/2.09k [00:00<00:00, 3.01kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.json (2KB)

Starting upload for file task.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/52.8M [00:00<?, ?B/s]

    
Uploading:   5%|█████████▌                                                                                                                                                                                         | 2.59M/52.8M [00:00<00:01, 25.9MB/s]

    
Uploading:  23%|████████████████████████████████████████████▌                                                                                                                                                      | 12.1M/52.8M [00:00<00:00, 65.4MB/s]

    
Uploading:  55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                        | 29.1M/52.8M [00:00<00:00, 105MB/s]

    
Uploading:  74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                 | 39.3M/52.8M [00:00<00:00, 98.5MB/s]

    
Uploading:  93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏             | 49.0M/52.8M [00:00<00:00, 87.2MB/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.8M/52.8M [00:01<00:00, 40.4MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.weights.h5 (50MB)

Starting upload for file model.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/17.6M [00:00<?, ?B/s]

    
Uploading:   5%|██████████▍                                                                                                                                                                                         | 934k/17.6M [00:00<00:01, 9.07MB/s]

    
Uploading:  47%|████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                      | 8.34M/17.6M [00:00<00:00, 45.9MB/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17.6M/17.6M [00:00<00:00, 21.6MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: model.weights.h5 (17MB)

Starting upload for file config.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/454 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [00:00<00:00, 679B/s]

    
<div class="k-default-codeblock">
```
Upload successful: config.json (454B)

Starting upload for file metadata.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/140 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140/140 [00:00<00:00, 214B/s]

    
<div class="k-default-codeblock">
```
Upload successful: metadata.json (140B)

Starting upload for file vocabulary.txt

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/232k [00:00<?, ?B/s]

    
Uploading: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 358kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: vocabulary.txt (226KB)

Your model instance version has been created.
Files are being processed...
See at: https://www.kaggle.com/models/smnsdt/bert/keras/bert_tiny_imdb

```
</div>
After verifying that the model is uploaded to Kaggle, we can load the model by calling `from_preset`.

```python
classifier = keras_nlp.models.Classifier.from_preset(
    f"kaggle://{kaggle_username}/bert/keras/bert_tiny_imdb"
)
```
