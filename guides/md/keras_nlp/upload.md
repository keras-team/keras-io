# Uploading Models with KerasNLP

**Author:** [Samaneh Saadat](https://github.com/SamanehSaadat/), [Matthew Watson](https://github.com/mattdangerw/)<br>
**Date created:** 2024/04/29<br>
**Last modified:** 2024/04/29<br>
**Description:** An introduction on how to upload a fine-tuned KerasNLP model to model hubs.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_nlp/upload.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_nlp/upload.py)



# Introduction

Fine-tuning a machine learning model can yield impressive results for specific tasks.
Uploading your fine-tuned model to a model hub allows you to share it with the broader community.
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

    
   1/100 [37m━━━━━━━━━━━━━━━━━━━━  49:12 30s/step - loss: 0.5498 - sparse_categorical_accuracy: 0.2670

<div class="k-default-codeblock">
```

```
</div>
   2/100 [37m━━━━━━━━━━━━━━━━━━━━  15:09 9s/step - loss: 0.7592 - sparse_categorical_accuracy: 0.2922 

<div class="k-default-codeblock">
```

```
</div>
   3/100 [37m━━━━━━━━━━━━━━━━━━━━  8:22 5s/step - loss: 0.8293 - sparse_categorical_accuracy: 0.2996 

<div class="k-default-codeblock">
```

```
</div>
   4/100 [37m━━━━━━━━━━━━━━━━━━━━  6:06 4s/step - loss: 0.8904 - sparse_categorical_accuracy: 0.3034

<div class="k-default-codeblock">
```

```
</div>
   5/100 ━[37m━━━━━━━━━━━━━━━━━━━  4:58 3s/step - loss: 0.9518 - sparse_categorical_accuracy: 0.3070

<div class="k-default-codeblock">
```

```
</div>
   6/100 ━[37m━━━━━━━━━━━━━━━━━━━  4:16 3s/step - loss: 0.9908 - sparse_categorical_accuracy: 0.3097

<div class="k-default-codeblock">
```

```
</div>
   7/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:48 2s/step - loss: 1.0098 - sparse_categorical_accuracy: 0.3112

<div class="k-default-codeblock">
```

```
</div>
   8/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:28 2s/step - loss: 1.0264 - sparse_categorical_accuracy: 0.3129

<div class="k-default-codeblock">
```

```
</div>
   9/100 ━[37m━━━━━━━━━━━━━━━━━━━  3:12 2s/step - loss: 1.0342 - sparse_categorical_accuracy: 0.3143

<div class="k-default-codeblock">
```

```
</div>
  10/100 ━━[37m━━━━━━━━━━━━━━━━━━  3:00 2s/step - loss: 1.0358 - sparse_categorical_accuracy: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  11/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:50 2s/step - loss: 1.0401 - sparse_categorical_accuracy: 0.3161

<div class="k-default-codeblock">
```

```
</div>
  12/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:42 2s/step - loss: 1.0431 - sparse_categorical_accuracy: 0.3168

<div class="k-default-codeblock">
```

```
</div>
  13/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:35 2s/step - loss: 1.0450 - sparse_categorical_accuracy: 0.3174

<div class="k-default-codeblock">
```

```
</div>
  14/100 ━━[37m━━━━━━━━━━━━━━━━━━  2:28 2s/step - loss: 1.0449 - sparse_categorical_accuracy: 0.3179

<div class="k-default-codeblock">
```

```
</div>
  15/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:23 2s/step - loss: 1.0473 - sparse_categorical_accuracy: 0.3184

<div class="k-default-codeblock">
```

```
</div>
  16/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:18 2s/step - loss: 1.0489 - sparse_categorical_accuracy: 0.3189

<div class="k-default-codeblock">
```

```
</div>
  17/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:14 2s/step - loss: 1.0499 - sparse_categorical_accuracy: 0.3192

<div class="k-default-codeblock">
```

```
</div>
  18/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:10 2s/step - loss: 1.0506 - sparse_categorical_accuracy: 0.3195

<div class="k-default-codeblock">
```

```
</div>
  19/100 ━━━[37m━━━━━━━━━━━━━━━━━  2:06 2s/step - loss: 1.0498 - sparse_categorical_accuracy: 0.3198

<div class="k-default-codeblock">
```

```
</div>
  20/100 ━━━━[37m━━━━━━━━━━━━━━━━  2:02 2s/step - loss: 1.0489 - sparse_categorical_accuracy: 0.3201

<div class="k-default-codeblock">
```

```
</div>
  21/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:59 2s/step - loss: 1.0489 - sparse_categorical_accuracy: 0.3202

<div class="k-default-codeblock">
```

```
</div>
  22/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:56 1s/step - loss: 1.0486 - sparse_categorical_accuracy: 0.3202

<div class="k-default-codeblock">
```

```
</div>
  23/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:53 1s/step - loss: 1.0487 - sparse_categorical_accuracy: 0.3204

<div class="k-default-codeblock">
```

```
</div>
  24/100 ━━━━[37m━━━━━━━━━━━━━━━━  1:51 1s/step - loss: 1.0489 - sparse_categorical_accuracy: 0.3206

<div class="k-default-codeblock">
```

```
</div>
  25/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:48 1s/step - loss: 1.0485 - sparse_categorical_accuracy: 0.3207

<div class="k-default-codeblock">
```

```
</div>
  26/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:46 1s/step - loss: 1.0483 - sparse_categorical_accuracy: 0.3209

<div class="k-default-codeblock">
```

```
</div>
  27/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:44 1s/step - loss: 1.0477 - sparse_categorical_accuracy: 0.3211

<div class="k-default-codeblock">
```

```
</div>
  28/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:41 1s/step - loss: 1.0475 - sparse_categorical_accuracy: 0.3213

<div class="k-default-codeblock">
```

```
</div>
  29/100 ━━━━━[37m━━━━━━━━━━━━━━━  1:39 1s/step - loss: 1.0465 - sparse_categorical_accuracy: 0.3215

<div class="k-default-codeblock">
```

```
</div>
  30/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:37 1s/step - loss: 1.0456 - sparse_categorical_accuracy: 0.3217

<div class="k-default-codeblock">
```

```
</div>
  31/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:35 1s/step - loss: 1.0451 - sparse_categorical_accuracy: 0.3219

<div class="k-default-codeblock">
```

```
</div>
  32/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:33 1s/step - loss: 1.0444 - sparse_categorical_accuracy: 0.3221

<div class="k-default-codeblock">
```

```
</div>
  33/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:31 1s/step - loss: 1.0435 - sparse_categorical_accuracy: 0.3223

<div class="k-default-codeblock">
```

```
</div>
  34/100 ━━━━━━[37m━━━━━━━━━━━━━━  1:30 1s/step - loss: 1.0426 - sparse_categorical_accuracy: 0.3224

<div class="k-default-codeblock">
```

```
</div>
  35/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:28 1s/step - loss: 1.0419 - sparse_categorical_accuracy: 0.3226

<div class="k-default-codeblock">
```

```
</div>
  36/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:26 1s/step - loss: 1.0413 - sparse_categorical_accuracy: 0.3227

<div class="k-default-codeblock">
```

```
</div>
  37/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:24 1s/step - loss: 1.0406 - sparse_categorical_accuracy: 0.3229

<div class="k-default-codeblock">
```

```
</div>
  38/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:23 1s/step - loss: 1.0400 - sparse_categorical_accuracy: 0.3231

<div class="k-default-codeblock">
```

```
</div>
  39/100 ━━━━━━━[37m━━━━━━━━━━━━━  1:21 1s/step - loss: 1.0394 - sparse_categorical_accuracy: 0.3232

<div class="k-default-codeblock">
```

```
</div>
  40/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:19 1s/step - loss: 1.0391 - sparse_categorical_accuracy: 0.3234

<div class="k-default-codeblock">
```

```
</div>
  41/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:18 1s/step - loss: 1.0386 - sparse_categorical_accuracy: 0.3235

<div class="k-default-codeblock">
```

```
</div>
  42/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:16 1s/step - loss: 1.0380 - sparse_categorical_accuracy: 0.3237

<div class="k-default-codeblock">
```

```
</div>
  43/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:15 1s/step - loss: 1.0372 - sparse_categorical_accuracy: 0.3238

<div class="k-default-codeblock">
```

```
</div>
  44/100 ━━━━━━━━[37m━━━━━━━━━━━━  1:13 1s/step - loss: 1.0364 - sparse_categorical_accuracy: 0.3240

<div class="k-default-codeblock">
```

```
</div>
  45/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:12 1s/step - loss: 1.0358 - sparse_categorical_accuracy: 0.3241

<div class="k-default-codeblock">
```

```
</div>
  46/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:10 1s/step - loss: 1.0352 - sparse_categorical_accuracy: 0.3242

<div class="k-default-codeblock">
```

```
</div>
  47/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:09 1s/step - loss: 1.0347 - sparse_categorical_accuracy: 0.3243

<div class="k-default-codeblock">
```

```
</div>
  48/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:07 1s/step - loss: 1.0343 - sparse_categorical_accuracy: 0.3244

<div class="k-default-codeblock">
```

```
</div>
  49/100 ━━━━━━━━━[37m━━━━━━━━━━━  1:06 1s/step - loss: 1.0339 - sparse_categorical_accuracy: 0.3245

<div class="k-default-codeblock">
```

```
</div>
  50/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:04 1s/step - loss: 1.0336 - sparse_categorical_accuracy: 0.3246

<div class="k-default-codeblock">
```

```
</div>
  51/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:03 1s/step - loss: 1.0333 - sparse_categorical_accuracy: 0.3247

<div class="k-default-codeblock">
```

```
</div>
  52/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:02 1s/step - loss: 1.0331 - sparse_categorical_accuracy: 0.3247

<div class="k-default-codeblock">
```

```
</div>
  53/100 ━━━━━━━━━━[37m━━━━━━━━━━  1:00 1s/step - loss: 1.0329 - sparse_categorical_accuracy: 0.3248

<div class="k-default-codeblock">
```

```
</div>
  54/100 ━━━━━━━━━━[37m━━━━━━━━━━  59s 1s/step - loss: 1.0325 - sparse_categorical_accuracy: 0.3249 

<div class="k-default-codeblock">
```

```
</div>
  55/100 ━━━━━━━━━━━[37m━━━━━━━━━  57s 1s/step - loss: 1.0322 - sparse_categorical_accuracy: 0.3249

<div class="k-default-codeblock">
```

```
</div>
  56/100 ━━━━━━━━━━━[37m━━━━━━━━━  56s 1s/step - loss: 1.0319 - sparse_categorical_accuracy: 0.3250

<div class="k-default-codeblock">
```

```
</div>
  57/100 ━━━━━━━━━━━[37m━━━━━━━━━  55s 1s/step - loss: 1.0315 - sparse_categorical_accuracy: 0.3251

<div class="k-default-codeblock">
```

```
</div>
  58/100 ━━━━━━━━━━━[37m━━━━━━━━━  53s 1s/step - loss: 1.0311 - sparse_categorical_accuracy: 0.3251

<div class="k-default-codeblock">
```

```
</div>
  59/100 ━━━━━━━━━━━[37m━━━━━━━━━  52s 1s/step - loss: 1.0306 - sparse_categorical_accuracy: 0.3252

<div class="k-default-codeblock">
```

```
</div>
  60/100 ━━━━━━━━━━━━[37m━━━━━━━━  51s 1s/step - loss: 1.0303 - sparse_categorical_accuracy: 0.3252

<div class="k-default-codeblock">
```

```
</div>
  61/100 ━━━━━━━━━━━━[37m━━━━━━━━  49s 1s/step - loss: 1.0299 - sparse_categorical_accuracy: 0.3253

<div class="k-default-codeblock">
```

```
</div>
  62/100 ━━━━━━━━━━━━[37m━━━━━━━━  48s 1s/step - loss: 1.0294 - sparse_categorical_accuracy: 0.3253

<div class="k-default-codeblock">
```

```
</div>
  63/100 ━━━━━━━━━━━━[37m━━━━━━━━  46s 1s/step - loss: 1.0289 - sparse_categorical_accuracy: 0.3254

<div class="k-default-codeblock">
```

```
</div>
  64/100 ━━━━━━━━━━━━[37m━━━━━━━━  45s 1s/step - loss: 1.0284 - sparse_categorical_accuracy: 0.3255

<div class="k-default-codeblock">
```

```
</div>
  65/100 ━━━━━━━━━━━━━[37m━━━━━━━  44s 1s/step - loss: 1.0278 - sparse_categorical_accuracy: 0.3255

<div class="k-default-codeblock">
```

```
</div>
  66/100 ━━━━━━━━━━━━━[37m━━━━━━━  42s 1s/step - loss: 1.0273 - sparse_categorical_accuracy: 0.3256

<div class="k-default-codeblock">
```

```
</div>
  67/100 ━━━━━━━━━━━━━[37m━━━━━━━  41s 1s/step - loss: 1.0268 - sparse_categorical_accuracy: 0.3256

<div class="k-default-codeblock">
```

```
</div>
  68/100 ━━━━━━━━━━━━━[37m━━━━━━━  40s 1s/step - loss: 1.0264 - sparse_categorical_accuracy: 0.3257

<div class="k-default-codeblock">
```

```
</div>
  69/100 ━━━━━━━━━━━━━[37m━━━━━━━  38s 1s/step - loss: 1.0259 - sparse_categorical_accuracy: 0.3258

<div class="k-default-codeblock">
```

```
</div>
  70/100 ━━━━━━━━━━━━━━[37m━━━━━━  37s 1s/step - loss: 1.0254 - sparse_categorical_accuracy: 0.3259

<div class="k-default-codeblock">
```

```
</div>
  71/100 ━━━━━━━━━━━━━━[37m━━━━━━  36s 1s/step - loss: 1.0251 - sparse_categorical_accuracy: 0.3259

<div class="k-default-codeblock">
```

```
</div>
  72/100 ━━━━━━━━━━━━━━[37m━━━━━━  35s 1s/step - loss: 1.0247 - sparse_categorical_accuracy: 0.3260

<div class="k-default-codeblock">
```

```
</div>
  73/100 ━━━━━━━━━━━━━━[37m━━━━━━  33s 1s/step - loss: 1.0243 - sparse_categorical_accuracy: 0.3261

<div class="k-default-codeblock">
```

```
</div>
  74/100 ━━━━━━━━━━━━━━[37m━━━━━━  32s 1s/step - loss: 1.0239 - sparse_categorical_accuracy: 0.3261

<div class="k-default-codeblock">
```

```
</div>
  75/100 ━━━━━━━━━━━━━━━[37m━━━━━  31s 1s/step - loss: 1.0237 - sparse_categorical_accuracy: 0.3262

<div class="k-default-codeblock">
```

```
</div>
  76/100 ━━━━━━━━━━━━━━━[37m━━━━━  29s 1s/step - loss: 1.0235 - sparse_categorical_accuracy: 0.3262

<div class="k-default-codeblock">
```

```
</div>
  77/100 ━━━━━━━━━━━━━━━[37m━━━━━  28s 1s/step - loss: 1.0233 - sparse_categorical_accuracy: 0.3263

<div class="k-default-codeblock">
```

```
</div>
  78/100 ━━━━━━━━━━━━━━━[37m━━━━━  27s 1s/step - loss: 1.0231 - sparse_categorical_accuracy: 0.3263

<div class="k-default-codeblock">
```

```
</div>
  79/100 ━━━━━━━━━━━━━━━[37m━━━━━  26s 1s/step - loss: 1.0228 - sparse_categorical_accuracy: 0.3263

<div class="k-default-codeblock">
```

```
</div>
  80/100 ━━━━━━━━━━━━━━━━[37m━━━━  24s 1s/step - loss: 1.0225 - sparse_categorical_accuracy: 0.3264

<div class="k-default-codeblock">
```

```
</div>
  81/100 ━━━━━━━━━━━━━━━━[37m━━━━  23s 1s/step - loss: 1.0223 - sparse_categorical_accuracy: 0.3264

<div class="k-default-codeblock">
```

```
</div>
  82/100 ━━━━━━━━━━━━━━━━[37m━━━━  22s 1s/step - loss: 1.0220 - sparse_categorical_accuracy: 0.3264

<div class="k-default-codeblock">
```

```
</div>
  83/100 ━━━━━━━━━━━━━━━━[37m━━━━  21s 1s/step - loss: 1.0218 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  84/100 ━━━━━━━━━━━━━━━━[37m━━━━  19s 1s/step - loss: 1.0215 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  85/100 ━━━━━━━━━━━━━━━━━[37m━━━  18s 1s/step - loss: 1.0214 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  86/100 ━━━━━━━━━━━━━━━━━[37m━━━  17s 1s/step - loss: 1.0212 - sparse_categorical_accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
  87/100 ━━━━━━━━━━━━━━━━━[37m━━━  16s 1s/step - loss: 1.0210 - sparse_categorical_accuracy: 0.3266

<div class="k-default-codeblock">
```

```
</div>
  88/100 ━━━━━━━━━━━━━━━━━[37m━━━  14s 1s/step - loss: 1.0208 - sparse_categorical_accuracy: 0.3266

<div class="k-default-codeblock">
```

```
</div>
  89/100 ━━━━━━━━━━━━━━━━━[37m━━━  13s 1s/step - loss: 1.0206 - sparse_categorical_accuracy: 0.3266

<div class="k-default-codeblock">
```

```
</div>
  90/100 ━━━━━━━━━━━━━━━━━━[37m━━  12s 1s/step - loss: 1.0204 - sparse_categorical_accuracy: 0.3267

<div class="k-default-codeblock">
```

```
</div>
  91/100 ━━━━━━━━━━━━━━━━━━[37m━━  11s 1s/step - loss: 1.0203 - sparse_categorical_accuracy: 0.3267

<div class="k-default-codeblock">
```

```
</div>
  92/100 ━━━━━━━━━━━━━━━━━━[37m━━  9s 1s/step - loss: 1.0202 - sparse_categorical_accuracy: 0.3267 

<div class="k-default-codeblock">
```

```
</div>
  93/100 ━━━━━━━━━━━━━━━━━━[37m━━  8s 1s/step - loss: 1.0201 - sparse_categorical_accuracy: 0.3268

<div class="k-default-codeblock">
```

```
</div>
  94/100 ━━━━━━━━━━━━━━━━━━[37m━━  7s 1s/step - loss: 1.0201 - sparse_categorical_accuracy: 0.3268

<div class="k-default-codeblock">
```

```
</div>
  95/100 ━━━━━━━━━━━━━━━━━━━[37m━  6s 1s/step - loss: 1.0200 - sparse_categorical_accuracy: 0.3268

<div class="k-default-codeblock">
```

```
</div>
  96/100 ━━━━━━━━━━━━━━━━━━━[37m━  4s 1s/step - loss: 1.0199 - sparse_categorical_accuracy: 0.3269

<div class="k-default-codeblock">
```

```
</div>
  97/100 ━━━━━━━━━━━━━━━━━━━[37m━  3s 1s/step - loss: 1.0199 - sparse_categorical_accuracy: 0.3269

<div class="k-default-codeblock">
```

```
</div>
  98/100 ━━━━━━━━━━━━━━━━━━━[37m━  2s 1s/step - loss: 1.0199 - sparse_categorical_accuracy: 0.3269

<div class="k-default-codeblock">
```

```
</div>
  99/100 ━━━━━━━━━━━━━━━━━━━[37m━  1s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3270

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3270

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 151s 1s/step - loss: 1.0198 - sparse_categorical_accuracy: 0.3271





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7fb15e164640>

```
</div>
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

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 834/834 [00:00<00:00, 1.41kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (834B)

Starting upload for file tokenizer.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/322 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 322/322 [00:00<00:00, 356B/s]

    
<div class="k-default-codeblock">
```
Upload successful: tokenizer.json (322B)

Starting upload for file task.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/1.82k [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.82k/1.82k [00:00<00:00, 3.13kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.json (2KB)

Starting upload for file model.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/498M [00:00<?, ?B/s]

    
Uploading:   2%|███▋                                                                                                                                                                                                | 9.45M/498M [00:00<00:09, 53.5MB/s]

    
Uploading:   6%|███████████▎                                                                                                                                                                                         | 28.7M/498M [00:00<00:04, 100MB/s]

    
Uploading:   8%|███████████████▍                                                                                                                                                                                    | 39.3M/498M [00:00<00:04, 98.4MB/s]

    
Uploading:  10%|███████████████████▍                                                                                                                                                                                | 49.4M/498M [00:00<00:04, 91.6MB/s]

    
Uploading:  13%|█████████████████████████▍                                                                                                                                                                          | 64.5M/498M [00:00<00:04, 96.3MB/s]

    
Uploading:  16%|██████████████████████████████▊                                                                                                                                                                      | 77.9M/498M [00:00<00:03, 107MB/s]

    
Uploading:  18%|███████████████████████████████████▎                                                                                                                                                                | 89.9M/498M [00:00<00:04, 99.5MB/s]

    
Uploading:  21%|█████████████████████████████████████████▌                                                                                                                                                            | 105M/498M [00:01<00:03, 112MB/s]

    
Uploading:  23%|██████████████████████████████████████████████▏                                                                                                                                                       | 116M/498M [00:01<00:03, 100MB/s]

    
Uploading:  26%|████████████████████████████████████████████████████▍                                                                                                                                                 | 132M/498M [00:01<00:03, 102MB/s]

    
Uploading:  29%|██████████████████████████████████████████████████████████▎                                                                                                                                           | 147M/498M [00:01<00:03, 101MB/s]

    
Uploading:  33%|████████████████████████████████████████████████████████████████▍                                                                                                                                     | 162M/498M [00:01<00:02, 114MB/s]

    
Uploading:  35%|█████████████████████████████████████████████████████████████████████▏                                                                                                                                | 174M/498M [00:01<00:03, 105MB/s]

    
Uploading:  38%|███████████████████████████████████████████████████████████████████████████                                                                                                                           | 189M/498M [00:01<00:02, 103MB/s]

    
Uploading:  41%|████████████████████████████████████████████████████████████████████████████████▊                                                                                                                     | 203M/498M [00:02<00:02, 102MB/s]

    
Uploading:  44%|██████████████████████████████████████████████████████████████████████████████████████▏                                                                                                               | 217M/498M [00:02<00:02, 110MB/s]

    
Uploading:  46%|███████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                          | 230M/498M [00:02<00:02, 116MB/s]

    
Uploading:  49%|████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                     | 242M/498M [00:02<00:02, 111MB/s]

    
Uploading:  51%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                 | 254M/498M [00:02<00:02, 103MB/s]

    
Uploading:  54%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                          | 270M/498M [00:02<00:02, 106MB/s]

    
Uploading:  57%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                     | 283M/498M [00:02<00:01, 111MB/s]

    
Uploading:  59%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                | 295M/498M [00:02<00:02, 101MB/s]

    
Uploading:  61%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                            | 306M/498M [00:02<00:01, 98.7MB/s]

    
Uploading:  65%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                     | 323M/498M [00:03<00:01, 102MB/s]

    
Uploading:  68%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                               | 340M/498M [00:03<00:01, 105MB/s]

    
Uploading:  72%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                        | 357M/498M [00:03<00:01, 107MB/s]

    
Uploading:  74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                  | 371M/498M [00:03<00:01, 115MB/s]

    
Uploading:  77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                              | 383M/498M [00:03<00:01, 108MB/s]

    
Uploading:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                       | 398M/498M [00:03<00:00, 109MB/s]

    
Uploading:  83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                 | 413M/498M [00:03<00:00, 118MB/s]

    
Uploading:  85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                             | 425M/498M [00:04<00:00, 109MB/s]

    
Uploading:  88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                       | 438M/498M [00:04<00:00, 104MB/s]

    
Uploading:  91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 455M/498M [00:04<00:00, 106MB/s]

    
Uploading:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏            | 466M/498M [00:04<00:00, 107MB/s]

    
Uploading:  96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌        | 477M/498M [00:04<00:00, 98.8MB/s]

    
Uploading:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉  | 493M/498M [00:04<00:00, 102MB/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 498M/498M [00:05<00:00, 92.1MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: model.weights.h5 (475MB)

Starting upload for file config.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/431 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 431/431 [00:00<00:00, 482B/s]

    
<div class="k-default-codeblock">
```
Upload successful: config.json (431B)

Starting upload for file metadata.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/142 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:00<00:00, 244B/s]

    
<div class="k-default-codeblock">
```
Upload successful: metadata.json (142B)

Starting upload for file merges.txt

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/456k [00:00<?, ?B/s]

    
Uploading: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:01<00:00, 366kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: merges.txt (446KB)

Starting upload for file vocabulary.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/1.04M [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 1.73MB/s]

    
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
This can in one of the following ways:
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
Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/task.json...

Downloading from https://www.kaggle.com/api/v1/models/keras/bert/keras/bert_tiny_en_uncased/2/download/preprocessor.json...

```
</div>
    
   1/100 [37m━━━━━━━━━━━━━━━━━━━━  7:04 4s/step - loss: 0.6897 - sparse_categorical_accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
   2/100 [37m━━━━━━━━━━━━━━━━━━━━  3:24 2s/step - loss: 0.6902 - sparse_categorical_accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
   6/100 ━[37m━━━━━━━━━━━━━━━━━━━  40s 429ms/step - loss: 0.6939 - sparse_categorical_accuracy: 0.6285

<div class="k-default-codeblock">
```

```
</div>
  10/100 ━━[37m━━━━━━━━━━━━━━━━━━  21s 244ms/step - loss: 0.6961 - sparse_categorical_accuracy: 0.5849

<div class="k-default-codeblock">
```

```
</div>
  14/100 ━━[37m━━━━━━━━━━━━━━━━━━  14s 173ms/step - loss: 0.6945 - sparse_categorical_accuracy: 0.5698

<div class="k-default-codeblock">
```

```
</div>
  18/100 ━━━[37m━━━━━━━━━━━━━━━━━  11s 135ms/step - loss: 0.6956 - sparse_categorical_accuracy: 0.5483

<div class="k-default-codeblock">
```

```
</div>
  22/100 ━━━━[37m━━━━━━━━━━━━━━━━  8s 112ms/step - loss: 0.6969 - sparse_categorical_accuracy: 0.5302 

<div class="k-default-codeblock">
```

```
</div>
  27/100 ━━━━━[37m━━━━━━━━━━━━━━━  6s 92ms/step - loss: 0.6979 - sparse_categorical_accuracy: 0.5192 

<div class="k-default-codeblock">
```

```
</div>
  33/100 ━━━━━━[37m━━━━━━━━━━━━━━  5s 77ms/step - loss: 0.6976 - sparse_categorical_accuracy: 0.5201

<div class="k-default-codeblock">
```

```
</div>
  39/100 ━━━━━━━[37m━━━━━━━━━━━━━  4s 66ms/step - loss: 0.6971 - sparse_categorical_accuracy: 0.5226

<div class="k-default-codeblock">
```

```
</div>
  45/100 ━━━━━━━━━[37m━━━━━━━━━━━  3s 58ms/step - loss: 0.6967 - sparse_categorical_accuracy: 0.5234

<div class="k-default-codeblock">
```

```
</div>
  51/100 ━━━━━━━━━━[37m━━━━━━━━━━  2s 52ms/step - loss: 0.6969 - sparse_categorical_accuracy: 0.5229

<div class="k-default-codeblock">
```

```
</div>
  58/100 ━━━━━━━━━━━[37m━━━━━━━━━  1s 47ms/step - loss: 0.6970 - sparse_categorical_accuracy: 0.5226

<div class="k-default-codeblock">
```

```
</div>
  64/100 ━━━━━━━━━━━━[37m━━━━━━━━  1s 43ms/step - loss: 0.6972 - sparse_categorical_accuracy: 0.5215

<div class="k-default-codeblock">
```

```
</div>
  70/100 ━━━━━━━━━━━━━━[37m━━━━━━  1s 40ms/step - loss: 0.6974 - sparse_categorical_accuracy: 0.5196

<div class="k-default-codeblock">
```

```
</div>
  77/100 ━━━━━━━━━━━━━━━[37m━━━━━  0s 37ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5175

<div class="k-default-codeblock">
```

```
</div>
  83/100 ━━━━━━━━━━━━━━━━[37m━━━━  0s 35ms/step - loss: 0.6976 - sparse_categorical_accuracy: 0.5169

<div class="k-default-codeblock">
```

```
</div>
  90/100 ━━━━━━━━━━━━━━━━━━[37m━━  0s 33ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5166

<div class="k-default-codeblock">
```

```
</div>
  97/100 ━━━━━━━━━━━━━━━━━━━[37m━  0s 31ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5162

<div class="k-default-codeblock">
```

```
</div>
 100/100 ━━━━━━━━━━━━━━━━━━━━ 7s 31ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.5164


<div class="k-default-codeblock">
```
Starting upload for file preprocessor.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/947 [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 947/947 [00:00<00:00, 1.43kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: preprocessor.json (947B)

Starting upload for file tokenizer.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/461 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 461/461 [00:00<00:00, 739B/s]

    
<div class="k-default-codeblock">
```
Upload successful: tokenizer.json (461B)

Starting upload for file task.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/2.09k [00:00<?, ?B/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.09k/2.09k [00:00<00:00, 2.38kB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.json (2KB)

Starting upload for file task.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/52.8M [00:00<?, ?B/s]

    
Uploading:   4%|███████▋                                                                                                                                                                                           | 2.08M/52.8M [00:00<00:02, 20.8MB/s]

    
Uploading:  23%|█████████████████████████████████████████████▎                                                                                                                                                     | 12.3M/52.8M [00:00<00:00, 66.8MB/s]

    
Uploading:  56%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                      | 29.4M/52.8M [00:00<00:00, 110MB/s]

    
Uploading:  76%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                              | 40.3M/52.8M [00:00<00:00, 97.4MB/s]

    
Uploading:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏         | 50.1M/52.8M [00:00<00:00, 96.9MB/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.8M/52.8M [00:01<00:00, 45.1MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: task.weights.h5 (50MB)

Starting upload for file model.weights.h5

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                            | 0.00/17.6M [00:00<?, ?B/s]

    
Uploading:  47%|████████████████████████████████████████████████████████████████████████████████████████████                                                                                                       | 8.32M/17.6M [00:00<00:00, 48.4MB/s]

    
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17.6M/17.6M [00:01<00:00, 16.1MB/s]

    
<div class="k-default-codeblock">
```
Upload successful: model.weights.h5 (17MB)

Starting upload for file config.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/454 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [00:00<00:00, 718B/s]

    
<div class="k-default-codeblock">
```
Upload successful: config.json (454B)

Starting upload for file metadata.json

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                              | 0.00/140 [00:00<?, ?B/s]

    
Uploading: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140/140 [00:01<00:00, 114B/s]

    
<div class="k-default-codeblock">
```
Upload successful: metadata.json (140B)

Starting upload for file vocabulary.txt

```
</div>
    
Uploading:   0%|                                                                                                                                                                                                             | 0.00/232k [00:00<?, ?B/s]

    
Uploading: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:01<00:00, 183kB/s]

    
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
