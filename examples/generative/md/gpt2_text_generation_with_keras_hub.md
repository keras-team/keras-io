# GPT2 Text Generation with KerasHub

**Author:** Chen Qian<br>
**Date created:** 2023/04/17<br>
**Last modified:** 2024/04/12<br>
**Description:** Use KerasHub GPT2 model and `samplers` to do text generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/gpt2_text_generation_with_kerashub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/gpt2_text_generation_with_kerashub.py)



In this tutorial, you will learn to use [KerasHub](https://keras.io/keras_hub/) to load a
pre-trained Large Language Model (LLM) - [GPT-2 model](https://openai.com/research/better-language-models)
(originally invented by OpenAI), finetune it to a specific text style, and
generate text based on users' input (also known as prompt). You will also learn
how GPT2 adapts quickly to non-English languages, such as Chinese.

---
##  Before we begin

Colab offers different kinds of runtimes. Make sure to go to **Runtime ->
Change runtime type** and choose the GPU Hardware Accelerator runtime
(which should have >12G host RAM and ~15G GPU RAM) since you will finetune the
GPT-2 model. Running this tutorial on CPU runtime will take hours.

---
## Install KerasHub, Choose Backend and Import Dependencies

This examples uses [Keras 3](https://keras.io/keras_3/) to work in any of
`"tensorflow"`, `"jax"` or `"torch"`. Support for Keras 3 is baked into
KerasHub, simply change the `"KERAS_BACKEND"` environment variable to select
the backend of your choice. We select the JAX backend below.


```python
!pip install git+https://github.com/keras-team/keras-hub.git -q
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras_hub
import keras
import tensorflow as tf
import time

keras.mixed_precision.set_global_policy("mixed_float16")
```
<div class="k-default-codeblock">
```

```
</div>
---
## Introduction to Generative Large Language Models (LLMs)

Large language models (LLMs) are a type of machine learning models that are
trained on a large corpus of text data to generate outputs for various natural
language processing (NLP) tasks, such as text generation, question answering,
and machine translation.

Generative LLMs are typically based on deep learning neural networks, such as
the [Transformer architecture](https://arxiv.org/abs/1706.03762) invented by
Google researchers in 2017, and are trained on massive amounts of text data,
often involving billions of words. These models, such as Google [LaMDA](https://blog.google/technology/ai/lamda/)
and [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html),
are trained with a large dataset from various data sources which allows them to
generate output for many tasks. The core of Generative LLMs is predicting the
next word in a sentence, often referred as **Causal LM Pretraining**. In this
way LLMs can generate coherent text based on user prompts. For a more
pedagogical discussion on language models, you can refer to the
[Stanford CS324 LLM class](https://stanford-cs324.github.io/winter2022/lectures/introduction/).

---
## Introduction to KerasHub

Large Language Models are complex to build and expensive to train from scratch.
Luckily there are pretrained LLMs available for use right away. [KerasHub](https://keras.io/keras_hub/)
provides a large number of pre-trained checkpoints that allow you to experiment
with SOTA models without needing to train them yourself.

KerasHub is a natural language processing library that supports users through
their entire development cycle. KerasHub offers both pretrained models and
modularized building blocks, so developers could easily reuse pretrained models
or stack their own LLM.

In a nutshell, for generative LLM, KerasHub offers:

- Pretrained models with `generate()` method, e.g.,
    `keras_hub.models.GPT2CausalLM` and `keras_hub.models.OPTCausalLM`.
- Sampler class that implements generation algorithms such as Top-K, Beam and
    contrastive search. These samplers can be used to generate text with
    custom models.

---
## Load a pre-trained GPT-2 model and generate some text

KerasHub provides a number of pre-trained models, such as [Google
Bert](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
and [GPT-2](https://openai.com/research/better-language-models). You can see
the list of models available in the [KerasHub repository](https://github.com/keras-team/keras-hub/tree/master/keras_hub/models).

It's very easy to load the GPT-2 model as you can see below:


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)
```

Once the model is loaded, you can use it to generate some text right away. Run
the cells below to give it a try. It's as simple as calling a single function
*generate()*:


```python
start = time.time()

output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
<div class="k-default-codeblock">
```
GPT-2 output:
My trip to Yosemite was pretty awesome. The first time I went I didn't know how to go and it was pretty hard to get around. It was a bit like going on an adventure with a friend. The only things I could do were hike and climb the mountain. It's really cool to know you're not alone in this world. It's a lot of fun. I'm a little worried that I might not get to the top of the mountain in time to see the sunrise and sunset of the day. I think the weather is going to get a little warmer in the coming years.
```
</div>
    
<div class="k-default-codeblock">
```
This post is a little more in-depth on how to go on the trail. It covers how to hike on the Sierra Nevada, how to hike with the Sierra Nevada, how to hike in the Sierra Nevada, how to get to the top of the mountain, and how to get to the top with your own gear.
```
</div>
    
<div class="k-default-codeblock">
```
The Sierra Nevada is a very popular trail in Yosemite
TOTAL TIME ELAPSED: 25.36s

```
</div>
Try another one:


```python
start = time.time()

output = gpt2_lm.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
<div class="k-default-codeblock">
```
GPT-2 output:
That Italian restaurant is known for its delicious food, and the best part is that it has a full bar, with seating for a whole host of guests. And that's only because it's located at the heart of the neighborhood.
```
</div>
    
<div class="k-default-codeblock">
```
The menu at the Italian restaurant is pretty straightforward:
```
</div>
    
<div class="k-default-codeblock">
```
The menu consists of three main dishes:
```
</div>
    
<div class="k-default-codeblock">
```
Italian sausage
```
</div>
    
<div class="k-default-codeblock">
```
Bolognese
```
</div>
    
<div class="k-default-codeblock">
```
Sausage
```
</div>
    
<div class="k-default-codeblock">
```
Bolognese with cheese
```
</div>
    
<div class="k-default-codeblock">
```
Sauce with cream
```
</div>
    
<div class="k-default-codeblock">
```
Italian sausage with cheese
```
</div>
    
<div class="k-default-codeblock">
```
Bolognese with cheese
```
</div>
    
<div class="k-default-codeblock">
```
And the main menu consists of a few other things.
```
</div>
    
<div class="k-default-codeblock">
```
There are two tables: the one that serves a menu of sausage and bolognese with cheese (the one that serves the menu of sausage and bolognese with cheese) and the one that serves the menu of sausage and bolognese with cheese. The two tables are also open 24 hours a day, 7 days a week.
```
</div>
    
    
<div class="k-default-codeblock">
```
TOTAL TIME ELAPSED: 1.55s

```
</div>
Notice how much faster the second call is. This is because the computational
graph is [XLA compiled](https://www.tensorflow.org/xla) in the 1st run and
re-used in the 2nd behind the scenes.

The quality of the generated text looks OK, but we can improve it via
fine-tuning.

---
## More on the GPT-2 model from KerasHub

Next up, we will actually fine-tune the model to update its parameters, but
before we do, let's take a look at the full set of tools we have to for working
with for GPT2.

The code of GPT2 can be found
[here](https://github.com/keras-team/keras-hub/blob/master/keras_hub/models/gpt2/).
Conceptually the `GPT2CausalLM` can be hierarchically broken down into several
modules in KerasHub, all of which have a *from_preset()* function that loads a
pretrained model:

- `keras_hub.models.GPT2Tokenizer`: The tokenizer used by GPT2 model, which is a
    [byte-pair encoder](https://huggingface.co/course/chapter6/5?fw=pt).
- `keras_hub.models.GPT2CausalLMPreprocessor`: the preprocessor used by GPT2
    causal LM training. It does the tokenization along with other preprocessing
    works such as creating the label and appending the end token.
- `keras_hub.models.GPT2Backbone`: the GPT2 model, which is a stack of
    `keras_hub.layers.TransformerDecoder`. This is usually just referred as
    `GPT2`.
- `keras_hub.models.GPT2CausalLM`: wraps `GPT2Backbone`, it multiplies the
    output of `GPT2Backbone` by embedding matrix to generate logits over
    vocab tokens.

---
## Finetune on Reddit dataset

Now you have the knowledge of the GPT-2 model from KerasHub, you can take one
step further to finetune the model so that it generates text in a specific
style, short or long, strict or casual. In this tutorial, we will use reddit
dataset for example.


```python
import tensorflow_datasets as tfds

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
```

Let's take a look inside sample data from the reddit TensorFlow Dataset. There
are two features:

- **__document__**: text of the post.
- **__title__**: the title.


```python
for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break
```

<div class="k-default-codeblock">
```
b"me and a friend decided to go to the beach last sunday. we loaded up and headed out. we were about half way there when i decided that i was not leaving till i had seafood. \n\nnow i'm not talking about red lobster. no friends i'm talking about a low country boil. i found the restaurant and got directions. i don't know if any of you have heard about the crab shack on tybee island but let me tell you it's worth it. \n\nwe arrived and was seated quickly. we decided to get a seafood sampler for two and split it. the waitress bought it out on separate platters for us. the amount of food was staggering. two types of crab, shrimp, mussels, crawfish, andouille sausage, red potatoes, and corn on the cob. i managed to finish it and some of my friends crawfish and mussels. it was a day to be a fat ass. we finished paid for our food and headed to the beach. \n\nfunny thing about seafood. it runs through me faster than a kenyan \n\nwe arrived and walked around a bit. it was about 45min since we arrived at the beach when i felt a rumble from the depths of my stomach. i ignored it i didn't want my stomach to ruin our fun. i pushed down the feeling and continued. about 15min later the feeling was back and stronger than before. again i ignored it and continued. 5min later it felt like a nuclear reactor had just exploded in my stomach. i started running. i yelled to my friend to hurry the fuck up. \n\nrunning in sand is extremely hard if you did not know this. we got in his car and i yelled at him to floor it. my stomach was screaming and if he didn't hurry i was gonna have this baby in his car and it wasn't gonna be pretty. after a few red lights and me screaming like a woman in labor we made it to the store. \n\ni practically tore his car door open and ran inside. i ran to the bathroom opened the door and barely got my pants down before the dam burst and a flood of shit poured from my ass. \n\ni finished up when i felt something wet on my ass. i rubbed it thinking it was back splash. no, mass was covered in the after math of me abusing the toilet. i grabbed all the paper towels i could and gave my self a whores bath right there. \n\ni sprayed the bathroom down with the air freshener and left. an elderly lady walked in quickly and closed the door. i was just about to walk away when i heard gag. instead of walking i ran. i got to the car and told him to get the hell out of there."
b'liking seafood'

```
</div>
In our case, we are performing next word prediction in a language model, so we
only need the 'document' feature.


```python
train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

Now you can finetune the model using the familiar *fit()* function. Note that
`preprocessor` will be automatically called inside `fit` method since
`GPT2CausalLM` is a `keras_hub.models.Task` instance.

This step takes quite a bit of GPU memory and a long time if we were to train
it all the way to a fully trained state. Here we just use part of the dataset
for demo purposes.


```python
train_ds = train_ds.take(500)
num_epochs = 1

# Linearly decaying learning rate.
learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)
```

<div class="k-default-codeblock">
```
 500/500 ━━━━━━━━━━━━━━━━━━━━ 75s 120ms/step - accuracy: 0.3189 - loss: 3.3653

<keras.src.callbacks.history.History at 0x7f2af3fda410>

```
</div>
After fine-tuning is finished, you can again generate text using the same
*generate()* function. This time, the text will be closer to Reddit writing
style, and the generated length will be close to our preset length in the
training set.


```python
start = time.time()

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
```

    
<div class="k-default-codeblock">
```
GPT-2 output:
I like basketball. it has the greatest shot of all time and the best shot of all time. i have to play a little bit more and get some practice time.
```
</div>
    
<div class="k-default-codeblock">
```
today i got the opportunity to play in a tournament in a city that is very close to my school so i was excited to see how it would go. i had just been playing with a few other guys, so i thought i would go and play a couple games with them. 
```
</div>
    
<div class="k-default-codeblock">
```
after a few games i was pretty confident and confident in myself. i had just gotten the opportunity and had to get some practice time. 
```
</div>
    
<div class="k-default-codeblock">
```
so i go to the
TOTAL TIME ELAPSED: 21.13s

```
</div>
---
## Into the Sampling Method

In KerasHub, we offer a few sampling methods, e.g., contrastive search,
Top-K and beam sampling. By default, our `GPT2CausalLM` uses Top-k search, but
you can choose your own sampling method.

Much like optimizer and activations, there are two ways to specify your custom
sampler:

- Use a string identifier, such as "greedy", you are using the default
configuration via this way.
- Pass a `keras_hub.samplers.Sampler` instance, you can use custom configuration
via this way.


```python
# Use a string identifier.
gpt2_lm.compile(sampler="top_k")
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# Use a `Sampler` instance. `GreedySampler` tends to repeat itself,
greedy_sampler = keras_hub.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)
```

    
<div class="k-default-codeblock">
```
GPT-2 output:
I like basketball, and this is a pretty good one. 
```
</div>
    
<div class="k-default-codeblock">
```
first off, my wife is pretty good, she is a very good basketball player and she is really, really good at playing basketball. 
```
</div>
    
<div class="k-default-codeblock">
```
she has an amazing game called basketball, it is a pretty fun game. 
```
</div>
    
<div class="k-default-codeblock">
```
i play it on the couch.  i'm sitting there, watching the game on the couch.  my wife is playing with her phone.  she's playing on the phone with a bunch of people. 
```
</div>
    
<div class="k-default-codeblock">
```
my wife is sitting there and watching basketball.  she's sitting there watching
```
</div>
    
<div class="k-default-codeblock">
```
GPT-2 output:
I like basketball, but i don't like to play it. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing basketball at my local high school, and i was playing with my friends. 
```
</div>
    
<div class="k-default-codeblock">
```
i was playing with my friends, and i was playing with my brother, who was playing basketball with his brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother's brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother's brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother's brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother's brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother's brother. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing with my brother, and he was playing with his brother

```
</div>
For more details on KerasHub `Sampler` class, you can check the code
[here](https://github.com/keras-team/keras-hub/tree/master/keras_hub/samplers).

---
## Finetune on Chinese Poem Dataset

We can also finetune GPT2 on non-English datasets. For readers knowing Chinese,
this part illustrates how to fine-tune GPT2 on Chinese poem dataset to teach our
model to become a poet!

Because GPT2 uses byte-pair encoder, and the original pretraining dataset
contains some Chinese characters, we can use the original vocab to finetune on
Chinese dataset.


```python
!# Load chinese poetry dataset.
!git clone https://github.com/chinese-poetry/chinese-poetry.git
```

<div class="k-default-codeblock">
```
Cloning into 'chinese-poetry'...

```
</div>
Load text from the json file. We only use《全唐诗》for demo purposes.


```python
import os
import json

poem_collection = []
for file in os.listdir("chinese-poetry/全唐诗"):
    if ".json" not in file or "poet" not in file:
        continue
    full_filename = "%s/%s" % ("chinese-poetry/全唐诗", file)
    with open(full_filename, "r") as f:
        content = json.load(f)
        poem_collection.extend(content)

paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]
```

Let's take a look at sample data.


```python
print(paragraphs[0])
```

<div class="k-default-codeblock">
```
毋謂支山險，此山能幾何。崎嶔十年夢，知歷幾蹉跎。

```
</div>
Similar as Reddit example, we convert to TF dataset, and only use partial data
to train.


```python
train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Running through the whole dataset takes long, only take `500` and run 1
# epochs for demo purposes.
train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)
```

<div class="k-default-codeblock">
```
 500/500 ━━━━━━━━━━━━━━━━━━━━ 49s 71ms/step - accuracy: 0.2357 - loss: 2.8196

<keras.src.callbacks.history.History at 0x7f2b2c192bc0>

```
</div>
Let's check the result!


```python
output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)
```

<div class="k-default-codeblock">
```
昨夜雨疏风骤，爲臨江山院短靜。石淡山陵長爲羣，臨石山非處臨羣。美陪河埃聲爲羣，漏漏漏邊陵塘

```
</div>
Not bad 😀
