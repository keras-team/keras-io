# GPT2 Text Generation with KerasNLP

**Author:** Chen Qian<br>
**Date created:** 04/17/2023<br>
**Last modified:** 04/17/2023<br>
**Description:** Use KerasNLP GPT2 model and `samplers` to do text generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/gpt2_text_generation_with_kerasnlp.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/gpt2_text_generation_with_kerasnlp.py)



In this tutorial, you will learn to use [KerasNLP](https://keras.io/keras_nlp/) to load a
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
## Install KerasNLP, Choose Backend and Import Dependencies

We can choose one of "tensorflow", "jax", "torch" as the backend. Let's go
ahead with JAX in this example.


```python
!pip install -q keras-nlp
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp
import tensorflow as tf
import keras_core as keras
import time
```

<div class="k-default-codeblock">
```
Using JAX backend.

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
## Introduction to KerasNLP

Large Language Models are complex to build and expensive to train from scratch.
Luckily there are pretrained LLMs available for use right away. [KerasNLP](https://keras.io/keras_nlp/)
provides a large number of pre-trained checkpoints that allow you to experiment
with SOTA models without needing to train them yourself.

KerasNLP is a natural language processing library that supports users through
their entire development cycle. KerasNLP offers both pretrained models and
modularized building blocks, so developers could easily reuse pretrained models
or stack their own LLM.

In a nutshell, for generative LLM, KerasNLP offers:

- Pretrained models with `generate()` method, e.g.,
    `keras_nlp.models.GPT2CausalLM` and `keras_nlp.models.OPTCausalLM`.
- Sampler class that implements generation algorithms such as Top-K, Beam and
    contrastive search. These samplers can be used to generate text with
    custom models.

---
## Load a pre-trained GPT-2 model and generate some text

KerasNLP provides a number of pre-trained models, such as [Google
Bert](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
and [GPT-2](https://openai.com/research/better-language-models). You can see
the list of models available in the [KerasNLP repository](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models).

It's very easy to load the GPT-2 model as you can see below:


```python
# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/vocab.json
 1042301/1042301 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step       
Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/merges.txt
 456318/456318 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step       
Downloading data from https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/model.h5
 497986112/497986112 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step

```
</div>
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
My trip to Yosemite was the perfect way to start. The first day I walked through the open-air park I saw the majestic view of Yosemite from the top. The view of Yosemite was breathtaking. It was the first time I ever saw the Yosemite mountains, and the first time I've experienced the views that were so beautiful. The first time I saw the Yosemite mountains was on a Sunday morning, and I was so happy I was back at the park. I walked through the open-air park, and I saw the beautiful Yosemite mountains. I was amazed by the beauty and the beauty of this place. I also was amazed at how well it was made of granite, and how much of it was made of rock. It was a great day for me.
```
</div>
    
<div class="k-default-codeblock">
```
The second morning of my trip I walked to the park's second-floor entrance. There I met the first lady and her husband, the first lady was very nice and they had some nice conversation with me and the couple
TOTAL TIME ELAPSED: 27.49s

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
That Italian restaurant is a good place to start. It's open every day from 9:15am to 5:30pm and has a great selection of pasta dishes. The food is delicious too. The only drawback is the price.
```
</div>
    
<div class="k-default-codeblock">
```
I was craving something different from what I was used to, so I ordered the Italian pasta, the Italian cheese and the Italian sauce. It was very good.
```
</div>
    
<div class="k-default-codeblock">
```
The restaurant has a very good selection of pasta dishes and I am always looking for a great place to eat. The food is very good, the service is great, and the staff are always friendly. The service has been good and the staff is always attentive.
```
</div>
    
<div class="k-default-codeblock">
```
I've been coming here in the past few years, and I love it. It's an Italian restaurant with a great selection, but the food is very different. I'm a fan of the food, so I decided to give it a try. The food was very good. I'm sure you will like
TOTAL TIME ELAPSED: 1.97s

```
</div>
Notice how much faster the second call is. This is because the computational
graph is [XLA compiled](https://www.tensorflow.org/xla) in the 1st run and
re-used in the 2nd behind the scenes.

The quality of the generated text looks OK, but we can improve it via
fine-tuning.

---
## More on the GPT-2 model from KerasNLP

Next up, we will actually fine-tune the model to update its parameters, but
before we do, let's take a look at the full set of tools we have to for working
with for GPT2.

The code of GPT2 can be found
[here](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/gpt2/).
Conceptually the `GPT2CausalLM` can be hierarchically broken down into several
modules in KerasNLP, all of which have a *from_preset()* function that loads a
pretrained model:

- `keras_nlp.models.GPT2Tokenizer`: The tokenizer used by GPT2 model, which is a
    [byte-pair encoder](https://huggingface.co/course/chapter6/5?fw=pt).
- `keras_nlp.models.GPT2CausalLMPreprocessor`: the preprocessor used by GPT2
    causal LM training. It does the tokenization along with other preprocessing
    works such as creating the label and appending the end token.
- `keras_nlp.models.GPT2Backbone`: the GPT2 model, which is a stack of
    `keras_nlp.layers.TransformerDecoder`. This is usually just referred as
    `GPT2`.
- `keras_nlp.models.GPT2CausalLM`: wraps `GPT2Backbone`, it multiplies the
    output of `GPT2Backbone` by embedding matrix to generate logits over
    vocab tokens.

---
## Finetune on Reddit dataset

Now you have the knowledge of the GPT-2 model from KerasNLP, you can take one
step further to finetune the model so that it generates text in a specific
style, short or long, strict or casual. In this tutorial, we will use reddit
dataset for example.


```python
import tensorflow_datasets as tfds

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
```

<div class="k-default-codeblock">
```
Downloading and preparing dataset 639.54 MiB (download: 639.54 MiB, generated: 141.46 MiB, total: 781.00 MiB) to /root/tensorflow_datasets/reddit_tifu/short/1.1.2...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...:   0%|          | 0/79740 [00:00<?, ? examples/s]

Shuffling /root/tensorflow_datasets/reddit_tifu/short/1.1.2.incompleteVF6MJX/reddit_tifu-train.tfrecord*...:  …

Dataset reddit_tifu downloaded and prepared to /root/tensorflow_datasets/reddit_tifu/short/1.1.2. Subsequent calls will reuse this data.

```
</div>
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
`GPT2CausalLM` is a `keras_nlp.models.Task` instance.

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
 500/500 ━━━━━━━━━━━━━━━━━━━━ 503s 949ms/step - loss: 3.3615

<keras_core.src.callbacks.history.History at 0x7f759cf29a50>

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
I like basketball. i'm not a big fan of the sport. i like to watch it, but i'm not a big fan of the game. 
```
</div>
    
<div class="k-default-codeblock">
```
so i was playing a game of basketball with my friend, and the other guys are all playing. i was playing with my brother, and he was talking to a guy, and we're talking. the guy was a freshman, and the other guys were seniors. he was really into the game, and we were really into it. he said that he thought that the other guys were going to score points, but the other guys weren't.
```
</div>
    
<div class="k-default-codeblock">
```
so we were talking, and he said,
```
</div>
    
<div class="k-default-codeblock">
```
"you're not going to score points,
TOTAL TIME ELAPSED: 20.27s

```
</div>
---
## Into the Sampling Method

In KerasNLP, we offer a few sampling methods, e.g., contrastive search,
Top-K and beam sampling. By default, our `GPT2CausalLM` uses Top-k search, but
you can choose your own sampling method.

Much like optimizer and activations, there are two ways to specify your custom
sampler:

- Use a string identifier, such as "greedy", you are using the default
configuration via this way.
- Pass a `keras_nlp.samplers.Sampler` instance, you can use custom configuration
via this way.


```python
# Use a string identifier.
gpt2_lm.compile(sampler="top_k")
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# Use a `Sampler` instance. `GreedySampler` tends to repeat itself,
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)
```

    
<div class="k-default-codeblock">
```
GPT-2 output:
I like basketball, and i'm a fan too.
```
</div>
    
<div class="k-default-codeblock">
```
so my friend and i were sitting at his place, watching the game. i was in the midst of my favorite game of the year and he said to me "hey, how about this?   
```
</div>
    
<div class="k-default-codeblock">
```
"i'll give you one of these, and it will make you happy."
```
</div>
    
<div class="k-default-codeblock">
```
he was right.
```
</div>
    
<div class="k-default-codeblock">
```
i was like "oh no!    
```
</div>
    
<div class="k-default-codeblock">
```
"oh my god.           
```
</div>
    
<div class="k-default-codeblock">
```
"oh no.              
```
</div>
    
<div class="k-default-codeblock">
```
"you can't make me happy."
```
</div>
    
<div class="k-default-codeblock">
```
he was right.
```
</div>
    
<div class="k-default-codeblock">
```
GPT-2 output:
I like basketball, but i don't really like the game. 
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
For more details on KerasNLP `Sampler` class, you can check the code
[here](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/samplers).

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
remote: Enumerating objects: 7249, done.[K
remote: Counting objects: 100% (54/54), done.[K
remote: Compressing objects: 100% (40/40), done.[K
remote: Total 7249 (delta 15), reused 40 (delta 11), pack-reused 7195[K
Receiving objects: 100% (7249/7249), 197.90 MiB | 22.87 MiB/s, done.
Resolving deltas: 100% (5303/5303), done.
Updating files: 100% (2285/2285), done.

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
喝不開兮鑿不開，春風幾度長莓苔。趙州拄杖雖粗糲，未必親曾靠著來。

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
 500/500 ━━━━━━━━━━━━━━━━━━━━ 252s 451ms/step - loss: 2.4777

<keras_core.src.callbacks.history.History at 0x7f75bc9441f0>

```
</div>
Let's check the result!


```python
output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)
```

<div class="k-default-codeblock">
```
昨夜雨疏风骤時。臨景處臨花，處江知青自。空聽聽聞色，草馬聲聽清自。曾欲面風深，面啊詩知。積樂知

```
</div>
Not bad 😀
