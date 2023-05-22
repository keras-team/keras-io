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
## Install KerasNLP and Import Dependencies


```python
!pip install -q keras-nlp
```


```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras
import time
```

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
WARNING:tensorflow:The following Variables were used in a Lambda layer's call (tf.linalg.matmul), but are not present in its tracked objects:   <tf.Variable 'token_embedding/embeddings:0' shape=(50257, 768) dtype=float32>. This is a strong indication that the Lambda layer should be rewritten as a subclassed Layer.

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
My trip to Yosemite was one of the best experiences of my life. I was so close to the top of the mountains, I could feel the sun shining through my eyes. I was so close to the top of the mountains, the sun had a nice view of the valley and I couldn't believe the sun came out of nowhere. The sun shone in all directions and I could feel it. I was so close to the top of the mountains, it felt like I was in the middle of a volcano. It was amazing to see all of that. I felt like a volcano. I felt so close to all of the things. I felt like an island in a sea of lava.
```
</div>
    
<div class="k-default-codeblock">
```
I didn't know what I was doing. I just thought I was going to get out of here and go home and see my family. I thought that I could go home and see my parents and I was just happy that I was here. I was so happy that I was here.
```
</div>
    
<div class="k-default-codeblock">
```
TOTAL TIME ELAPSED: 18.25s

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
That Italian restaurant is now closed, according to a report from Bloomberg.
```
</div>
    
<div class="k-default-codeblock">
```
The eatery is located at 5100 N. Broadway in New York City, according to the New York Post. The restaurant is open from 11 a.m. to 4 p.m.
```
</div>
    
<div class="k-default-codeblock">
```
"The owner is very upset and we are trying to find out what happened to our place," an Italian restaurant employee said in an email to the Post. "He said he is going to close the restaurant. We are not going to let him get away from us."
```
</div>
    
<div class="k-default-codeblock">
```
"I don't know what the problem is but it's sad and it makes me feel like I have to go," the Italian owner told the Post.
```
</div>
    
<div class="k-default-codeblock">
```
The restaurant, which has a large Italian menu, was closed in April after the owner, who is Italian, told the Post that the restaurant was "not a good place," but that he was "working on a new restaurant."
TOTAL TIME ELAPSED: 1.71s

```
</div>
Notice how much faster the second call is. This is because the computational
graph is [XLA compiled](https://www.tensorflow.org/xla) in the 1st run and
re-used in the 2nd behind the scenes.

The quality of the generated text looks OK, but we can improved it via
finetuning.

---
## More on the GPT-2 model from KerasNLP

Next up, we will actually fine-tune the model to update it's parameters, but
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
500/500 [==============================] - 212s 311ms/step - loss: 3.3057 - accuracy: 0.3265

<keras.callbacks.History at 0x7f3fead74b20>

```
</div>
After finetuning is finished, you can again generate text using the same
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
I like basketball. i've been a big fan of it since high school, and it's been pretty cool to me.
```
</div>
    
<div class="k-default-codeblock">
```
i've been playing basketball with my brother since high school, and my dad is a big fan of the game.
```
</div>
    
<div class="k-default-codeblock">
```
so, i'm in the middle of a game, and i get a little frustrated, so i just try to play basketball. so, i start to go up on the court, and when i see the basket, i'm like, "what the hell, this kid has to go!" so i start to get up and go up on the floor.
```
</div>
    
<div class="k-default-codeblock">
```
it's like a giant
TOTAL TIME ELAPSED: 16.22s

```
</div>
---
## Into the Sampling Method

In KerasNLP, we offer a few sampling methods, e.g., contrastive search,
Top-K and beam sampling. By default our `GPT2CausalLM` uses Top-k search, but
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
I like basketball and it's a good sport to have.
```
</div>
    
<div class="k-default-codeblock">
```
i was playing basketball in my hometown of texas and was playing a little bit of a game.
```
</div>
    
<div class="k-default-codeblock">
```
my team had won the title and i was just getting ready to go out to the court. 
```
</div>
    
<div class="k-default-codeblock">
```
i was sitting there watching my team play and was about to jump on the court, when the ball came in the other direction.
```
</div>
    
<div class="k-default-codeblock">
```
so my buddy was standing right behind me and was looking at me with a look of surprise on his face.
```
</div>
    
<div class="k-default-codeblock">
```
so he looked up at me and said "hey guys, you're going to have to play the next game, right?"
```
</div>
    
<div class="k-default-codeblock">
```
so i was like "yeah, i know, i guess i'll go." 
so i
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
For more details on KerasNLP `Sampler` class, you can check the code
[here](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/samplers).

---
## Finetune on Chinese Poem Dataset

We can also finetune GPT2 on non-English datasets. For readers knowing Chinese,
this part illustrates how to finetung GPT2 on Chinese poem dataset to teach our
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
remote: Enumerating objects: 7222, done.[K
remote: Counting objects: 100% (27/27), done.[K
remote: Compressing objects: 100% (19/19), done.[K
remote: Total 7222 (delta 5), reused 20 (delta 5), pack-reused 7195[K
Receiving objects: 100% (7222/7222), 197.75 MiB | 15.15 MiB/s, done.
Resolving deltas: 100% (5295/5295), done.
Checking out files: 100% (2283/2283), done.

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
十月十六誰宗旨，無聲三昧重拈起。十方三世側耳聽，刹刹塵塵俱解義。雙林樹開榮枯枝，寶塔佛分生滅理。一絲不挂露堂堂，要識雲菴今日是。

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
500/500 [==============================] - 160s 211ms/step - loss: 2.4432 - accuracy: 0.2811

<keras.callbacks.History at 0x7f40802d8e20>

```
</div>
Let's check the result!


```python
output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:5 out of the last 6 calls to <bound method GPT2CausalLM.generate_step of <keras_nlp.models.gpt2.gpt2_causal_lm.GPT2CausalLM object at 0x7f40804406a0>> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:5 out of the last 6 calls to <bound method GPT2CausalLM.generate_step of <keras_nlp.models.gpt2.gpt2_causal_lm.GPT2CausalLM object at 0x7f40804406a0>> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

昨夜雨疏风骤紛，白萬聞城清山。江清知非頭自花，爲花萬屋清清除。白處樂深紅花，清處馬自自自。

```
</div>
Not bad 😀
