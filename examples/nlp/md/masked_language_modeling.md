# End-to-end Masked Language Modeling with BERT

**Author:** [Ankur Singh](https://twitter.com/ankur310794)<br>
**Date created:** 2020/09/18<br>
**Last modified:** 2024/03/15<br>
**Description:** Implement a Masked Language Model (MLM) with BERT and fine-tune it on the IMDB Reviews dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/masked_language_modeling.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/masked_language_modeling.py)



---
## Introduction

Masked Language Modeling is a fill-in-the-blank task,
where a model uses the context words surrounding a mask token to try to predict what the
masked word should be.

For an input that contains one or more mask tokens,
the model will generate the most likely substitution for each.

Example:

- Input: "I have watched this [MASK] and it was awesome."
- Output: "I have watched this movie and it was awesome."

Masked language modeling is a great way to train a language
model in a self-supervised setting (without human-annotated labels).
Such a model can then be fine-tuned to accomplish various supervised
NLP tasks.

This example teaches you how to build a BERT model from scratch,
train it with the masked language modeling task,
and then fine-tune this model on a sentiment classification task.

We will use the Keras `TextVectorization` and `MultiHeadAttention` layers
to create a BERT Transformer-Encoder network architecture.

Note: This example should be run with `tf-nightly`.

---
## Setup

Install `tf-nightly` via `pip install tf-nightly`.


```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import keras_hub

import keras
from keras import layers
from keras.layers import TextVectorization

from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
```

---
## Set-up Configuration


```python

@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1


config = Config()
```

---
## Load the data

We will first download the IMDB data and load into a Pandas dataframe.


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
```
</div>
    
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

    
  0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0

    
  0 80.2M    0 16384    0     0   8170      0  2:51:36  0:00:02  2:51:34  8171

    
  0 80.2M    0 32768    0     0  11611      0  2:00:45  0:00:02  2:00:43 11611

    
  0 80.2M    0  144k    0     0  36325      0  0:38:35  0:00:04  0:38:31 36328

    
  0 80.2M    0  240k    0     0  50462      0  0:27:47  0:00:04  0:27:43 50464

    
  0 80.2M    0  464k    0     0  81947      0  0:17:06  0:00:05  0:17:01  100k

    
  0 80.2M    0  672k    

<div class="k-default-codeblock">
```
0     0  97992      0  0:14:18  0:00:07  0:14:11  130k

```
</div>
    
  1 80.2M    1  912k    0     0   116k      0  0:11:42  0:00:07  0:11:35  176k

    
  1 80.2M    1 1104k    0     0   123k      0  0:11:07  0:00:08  0:10:59  195k

    
  1 80.2M    1 1424k    0     0   145k      0  0:09:25  0:00:09  0:09:16  240k

    
  2 80.2M    2 1696k    0     0   153k      0  0:08:53  0:00:11  0:08:42  236k

    
  2 80.2M    2 2016k    0     0   170k      0  0:08:02  0:00:11  0:07:51  279k

    
  3 80.2M    3 2544k    0     0   194k      0  0:07:01  0:00:13  0:06:48  310k

    
  3 80.2M    3 2816k    0     0   202k      0  0:06:45  0:00:13  0:06:32  348k

    
  3 80.2M    3 3264k    0     0   215k      0  0:06:20  0:00:15  0:06:05  346k

    
  4 80.2M    4 3632k    0     0   227k      0  0:06:00  0:00:15  0:05:45  393k

    
  5 80.2M    5 4128k    0     0   240k      0  0:05:40  0:00:17  0:05:23  398k

    
  5 80.2M    5 4384k    0     0   243k      0  0:05:36  0:00:17  0:05:19  374k

    
  5 80.2M    5 4832k    0     0   252k      0  0:05:24  0:00:19  0:05:05  386k

    
  6 80.2M    6 5152k    0     0   258k      0  0:05:17  0:00:19  0:04:58  392k

    
  6 80.2M    6 5632k    0     0   267k      0  0:05:07  0:00:21  0:04:46  389k

    
  7 80.2M    7 5952k    0     0   272k      0  0:05:01  0:00:21  0:04:40  385k

    
  7 80.2M    7 6432k    0     0   278k      0  0:04:55  0:00:23  0:04:32  400k

    
  8 80.2M    8 6768k    0     0   284k      0  0:04:49  0:00:23  0:04:26  411k

    
  9 80.2M    9 7408k    0     0   298k      0  0:04:34  0:00:24  0:04:10  462k

    
  9 80.2M    9 7952k    0     0   308k      0  0:04:26  0:00:25  0:04:01  489k

    
 10 80.2M   10 8896k    0     0   331k      0  0:04:07  0:00:26  0:03:41  597k

    
 11 80.2M   11 9392k    0     0   337k      0  0:04:03  0:00:27  0:03:36  625k

    
 12 80.2M   12 10.0M    0     0   355k      0  0:03:51  0:00:28  0:03:23  696k

    
 13 80.2M   13 10.7M    0     0   368k      0  0:03:42  0:00:29  0:03:13  705k

    
 14 80.2M   14 11.4M    0     0   377k      0  0:03:37  0:00:31  0:03:06  717k

    
 14 80.2M   14 11.6M    0     0   373k      0  0:03:40  0:00:31  0:03:09  589k

    
 15 80.2M   15 12.3M    0     0   380k      0  0:03:35  0:00:33  0:03:02  606k

    
 15 80.2M   15 12.6M    0     0   383k      0  0:03:34  0:00:33  0:03:01  542k

    
 16 80.2M   16 13.0M    0     0   381k      0  0:03:35  0:00:34  0:03:01  455k

    
 16 80.2M   16 13.2M    0     0   378k      0  0:03:37  0:00:35  0:03:02  383k

    
 17 80.2M   17 13.9M    0     0   387k      0  0:03:31  0:00:36  0:02:55  480k

    
 18 80.2M   18 14.4M    0     0   388k      0  0:03:31  0:00:38  0:02:53  440k

    
 18 80.2M   18 14.8M    0     0   389k      0  0:03:30  0:00:38  0:02:52  431k

    
 19 80.2M   19 15.3M    0     0   394k      0  0:03:28  0:00:39  0:02:49  491k

    
 19 80.2M   19 15.6M    0     0   390k      0  0:03:30  0:00:41  0:02:49  468k

    
 19 80.2M   19 15.8M    0     0   388k      0  0:03:31  0:00:41  0:02:50  397k

    
 20 80.2M   20 16.4M    0     0   390k      0  0:03:30  0:00:42  0:02:48  411k

    
 20 80.2M   20 16.8M    0     0   393k      0  0:03:28  0:00:43  0:02:45  427k

    
 21 80.2M   21 17.2M    0     0   394k      0  0:03:28  0:00:44  0:02:44  393k

    
 22 80.2M   22 17.6M    0     0   393k      0  0:03:29  0:00:46  0:02:43  417k

    
 22 80.2M   22 17.9M    0     0   393k      0  0:03:29  0:00:46  0:02:43  427k

    
 23 80.2M   23 18.4M    0     0   393k      0  0:03:28  0:00:48  0:02:40  416k

    
 23 80.2M   23 18.8M    0     0   390k      0  0:03:30  0:00:49  0:02:41  364k

    
 24 80.2M   24 19.2M    0     0   394k      0  0:03:28  0:00:50  0:02:38  398k

    
 24 80.2M   24 19.5M    0     0   393k      0  0:03:28  0:00:50  0:02:38  402k

    
 24 80.2M   24 20.0M    0     0   393k      0  0:03:28  0:00:52  0:02:36  401k

    
 25 80.2M   25 20.3M    0     0   394k      0  0:03:28  0:00:52  0:02:36  405k

    
 26 80.2M   26 21.2M    0     0   398k      0  0:03:25  0:00:54  0:02:31  478k

    
 26 80.2M   26 21.2M    0     0   396k      0  0:03:27  0:00:54  0:02:33  407k

    
 26 80.2M   26 21.6M    0     0   394k      0  0:03:28  0:00:56  0:02:32  399k

    
 27 80.2M   27 21.9M    0     0   394k      0  0:03:28  0:00:56  0:02:32  400k

    
 27 80.2M   27 22.2M    0     0   394k      0  0:03:28  0:00:57  0:02:31  396k

    
 28 80.2M   28 23.1M    0     0   398k      0  0:03:26  0:00:59  0:02:27  396k

    
 28 80.2M   28 23.1M    0     0   396k      0  0:03:27  0:00:59  0:02:28  397k

    
 29 80.2M   29 23.5M    0     0   395k      0  0:03:27  0:01:00  0:02:27  405k

    
 29 80.2M   29 23.8M    0     0   395k      0  0:03:27  0:01:01  0:02:26  405k

    
 30 80.2M   30 24.3M    0     0   395k      0  0:03:27  0:01:03  0:02:24  407k

    
 31 80.2M   31 24.8M    0     0   397k      0  0:03:26  0:01:04  0:02:22  387k

    
 31 80.2M   31 25.5M    0     0   401k      0  0:03:24  0:01:05  0:02:19  465k

    
 32 80.2M   32 25.7M    0     0   400k      0  0:03:25  0:01:05  0:02:20  465k

    
 32 80.2M   32 26.1M    0     0   399k      0  0:03:25  0:01:06  0:02:19  456k

    
 33 80.2M   33 26.6M    0     0   400k      0  0:03:25  0:01:08  0:02:17  450k

    
 33 80.2M   33 26.9M    0     0   400k      0  0:03:25  0:01:08  0:02:17  432k

    
 34 80.2M   34 27.6M    0     0   403k      0  0:03:23  0:01:10  0:02:13  422k

    
 34 80.2M   34 27.7M    0     0   401k      0  0:03:24  0:01:10  0:02:14  412k

    
 35 80.2M   35 28.2M    0     0   401k      0  0:03:24  0:01:12  0:02:12  418k

    
 35 80.2M   35 28.5M    0     0   400k      0  0:03:24  0:01:12  0:02:12  414k

    
 36 80.2M   36 29.0M    0     0   401k      0  0:03:24  0:01:13  0:02:11  424k

    
 36 80.2M   36 29.3M    0     0   400k      0  0:03:25  0:01:15  0:02:10  360k

    
 37 80.2M   37 29.8M    0     0   403k      0  0:03:23  0:01:15  0:02:08  432k

    
 37 80.2M   37 30.2M    0     0   402k      0  0:03:24  0:01:17  0:02:07  417k

    
 38 80.2M   38 30.6M    0     0   402k      0  0:03:24  0:01:17  0:02:07  424k

    
 38 80.2M   38 31.1M    0     0   403k      0  0:03:23  0:01:18  0:02:05  432k

    
 39 80.2M   39 31.6M    0     0   404k      0  0:03:23  0:01:20  0:02:03  465k

    
 39 80.2M   39 31.9M    0     0   404k      0  0:03:23  0:01:20  0:02:03  415k

    
 40 80.2M   40 32.5M    0     0   404k      0  0:03:22  0:01:22  0:02:00  443k

    
 40 80.2M   40 32.8M    0     0   405k      0  0:03:22  0:01:23  0:01:59  446k

    
 41 80.2M   41 33.1M    0     0   405k      0  0:03:22  0:01:23  0:01:59  429k

    
 42 80.2M   42 33.8M    0     0   408k      0  0:03:21  0:01:24  0:01:57  476k

    
 42 80.2M   42 34.1M    0     0   406k      0  0:03:22  0:01:25  0:01:57  446k

    
 42 80.2M   42 34.4M    0     0   406k      0  0:03:21  0:01:26  0:01:55  440k

    
 43 80.2M   43 34.9M    0     0   407k      0  0:03:21  0:01:27  0:01:54  439k

    
 44 80.2M   44 35.4M    0     0   408k      0  0:03:21  0:01:28  0:01:53  459k

    
 44 80.2M   44 36.0M    0     0   408k      0  0:03:20  0:01:30  0:01:50  414k

    
 45 80.2M   45 36.2M    0     0   408k      0  0:03:21  0:01:30  0:01:51  436k

    
 45 80.2M   45 36.5M    0     0   407k      0  0:03:21  0:01:31  0:01:50  424k

    
 46 80.2M   46 37.0M    0     0   408k      0  0:03:21  0:01:33  0:01:48  425k

    
 46 80.2M   46 37.6M    0     0   408k      0  0:03:20  0:01:34  0:01:46  416k

    
 47 80.2M   47 38.1M    0     0   410k      0  0:03:20  0:01:35  0:01:45  446k

    
 47 80.2M   47 38.2M    0     0   409k      0  0:03:20  0:01:35  0:01:45  422k

    
 48 80.2M   48 38.6M    0     0   408k      0  0:03:21  0:01:37  0:01:44  416k

    
 48 80.2M   48 38.9M    0     0   408k      0  0:03:21  0:01:37  0:01:44  409k

    
 49 80.2M   49 39.5M    0     0   408k      0  0:03:21  0:01:38  0:01:43  402k

    
 50 80.2M   50 40.2M    0     0   410k      0  0:03:20  0:01:40  0:01:40  410k

    
 50 80.2M   50 40.3M    0     0   408k      0  0:03:20  0:01:41  0:01:39  407k

    
 50 80.2M   50 40.6M    0     0   408k      0  0:03:21  0:01:41  0:01:40  412k

    
 51 80.2M   51 41.0M    0     0   408k      0  0:03:21  0:01:42  0:01:39  418k

    
 51 80.2M   51 41.5M    0     0   409k      0  0:03:20  0:01:43  0:01:37  431k

    
 52 80.2M   52 42.1M    0     0   411k      0  0:03:19  0:01:44  0:01:35  426k

    
 52 80.2M   52 42.3M    0     0   409k      0  0:03:20  0:01:45  0:01:35  426k

    
 53 80.2M   53 42.8M    0     0   409k      0  0:03:20  0:01:47  0:01:33  432k

    
 53 80.2M   53 43.1M    0     0   409k      0  0:03:20  0:01:47  0:01:33  432k

    
 54 80.2M   54 44.0M    0     0   412k      0  0:03:19  0:01:49  0:01:30  453k

    
 54 80.2M   54 44.0M    0     0   410k      0  0:03:20  0:01:49  0:01:31  395k

    
 55 80.2M   55 44.4M    0     0   409k      0  0:03:20  0:01:51  0:01:29  410k

    
 55 80.2M   55 44.7M    0     0   409k      0  0:03:20  0:01:51  0:01:29  405k

    
 56 80.2M   56 45.2M    0     0   410k      0  0:03:20  0:01:53  0:01:27  421k

    
 57 80.2M   57 45.9M    0     0   412k      0  0:03:19  0:01:54  0:01:25  432k

    
 57 80.2M   57 46.1M    0     0   411k      0  0:03:19  0:01:54  0:01:25  423k

    
 57 80.2M   57 46.5M    0     0   410k      0  0:03:20  0:01:56  0:01:24  426k

    
 58 80.2M   58 46.8M    0     0   410k      0  0:03:19  0:01:56  0:01:23  445k

    
 59 80.2M   59 47.3M    0     0   411k      0  0:03:19  0:01:57  0:01:22  440k

    
 59 80.2M   59 47.4M    0     0   409k      0  0:03:20  0:01:58  0:01:22  321k

    
 59 80.2M   59 48.1M    0     0   410k      0  0:03:20  0:02:00  0:01:20  389k

    
 60 80.2M   60 48.4M    0     0   410k      0  0:03:20  0:02:00  0:01:20  413k

    
 60 80.2M   60 48.9M    0     0   410k      0  0:03:20  0:02:02  0:01:18  398k

    
 61 80.2M   61 49.2M    0     0   410k      0  0:03:20  0:02:02  0:01:18  390k

    
 62 80.2M   62 49.7M    0     0   410k      0  0:03:19  0:02:04  0:01:15  449k

    
 62 80.2M   62 50.0M    0     0   410k      0  0:03:20  0:02:04  0:01:16  411k

    
 62 80.2M   62 50.4M    0     0   410k      0  0:03:20  0:02:05  0:01:15  406k

    
 63 80.2M   63 50.7M    0     0   410k      0  0:03:20  0:02:06  0:01:14  401k

    
 63 80.2M   63 51.1M    0     0   409k      0  0:03:20  0:02:08  0:01:12  380k

    
 64 80.2M   64 51.5M    0     0   409k      0  0:03:20  0:02:08  0:01:12  365k

    
 64 80.2M   64 51.8M    0     0   409k      0  0:03:20  0:02:09  0:01:11  384k

    
 65 80.2M   65 52.4M    0     0   409k      0  0:03:20  0:02:11  0:01:09  388k

    
 65 80.2M   65 52.6M    0     0   409k      0  0:03:20  0:02:11  0:01:09  383k

    
 66 80.2M   66 53.1M    0     0   408k      0  0:03:20  0:02:13  0:01:07  394k

    
 66 80.2M   66 53.4M    0     0   408k      0  0:03:20  0:02:13  0:01:07  400k

    
 67 80.2M   67 54.0M    0     0   409k      0  0:03:20  0:02:15  0:01:05  412k

    
 68 80.2M   68 54.6M    0     0   410k      0  0:03:19  0:02:16  0:01:03  443k

    
 68 80.2M   68 54.8M    0     0   409k      0  0:03:20  0:02:17  0:01:03  422k

    
 68 80.2M   68 55.0M    0     0   409k      0  0:03:20  0:02:17  0:01:03  421k

    
 69 80.2M   69 55.3M    0     0   408k      0  0:03:21  0:02:18  0:01:03  403k

    
 69 80.2M   69 55.9M    0     0   408k      0  0:03:20  0:02:20  0:01:00  389k

    
 70 80.2M   70 56.4M    0     0   410k      0  0:03:20  0:02:20  0:01:00  386k

    
 70 80.2M   70 56.7M    0     0   408k      0  0:03:21  0:02:22  0:00:59  383k

    
 71 80.2M   71 57.0M    0     0   408k      0  0:03:21  0:02:22  0:00:59  384k

    
 71 80.2M   71 57.4M    0     0   408k      0  0:03:21  0:02:24  0:00:57  402k

    
 72 80.2M   72 57.8M    0     0   408k      0  0:03:21  0:02:24  0:00:57  403k

    
 72 80.2M   72 58.3M    0     0   409k      0  0:03:20  0:02:26  0:00:54  383k

    
 73 80.2M   73 58.6M    0     0   408k      0  0:03:21  0:02:26  0:00:55  408k

    
 73 80.2M   73 59.0M    0     0   408k      0  0:03:21  0:02:28  0:00:53  407k

    
 74 80.2M   74 59.4M    0     0   408k      0  0:03:21  0:02:28  0:00:53  408k

    
 74 80.2M   74 59.8M    0     0   409k      0  0:03:20  0:02:29  0:00:51  429k

    
 75 80.2M   75 60.2M    0     0   408k      0  0:03:20  0:02:30  0:00:50  400k

    
 75 80.2M   75 60.5M    0     0   408k      0  0:03:21  0:02:31  0:00:50  397k

    
 76 80.2M   76 60.9M    0     0   408k      0  0:03:21  0:02:33  0:00:48  397k

    
 76 80.2M   76 61.3M    0     0   408k      0  0:03:21  0:02:33  0:00:48  412k

    
 77 80.2M   77 61.8M    0     0   407k      0  0:03:21  0:02:35  0:00:46  357k

    
 77 80.2M   77 62.1M    0     0   408k      0  0:03:21  0:02:35  0:00:46  405k

    
 78 80.2M   78 62.5M    0     0   408k      0  0:03:21  0:02:37  0:00:44  404k

    
 78 80.2M   78 62.8M    0     0   407k      0  0:03:21  0:02:37  0:00:44  401k

    
 78 80.2M   78 63.2M    0     0   407k      0  0:03:21  0:02:38  0:00:43  377k

    
 79 80.2M   79 63.9M    0     0   408k      0  0:03:20  0:02:40  0:00:40  454k

    
 79 80.2M   79 64.0M    0     0   407k      0  0:03:21  0:02:40  0:00:41  375k

    
 80 80.2M   80 64.4M    0     0   407k      0  0:03:21  0:02:42  0:00:39  378k

    
 80 80.2M   80 64.7M    0     0   407k      0  0:03:21  0:02:42  0:00:39  384k

    
 81 80.2M   81 65.1M    0     0   407k      0  0:03:21  0:02:43  0:00:38  400k

    
 81 80.2M   81 65.5M    0     0   406k      0  0:03:22  0:02:44  0:00:38  333k

    
 82 80.2M   82 66.1M    0     0   407k      0  0:03:21  0:02:46  0:00:35  409k

    
 82 80.2M   82 66.4M    0     0   407k      0  0:03:21  0:02:46  0:00:35  417k

    
 83 80.2M   83 66.7M    0     0   407k      0  0:03:21  0:02:47  0:00:34  406k

    
 83 80.2M   83 66.8M    0     0   405k      0  0:03:22  0:02:48  0:00:34  348k

    
 83 80.2M   83 66.8M    0     0   402k      0  0:03:24  0:02:50  0:00:34  267k

    
 83 80.2M   83 66.9M    0     0   400k      0  0:03:24  0:02:51  0:00:33  178k

    
 83 80.2M   83 67.0M    0     0   399k      0  0:03:25  0:02:51  0:00:34  121k

    
 83 80.2M   83 67.0M    0     0   396k      0  0:03:27  0:02:53  0:00:34 68552

    
 83 80.2M   83 67

<div class="k-default-codeblock">
```
.1M    0     0   395k      0  0:03:27  0:02:53  0:00:34 55340

```
</div>
    
 83 80.2M   83 67.3M    0     0   393k      0  0:03:28  0:02:55  0:00:33   97k

    
 84 80.2M   84 67.5M    0     0   393k      0  0:03:28  0:02:55  0:00:33  123k

    
 84 80.2M   84 67.8M    0     0   392k      0  0:03:29  0:02:56  0:00:33  169k

    
 85 80.2M   85 68.2M    0     0   392k      0  0:03:29  0:02:57  0:00:32  244k

    
 85 80.2M   85 68.7M    0     0   393k      0  0:03:28  0:02:58  0:00:30  318k

    
 86 80.2M   86 69.0M    0     0   393k      0  0:03:28  0:02:59  0:00:29  379k

    
 86 80.2M   86 69.1M    0     0   391k      0  0:03:29  0:03:00  0:00:29  331k

    
 86 80.2M   86 69.1M    0     0   388k      0  0:03:31  0:03:02  0:00:29  260k

    
 86 80.2M   86 69.1M    0     0   387k      0  0:03:32  0:03:03  0:00:29  191k

    
 86 80.2M   86 69.2M    0     0   385k      0  0:03:33  0:03:03  0:00:30  104k

    
 86 80.2M   86 69.4M    0     0   383k      0  0:03:33  0:03:05  0:00:28 64954

    
 86 80.2M   86 69.4M    0     0   382k      0  0:03:34  0:03:05  0:00:29 65431

    
 86 80.2M   86 69.5M    0     0   381k      0  0:03:35  0:03:07  0:00:28 94686

    
 86 80.2M   86 69.7M    0     0   380k      0  0:03:36  0:03:07  0:00:29  115k

    
 87 80.2M   87 69.9M    0     0   379k      0  0:03:36  0:03:08  0:00:28  149k

    
 87 80.2M   87 70.1M    0     0   378k      0  0:03:37  0:03:10  0:00:27  159k

    
 87 80.2M   87 70.3M    0     0   377k      0  0:03:37  0:03:10  0:00:27  188k

    
 88 80.2M   88 70.7M    0     0   377k      0  0:03:37  0:03:12  0:00:25  229k

    
 88 80.2M   88 70.9M    0     0   376k      0  0:03:37  0:03:12  0:00:25  257k

    
 89 80.2M   89 71.5M    0     0   376k      0  0:03:38  0:03:14  0:00:24  279k

    
 89 80.2M   89 71.7M    0     0   376k      0  0:03:38  0:03:14  0:00:24  318k

    
 90 80.2M   90 72.2M    0     0   377k      0  0:03:37  0:03:16  0:00:21  366k

    
 90 80.2M   90 72.4M    0     0   376k      0  0:03:38  0:03:16  0:00:22  361k

    
 90 80.2M   90 72.8M    0     0   377k      0  0:03:37  0:03:17  0:00:20  386k

    
 91 80.2M   91 73.1M    0     0   376k      0  0:03:38  0:03:18  0:00:20  388k

    
 91 80.2M   91 73.6M    0     0   376k      0  0:03:37  0:03:20  0:00:17  384k

    
 92 80.2M   92 73.9M    0     0   376k      0  0:03:37  0:03:20  0:00:17  359k

    
 92 80.2M   92 74.4M    0     0   377k      0  0:03:37  0:03:21  0:00:16  401k

    
 93 80.2M   93 74.7M    0     0   377k      0  0:03:37  0:03:22  0:00:15  386k

    
 93 80.2M   93 75.2M    0     0   377k      0  0:03:37  0:03:24  0:00:13  409k

    
 94 80.2M   94 75.5M    0     0   377k      0  0:03:37  0:03:24  0:00:13  411k

    
 94 80.2M   94 75.9M    0     0   378k      0  0:03:37  0:03:25  0:00:12  425k

    
 95 80.2M   95 76.3M    0     0   377k      0  0:03:37  0:03:27  0:00:10  395k

    
 95 80.2M   95 76.7M    0     0   377k      0  0:03:37  0:03:27  0:00:10  400k

    
 96 80.2M   96 77.0M    0     0   377k      0  0:03:37  0:03:28  0:00:09  383k

    
 96 80.2M   96 77.2M    0     0   376k      0  0:03:38  0:03:30  0:00:08  325k

    
 96 80.2M   96 77.4M    0     0   375k      0  0:03:38  0:03:30  0:00:08  290k

    
 97 80.2M   97 77.8M    0     0   376k      0  0:03:38  0:03:32  0:00:06  306k

    
 97 80.2M   97 78.1M    0     0   376k      0  0:03:38  0:03:32  0:00:06  297k

    
 98 80.2M   98 78.6M    0     0   376k      0  0:03:38  0:03:33  0:00:05  318k

    
 98 80.2M   98 78.9M    0     0   376k      0  0:03:38  0:03:34  0:00:04  374k

    
 98 80.2M   98 79.4M    0     0   376k      0  0:03:38  0:03:35  0:00:03  410k

    
 99 80.2M   99 79.9M    0     0   377k      0  0:03:37  0:03:36  0:00:01  444k

    
100 80.2M  100 80.2M    0     0   378k      0  0:03:37  0:03:37 --:--:--  473k



```python

def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):
    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


train_df = get_data_from_text_files("train")
test_df = get_data_from_text_files("test")

all_data = pd.concat([train_df, test_df], ignore_index=True)
```

---
## Dataset preparation

We will use the `TextVectorization` layer to vectorize the text into integer token ids.
It transforms a batch of strings into either
a sequence of token indices (one sample = 1D array of integer token indices, in order)
or a dense representation (one sample = 1D array of float values encoding an unordered set of tokens).

Below, we define 3 preprocessing functions.

1.  The `get_vectorize_layer` function builds the `TextVectorization` layer.
2.  The `encode` function encodes raw text into integer token ids.
3.  The `get_masked_input_and_labels` function will mask input token ids.
It masks 15% of all input tokens in each sequence at random.


```python
# For data pre-processing and tf.data.Dataset
import tensorflow as tf


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence length.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


vectorize_layer = get_vectorize_layer(
    all_data.review.values.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    special_tokens=["[mask]"],
)

# Get mask token id for masked language model
mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]


def encode(texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()


def get_masked_input_and_labels(encoded_texts):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[inp_mask_2mask] = (
        mask_token_id  # mask token is the last in the dict
    )

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# We have 25000 examples for training
x_train = encode(train_df.review.values)  # encode reviews with vectorizer
y_train = train_df.sentiment.values
train_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .batch(config.BATCH_SIZE)
)

# We have 25000 examples for testing
x_test = encode(test_df.review.values)
y_test = test_df.sentiment.values
test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    config.BATCH_SIZE
)

# Dataset for end to end model input (will be used at the end)
test_raw_classifier_ds = test_df

# Prepare data for masked language model
x_all_review = encode(all_data.review.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
    x_all_review
)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
```

---
## Create BERT model (Pretraining Model) for masked language modeling

We will create a BERT-like pretraining model architecture
using the `MultiHeadAttention` layer.
It will take token ids as inputs (including masked tokens)
and it will predict the correct ids for the masked input tokens.


```python

def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}_multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}_att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}_ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}_ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=None)
loss_tracker = keras.metrics.Mean(name="loss")


class MaskedLanguageModel(keras.Model):

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):

        loss = loss_fn(y, y_pred, sample_weight)
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        return keras.ops.sum(loss)

    def compute_metrics(self, x, y, y_pred, sample_weight):

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


def create_masked_language_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype="int64")

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = keras_hub.layers.PositionEmbedding(
        sequence_length=config.MAX_LEN
    )(word_embeddings)
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model


id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)


sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])
generator_callback = MaskedTextGenerator(sample_tokens.numpy())

bert_masked_model = create_masked_language_bert_model()
bert_masked_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "masked_bert_model"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ word_embedding      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │  <span style="color: #00af00; text-decoration-color: #00af00">3,840,000</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ position_embedding  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">32,768</span> │ word_embedding[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">PositionEmbedding</span>) │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ word_embedding[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                     │                   │            │ position_embeddi… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_multihea… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">66,048</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MultiHeadAttentio…</span> │                   │            │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│                     │                   │            │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_att_drop… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ encoder_0_multih… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│                     │                   │            │ encoder_0_att_dr… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_att_laye… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_ffn       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">33,024</span> │ encoder_0_att_la… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_ffn_drop… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ encoder_0_ffn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ encoder_0_att_la… │
│                     │                   │            │ encoder_0_ffn_dr… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ encoder_0_ffn_laye… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ add_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ mlm_cls (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>,       │  <span style="color: #00af00; text-decoration-color: #00af00">3,870,000</span> │ encoder_0_ffn_la… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">30000</span>)            │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,842,352</span> (29.92 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,842,352</span> (29.92 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



---
## Train and Save


```python
bert_masked_model.fit(mlm_ds, epochs=5, callbacks=[generator_callback])
bert_masked_model.save("bert_mlm_imdb.keras")
```

    
  1/16 ━[37m━━━━━━━━━━━━━━━━━━━  3:02 12s/step - loss: 10.3103

<div class="k-default-codeblock">
```

```
</div>
  2/16 ━━[37m━━━━━━━━━━━━━━━━━━  3:31 15s/step - loss: 10.2979

<div class="k-default-codeblock">
```

```
</div>
  3/16 ━━━[37m━━━━━━━━━━━━━━━━━  3:25 16s/step - loss: 10.2859

<div class="k-default-codeblock">
```

```
</div>
  4/16 ━━━━━[37m━━━━━━━━━━━━━━━  3:14 16s/step - loss: 10.2727

<div class="k-default-codeblock">
```

```
</div>
  5/16 ━━━━━━[37m━━━━━━━━━━━━━━  2:57 16s/step - loss: 10.2564

<div class="k-default-codeblock">
```

```
</div>
  6/16 ━━━━━━━[37m━━━━━━━━━━━━━  2:42 16s/step - loss: 10.2378

<div class="k-default-codeblock">
```

```
</div>
  7/16 ━━━━━━━━[37m━━━━━━━━━━━━  2:26 16s/step - loss: 10.2182

<div class="k-default-codeblock">
```

```
</div>
  8/16 ━━━━━━━━━━[37m━━━━━━━━━━  2:10 16s/step - loss: 10.1975

<div class="k-default-codeblock">
```

```
</div>
  9/16 ━━━━━━━━━━━[37m━━━━━━━━━  1:55 16s/step - loss: 10.1745

<div class="k-default-codeblock">
```

```
</div>
 10/16 ━━━━━━━━━━━━[37m━━━━━━━━  1:39 17s/step - loss: 10.1503

<div class="k-default-codeblock">
```

```
</div>
 11/16 ━━━━━━━━━━━━━[37m━━━━━━━  1:23 17s/step - loss: 10.1254

<div class="k-default-codeblock">
```

```
</div>
 12/16 ━━━━━━━━━━━━━━━[37m━━━━━  1:07 17s/step - loss: 10.0993

<div class="k-default-codeblock">
```

```
</div>
 13/16 ━━━━━━━━━━━━━━━━[37m━━━━  50s 17s/step - loss: 10.0726 

<div class="k-default-codeblock">
```

```
</div>
 14/16 ━━━━━━━━━━━━━━━━━[37m━━━  33s 17s/step - loss: 10.0452

<div class="k-default-codeblock">
```

```
</div>
 15/16 ━━━━━━━━━━━━━━━━━━[37m━━  16s 17s/step - loss: 10.0174

<div class="k-default-codeblock">
```

```
</div>
 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 17s/step - loss: 9.9899  

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step


<div class="k-default-codeblock">
```
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'a',
 'prediction': 'i have watched this a and it was awesome',
 'probability': 0.0013674975}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'i',
 'prediction': 'i have watched this i and it was awesome',
 'probability': 0.0012694978}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'is',
 'prediction': 'i have watched this is and it was awesome',
 'probability': 0.0012668626}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'to',
 'prediction': 'i have watched this to and it was awesome',
 'probability': 0.0012651902}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'of',
 'prediction': 'i have watched this of and it was awesome',
 'probability': 0.0011966776}

```
</div>
 16/16 ━━━━━━━━━━━━━━━━━━━━ 261s 17s/step - loss: 9.9656


---
## Fine-tune a sentiment classification model

We will fine-tune our self-supervised model on a downstream task of sentiment classification.
To do this, let's create a classifier by adding a pooling layer and a `Dense` layer on top of the
pretrained BERT features.


```python
# Load pretrained bert model
mlm_model = keras.models.load_model(
    "bert_mlm_imdb.keras", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
)
pretrained_bert_model = keras.Model(
    mlm_model.input, mlm_model.get_layer("encoder_0_ffn_layernormalization").output
)

# Freeze it
pretrained_bert_model.trainable = False


def create_classifier_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype="int64")
    sequence_output = pretrained_bert_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output)
    outputs = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam()
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


classifer_model = create_classifier_bert_model()
classifer_model.summary()

# Train the classifier with frozen BERT stage
classifer_model.fit(
    train_classifier_ds,
    epochs=5,
    validation_data=test_classifier_ds,
)

# Unfreeze the BERT model for fine-tuning
pretrained_bert_model.trainable = True
optimizer = keras.optimizers.Adam()
classifer_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
classifer_model.fit(
    train_classifier_ds,
    epochs=5,
    validation_data=test_classifier_ds,
)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "classification"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ functional_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │     <span style="color: #00af00; text-decoration-color: #00af00">3,972,352</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling1d            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalMaxPooling1D</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">65</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,980,673</span> (15.19 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">8,321</span> (32.50 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,972,352</span> (15.15 MB)
</pre>



    
 1/8 ━━[37m━━━━━━━━━━━━━━━━━━  0s 140ms/step - accuracy: 0.5312 - loss: 0.7599

<div class="k-default-codeblock">
```

```
</div>
 2/8 ━━━━━[37m━━━━━━━━━━━━━━━  1s 184ms/step - accuracy: 0.5703 - loss: 0.7296

<div class="k-default-codeblock">
```

```
</div>
 3/8 ━━━━━━━[37m━━━━━━━━━━━━━  0s 164ms/step - accuracy: 0.5851 - loss: 0.7164

<div class="k-default-codeblock">
```

```
</div>
 4/8 ━━━━━━━━━━[37m━━━━━━━━━━  0s 161ms/step - accuracy: 0.5794 - loss: 0.7125

<div class="k-default-codeblock">
```

```
</div>
 5/8 ━━━━━━━━━━━━[37m━━━━━━━━  0s 158ms/step - accuracy: 0.5685 - loss: 0.7105

<div class="k-default-codeblock">
```

```
</div>
 6/8 ━━━━━━━━━━━━━━━[37m━━━━━  0s 158ms/step - accuracy: 0.5589 - loss: 0.7090

<div class="k-default-codeblock">
```

```
</div>
 7/8 ━━━━━━━━━━━━━━━━━[37m━━━  0s 156ms/step - accuracy: 0.5504 - loss: 0.7080

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 151ms/step - accuracy: 0.5426 - loss: 0.7076

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 2s 288ms/step - accuracy: 0.5366 - loss: 0.7073 - val_accuracy: 0.4920 - val_loss: 0.6975


    
 1/8 ━━[37m━━━━━━━━━━━━━━━━━━  3s 436ms/step - accuracy: 0.5000 - loss: 0.7119

<div class="k-default-codeblock">
```

```
</div>
 2/8 ━━━━━[37m━━━━━━━━━━━━━━━  3s 534ms/step - accuracy: 0.5469 - loss: 0.6903

<div class="k-default-codeblock">
```

```
</div>
 3/8 ━━━━━━━[37m━━━━━━━━━━━━━  2s 472ms/step - accuracy: 0.5660 - loss: 0.6913

<div class="k-default-codeblock">
```

```
</div>
 4/8 ━━━━━━━━━━[37m━━━━━━━━━━  1s 461ms/step - accuracy: 0.5671 - loss: 0.7032

<div class="k-default-codeblock">
```

```
</div>
 5/8 ━━━━━━━━━━━━[37m━━━━━━━━  1s 459ms/step - accuracy: 0.5636 - loss: 0.7116

<div class="k-default-codeblock">
```

```
</div>
 6/8 ━━━━━━━━━━━━━━━[37m━━━━━  0s 468ms/step - accuracy: 0.5626 - loss: 0.7156

<div class="k-default-codeblock">
```

```
</div>
 7/8 ━━━━━━━━━━━━━━━━━[37m━━━  0s 476ms/step - accuracy: 0.5600 - loss: 0.7183

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 476ms/step - accuracy: 0.5580 - loss: 0.7198

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 5s 650ms/step - accuracy: 0.5565 - loss: 0.7210 - val_accuracy: 0.5160 - val_loss: 0.6895





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7a0e5fd9bf50>

```
</div>
---
## Create an end-to-end model and evaluate it

When you want to deploy a model, it's best if it already includes its preprocessing
pipeline, so that you don't have to reimplement the preprocessing logic in your
production environment. Let's create an end-to-end model that incorporates
the `TextVectorization` layer inside evalaute method, and let's evaluate. We will pass raw strings as input.


```python

# We create a custom Model to override the evaluate method so
# that it first pre-process text data
class ModelEndtoEnd(keras.Model):

    def evaluate(self, inputs):
        features = encode(inputs.review.values)
        labels = inputs.sentiment.values
        test_classifier_ds = (
            tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(1000)
            .batch(config.BATCH_SIZE)
        )
        return super().evaluate(test_classifier_ds)

    # Build the model
    def build(self, input_shape):
        self.built = True


def get_end_to_end(model):
    inputs = classifer_model.inputs[0]
    outputs = classifer_model.outputs
    end_to_end_model = ModelEndtoEnd(inputs, outputs, name="end_to_end_model")
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(classifer_model)
# Pass raw text dataframe to the model
end_to_end_classification_model.evaluate(test_raw_classifier_ds)
```

    
 1/8 ━━[37m━━━━━━━━━━━━━━━━━━  0s 138ms/step - accuracy: 0.6875 - loss: 0.6684

<div class="k-default-codeblock">
```

```
</div>
 2/8 ━━━━━[37m━━━━━━━━━━━━━━━  1s 225ms/step - accuracy: 0.6250 - loss: 0.6761

<div class="k-default-codeblock">
```

```
</div>
 3/8 ━━━━━━━[37m━━━━━━━━━━━━━  0s 190ms/step - accuracy: 0.5833 - loss: 0.6820

<div class="k-default-codeblock">
```

```
</div>
 4/8 ━━━━━━━━━━[37m━━━━━━━━━━  0s 184ms/step - accuracy: 0.5605 - loss: 0.6848

<div class="k-default-codeblock">
```

```
</div>
 5/8 ━━━━━━━━━━━━[37m━━━━━━━━  0s 178ms/step - accuracy: 0.5422 - loss: 0.6871

<div class="k-default-codeblock">
```

```
</div>
 6/8 ━━━━━━━━━━━━━━━[37m━━━━━  0s 174ms/step - accuracy: 0.5352 - loss: 0.6880

<div class="k-default-codeblock">
```

```
</div>
 7/8 ━━━━━━━━━━━━━━━━━[37m━━━  0s 169ms/step - accuracy: 0.5320 - loss: 0.6883

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 165ms/step - accuracy: 0.5300 - loss: 0.6885

<div class="k-default-codeblock">
```

```
</div>
 8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 166ms/step - accuracy: 0.5285 - loss: 0.6886





<div class="k-default-codeblock">
```
[0.6894814372062683, 0.515999972820282]

```
</div>

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
