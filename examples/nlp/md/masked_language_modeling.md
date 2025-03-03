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

    
  0 80.2M    0 16384    0     0   8145      0  2:52:08  0:00:02  2:52:06  8147

    
  0 80.2M    0 49152    0     0  17366      0  1:20:44  0:00:02  1:20:42 17362

    
  0 80.2M    0  128k    0     0  32289      0  0:43:25  0:00:04  0:43:21 32291

    
  0 80.2M    0  208k    0     0  43659      0  0:32:06  0:00:04  0:32:02 43654

    
  0 80.2M    0  416k    0     0  70937      0  0:19:45  0:00:06  0:19:39 88470

    
  0 80.2M    0  688k    0     0   101k      0  0:13:31  0:00:06  0:13:25  140k

    
  1 80.2M    1  912k    0     0   116k      0  0:11:47  0:00:07  0:11:40  172k

    
  1 80.2M    1 1296k    0     0   144k      0  0:09:28  0:00:08  0:09:20  237k

    
  2 80.2M    2 1680k    0     0   171k      0  0:07:58  0:00:09  0:07:49  299k

    
  2 80.2M    2 2032k    0     0   187k      0  0:07:17  0:00:10  0:07:07  335k

    
  3 80.2M    3 2736k    0     0   219k      0  0:06:14  0:00:12  0:06:02  361k

    
  3 80.2M    3 2736k    0     0   212k      0  0:06:26  0:00:12  0:06:14  363k

    
  4 80.2M    4 3328k    0     0   236k      0  0:05:46  0:00:14  0:05:32  399k

    
  4 80.2M    4 3680k    0     0   248k      0  0:05:30  0:00:14  0:05:16  398k

    
  5 80.2M    5 4176k    0     0   260k      0  0:05:15  0:00:16  0:04:59  410k

    
  5 80.2M    5 4448k    0     0   263k      0  0:05:11  0:00:16  0:04:55  388k

    
  5 80.2M    5 4896k    0     0   272k      0  0:05:01  0:00:17  0:04:44  422k

    
  6 80.2M    6 5232k    0     0   278k      0  0:04:55  0:00:18  0:04:37  400k

    
  6 80.2M    6 5728k    0     0   285k      0  0:04:47  0:00:20  0:04:27  392k

    
  7 80.2M    7 6048k    0     0   290k      0  0:04:43  0:00:20  0:04:23  388k

    
  7 80.2M    7 6544k    0     0   296k      0  0:04:37  0:00:22  0:04:15  401k

    
  8 80.2M    8 6896k    0     0   301k      0  0:04:32  0:00:22  0:04:10  406k

    
  9 80.2M    9 7424k    0     0   307k      0  0:04:27  0:00:24  0:04:03  411k

    
  9 80.2M    9 7680k    0     0   308k      0  0:04:25  0:00:24  0:04:01  404k

    
  9 80.2M    9 8144k    0     0   313k      0  0:04:22  0:00:25  0:03:57  409k

    
 10 80.2M   10 8512k    0     0   317k      0  0:04:18  0:00:26  0:03:52  417k

    
 10 80.2M   10 9024k    0     0   322k      0  0:04:14  0:00:27  0:03:47  418k

    
 11 80.2M   11 9360k    0     0   316k      0  0:04:19  0:00:29  0:03:50  356k

    
 11 80.2M   11 9520k    0     0   317k      0  0:04:18  0:00:29  0:03:49  360k

    
 11 80.2M   11 9744k    0     0   312k      0  0:04:23  0:00:31  0:03:52  306k

    
 12 80.2M   12 9904k    0     0   309k      0  0:04:25  0:00:32  0:03:53  266k

    
 12 80.2M   12  9.8M    0     0   306k      0  0:04:28  0:00:32  0:03:56  214k

    
 12 80.2M   12 10.0M    0     0   302k      0  0:04:31  0:00:34  0:03:57  209k

    
 12 80.2M   12 10.2M    0     0   300k      0  0:04:33  0:00:34  0:03:59  195k

    
 13 80.2M   13 10.5M    0     0   297k      0  0:04:35  0:00:36  0:03:59  205k

    
 13 80.2M   13 10.7M    0     0   296k      0  0:04:36  0:00:36  0:04:00  214k

    
 13 80.2M   13 11.0M    0     0   297k      0  0:04:36  0:00:38  0:03:58  241k

    
 14 80.2M   14 11.3M    0     0   300k      0  0:04:33  0:00:38  0:03:55  282k

    
 14 80.2M   14 11.9M    0     0   304k      0  0:04:29  0:00:40  0:03:49  330k

    
 15 80.2M   15 12.1M    0     0   304k      0  0:04:29  0:00:40  0:03:49  359k

    
 15 80.2M   15 12.6M    0     0   307k      0  0:04:26  0:00:42  0:03:44  386k

    
 16 80.2M   16 13.0M    0     0   309k      0  0:04:25  0:00:42  0:03:43  406k

    
 16 80.2M   16 13.3M    0     0   311k      0  0:04:23  0:00:43  0:03:40  399k

    
 17 80.2M   17 13.6M    0     0   311k      0  0:04:23  0:00:45  0:03:38  367k

    
 17 80.2M   17 14.0M    0     0   312k      0  0:04:22  0:00:45  0:03:37  377k

    
 18 80.2M   18 14.4M    0     0   314k      0  0:04:20  0:00:47  0:03:33  374k

    
 18 80.2M   18 14.7M    0     0   316k      0  0:04:19  0:00:47  0:03:32  371k

    
 19 80.2M   19 15.2M    0     0   318k      0  0:04:17  0:00:49  0:03:28  378k

    
 19 80.2M   19 15.2M    0     0   313k      0  0:04:22  0:00:49  0:03:33  328k

    
 19 80.2M   19 15.2M    0     0   305k      0  0:04:28  0:00:51  0:03:37  246k

    
 19 80.2M   19 15.2M    0     0   301k      0  0:04:32  0:00:51  0:03:41  169k

    
 19 80.2M   19 15.2M    0     0   296k      0  0:04:36  0:00:52  0:03:44  107k

    
 19 80.2M   19 15.3M    0     0   290k      0  0:04:43  0:00:54  0:03:49 13115

    
 19 80.2M   19 15.3M    0     0   285k      0  0:04:47  0:00:54  0:03:53  9943

    
 19 80.2M   19 15.3M    0     0   279k      0  0:04:53  0:00:56  0:03:57  9998

    
 19 80.2M   19 15.3M    0     0   275k      0  0:04:58  0:00:57  0:04:01 12800

    
 19 80.2M   19 15.3M    0     0   271k      0  0:05:02  0:00:57  0:04:05  9601

    
 19 80.2M   19 15.3M    0     0   266k      0  0:05:07  0:00:58  0:04:09  9982

    
 19 80.2M   19 15.3M    0     0   260k      0  0:05:15  0:01:00  0:04:15  8974

    
 19 80.2M   19 15.3M    0     0   257k      0  0:05:18  0:01:01  0:04:17  9797

    
 19 80.2M   19 15.3M    0     0   254k      0  0:05:23  0:01:01  0:04:22  6789

    
 19 80.2M   19 15.3M    0     0   249k      0  0:05:29  0:01:03  0:04:26  9412

    
 19 80.2M   19 15.4M    0     0   247k      0  0:05:31  0:01:03  0:04:28 22832

    
 19 80.2M   19 15.4M    0     0   244k      0  0:05:36  0:01:04  0:04:32 28908

    
 19 80.2M   19 15.5M    0     0   240k      0  0:05:42  0:01:06  0:04:36 26125

    
 19 80.2M   19 15.5M    0     0   237k      0  0:05:46  0:01:06  0:04:40 29408

    
 19 80.2M   19 15.5M    0     0   233k      0  0:05:51  0:01:08  0:04:43 29657

    
 19 80.2M   19 15.5M    0     0   231k      0  0:05:55  0:01:08  0:04:47 16697

    
 19 80.2M   19 15.5M    0     0   227k      0  0:06:01  0:01:10  0:04:51 15702

    
 19 80.2M   19 15.5M    0     0   224k      0  0:06:05  0:01:11  0:04:54 19590

    
 19 80.2M   19 15.6M    0     0   222k      0  0:06:09  0:01:11  0:04:58 19609

    
 19 80.2M   19 15.6M    0     0   218k      0  0:06:15  0:01:13  0:05:02 22647

    
 19 80.2M   19 15.6M    0     0   216k      0  0:06:18  0:01:13  0:05:05 25615

    
 19 80.2M   19 15.6M    0     0   214k      0  0:06:22  0:01:14  0:05:08 27816

    
 19 80.2M   19 15.7M    0     0   211k      0  0:06:27  0:01:16  0:05:11 29995

    
 19 80.2M   19 15.7M    0     0   210k      0  0:06:30  0:01:16  0:05:14 40026

    
 19 80.2M   19 15.8M    0     0   208k      0  0:06:34  0:01:17  0:05:17 51157

    
 19 80.2M   19 16.0M    0     0   207k      0  0:06:36  0:01:19  0:05:17 68579

    
 20 80.2M   20 16.0M    0     0   206k      0  0:06:37  0:01:19  0:05:18 84908

    
 20 80.2M   20 16.2M    0     0   205k      0  0:06:39  0:01:21  0:05:18  114k

    
 20 80.2M   20 16.4M    0     0   205k      0  0:06:39  0:01:21  0:05:18  130k

    
 20 80.2M   20 16.7M    0     0   206k      0  0:06:38  0:01:23  0:05:15  174k

    
 21 80.2M   21 16.9M    0     0   207k      0  0:06:36  0:01:23  0:05:13  205k

    
 21 80.2M   21 17.4M    0     0   210k      0  0:06:31  0:01:25  0:05:06  265k

    
 22 80.2M   22 17.8M    0     0   211k      0  0:06:28  0:01:26  0:05:02  297k

    
 22 80.2M   22 18.2M    0     0   214k      0  0:06:22  0:01:27  0:04:55  355k

    
 23 80.2M   23 18.5M    0     0   216k      0  0:06:19  0:01:27  0:04:52  388k

    
 23 80.2M   23 19.1M    0     0   218k      0  0:06:16  0:01:29  0:04:47  386k

    
 24 80.2M   24 19.4M    0     0   221k      0  0:06:11  0:01:29  0:04:42  414k

    
 24 80.2M   24 19.7M    0     0   222k      0  0:06:09  0:01:30  0:04:39  432k

    
 25 80.2M   25 20.1M    0     0   224k      0  0:06:06  0:01:32  0:04:34  397k

    
 25 80.2M   25 20.4M    0     0   225k      0  0:06:03  0:01:32  0:04:31  397k

    
 26 80.2M   26 20.9M    0     0   228k      0  0:05:59  0:01:34  0:04:25  426k

    
 26 80.2M   26 21.3M    0     0   230k      0  0:05:56  0:01:34  0:04:22  393k

    
 27 80.2M   27 21.8M    0     0   232k      0  0:05:53  0:01:36  0:04:17  403k

    
 27 80.2M   27 22.1M    0     0   234k      0  0:05:51  0:01:36  0:04:15  416k

    
 28 80.2M   28 22.6M    0     0   236k      0  0:05:47  0:01:38  0:04:09  418k

    
 28 80.2M   28 22.9M    0     0   237k      0  0:05:46  0:01:38  0:04:08  408k

    
 29 80.2M   29 23.3M    0     0   239k      0  0:05:43  0:01:40  0:04:03  401k

    
 29 80.2M   29 23.7M    0     0   240k      0  0:05:41  0:01:40  0:04:01  405k

    
 30 80.2M   30 24.2M    0     0   242k      0  0:05:38  0:01:42  0:03:56  401k

    
 30 80.2M   30 24.4M    0     0   243k      0  0:05:37  0:01:42  0:03:55  381k

    
 30 80.2M   30 24.7M    0     0   244k      0  0:05:36  0:01:43  0:03:53  380k

    
 31 80.2M   31 25.2M    0     0   245k      0  0:05:33  0:01:44  0:03:49  388k

    
 31 80.2M   31 25.6M    0     0   247k      0  0:05:31  0:01:46  0:03:45  390k

    
 32 80.2M   32 25.9M    0     0   248k      0  0:05:29  0:01:46  0:03:43  387k

    
 32 80.2M   32 26.4M    0     0   250k      0  0:05:27  0:01:48  0:03:39  407k

    
 33 80.2M   33 26.9M    0     0   252k      0  0:05:24  0:01:49  0:03:35  426k

    
 34 80.2M   34 27.4M    0     0   254k      0  0:05:22  0:01:50  0:03:32  424k

    
 34 80.2M   34 27.7M    0     0   255k      0  0:05:21  0:01:51  0:03:30  409k

    
 34 80.2M   34 28.0M    0     0   256k      0  0:05:20  0:01:51  0:03:29  403k

    
 35 80.2M   35 28.3M    0     0   257k      0  0:05:19  0:01:52  0:03:27  399k

    
 35 80.2M   35 28.8M    0     0   259k      0  0:05:17  0:01:54  0:03:23  393k

    
 36 80.2M   36 29.0M    0     0   259k      0  0:05:16  0:01:54  0:03:22  372k

    
 36 80.2M   36 29.5M    0     0   260k      0  0:05:15  0:01:56  0:03:19  374k

    
 37 80.2M   37 29.8M    0     0   261k      0  0:05:14  0:01:56  0:03:18  379k

    
 37 80.2M   37 30.2M    0     0   262k      0  0:05:13  0:01:58  0:03:15  371k

    
 38 80.2M   38 30.5M    0     0   263k      0  0:05:12  0:01:58  0:03:14  361k

    
 38 80.2M   38 31.0M    0     0   264k      0  0:05:10  0:02:00  0:03:10  372k

    
 39 80.2M   39 31.3M    0     0   265k      0  0:05:09  0:02:00  0:03:09  380k

    
 39 80.2M   39 31.6M    0     0   266k      0  0:05:08  0:02:01  0:03:07  388k

    
 40 80.2M   40 32.2M    0     0   268k      0  0:05:06  0:02:03  0:03:03  407k

    
 40 80.2M   40 32.4M    0     0   268k      0  0:05:05  0:02:03  0:03:02  397k

    
 41 80.2M   41 32.9M    0     0   269k      0  0:05:04  0:02:05  0:02:59  397k

    
 41 80.2M   41 33.2M    0     0   269k      0  0:05:04  0:02:06  0:02:58  372k

    
 42 80.2M   42 33.7M    0     0   271k      0  0:05:02  0:02:07  0:02:55  387k

    
 42 80.2M   42 33.9M    0     0   272k      0  0:05:01  0:02:07  0:02:54  375k

    
 42 80.2M   42 34.4M    0     0   273k      0  0:05:00  0:02:08  0:02:52  387k

    
 43 80.2M   43 34.8M    0     0   275k      0  0:04:58  0:02:09  0:02:49  418k

    
 43 80.2M   43 35.2M    0     0   275k      0  0:04:58  0:02:10  0:02:48  427k

    
 44 80.2M   44 35.6M    0     0   276k      0  0:04:57  0:02:12  0:02:45  401k

    
 44 80.2M   44 35.9M    0     0   277k      0  0:04:56  0:02:12  0:02:44  400k

    
 45 80.2M   45 36.3M    0     0   278k      0  0:04:55  0:02:13  0:02:42  405k

    
 45 80.2M   45 36.8M    0     0   279k      0  0:04:54  0:02:15  0:02:39  388k

    
 46 80.2M   46 37.0M    0     0   279k      0  0:04:53  0:02:15  0:02:38  393k

    
 46 80.2M   46 37.5M    0     0   280k      0  0:04:53  0:02:17  0:02:36  387k

    
 47 80.2M   47 37.8M    0     0   281k      0  0:04:52  0:02:17  0:02:35  393k

    
 47 80.2M   47 38.3M    0     0   282k      0  0:04:50  0:02:19  0:02:31  398k

    
 48 80.2M   48 38.6M    0     0   282k      0  0:04:50  0:02:19  0:02:31  382k

    
 48 80.2M   48 39.0M    0     0   283k      0  0:04:49  0:02:21  0:02:28  386k

    
 49 80.2M   49 39.3M    0     0   284k      0  0:04:49  0:02:21  0:02:28  392k

    
 49 80.2M   49 39.8M    0     0   285k      0  0:04:48  0:02:23  0:02:25  386k

    
 50 80.2M   50 40.1M    0     0   285k      0  0:04:47  0:02:23  0:02:24  374k

    
 50 80.2M   50 40.6M    0     0   286k      0  0:04:46  0:02:25  0:02:21  385k

    
 51 80.2M   51 40.9M    0     0   287k      0  0:04:45  0:02:25  0:02:20  402k

    
 51 80.2M   51 41.4M    0     0   288k      0  0:04:44  0:02:27  0:02:17  410k

    
 52 80.2M   52 41.7M    0     0   288k      0  0:04:44  0:02:27  0:02:17  402k

    
 52 80.2M   52 42.0M    0     0   288k      0  0:04:44  0:02:29  0:02:15  364k

    
 52 80.2M   52 42.3M    0     0   289k      0  0:04:44  0:02:29  0:02:15  358k

    
 53 80.2M   53 42.5M    0     0   289k      0  0:04:44  0:02:30  0:02:14  335k

    
 53 80.2M   53 42.9M    0     0   289k      0  0:04:44  0:02:32  0:02:12  306k

    
 53 80.2M   53 43.1M    0     0   289k      0  0:04:43  0:02:32  0:02:11  302k

    
 54 80.2M   54 43.5M    0     0   289k      0  0:04:43  0:02:33  0:02:10  332k

    
 54 80.2M   54 43.8M    0     0   290k      0  0:04:43  0:02:34  0:02:09  322k

    
 55 80.2M   55 44.2M    0     0   290k      0  0:04:43  0:02:36  0:02:07  324k

    
 55 80.2M   55 44.5M    0     0   290k      0  0:04:42  0:02:36  0:02:06  332k

    
 56 80.2M   56 45.0M    0     0   291k      0  0:04:42  0:02:38  0:02:04  341k

    
 56 80.2M   56 45.4M    0     0   292k      0  0:04:40  0:02:39  0:02:01  381k

    
 57 80.2M   57 45.8M    0     0   293k      0  0:04:39  0:02:39  0:02:00  397k

    
 57 80.2M   57 46.3M    0     0   294k      0  0:04:38  0:02:41  0:01:57  425k

    
 58 80.2M   58 46.5M    0     0   294k      0  0:04:38  0:02:41  0:01:57  418k

    
 58 80.2M   58 47.0M    0     0   295k      0  0:04:37  0:02:42  0:01:55  455k

    
 58 80.2M   58 47.3M    0     0   295k      0  0:04:38  0:02:44  0:01:54  384k

    
 59 80.2M   59 47.6M    0     0   296k      0  0:04:37  0:02:44  0:01:53  380k

    
 59 80.2M   59 48.1M    0     0   296k      0  0:04:36  0:02:46  0:01:50  371k

    
 60 80.2M   60 48.4M    0     0   297k      0  0:04:36  0:02:46  0:01:50  387k

    
 60 80.2M   60 48.9M    0     0   298k      0  0:04:35  0:02:48  0:01:47  370k

    
 61 80.2M   61 49.0M    0     0   297k      0  0:04:36  0:02:48  0:01:48  354k

    
 61 80.2M   61 49.3M    0     0   297k      0  0:04:36  0:02:49  0:01:47  344k

    
 62 80.2M   62 49.7M    0     0   298k      0  0:04:35  0:02:50  0:01:45  341k

    
 62 80.2M   62 50.1M    0     0   298k      0  0:04:34  0:02:51  0:01:43  351k

    
 63 80.2M   63 50.5M    0     0   299k      0  0:04:34  0:02:53  0:01:41  348k

    
 63 80.2M   63 51.0M    0     0   300k      0  0:04:33  0:02:54  0:01:39  394k

    
 63 80.2M   63 51.3M    0     0   300k      0  0:04:33  0:02:54  0:01:39  393k

    
 64 80.2M   64 51.7M    0     0   301k      0  0:04:32  0:02:55  0:01:37  414k

    
 64 80.2M   64 52.1M    0     0   301k      0  0:04:32  0:02:56  0:01:36  397k

    
 65 80.2M   65 52.6M    0     0   302k      0  0:04:31  0:02:57  0:01:34  419k

    
 66 80.2M   66 52.9M    0     0   303k      0  0:04:30  0:02:58  0:01:32  418k

    
 66 80.2M   66 53.3M    0     0   303k      0  0:04:30  0:03:00  0:01:30  415k

    
 66 80.2M   66 53.6M    0     0   303k      0  0:04:30  0:03:00  0:01:30  388k

    
 67 80.2M   67 54.1M    0     0   304k      0  0:04:30  0:03:02  0:01:28  388k

    
 67 80.2M   67 54.4M    0     0   304k      0  0:04:29  0:03:02  0:01:27  376k

    
 68 80.2M   68 54.9M    0     0   306k      0  0:04:28  0:03:03  0:01:25  411k

    
 68 80.2M   68 55.2M    0     0   306k      0  0:04:28  0:03:04  0:01:24  397k

    
 69 80.2M   69 55.6M    0     0   306k      0  0:04:27  0:03:05  0:01:22  413k

    
 70 80.2M   70 56.3M    0     0   307k      0  0:04:26  0:03:07  0:01:19  429k

    
 70 80.2M   70 56.4M    0     0   307k      0  0:04:26  0:03:07  0:01:19  423k

    
 70 80.2M   70 56.8M    0     0   308k      0  0:04:26  0:03:08  0:01:18  378k

    
 71 80.2M   71 57.1M    0     0   308k      0  0:04:26  0:03:09  0:01:17  402k

    
 71 80.2M   71 57.7M    0     0   309k      0  0:04:25  0:03:10  0:01:15  412k

    
 72 80.2M   72 58.0M    0     0   309k      0  0:04:25  0:03:11  0:01:14  396k

    
 72 80.2M   72 58.4M    0     0   309k      0  0:04:25  0:03:13  0:01:12  383k

    
 73 80.2M   73 58.7M    0     0   310k      0  0:04:25  0:03:13  0:01:12  380k

    
 73 80.2M   73 59.1M    0     0   310k      0  0:04:24  0:03:15  0:01:09  378k

    
 74 80.2M   74 59.4M    0     0   310k      0  0:04:24  0:03:15  0:01:09  363k

    
 74 80.2M   74 59.9M    0     0   311k      0  0:04:23  0:03:16  0:01:07  388k

    
 75 80.2M   75 60.2M    0     0   311k      0  0:04:23  0:03:17  0:01:06  379k

    
 75 80.2M   75 60.7M    0     0   312k      0  0:04:23  0:03:19  0:01:04  395k

    
 76 80.2M   76 61.0M    0     0   312k      0  0:04:22  0:03:19  0:01:03  405k

    
 76 80.2M   76 61.4M    0     0   313k      0  0:04:22  0:03:20  0:01:02  400k

    
 77 80.2M   77 61.8M    0     0   313k      0  0:04:22  0:03:22  0:01:00  365k

    
 77 80.2M   77 62.1M    0     0   313k      0  0:04:22  0:03:22  0:01:00  385k

    
 78 80.2M   78 62.7M    0     0   313k      0  0:04:21  0:03:24  0:00:57  374k

    
 78 80.2M   78 62.9M    0     0   314k      0  0:04:21  0:03:24  0:00:57  385k

    
 78 80.2M   78 63.3M    0     0   314k      0  0:04:21  0:03:26  0:00:55  379k

    
 79 80.2M   79 63.5M    0     0   314k      0  0:04:21  0:03:26  0:00:55  375k

    
 79 80.2M   79 64.0M    0     0   315k      0  0:04:20  0:03:28  0:00:52  377k

    
 80 80.2M   80 64.3M    0     0   315k      0  0:04:20  0:03:28  0:00:52  382k

    
 80 80.2M   80 64.7M    0     0   316k      0  0:04:19  0:03:29  0:00:50  381k

    
 81 80.2M   81 65.1M    0     0   316k      0  0:04:19  0:03:31  0:00:48  374k

    
 81 80.2M   81 65.4M    0     0   316k      0  0:04:19  0:03:31  0:00:48  396k

    
 82 80.2M   82 66.0M    0     0   317k      0  0:04:19  0:03:33  0:00:46  401k

    
 82 80.2M   82 66

<div class="k-default-codeblock">
```
.2M    0     0   317k      0  0:04:19  0:03:33  0:00:46  392k

```
</div>
    
 82 80.2M   82 66.5M    0     0   317k      0  0:04:18  0:03:34  0:00:44  374k

    
 83 80.2M   83 67.2M    0     0   318k      0  0:04:18  0:03:36  0:00:42  397k

    
 83 80.2M   83 67.3M    0     0   318k      0  0:04:18  0:03:36  0:00:42  389k

    
 84 80.2M   84 67.6M    0     0   317k      0  0:04:18  0:03:37  0:00:41  350k

    
 84 80.2M   84 67.8M    0     0   317k      0  0:04:18  0:03:38  0:00:40  331k

    
 85 80.2M   85 68.2M    0     0   317k      0  0:04:18  0:03:40  0:00:38  327k

    
 85 80.2M   85 68.5M    0     0   317k      0  0:04:18  0:03:40  0:00:38  290k

    
 85 80.2M   85 68.8M    0     0   317k      0  0:04:18  0:03:42  0:00:36  294k

    
 86 80.2M   86 69.1M    0     0   317k      0  0:04:18  0:03:42  0:00:36  313k

    
 86 80.2M   86 69.5M    0     0   317k      0  0:04:18  0:03:44  0:00:34  328k

    
 86 80.2M   86 69.7M    0     0   317k      0  0:04:18  0:03:44  0:00:34  322k

    
 87 80.2M   87 70.2M    0     0   317k      0  0:04:18  0:03:46  0:00:32  330k

    
 87 80.2M   87 70.5M    0     0   318k      0  0:04:18  0:03:46  0:00:32  344k

    
 88 80.2M   88 70.8M    0     0   318k      0  0:04:17  0:03:47  0:00:30  361k

    
 88 80.2M   88 71.0M    0     0   317k      0  0:04:18  0:03:49  0:00:29  322k

    
 88 80.2M   88 71.2M    0     0   317k      0  0:04:18  0:03:49  0:00:29  312k

    
 89 80.2M   89 71.6M    0     0   317k      0  0:04:18  0:03:51  0:00:27  292k

    
 89 80.2M   89 71.8M    0     0   317k      0  0:04:19  0:03:51  0:00:28  266k

    
 89 80.2M   89 72.1M    0     0   317k      0  0:04:19  0:03:53  0:00:26  249k

    
 90 80.2M   90 72.4M    0     0   317k      0  0:04:19  0:03:53  0:00:26  282k

    
 90 80.2M   90 72.7M    0     0   316k      0  0:04:19  0:03:55  0:00:24  288k

    
 91 80.2M   91 73.0M    0     0   316k      0  0:04:19  0:03:55  0:00:24  295k

    
 91 80.2M   91 73.4M    0     0   317k      0  0:04:18  0:03:56  0:00:22  328k

    
 92 80.2M   92 73.9M    0     0   317k      0  0:04:18  0:03:58  0:00:20  335k

    
 92 80.2M   92 74.2M    0     0   318k      0  0:04:17  0:03:58  0:00:19  383k

    
 92 80.2M   92 74.4M    0     0   317k      0  0:04:18  0:04:00  0:00:18  340k

    
 93 80.2M   93 74.9M    0     0   318k      0  0:04:17  0:04:00  0:00:17  395k

    
 93 80.2M   93 75.3M    0     0   318k      0  0:04:17  0:04:02  0:00:15  367k

    
 94 80.2M   94 75.5M    0     0   318k      0  0:04:17  0:04:02  0:00:15  378k

    
 94 80.2M   94 75.9M    0     0   319k      0  0:04:17  0:04:03  0:00:14  353k

    
 95 80.2M   95 76.3M    0     0   319k      0  0:04:17  0:04:05  0:00:12  395k

    
 95 80.2M   95 76.6M    0     0   319k      0  0:04:17  0:04:05  0:00:12  348k

    
 96 80.2M   96 77.0M    0     0   319k      0  0:04:17  0:04:07  0:00:10  364k

    
 96 80.2M   96 77.4M    0     0   320k      0  0:04:16  0:04:07  0:00:09  390k

    
 96 80.2M   96 77.6M    0     0   318k      0  0:04:17  0:04:09  0:00:08  308k

    
 97 80.2M   97 78.1M    0     0   319k      0  0:04:16  0:04:10  0:00:06  365k

    
 97 80.2M   97 78.4M    0     0   320k      0  0:04:16  0:04:10  0:00:06  362k

    
 98 80.2M   98 78.9M    0     0   320k      0  0:04:16  0:04:12  0:00:04  368k

    
 9

<div class="k-default-codeblock">
```
8 80.2M   98 79.2M    0     0   320k      0  0:04:16  0:04:12  0:00:04  360k

```
</div>
    
 99 80.2M   99 79.5M    0     0   321k      0  0:04:15  0:04:13  0:00:02  442k

    
 99 80.2M   99 80.1M    0     0   321k      0  0:04:15  0:04:15 --:--:--  410k
100 80.2M  100 80.2M    0     0   322k      0  0:04:15  0:04:15 --:--:--  446k



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

    
  1/13 ━[37m━━━━━━━━━━━━━━━━━━━  2:15 11s/step - loss: 10.3156

<div class="k-default-codeblock">
```

```
</div>
  2/13 ━━━[37m━━━━━━━━━━━━━━━━━  2:15 12s/step - loss: 10.3044

<div class="k-default-codeblock">
```

```
</div>
  3/13 ━━━━[37m━━━━━━━━━━━━━━━━  2:02 12s/step - loss: 10.2915

<div class="k-default-codeblock">
```

```
</div>
  4/13 ━━━━━━[37m━━━━━━━━━━━━━━  1:50 12s/step - loss: 10.2769

<div class="k-default-codeblock">
```

```
</div>
  5/13 ━━━━━━━[37m━━━━━━━━━━━━━  1:39 12s/step - loss: 10.2610

<div class="k-default-codeblock">
```

```
</div>
  6/13 ━━━━━━━━━[37m━━━━━━━━━━━  1:26 12s/step - loss: 10.2437

<div class="k-default-codeblock">
```

```
</div>
  7/13 ━━━━━━━━━━[37m━━━━━━━━━━  1:13 12s/step - loss: 10.2260

<div class="k-default-codeblock">
```

```
</div>
  8/13 ━━━━━━━━━━━━[37m━━━━━━━━  1:01 12s/step - loss: 10.2068

<div class="k-default-codeblock">
```

```
</div>
  9/13 ━━━━━━━━━━━━━[37m━━━━━━━  49s 12s/step - loss: 10.1861 

<div class="k-default-codeblock">
```

```
</div>
 10/13 ━━━━━━━━━━━━━━━[37m━━━━━  38s 13s/step - loss: 10.1633

<div class="k-default-codeblock">
```

```
</div>
 11/13 ━━━━━━━━━━━━━━━━[37m━━━━  26s 13s/step - loss: 10.1398

<div class="k-default-codeblock">
```

```
</div>
 12/13 ━━━━━━━━━━━━━━━━━━[37m━━  13s 13s/step - loss: 10.1148

<div class="k-default-codeblock">
```

```
</div>
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 13s/step - loss: 10.0910 

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 178ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 179ms/step


<div class="k-default-codeblock">
```
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'a',
 'prediction': 'i have watched this a and it was awesome',
 'probability': 0.0006833174}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'to',
 'prediction': 'i have watched this to and it was awesome',
 'probability': 0.00058719254}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'it',
 'prediction': 'i have watched this it and it was awesome',
 'probability': 0.00056189025}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'in',
 'prediction': 'i have watched this in and it was awesome',
 'probability': 0.0005278979}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'of',
 'prediction': 'i have watched this of and it was awesome',
 'probability': 0.00050748244}

```
</div>
 13/13 ━━━━━━━━━━━━━━━━━━━━ 164s 13s/step - loss: 10.0707


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



    
 1/7 ━━[37m━━━━━━━━━━━━━━━━━━  0s 130ms/step - accuracy: 0.3125 - loss: 0.8427

<div class="k-default-codeblock">
```

```
</div>
 2/7 ━━━━━[37m━━━━━━━━━━━━━━━  0s 95ms/step - accuracy: 0.3359 - loss: 0.8267 

<div class="k-default-codeblock">
```

```
</div>
 3/7 ━━━━━━━━[37m━━━━━━━━━━━━  0s 107ms/step - accuracy: 0.3524 - loss: 0.8178

<div class="k-default-codeblock">
```

```
</div>
 4/7 ━━━━━━━━━━━[37m━━━━━━━━━  0s 107ms/step - accuracy: 0.3678 - loss: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 5/7 ━━━━━━━━━━━━━━[37m━━━━━━  0s 113ms/step - accuracy: 0.3843 - loss: 0.7989

<div class="k-default-codeblock">
```

```
</div>
 6/7 ━━━━━━━━━━━━━━━━━[37m━━━  0s 110ms/step - accuracy: 0.3984 - loss: 0.7904

<div class="k-default-codeblock">
```

```
</div>
 7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 210ms/step - accuracy: 0.4163 - loss: 0.7792 - val_accuracy: 0.5250 - val_loss: 0.7072


    
 1/7 ━━[37m━━━━━━━━━━━━━━━━━━  2s 375ms/step - accuracy: 0.4688 - loss: 0.7607

<div class="k-default-codeblock">
```

```
</div>
 2/7 ━━━━━[37m━━━━━━━━━━━━━━━  1s 377ms/step - accuracy: 0.4766 - loss: 0.7908

<div class="k-default-codeblock">
```

```
</div>
 3/7 ━━━━━━━━[37m━━━━━━━━━━━━  1s 373ms/step - accuracy: 0.4740 - loss: 0.7972

<div class="k-default-codeblock">
```

```
</div>
 4/7 ━━━━━━━━━━━[37m━━━━━━━━━  1s 395ms/step - accuracy: 0.4805 - loss: 0.7913

<div class="k-default-codeblock">
```

```
</div>
 5/7 ━━━━━━━━━━━━━━[37m━━━━━━  0s 400ms/step - accuracy: 0.4806 - loss: 0.7965

<div class="k-default-codeblock">
```

```
</div>
 6/7 ━━━━━━━━━━━━━━━━━[37m━━━  0s 396ms/step - accuracy: 0.4804 - loss: 0.8007

<div class="k-default-codeblock">
```

```
</div>
 7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 360ms/step - accuracy: 0.4810 - loss: 0.8027

<div class="k-default-codeblock">
```

```
</div>
 7/7 ━━━━━━━━━━━━━━━━━━━━ 3s 480ms/step - accuracy: 0.4815 - loss: 0.8043 - val_accuracy: 0.5300 - val_loss: 0.6930





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x72eec8852350>

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
    inputs = classifer_model.inputs
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

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:252: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: ['keras_tensor_48']
Received: inputs=Tensor(shape=torch.Size([32, 256]))
  warnings.warn(msg)

```
</div>
    
 1/7 ━━[37m━━━━━━━━━━━━━━━━━━  0s 102ms/step - accuracy: 0.4688 - loss: 0.7020

<div class="k-default-codeblock">
```

```
</div>
 2/7 ━━━━━[37m━━━━━━━━━━━━━━━  0s 119ms/step - accuracy: 0.4766 - loss: 0.7083

<div class="k-default-codeblock">
```

```
</div>
 3/7 ━━━━━━━━[37m━━━━━━━━━━━━  0s 110ms/step - accuracy: 0.4774 - loss: 0.7089

<div class="k-default-codeblock">
```

```
</div>
 4/7 ━━━━━━━━━━━[37m━━━━━━━━━  0s 107ms/step - accuracy: 0.4811 - loss: 0.7079

<div class="k-default-codeblock">
```

```
</div>
 5/7 ━━━━━━━━━━━━━━[37m━━━━━━  0s 109ms/step - accuracy: 0.4911 - loss: 0.7047

<div class="k-default-codeblock">
```

```
</div>
 6/7 ━━━━━━━━━━━━━━━━━[37m━━━  0s 110ms/step - accuracy: 0.4978 - loss: 0.7026

<div class="k-default-codeblock">
```

```
</div>
 7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 97ms/step - accuracy: 0.5059 - loss: 0.7002 


<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:252: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: ['keras_tensor_48']
Received: inputs=Tensor(shape=torch.Size([8, 256]))
  warnings.warn(msg)

[0.6929563283920288, 0.5299999713897705]

```
</div>