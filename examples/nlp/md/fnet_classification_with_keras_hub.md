# Text Classification using FNet

**Author:** [Abheesht Sharma](https://github.com/abheesht17/)<br>
**Date created:** 2022/06/01<br>
**Last modified:** 2025/01/06<br>
**Description:** Text Classification on the IMDb Dataset using `keras_hub.layers.FNetEncoder` layer.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/fnet_classification_with_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/fnet_classification_with_keras_hub.py)



---
## Introduction

In this example, we will demonstrate the ability of FNet to achieve comparable
results with a vanilla Transformer model on the text classification task.
We will be using the IMDb dataset, which is a
collection of movie reviews labelled either positive or negative (sentiment
analysis).

To build the tokenizer, model, etc., we will use components from
[KerasHub](https://github.com/keras-team/keras-hub). KerasHub makes life easier
for people who want to build NLP pipelines! :)

### Model

Transformer-based language models (LMs) such as BERT, RoBERTa, XLNet, etc. have
demonstrated the effectiveness of the self-attention mechanism for computing
rich embeddings for input text. However, the self-attention mechanism is an
expensive operation, with a time complexity of `O(n^2)`, where `n` is the number
of tokens in the input. Hence, there has been an effort to reduce the time
complexity of the self-attention mechanism and improve performance without
sacrificing the quality of results.

In 2020, a paper titled
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
replaced the self-attention layer in BERT with a simple Fourier Transform layer
for "token mixing". This resulted in comparable accuracy and a speed-up during
training. In particular, a couple of points from the paper stand out:

* The authors claim that FNet is 80% faster than BERT on GPUs and 70% faster on
TPUs. The reason for this speed-up is two-fold: a) the Fourier Transform layer
is unparametrized, it does not have any parameters, and b) the authors use Fast
Fourier Transform (FFT); this reduces the time complexity from `O(n^2)`
(in the case of self-attention) to `O(n log n)`.
* FNet manages to achieve 92-97% of the accuracy of BERT on the GLUE benchmark.

---
## Setup

Before we start with the implementation, let's import all the necessary packages.


```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```

    
<div class="k-default-codeblock">
```
 [[34;49mnotice[1;39;49m][39;49m A new release of pip is available: [31;49m24.0[39;49m -> [32;49m24.3.1
 [[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip

```
</div>
    
```python
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import keras_hub
import keras

keras.utils.set_random_seed(42)
```
<div class="k-default-codeblock">
```
 [[34;49mnotice[1;39;49m][39;49m A new release of pip is available: [31;49m24.0[39;49m -> [32;49m24.3.1
 [[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip

```
</div>
Let's also define our hyperparameters.


```python
BATCH_SIZE = 64
EPOCHS = 1  # maybe adjusted as desired
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512
```

---
## Loading the dataset

First, let's download the IMDB dataset and extract it.


```python
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf aclImdb_v1.tar.gz
```

<div class="k-default-codeblock">
```
--2025-01-06 21:32:09--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... 

connected.
HTTP request sent, awaiting response... 

200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘aclImdb_v1.tar.gz’
```
</div>
    
    
aclImdb_v1.tar.gz     0%[                    ]       0  --.-KB/s               

    
aclImdb_v1.tar.gz     0%[                    ]  22.19K  36.1KB/s               

    
aclImdb_v1.tar.gz     0%[                    ]  44.69K  36.4KB/s               

    
aclImdb_v1.tar.gz     0%[                    ]  64.38K  34.9KB/s               

    
aclImdb_v1.tar.gz     0%[                    ]  79.85K  32.5KB/s               

    
aclImdb_v1.tar.gz     0%[                    ] 119.23K  38.7KB/s    eta 35m 17s

    
aclImdb_v1.tar.gz     0%[                    ] 158.60K  43.0KB/s    eta 35m 17s

    
aclImdb_v1.tar.gz     0%[                    ] 219.07K  54.9KB/s    eta 35m 17s

    
aclImdb_v1.tar.gz     0%[                    ] 271.10K  58.8KB/s    eta 23m 12s

    
aclImdb_v1.tar.gz     0%[                    ] 330.16K  63.2KB/s    eta 23m 12s

    
aclImdb_v1.tar.gz     0%[                    ] 396.26K  67.9KB/s    eta 20m 4s 

    
aclImdb_v1.tar.gz     0%[                    ] 456.73K  70.8KB/s    eta 20m 4s 

    
aclImdb_v1.tar.gz     0%[                    ] 528.44K  74.8KB/s    eta 18m 11s

    
aclImdb_v1.tar.gz     0%[                    ] 601.57K  78.3KB/s    eta 18m 11s

    
aclImdb_v1.tar.gz     0%[                    ] 677.51K  81.7KB/s    eta 16m 37s

    
aclImdb_v1.tar.gz     0%[                    ] 756.26K  84.9KB/s    eta 16m 37s

    
aclImdb_v1.tar.gz     1%[                    ] 840.63K  88.2KB/s    eta 15m 21s

    
aclImdb_v1.tar.gz     1%[                    ] 930.63K  91.8KB/s    eta 15m 21s

    
aclImdb_v1.tar.gz     1%[                    ]   1023K  95.2KB/s    eta 14m 12s

    
aclImdb_v1.tar.gz     1%[                    ]   1.09M  98.7KB/s    eta 14m 12s

    
aclImdb_v1.tar.gz     1%[                    ]   1.18M   101KB/s    eta 13m 19s

    
aclImdb_v1.tar.gz     1%[                    ]   1.30M   109KB/s    eta 13m 19s

    
aclImdb_v1.tar.gz     1%[                    ]   1.41M   117KB/s    eta 12m 19s

    
aclImdb_v1.tar.gz     1%[                    ]   1.51M   124KB/s    eta 12m 19s

    
aclImdb_v1.tar.gz     2%[                    ]   1.65M   134KB/s    eta 11m 29s

    
aclImdb_v1.tar.gz     2%[                    ]   1.77M   142KB/s    eta 11m 29s

    
aclImdb_v1.tar.gz     2%[                    ]   1.91M   150KB/s    eta 10m 42s

    
aclImdb_v1.tar.gz     2%[                    ]   2.05M   153KB/s    eta 10m 42s

    
aclImdb_v1.tar.gz     2%[                    ]   2.21M   162KB/s    eta 9m 58s 

    
aclImdb_v1.tar.gz     2%[                    ]   2.36M   170KB/s    eta 9m 58s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.52M   182KB/s    eta 9m 58s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.71M   193KB/s    eta 8m 48s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.87M   191KB/s    eta 8m 48s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.92M   195KB/s    eta 8m 41s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.97M   194KB/s    eta 8m 41s 

    
aclImdb_v1.tar.gz     3%[                    ]   3.12M   200KB/s    eta 8m 37s 

    
aclImdb_v1.tar.gz     4%[                    ]   3.27M   206KB/s    eta 8m 37s 

    
aclImdb_v1.tar.gz     4%[                    ]   3.42M   212KB/s    eta 8m 16s 

    
aclImdb_v1.tar.gz     4%[                    ]   3.61M   220KB/s    eta 8m 16s 

    
aclImdb_v1.tar.gz     4%[                    ]   3.77M   216KB/s    eta 8m 1s  

    
aclImdb_v1.tar.gz     4%[                    ]   3.83M   220KB/s    eta 8m 1s  

    
aclImdb_v1.tar.gz     5%[>                   ]   4.01M   226KB/s    eta 8m 1s  

    
aclImdb_v1.tar.gz     5%[>                   ]   4.21M   235KB/s    eta 7m 34s 

    
aclImdb_v1.tar.gz     5%[>                   ]   4.35M   237KB/s    eta 7m 34s 

    
aclImdb_v1.tar.gz     5%[>                   ]   4.48M   238KB/s    eta 7m 27s 

    
aclImdb_v1.tar.gz     5%[>                   ]   4.64M   242KB/s    eta 7m 27s 

    
aclImdb_v1.tar.gz     5%[>                   ]   4.78M   243KB/s    eta 7m 16s 

    
aclImdb_v1.tar.gz     6%[>                   ]   4.95M   251KB/s    eta 7m 16s 

    
aclImdb_v1.tar.gz     6%[>                   ]   5.11M   254KB/s    eta 7m 16s 

    
aclImdb_v1.tar.gz     6%[>                   ]   5.26M   255KB/s    eta 6m 56s 

    
aclImdb_v1.tar.gz     6%[>                   ]   5.42M   255KB/s    eta 6m 56s 

    
aclImdb_v1.tar.gz     6%[>                   ]   5.57M   254KB/s    eta 6m 48s 

    
aclImdb_v1.tar.gz     7%[>                   ]   5.73M   249KB/s    eta 6m 48s 

    
aclImdb_v1.tar.gz     7%[>                   ]   5.89M   247KB/s    eta 6m 40s 

    
aclImdb_v1.tar.gz     7%[>                   ]   6.05M   255KB/s    eta 6m 40s 

    
aclImdb_v1.tar.gz     7%[>                   ]   6.23M   250KB/s    eta 6m 35s 

    
aclImdb_v1.tar.gz     7%[>                   ]   6.29M   257KB/s    eta 6m 35s 

    
aclImdb_v1.tar.gz     7%[>                   ]   6.41M   256KB/s    eta 6m 35s 

    
aclImdb_v1.tar.gz     8%[>                   ]   6.67M   264KB/s    eta 6m 23s 

    
aclImdb_v1.tar.gz     8%[>                   ]   6.80M   261KB/s    eta 6m 23s 

    
aclImdb_v1.tar.gz     8%[>                   ]   6.96M   260KB/s    eta 6m 19s 

    
aclImdb_v1.tar.gz     8%[>                   ]   7.11M   266KB/s    eta 6m 19s 

    
aclImdb_v1.tar.gz     9%[>                   ]   7.28M   265KB/s    eta 6m 13s 

    
aclImdb_v1.tar.gz     9%[>                   ]   7.45M   263KB/s    eta 6m 13s 

    
aclImdb_v1.tar.gz     9%[>                   ]   7.61M   256KB/s    eta 6m 10s 

    
aclImdb_v1.tar.gz     9%[>                   ]   7.67M   258KB/s    eta 6m 10s 

    
aclImdb_v1.tar.gz     9%[>                   ]   7.86M   262KB/s    eta 6m 10s 

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.06M   265KB/s    eta 6m 0s  

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.19M   265KB/s    eta 6m 0s  

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.32M   257KB/s    eta 5m 58s 

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.48M   256KB/s    eta 5m 58s 

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.63M   255KB/s    eta 5m 54s 

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.79M   255KB/s    eta 5m 54s 

    
aclImdb_v1.tar.gz    11%[=>                  ]   8.95M   256KB/s    eta 5m 50s 

    
aclImdb_v1.tar.gz    11%[=>                  ]   9.10M   255KB/s    eta 5m 50s 

    
aclImdb_v1.tar.gz    11%[=>                  ]   9.25M   255KB/s    eta 5m 46s 

    
aclImdb_v1.tar.gz    11%[=>                  ]   9.41M   254KB/s    eta 5m 46s 

    
aclImdb_v1.tar.gz    11%[=>                  ]   9.58M   261KB/s    eta 5m 42s 

    
aclImdb_v1.tar.gz    12%[=>                  ]   9.75M   265KB/s    eta 5m 42s 

    
aclImdb_v1.tar.gz    12%[=>                  ]   9.92M   258KB/s    eta 5m 37s 

    
aclImdb_v1.tar.gz    12%[=>                  ]  10.08M   252KB/s    eta 5m 37s 

    
aclImdb_v1.tar.gz    12%[=>                  ]  10.14M   254KB/s    eta 5m 37s 

    
aclImdb_v1.tar.gz    12%[=>                  ]  10.27M   257KB/s    eta 5m 37s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.53M   265KB/s    eta 5m 37s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.66M   268KB/s    eta 5m 27s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.68M   256KB/s    eta 5m 27s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.70M   252KB/s    eta 5m 33s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.74M   240KB/s    eta 5m 33s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.77M   226KB/s    eta 5m 39s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.82M   219KB/s    eta 5m 39s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.86M   211KB/s    eta 5m 43s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.91M   202KB/s    eta 5m 43s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.95M   194KB/s    eta 5m 48s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  11.02M   186KB/s    eta 5m 48s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  11.08M   178KB/s    eta 5m 51s 

    
aclImdb_v1.tar.gz    13%[=>                  ]  11.17M   172KB/s    eta 5m 51s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.25M   166KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.35M   162KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.45M   156KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.56M   151KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.67M   146KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.79M   144KB/s    eta 5m 52s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  11.90M   140KB/s    eta 5m 51s 

    
aclImdb_v1.tar.gz    14%[=>                  ]  12.02M   127KB/s    eta 5m 51s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.14M   124KB/s    eta 5m 50s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.26M   135KB/s    eta 5m 50s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.38M   143KB/s    eta 5m 50s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.50M   151KB/s    eta 5m 46s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.63M   159KB/s    eta 5m 46s 

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.76M   166KB/s    eta 5m 44s 

    
aclImdb_v1.tar.gz    16%[==>                 ]  12.91M   175KB/s    eta 5m 44s 

    
aclImdb_v1.tar.gz    16%[==>                 ]  13.06M   184KB/s    eta 5m 41s 

    
aclImdb_v1.tar.gz    16%[==>                 ]  13.21M   193KB/s    eta 5m 41s 

    
aclImdb_v1.tar.gz    16%[==>                 ]  13.37M   201KB/s    eta 5m 38s 

    
aclImdb_v1.tar.gz    16%[==>                 ]  13.55M   211KB/s    eta 5m 38s 

    
aclImdb_v1.tar.gz    17%[==>                 ]  13.71M   208KB/s    eta 5m 36s 

    
aclImdb_v1.tar.gz    17%[==>                 ]  13.77M   212KB/s    eta 5m 36s 

    
aclImdb_v1.tar.gz    17%[==>                 ]  13.90M   216KB/s    eta 5m 36s 

    
aclImdb_v1.tar.gz    17%[==>                 ]  14.16M   229KB/s    eta 5m 30s 

    
aclImdb_v1.tar.gz    17%[==>                 ]  14.30M   232KB/s    eta 5m 30s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  14.45M   235KB/s    eta 5m 27s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  14.61M   239KB/s    eta 5m 27s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  14.77M   243KB/s    eta 5m 24s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  14.94M   247KB/s    eta 5m 24s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  15.10M   241KB/s    eta 5m 22s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  15.16M   244KB/s    eta 5m 22s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  15.34M   247KB/s    eta 5m 22s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  15.54M   249KB/s    eta 5m 17s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  15.68M   250KB/s    eta 5m 17s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  15.82M   251KB/s    eta 5m 15s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  15.97M   253KB/s    eta 5m 15s 

    
aclImdb_v1.tar.gz    20%[===>                ]  16.12M   255KB/s    eta 5m 13s 

    
aclImdb_v1.tar.gz    20%[===>                ]  16.28M   256KB/s    eta 5m 13s 

    
aclImdb_v1.tar.gz    20%[===>                ]  16.43M   256KB/s    eta 5m 10s 

    
aclImdb_v1.tar.gz    20%[===>                ]  16.59M   256KB/s    eta 5m 10s 

    
aclImdb_v1.tar.gz    20%[===>                ]  16.75M   256KB/s    eta 5m 7s  

    
aclImdb_v1.tar.gz    21%[===>                ]  16.89M   253KB/s    eta 5m 7s  

    
aclImdb_v1.tar.gz    21%[===>                ]  17.07M   262KB/s    eta 5m 5s  

    
aclImdb_v1.tar.gz    21%[===>                ]  17.23M   265KB/s    eta 5m 5s  

    
aclImdb_v1.tar.gz    21%[===>                ]  17.41M   257KB/s    eta 5m 2s  

    
aclImdb_v1.tar.gz    21%[===>                ]  17.57M   251KB/s    eta 5m 2s  

    
aclImdb_v1.tar.gz    21%[===>                ]  17.63M   252KB/s    eta 5m 1s  

    
aclImdb_v1.tar.gz    22%[===>                ]  17.81M   255KB/s    eta 5m 1s  

    
aclImdb_v1.tar.gz    22%[===>                ]  18.02M   258KB/s    eta 4m 57s 

    
aclImdb_v1.tar.gz    22%[===>                ]  18.16M   257KB/s    eta 4m 57s 

    
aclImdb_v1.tar.gz    22%[===>                ]  18.31M   255KB/s    eta 4m 55s 

    
aclImdb_v1.tar.gz    23%[===>                ]  18.45M   261KB/s    eta 4m 55s 

    
aclImdb_v1.tar.gz    23%[===>                ]  18.62M   260KB/s    eta 4m 53s 

    
aclImdb_v1.tar.gz    23%[===>                ]  18.79M   258KB/s    eta 4m 53s 

    
aclImdb_v1.tar.gz    23%[===>                ]  18.95M   252KB/s    eta 4m 51s 

    
aclImdb_v1.tar.gz    23%[===>                ]  19.01M   253KB/s    eta 4m 51s 

    
aclImdb_v1.tar.gz    23%[===>                ]  19.20M   256KB/s    eta 4m 51s 

    
aclImdb_v1.tar.gz    24%[===>                ]  19.40M   260KB/s    eta 4m 47s 

    
aclImdb_v1.tar.gz    24%[===>                ]  19.53M   258KB/s    eta 4m 47s 

    
aclImdb_v1.tar.gz    24%[===>                ]  19.68M   257KB/s    eta 4m 46s 

    
aclImdb_v1.tar.gz    24%[===>                ]  19.82M   257KB/s    eta 4m 46s 

    
aclImdb_v1.tar.gz    24%[===>                ]  19.97M   256KB/s    eta 4m 44s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.13M   257KB/s    eta 4m 44s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.28M   256KB/s    eta 4m 41s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.44M   260KB/s    eta 4m 41s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.60M   261KB/s    eta 4m 41s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.76M   263KB/s    eta 4m 37s 

    
aclImdb_v1.tar.gz    26%[====>               ]  20.92M   274KB/s    eta 4m 37s 

    
aclImdb_v1.tar.gz    26%[====>               ]  21.08M   273KB/s    eta 4m 34s 

    
aclImdb_v1.tar.gz    26%[====>               ]  21.25M   269KB/s    eta 4m 34s 

    
aclImdb_v1.tar.gz    26%[====>               ]  21.42M   272KB/s    eta 4m 32s 

    
aclImdb_v1.tar.gz    26%[====>               ]  21.58M   265KB/s    eta 4m 32s 

    
aclImdb_v1.tar.gz    26%[====>               ]  21.65M   265KB/s    eta 4m 31s 

    
aclImdb_v1.tar.gz    27%[====>               ]  21.83M   268KB/s    eta 4m 31s 

    
aclImdb_v1.tar.gz    27%[====>               ]  22.04M   271KB/s    eta 4m 28s 

    
aclImdb_v1.tar.gz    27%[====>               ]  22.17M   269KB/s    eta 4m 28s 

    
aclImdb_v1.tar.gz    27%[====>               ]  22.32M   276KB/s    eta 4m 26s 

    
aclImdb_v1.tar.gz    28%[====>               ]  22.47M   273KB/s    eta 4m 26s 

    
aclImdb_v1.tar.gz    28%[====>               ]  22.65M   271KB/s    eta 4m 24s 

    
aclImdb_v1.tar.gz    28%[====>               ]  22.83M   274KB/s    eta 4m 24s 

    
aclImdb_v1.tar.gz    28%[====>               ]  22.99M   267KB/s    eta 4m 23s 

    
aclImdb_v1.tar.gz    28%[====>               ]  23.05M   268KB/s    eta 4m 23s 

    
aclImdb_v1.tar.gz    28%[====>               ]  23.23M   271KB/s    eta 4m 23s 

    
aclImdb_v1.tar.gz    29%[====>               ]  23.44M   275KB/s    eta 4m 19s 

    
aclImdb_v1.tar.gz    29%[====>               ]  23.57M   273KB/s    eta 4m 19s 

    
aclImdb_v1.tar.gz    29%[====>               ]  23.71M   272KB/s    eta 4m 18s 

    
aclImdb_v1.tar.gz    29%[====>               ]  23.85M   265KB/s    eta 4m 18s 

    
aclImdb_v1.tar.gz    29%[====>               ]  24.02M   264KB/s    eta 4m 16s 

    
aclImdb_v1.tar.gz    30%[=====>              ]  24.16M   261KB/s    eta 4m 16s 

    
aclImdb_v1.tar.gz    30%[=====>              ]  24.32M   258KB/s    eta 4m 14s 

    
aclImdb_v1.tar.gz    30%[=====>              ]  24.48M   257KB/s    eta 4m 14s 

    
aclImdb_v1.tar.gz    30%[=====>              ]  24.65M   258KB/s    eta 4m 12s 

    
aclImdb_v1.tar.gz    30%[=====>              ]  24.81M   257KB/s    eta 4m 12s 

    
aclImdb_v1.tar.gz    31%[=====>              ]  24.96M   263KB/s    eta 4m 10s 

    
aclImdb_v1.tar.gz    31%[=====>              ]  25.14M   263KB/s    eta 4m 10s 

    
aclImdb_v1.tar.gz    31%[=====>              ]  25.30M   260KB/s    eta 4m 8s  

    
aclImdb_v1.tar.gz    31%[=====>              ]  25.47M   254KB/s    eta 4m 8s  

    
aclImdb_v1.tar.gz    32%[=====>              ]  25.72M   259KB/s    eta 4m 6s  

    
aclImdb_v1.tar.gz    32%[=====>              ]  25.92M   263KB/s    eta 4m 6s  

    
aclImdb_v1.tar.gz    32%[=====>              ]  26.06M   260KB/s    eta 4m 4s  

    
aclImdb_v1.tar.gz    32%[=====>              ]  26.21M   258KB/s    eta 4m 4s  

    
aclImdb_v1.tar.gz    32%[=====>              ]  26.29M   265KB/s    eta 4m 4s  

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.50M   267KB/s    eta 4m 1s  

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.61M   266KB/s    eta 4m 1s  

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.72M   265KB/s    eta 4m 1s  

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.84M   263KB/s    eta 3m 59s 

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.97M   262KB/s    eta 3m 59s 

    
aclImdb_v1.tar.gz    33%[=====>              ]  27.10M   259KB/s    eta 3m 58s 

    
aclImdb_v1.tar.gz    33%[=====>              ]  27.23M   258KB/s    eta 3m 58s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.35M   248KB/s    eta 3m 58s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.41M   247KB/s    eta 3m 58s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.58M   248KB/s    eta 3m 58s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.66M   241KB/s    eta 3m 57s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.77M   237KB/s    eta 3m 57s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.87M   233KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.96M   226KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.08M   222KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.18M   221KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.29M   214KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.41M   207KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.52M   205KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.63M   202KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.76M   201KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    35%[======>             ]  28.87M   195KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.01M   197KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.16M   199KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.30M   206KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.32M   202KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.35M   194KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.39M   186KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.42M   183KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.48M   173KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.52M   169KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.58M   167KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.65M   163KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  29.72M   161KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  29.79M   157KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  29.89M   157KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  29.97M   155KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  30.09M   155KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    37%[======>             ]  30.20M   154KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    37%[======>             ]  30.31M   158KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    37%[======>             ]  30.44M   159KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.58M   160KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.59M   144KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.62M   133KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.65M   120KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.68M   118KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.72M   118KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.77M   119KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.82M   120KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.87M   120KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    38%[======>             ]  30.93M   122KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    38%[======>             ]  31.00M   121KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    38%[======>             ]  31.07M   121KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    38%[======>             ]  31.15M   122KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    38%[======>             ]  31.23M   123KB/s    eta 3m 56s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.33M   123KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.44M   125KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.54M   124KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.65M   125KB/s    eta 3m 55s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.77M   121KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.88M   120KB/s    eta 3m 54s 

    
aclImdb_v1.tar.gz    39%[======>             ]  31.99M   117KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.10M   126KB/s    eta 3m 53s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.22M   133KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.34M   141KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.46M   148KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.59M   156KB/s    eta 3m 51s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.73M   164KB/s    eta 3m 50s 

    
aclImdb_v1.tar.gz    40%[=======>            ]  32.88M   172KB/s    eta 3m 50s 

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.02M   183KB/s    eta 3m 48s 

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.19M   195KB/s    eta 3m 48s 

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.36M   206KB/s    eta 3m 48s 

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.53M   203KB/s    eta 3m 44s 

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.53M   203KB/s    eta 3m 44s 

    
aclImdb_v1.tar.gz    42%[=======>            ]  33.77M   216KB/s    eta 3m 44s 

    
aclImdb_v1.tar.gz    42%[=======>            ]  33.97M   227KB/s    eta 3m 41s 

    
aclImdb_v1.tar.gz    42%[=======>            ]  34.11M   230KB/s    eta 3m 41s 

    
aclImdb_v1.tar.gz    42%[=======>            ]  34.25M   233KB/s    eta 3m 40s 

    
aclImdb_v1.tar.gz    42%[=======>            ]  34.41M   237KB/s    eta 3m 40s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.56M   246KB/s    eta 3m 40s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.74M   252KB/s    eta 3m 37s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.90M   246KB/s    eta 3m 37s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.94M   247KB/s    eta 3m 36s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.97M   240KB/s    eta 3m 36s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  35.10M   242KB/s    eta 3m 36s 

    
aclImdb_v1.tar.gz    43%[=======>            ]  35.24M   243KB/s    eta 3m 36s 

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.37M   244KB/s    eta 3m 34s 

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.53M   247KB/s    eta 3m 34s 

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.68M   249KB/s    eta 3m 33s 

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.83M   250KB/s    eta 3m 33s 

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.98M   250KB/s    eta 3m 31s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.14M   247KB/s    eta 3m 31s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.29M   242KB/s    eta 3m 29s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.45M   239KB/s    eta 3m 29s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.60M   249KB/s    eta 3m 27s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.78M   245KB/s    eta 3m 27s 

    
aclImdb_v1.tar.gz    46%[========>           ]  36.94M   239KB/s    eta 3m 26s 

    
aclImdb_v1.tar.gz    46%[========>           ]  37.00M   235KB/s    eta 3m 26s 

    
aclImdb_v1.tar.gz    46%[========>           ]  37.18M   238KB/s    eta 3m 26s 

    
aclImdb_v1.tar.gz    46%[========>           ]  37.38M   243KB/s    eta 3m 23s 

    
aclImdb_v1.tar.gz    46%[========>           ]  37.52M   241KB/s    eta 3m 23s 

    
aclImdb_v1.tar.gz    46%[========>           ]  37.67M   235KB/s    eta 3m 21s 

    
aclImdb_v1.tar.gz    47%[========>           ]  37.82M   233KB/s    eta 3m 21s 

    
aclImdb_v1.tar.gz    47%[========>           ]  37.98M   241KB/s    eta 3m 20s 

    
aclImdb_v1.tar.gz    47%[========>           ]  38.14M   252KB/s    eta 3m 20s 

    
aclImdb_v1.tar.gz    47%[========>           ]  38.31M   254KB/s    eta 3m 18s 

    
aclImdb_v1.tar.gz    47%[========>           ]  38.48M   249KB/s    eta 3m 18s 

    
aclImdb_v1.tar.gz    47%[========>           ]  38.48M   246KB/s    eta 3m 17s 

    
aclImdb_v1.tar.gz    48%[========>           ]  38.72M   254KB/s    eta 3m 17s 

    
aclImdb_v1.tar.gz    48%[========>           ]  38.93M   258KB/s    eta 3m 14s 

    
aclImdb_v1.tar.gz    48%[========>           ]  39.06M   256KB/s    eta 3m 14s 

    
aclImdb_v1.tar.gz    48%[========>           ]  39.20M   256KB/s    eta 3m 13s 

    
aclImdb_v1.tar.gz    49%[========>           ]  39.35M   255KB/s    eta 3m 13s 

    
aclImdb_v1.tar.gz    49%[========>           ]  39.51M   255KB/s    eta 3m 11s 

    
aclImdb_v1.tar.gz    49%[========>           ]  39.66M   256KB/s    eta 3m 11s 

    
aclImdb_v1.tar.gz    49%[========>           ]  39.82M   256KB/s    eta 3m 10s 

    
aclImdb_v1.tar.gz    49%[========>           ]  39.97M   255KB/s    eta 3m 10s 

    
aclImdb_v1.tar.gz    50%[=========>          ]  40.13M   254KB/s    eta 3m 8s  

    
aclImdb_v1.tar.gz    50%[=========>          ]  40.29M   261KB/s    eta 3m 8s  

    
aclImdb_v1.tar.gz    50%[=========>          ]  40.45M   260KB/s    eta 3m 6s  

    
aclImdb_v1.tar.gz    50%[=========>          ]  40.62M   257KB/s    eta 3m 6s  

    
aclImdb_v1.tar.gz    50%[=========>          ]  40.79M   260KB/s    eta 3m 4s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  40.96M   253KB/s    eta 3m 4s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  40.96M   250KB/s    eta 3m 4s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  41.20M   256KB/s    eta 3m 4s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  41.41M   266KB/s    eta 3m 4s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  41.55M   264KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  41.69M   263KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    51%[=========>          ]  41.71M   263KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.74M   245KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.77M   231KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.80M   223KB/s    eta 3m 0s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.85M   215KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.89M   207KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.94M   198KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.99M   190KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.05M   186KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.11M   184KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.20M   180KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.29M   179KB/s    eta 3m 1s  

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.38M   177KB/s    eta 2m 59s 

    
aclImdb_v1.tar.gz    52%[=========>          ]  42.50M   172KB/s    eta 2m 59s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  42.61M   167KB/s    eta 2m 58s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  42.72M   171KB/s    eta 2m 58s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  42.83M   158KB/s    eta 2m 57s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  42.96M   147KB/s    eta 2m 57s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  43.09M   146KB/s    eta 2m 56s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  43.20M   147KB/s    eta 2m 56s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.33M   157KB/s    eta 2m 56s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.46M   168KB/s    eta 2m 54s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.58M   176KB/s    eta 2m 54s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.72M   186KB/s    eta 2m 53s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.85M   195KB/s    eta 2m 53s 

    
aclImdb_v1.tar.gz    54%[=========>          ]  43.98M   203KB/s    eta 2m 52s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.14M   213KB/s    eta 2m 52s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.30M   224KB/s    eta 2m 50s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.47M   229KB/s    eta 2m 50s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.64M   224KB/s    eta 2m 48s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.70M   225KB/s    eta 2m 48s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.88M   229KB/s    eta 2m 48s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.08M   232KB/s    eta 2m 46s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.21M   230KB/s    eta 2m 46s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.37M   234KB/s    eta 2m 44s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.53M   237KB/s    eta 2m 44s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.69M   241KB/s    eta 2m 42s 

    
aclImdb_v1.tar.gz    57%[==========>         ]  45.87M   247KB/s    eta 2m 42s 

    
aclImdb_v1.tar.gz    57%[==========>         ]  46.03M   241KB/s    eta 2m 41s 

    
aclImdb_v1.tar.gz    57%[==========>         ]  46.09M   243KB/s    eta 2m 41s 

    
aclImdb_v1.tar.gz    57%[==========>         ]  46.28M   247KB/s    eta 2m 41s 

    
aclImdb_v1.tar.gz    57%[==========>         ]  46.48M   248KB/s    eta 2m 38s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  46.61M   249KB/s    eta 2m 38s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  46.76M   250KB/s    eta 2m 37s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  46.90M   251KB/s    eta 2m 37s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  47.05M   252KB/s    eta 2m 36s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  47.21M   255KB/s    eta 2m 36s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  47.37M   257KB/s    eta 2m 34s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  47.53M   257KB/s    eta 2m 34s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  47.69M   257KB/s    eta 2m 32s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  47.85M   256KB/s    eta 2m 32s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  48.01M   263KB/s    eta 2m 30s 

    
aclImdb_v1.tar.gz    60%[===========>        ]  48.18M   262KB/s    eta 2m 30s 

    
aclImdb_v1.tar.gz    60%[===========>        ]  48.36M   260KB/s    eta 2m 29s 

    
aclImdb_v1.tar.gz    60%[===========>        ]  48.52M   254KB/s    eta 2m 29s 

    
aclImdb_v1.tar.gz    60%[===========>        ]  48.58M   255KB/s    eta 2m 28s 

    
aclImdb_v1.tar.gz    60%[===========>        ]  48.76M   257KB/s    eta 2m 28s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  48.97M   261KB/s    eta 2m 25s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  49.10M   258KB/s    eta 2m 25s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  49.26M   257KB/s    eta 2m 24s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  49.41M   264KB/s    eta 2m 24s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  49.58M   262KB/s    eta 2m 22s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  49.75M   260KB/s    eta 2m 22s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  49.91M   253KB/s    eta 2m 21s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  49.98M   255KB/s    eta 2m 21s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  50.16M   258KB/s    eta 2m 21s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  50.36M   262KB/s    eta 2m 18s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  50.49M   261KB/s    eta 2m 18s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  50.63M   264KB/s    eta 2m 18s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  50.78M   264KB/s    eta 2m 16s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  50.94M   264KB/s    eta 2m 16s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  51.09M   264KB/s    eta 2m 15s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  51.24M   263KB/s    eta 2m 15s 

    
aclImdb_v1.tar.gz    64%[===========>        ]  51.41M   263KB/s    eta 2m 13s 

    
aclImdb_v1.tar.gz    64%[===========>        ]  51.57M   263KB/s    eta 2m 13s 

    
aclImdb_v1.tar.gz    64%[===========>        ]  51.73M   261KB/s    eta 2m 11s 

    
aclImdb_v1.tar.gz    64%[===========>        ]  51.89M   269KB/s    eta 2m 11s 

    
aclImdb_v1.tar.gz    64%[===========>        ]  52.05M   267KB/s    eta 2m 10s 

    
aclImdb_v1.tar.gz    65%[============>       ]  52.22M   265KB/s    eta 2m 10s 

    
aclImdb_v1.tar.gz    65%[============>       ]  52.38M   258KB/s    eta 2m 8s  

    
aclImdb_v1.tar.gz    65%[============>       ]  52.45M   259KB/s    eta 2m 8s  

    
aclImdb_v1.tar.gz    65%[============>       ]  52.63M   261KB/s    eta 2m 8s  

    
aclImdb_v1.tar.gz    65%[============>       ]  52.83M   265KB/s    eta 2m 6s  

    
aclImdb_v1.tar.gz    66%[============>       ]  52.96M   262KB/s    eta 2m 6s  

    
aclImdb_v1.tar.gz    66%[============>       ]  53.10M   260KB/s    eta 2m 5s  

    
aclImdb_v1.tar.gz    66%[============>       ]  53.26M   267KB/s    eta 2m 5s  

    
aclImdb_v1.tar.gz    66%[============>       ]  53.42M   271KB/s    eta 2m 5s  

    
aclImdb_v1.tar.gz    66%[============>       ]  53.58M   275KB/s    eta 2m 2s  

    
aclImdb_v1.tar.gz    67%[============>       ]  53.75M   286KB/s    eta 2m 2s  

    
aclImdb_v1.tar.gz    67%[============>       ]  53.94M   292KB/s    eta 2m 2s  

    
aclImdb_v1.tar.gz    67%[============>       ]  54.14M   299KB/s    eta 1m 59s 

    
aclImdb_v1.tar.gz    67%[============>       ]  54.30M   289KB/s    eta 1m 59s 

    
aclImdb_v1.tar.gz    67%[============>       ]  54.36M   290KB/s    eta 1m 58s 

    
aclImdb_v1.tar.gz    68%[============>       ]  54.61M   306KB/s    eta 1m 58s 

    
aclImdb_v1.tar.gz    68%[============>       ]  54.75M   313KB/s    eta 1m 58s 

    
aclImdb_v1.tar.gz    68%[============>       ]  54.90M   311KB/s    eta 1m 55s 

    
aclImdb_v1.tar.gz    68%[============>       ]  55.04M   313KB/s    eta 1m 55s 

    
aclImdb_v1.tar.gz    68%[============>       ]  55.21M   314KB/s    eta 1m 53s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.38M   315KB/s    eta 1m 53s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.54M   303KB/s    eta 1m 52s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.61M   304KB/s    eta 1m 52s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.78M   310KB/s    eta 1m 52s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.99M   325KB/s    eta 1m 49s 

    
aclImdb_v1.tar.gz    69%[============>       ]  56.12M   320KB/s    eta 1m 49s 

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.26M   314KB/s    eta 1m 48s 

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.40M   316KB/s    eta 1m 48s 

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.55M   316KB/s    eta 1m 47s 

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.69M   322KB/s    eta 1m 47s 

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.86M   316KB/s    eta 1m 47s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.01M   307KB/s    eta 1m 45s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.17M   297KB/s    eta 1m 45s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.32M   286KB/s    eta 1m 43s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.48M   278KB/s    eta 1m 43s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.63M   287KB/s    eta 1m 42s 

    
aclImdb_v1.tar.gz    72%[=============>      ]  57.80M   273KB/s    eta 1m 42s 

    
aclImdb_v1.tar.gz    72%[=============>      ]  57.97M   269KB/s    eta 1m 40s 

    
aclImdb_v1.tar.gz    72%[=============>      ]  58.14M   262KB/s    eta 1m 40s 

    
aclImdb_v1.tar.gz    72%[=============>      ]  58.14M   258KB/s    eta 99s    

    
aclImdb_v1.tar.gz    72%[=============>      ]  58.33M   259KB/s    eta 99s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.58M   272KB/s    eta 99s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.73M   270KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.88M   278KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.90M   260KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.93M   245KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  58.96M   237KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.00M   228KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.04M   220KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.07M   210KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.13M   199KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.19M   192KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.25M   187KB/s    eta 96s    

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.31M   179KB/s    eta 95s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.40M   173KB/s    eta 95s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.49M   168KB/s    eta 94s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.58M   163KB/s    eta 94s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.69M   157KB/s    eta 94s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.80M   152KB/s    eta 94s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  59.91M   155KB/s    eta 93s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  60.04M   149KB/s    eta 93s    

    
aclImdb_v1.tar.gz    74%[=============>      ]  60.16M   135KB/s    eta 92s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.29M   133KB/s    eta 92s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.41M   131KB/s    eta 90s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.54M   140KB/s    eta 90s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.67M   152KB/s    eta 89s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.79M   159KB/s    eta 89s    

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.93M   168KB/s    eta 88s    

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.07M   177KB/s    eta 88s    

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.21M   186KB/s    eta 87s    

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.36M   194KB/s    eta 87s    

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.52M   200KB/s    eta 85s    

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.69M   206KB/s    eta 85s    

    
aclImdb_v1.tar.gz    77%[==============>     ]  61.92M   215KB/s    eta 83s    

    
aclImdb_v1.tar.gz    77%[==============>     ]  62.09M   223KB/s    eta 83s    

    
aclImdb_v1.tar.gz    77%[==============>     ]  62.30M   232KB/s    eta 82s    

    
aclImdb_v1.tar.gz    77%[==============>     ]  62.43M   235KB/s    eta 82s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  62.58M   239KB/s    eta 80s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  62.73M   242KB/s    eta 80s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  62.88M   245KB/s    eta 79s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  63.06M   250KB/s    eta 79s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  63.22M   244KB/s    eta 77s    

    
aclImdb_v1.tar.gz    78%[==============>     ]  63.28M   246KB/s    eta 77s    

    
aclImdb_v1.tar.gz    79%[==============>     ]  63.46M   250KB/s    eta 77s    

    
aclImdb_v1.tar.gz    79%[==============>     ]  63.66M   256KB/s    eta 75s    

    
aclImdb_v1.tar.gz    79%[==============>     ]  63.79M   256KB/s    eta 75s    

    
aclImdb_v1.tar.gz    79%[==============>     ]  63.93M   253KB/s    eta 74s    

    
aclImdb_v1.tar.gz    79%[==============>     ]  64.06M   254KB/s    eta 74s    

    
aclImdb_v1.tar.gz    80%[===============>    ]  64.23M   255KB/s    eta 73s    

    
aclImdb_v1.tar.gz    80%[===============>    ]  64.38M   257KB/s    eta 73s    

    
aclImdb_v1.tar.gz    80%[===============>    ]  64.53M   257KB/s    eta 71s    

    
aclImdb_v1.tar.gz    80%[===============>    ]  64.69M   258KB/s    eta 71s    

    
aclImdb_v1.tar.gz    80%[===============>    ]  64.84M   257KB/s    eta 70s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.00M   256KB/s    eta 70s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.15M   257KB/s    eta 68s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.32M   256KB/s    eta 68s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.49M   254KB/s    eta 67s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.65M   248KB/s    eta 67s    

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.71M   248KB/s    eta 66s    

    
aclImdb_v1.tar.gz    82%[===============>    ]  65.89M   251KB/s    eta 66s    

    
aclImdb_v1.tar.gz    82%[===============>    ]  66.09M   255KB/s    eta 64s    

    
aclImdb_v1.tar.gz    82%[===============>    ]  66.23M   254KB/s    eta 64s    

    
aclImdb_v1.tar.gz    82%[===============>    ]  66.38M   251KB/s    eta 63s    

    
aclImdb_v1.tar.gz    82%[===============>    ]  66.53M   258KB/s    eta 63s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  66.69M   257KB/s    eta 61s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  66.85M   250KB/s    eta 61s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  66.86M   243KB/s    eta 60s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  67.05M   242KB/s    eta 60s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  67.09M   234KB/s    eta 60s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  67.22M   234KB/s    eta 60s    

    
aclImdb_v1.tar.gz    83%[===============>    ]  67.35M   231KB/s    eta 58s    

    
aclImdb_v1.tar.gz    84%[===============>    ]  67.49M   230KB/s    eta 58s    

    
aclImdb_v1.tar.gz    84%[===============>    ]  67.63M   230KB/s    eta 57s    

    
aclImdb_v1.tar.gz    84%[===============>    ]  67.78M   229KB/s    eta 57s    

    
aclImdb_v1.tar.gz    84%[===============>    ]  67.93M   229KB/s    eta 56s    

    
aclImdb_v1.tar.gz    84%[===============>    ]  68.09M   229KB/s    eta 56s    

    
aclImdb_v1.tar.gz    85%[================>   ]  68.24M   229KB/s    eta 54s    

    
aclImdb_v1.tar.gz    85%[================>   ]  68.38M   227KB/s    eta 54s    

    
aclImdb_v1.tar.gz    85%[================>   ]  68.55M   226KB/s    eta 53s    

    
aclImdb_v1.tar.gz    85%[================>   ]  68.71M   232KB/s    eta 53s    

    
aclImdb_v1.tar.gz    85%[================>   ]  68.88M   231KB/s    eta 51s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.04M   227KB/s    eta 51s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.10M   223KB/s    eta 50s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.28M   226KB/s    eta 50s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.48M   230KB/s    eta 49s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.61M   228KB/s    eta 49s    

    
aclImdb_v1.tar.gz    86%[================>   ]  69.76M   227KB/s    eta 47s    

    
aclImdb_v1.tar.gz    87%[================>   ]  69.92M   237KB/s    eta 47s    

    
aclImdb_v1.tar.gz    87%[================>   ]  70.08M   240KB/s    eta 46s    

    
aclImdb_v1.tar.gz    87%[================>   ]  70.25M   251KB/s    eta 46s    

    
aclImdb_v1.tar.gz    87%[================>   ]  70.41M   245KB/s    eta 44s    

    
aclImdb_v1.tar.gz    87%[================>   ]  70.47M   247KB/s    eta 44s    

    
aclImdb_v1.tar.gz    88%[================>   ]  70.65M   256KB/s    eta 44s    

    
aclImdb_v1.tar.gz    88%[================>   ]  70.84M   260KB/s    eta 42s    

    
aclImdb_v1.tar.gz    88%[================>   ]  70.98M   266KB/s    eta 42s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.01M   256KB/s    eta 42s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.03M   246KB/s    eta 42s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.07M   237KB/s    eta 42s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.11M   228KB/s    eta 41s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.16M   220KB/s    eta 41s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.20M   211KB/s    eta 41s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.26M   203KB/s    eta 41s    

    
aclImdb_v1.tar.gz    88%[================>   ]  71.33M   195KB/s    eta 40s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.40M   192KB/s    eta 40s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.48M   184KB/s    eta 40s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.59M   176KB/s    eta 40s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.69M   173KB/s    eta 39s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.79M   169KB/s    eta 39s    

    
aclImdb_v1.tar.gz    89%[================>   ]  71.90M   165KB/s    eta 38s    

    
aclImdb_v1.tar.gz    89%[================>   ]  72.01M   161KB/s    eta 38s    

    
aclImdb_v1.tar.gz    89%[================>   ]  72.12M   156KB/s    eta 37s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.23M   154KB/s    eta 37s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.35M   145KB/s    eta 36s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.45M   137KB/s    eta 36s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.57M   133KB/s    eta 35s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.69M   140KB/s    eta 35s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.82M   149KB/s    eta 34s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.95M   156KB/s    eta 34s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.09M   165KB/s    eta 33s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.23M   173KB/s    eta 33s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.39M   182KB/s    eta 31s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.55M   191KB/s    eta 31s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.72M   199KB/s    eta 30s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  73.88M   196KB/s    eta 30s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  74.12M   211KB/s    eta 28s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  74.33M   219KB/s    eta 28s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  74.46M   222KB/s    eta 26s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  74.60M   225KB/s    eta 26s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  74.75M   234KB/s    eta 26s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  74.92M   238KB/s    eta 24s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.09M   243KB/s    eta 24s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.11M   242KB/s    eta 24s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.14M   235KB/s    eta 23s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.19M   230KB/s    eta 23s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.23M   228KB/s    eta 23s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.29M   224KB/s    eta 23s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.34M   223KB/s    eta 23s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  75.40M   217KB/s    eta 23s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.49M   213KB/s    eta 22s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.57M   212KB/s    eta 22s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.67M   208KB/s    eta 22s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.78M   203KB/s    eta 20s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.90M   199KB/s    eta 20s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  76.03M   199KB/s    eta 19s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  76.17M   195KB/s    eta 19s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.31M   189KB/s    eta 18s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.44M   194KB/s    eta 18s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.59M   196KB/s    eta 18s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.74M   189KB/s    eta 16s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.88M   187KB/s    eta 16s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.03M   185KB/s    eta 15s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.18M   191KB/s    eta 15s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.34M   204KB/s    eta 13s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.50M   214KB/s    eta 13s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.57M   212KB/s    eta 12s    

    
aclImdb_v1.tar.gz    96%[==================> ]  77.80M   226KB/s    eta 12s    

    
aclImdb_v1.tar.gz    97%[==================> ]  77.92M   227KB/s    eta 11s    

    
aclImdb_v1.tar.gz    97%[==================> ]  78.05M   233KB/s    eta 11s    

    
aclImdb_v1.tar.gz    97%[==================> ]  78.18M   238KB/s    eta 9s     

    
aclImdb_v1.tar.gz    97%[==================> ]  78.33M   239KB/s    eta 9s     

    
aclImdb_v1.tar.gz    97%[==================> ]  78.49M   243KB/s    eta 8s     

    
aclImdb_v1.tar.gz    98%[==================> ]  78.65M   248KB/s    eta 8s     

    
aclImdb_v1.tar.gz    98%[==================> ]  78.81M   251KB/s    eta 6s     

    
aclImdb_v1.tar.gz    98%[==================> ]  78.99M   255KB/s    eta 6s     

    
aclImdb_v1.tar.gz    98%[==================> ]  79.16M   258KB/s    eta 5s     

    
aclImdb_v1.tar.gz    98%[==================> ]  79.34M   261KB/s    eta 5s     

    
aclImdb_v1.tar.gz    99%[==================> ]  79.50M   264KB/s    eta 5s     

    
aclImdb_v1.tar.gz    99%[==================> ]  79.69M   267KB/s    eta 2s     

    
aclImdb_v1.tar.gz    99%[==================> ]  79.89M   265KB/s    eta 2s     

    
aclImdb_v1.tar.gz    99%[==================> ]  79.92M   256KB/s    eta 1s     

    
aclImdb_v1.tar.gz    99%[==================> ]  80.05M   253KB/s    eta 1s     

    
aclImdb_v1.tar.gz    99%[==================> ]  80.18M   253KB/s    eta 0s     
aclImdb_v1.tar.gz   100%[===================>]  80.23M   256KB/s    in 6m 5s   
    
<div class="k-default-codeblock">
```
2025-01-06 21:38:16 (225 KB/s) - ‘aclImdb_v1.tar.gz’ saved [84125825/84125825]
```
</div>
    


Samples are present in the form of text files. Let's inspect the structure of
the directory.


```python
print(os.listdir("./aclImdb"))
print(os.listdir("./aclImdb/train"))
print(os.listdir("./aclImdb/test"))
```

<div class="k-default-codeblock">
```
['train', 'README', 'imdb.vocab', 'test', 'imdbEr.txt']
['labeledBow.feat', 'urls_neg.txt', 'unsupBow.feat', 'urls_pos.txt', 'pos', 'urls_unsup.txt', 'neg', 'unsup']
['labeledBow.feat', 'urls_neg.txt', 'urls_pos.txt', 'pos', 'neg']

```
</div>
The directory contains two sub-directories: `train` and `test`. Each subdirectory
in turn contains two folders: `pos` and `neg` for positive and negative reviews,
respectively. Before we load the dataset, let's delete the `./aclImdb/train/unsup`
folder since it has unlabelled samples.


```python
!rm -rf aclImdb/train/unsup
```

We'll use Pandas to avoid dependency on `keras.utils.text_dataset_from_directory`
since it is merely a wrapper that returns `tf.data.Dataset` and we wish to make
this script backend-agnostic


```python

# Function to read files and label them
def load_data_from_directory(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                data.append((text, label))
    return data


# Function to create Pandas dataframes
def create_df(directory_pos, directory_neg):

    # Load data from both directories
    pos_data = load_data_from_directory(directory_pos, label=1)
    neg_data = load_data_from_directory(directory_neg, label=0)

    # Combine data from both pos and neg
    all_data = pos_data + neg_data

    # Shuffle the data randomly
    random.shuffle(all_data)

    # Create a pandas DataFrame
    df = pd.DataFrame(all_data, columns=["text", "label"])
    return df


# Create the train and test dataframes, iloc maybe adjusted to a desired size
train_df = create_df("./aclImdb/train/pos", "./aclImdb/train/neg").iloc[0:5000]
test_df = create_df("./aclImdb/test/pos", "./aclImdb/test/neg").iloc[0:2500]

# Train and Validation splits, 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

# Data statistics
print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")
```

<div class="k-default-codeblock">
```
Total training examples: 4750
Total validation examples: 250
Total test examples: 2500

```
</div>
Let's print a few samples.


```python
for i in range(3):
    print(f"text: {train_df['text'][i]}, label: {train_df['label'][i]}")
```

<div class="k-default-codeblock">
```
text: Germans think smirking is funny (just like Americans think mumbling is sexy and that women with English accents are acting). I had to cross my eyes whenever the screen was filled yet again with a giant close-up of a smirking face. One of those 'housewife hacks corporate mainframe' tales where she defrauds a bank by tapping a few random keys on her home PC which is connected only to a power socket. The director obviously loves the rather large leading lady. Can't say I share his feelings. There's quite a funny bit when the entire family sit in front of the television chanting tonelessly along with the adverts. Apparently this review needs to be one line longer so here it is., label: 0
text: I remember the original series vividly mostly due to it's unique blend of wry humor and macabre subject matter. Kolchak was hard-bitten newsman from the Ben Hecht school of big-city reporting, and his gritty determination and wise-ass demeanor made even the most mundane episode eminently watchable. My personal fave was "The Spanish Moss Murders" due to it's totally original storyline. A poor,troubled Cajun youth from Louisiana bayou country, takes part in a sleep research experiment, for the purpose of dream analysis. Something goes inexplicably wrong, and he literally dreams to life a swamp creature inhabiting the dark folk tales of his youth. This malevolent manifestation seeks out all persons who have wronged the dreamer in his conscious state, and brutally suffocates them to death. Kolchak investigates and uncovers this horrible truth, much to the chagrin of police captain Joe "Mad Dog" Siska(wonderfully essayed by a grumpy Keenan Wynn)and the head sleep researcher played by Second City improv founder, Severn Darden, to droll, understated perfection. The wickedly funny, harrowing finale takes place in the Chicago sewer system, and is a series highlight. Kolchak never got any better. Timeless., label: 1
text: Believe it or not, at 12 minutes, this film (for 1912) is a full-length film. Very, very few films were longer than that back then, but that is definitely NOT what sets this odd little film apart from the rest! No, what's different is that all the actors (with the exception of one frog) are bugs...yes, bugs! This simple little domestic comedy could have looked much like productions starring the likes of Chaplin, Laurel and Hardy or Max Linder but instead this Russian production uses bugs (or, I think, models that looked just like bugs). Chaplin and Laurel and Hardy were yet to be discovered and I assume Linder was busy, so perhaps that's why they used bugs! Using stop-motion, the bugs moved and danced and fought amazingly well--and a heck of a lot more realistically than King Kong 21 years later! <br /><br />The film starts with Mr. Beetle sneaking off for a good time. He goes to a bawdy club while his wife supposedly waits at home. But, unfortunately for Mr. Beetle, he is caught on camera by a local film buff. Plus, he doesn't know it but Mrs. Beetle is also carrying on with a bohemian grasshopper painter. Of course, there's a lot more to this domestic comedy than this, but the plot is age-old and very entertaining for adults and kids alike.<br /><br />Weird but also very amazing and watchable., label: 1

```
</div>
### Tokenizing the data

We'll be using the `keras_hub.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_hub.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization algorithm;
training it on a corpus gives us a vocabulary of subwords. A subword tokenizer
is a compromise between word tokenizers (word tokenizers need very large
vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, KerasHub
makes it very simple to train WordPiece on a corpus with the
`keras_hub.tokenizers.compute_word_piece_vocabulary` utility.

Note: The official implementation of FNet uses the SentencePiece Tokenizer.

Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence length
when the input sequence length is shorter than the maximum sequence length.
- `"[UNK]"` - Unknown token.


```python
reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_df]

vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
    data=[
        "./aclImdb/train/pos/" + f
        for f in os.listdir("./aclImdb/train/pos")
        if os.path.isfile(os.path.join("./aclImdb/train/pos", f))
    ],  # list of files in the specified directory used to create the vocab
    vocabulary_size=VOCAB_SIZE,
    reserved_tokens=reserved_tokens,
    lowercase=True,
)

```

Let's see some tokens!


```python
print("Tokens: ", vocab[100:110])
```

<div class="k-default-codeblock">
```
Tokens:  ['ó', 'ô', 'õ', 'ö', 'ø', 'ú', 'ü', 'ý', '́', '̈']

```
</div>
Now, let's define the tokenizer. We will configure the tokenizer with the
the vocabularies trained above. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.


```python
tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)

```

Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.


```python
input_sentence_ex = train_df["text"][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))

```

<div class="k-default-codeblock">
```
Sentence:  Germans think smirking is funny (just like Americans think mumbling is sexy and that women with English accents are acting). I had to cross my eyes whenever the screen was filled yet again with a giant close-up of a smirking face. One of those 'housewife hacks corporate mainframe' tales where she defrauds a bank by tapping a few random keys on her home PC which is connected only to a power socket. The director obviously loves the rather large leading lady. Can't say I share his feelings. There's quite a funny bit when the entire family sit in front of the television chanting tonelessly along with the adverts. Apparently this review needs to be one line longer so here it is.
Tokens:  tensor([    1,   214,    59,  5480,  4817,   223,   121,   295,    10,   167,
          154,     1,   214,  5964,  5476,   121,  1388,   118,   125,   500,
          128,     1,  3994,   138,   255,    11,    16,     1,   184,   120,
         1827,   171,   609,  2364,   117,   378,   130,  1102,   341,   274,
          128,    41,  2132,   588,    15,   174,   119,    41,    59,  5480,
         4817,   223,   513,    16,     1,   119,   249,     9,  6062, 11135,
          145,  4102,   434,  1384,  5897,   290,     9,  2723,   231,   159,
        13412,  1413,  8535,   145,    41,  2540,   143, 10416,    41,   298,
         3409,  8408,   135,   150,   437,     1,   175,   121,  3669,   185,
          120,    41,   723,   153,  2030,  2264,    16,     1,   285,   854,
         1318,   117,   369,  1162,  1006,   813,    16,     1,     9,    60,
          259,     1,  1419,   134,  1189,    16,     1,     9,    59,   276,
           41,   295,   311,   164,   117,   651,   286,  1296,   122,  1196,
          119,   117,   721,  1874,  1229,  1199,  8278,   454,   128,   117,
         4751,  5572,   145,    16,     1,   126,  1088,   841,   120,   142,
          140,   506,  1382,   153,   241,   124,   121,    16,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0], dtype=torch.int32)
Recovered text after detokenizing:  [UNK] think smirking is funny ( just like [UNK] think mumbling is sexy and that women with [UNK] accents are acting ) . [UNK] had to cross my eyes whenever the screen was filled yet again with a giant close - up of a smirking face . [UNK] of those ' housewife hacks corporate mainframe ' tales where she defrauds a bank by tapping a few random keys on her home [UNK] which is connected only to a power socket . [UNK] director obviously loves the rather large leading lady . [UNK] ' t say [UNK] share his feelings . [UNK] ' s quite a funny bit when the entire family sit in front of the television chanting tonelessly along with the adverts . [UNK] this review needs to be one line longer so here it is . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

```
</div>
Create the final datasets, method adapted from `PyDataset` doc string


```python

class UnifiedPyDataset(keras.utils.PyDataset):
    """A Keras-compatible dataset that processes a DataFrame for TensorFlow, JAX, and PyTorch."""

    def __init__(
        self,
        df,
        batch_size=BATCH_SIZE,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
        **kwargs,
    ):
        """
        Args:
            df: pandas DataFrame with data
            batch_size: Batch size for dataset
            workers: Number of workers to use for parallel loading (Keras)
            use_multiprocessing: Whether to use multiprocessing
            max_queue_size: Maximum size of the data queue for parallel loading
        """
        super().__init__(**kwargs)
        self.dataframe = df
        columns = ["text", "label"]

        # text files
        self.text_x = self.dataframe["text"]
        self.text_y = self.dataframe["label"]

        # general
        self.batch_size = batch_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def __getitem__(self, index):
        """
        Fetches a batch of data from the dataset at the given index.

        Args:
            index: position of the batch in the dataset
        """

        # Return x, y for batch idx.
        low = index * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.

        high_text = min(low + self.batch_size, len(self.text_x))

        # text files batches
        batch_text_x = self.text_x[low:high_text]
        batch_text_y = self.text_y[low:high_text]

        # Tokenize the data in dataframe and stack them into an nd.array of axis_0=batch
        text = np.array([[tokenizer(text) for text in batch_text_x]])
        return (
            {
                "input_ids": keras.ops.squeeze(text),
            },
            # Target lables
            np.array(batch_text_y),
        )

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return math.ceil(len(self.dataframe) / self.batch_size)


def prepare_dataset(dataframe, training=True):
    ds = UnifiedPyDataset(
        dataframe,
        batch_size=32,
        workers=4,
    )
    return ds


train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)
test_ds = prepare_dataset(test_df)
```

---
## Building the model

Now, let's move on to the exciting part - defining our model!
We first need an embedding layer, i.e., a layer that maps every token in the input
sequence to a vector. This embedding layer can be initialised randomly. We also
need a positional embedding layer which encodes the word order in the sequence.
The convention is to add, i.e., sum, these two embeddings. KerasHub has a
`keras_hub.layers.TokenAndPositionEmbedding ` layer which does all of the above
steps for us.

Our FNet classification model consists of three `keras_hub.layers.FNetEncoder`
layers with a `keras.layers.Dense` layer on top.

Note: For FNet, masking the padding tokens has a minimal effect on results. In the
official implementation, the padding tokens are not masked.


```python
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
```

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:932: UserWarning: Layer 'f_net_encoder' (of type FNetEncoder) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(

```
</div>
---
## Training our model

We'll use accuracy to monitor training progress on the validation data. Let's
train our model for 3 epochs.


```python
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "fnet_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ token_and_position_embedding    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │     <span style="color: #00af00; text-decoration-color: #00af00">1,985,536</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbedding</span>)     │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ f_net_encoder (<span style="color: #0087ff; text-decoration-color: #0087ff">FNetEncoder</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">132,224</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ f_net_encoder_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">FNetEncoder</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">132,224</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ f_net_encoder_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">FNetEncoder</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">132,224</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">129</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,382,337</span> (9.09 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,382,337</span> (9.09 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:248: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: input_ids
Received: inputs=['Tensor(shape=torch.Size([32, 512]))']
  warnings.warn(msg)

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:932: UserWarning: Layer 'position_embedding' (of type PositionEmbedding) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(

```
</div>
    
   1/149 [37m━━━━━━━━━━━━━━━━━━━━  11:50 5s/step - accuracy: 0.4062 - loss: 0.6976

<div class="k-default-codeblock">
```

```
</div>
   2/149 [37m━━━━━━━━━━━━━━━━━━━━  2:52 1s/step - accuracy: 0.4297 - loss: 0.8143 

<div class="k-default-codeblock">
```

```
</div>
   3/149 [37m━━━━━━━━━━━━━━━━━━━━  2:55 1s/step - accuracy: 0.4427 - loss: 0.8336

<div class="k-default-codeblock">
```

```
</div>
   4/149 [37m━━━━━━━━━━━━━━━━━━━━  3:18 1s/step - accuracy: 0.4512 - loss: 0.8359

<div class="k-default-codeblock">
```

```
</div>
   5/149 [37m━━━━━━━━━━━━━━━━━━━━  3:27 1s/step - accuracy: 0.4609 - loss: 0.8350

<div class="k-default-codeblock">
```

```
</div>
   6/149 [37m━━━━━━━━━━━━━━━━━━━━  3:32 1s/step - accuracy: 0.4692 - loss: 0.8342

<div class="k-default-codeblock">
```

```
</div>
   7/149 [37m━━━━━━━━━━━━━━━━━━━━  3:28 1s/step - accuracy: 0.4755 - loss: 0.8332

<div class="k-default-codeblock">
```

```
</div>
   8/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:26 1s/step - accuracy: 0.4781 - loss: 0.8314

<div class="k-default-codeblock">
```

```
</div>
   9/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:20 1s/step - accuracy: 0.4782 - loss: 0.8297

<div class="k-default-codeblock">
```

```
</div>
  10/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:16 1s/step - accuracy: 0.4779 - loss: 0.8283

<div class="k-default-codeblock">
```

```
</div>
  11/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:15 1s/step - accuracy: 0.4786 - loss: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  12/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:18 1s/step - accuracy: 0.4802 - loss: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  13/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:20 1s/step - accuracy: 0.4819 - loss: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  14/149 ━[37m━━━━━━━━━━━━━━━━━━━  3:22 2s/step - accuracy: 0.4835 - loss: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  15/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:24 2s/step - accuracy: 0.4846 - loss: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  16/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:25 2s/step - accuracy: 0.4854 - loss: 0.8109

<div class="k-default-codeblock">
```

```
</div>
  17/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:25 2s/step - accuracy: 0.4866 - loss: 0.8080

<div class="k-default-codeblock">
```

```
</div>
  18/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:26 2s/step - accuracy: 0.4875 - loss: 0.8053

<div class="k-default-codeblock">
```

```
</div>
  19/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:25 2s/step - accuracy: 0.4882 - loss: 0.8030

<div class="k-default-codeblock">
```

```
</div>
  20/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:24 2s/step - accuracy: 0.4888 - loss: 0.8007

<div class="k-default-codeblock">
```

```
</div>
  21/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:23 2s/step - accuracy: 0.4891 - loss: 0.7986

<div class="k-default-codeblock">
```

```
</div>
  22/149 ━━[37m━━━━━━━━━━━━━━━━━━  3:23 2s/step - accuracy: 0.4895 - loss: 0.7965

<div class="k-default-codeblock">
```

```
</div>
  23/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:22 2s/step - accuracy: 0.4898 - loss: 0.7945

<div class="k-default-codeblock">
```

```
</div>
  24/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:22 2s/step - accuracy: 0.4902 - loss: 0.7926

<div class="k-default-codeblock">
```

```
</div>
  25/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:21 2s/step - accuracy: 0.4908 - loss: 0.7907

<div class="k-default-codeblock">
```

```
</div>
  26/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:22 2s/step - accuracy: 0.4915 - loss: 0.7888

<div class="k-default-codeblock">
```

```
</div>
  27/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:21 2s/step - accuracy: 0.4922 - loss: 0.7870

<div class="k-default-codeblock">
```

```
</div>
  28/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:21 2s/step - accuracy: 0.4929 - loss: 0.7853

<div class="k-default-codeblock">
```

```
</div>
  29/149 ━━━[37m━━━━━━━━━━━━━━━━━  3:20 2s/step - accuracy: 0.4935 - loss: 0.7837

<div class="k-default-codeblock">
```

```
</div>
  30/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:19 2s/step - accuracy: 0.4941 - loss: 0.7821

<div class="k-default-codeblock">
```

```
</div>
  31/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:17 2s/step - accuracy: 0.4945 - loss: 0.7806

<div class="k-default-codeblock">
```

```
</div>
  32/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:16 2s/step - accuracy: 0.4948 - loss: 0.7792

<div class="k-default-codeblock">
```

```
</div>
  33/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:15 2s/step - accuracy: 0.4952 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
  34/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:15 2s/step - accuracy: 0.4956 - loss: 0.7765

<div class="k-default-codeblock">
```

```
</div>
  35/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:14 2s/step - accuracy: 0.4961 - loss: 0.7752

<div class="k-default-codeblock">
```

```
</div>
  36/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:13 2s/step - accuracy: 0.4964 - loss: 0.7739

<div class="k-default-codeblock">
```

```
</div>
  37/149 ━━━━[37m━━━━━━━━━━━━━━━━  3:11 2s/step - accuracy: 0.4967 - loss: 0.7727

<div class="k-default-codeblock">
```

```
</div>
  38/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:11 2s/step - accuracy: 0.4971 - loss: 0.7715

<div class="k-default-codeblock">
```

```
</div>
  39/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:09 2s/step - accuracy: 0.4975 - loss: 0.7704

<div class="k-default-codeblock">
```

```
</div>
  40/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:08 2s/step - accuracy: 0.4978 - loss: 0.7693

<div class="k-default-codeblock">
```

```
</div>
  41/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:07 2s/step - accuracy: 0.4981 - loss: 0.7682

<div class="k-default-codeblock">
```

```
</div>
  42/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:06 2s/step - accuracy: 0.4984 - loss: 0.7672

<div class="k-default-codeblock">
```

```
</div>
  43/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:05 2s/step - accuracy: 0.4986 - loss: 0.7662

<div class="k-default-codeblock">
```

```
</div>
  44/149 ━━━━━[37m━━━━━━━━━━━━━━━  3:03 2s/step - accuracy: 0.4988 - loss: 0.7653

<div class="k-default-codeblock">
```

```
</div>
  45/149 ━━━━━━[37m━━━━━━━━━━━━━━  3:02 2s/step - accuracy: 0.4990 - loss: 0.7644

<div class="k-default-codeblock">
```

```
</div>
  46/149 ━━━━━━[37m━━━━━━━━━━━━━━  3:01 2s/step - accuracy: 0.4992 - loss: 0.7635

<div class="k-default-codeblock">
```

```
</div>
  47/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:59 2s/step - accuracy: 0.4993 - loss: 0.7626

<div class="k-default-codeblock">
```

```
</div>
  48/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:58 2s/step - accuracy: 0.4996 - loss: 0.7617

<div class="k-default-codeblock">
```

```
</div>
  49/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:56 2s/step - accuracy: 0.4998 - loss: 0.7609

<div class="k-default-codeblock">
```

```
</div>
  50/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:55 2s/step - accuracy: 0.5000 - loss: 0.7601

<div class="k-default-codeblock">
```

```
</div>
  51/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:54 2s/step - accuracy: 0.5003 - loss: 0.7593

<div class="k-default-codeblock">
```

```
</div>
  52/149 ━━━━━━[37m━━━━━━━━━━━━━━  2:52 2s/step - accuracy: 0.5004 - loss: 0.7585

<div class="k-default-codeblock">
```

```
</div>
  53/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:51 2s/step - accuracy: 0.5007 - loss: 0.7578

<div class="k-default-codeblock">
```

```
</div>
  54/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:49 2s/step - accuracy: 0.5008 - loss: 0.7571

<div class="k-default-codeblock">
```

```
</div>
  55/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:48 2s/step - accuracy: 0.5011 - loss: 0.7564

<div class="k-default-codeblock">
```

```
</div>
  56/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:46 2s/step - accuracy: 0.5013 - loss: 0.7557

<div class="k-default-codeblock">
```

```
</div>
  57/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:45 2s/step - accuracy: 0.5015 - loss: 0.7550

<div class="k-default-codeblock">
```

```
</div>
  58/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:43 2s/step - accuracy: 0.5016 - loss: 0.7543

<div class="k-default-codeblock">
```

```
</div>
  59/149 ━━━━━━━[37m━━━━━━━━━━━━━  2:42 2s/step - accuracy: 0.5018 - loss: 0.7537

<div class="k-default-codeblock">
```

```
</div>
  60/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:40 2s/step - accuracy: 0.5020 - loss: 0.7531

<div class="k-default-codeblock">
```

```
</div>
  61/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:38 2s/step - accuracy: 0.5023 - loss: 0.7525

<div class="k-default-codeblock">
```

```
</div>
  62/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:37 2s/step - accuracy: 0.5025 - loss: 0.7519

<div class="k-default-codeblock">
```

```
</div>
  63/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:36 2s/step - accuracy: 0.5027 - loss: 0.7513

<div class="k-default-codeblock">
```

```
</div>
  64/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:34 2s/step - accuracy: 0.5029 - loss: 0.7507

<div class="k-default-codeblock">
```

```
</div>
  65/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:32 2s/step - accuracy: 0.5032 - loss: 0.7501

<div class="k-default-codeblock">
```

```
</div>
  66/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:31 2s/step - accuracy: 0.5034 - loss: 0.7496

<div class="k-default-codeblock">
```

```
</div>
  67/149 ━━━━━━━━[37m━━━━━━━━━━━━  2:30 2s/step - accuracy: 0.5037 - loss: 0.7490

<div class="k-default-codeblock">
```

```
</div>
  68/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:28 2s/step - accuracy: 0.5039 - loss: 0.7484

<div class="k-default-codeblock">
```

```
</div>
  69/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:26 2s/step - accuracy: 0.5041 - loss: 0.7479

<div class="k-default-codeblock">
```

```
</div>
  70/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:25 2s/step - accuracy: 0.5043 - loss: 0.7474

<div class="k-default-codeblock">
```

```
</div>
  71/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:23 2s/step - accuracy: 0.5045 - loss: 0.7469

<div class="k-default-codeblock">
```

```
</div>
  72/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:22 2s/step - accuracy: 0.5046 - loss: 0.7464

<div class="k-default-codeblock">
```

```
</div>
  73/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:20 2s/step - accuracy: 0.5048 - loss: 0.7460

<div class="k-default-codeblock">
```

```
</div>
  74/149 ━━━━━━━━━[37m━━━━━━━━━━━  2:18 2s/step - accuracy: 0.5050 - loss: 0.7455

<div class="k-default-codeblock">
```

```
</div>
  75/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:17 2s/step - accuracy: 0.5052 - loss: 0.7451

<div class="k-default-codeblock">
```

```
</div>
  76/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:15 2s/step - accuracy: 0.5053 - loss: 0.7447

<div class="k-default-codeblock">
```

```
</div>
  77/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:13 2s/step - accuracy: 0.5055 - loss: 0.7443

<div class="k-default-codeblock">
```

```
</div>
  78/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:11 2s/step - accuracy: 0.5056 - loss: 0.7439

<div class="k-default-codeblock">
```

```
</div>
  79/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:10 2s/step - accuracy: 0.5058 - loss: 0.7435

<div class="k-default-codeblock">
```

```
</div>
  80/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:08 2s/step - accuracy: 0.5059 - loss: 0.7431

<div class="k-default-codeblock">
```

```
</div>
  81/149 ━━━━━━━━━━[37m━━━━━━━━━━  2:06 2s/step - accuracy: 0.5061 - loss: 0.7428

<div class="k-default-codeblock">
```

```
</div>
  82/149 ━━━━━━━━━━━[37m━━━━━━━━━  2:04 2s/step - accuracy: 0.5062 - loss: 0.7424

<div class="k-default-codeblock">
```

```
</div>
  83/149 ━━━━━━━━━━━[37m━━━━━━━━━  2:02 2s/step - accuracy: 0.5063 - loss: 0.7420

<div class="k-default-codeblock">
```

```
</div>
  84/149 ━━━━━━━━━━━[37m━━━━━━━━━  2:01 2s/step - accuracy: 0.5065 - loss: 0.7417

<div class="k-default-codeblock">
```

```
</div>
  85/149 ━━━━━━━━━━━[37m━━━━━━━━━  1:59 2s/step - accuracy: 0.5066 - loss: 0.7413

<div class="k-default-codeblock">
```

```
</div>
  86/149 ━━━━━━━━━━━[37m━━━━━━━━━  1:57 2s/step - accuracy: 0.5068 - loss: 0.7410

<div class="k-default-codeblock">
```

```
</div>
  87/149 ━━━━━━━━━━━[37m━━━━━━━━━  1:56 2s/step - accuracy: 0.5070 - loss: 0.7406

<div class="k-default-codeblock">
```

```
</div>
  88/149 ━━━━━━━━━━━[37m━━━━━━━━━  1:54 2s/step - accuracy: 0.5071 - loss: 0.7403

<div class="k-default-codeblock">
```

```
</div>
  89/149 ━━━━━━━━━━━[37m━━━━━━━━━  1:52 2s/step - accuracy: 0.5073 - loss: 0.7399

<div class="k-default-codeblock">
```

```
</div>
  90/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:51 2s/step - accuracy: 0.5075 - loss: 0.7396

<div class="k-default-codeblock">
```

```
</div>
  91/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:49 2s/step - accuracy: 0.5077 - loss: 0.7392

<div class="k-default-codeblock">
```

```
</div>
  92/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:47 2s/step - accuracy: 0.5079 - loss: 0.7389

<div class="k-default-codeblock">
```

```
</div>
  93/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:45 2s/step - accuracy: 0.5081 - loss: 0.7385

<div class="k-default-codeblock">
```

```
</div>
  94/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:44 2s/step - accuracy: 0.5083 - loss: 0.7382

<div class="k-default-codeblock">
```

```
</div>
  95/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:42 2s/step - accuracy: 0.5085 - loss: 0.7379

<div class="k-default-codeblock">
```

```
</div>
  96/149 ━━━━━━━━━━━━[37m━━━━━━━━  1:40 2s/step - accuracy: 0.5087 - loss: 0.7376

<div class="k-default-codeblock">
```

```
</div>
  97/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:38 2s/step - accuracy: 0.5088 - loss: 0.7373

<div class="k-default-codeblock">
```

```
</div>
  98/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:37 2s/step - accuracy: 0.5089 - loss: 0.7370

<div class="k-default-codeblock">
```

```
</div>
  99/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:35 2s/step - accuracy: 0.5091 - loss: 0.7367

<div class="k-default-codeblock">
```

```
</div>
 100/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:33 2s/step - accuracy: 0.5092 - loss: 0.7364

<div class="k-default-codeblock">
```

```
</div>
 101/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:31 2s/step - accuracy: 0.5093 - loss: 0.7362

<div class="k-default-codeblock">
```

```
</div>
 102/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:29 2s/step - accuracy: 0.5095 - loss: 0.7359

<div class="k-default-codeblock">
```

```
</div>
 103/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:28 2s/step - accuracy: 0.5096 - loss: 0.7356

<div class="k-default-codeblock">
```

```
</div>
 104/149 ━━━━━━━━━━━━━[37m━━━━━━━  1:26 2s/step - accuracy: 0.5097 - loss: 0.7353

<div class="k-default-codeblock">
```

```
</div>
 105/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:24 2s/step - accuracy: 0.5099 - loss: 0.7351

<div class="k-default-codeblock">
```

```
</div>
 106/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:22 2s/step - accuracy: 0.5100 - loss: 0.7348

<div class="k-default-codeblock">
```

```
</div>
 107/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:20 2s/step - accuracy: 0.5102 - loss: 0.7345

<div class="k-default-codeblock">
```

```
</div>
 108/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:18 2s/step - accuracy: 0.5104 - loss: 0.7342

<div class="k-default-codeblock">
```

```
</div>
 109/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:16 2s/step - accuracy: 0.5105 - loss: 0.7340

<div class="k-default-codeblock">
```

```
</div>
 110/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:15 2s/step - accuracy: 0.5107 - loss: 0.7337

<div class="k-default-codeblock">
```

```
</div>
 111/149 ━━━━━━━━━━━━━━[37m━━━━━━  1:13 2s/step - accuracy: 0.5109 - loss: 0.7334

<div class="k-default-codeblock">
```

```
</div>
 112/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:11 2s/step - accuracy: 0.5111 - loss: 0.7332

<div class="k-default-codeblock">
```

```
</div>
 113/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:09 2s/step - accuracy: 0.5113 - loss: 0.7329

<div class="k-default-codeblock">
```

```
</div>
 114/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:07 2s/step - accuracy: 0.5114 - loss: 0.7327

<div class="k-default-codeblock">
```

```
</div>
 115/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:05 2s/step - accuracy: 0.5116 - loss: 0.7324

<div class="k-default-codeblock">
```

```
</div>
 116/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:03 2s/step - accuracy: 0.5118 - loss: 0.7322

<div class="k-default-codeblock">
```

```
</div>
 117/149 ━━━━━━━━━━━━━━━[37m━━━━━  1:01 2s/step - accuracy: 0.5120 - loss: 0.7319

<div class="k-default-codeblock">
```

```
</div>
 118/149 ━━━━━━━━━━━━━━━[37m━━━━━  59s 2s/step - accuracy: 0.5121 - loss: 0.7317 

<div class="k-default-codeblock">
```

```
</div>
 119/149 ━━━━━━━━━━━━━━━[37m━━━━━  58s 2s/step - accuracy: 0.5123 - loss: 0.7314

<div class="k-default-codeblock">
```

```
</div>
 120/149 ━━━━━━━━━━━━━━━━[37m━━━━  56s 2s/step - accuracy: 0.5125 - loss: 0.7311

<div class="k-default-codeblock">
```

```
</div>
 121/149 ━━━━━━━━━━━━━━━━[37m━━━━  54s 2s/step - accuracy: 0.5127 - loss: 0.7309

<div class="k-default-codeblock">
```

```
</div>
 122/149 ━━━━━━━━━━━━━━━━[37m━━━━  52s 2s/step - accuracy: 0.5129 - loss: 0.7306

<div class="k-default-codeblock">
```

```
</div>
 123/149 ━━━━━━━━━━━━━━━━[37m━━━━  50s 2s/step - accuracy: 0.5131 - loss: 0.7304

<div class="k-default-codeblock">
```

```
</div>
 124/149 ━━━━━━━━━━━━━━━━[37m━━━━  48s 2s/step - accuracy: 0.5133 - loss: 0.7302

<div class="k-default-codeblock">
```

```
</div>
 125/149 ━━━━━━━━━━━━━━━━[37m━━━━  46s 2s/step - accuracy: 0.5135 - loss: 0.7299

<div class="k-default-codeblock">
```

```
</div>
 126/149 ━━━━━━━━━━━━━━━━[37m━━━━  44s 2s/step - accuracy: 0.5137 - loss: 0.7296

<div class="k-default-codeblock">
```

```
</div>
 127/149 ━━━━━━━━━━━━━━━━━[37m━━━  42s 2s/step - accuracy: 0.5139 - loss: 0.7294

<div class="k-default-codeblock">
```

```
</div>
 128/149 ━━━━━━━━━━━━━━━━━[37m━━━  40s 2s/step - accuracy: 0.5142 - loss: 0.7291

<div class="k-default-codeblock">
```

```
</div>
 129/149 ━━━━━━━━━━━━━━━━━[37m━━━  38s 2s/step - accuracy: 0.5144 - loss: 0.7289

<div class="k-default-codeblock">
```

```
</div>
 130/149 ━━━━━━━━━━━━━━━━━[37m━━━  36s 2s/step - accuracy: 0.5147 - loss: 0.7286

<div class="k-default-codeblock">
```

```
</div>
 131/149 ━━━━━━━━━━━━━━━━━[37m━━━  35s 2s/step - accuracy: 0.5149 - loss: 0.7284

<div class="k-default-codeblock">
```

```
</div>
 132/149 ━━━━━━━━━━━━━━━━━[37m━━━  33s 2s/step - accuracy: 0.5152 - loss: 0.7281

<div class="k-default-codeblock">
```

```
</div>
 133/149 ━━━━━━━━━━━━━━━━━[37m━━━  31s 2s/step - accuracy: 0.5154 - loss: 0.7279

<div class="k-default-codeblock">
```

```
</div>
 134/149 ━━━━━━━━━━━━━━━━━[37m━━━  29s 2s/step - accuracy: 0.5157 - loss: 0.7276

<div class="k-default-codeblock">
```

```
</div>
 135/149 ━━━━━━━━━━━━━━━━━━[37m━━  27s 2s/step - accuracy: 0.5159 - loss: 0.7274

<div class="k-default-codeblock">
```

```
</div>
 136/149 ━━━━━━━━━━━━━━━━━━[37m━━  25s 2s/step - accuracy: 0.5162 - loss: 0.7271

<div class="k-default-codeblock">
```

```
</div>
 137/149 ━━━━━━━━━━━━━━━━━━[37m━━  23s 2s/step - accuracy: 0.5165 - loss: 0.7268

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:248: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: input_ids
Received: inputs=['Tensor(shape=torch.Size([14, 512]))']
  warnings.warn(msg)


```
</div>
 138/149 ━━━━━━━━━━━━━━━━━━[37m━━  21s 2s/step - accuracy: 0.5167 - loss: 0.7266

<div class="k-default-codeblock">
```

```
</div>
 139/149 ━━━━━━━━━━━━━━━━━━[37m━━  19s 2s/step - accuracy: 0.5170 - loss: 0.7263

<div class="k-default-codeblock">
```

```
</div>
 140/149 ━━━━━━━━━━━━━━━━━━[37m━━  17s 2s/step - accuracy: 0.5173 - loss: 0.7261

<div class="k-default-codeblock">
```

```
</div>
 141/149 ━━━━━━━━━━━━━━━━━━[37m━━  15s 2s/step - accuracy: 0.5175 - loss: 0.7258

<div class="k-default-codeblock">
```

```
</div>
 142/149 ━━━━━━━━━━━━━━━━━━━[37m━  13s 2s/step - accuracy: 0.5178 - loss: 0.7255

<div class="k-default-codeblock">
```

```
</div>
 143/149 ━━━━━━━━━━━━━━━━━━━[37m━  11s 2s/step - accuracy: 0.5181 - loss: 0.7253

<div class="k-default-codeblock">
```

```
</div>
 144/149 ━━━━━━━━━━━━━━━━━━━[37m━  9s 2s/step - accuracy: 0.5184 - loss: 0.7250 

<div class="k-default-codeblock">
```

```
</div>
 145/149 ━━━━━━━━━━━━━━━━━━━[37m━  7s 2s/step - accuracy: 0.5187 - loss: 0.7248

<div class="k-default-codeblock">
```

```
</div>
 146/149 ━━━━━━━━━━━━━━━━━━━[37m━  5s 2s/step - accuracy: 0.5190 - loss: 0.7245

<div class="k-default-codeblock">
```

```
</div>
 147/149 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - accuracy: 0.5192 - loss: 0.7242

<div class="k-default-codeblock">
```

```
</div>
 148/149 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - accuracy: 0.5195 - loss: 0.7240

<div class="k-default-codeblock">
```

```
</div>
 149/149 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.5198 - loss: 0.7237

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:248: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: input_ids
Received: inputs=['Tensor(shape=torch.Size([26, 512]))']
  warnings.warn(msg)


```
</div>
 149/149 ━━━━━━━━━━━━━━━━━━━━ 305s 2s/step - accuracy: 0.5201 - loss: 0.7234 - val_accuracy: 0.7400 - val_loss: 0.5591





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7c51e823fd10>

```
</div>
We obtain a train accuracy of around 92% and a validation accuracy of around
85%. Moreover, for 3 epochs, it takes around 86 seconds to train the model
(on Colab with a 16 GB Tesla T4 GPU).

Let's calculate the test accuracy.


```python
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

```

    
  1/79 [37m━━━━━━━━━━━━━━━━━━━━  5:55 5s/step - accuracy: 0.6875 - loss: 0.5899

<div class="k-default-codeblock">
```

```
</div>
  2/79 [37m━━━━━━━━━━━━━━━━━━━━  1:21 1s/step - accuracy: 0.6641 - loss: 0.5981

<div class="k-default-codeblock">
```

```
</div>
  3/79 [37m━━━━━━━━━━━━━━━━━━━━  1:16 1s/step - accuracy: 0.6788 - loss: 0.5866

<div class="k-default-codeblock">
```

```
</div>
  4/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:14 996ms/step - accuracy: 0.6888 - loss: 0.5795

<div class="k-default-codeblock">
```

```
</div>
  5/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:12 985ms/step - accuracy: 0.6885 - loss: 0.5754

<div class="k-default-codeblock">
```

```
</div>
  6/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:11 982ms/step - accuracy: 0.6918 - loss: 0.5718

<div class="k-default-codeblock">
```

```
</div>
  7/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:10 981ms/step - accuracy: 0.6950 - loss: 0.5693

<div class="k-default-codeblock">
```

```
</div>
  8/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:09 985ms/step - accuracy: 0.6961 - loss: 0.5685

<div class="k-default-codeblock">
```

```
</div>
  9/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:08 978ms/step - accuracy: 0.6963 - loss: 0.5682

<div class="k-default-codeblock">
```

```
</div>
 10/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:08 987ms/step - accuracy: 0.6963 - loss: 0.5681

<div class="k-default-codeblock">
```

```
</div>
 11/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:06 978ms/step - accuracy: 0.6978 - loss: 0.5673

<div class="k-default-codeblock">
```

```
</div>
 12/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:05 979ms/step - accuracy: 0.6998 - loss: 0.5666

<div class="k-default-codeblock">
```

```
</div>
 13/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:04 983ms/step - accuracy: 0.7016 - loss: 0.5662

<div class="k-default-codeblock">
```

```
</div>
 14/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:03 981ms/step - accuracy: 0.7025 - loss: 0.5661

<div class="k-default-codeblock">
```

```
</div>
 15/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:02 982ms/step - accuracy: 0.7035 - loss: 0.5661

<div class="k-default-codeblock">
```

```
</div>
 16/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:01 982ms/step - accuracy: 0.7043 - loss: 0.5662

<div class="k-default-codeblock">
```

```
</div>
 17/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:00 982ms/step - accuracy: 0.7053 - loss: 0.5663

<div class="k-default-codeblock">
```

```
</div>
 18/79 ━━━━[37m━━━━━━━━━━━━━━━━  59s 979ms/step - accuracy: 0.7064 - loss: 0.5662 

<div class="k-default-codeblock">
```

```
</div>
 19/79 ━━━━[37m━━━━━━━━━━━━━━━━  58s 975ms/step - accuracy: 0.7074 - loss: 0.5662

<div class="k-default-codeblock">
```

```
</div>
 20/79 ━━━━━[37m━━━━━━━━━━━━━━━  57s 975ms/step - accuracy: 0.7082 - loss: 0.5662

<div class="k-default-codeblock">
```

```
</div>
 21/79 ━━━━━[37m━━━━━━━━━━━━━━━  56s 973ms/step - accuracy: 0.7092 - loss: 0.5663

<div class="k-default-codeblock">
```

```
</div>
 22/79 ━━━━━[37m━━━━━━━━━━━━━━━  55s 975ms/step - accuracy: 0.7098 - loss: 0.5665

<div class="k-default-codeblock">
```

```
</div>
 23/79 ━━━━━[37m━━━━━━━━━━━━━━━  54s 977ms/step - accuracy: 0.7103 - loss: 0.5668

<div class="k-default-codeblock">
```

```
</div>
 24/79 ━━━━━━[37m━━━━━━━━━━━━━━  53s 979ms/step - accuracy: 0.7107 - loss: 0.5670

<div class="k-default-codeblock">
```

```
</div>
 25/79 ━━━━━━[37m━━━━━━━━━━━━━━  52s 978ms/step - accuracy: 0.7110 - loss: 0.5671

<div class="k-default-codeblock">
```

```
</div>
 26/79 ━━━━━━[37m━━━━━━━━━━━━━━  51s 978ms/step - accuracy: 0.7113 - loss: 0.5673

<div class="k-default-codeblock">
```

```
</div>
 27/79 ━━━━━━[37m━━━━━━━━━━━━━━  50s 977ms/step - accuracy: 0.7117 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 28/79 ━━━━━━━[37m━━━━━━━━━━━━━  49s 974ms/step - accuracy: 0.7121 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 29/79 ━━━━━━━[37m━━━━━━━━━━━━━  48s 973ms/step - accuracy: 0.7126 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 30/79 ━━━━━━━[37m━━━━━━━━━━━━━  47s 972ms/step - accuracy: 0.7130 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 31/79 ━━━━━━━[37m━━━━━━━━━━━━━  46s 971ms/step - accuracy: 0.7136 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 32/79 ━━━━━━━━[37m━━━━━━━━━━━━  45s 972ms/step - accuracy: 0.7140 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 33/79 ━━━━━━━━[37m━━━━━━━━━━━━  44s 973ms/step - accuracy: 0.7145 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 34/79 ━━━━━━━━[37m━━━━━━━━━━━━  43s 973ms/step - accuracy: 0.7148 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 35/79 ━━━━━━━━[37m━━━━━━━━━━━━  43s 992ms/step - accuracy: 0.7152 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 36/79 ━━━━━━━━━[37m━━━━━━━━━━━  42s 999ms/step - accuracy: 0.7157 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 37/79 ━━━━━━━━━[37m━━━━━━━━━━━  41s 1000ms/step - accuracy: 0.7160 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 38/79 ━━━━━━━━━[37m━━━━━━━━━━━  40s 998ms/step - accuracy: 0.7163 - loss: 0.5673 

<div class="k-default-codeblock">
```

```
</div>
 39/79 ━━━━━━━━━[37m━━━━━━━━━━━  39s 998ms/step - accuracy: 0.7166 - loss: 0.5673

<div class="k-default-codeblock">
```

```
</div>
 40/79 ━━━━━━━━━━[37m━━━━━━━━━━  38s 998ms/step - accuracy: 0.7168 - loss: 0.5673

<div class="k-default-codeblock">
```

```
</div>
 41/79 ━━━━━━━━━━[37m━━━━━━━━━━  37s 999ms/step - accuracy: 0.7169 - loss: 0.5673

<div class="k-default-codeblock">
```

```
</div>
 42/79 ━━━━━━━━━━[37m━━━━━━━━━━  36s 997ms/step - accuracy: 0.7170 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 43/79 ━━━━━━━━━━[37m━━━━━━━━━━  35s 996ms/step - accuracy: 0.7171 - loss: 0.5674

<div class="k-default-codeblock">
```

```
</div>
 44/79 ━━━━━━━━━━━[37m━━━━━━━━━  34s 995ms/step - accuracy: 0.7171 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 45/79 ━━━━━━━━━━━[37m━━━━━━━━━  33s 994ms/step - accuracy: 0.7171 - loss: 0.5675

<div class="k-default-codeblock">
```

```
</div>
 46/79 ━━━━━━━━━━━[37m━━━━━━━━━  32s 994ms/step - accuracy: 0.7172 - loss: 0.5676

<div class="k-default-codeblock">
```

```
</div>
 47/79 ━━━━━━━━━━━[37m━━━━━━━━━  31s 992ms/step - accuracy: 0.7173 - loss: 0.5676

<div class="k-default-codeblock">
```

```
</div>
 48/79 ━━━━━━━━━━━━[37m━━━━━━━━  30s 991ms/step - accuracy: 0.7174 - loss: 0.5676

<div class="k-default-codeblock">
```

```
</div>
 49/79 ━━━━━━━━━━━━[37m━━━━━━━━  29s 990ms/step - accuracy: 0.7175 - loss: 0.5676

<div class="k-default-codeblock">
```

```
</div>
 50/79 ━━━━━━━━━━━━[37m━━━━━━━━  28s 990ms/step - accuracy: 0.7176 - loss: 0.5677

<div class="k-default-codeblock">
```

```
</div>
 51/79 ━━━━━━━━━━━━[37m━━━━━━━━  27s 989ms/step - accuracy: 0.7176 - loss: 0.5677

<div class="k-default-codeblock">
```

```
</div>
 52/79 ━━━━━━━━━━━━━[37m━━━━━━━  26s 989ms/step - accuracy: 0.7176 - loss: 0.5677

<div class="k-default-codeblock">
```

```
</div>
 53/79 ━━━━━━━━━━━━━[37m━━━━━━━  25s 989ms/step - accuracy: 0.7176 - loss: 0.5678

<div class="k-default-codeblock">
```

```
</div>
 54/79 ━━━━━━━━━━━━━[37m━━━━━━━  24s 989ms/step - accuracy: 0.7176 - loss: 0.5678

<div class="k-default-codeblock">
```

```
</div>
 55/79 ━━━━━━━━━━━━━[37m━━━━━━━  23s 988ms/step - accuracy: 0.7177 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 56/79 ━━━━━━━━━━━━━━[37m━━━━━━  22s 988ms/step - accuracy: 0.7177 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 57/79 ━━━━━━━━━━━━━━[37m━━━━━━  21s 988ms/step - accuracy: 0.7178 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 58/79 ━━━━━━━━━━━━━━[37m━━━━━━  20s 986ms/step - accuracy: 0.7179 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 59/79 ━━━━━━━━━━━━━━[37m━━━━━━  19s 985ms/step - accuracy: 0.7180 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 60/79 ━━━━━━━━━━━━━━━[37m━━━━━  18s 985ms/step - accuracy: 0.7181 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 61/79 ━━━━━━━━━━━━━━━[37m━━━━━  17s 984ms/step - accuracy: 0.7182 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 62/79 ━━━━━━━━━━━━━━━[37m━━━━━  16s 984ms/step - accuracy: 0.7182 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 63/79 ━━━━━━━━━━━━━━━[37m━━━━━  15s 984ms/step - accuracy: 0.7183 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 64/79 ━━━━━━━━━━━━━━━━[37m━━━━  14s 984ms/step - accuracy: 0.7184 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 65/79 ━━━━━━━━━━━━━━━━[37m━━━━  13s 983ms/step - accuracy: 0.7185 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 66/79 ━━━━━━━━━━━━━━━━[37m━━━━  12s 983ms/step - accuracy: 0.7187 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 67/79 ━━━━━━━━━━━━━━━━[37m━━━━  11s 994ms/step - accuracy: 0.7188 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 68/79 ━━━━━━━━━━━━━━━━━[37m━━━  10s 995ms/step - accuracy: 0.7189 - loss: 0.5680

<div class="k-default-codeblock">
```

```
</div>
 69/79 ━━━━━━━━━━━━━━━━━[37m━━━  9s 992ms/step - accuracy: 0.7190 - loss: 0.5680 

<div class="k-default-codeblock">
```

```
</div>
 70/79 ━━━━━━━━━━━━━━━━━[37m━━━  8s 987ms/step - accuracy: 0.7192 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 71/79 ━━━━━━━━━━━━━━━━━[37m━━━  7s 982ms/step - accuracy: 0.7194 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 72/79 ━━━━━━━━━━━━━━━━━━[37m━━  6s 977ms/step - accuracy: 0.7195 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 73/79 ━━━━━━━━━━━━━━━━━━[37m━━  5s 974ms/step - accuracy: 0.7197 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 74/79 ━━━━━━━━━━━━━━━━━━[37m━━  4s 969ms/step - accuracy: 0.7198 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 75/79 ━━━━━━━━━━━━━━━━━━[37m━━  3s 964ms/step - accuracy: 0.7200 - loss: 0.5679

<div class="k-default-codeblock">
```

```
</div>
 76/79 ━━━━━━━━━━━━━━━━━━━[37m━  2s 960ms/step - accuracy: 0.7201 - loss: 0.5678

<div class="k-default-codeblock">
```

```
</div>
 77/79 ━━━━━━━━━━━━━━━━━━━[37m━  1s 957ms/step - accuracy: 0.7203 - loss: 0.5678

<div class="k-default-codeblock">
```

```
</div>
 78/79 ━━━━━━━━━━━━━━━━━━━[37m━  0s 954ms/step - accuracy: 0.7205 - loss: 0.5678

<div class="k-default-codeblock">
```

```
</div>
 79/79 ━━━━━━━━━━━━━━━━━━━━ 0s 944ms/step - accuracy: 0.7206 - loss: 0.5678

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:248: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: input_ids
Received: inputs=['Tensor(shape=torch.Size([4, 512]))']
  warnings.warn(msg)


```
</div>
 79/79 ━━━━━━━━━━━━━━━━━━━━ 78s 945ms/step - accuracy: 0.7208 - loss: 0.5678





<div class="k-default-codeblock">
```
[0.5670130252838135, 0.7336000204086304]

```
</div>
---
## Comparison with Transformer model

Let's compare our FNet Classifier model with a Transformer Classifier model. We
keep all the parameters/hyperparameters the same. For example, we use three
`TransformerEncoder` layers.

We set the number of heads to 2.


```python
NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")


x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transformer_classifier = keras.Model(input_ids, outputs, name="transformer_classifier")


transformer_classifier.summary()
transformer_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "transformer_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_ids           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ token_and_position… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">1,985,536</span> │ input_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionE…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ transformer_encoder │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">198,272</span> │ token_and_positi… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncode…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ transformer_encode… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">198,272</span> │ transformer_enco… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncode…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ transformer_encode… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">198,272</span> │ transformer_enco… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncode…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ not_equal_1         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">NotEqual</span>)          │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ transformer_enco… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │            │ not_equal_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">129</span> │ dropout_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,580,481</span> (9.84 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,580,481</span> (9.84 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
   1/149 [37m━━━━━━━━━━━━━━━━━━━━  19:16 8s/step - accuracy: 0.4375 - loss: 0.7813

<div class="k-default-codeblock">
```

```
</div>
   2/149 [37m━━━━━━━━━━━━━━━━━━━━  9:50 4s/step - accuracy: 0.4531 - loss: 1.3017 

<div class="k-default-codeblock">
```

```
</div>
   3/149 [37m━━━━━━━━━━━━━━━━━━━━  9:32 4s/step - accuracy: 0.4618 - loss: 1.4133

<div class="k-default-codeblock">
```

```
</div>
   4/149 [37m━━━━━━━━━━━━━━━━━━━━  9:45 4s/step - accuracy: 0.4616 - loss: 1.4361

<div class="k-default-codeblock">
```

```
</div>
   5/149 [37m━━━━━━━━━━━━━━━━━━━━  9:36 4s/step - accuracy: 0.4605 - loss: 1.4379

<div class="k-default-codeblock">
```

```
</div>
   6/149 [37m━━━━━━━━━━━━━━━━━━━━  9:34 4s/step - accuracy: 0.4593 - loss: 1.4202

<div class="k-default-codeblock">
```

```
</div>
   7/149 [37m━━━━━━━━━━━━━━━━━━━━  9:32 4s/step - accuracy: 0.4619 - loss: 1.3940

<div class="k-default-codeblock">
```

```
</div>
   8/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:24 4s/step - accuracy: 0.4633 - loss: 1.3713

<div class="k-default-codeblock">
```

```
</div>
   9/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:08 4s/step - accuracy: 0.4662 - loss: 1.3476

<div class="k-default-codeblock">
```

```
</div>
  10/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:00 4s/step - accuracy: 0.4686 - loss: 1.3250

<div class="k-default-codeblock">
```

```
</div>
  11/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:01 4s/step - accuracy: 0.4702 - loss: 1.3035

<div class="k-default-codeblock">
```

```
</div>
  12/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:01 4s/step - accuracy: 0.4733 - loss: 1.2824

<div class="k-default-codeblock">
```

```
</div>
  13/149 ━[37m━━━━━━━━━━━━━━━━━━━  9:00 4s/step - accuracy: 0.4759 - loss: 1.2641

<div class="k-default-codeblock">
```

```
</div>
  14/149 ━[37m━━━━━━━━━━━━━━━━━━━  8:58 4s/step - accuracy: 0.4783 - loss: 1.2478

<div class="k-default-codeblock">
```

```
</div>
  15/149 ━━[37m━━━━━━━━━━━━━━━━━━  8:58 4s/step - accuracy: 0.4808 - loss: 1.2323

<div class="k-default-codeblock">
```

```
</div>
  16/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:04 4s/step - accuracy: 0.4840 - loss: 1.2169

<div class="k-default-codeblock">
```

```
</div>
  17/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:03 4s/step - accuracy: 0.4867 - loss: 1.2025

<div class="k-default-codeblock">
```

```
</div>
  18/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:01 4s/step - accuracy: 0.4890 - loss: 1.1889

<div class="k-default-codeblock">
```

```
</div>
  19/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:06 4s/step - accuracy: 0.4913 - loss: 1.1760

<div class="k-default-codeblock">
```

```
</div>
  20/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:06 4s/step - accuracy: 0.4932 - loss: 1.1638

<div class="k-default-codeblock">
```

```
</div>
  21/149 ━━[37m━━━━━━━━━━━━━━━━━━  9:04 4s/step - accuracy: 0.4952 - loss: 1.1521

<div class="k-default-codeblock">
```

```
</div>
  22/149 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 4s/step - accuracy: 0.4973 - loss: 1.1409

<div class="k-default-codeblock">
```

```
</div>
  23/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:52 4s/step - accuracy: 0.4991 - loss: 1.1304

<div class="k-default-codeblock">
```

```
</div>
  24/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:47 4s/step - accuracy: 0.5009 - loss: 1.1205

<div class="k-default-codeblock">
```

```
</div>
  25/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:50 4s/step - accuracy: 0.5025 - loss: 1.1111

<div class="k-default-codeblock">
```

```
</div>
  26/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:48 4s/step - accuracy: 0.5040 - loss: 1.1022

<div class="k-default-codeblock">
```

```
</div>
  27/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:43 4s/step - accuracy: 0.5053 - loss: 1.0936

<div class="k-default-codeblock">
```

```
</div>
  28/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:40 4s/step - accuracy: 0.5066 - loss: 1.0855

<div class="k-default-codeblock">
```

```
</div>
  29/149 ━━━[37m━━━━━━━━━━━━━━━━━  8:36 4s/step - accuracy: 0.5076 - loss: 1.0778

<div class="k-default-codeblock">
```

```
</div>
  30/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:33 4s/step - accuracy: 0.5086 - loss: 1.0703

<div class="k-default-codeblock">
```

```
</div>
  31/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:27 4s/step - accuracy: 0.5095 - loss: 1.0633

<div class="k-default-codeblock">
```

```
</div>
  32/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:22 4s/step - accuracy: 0.5104 - loss: 1.0565

<div class="k-default-codeblock">
```

```
</div>
  33/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:18 4s/step - accuracy: 0.5112 - loss: 1.0501

<div class="k-default-codeblock">
```

```
</div>
  34/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:13 4s/step - accuracy: 0.5119 - loss: 1.0438

<div class="k-default-codeblock">
```

```
</div>
  35/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:07 4s/step - accuracy: 0.5126 - loss: 1.0379

<div class="k-default-codeblock">
```

```
</div>
  36/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:02 4s/step - accuracy: 0.5133 - loss: 1.0321

<div class="k-default-codeblock">
```

```
</div>
  37/149 ━━━━[37m━━━━━━━━━━━━━━━━  8:00 4s/step - accuracy: 0.5140 - loss: 1.0265

<div class="k-default-codeblock">
```

```
</div>
  38/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:56 4s/step - accuracy: 0.5146 - loss: 1.0212

<div class="k-default-codeblock">
```

```
</div>
  39/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:52 4s/step - accuracy: 0.5152 - loss: 1.0161

<div class="k-default-codeblock">
```

```
</div>
  40/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:47 4s/step - accuracy: 0.5157 - loss: 1.0112

<div class="k-default-codeblock">
```

```
</div>
  41/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:42 4s/step - accuracy: 0.5161 - loss: 1.0065

<div class="k-default-codeblock">
```

```
</div>
  42/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:41 4s/step - accuracy: 0.5164 - loss: 1.0020

<div class="k-default-codeblock">
```

```
</div>
  43/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:37 4s/step - accuracy: 0.5167 - loss: 0.9976

<div class="k-default-codeblock">
```

```
</div>
  44/149 ━━━━━[37m━━━━━━━━━━━━━━━  7:32 4s/step - accuracy: 0.5170 - loss: 0.9934

<div class="k-default-codeblock">
```

```
</div>
  45/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:27 4s/step - accuracy: 0.5173 - loss: 0.9893

<div class="k-default-codeblock">
```

```
</div>
  46/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:24 4s/step - accuracy: 0.5175 - loss: 0.9854

<div class="k-default-codeblock">
```

```
</div>
  47/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:19 4s/step - accuracy: 0.5177 - loss: 0.9818

<div class="k-default-codeblock">
```

```
</div>
  48/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:14 4s/step - accuracy: 0.5178 - loss: 0.9781

<div class="k-default-codeblock">
```

```
</div>
  49/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:12 4s/step - accuracy: 0.5180 - loss: 0.9746

<div class="k-default-codeblock">
```

```
</div>
  50/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:07 4s/step - accuracy: 0.5181 - loss: 0.9713

<div class="k-default-codeblock">
```

```
</div>
  51/149 ━━━━━━[37m━━━━━━━━━━━━━━  7:03 4s/step - accuracy: 0.5182 - loss: 0.9679

<div class="k-default-codeblock">
```

```
</div>
  52/149 ━━━━━━[37m━━━━━━━━━━━━━━  6:59 4s/step - accuracy: 0.5183 - loss: 0.9647

<div class="k-default-codeblock">
```

```
</div>
  53/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:53 4s/step - accuracy: 0.5184 - loss: 0.9616

<div class="k-default-codeblock">
```

```
</div>
  54/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:51 4s/step - accuracy: 0.5184 - loss: 0.9586

<div class="k-default-codeblock">
```

```
</div>
  55/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:47 4s/step - accuracy: 0.5185 - loss: 0.9556

<div class="k-default-codeblock">
```

```
</div>
  56/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:45 4s/step - accuracy: 0.5186 - loss: 0.9527

<div class="k-default-codeblock">
```

```
</div>
  57/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:40 4s/step - accuracy: 0.5187 - loss: 0.9499

<div class="k-default-codeblock">
```

```
</div>
  58/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:36 4s/step - accuracy: 0.5189 - loss: 0.9471

<div class="k-default-codeblock">
```

```
</div>
  59/149 ━━━━━━━[37m━━━━━━━━━━━━━  6:32 4s/step - accuracy: 0.5191 - loss: 0.9444

<div class="k-default-codeblock">
```

```
</div>
  60/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:27 4s/step - accuracy: 0.5191 - loss: 0.9418

<div class="k-default-codeblock">
```

```
</div>
  61/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:23 4s/step - accuracy: 0.5192 - loss: 0.9392

<div class="k-default-codeblock">
```

```
</div>
  62/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:18 4s/step - accuracy: 0.5193 - loss: 0.9367

<div class="k-default-codeblock">
```

```
</div>
  63/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:13 4s/step - accuracy: 0.5193 - loss: 0.9342

<div class="k-default-codeblock">
```

```
</div>
  64/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:09 4s/step - accuracy: 0.5194 - loss: 0.9318

<div class="k-default-codeblock">
```

```
</div>
  65/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:05 4s/step - accuracy: 0.5194 - loss: 0.9295

<div class="k-default-codeblock">
```

```
</div>
  66/149 ━━━━━━━━[37m━━━━━━━━━━━━  6:00 4s/step - accuracy: 0.5194 - loss: 0.9272

<div class="k-default-codeblock">
```

```
</div>
  67/149 ━━━━━━━━[37m━━━━━━━━━━━━  5:56 4s/step - accuracy: 0.5195 - loss: 0.9250

<div class="k-default-codeblock">
```

```
</div>
  68/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:52 4s/step - accuracy: 0.5195 - loss: 0.9228

<div class="k-default-codeblock">
```

```
</div>
  69/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:48 4s/step - accuracy: 0.5195 - loss: 0.9207

<div class="k-default-codeblock">
```

```
</div>
  70/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:43 4s/step - accuracy: 0.5196 - loss: 0.9186

<div class="k-default-codeblock">
```

```
</div>
  71/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:40 4s/step - accuracy: 0.5196 - loss: 0.9165

<div class="k-default-codeblock">
```

```
</div>
  72/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:36 4s/step - accuracy: 0.5197 - loss: 0.9145

<div class="k-default-codeblock">
```

```
</div>
  73/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:31 4s/step - accuracy: 0.5197 - loss: 0.9126

<div class="k-default-codeblock">
```

```
</div>
  74/149 ━━━━━━━━━[37m━━━━━━━━━━━  5:27 4s/step - accuracy: 0.5197 - loss: 0.9107

<div class="k-default-codeblock">
```

```
</div>
  75/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:23 4s/step - accuracy: 0.5197 - loss: 0.9088

<div class="k-default-codeblock">
```

```
</div>
  76/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:19 4s/step - accuracy: 0.5197 - loss: 0.9069

<div class="k-default-codeblock">
```

```
</div>
  77/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:15 4s/step - accuracy: 0.5197 - loss: 0.9052

<div class="k-default-codeblock">
```

```
</div>
  78/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:11 4s/step - accuracy: 0.5197 - loss: 0.9034

<div class="k-default-codeblock">
```

```
</div>
  79/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:08 4s/step - accuracy: 0.5197 - loss: 0.9017

<div class="k-default-codeblock">
```

```
</div>
  80/149 ━━━━━━━━━━[37m━━━━━━━━━━  5:01 4s/step - accuracy: 0.5197 - loss: 0.9000

<div class="k-default-codeblock">
```

```
</div>
  81/149 ━━━━━━━━━━[37m━━━━━━━━━━  4:57 4s/step - accuracy: 0.5197 - loss: 0.8983

<div class="k-default-codeblock">
```

```
</div>
  82/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:53 4s/step - accuracy: 0.5197 - loss: 0.8967

<div class="k-default-codeblock">
```

```
</div>
  83/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:48 4s/step - accuracy: 0.5197 - loss: 0.8951

<div class="k-default-codeblock">
```

```
</div>
  84/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:44 4s/step - accuracy: 0.5197 - loss: 0.8935

<div class="k-default-codeblock">
```

```
</div>
  85/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:40 4s/step - accuracy: 0.5198 - loss: 0.8919

<div class="k-default-codeblock">
```

```
</div>
  86/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:36 4s/step - accuracy: 0.5199 - loss: 0.8903

<div class="k-default-codeblock">
```

```
</div>
  87/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:31 4s/step - accuracy: 0.5200 - loss: 0.8888

<div class="k-default-codeblock">
```

```
</div>
  88/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:27 4s/step - accuracy: 0.5200 - loss: 0.8873

<div class="k-default-codeblock">
```

```
</div>
  89/149 ━━━━━━━━━━━[37m━━━━━━━━━  4:23 4s/step - accuracy: 0.5201 - loss: 0.8858

<div class="k-default-codeblock">
```

```
</div>
  90/149 ━━━━━━━━━━━━[37m━━━━━━━━  4:19 4s/step - accuracy: 0.5202 - loss: 0.8843

<div class="k-default-codeblock">
```

```
</div>
  91/149 ━━━━━━━━━━━━[37m━━━━━━━━  4:14 4s/step - accuracy: 0.5204 - loss: 0.8829

<div class="k-default-codeblock">
```

```
</div>
  92/149 ━━━━━━━━━━━━[37m━━━━━━━━  4:10 4s/step - accuracy: 0.5205 - loss: 0.8815

<div class="k-default-codeblock">
```

```
</div>
  93/149 ━━━━━━━━━━━━[37m━━━━━━━━  4:05 4s/step - accuracy: 0.5206 - loss: 0.8800

<div class="k-default-codeblock">
```

```
</div>
  94/149 ━━━━━━━━━━━━[37m━━━━━━━━  4:02 4s/step - accuracy: 0.5208 - loss: 0.8786

<div class="k-default-codeblock">
```

```
</div>
  95/149 ━━━━━━━━━━━━[37m━━━━━━━━  3:57 4s/step - accuracy: 0.5209 - loss: 0.8772

<div class="k-default-codeblock">
```

```
</div>
  96/149 ━━━━━━━━━━━━[37m━━━━━━━━  3:53 4s/step - accuracy: 0.5211 - loss: 0.8758

<div class="k-default-codeblock">
```

```
</div>
  97/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:49 4s/step - accuracy: 0.5213 - loss: 0.8744

<div class="k-default-codeblock">
```

```
</div>
  98/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:44 4s/step - accuracy: 0.5215 - loss: 0.8731

<div class="k-default-codeblock">
```

```
</div>
  99/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:40 4s/step - accuracy: 0.5218 - loss: 0.8717

<div class="k-default-codeblock">
```

```
</div>
 100/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:36 4s/step - accuracy: 0.5220 - loss: 0.8704

<div class="k-default-codeblock">
```

```
</div>
 101/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:32 4s/step - accuracy: 0.5223 - loss: 0.8690

<div class="k-default-codeblock">
```

```
</div>
 102/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:27 4s/step - accuracy: 0.5225 - loss: 0.8677

<div class="k-default-codeblock">
```

```
</div>
 103/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:23 4s/step - accuracy: 0.5228 - loss: 0.8664

<div class="k-default-codeblock">
```

```
</div>
 104/149 ━━━━━━━━━━━━━[37m━━━━━━━  3:19 4s/step - accuracy: 0.5231 - loss: 0.8651

<div class="k-default-codeblock">
```

```
</div>
 105/149 ━━━━━━━━━━━━━━[37m━━━━━━  3:15 4s/step - accuracy: 0.5234 - loss: 0.8638

<div class="k-default-codeblock">
```

```
</div>
 106/149 ━━━━━━━━━━━━━━[37m━━━━━━  3:11 4s/step - accuracy: 0.5236 - loss: 0.8626

<div class="k-default-codeblock">
```

```
</div>
 107/149 ━━━━━━━━━━━━━━[37m━━━━━━  3:06 4s/step - accuracy: 0.5239 - loss: 0.8614

<div class="k-default-codeblock">
```

```
</div>
 108/149 ━━━━━━━━━━━━━━[37m━━━━━━  3:02 4s/step - accuracy: 0.5242 - loss: 0.8601

<div class="k-default-codeblock">
```

```
</div>
 109/149 ━━━━━━━━━━━━━━[37m━━━━━━  2:57 4s/step - accuracy: 0.5245 - loss: 0.8589

<div class="k-default-codeblock">
```

```
</div>
 110/149 ━━━━━━━━━━━━━━[37m━━━━━━  2:53 4s/step - accuracy: 0.5249 - loss: 0.8577

<div class="k-default-codeblock">
```

```
</div>
 111/149 ━━━━━━━━━━━━━━[37m━━━━━━  2:48 4s/step - accuracy: 0.5252 - loss: 0.8565

<div class="k-default-codeblock">
```

```
</div>
 112/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:44 4s/step - accuracy: 0.5255 - loss: 0.8553

<div class="k-default-codeblock">
```

```
</div>
 113/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:40 4s/step - accuracy: 0.5259 - loss: 0.8541

<div class="k-default-codeblock">
```

```
</div>
 114/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:35 4s/step - accuracy: 0.5262 - loss: 0.8530

<div class="k-default-codeblock">
```

```
</div>
 115/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:31 4s/step - accuracy: 0.5265 - loss: 0.8518

<div class="k-default-codeblock">
```

```
</div>
 116/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:26 4s/step - accuracy: 0.5269 - loss: 0.8507

<div class="k-default-codeblock">
```

```
</div>
 117/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:22 4s/step - accuracy: 0.5272 - loss: 0.8495

<div class="k-default-codeblock">
```

```
</div>
 118/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:18 4s/step - accuracy: 0.5276 - loss: 0.8484

<div class="k-default-codeblock">
```

```
</div>
 119/149 ━━━━━━━━━━━━━━━[37m━━━━━  2:13 4s/step - accuracy: 0.5279 - loss: 0.8473

<div class="k-default-codeblock">
```

```
</div>
 120/149 ━━━━━━━━━━━━━━━━[37m━━━━  2:09 4s/step - accuracy: 0.5283 - loss: 0.8462

<div class="k-default-codeblock">
```

```
</div>
 121/149 ━━━━━━━━━━━━━━━━[37m━━━━  2:05 4s/step - accuracy: 0.5287 - loss: 0.8450

<div class="k-default-codeblock">
```

```
</div>
 122/149 ━━━━━━━━━━━━━━━━[37m━━━━  2:00 4s/step - accuracy: 0.5291 - loss: 0.8439

<div class="k-default-codeblock">
```

```
</div>
 123/149 ━━━━━━━━━━━━━━━━[37m━━━━  1:56 4s/step - accuracy: 0.5295 - loss: 0.8428

<div class="k-default-codeblock">
```

```
</div>
 124/149 ━━━━━━━━━━━━━━━━[37m━━━━  1:51 4s/step - accuracy: 0.5299 - loss: 0.8417

<div class="k-default-codeblock">
```

```
</div>
 125/149 ━━━━━━━━━━━━━━━━[37m━━━━  1:47 4s/step - accuracy: 0.5303 - loss: 0.8405

<div class="k-default-codeblock">
```

```
</div>
 126/149 ━━━━━━━━━━━━━━━━[37m━━━━  1:42 4s/step - accuracy: 0.5307 - loss: 0.8394

<div class="k-default-codeblock">
```

```
</div>
 127/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:38 4s/step - accuracy: 0.5311 - loss: 0.8383

<div class="k-default-codeblock">
```

```
</div>
 128/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:33 4s/step - accuracy: 0.5316 - loss: 0.8372

<div class="k-default-codeblock">
```

```
</div>
 129/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:29 4s/step - accuracy: 0.5320 - loss: 0.8361

<div class="k-default-codeblock">
```

```
</div>
 130/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:24 4s/step - accuracy: 0.5325 - loss: 0.8350

<div class="k-default-codeblock">
```

```
</div>
 131/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:20 4s/step - accuracy: 0.5329 - loss: 0.8340

<div class="k-default-codeblock">
```

```
</div>
 132/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:15 4s/step - accuracy: 0.5334 - loss: 0.8329

<div class="k-default-codeblock">
```

```
</div>
 133/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:11 4s/step - accuracy: 0.5338 - loss: 0.8318

<div class="k-default-codeblock">
```

```
</div>
 134/149 ━━━━━━━━━━━━━━━━━[37m━━━  1:06 4s/step - accuracy: 0.5343 - loss: 0.8308

<div class="k-default-codeblock">
```

```
</div>
 135/149 ━━━━━━━━━━━━━━━━━━[37m━━  1:02 4s/step - accuracy: 0.5347 - loss: 0.8297

<div class="k-default-codeblock">
```

```
</div>
 136/149 ━━━━━━━━━━━━━━━━━━[37m━━  58s 4s/step - accuracy: 0.5352 - loss: 0.8287 

<div class="k-default-codeblock">
```

```
</div>
 137/149 ━━━━━━━━━━━━━━━━━━[37m━━  53s 4s/step - accuracy: 0.5356 - loss: 0.8276

<div class="k-default-codeblock">
```

```
</div>
 138/149 ━━━━━━━━━━━━━━━━━━[37m━━  49s 4s/step - accuracy: 0.5361 - loss: 0.8266

<div class="k-default-codeblock">
```

```
</div>
 139/149 ━━━━━━━━━━━━━━━━━━[37m━━  44s 4s/step - accuracy: 0.5366 - loss: 0.8255

<div class="k-default-codeblock">
```

```
</div>
 140/149 ━━━━━━━━━━━━━━━━━━[37m━━  40s 4s/step - accuracy: 0.5370 - loss: 0.8245

<div class="k-default-codeblock">
```

```
</div>
 141/149 ━━━━━━━━━━━━━━━━━━[37m━━  35s 4s/step - accuracy: 0.5375 - loss: 0.8235

<div class="k-default-codeblock">
```

```
</div>
 142/149 ━━━━━━━━━━━━━━━━━━━[37m━  31s 4s/step - accuracy: 0.5380 - loss: 0.8224

<div class="k-default-codeblock">
```

```
</div>
 143/149 ━━━━━━━━━━━━━━━━━━━[37m━  26s 4s/step - accuracy: 0.5385 - loss: 0.8214

<div class="k-default-codeblock">
```

```
</div>
 144/149 ━━━━━━━━━━━━━━━━━━━[37m━  22s 4s/step - accuracy: 0.5390 - loss: 0.8204

<div class="k-default-codeblock">
```

```
</div>
 145/149 ━━━━━━━━━━━━━━━━━━━[37m━  17s 4s/step - accuracy: 0.5395 - loss: 0.8193

<div class="k-default-codeblock">
```

```
</div>
 146/149 ━━━━━━━━━━━━━━━━━━━[37m━  13s 4s/step - accuracy: 0.5400 - loss: 0.8183

<div class="k-default-codeblock">
```

```
</div>
 147/149 ━━━━━━━━━━━━━━━━━━━[37m━  8s 4s/step - accuracy: 0.5405 - loss: 0.8173 

<div class="k-default-codeblock">
```

```
</div>
 148/149 ━━━━━━━━━━━━━━━━━━━[37m━  4s 4s/step - accuracy: 0.5410 - loss: 0.8163

<div class="k-default-codeblock">
```

```
</div>
 149/149 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.5415 - loss: 0.8152

<div class="k-default-codeblock">
```

```
</div>
 149/149 ━━━━━━━━━━━━━━━━━━━━ 684s 5s/step - accuracy: 0.5420 - loss: 0.8142 - val_accuracy: 0.8560 - val_loss: 0.3914





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7c51ab962990>

```
</div>
We obtain a train accuracy of around 94% and a validation accuracy of around
86.5%. It takes around 146 seconds to train the model (on Colab with a 16 GB Tesla
T4 GPU).

Let's calculate the test accuracy.


```python
transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
```

    
  1/79 [37m━━━━━━━━━━━━━━━━━━━━  7:39 6s/step - accuracy: 0.8125 - loss: 0.3880

<div class="k-default-codeblock">
```

```
</div>
  2/79 [37m━━━━━━━━━━━━━━━━━━━━  1:58 2s/step - accuracy: 0.8047 - loss: 0.4233

<div class="k-default-codeblock">
```

```
</div>
  3/79 [37m━━━━━━━━━━━━━━━━━━━━  1:57 2s/step - accuracy: 0.8212 - loss: 0.4146

<div class="k-default-codeblock">
```

```
</div>
  4/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:55 2s/step - accuracy: 0.8268 - loss: 0.4106

<div class="k-default-codeblock">
```

```
</div>
  5/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:53 2s/step - accuracy: 0.8340 - loss: 0.4023

<div class="k-default-codeblock">
```

```
</div>
  6/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:51 2s/step - accuracy: 0.8399 - loss: 0.3957

<div class="k-default-codeblock">
```

```
</div>
  7/79 ━[37m━━━━━━━━━━━━━━━━━━━  1:49 2s/step - accuracy: 0.8437 - loss: 0.3908

<div class="k-default-codeblock">
```

```
</div>
  8/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:47 2s/step - accuracy: 0.8437 - loss: 0.3906

<div class="k-default-codeblock">
```

```
</div>
  9/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:45 2s/step - accuracy: 0.8433 - loss: 0.3911

<div class="k-default-codeblock">
```

```
</div>
 10/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 2s/step - accuracy: 0.8418 - loss: 0.3920

<div class="k-default-codeblock">
```

```
</div>
 11/79 ━━[37m━━━━━━━━━━━━━━━━━━  1:49 2s/step - accuracy: 0.8414 - loss: 0.3910

<div class="k-default-codeblock">
```

```
</div>
 12/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:48 2s/step - accuracy: 0.8412 - loss: 0.3896

<div class="k-default-codeblock">
```

```
</div>
 13/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:48 2s/step - accuracy: 0.8405 - loss: 0.3891

<div class="k-default-codeblock">
```

```
</div>
 14/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:46 2s/step - accuracy: 0.8398 - loss: 0.3898

<div class="k-default-codeblock">
```

```
</div>
 15/79 ━━━[37m━━━━━━━━━━━━━━━━━  1:44 2s/step - accuracy: 0.8386 - loss: 0.3912

<div class="k-default-codeblock">
```

```
</div>
 16/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:42 2s/step - accuracy: 0.8376 - loss: 0.3924

<div class="k-default-codeblock">
```

```
</div>
 17/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:43 2s/step - accuracy: 0.8366 - loss: 0.3935

<div class="k-default-codeblock">
```

```
</div>
 18/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:41 2s/step - accuracy: 0.8357 - loss: 0.3940

<div class="k-default-codeblock">
```

```
</div>
 19/79 ━━━━[37m━━━━━━━━━━━━━━━━  1:39 2s/step - accuracy: 0.8347 - loss: 0.3946

<div class="k-default-codeblock">
```

```
</div>
 20/79 ━━━━━[37m━━━━━━━━━━━━━━━  1:37 2s/step - accuracy: 0.8338 - loss: 0.3956

<div class="k-default-codeblock">
```

```
</div>
 21/79 ━━━━━[37m━━━━━━━━━━━━━━━  1:38 2s/step - accuracy: 0.8328 - loss: 0.3967

<div class="k-default-codeblock">
```

```
</div>
 22/79 ━━━━━[37m━━━━━━━━━━━━━━━  1:36 2s/step - accuracy: 0.8317 - loss: 0.3979

<div class="k-default-codeblock">
```

```
</div>
 23/79 ━━━━━[37m━━━━━━━━━━━━━━━  1:34 2s/step - accuracy: 0.8308 - loss: 0.3990

<div class="k-default-codeblock">
```

```
</div>
 24/79 ━━━━━━[37m━━━━━━━━━━━━━━  1:32 2s/step - accuracy: 0.8301 - loss: 0.3996

<div class="k-default-codeblock">
```

```
</div>
 25/79 ━━━━━━[37m━━━━━━━━━━━━━━  1:32 2s/step - accuracy: 0.8297 - loss: 0.3999

<div class="k-default-codeblock">
```

```
</div>
 26/79 ━━━━━━[37m━━━━━━━━━━━━━━  1:30 2s/step - accuracy: 0.8293 - loss: 0.4000

<div class="k-default-codeblock">
```

```
</div>
 27/79 ━━━━━━[37m━━━━━━━━━━━━━━  1:28 2s/step - accuracy: 0.8292 - loss: 0.3999

<div class="k-default-codeblock">
```

```
</div>
 28/79 ━━━━━━━[37m━━━━━━━━━━━━━  1:26 2s/step - accuracy: 0.8291 - loss: 0.3998

<div class="k-default-codeblock">
```

```
</div>
 29/79 ━━━━━━━[37m━━━━━━━━━━━━━  1:26 2s/step - accuracy: 0.8292 - loss: 0.3994

<div class="k-default-codeblock">
```

```
</div>
 30/79 ━━━━━━━[37m━━━━━━━━━━━━━  1:24 2s/step - accuracy: 0.8292 - loss: 0.3992

<div class="k-default-codeblock">
```

```
</div>
 31/79 ━━━━━━━[37m━━━━━━━━━━━━━  1:22 2s/step - accuracy: 0.8293 - loss: 0.3989

<div class="k-default-codeblock">
```

```
</div>
 32/79 ━━━━━━━━[37m━━━━━━━━━━━━  1:21 2s/step - accuracy: 0.8294 - loss: 0.3987

<div class="k-default-codeblock">
```

```
</div>
 33/79 ━━━━━━━━[37m━━━━━━━━━━━━  1:19 2s/step - accuracy: 0.8295 - loss: 0.3985

<div class="k-default-codeblock">
```

```
</div>
 34/79 ━━━━━━━━[37m━━━━━━━━━━━━  1:17 2s/step - accuracy: 0.8294 - loss: 0.3986

<div class="k-default-codeblock">
```

```
</div>
 35/79 ━━━━━━━━[37m━━━━━━━━━━━━  1:15 2s/step - accuracy: 0.8295 - loss: 0.3986

<div class="k-default-codeblock">
```

```
</div>
 36/79 ━━━━━━━━━[37m━━━━━━━━━━━  1:14 2s/step - accuracy: 0.8296 - loss: 0.3985

<div class="k-default-codeblock">
```

```
</div>
 37/79 ━━━━━━━━━[37m━━━━━━━━━━━  1:12 2s/step - accuracy: 0.8295 - loss: 0.3986

<div class="k-default-codeblock">
```

```
</div>
 38/79 ━━━━━━━━━[37m━━━━━━━━━━━  1:11 2s/step - accuracy: 0.8295 - loss: 0.3986

<div class="k-default-codeblock">
```

```
</div>
 39/79 ━━━━━━━━━[37m━━━━━━━━━━━  1:09 2s/step - accuracy: 0.8294 - loss: 0.3987

<div class="k-default-codeblock">
```

```
</div>
 40/79 ━━━━━━━━━━[37m━━━━━━━━━━  1:07 2s/step - accuracy: 0.8292 - loss: 0.3989

<div class="k-default-codeblock">
```

```
</div>
 41/79 ━━━━━━━━━━[37m━━━━━━━━━━  1:06 2s/step - accuracy: 0.8290 - loss: 0.3992

<div class="k-default-codeblock">
```

```
</div>
 42/79 ━━━━━━━━━━[37m━━━━━━━━━━  1:04 2s/step - accuracy: 0.8287 - loss: 0.3995

<div class="k-default-codeblock">
```

```
</div>
 43/79 ━━━━━━━━━━[37m━━━━━━━━━━  1:02 2s/step - accuracy: 0.8285 - loss: 0.3998

<div class="k-default-codeblock">
```

```
</div>
 44/79 ━━━━━━━━━━━[37m━━━━━━━━━  1:01 2s/step - accuracy: 0.8283 - loss: 0.4001

<div class="k-default-codeblock">
```

```
</div>
 45/79 ━━━━━━━━━━━[37m━━━━━━━━━  59s 2s/step - accuracy: 0.8280 - loss: 0.4004 

<div class="k-default-codeblock">
```

```
</div>
 46/79 ━━━━━━━━━━━[37m━━━━━━━━━  57s 2s/step - accuracy: 0.8278 - loss: 0.4007

<div class="k-default-codeblock">
```

```
</div>
 47/79 ━━━━━━━━━━━[37m━━━━━━━━━  56s 2s/step - accuracy: 0.8275 - loss: 0.4010

<div class="k-default-codeblock">
```

```
</div>
 48/79 ━━━━━━━━━━━━[37m━━━━━━━━  54s 2s/step - accuracy: 0.8272 - loss: 0.4013

<div class="k-default-codeblock">
```

```
</div>
 49/79 ━━━━━━━━━━━━[37m━━━━━━━━  52s 2s/step - accuracy: 0.8270 - loss: 0.4016

<div class="k-default-codeblock">
```

```
</div>
 50/79 ━━━━━━━━━━━━[37m━━━━━━━━  50s 2s/step - accuracy: 0.8267 - loss: 0.4020

<div class="k-default-codeblock">
```

```
</div>
 51/79 ━━━━━━━━━━━━[37m━━━━━━━━  49s 2s/step - accuracy: 0.8263 - loss: 0.4025

<div class="k-default-codeblock">
```

```
</div>
 52/79 ━━━━━━━━━━━━━[37m━━━━━━━  47s 2s/step - accuracy: 0.8260 - loss: 0.4028

<div class="k-default-codeblock">
```

```
</div>
 53/79 ━━━━━━━━━━━━━[37m━━━━━━━  45s 2s/step - accuracy: 0.8257 - loss: 0.4032

<div class="k-default-codeblock">
```

```
</div>
 54/79 ━━━━━━━━━━━━━[37m━━━━━━━  43s 2s/step - accuracy: 0.8254 - loss: 0.4036

<div class="k-default-codeblock">
```

```
</div>
 55/79 ━━━━━━━━━━━━━[37m━━━━━━━  41s 2s/step - accuracy: 0.8252 - loss: 0.4040

<div class="k-default-codeblock">
```

```
</div>
 56/79 ━━━━━━━━━━━━━━[37m━━━━━━  40s 2s/step - accuracy: 0.8249 - loss: 0.4043

<div class="k-default-codeblock">
```

```
</div>
 57/79 ━━━━━━━━━━━━━━[37m━━━━━━  38s 2s/step - accuracy: 0.8247 - loss: 0.4046

<div class="k-default-codeblock">
```

```
</div>
 58/79 ━━━━━━━━━━━━━━[37m━━━━━━  36s 2s/step - accuracy: 0.8245 - loss: 0.4049

<div class="k-default-codeblock">
```

```
</div>
 59/79 ━━━━━━━━━━━━━━[37m━━━━━━  34s 2s/step - accuracy: 0.8243 - loss: 0.4052

<div class="k-default-codeblock">
```

```
</div>
 60/79 ━━━━━━━━━━━━━━━[37m━━━━━  33s 2s/step - accuracy: 0.8242 - loss: 0.4055

<div class="k-default-codeblock">
```

```
</div>
 61/79 ━━━━━━━━━━━━━━━[37m━━━━━  31s 2s/step - accuracy: 0.8240 - loss: 0.4057

<div class="k-default-codeblock">
```

```
</div>
 62/79 ━━━━━━━━━━━━━━━[37m━━━━━  29s 2s/step - accuracy: 0.8239 - loss: 0.4059

<div class="k-default-codeblock">
```

```
</div>
 63/79 ━━━━━━━━━━━━━━━[37m━━━━━  28s 2s/step - accuracy: 0.8238 - loss: 0.4061

<div class="k-default-codeblock">
```

```
</div>
 64/79 ━━━━━━━━━━━━━━━━[37m━━━━  26s 2s/step - accuracy: 0.8236 - loss: 0.4063

<div class="k-default-codeblock">
```

```
</div>
 65/79 ━━━━━━━━━━━━━━━━[37m━━━━  24s 2s/step - accuracy: 0.8235 - loss: 0.4065

<div class="k-default-codeblock">
```

```
</div>
 66/79 ━━━━━━━━━━━━━━━━[37m━━━━  22s 2s/step - accuracy: 0.8234 - loss: 0.4066

<div class="k-default-codeblock">
```

```
</div>
 67/79 ━━━━━━━━━━━━━━━━[37m━━━━  21s 2s/step - accuracy: 0.8233 - loss: 0.4068

<div class="k-default-codeblock">
```

```
</div>
 68/79 ━━━━━━━━━━━━━━━━━[37m━━━  19s 2s/step - accuracy: 0.8231 - loss: 0.4070

<div class="k-default-codeblock">
```

```
</div>
 69/79 ━━━━━━━━━━━━━━━━━[37m━━━  17s 2s/step - accuracy: 0.8230 - loss: 0.4072

<div class="k-default-codeblock">
```

```
</div>
 70/79 ━━━━━━━━━━━━━━━━━[37m━━━  15s 2s/step - accuracy: 0.8228 - loss: 0.4073

<div class="k-default-codeblock">
```

```
</div>
 71/79 ━━━━━━━━━━━━━━━━━[37m━━━  13s 2s/step - accuracy: 0.8227 - loss: 0.4075

<div class="k-default-codeblock">
```

```
</div>
 72/79 ━━━━━━━━━━━━━━━━━━[37m━━  12s 2s/step - accuracy: 0.8226 - loss: 0.4076

<div class="k-default-codeblock">
```

```
</div>
 73/79 ━━━━━━━━━━━━━━━━━━[37m━━  10s 2s/step - accuracy: 0.8225 - loss: 0.4077

<div class="k-default-codeblock">
```

```
</div>
 74/79 ━━━━━━━━━━━━━━━━━━[37m━━  8s 2s/step - accuracy: 0.8224 - loss: 0.4078 

<div class="k-default-codeblock">
```

```
</div>
 75/79 ━━━━━━━━━━━━━━━━━━[37m━━  6s 2s/step - accuracy: 0.8223 - loss: 0.4079

<div class="k-default-codeblock">
```

```
</div>
 76/79 ━━━━━━━━━━━━━━━━━━━[37m━  5s 2s/step - accuracy: 0.8223 - loss: 0.4080

<div class="k-default-codeblock">
```

```
</div>
 77/79 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - accuracy: 0.8222 - loss: 0.4080

<div class="k-default-codeblock">
```

```
</div>
 78/79 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - accuracy: 0.8221 - loss: 0.4081

<div class="k-default-codeblock">
```

```
</div>
 79/79 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.8220 - loss: 0.4081

<div class="k-default-codeblock">
```

```
</div>
 79/79 ━━━━━━━━━━━━━━━━━━━━ 138s 2s/step - accuracy: 0.8219 - loss: 0.4082





<div class="k-default-codeblock">
```
[0.41192713379859924, 0.8163999915122986]

```
</div>
Let's make a table and compare the two models. We can see that FNet
significantly speeds up our run time (1.7x), with only a small sacrifice in
overall accuracy (drop of 0.75%).

|                         | **FNet Classifier** | **Transformer Classifier** |
|:-----------------------:|:-------------------:|:--------------------------:|
|    **Training Time**    |      86 seconds     |         146 seconds        |
|    **Train Accuracy**   |        92.34%       |           93.85%           |
| **Validation Accuracy** |        85.21%       |           86.42%           |
|    **Test Accuracy**    |        83.94%       |           84.69%           |
|       **#Params**       |      2,321,921      |          2,520,065         |
