# Float8 training and inference with a simple Transformer model

**Author:** [Hongyu Chiu](https://github.com/james77777778)<br>
**Date created:** 2024/05/14<br>
**Last modified:** 2024/05/14<br>
**Description:** Train a simple Transformer model with the float8 quantization.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/float8_training_and_inference_with_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/float8_training_and_inference_with_transformer.py)



---
## Introduction

As the number of parameters in Transformer models continues to grow, training
and inference become highly memory and compute-intensive. Therefore, 8-bit
floating point (FP8) was introduced, offering improved performance over 16-bit
floating point with nearly no degradation in accuracy.

In detail, there are two distinct types of FP8: E4M3 and E5M2, useful in
different parts of training.
- E4M3: It consists of 1 sign bit, 4 exponent bits and 3 bits of mantissa. It
    can store values up to +/-448 and nan.
- E5M2: It consists of 1 sign bit, 5 exponent bits and 2 bits of mantissa. It
    can store values up to +/-57344, +/-inf and nan. The tradeoff of the
    increased dynamic range is lower precision of the stored values.
Typically, E4M3 is best used during the forward pass because activations and
weights require more precision. In the backward pass, however, E5M2 is utilized
because gradients are less susceptible to the loss of precision but require
higher dynamic range.

It is worth noting that FP8 inference deployment is greatly simplified, as
inference and training use the same datatype. This is in contrast to INT8
inference with networks trained in 32- or 16-bit floating point, which require
post-training quantization (PTQ) calibration and even quantization-aware
training (QAT) in order to maintain model accuracy.

In this example, we will build a simple Transformer model and train it with
both FP16 and FP8 precision. You will observe that the accuracy doesn't decrease
with lower precision.

Note: You will need a decent GPU with FP8 Tensor Cores support for the expected
performance improvement.

---
## Setup

We will use KerasNLP library to simplify the model implementation. Additionally,
use mixed precision training to reduce the training time.

Note: The dependency on TensorFlow is only required for data processing.


```python
!pip install -q --upgrade git+https://github.com/keras-team/keras-nlp.git  # Get the latest version of KerasNLP
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import keras
import keras_nlp
import tensorflow as tf

keras.config.set_dtype_policy("mixed_bfloat16")
```

Define some hyperparameters.


```python
EPOCHS = 3
BATCH_SIZE = 32
VOCABULARY_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
MODEL_KWARGS = dict(
    vocabulary_size=VOCABULARY_SIZE,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    hidden_dim=32,  # Hidden size for each token
    num_heads=2,  # Number of attention heads
    intermediate_dim=32,  # Intermediate size in feedforward network
    dropout=0.1,  # Dropout rate
)
```

---
## Dataset

First, let's download the IMDB dataset and extract it.


```python
!mkdir -p datasets
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O datasets/aclImdb_v1.tar.gz
!mkdir -p datasets/aclImdb
!tar -xzf datasets/aclImdb_v1.tar.gz -C datasets
!rm -rf datasets/aclImdb/train/unsup
```

<div class="k-default-codeblock">
```
--2024-05-20 10:21:19--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... 

connected.
HTTP request sent, awaiting response... 

200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘datasets/aclImdb_v1.tar.gz’
```
</div>
    
    
<div class="k-default-codeblock">
```
      datasets/   0%[                    ]       0  --.-KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   0%[                    ]  19.49K  80.9KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   0%[                    ]  36.46K  76.1KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   0%[                    ]  62.84K  87.7KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   0%[                    ]  88.30K  92.4KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   0%[                    ] 116.58K  97.6KB/s               

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   0%[                    ] 146.76K   102KB/s               

```
</div>
    
   datasets/aclImdb   0%[                    ] 180.69K   108KB/s               

    
  datasets/aclImdb_   0%[                    ] 211.80K   110KB/s               

    
 datasets/aclImdb_v   0%[                    ] 251.40K   116KB/s               

    
datasets/aclImdb_v1   0%[                    ] 291.92K   122KB/s               

    
atasets/aclImdb_v1.   0%[                    ] 327.76K   124KB/s               

    
tasets/aclImdb_v1.t   0%[                    ] 367.35K   128KB/s               

    
asets/aclImdb_v1.ta   0%[                    ] 406.94K   131KB/s    eta 10m 25s

    
sets/aclImdb_v1.tar   0%[                    ] 453.12K   135KB/s    eta 10m 25s

    
ets/aclImdb_v1.tar.   0%[                    ] 495.55K   138KB/s    eta 10m 25s

    
ts/aclImdb_v1.tar.g   0%[                    ] 537.04K   140KB/s    eta 10m 25s

    
s/aclImdb_v1.tar.gz   0%[                    ] 586.53K   144KB/s    eta 10m 25s

    
/aclImdb_v1.tar.gz    0%[                    ] 636.02K   148KB/s    eta 9m 12s 

    
aclImdb_v1.tar.gz     0%[                    ] 696.83K   153KB/s    eta 9m 12s 

    
clImdb_v1.tar.gz      0%[                    ] 754.80K   157KB/s    eta 9m 12s 

    
lImdb_v1.tar.gz       0%[                    ] 818.44K   166KB/s    eta 9m 12s 

    
Imdb_v1.tar.gz        1%[                    ] 886.31K   176KB/s    eta 8m 8s  

    
mdb_v1.tar.gz         1%[                    ] 955.60K   183KB/s    eta 8m 8s  

    
db_v1.tar.gz          1%[                    ]   1.01M   193KB/s    eta 8m 8s  

    
b_v1.tar.gz           1%[                    ]   1.09M   203KB/s    eta 8m 8s  

    
_v1.tar.gz            1%[                    ]   1.12M   199KB/s    eta 7m 36s 

    
v1.tar.gz             1%[                    ]   1.25M   220KB/s    eta 7m 36s 

    
1.tar.gz              1%[                    ]   1.29M   221KB/s    eta 7m 36s 

    
.tar.gz               1%[                    ]   1.38M   232KB/s    eta 7m 36s 

    
tar.gz                1%[                    ]   1.43M   234KB/s    eta 7m 36s 

    
ar.gz                 1%[                    ]   1.48M   238KB/s    eta 6m 46s 

    
r.gz                  1%[                    ]   1.54M   242KB/s    eta 6m 46s 

    
.gz                   1%[                    ]   1.60M   245KB/s    eta 6m 46s 

    
gz                    2%[                    ]   1.62M   229KB/s    eta 6m 46s 

    
z                     2%[                    ]   1.71M   239KB/s    eta 6m 46s 

    
<div class="k-default-codeblock">
```
                  2%[                    ]   1.76M   241KB/s    eta 6m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
              d   2%[                    ]   1.79M   237KB/s    eta 6m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
             da   2%[                    ]   1.82M   234KB/s    eta 6m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat   2%[                    ]   1.85M   230KB/s    eta 6m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
           data   2%[                    ]   1.89M   225KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas   2%[                    ]   1.92M   220KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase   2%[                    ]   1.95M   214KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   2%[                    ]   1.99M   207KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   2%[                    ]   2.02M   198KB/s    eta 7m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   2%[                    ]   2.05M   189KB/s    eta 7m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   2%[                    ]   2.09M   182KB/s    eta 7m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   2%[                    ]   2.13M   164KB/s    eta 7m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   2%[                    ]   2.15M   160KB/s    eta 7m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   2%[                    ]   2.17M   150KB/s    eta 7m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   2%[                    ]   2.19M   143KB/s    eta 7m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   2%[                    ]   2.21M   136KB/s    eta 7m 26s 

```
</div>
    
   datasets/aclImdb   2%[                    ]   2.23M   129KB/s    eta 7m 47s 

    
  datasets/aclImdb_   2%[                    ]   2.25M   122KB/s    eta 7m 47s 

    
 datasets/aclImdb_v   2%[                    ]   2.27M   127KB/s    eta 7m 47s 

    
datasets/aclImdb_v1   2%[                    ]   2.29M   112KB/s    eta 7m 47s 

    
atasets/aclImdb_v1.   2%[                    ]   2.31M   107KB/s    eta 7m 47s 

    
tasets/aclImdb_v1.t   2%[                    ]   2.33M   105KB/s    eta 8m 8s  

    
asets/aclImdb_v1.ta   2%[                    ]   2.35M   102KB/s    eta 8m 8s  

    
sets/aclImdb_v1.tar   2%[                    ]   2.37M  99.1KB/s    eta 8m 8s  

    
ets/aclImdb_v1.tar.   2%[                    ]   2.39M  96.4KB/s    eta 8m 8s  

    
ts/aclImdb_v1.tar.g   3%[                    ]   2.41M  94.0KB/s    eta 8m 24s 

    
s/aclImdb_v1.tar.gz   3%[                    ]   2.43M  91.1KB/s    eta 8m 24s 

    
/aclImdb_v1.tar.gz    3%[                    ]   2.45M  89.3KB/s    eta 8m 24s 

    
aclImdb_v1.tar.gz     3%[                    ]   2.48M  87.6KB/s    eta 8m 24s 

    
clImdb_v1.tar.gz      3%[                    ]   2.50M  86.7KB/s    eta 8m 38s 

    
lImdb_v1.tar.gz       3%[                    ]   2.53M  90.4KB/s    eta 8m 38s 

    
Imdb_v1.tar.gz        3%[                    ]   2.56M  87.7KB/s    eta 8m 38s 

    
mdb_v1.tar.gz         3%[                    ]   2.57M  87.1KB/s    eta 8m 38s 

    
db_v1.tar.gz          3%[                    ]   2.60M  87.4KB/s    eta 8m 47s 

    
b_v1.tar.gz           3%[                    ]   2.63M  89.5KB/s    eta 8m 47s 

    
_v1.tar.gz            3%[                    ]   2.64M  88.7KB/s    eta 8m 47s 

    
v1.tar.gz             3%[                    ]   2.66M  87.6KB/s    eta 8m 47s 

    
1.tar.gz              3%[                    ]   2.67M  86.8KB/s    eta 8m 47s 

    
.tar.gz               3%[                    ]   2.69M  86.5KB/s    eta 9m 3s  

    
tar.gz                3%[                    ]   2.71M  86.4KB/s    eta 9m 3s  

    
ar.gz                 3%[                    ]   2.73M  85.8KB/s    eta 9m 3s  

    
r.gz                  3%[                    ]   2.75M  85.3KB/s    eta 9m 3s  

    
.gz                   3%[                    ]   2.77M  85.1KB/s    eta 9m 17s 

    
gz                    3%[                    ]   2.79M  84.7KB/s    eta 9m 17s 

    
z                     3%[                    ]   2.81M  84.1KB/s    eta 9m 17s 

    
<div class="k-default-codeblock">
```
                  3%[                    ]   2.83M  83.9KB/s    eta 9m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
              d   3%[                    ]   2.85M  80.1KB/s    eta 9m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
             da   3%[                    ]   2.88M  81.4KB/s    eta 9m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat   3%[                    ]   2.89M  80.6KB/s    eta 9m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
           data   3%[                    ]   2.91M  78.5KB/s    eta 9m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas   3%[                    ]   2.93M  76.8KB/s    eta 9m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase   3%[                    ]   2.95M  74.7KB/s    eta 9m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   3%[                    ]   2.97M  75.9KB/s    eta 9m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   3%[                    ]   2.99M  73.5KB/s    eta 9m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   3%[                    ]   3.01M  72.0KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   3%[                    ]   3.03M  73.2KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   3%[                    ]   3.05M  73.7KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   3%[                    ]   3.06M  71.8KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   3%[                    ]   3.09M  74.6KB/s    eta 10m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   3%[                    ]   3.10M  74.0KB/s    eta 10m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   3%[                    ]   3.11M  73.2KB/s    eta 10m 7s 

```
</div>
    
   datasets/aclImdb   3%[                    ]   3.13M  72.8KB/s    eta 10m 7s 

    
  datasets/aclImdb_   3%[                    ]   3.15M  72.7KB/s    eta 10m 7s 

    
 datasets/aclImdb_v   3%[                    ]   3.17M  73.2KB/s    eta 10m 20s

    
datasets/aclImdb_v1   3%[                    ]   3.19M  73.7KB/s    eta 10m 20s

    
atasets/aclImdb_v1.   3%[                    ]   3.21M  73.8KB/s    eta 10m 20s

    
tasets/aclImdb_v1.t   4%[                    ]   3.23M  77.9KB/s    eta 10m 20s

    
asets/aclImdb_v1.ta   4%[                    ]   3.25M  76.0KB/s    eta 10m 20s

    
sets/aclImdb_v1.tar   4%[                    ]   3.26M  76.1KB/s    eta 10m 30s

    
ets/aclImdb_v1.tar.   4%[                    ]   3.28M  76.8KB/s    eta 10m 30s

    
ts/aclImdb_v1.tar.g   4%[                    ]   3.30M  76.8KB/s    eta 10m 30s

    
s/aclImdb_v1.tar.gz   4%[                    ]   3.32M  76.8KB/s    eta 10m 30s

    
/aclImdb_v1.tar.gz    4%[                    ]   3.34M  77.9KB/s    eta 10m 37s

    
aclImdb_v1.tar.gz     4%[                    ]   3.37M  78.0KB/s    eta 10m 37s

    
clImdb_v1.tar.gz      4%[                    ]   3.39M  78.7KB/s    eta 10m 37s

    
lImdb_v1.tar.gz       4%[                    ]   3.42M  78.9KB/s    eta 10m 37s

    
Imdb_v1.tar.gz        4%[                    ]   3.44M  80.0KB/s    eta 10m 43s

    
mdb_v1.tar.gz         4%[                    ]   3.47M  83.4KB/s    eta 10m 43s

    
db_v1.tar.gz          4%[                    ]   3.50M  83.8KB/s    eta 10m 43s

    
b_v1.tar.gz           4%[                    ]   3.53M  88.1KB/s    eta 10m 43s

    
_v1.tar.gz            4%[                    ]   3.57M  92.4KB/s    eta 10m 41s

    
v1.tar.gz             4%[                    ]   3.59M  91.8KB/s    eta 10m 41s

    
1.tar.gz              4%[                    ]   3.64M  99.1KB/s    eta 10m 41s

    
.tar.gz               4%[                    ]   3.67M   102KB/s    eta 10m 41s

    
tar.gz                4%[                    ]   3.70M   104KB/s    eta 10m 41s

    
ar.gz                 4%[                    ]   3.74M   107KB/s    eta 10m 35s

    
r.gz                  4%[                    ]   3.77M   110KB/s    eta 10m 35s

    
.gz                   4%[                    ]   3.82M   115KB/s    eta 10m 35s

    
gz                    4%[                    ]   3.86M   119KB/s    eta 10m 35s

    
z                     4%[                    ]   3.91M   118KB/s    eta 10m 32s

    
<div class="k-default-codeblock">
```
                  4%[                    ]   3.96M   126KB/s    eta 10m 32s

```
</div>
    
<div class="k-default-codeblock">
```
              d   4%[                    ]   4.00M   129KB/s    eta 10m 32s

```
</div>
    
<div class="k-default-codeblock">
```
             da   5%[>                   ]   4.04M   132KB/s    eta 10m 32s

```
</div>
    
<div class="k-default-codeblock">
```
            dat   5%[>                   ]   4.05M   129KB/s    eta 10m 28s

```
</div>
    
<div class="k-default-codeblock">
```
           data   5%[>                   ]   4.11M   137KB/s    eta 10m 28s

```
</div>
    
<div class="k-default-codeblock">
```
          datas   5%[>                   ]   4.11M   135KB/s    eta 10m 28s

```
</div>
    
<div class="k-default-codeblock">
```
         datase   5%[>                   ]   4.14M   137KB/s    eta 10m 28s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   5%[>                   ]   4.17M   139KB/s    eta 10m 28s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   5%[>                   ]   4.20M   138KB/s    eta 10m 25s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   5%[>                   ]   4.21M   133KB/s    eta 10m 25s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   5%[>                   ]   4.24M   132KB/s    eta 10m 25s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   5%[>                   ]   4.25M   132KB/s    eta 10m 25s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   5%[>                   ]   4.29M   128KB/s    eta 10m 25s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   5%[>                   ]   4.30M   125KB/s    eta 10m 30s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   5%[>                   ]   4.31M   122KB/s    eta 10m 30s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   5%[>                   ]   4.33M   118KB/s    eta 10m 30s

```
</div>
    
   datasets/aclImdb   5%[>                   ]   4.35M   116KB/s    eta 10m 30s

    
  datasets/aclImdb_   5%[>                   ]   4.37M   111KB/s    eta 10m 30s

    
 datasets/aclImdb_v   5%[>                   ]   4.39M   106KB/s    eta 10m 38s

    
datasets/aclImdb_v1   5%[>                   ]   4.42M   108KB/s    eta 10m 38s

    
atasets/aclImdb_v1.   5%[>                   ]   4.44M   100KB/s    eta 10m 38s

    
tasets/aclImdb_v1.t   5%[>                   ]   4.46M  95.9KB/s    eta 10m 38s

    
asets/aclImdb_v1.ta   5%[>                   ]   4.47M  92.5KB/s    eta 10m 42s

    
sets/aclImdb_v1.tar   5%[>                   ]   4.49M  93.6KB/s    eta 10m 42s

    
ets/aclImdb_v1.tar.   5%[>                   ]   4.51M  87.9KB/s    eta 10m 42s

    
ts/aclImdb_v1.tar.g   5%[>                   ]   4.53M  86.5KB/s    eta 10m 42s

    
s/aclImdb_v1.tar.gz   5%[>                   ]   4.55M  83.9KB/s    eta 10m 49s

    
/aclImdb_v1.tar.gz    5%[>                   ]   4.57M  79.2KB/s    eta 10m 49s

    
aclImdb_v1.tar.gz     5%[>                   ]   4.60M  78.4KB/s    eta 10m 49s

    
clImdb_v1.tar.gz      5%[>                   ]   4.61M  80.2KB/s    eta 10m 49s

    
lImdb_v1.tar.gz       5%[>                   ]   4.62M  75.5KB/s    eta 10m 57s

    
Imdb_v1.tar.gz        5%[>                   ]   4.63M  75.4KB/s    eta 10m 57s

    
mdb_v1.tar.gz         5%[>                   ]   4.65M  70.8KB/s    eta 10m 57s

    
db_v1.tar.gz          5%[>                   ]   4.66M  69.7KB/s    eta 10m 57s

    
b_v1.tar.gz           5%[>                   ]   4.67M  69.6KB/s    eta 10m 57s

    
_v1.tar.gz            5%[>                   ]   4.68M  68.0KB/s    eta 11m 9s 

    
v1.tar.gz             5%[>                   ]   4.69M  66.3KB/s    eta 11m 9s 

    
1.tar.gz              5%[>                   ]   4.70M  64.6KB/s    eta 11m 9s 

    
.tar.gz               5%[>                   ]   4.71M  63.1KB/s    eta 11m 9s 

    
tar.gz                5%[>                   ]   4.72M  60.4KB/s    eta 11m 9s 

    
ar.gz                 5%[>                   ]   4.73M  59.0KB/s    eta 11m 20s

    
r.gz                  5%[>                   ]   4.75M  58.0KB/s    eta 11m 20s

    
.gz                   5%[>                   ]   4.76M  57.1KB/s    eta 11m 20s

    
gz                    5%[>                   ]   4.78M  57.0KB/s    eta 11m 20s

    
z                     5%[>                   ]   4.79M  56.4KB/s    eta 11m 20s

    
<div class="k-default-codeblock">
```
                  5%[>                   ]   4.81M  56.4KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
              d   6%[>                   ]   4.83M  55.7KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
             da   6%[>                   ]   4.85M  57.4KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
            dat   6%[>                   ]   4.87M  56.9KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
           data   6%[>                   ]   4.90M  58.3KB/s    eta 11m 31s

```
</div>
    
<div class="k-default-codeblock">
```
          datas   6%[>                   ]   4.92M  61.7KB/s    eta 11m 31s

```
</div>
    
<div class="k-default-codeblock">
```
         datase   6%[>                   ]   4.96M  65.9KB/s    eta 11m 31s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   6%[>                   ]   4.99M  71.0KB/s    eta 11m 31s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   6%[>                   ]   5.03M  76.5KB/s    eta 11m 31s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   6%[>                   ]   5.07M  83.3KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   6%[>                   ]   5.12M  90.6KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   6%[>                   ]   5.17M  98.6KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   6%[>                   ]   5.23M   108KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   6%[>                   ]   5.29M   119KB/s    eta 11m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   6%[>                   ]   5.37M   131KB/s    eta 11m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   6%[>                   ]   5.45M   144KB/s    eta 11m 7s 

```
</div>
    
   datasets/aclImdb   6%[>                   ]   5.48M   146KB/s    eta 11m 7s 

    
  datasets/aclImdb_   6%[>                   ]   5.60M   168KB/s    eta 10m 42s

    
 datasets/aclImdb_v   7%[>                   ]   5.67M   176KB/s    eta 10m 42s

    
datasets/aclImdb_v1   7%[>                   ]   5.76M   194KB/s    eta 10m 42s

    
atasets/aclImdb_v1.   7%[>                   ]   5.79M   198KB/s    eta 10m 42s

    
tasets/aclImdb_v1.t   7%[>                   ]   5.85M   206KB/s    eta 10m 42s

    
asets/aclImdb_v1.ta   7%[>                   ]   5.90M   215KB/s    eta 10m 22s

    
sets/aclImdb_v1.tar   7%[>                   ]   5.96M   224KB/s    eta 10m 22s

    
ets/aclImdb_v1.tar.   7%[>                   ]   6.02M   232KB/s    eta 10m 22s

    
ts/aclImdb_v1.tar.g   7%[>                   ]   6.08M   238KB/s    eta 10m 22s

    
s/aclImdb_v1.tar.gz   7%[>                   ]   6.14M   242KB/s    eta 10m 9s 

    
/aclImdb_v1.tar.gz    7%[>                   ]   6.20M   246KB/s    eta 10m 9s 

    
aclImdb_v1.tar.gz     7%[>                   ]   6.26M   251KB/s    eta 10m 9s 

    
clImdb_v1.tar.gz      7%[>                   ]   6.33M   255KB/s    eta 10m 9s 

    
lImdb_v1.tar.gz       7%[>                   ]   6.39M   257KB/s    eta 9m 55s 

    
Imdb_v1.tar.gz        7%[>                   ]   6.40M   248KB/s    eta 9m 55s 

    
mdb_v1.tar.gz         8%[>                   ]   6.48M   253KB/s    eta 9m 55s 

    
db_v1.tar.gz          8%[>                   ]   6.49M   241KB/s    eta 9m 55s 

    
b_v1.tar.gz           8%[>                   ]   6.51M   230KB/s    eta 9m 54s 

    
_v1.tar.gz            8%[>                   ]   6.59M   233KB/s    eta 9m 54s 

    
v1.tar.gz             8%[>                   ]   6.61M   234KB/s    eta 9m 54s 

    
1.tar.gz              8%[>                   ]   6.64M   216KB/s    eta 9m 54s 

    
.tar.gz               8%[>                   ]   6.68M   212KB/s    eta 9m 54s 

    
tar.gz                8%[>                   ]   6.72M   216KB/s    eta 9m 47s 

    
ar.gz                 8%[>                   ]   6.76M   190KB/s    eta 9m 47s 

    
r.gz                  8%[>                   ]   6.78M   182KB/s    eta 9m 47s 

    
.gz                   8%[>                   ]   6.81M   176KB/s    eta 9m 50s 

    
gz                    8%[>                   ]   6.86M   176KB/s    eta 9m 50s 

    
z                     8%[>                   ]   6.87M   167KB/s    eta 9m 50s 

    
<div class="k-default-codeblock">
```
                  8%[>                   ]   6.87M   156KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
              d   8%[>                   ]   6.90M   151KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
             da   8%[>                   ]   6.91M   142KB/s    eta 9m 52s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat   8%[>                   ]   6.93M   133KB/s    eta 9m 52s 

```
</div>
    
<div class="k-default-codeblock">
```
           data   8%[>                   ]   6.94M   124KB/s    eta 9m 52s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas   8%[>                   ]   6.96M   110KB/s    eta 9m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase   8%[>                   ]   6.98M   111KB/s    eta 9m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   8%[>                   ]   7.00M  98.2KB/s    eta 9m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   8%[>                   ]   7.01M  99.4KB/s    eta 9m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   8%[>                   ]   7.03M  99.4KB/s    eta 10m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   8%[>                   ]   7.04M  88.6KB/s    eta 10m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   8%[>                   ]   7.06M  85.9KB/s    eta 10m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   8%[>                   ]   7.08M  81.8KB/s    eta 10m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   8%[>                   ]   7.09M  77.6KB/s    eta 10m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   8%[>                   ]   7.11M  73.0KB/s    eta 10m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   8%[>                   ]   7.12M  71.9KB/s    eta 10m 7s 

```
</div>
    
   datasets/aclImdb   8%[>                   ]   7.14M  70.7KB/s    eta 10m 7s 

    
  datasets/aclImdb_   8%[>                   ]   7.16M  68.6KB/s    eta 10m 11s

    
 datasets/aclImdb_v   8%[>                   ]   7.17M  63.3KB/s    eta 10m 11s

    
datasets/aclImdb_v1   8%[>                   ]   7.19M  62.7KB/s    eta 10m 11s

    
atasets/aclImdb_v1.   8%[>                   ]   7.20M  62.9KB/s    eta 10m 11s

    
tasets/aclImdb_v1.t   8%[>                   ]   7.22M  62.5KB/s    eta 10m 15s

    
asets/aclImdb_v1.ta   9%[>                   ]   7.23M  58.9KB/s    eta 10m 15s

    
sets/aclImdb_v1.tar   9%[>                   ]   7.25M  58.7KB/s    eta 10m 15s

    
ets/aclImdb_v1.tar.   9%[>                   ]   7.26M  58.0KB/s    eta 10m 15s

    
ts/aclImdb_v1.tar.g   9%[>                   ]   7.27M  60.0KB/s    eta 10m 23s

    
s/aclImdb_v1.tar.gz   9%[>                   ]   7.28M  58.1KB/s    eta 10m 23s

    
/aclImdb_v1.tar.gz    9%[>                   ]   7.29M  58.0KB/s    eta 10m 23s

    
aclImdb_v1.tar.gz     9%[>                   ]   7.31M  57.5KB/s    eta 10m 23s

    
clImdb_v1.tar.gz      9%[>                   ]   7.32M  56.5KB/s    eta 10m 23s

    
lImdb_v1.tar.gz       9%[>                   ]   7.33M  55.6KB/s    eta 10m 30s

    
Imdb_v1.tar.gz        9%[>                   ]   7.34M  54.7KB/s    eta 10m 30s

    
mdb_v1.tar.gz         9%[>                   ]   7.35M  53.0KB/s    eta 10m 30s

    
db_v1.tar.gz          9%[>                   ]   7.36M  53.1KB/s    eta 10m 30s

    
b_v1.tar.gz           9%[>                   ]   7.37M  49.9KB/s    eta 10m 38s

    
_v1.tar.gz            9%[>                   ]   7.38M  49.0KB/s    eta 10m 38s

    
v1.tar.gz             9%[>                   ]   7.39M  47.6KB/s    eta 10m 38s

    
1.tar.gz              9%[>                   ]   7.40M  46.6KB/s    eta 10m 38s

    
.tar.gz               9%[>                   ]   7.41M  45.1KB/s    eta 10m 44s

    
tar.gz                9%[>                   ]   7.42M  42.6KB/s    eta 10m 44s

    
ar.gz                 9%[>                   ]   7.42M  40.6KB/s    eta 10m 44s

    
r.gz                  9%[>                   ]   7.43M  38.3KB/s    eta 10m 44s

    
.gz                   9%[>                   ]   7.44M  39.3KB/s    eta 10m 54s

    
gz                    9%[>                   ]   7.45M  37.8KB/s    eta 10m 54s

    
z                     9%[>                   ]   7.46M  37.0KB/s    eta 10m 54s

    
<div class="k-default-codeblock">
```
                  9%[>                   ]   7.46M  35.8KB/s    eta 10m 54s

```
</div>
    
<div class="k-default-codeblock">
```
              d   9%[>                   ]   7.47M  34.1KB/s    eta 11m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
             da   9%[>                   ]   7.48M  33.4KB/s    eta 11m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat   9%[>                   ]   7.48M  31.7KB/s    eta 11m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
           data   9%[>                   ]   7.49M  31.4KB/s    eta 11m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas   9%[>                   ]   7.50M  30.7KB/s    eta 11m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase   9%[>                   ]   7.51M  30.1KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   9%[>                   ]   7.51M  30.3KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   9%[>                   ]   7.52M  27.9KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   9%[>                   ]   7.53M  29.1KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   9%[>                   ]   7.54M  28.0KB/s    eta 11m 19s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   9%[>                   ]   7.54M  27.5KB/s    eta 11m 19s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   9%[>                   ]   7.55M  25.9KB/s    eta 11m 19s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   9%[>                   ]   7.56M  26.2KB/s    eta 11m 19s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   9%[>                   ]   7.57M  26.7KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   9%[>                   ]   7.58M  27.9KB/s    eta 11m 28s

```
</div>
    
   datasets/aclImdb   9%[>                   ]   7.58M  27.8KB/s    eta 11m 28s

    
  datasets/aclImdb_   9%[>                   ]   7.59M  28.3KB/s    eta 11m 28s

    
 datasets/aclImdb_v   9%[>                   ]   7.60M  28.3KB/s    eta 11m 28s

    
datasets/aclImdb_v1   9%[>                   ]   7.60M  27.2KB/s    eta 11m 36s

    
atasets/aclImdb_v1.   9%[>                   ]   7.60M  26.0KB/s    eta 11m 36s

    
tasets/aclImdb_v1.t   9%[>                   ]   7.61M  26.4KB/s    eta 11m 36s

    
asets/aclImdb_v1.ta   9%[>                   ]   7.62M  26.2KB/s    eta 11m 36s

    
sets/aclImdb_v1.tar   9%[>                   ]   7.62M  26.9KB/s    eta 11m 36s

    
ets/aclImdb_v1.tar.   9%[>                   ]   7.63M  26.2KB/s    eta 11m 45s

    
ts/aclImdb_v1.tar.g   9%[>                   ]   7.63M  26.3KB/s    eta 11m 45s

    
s/aclImdb_v1.tar.gz   9%[>                   ]   7.64M  26.1KB/s    eta 11m 45s

    
/aclImdb_v1.tar.gz    9%[>                   ]   7.65M  25.1KB/s    eta 11m 45s

    
aclImdb_v1.tar.gz     9%[>                   ]   7.66M  26.7KB/s    eta 11m 53s

    
clImdb_v1.tar.gz      9%[>                   ]   7.67M  25.8KB/s    eta 11m 53s

    
lImdb_v1.tar.gz       9%[>                   ]   7.67M  25.5KB/s    eta 11m 53s

    
Imdb_v1.tar.gz        9%[>                   ]   7.68M  26.5KB/s    eta 11m 53s

    
mdb_v1.tar.gz         9%[>                   ]   7.69M  27.9KB/s    eta 11m 53s

    
db_v1.tar.gz          9%[>                   ]   7.70M  27.5KB/s    eta 12m 0s 

    
b_v1.tar.gz           9%[>                   ]   7.71M  28.0KB/s    eta 12m 0s 

    
_v1.tar.gz            9%[>                   ]   7.71M  28.0KB/s    eta 12m 0s 

    
v1.tar.gz             9%[>                   ]   7.72M  27.2KB/s    eta 12m 0s 

    
1.tar.gz              9%[>                   ]   7.73M  25.9KB/s    eta 12m 9s 

    
.tar.gz               9%[>                   ]   7.74M  26.8KB/s    eta 12m 9s 

    
tar.gz                9%[>                   ]   7.75M  27.5KB/s    eta 12m 9s 

    
ar.gz                 9%[>                   ]   7.75M  28.3KB/s    eta 12m 9s 

    
r.gz                  9%[>                   ]   7.76M  28.7KB/s    eta 12m 9s 

    
.gz                   9%[>                   ]   7.77M  29.2KB/s    eta 12m 16s

    
gz                    9%[>                   ]   7.77M  28.6KB/s    eta 12m 16s

    
z                     9%[>                   ]   7.78M  28.9KB/s    eta 12m 16s

    
<div class="k-default-codeblock">
```
                  9%[>                   ]   7.79M  28.5KB/s    eta 12m 16s

```
</div>
    
<div class="k-default-codeblock">
```
              d   9%[>                   ]   7.80M  28.5KB/s    eta 12m 25s

```
</div>
    
<div class="k-default-codeblock">
```
             da   9%[>                   ]   7.81M  30.0KB/s    eta 12m 25s

```
</div>
    
<div class="k-default-codeblock">
```
            dat   9%[>                   ]   7.81M  29.1KB/s    eta 12m 25s

```
</div>
    
<div class="k-default-codeblock">
```
           data   9%[>                   ]   7.82M  29.5KB/s    eta 12m 25s

```
</div>
    
<div class="k-default-codeblock">
```
          datas   9%[>                   ]   7.83M  30.3KB/s    eta 12m 25s

```
</div>
    
<div class="k-default-codeblock">
```
         datase   9%[>                   ]   7.83M  28.7KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset   9%[>                   ]   7.84M  28.7KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets   9%[>                   ]   7.85M  28.7KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/   9%[>                   ]   7.86M  28.7KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a   9%[>                   ]   7.87M  28.8KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac   9%[>                   ]   7.88M  29.6KB/s    eta 12m 39s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl   9%[>                   ]   7.88M  31.0KB/s    eta 12m 39s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI   9%[>                   ]   7.89M  30.1KB/s    eta 12m 39s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm   9%[>                   ]   7.90M  30.6KB/s    eta 12m 39s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd   9%[>                   ]   7.91M  31.2KB/s    eta 12m 39s

```
</div>
    
   datasets/aclImdb   9%[>                   ]   7.92M  31.0KB/s    eta 12m 45s

    
  datasets/aclImdb_   9%[>                   ]   7.92M  31.4KB/s    eta 12m 45s

    
 datasets/aclImdb_v   9%[>                   ]   7.94M  33.1KB/s    eta 12m 45s

    
datasets/aclImdb_v1   9%[>                   ]   7.95M  33.9KB/s    eta 12m 45s

    
atasets/aclImdb_v1.   9%[>                   ]   7.96M  35.5KB/s    eta 12m 45s

    
tasets/aclImdb_v1.t   9%[>                   ]   7.97M  36.5KB/s    eta 12m 50s

    
asets/aclImdb_v1.ta   9%[>                   ]   7.98M  36.0KB/s    eta 12m 50s

    
sets/aclImdb_v1.tar   9%[>                   ]   7.98M  36.6KB/s    eta 12m 50s

    
ets/aclImdb_v1.tar.   9%[>                   ]   7.99M  36.8KB/s    eta 12m 50s

    
ts/aclImdb_v1.tar.g   9%[>                   ]   8.00M  36.2KB/s    eta 12m 50s

    
s/aclImdb_v1.tar.gz   9%[>                   ]   8.01M  37.3KB/s    eta 12m 57s

    
/aclImdb_v1.tar.gz    9%[>                   ]   8.02M  37.9KB/s    eta 12m 57s

    
aclImdb_v1.tar.gz    10%[=>                  ]   8.03M  38.4KB/s    eta 12m 57s

    
clImdb_v1.tar.gz     10%[=>                  ]   8.04M  39.0KB/s    eta 12m 57s

    
lImdb_v1.tar.gz      10%[=>                  ]   8.05M  39.5KB/s    eta 12m 57s

    
Imdb_v1.tar.gz       10%[=>                  ]   8.06M  39.9KB/s    eta 13m 2s 

    
mdb_v1.tar.gz        10%[=>                  ]   8.07M  40.3KB/s    eta 13m 2s 

    
db_v1.tar.gz         10%[=>                  ]   8.08M  40.6KB/s    eta 13m 2s 

    
b_v1.tar.gz          10%[=>                  ]   8.10M  40.9KB/s    eta 13m 2s 

    
_v1.tar.gz           10%[=>                  ]   8.11M  41.2KB/s    eta 13m 7s 

    
v1.tar.gz            10%[=>                  ]   8.12M  41.7KB/s    eta 13m 7s 

    
1.tar.gz             10%[=>                  ]   8.13M  41.9KB/s    eta 13m 7s 

    
.tar.gz              10%[=>                  ]   8.15M  42.5KB/s    eta 13m 7s 

    
tar.gz               10%[=>                  ]   8.16M  43.3KB/s    eta 13m 11s

    
ar.gz                10%[=>                  ]   8.18M  43.9KB/s    eta 13m 11s

    
r.gz                 10%[=>                  ]   8.20M  45.8KB/s    eta 13m 11s

    
.gz                  10%[=>                  ]   8.22M  48.4KB/s    eta 13m 11s

    
gz                   10%[=>                  ]   8.24M  51.1KB/s    eta 13m 11s

    
z                    10%[=>                  ]   8.27M  51.3KB/s    eta 13m 11s

    
<div class="k-default-codeblock">
```
                 10%[=>                  ]   8.31M  59.1KB/s    eta 13m 11s

```
</div>
    
<div class="k-default-codeblock">
```
              d  10%[=>                  ]   8.34M  62.4KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
             da  10%[=>                  ]   8.37M  65.6KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  10%[=>                  ]   8.39M  68.8KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
           data  10%[=>                  ]   8.41M  68.3KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  10%[=>                  ]   8.45M  74.9KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  10%[=>                  ]   8.47M  76.5KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  10%[=>                  ]   8.48M  77.5KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  10%[=>                  ]   8.51M  80.4KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  10%[=>                  ]   8.53M  82.9KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  10%[=>                  ]   8.56M  85.6KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  10%[=>                  ]   8.58M  86.4KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  10%[=>                  ]   8.61M  89.2KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  10%[=>                  ]   8.64M  91.3KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  10%[=>                  ]   8.65M  90.7KB/s    eta 13m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  10%[=>                  ]   8.66M  89.3KB/s    eta 13m 7s 

```
</div>
    
   datasets/aclImdb  10%[=>                  ]   8.68M  87.9KB/s    eta 13m 9s 

    
  datasets/aclImdb_  10%[=>                  ]   8.69M  85.8KB/s    eta 13m 9s 

    
 datasets/aclImdb_v  10%[=>                  ]   8.71M  88.3KB/s    eta 13m 9s 

    
datasets/aclImdb_v1  10%[=>                  ]   8.72M  80.9KB/s    eta 13m 9s 

    
atasets/aclImdb_v1.  10%[=>                  ]   8.74M  75.8KB/s    eta 13m 12s

    
tasets/aclImdb_v1.t  10%[=>                  ]   8.75M  72.9KB/s    eta 13m 12s

    
asets/aclImdb_v1.ta  10%[=>                  ]   8.76M  73.9KB/s    eta 13m 12s

    
sets/aclImdb_v1.tar  10%[=>                  ]   8.78M  69.3KB/s    eta 13m 12s

    
ets/aclImdb_v1.tar.  10%[=>                  ]   8.80M  68.1KB/s    eta 13m 15s

    
ts/aclImdb_v1.tar.g  10%[=>                  ]   8.81M  66.9KB/s    eta 13m 15s

    
s/aclImdb_v1.tar.gz  10%[=>                  ]   8.82M  60.2KB/s    eta 13m 15s

    
/aclImdb_v1.tar.gz   11%[=>                  ]   8.83M  57.4KB/s    eta 13m 20s

    
aclImdb_v1.tar.gz    11%[=>                  ]   8.87M  60.2KB/s    eta 13m 20s

    
clImdb_v1.tar.gz     11%[=>                  ]   8.88M  59.2KB/s    eta 13m 20s

    
lImdb_v1.tar.gz      11%[=>                  ]   8.90M  57.2KB/s    eta 13m 20s

    
Imdb_v1.tar.gz       11%[=>                  ]   8.92M  59.0KB/s    eta 13m 19s

    
mdb_v1.tar.gz        11%[=>                  ]   8.93M  57.7KB/s    eta 13m 19s

    
db_v1.tar.gz         11%[=>                  ]   8.95M  58.4KB/s    eta 13m 19s

    
b_v1.tar.gz          11%[=>                  ]   8.97M  59.2KB/s    eta 13m 19s

    
_v1.tar.gz           11%[=>                  ]   8.99M  60.5KB/s    eta 13m 20s

    
v1.tar.gz            11%[=>                  ]   9.01M  61.7KB/s    eta 13m 20s

    
1.tar.gz             11%[=>                  ]   9.03M  61.5KB/s    eta 13m 20s

    
.tar.gz              11%[=>                  ]   9.05M  62.8KB/s    eta 13m 20s

    
tar.gz               11%[=>                  ]   9.07M  63.5KB/s    eta 13m 21s

    
ar.gz                11%[=>                  ]   9.08M  64.6KB/s    eta 13m 21s

    
r.gz                 11%[=>                  ]   9.10M  65.7KB/s    eta 13m 21s

    
.gz                  11%[=>                  ]   9.13M  67.7KB/s    eta 13m 21s

    
gz                   11%[=>                  ]   9.15M  68.7KB/s    eta 13m 21s

    
z                    11%[=>                  ]   9.17M  69.9KB/s    eta 13m 21s

    
<div class="k-default-codeblock">
```
                 11%[=>                  ]   9.20M  71.3KB/s    eta 13m 21s

```
</div>
    
<div class="k-default-codeblock">
```
              d  11%[=>                  ]   9.22M  74.3KB/s    eta 13m 22s

```
</div>
    
<div class="k-default-codeblock">
```
             da  11%[=>                  ]   9.27M  81.6KB/s    eta 13m 22s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  11%[=>                  ]   9.28M  77.4KB/s    eta 13m 22s

```
</div>
    
<div class="k-default-codeblock">
```
           data  11%[=>                  ]   9.31M  80.6KB/s    eta 13m 22s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  11%[=>                  ]   9.33M  81.6KB/s    eta 13m 22s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  11%[=>                  ]   9.36M  83.6KB/s    eta 13m 18s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  11%[=>                  ]   9.38M  85.6KB/s    eta 13m 18s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  11%[=>                  ]   9.41M  87.4KB/s    eta 13m 18s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  11%[=>                  ]   9.44M  89.5KB/s    eta 13m 18s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  11%[=>                  ]   9.47M  91.2KB/s    eta 13m 18s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  11%[=>                  ]   9.49M  92.3KB/s    eta 13m 14s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  11%[=>                  ]   9.52M  92.0KB/s    eta 13m 14s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  11%[=>                  ]   9.54M  92.2KB/s    eta 13m 14s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  11%[=>                  ]   9.58M  95.4KB/s    eta 13m 14s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  11%[=>                  ]   9.60M  96.1KB/s    eta 13m 14s

```
</div>
    
   datasets/aclImdb  11%[=>                  ]   9.62M  97.1KB/s    eta 13m 14s

    
  datasets/aclImdb_  12%[=>                  ]   9.65M  97.4KB/s    eta 13m 14s

    
 datasets/aclImdb_v  12%[=>                  ]   9.67M  98.7KB/s    eta 13m 14s

    
datasets/aclImdb_v1  12%[=>                  ]   9.70M   100KB/s    eta 13m 14s

    
atasets/aclImdb_v1.  12%[=>                  ]   9.73M   101KB/s    eta 13m 10s

    
tasets/aclImdb_v1.t  12%[=>                  ]   9.75M   104KB/s    eta 13m 10s

    
asets/aclImdb_v1.ta  12%[=>                  ]   9.78M   101KB/s    eta 13m 10s

    
sets/aclImdb_v1.tar  12%[=>                  ]   9.80M   103KB/s    eta 13m 10s

    
ets/aclImdb_v1.tar.  12%[=>                  ]   9.81M   101KB/s    eta 13m 10s

    
ts/aclImdb_v1.tar.g  12%[=>                  ]   9.81M  96.2KB/s    eta 13m 11s

    
s/aclImdb_v1.tar.gz  12%[=>                  ]   9.85M  99.0KB/s    eta 13m 11s

    
/aclImdb_v1.tar.gz   12%[=>                  ]   9.87M  97.3KB/s    eta 13m 11s

    
aclImdb_v1.tar.gz    12%[=>                  ]   9.89M  96.0KB/s    eta 13m 11s

    
clImdb_v1.tar.gz     12%[=>                  ]   9.91M  94.5KB/s    eta 13m 11s

    
lImdb_v1.tar.gz      12%[=>                  ]   9.93M  93.2KB/s    eta 13m 9s 

    
Imdb_v1.tar.gz       12%[=>                  ]   9.95M  91.9KB/s    eta 13m 9s 

    
mdb_v1.tar.gz        12%[=>                  ]   9.97M  94.4KB/s    eta 13m 9s 

    
db_v1.tar.gz         12%[=>                  ]   9.99M  94.3KB/s    eta 13m 9s 

    
b_v1.tar.gz          12%[=>                  ]  10.01M  91.5KB/s    eta 13m 9s 

    
_v1.tar.gz           12%[=>                  ]  10.02M  90.2KB/s    eta 13m 9s 

    
v1.tar.gz            12%[=>                  ]  10.04M  89.6KB/s    eta 13m 9s 

    
1.tar.gz             12%[=>                  ]  10.06M  87.1KB/s    eta 13m 9s 

    
.tar.gz              12%[=>                  ]  10.08M  85.9KB/s    eta 13m 9s 

    
tar.gz               12%[=>                  ]  10.11M  84.1KB/s    eta 13m 9s 

    
ar.gz                12%[=>                  ]  10.13M  82.5KB/s    eta 13m 9s 

    
r.gz                 12%[=>                  ]  10.15M  83.9KB/s    eta 13m 9s 

    
.gz                  12%[=>                  ]  10.17M  81.8KB/s    eta 13m 9s 

    
gz                   12%[=>                  ]  10.18M  78.4KB/s    eta 13m 10s

    
z                    12%[=>                  ]  10.21M  79.8KB/s    eta 13m 10s

    
<div class="k-default-codeblock">
```
                 12%[=>                  ]  10.23M  84.4KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
              d  12%[=>                  ]  10.26M  81.4KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
             da  12%[=>                  ]  10.28M  82.7KB/s    eta 13m 10s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  12%[=>                  ]  10.30M  82.9KB/s    eta 13m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  12%[=>                  ]  10.33M  84.5KB/s    eta 13m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  12%[=>                  ]  10.36M  86.1KB/s    eta 13m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  12%[=>                  ]  10.38M  87.6KB/s    eta 13m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  12%[=>                  ]  10.39M  84.7KB/s    eta 13m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  12%[=>                  ]  10.42M  86.1KB/s    eta 13m 6s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  13%[=>                  ]  10.45M  89.1KB/s    eta 13m 6s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  13%[=>                  ]  10.47M  90.5KB/s    eta 13m 6s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  13%[=>                  ]  10.50M  90.6KB/s    eta 13m 6s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  13%[=>                  ]  10.52M  92.8KB/s    eta 13m 6s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  13%[=>                  ]  10.54M  93.7KB/s    eta 13m 3s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  13%[=>                  ]  10.57M  94.2KB/s    eta 13m 3s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  13%[=>                  ]  10.57M  90.8KB/s    eta 13m 3s 

```
</div>
    
   datasets/aclImdb  13%[=>                  ]  10.61M  96.0KB/s    eta 13m 3s 

    
  datasets/aclImdb_  13%[=>                  ]  10.63M  95.6KB/s    eta 13m 3s 

    
 datasets/aclImdb_v  13%[=>                  ]  10.65M   100KB/s    eta 13m 1s 

    
datasets/aclImdb_v1  13%[=>                  ]  10.68M   103KB/s    eta 13m 1s 

    
atasets/aclImdb_v1.  13%[=>                  ]  10.70M   101KB/s    eta 13m 1s 

    
tasets/aclImdb_v1.t  13%[=>                  ]  10.73M   102KB/s    eta 13m 1s 

    
asets/aclImdb_v1.ta  13%[=>                  ]  10.74M  99.8KB/s    eta 13m 1s 

    
sets/aclImdb_v1.tar  13%[=>                  ]  10.77M  99.8KB/s    eta 13m 0s 

    
ets/aclImdb_v1.tar.  13%[=>                  ]  10.79M  99.7KB/s    eta 13m 0s 

    
ts/aclImdb_v1.tar.g  13%[=>                  ]  10.82M  99.6KB/s    eta 13m 0s 

    
s/aclImdb_v1.tar.gz  13%[=>                  ]  10.85M  98.9KB/s    eta 13m 0s 

    
/aclImdb_v1.tar.gz   13%[=>                  ]  10.88M   102KB/s    eta 12m 57s

    
aclImdb_v1.tar.gz    13%[=>                  ]  10.90M   103KB/s    eta 12m 57s

    
clImdb_v1.tar.gz     13%[=>                  ]  10.93M   101KB/s    eta 12m 57s

    
lImdb_v1.tar.gz      13%[=>                  ]  10.96M   103KB/s    eta 12m 57s

    
Imdb_v1.tar.gz       13%[=>                  ]  10.99M   105KB/s    eta 12m 54s

    
mdb_v1.tar.gz        13%[=>                  ]  11.02M   105KB/s    eta 12m 54s

    
db_v1.tar.gz         13%[=>                  ]  11.05M   105KB/s    eta 12m 54s

    
b_v1.tar.gz          13%[=>                  ]  11.08M   106KB/s    eta 12m 54s

    
_v1.tar.gz           13%[=>                  ]  11.11M   111KB/s    eta 12m 51s

    
v1.tar.gz            13%[=>                  ]  11.15M   110KB/s    eta 12m 51s

    
1.tar.gz             13%[=>                  ]  11.19M   113KB/s    eta 12m 51s

    
.tar.gz              13%[=>                  ]  11.21M   111KB/s    eta 12m 51s

    
tar.gz               14%[=>                  ]  11.28M   119KB/s    eta 12m 45s

    
ar.gz                14%[=>                  ]  11.31M   121KB/s    eta 12m 45s

    
r.gz                 14%[=>                  ]  11.34M   122KB/s    eta 12m 45s

    
.gz                  14%[=>                  ]  11.37M   122KB/s    eta 12m 45s

    
gz                   14%[=>                  ]  11.40M   126KB/s    eta 12m 45s

    
z                    14%[=>                  ]  11.43M   124KB/s    eta 12m 41s

    
<div class="k-default-codeblock">
```
                 14%[=>                  ]  11.49M   133KB/s    eta 12m 41s

```
</div>
    
<div class="k-default-codeblock">
```
              d  14%[=>                  ]  11.51M   133KB/s    eta 12m 41s

```
</div>
    
<div class="k-default-codeblock">
```
             da  14%[=>                  ]  11.54M   134KB/s    eta 12m 41s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  14%[=>                  ]  11.58M   135KB/s    eta 12m 41s

```
</div>
    
<div class="k-default-codeblock">
```
           data  14%[=>                  ]  11.61M   137KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  14%[=>                  ]  11.65M   139KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  14%[=>                  ]  11.68M   141KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  14%[=>                  ]  11.72M   143KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  14%[=>                  ]  11.75M   146KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  14%[=>                  ]  11.76M   141KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  14%[=>                  ]  11.81M   144KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  14%[=>                  ]  11.84M   142KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  14%[=>                  ]  11.86M   141KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  14%[=>                  ]  11.89M   146KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  14%[=>                  ]  11.93M   141KB/s    eta 12m 24s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  14%[=>                  ]  11.94M   135KB/s    eta 12m 24s

```
</div>
    
   datasets/aclImdb  14%[=>                  ]  11.97M   134KB/s    eta 12m 24s

    
  datasets/aclImdb_  14%[=>                  ]  12.01M   134KB/s    eta 12m 24s

    
 datasets/aclImdb_v  14%[=>                  ]  12.03M   130KB/s    eta 12m 23s

    
datasets/aclImdb_v1  15%[==>                 ]  12.04M   130KB/s    eta 12m 23s

    
atasets/aclImdb_v1.  15%[==>                 ]  12.06M   121KB/s    eta 12m 23s

    
tasets/aclImdb_v1.t  15%[==>                 ]  12.08M   108KB/s    eta 12m 25s

    
asets/aclImdb_v1.ta  15%[==>                 ]  12.11M   108KB/s    eta 12m 25s

    
sets/aclImdb_v1.tar  15%[==>                 ]  12.13M   104KB/s    eta 12m 25s

    
ets/aclImdb_v1.tar.  15%[==>                 ]  12.14M  99.3KB/s    eta 12m 25s

    
ts/aclImdb_v1.tar.g  15%[==>                 ]  12.16M  94.9KB/s    eta 12m 25s

    
s/aclImdb_v1.tar.gz  15%[==>                 ]  12.18M  91.7KB/s    eta 12m 25s

    
/aclImdb_v1.tar.gz   15%[==>                 ]  12.19M  86.0KB/s    eta 12m 25s

    
aclImdb_v1.tar.gz    15%[==>                 ]  12.19M  86.1KB/s    eta 12m 25s

    
clImdb_v1.tar.gz     15%[==>                 ]  12.21M  79.8KB/s    eta 12m 25s

    
lImdb_v1.tar.gz      15%[==>                 ]  12.23M  79.2KB/s    eta 12m 25s

    
Imdb_v1.tar.gz       15%[==>                 ]  12.24M  75.9KB/s    eta 12m 27s

    
mdb_v1.tar.gz        15%[==>                 ]  12.26M  74.2KB/s    eta 12m 27s

    
db_v1.tar.gz         15%[==>                 ]  12.28M  71.7KB/s    eta 12m 27s

    
b_v1.tar.gz          15%[==>                 ]  12.30M  73.1KB/s    eta 12m 27s

    
_v1.tar.gz           15%[==>                 ]  12.32M  70.8KB/s    eta 12m 27s

    
v1.tar.gz            15%[==>                 ]  12.34M  66.9KB/s    eta 12m 27s

    
1.tar.gz             15%[==>                 ]  12.36M  67.8KB/s    eta 12m 27s

    
.tar.gz              15%[==>                 ]  12.38M  68.0KB/s    eta 12m 27s

    
tar.gz               15%[==>                 ]  12.39M  66.2KB/s    eta 12m 27s

    
ar.gz                15%[==>                 ]  12.42M  70.2KB/s    eta 12m 28s

    
r.gz                 15%[==>                 ]  12.44M  67.8KB/s    eta 12m 28s

    
.gz                  15%[==>                 ]  12.45M  66.4KB/s    eta 12m 28s

    
gz                   15%[==>                 ]  12.47M  67.6KB/s    eta 12m 28s

    
z                    15%[==>                 ]  12.48M  67.7KB/s    eta 12m 28s

    
<div class="k-default-codeblock">
```
                 15%[==>                 ]  12.50M  67.2KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
              d  15%[==>                 ]  12.50M  65.6KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
             da  15%[==>                 ]  12.53M  68.5KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  15%[==>                 ]  12.53M  64.8KB/s    eta 12m 29s

```
</div>
    
<div class="k-default-codeblock">
```
           data  15%[==>                 ]  12.54M  62.8KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  15%[==>                 ]  12.55M  62.4KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  15%[==>                 ]  12.56M  59.6KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  15%[==>                 ]  12.57M  57.5KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  15%[==>                 ]  12.58M  55.2KB/s    eta 12m 32s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  15%[==>                 ]  12.59M  53.3KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  15%[==>                 ]  12.59M  51.3KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  15%[==>                 ]  12.60M  49.2KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  15%[==>                 ]  12.61M  47.8KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  15%[==>                 ]  12.62M  48.4KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  15%[==>                 ]  12.64M  46.3KB/s    eta 12m 38s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  15%[==>                 ]  12.66M  48.8KB/s    eta 12m 38s

```
</div>
    
   datasets/aclImdb  15%[==>                 ]  12.67M  47.2KB/s    eta 12m 38s

    
  datasets/aclImdb_  15%[==>                 ]  12.69M  47.9KB/s    eta 12m 38s

    
 datasets/aclImdb_v  15%[==>                 ]  12.72M  50.0KB/s    eta 12m 38s

    
datasets/aclImdb_v1  15%[==>                 ]  12.74M  52.1KB/s    eta 12m 37s

    
atasets/aclImdb_v1.  15%[==>                 ]  12.77M  57.8KB/s    eta 12m 37s

    
tasets/aclImdb_v1.t  15%[==>                 ]  12.81M  58.0KB/s    eta 12m 37s

    
asets/aclImdb_v1.ta  16%[==>                 ]  12.87M  70.6KB/s    eta 12m 37s

    
sets/aclImdb_v1.tar  16%[==>                 ]  12.90M  74.0KB/s    eta 12m 33s

    
ets/aclImdb_v1.tar.  16%[==>                 ]  12.93M  78.3KB/s    eta 12m 33s

    
ts/aclImdb_v1.tar.g  16%[==>                 ]  12.96M  84.0KB/s    eta 12m 33s

    
s/aclImdb_v1.tar.gz  16%[==>                 ]  13.00M  89.1KB/s    eta 12m 33s

    
/aclImdb_v1.tar.gz   16%[==>                 ]  13.04M  94.5KB/s    eta 12m 28s

    
aclImdb_v1.tar.gz    16%[==>                 ]  13.08M   100KB/s    eta 12m 28s

    
clImdb_v1.tar.gz     16%[==>                 ]  13.11M   105KB/s    eta 12m 28s

    
lImdb_v1.tar.gz      16%[==>                 ]  13.15M   114KB/s    eta 12m 28s

    
Imdb_v1.tar.gz       16%[==>                 ]  13.19M   119KB/s    eta 12m 24s

    
mdb_v1.tar.gz        16%[==>                 ]  13.23M   124KB/s    eta 12m 24s

    
db_v1.tar.gz         16%[==>                 ]  13.27M   132KB/s    eta 12m 24s

    
b_v1.tar.gz          16%[==>                 ]  13.31M   137KB/s    eta 12m 24s

    
_v1.tar.gz           16%[==>                 ]  13.32M   132KB/s    eta 12m 20s

    
v1.tar.gz            16%[==>                 ]  13.36M   131KB/s    eta 12m 20s

    
1.tar.gz             16%[==>                 ]  13.40M   134KB/s    eta 12m 20s

    
.tar.gz              16%[==>                 ]  13.43M   133KB/s    eta 12m 20s

    
tar.gz               16%[==>                 ]  13.45M   136KB/s    eta 12m 18s

    
ar.gz                16%[==>                 ]  13.47M   128KB/s    eta 12m 18s

    
r.gz                 16%[==>                 ]  13.49M   125KB/s    eta 12m 18s

    
.gz                  16%[==>                 ]  13.51M   121KB/s    eta 12m 18s

    
gz                   16%[==>                 ]  13.53M   118KB/s    eta 12m 17s

    
z                    16%[==>                 ]  13.53M   111KB/s    eta 12m 17s

    
<div class="k-default-codeblock">
```
                 16%[==>                 ]  13.56M   104KB/s    eta 12m 17s

```
</div>
    
<div class="k-default-codeblock">
```
              d  16%[==>                 ]  13.58M   105KB/s    eta 12m 17s

```
</div>
    
<div class="k-default-codeblock">
```
             da  16%[==>                 ]  13.59M   101KB/s    eta 12m 18s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  16%[==>                 ]  13.60M  95.4KB/s    eta 12m 18s

```
</div>
    
<div class="k-default-codeblock">
```
           data  16%[==>                 ]  13.61M  89.2KB/s    eta 12m 18s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  16%[==>                 ]  13.62M  87.7KB/s    eta 12m 18s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  16%[==>                 ]  13.63M  82.3KB/s    eta 12m 18s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  17%[==>                 ]  13.64M  77.5KB/s    eta 12m 21s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  17%[==>                 ]  13.66M  71.7KB/s    eta 12m 21s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  17%[==>                 ]  13.67M  72.6KB/s    eta 12m 21s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  17%[==>                 ]  13.68M  68.9KB/s    eta 12m 21s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  17%[==>                 ]  13.69M  62.3KB/s    eta 12m 21s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  17%[==>                 ]  13.70M  59.7KB/s    eta 12m 23s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  17%[==>                 ]  13.71M  58.0KB/s    eta 12m 23s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  17%[==>                 ]  13.72M  56.9KB/s    eta 12m 23s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  17%[==>                 ]  13.74M  55.2KB/s    eta 12m 23s

```
</div>
    
   datasets/aclImdb  17%[==>                 ]  13.76M  53.6KB/s    eta 12m 24s

    
  datasets/aclImdb_  17%[==>                 ]  13.77M  54.1KB/s    eta 12m 24s

    
 datasets/aclImdb_v  17%[==>                 ]  13.79M  53.9KB/s    eta 12m 24s

    
datasets/aclImdb_v1  17%[==>                 ]  13.82M  58.4KB/s    eta 12m 24s

    
atasets/aclImdb_v1.  17%[==>                 ]  13.84M  59.5KB/s    eta 12m 24s

    
tasets/aclImdb_v1.t  17%[==>                 ]  13.87M  60.4KB/s    eta 12m 24s

    
asets/aclImdb_v1.ta  17%[==>                 ]  13.90M  65.3KB/s    eta 12m 24s

    
sets/aclImdb_v1.tar  17%[==>                 ]  13.94M  69.8KB/s    eta 12m 24s

    
ets/aclImdb_v1.tar.  17%[==>                 ]  13.98M  72.2KB/s    eta 12m 21s

    
ts/aclImdb_v1.tar.g  17%[==>                 ]  14.04M  83.6KB/s    eta 12m 21s

    
s/aclImdb_v1.tar.gz  17%[==>                 ]  14.08M  87.9KB/s    eta 12m 21s

    
/aclImdb_v1.tar.gz   17%[==>                 ]  14.08M  86.3KB/s    eta 12m 21s

    
aclImdb_v1.tar.gz    17%[==>                 ]  14.09M  84.1KB/s    eta 12m 19s

    
clImdb_v1.tar.gz     17%[==>                 ]  14.17M  98.3KB/s    eta 12m 19s

    
lImdb_v1.tar.gz      17%[==>                 ]  14.20M   101KB/s    eta 12m 19s

    
Imdb_v1.tar.gz       17%[==>                 ]  14.23M   104KB/s    eta 12m 19s

    
mdb_v1.tar.gz        17%[==>                 ]  14.26M   107KB/s    eta 12m 19s

    
db_v1.tar.gz         17%[==>                 ]  14.29M   114KB/s    eta 12m 12s

    
b_v1.tar.gz          17%[==>                 ]  14.32M   117KB/s    eta 12m 12s

    
_v1.tar.gz           17%[==>                 ]  14.34M   123KB/s    eta 12m 12s

    
v1.tar.gz            17%[==>                 ]  14.37M   125KB/s    eta 12m 12s

    
1.tar.gz             17%[==>                 ]  14.39M   124KB/s    eta 12m 12s

    
.tar.gz              17%[==>                 ]  14.42M   125KB/s    eta 12m 9s 

    
tar.gz               18%[==>                 ]  14.45M   125KB/s    eta 12m 9s 

    
ar.gz                18%[==>                 ]  14.46M   120KB/s    eta 12m 9s 

    
r.gz                 18%[==>                 ]  14.50M   120KB/s    eta 12m 9s 

    
.gz                  18%[==>                 ]  14.51M   123KB/s    eta 12m 8s 

    
gz                   18%[==>                 ]  14.54M   115KB/s    eta 12m 8s 

    
z                    18%[==>                 ]  14.56M   111KB/s    eta 12m 8s 

    
<div class="k-default-codeblock">
```
                 18%[==>                 ]  14.58M   115KB/s    eta 12m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  18%[==>                 ]  14.59M   120KB/s    eta 12m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  18%[==>                 ]  14.61M   105KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  18%[==>                 ]  14.63M   106KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  18%[==>                 ]  14.65M   103KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  18%[==>                 ]  14.67M   101KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  18%[==>                 ]  14.69M  98.4KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  18%[==>                 ]  14.71M  95.9KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  18%[==>                 ]  14.73M  96.0KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  18%[==>                 ]  14.75M  93.1KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  18%[==>                 ]  14.76M  90.0KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  18%[==>                 ]  14.78M  90.0KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  18%[==>                 ]  14.80M  87.3KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  18%[==>                 ]  14.82M  85.1KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  18%[==>                 ]  14.84M  83.5KB/s    eta 12m 7s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  18%[==>                 ]  14.87M  85.3KB/s    eta 12m 7s 

```
</div>
    
   datasets/aclImdb  18%[==>                 ]  14.90M  81.9KB/s    eta 12m 5s 

    
  datasets/aclImdb_  18%[==>                 ]  14.92M  87.4KB/s    eta 12m 5s 

    
 datasets/aclImdb_v  18%[==>                 ]  14.96M  86.8KB/s    eta 12m 5s 

    
datasets/aclImdb_v1  18%[==>                 ]  14.97M  85.8KB/s    eta 12m 5s 

    
atasets/aclImdb_v1.  18%[==>                 ]  15.02M  93.0KB/s    eta 12m 2s 

    
tasets/aclImdb_v1.t  18%[==>                 ]  15.04M  92.8KB/s    eta 12m 2s 

    
asets/aclImdb_v1.ta  18%[==>                 ]  15.07M  94.2KB/s    eta 12m 2s 

    
sets/aclImdb_v1.tar  18%[==>                 ]  15.10M  96.7KB/s    eta 12m 2s 

    
ets/aclImdb_v1.tar.  18%[==>                 ]  15.13M  99.1KB/s    eta 12m 0s 

    
ts/aclImdb_v1.tar.g  18%[==>                 ]  15.14M  96.1KB/s    eta 12m 0s 

    
s/aclImdb_v1.tar.gz  18%[==>                 ]  15.18M  96.9KB/s    eta 12m 0s 

    
/aclImdb_v1.tar.gz   18%[==>                 ]  15.22M  99.3KB/s    eta 12m 0s 

    
aclImdb_v1.tar.gz    18%[==>                 ]  15.24M  98.8KB/s    eta 12m 0s 

    
clImdb_v1.tar.gz     19%[==>                 ]  15.25M  98.4KB/s    eta 12m 0s 

    
lImdb_v1.tar.gz      19%[==>                 ]  15.27M  98.8KB/s    eta 12m 0s 

    
Imdb_v1.tar.gz       19%[==>                 ]  15.29M  98.9KB/s    eta 12m 0s 

    
mdb_v1.tar.gz        19%[==>                 ]  15.31M  98.9KB/s    eta 12m 0s 

    
db_v1.tar.gz         19%[==>                 ]  15.33M  98.7KB/s    eta 12m 0s 

    
b_v1.tar.gz          19%[==>                 ]  15.33M  94.1KB/s    eta 12m 0s 

    
_v1.tar.gz           19%[==>                 ]  15.36M  94.2KB/s    eta 12m 0s 

    
v1.tar.gz            19%[==>                 ]  15.37M  90.7KB/s    eta 12m 0s 

    
1.tar.gz             19%[==>                 ]  15.38M  87.6KB/s    eta 12m 0s 

    
.tar.gz              19%[==>                 ]  15.40M  87.9KB/s    eta 12m 0s 

    
tar.gz               19%[==>                 ]  15.42M  83.1KB/s    eta 12m 0s 

    
ar.gz                19%[==>                 ]  15.43M  80.8KB/s    eta 12m 0s 

    
r.gz                 19%[==>                 ]  15.46M  78.8KB/s    eta 12m 0s 

    
.gz                  19%[==>                 ]  15.47M  78.6KB/s    eta 12m 0s 

    
gz                   19%[==>                 ]  15.49M  78.4KB/s    eta 12m 0s 

    
z                    19%[==>                 ]  15.51M  74.9KB/s    eta 12m 0s 

    
<div class="k-default-codeblock">
```
                 19%[==>                 ]  15.53M  78.3KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  19%[==>                 ]  15.55M  76.5KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  19%[==>                 ]  15.57M  73.6KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  19%[==>                 ]  15.59M  73.8KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  19%[==>                 ]  15.60M  71.5KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  19%[==>                 ]  15.62M  72.1KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  19%[==>                 ]  15.63M  71.1KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  19%[==>                 ]  15.65M  70.0KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  19%[==>                 ]  15.66M  69.5KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  19%[==>                 ]  15.68M  72.7KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  19%[==>                 ]  15.70M  74.0KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  19%[==>                 ]  15.72M  72.5KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  19%[==>                 ]  15.74M  74.1KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  19%[==>                 ]  15.76M  68.8KB/s    eta 12m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  19%[==>                 ]  15.78M  69.8KB/s    eta 12m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  19%[==>                 ]  15.80M  70.0KB/s    eta 12m 2s 

```
</div>
    
   datasets/aclImdb  19%[==>                 ]  15.81M  68.9KB/s    eta 12m 2s 

    
  datasets/aclImdb_  19%[==>                 ]  15.83M  68.3KB/s    eta 12m 2s 

    
 datasets/aclImdb_v  19%[==>                 ]  15.85M  68.4KB/s    eta 12m 2s 

    
datasets/aclImdb_v1  19%[==>                 ]  15.87M  68.6KB/s    eta 12m 2s 

    
atasets/aclImdb_v1.  19%[==>                 ]  15.87M  65.5KB/s    eta 12m 2s 

    
tasets/aclImdb_v1.t  19%[==>                 ]  15.90M  66.7KB/s    eta 12m 2s 

    
asets/aclImdb_v1.ta  19%[==>                 ]  15.91M  65.5KB/s    eta 12m 2s 

    
sets/aclImdb_v1.tar  19%[==>                 ]  15.93M  67.4KB/s    eta 12m 2s 

    
ets/aclImdb_v1.tar.  19%[==>                 ]  15.94M  65.2KB/s    eta 12m 2s 

    
ts/aclImdb_v1.tar.g  19%[==>                 ]  15.96M  67.0KB/s    eta 12m 2s 

    
s/aclImdb_v1.tar.gz  19%[==>                 ]  15.98M  67.9KB/s    eta 12m 2s 

    
/aclImdb_v1.tar.gz   19%[==>                 ]  16.00M  69.1KB/s    eta 12m 2s 

    
aclImdb_v1.tar.gz    19%[==>                 ]  16.02M  70.3KB/s    eta 12m 2s 

    
clImdb_v1.tar.gz     19%[==>                 ]  16.04M  70.8KB/s    eta 12m 2s 

    
lImdb_v1.tar.gz      20%[===>                ]  16.05M  71.0KB/s    eta 12m 2s 

    
Imdb_v1.tar.gz       20%[===>                ]  16.07M  71.0KB/s    eta 12m 2s 

    
mdb_v1.tar.gz        20%[===>                ]  16.09M  70.8KB/s    eta 12m 2s 

    
db_v1.tar.gz         20%[===>                ]  16.11M  74.9KB/s    eta 12m 2s 

    
b_v1.tar.gz          20%[===>                ]  16.13M  73.9KB/s    eta 12m 2s 

    
_v1.tar.gz           20%[===>                ]  16.15M  74.6KB/s    eta 12m 2s 

    
v1.tar.gz            20%[===>                ]  16.17M  74.7KB/s    eta 12m 2s 

    
1.tar.gz             20%[===>                ]  16.19M  75.2KB/s    eta 12m 1s 

    
.tar.gz              20%[===>                ]  16.21M  74.9KB/s    eta 12m 1s 

    
tar.gz               20%[===>                ]  16.23M  75.3KB/s    eta 12m 1s 

    
ar.gz                20%[===>                ]  16.26M  79.2KB/s    eta 12m 1s 

    
r.gz                 20%[===>                ]  16.27M  74.9KB/s    eta 12m 1s 

    
.gz                  20%[===>                ]  16.29M  76.8KB/s    eta 12m 1s 

    
gz                   20%[===>                ]  16.30M  75.1KB/s    eta 12m 1s 

    
z                    20%[===>                ]  16.33M  78.2KB/s    eta 12m 1s 

    
<div class="k-default-codeblock">
```
                 20%[===>                ]  16.34M  76.4KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  20%[===>                ]  16.35M  75.3KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  20%[===>                ]  16.36M  73.6KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  20%[===>                ]  16.37M  72.1KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  20%[===>                ]  16.39M  71.4KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  20%[===>                ]  16.40M  70.0KB/s    eta 12m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  20%[===>                ]  16.42M  68.5KB/s    eta 12m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  20%[===>                ]  16.43M  64.1KB/s    eta 12m 2s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  20%[===>                ]  16.44M  63.7KB/s    eta 12m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  20%[===>                ]  16.46M  62.6KB/s    eta 12m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  20%[===>                ]  16.47M  61.7KB/s    eta 12m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  20%[===>                ]  16.49M  61.0KB/s    eta 12m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  20%[===>                ]  16.49M  58.8KB/s    eta 12m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  20%[===>                ]  16.51M  58.7KB/s    eta 12m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  20%[===>                ]  16.53M  56.9KB/s    eta 12m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  20%[===>                ]  16.54M  54.8KB/s    eta 12m 5s 

```
</div>
    
   datasets/aclImdb  20%[===>                ]  16.55M  56.1KB/s    eta 12m 5s 

    
  datasets/aclImdb_  20%[===>                ]  16.56M  53.1KB/s    eta 12m 6s 

    
 datasets/aclImdb_v  20%[===>                ]  16.57M  53.0KB/s    eta 12m 6s 

    
datasets/aclImdb_v1  20%[===>                ]  16.58M  51.7KB/s    eta 12m 6s 

    
atasets/aclImdb_v1.  20%[===>                ]  16.59M  49.3KB/s    eta 12m 6s 

    
tasets/aclImdb_v1.t  20%[===>                ]  16.60M  48.3KB/s    eta 12m 6s 

    
asets/aclImdb_v1.ta  20%[===>                ]  16.61M  47.9KB/s    eta 12m 9s 

    
sets/aclImdb_v1.tar  20%[===>                ]  16.62M  47.2KB/s    eta 12m 9s 

    
ets/aclImdb_v1.tar.  20%[===>                ]  16.62M  45.6KB/s    eta 12m 9s 

    
ts/aclImdb_v1.tar.g  20%[===>                ]  16.63M  44.6KB/s    eta 12m 9s 

    
s/aclImdb_v1.tar.gz  20%[===>                ]  16.64M  43.3KB/s    eta 12m 11s

    
/aclImdb_v1.tar.gz   20%[===>                ]  16.65M  45.3KB/s    eta 12m 11s

    
aclImdb_v1.tar.gz    20%[===>                ]  16.66M  44.4KB/s    eta 12m 11s

    
clImdb_v1.tar.gz     20%[===>                ]  16.68M  44.1KB/s    eta 12m 11s

    
lImdb_v1.tar.gz      20%[===>                ]  16.69M  43.3KB/s    eta 12m 12s

    
Imdb_v1.tar.gz       20%[===>                ]  16.70M  42.6KB/s    eta 12m 12s

    
mdb_v1.tar.gz        20%[===>                ]  16.72M  44.9KB/s    eta 12m 12s

    
db_v1.tar.gz         20%[===>                ]  16.74M  45.5KB/s    eta 12m 12s

    
b_v1.tar.gz          20%[===>                ]  16.76M  47.3KB/s    eta 12m 12s

    
_v1.tar.gz           20%[===>                ]  16.79M  49.6KB/s    eta 12m 12s

    
v1.tar.gz            20%[===>                ]  16.82M  52.3KB/s    eta 12m 12s

    
1.tar.gz             20%[===>                ]  16.84M  55.0KB/s    eta 12m 12s

    
.tar.gz              21%[===>                ]  16.85M  55.1KB/s    eta 12m 11s

    
tar.gz               21%[===>                ]  16.88M  59.0KB/s    eta 12m 11s

    
ar.gz                21%[===>                ]  16.91M  63.2KB/s    eta 12m 11s

    
r.gz                 21%[===>                ]  16.93M  64.1KB/s    eta 12m 10s

    
.gz                  21%[===>                ]  16.96M  66.9KB/s    eta 12m 10s

    
gz                   21%[===>                ]  16.97M  68.1KB/s    eta 12m 10s

    
z                    21%[===>                ]  16.99M  69.8KB/s    eta 12m 10s

    
<div class="k-default-codeblock">
```
                 21%[===>                ]  17.00M  69.3KB/s    eta 12m 11s

```
</div>
    
<div class="k-default-codeblock">
```
              d  21%[===>                ]  17.03M  72.4KB/s    eta 12m 11s

```
</div>
    
<div class="k-default-codeblock">
```
             da  21%[===>                ]  17.04M  69.5KB/s    eta 12m 11s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  21%[===>                ]  17.05M  71.5KB/s    eta 12m 11s

```
</div>
    
<div class="k-default-codeblock">
```
           data  21%[===>                ]  17.06M  71.0KB/s    eta 12m 12s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  21%[===>                ]  17.07M  67.6KB/s    eta 12m 12s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  21%[===>                ]  17.08M  66.0KB/s    eta 12m 12s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  21%[===>                ]  17.09M  63.5KB/s    eta 12m 12s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  21%[===>                ]  17.10M  61.1KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  21%[===>                ]  17.11M  57.1KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  21%[===>                ]  17.11M  53.9KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  21%[===>                ]  17.12M  53.9KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  21%[===>                ]  17.13M  50.1KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  21%[===>                ]  17.14M  46.0KB/s    eta 12m 16s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  21%[===>                ]  17.15M  44.4KB/s    eta 12m 16s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  21%[===>                ]  17.16M  43.4KB/s    eta 12m 16s

```
</div>
    
   datasets/aclImdb  21%[===>                ]  17.17M  41.6KB/s    eta 12m 16s

    
  datasets/aclImdb_  21%[===>                ]  17.18M  40.4KB/s    eta 12m 18s

    
 datasets/aclImdb_v  21%[===>                ]  17.19M  38.4KB/s    eta 12m 18s

    
datasets/aclImdb_v1  21%[===>                ]  17.20M  39.8KB/s    eta 12m 18s

    
atasets/aclImdb_v1.  21%[===>                ]  17.21M  37.3KB/s    eta 12m 18s

    
tasets/aclImdb_v1.t  21%[===>                ]  17.22M  38.8KB/s    eta 12m 18s

    
asets/aclImdb_v1.ta  21%[===>                ]  17.23M  37.2KB/s    eta 12m 20s

    
sets/aclImdb_v1.tar  21%[===>                ]  17.25M  38.1KB/s    eta 12m 20s

    
ets/aclImdb_v1.tar.  21%[===>                ]  17.26M  39.4KB/s    eta 12m 20s

    
ts/aclImdb_v1.tar.g  21%[===>                ]  17.26M  38.1KB/s    eta 12m 20s

    
s/aclImdb_v1.tar.gz  21%[===>                ]  17.27M  37.6KB/s    eta 12m 22s

    
/aclImdb_v1.tar.gz   21%[===>                ]  17.28M  38.3KB/s    eta 12m 22s

    
aclImdb_v1.tar.gz    21%[===>                ]  17.29M  37.9KB/s    eta 12m 22s

    
clImdb_v1.tar.gz     21%[===>                ]  17.30M  39.2KB/s    eta 12m 22s

    
lImdb_v1.tar.gz      21%[===>                ]  17.31M  39.9KB/s    eta 12m 23s

    
Imdb_v1.tar.gz       21%[===>                ]  17.33M  40.1KB/s    eta 12m 23s

    
mdb_v1.tar.gz        21%[===>                ]  17.33M  39.8KB/s    eta 12m 23s

    
db_v1.tar.gz         21%[===>                ]  17.35M  41.3KB/s    eta 12m 23s

    
b_v1.tar.gz          21%[===>                ]  17.36M  42.8KB/s    eta 12m 23s

    
_v1.tar.gz           21%[===>                ]  17.37M  41.8KB/s    eta 12m 25s

    
v1.tar.gz            21%[===>                ]  17.38M  42.9KB/s    eta 12m 25s

    
1.tar.gz             21%[===>                ]  17.39M  43.5KB/s    eta 12m 25s

    
.tar.gz              21%[===>                ]  17.40M  43.5KB/s    eta 12m 25s

    
tar.gz               21%[===>                ]  17.41M  43.5KB/s    eta 12m 25s

    
ar.gz                21%[===>                ]  17.42M  43.5KB/s    eta 12m 26s

    
r.gz                 21%[===>                ]  17.43M  41.5KB/s    eta 12m 26s

    
.gz                  21%[===>                ]  17.45M  41.7KB/s    eta 12m 26s

    
gz                   21%[===>                ]  17.46M  41.5KB/s    eta 12m 28s

    
z                    21%[===>                ]  17.47M  41.6KB/s    eta 12m 28s

    
<div class="k-default-codeblock">
```
                 21%[===>                ]  17.47M  39.4KB/s    eta 12m 28s

```
</div>
    
<div class="k-default-codeblock">
```
              d  21%[===>                ]  17.48M  39.6KB/s    eta 12m 28s

```
</div>
    
<div class="k-default-codeblock">
```
             da  21%[===>                ]  17.49M  39.0KB/s    eta 12m 30s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  21%[===>                ]  17.49M  37.7KB/s    eta 12m 30s

```
</div>
    
<div class="k-default-codeblock">
```
           data  21%[===>                ]  17.50M  35.9KB/s    eta 12m 30s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  21%[===>                ]  17.50M  34.6KB/s    eta 12m 30s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  21%[===>                ]  17.51M  33.9KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  21%[===>                ]  17.52M  33.1KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  21%[===>                ]  17.53M  32.7KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  21%[===>                ]  17.54M  32.8KB/s    eta 12m 33s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  21%[===>                ]  17.55M  32.1KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  21%[===>                ]  17.55M  31.4KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  21%[===>                ]  17.56M  30.8KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  21%[===>                ]  17.57M  29.3KB/s    eta 12m 35s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  21%[===>                ]  17.58M  28.0KB/s    eta 12m 38s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  21%[===>                ]  17.58M  28.7KB/s    eta 12m 38s

```
</div>
    
   datasets/aclImdb  21%[===>                ]  17.59M  27.6KB/s    eta 12m 38s

    
  datasets/aclImdb_  21%[===>                ]  17.60M  26.8KB/s    eta 12m 38s

    
 datasets/aclImdb_v  21%[===>                ]  17.61M  27.5KB/s    eta 12m 38s

    
datasets/aclImdb_v1  21%[===>                ]  17.61M  26.6KB/s    eta 12m 40s

    
atasets/aclImdb_v1.  21%[===>                ]  17.63M  28.7KB/s    eta 12m 40s

    
tasets/aclImdb_v1.t  21%[===>                ]  17.63M  28.6KB/s    eta 12m 40s

    
asets/aclImdb_v1.ta  21%[===>                ]  17.64M  29.1KB/s    eta 12m 40s

    
sets/aclImdb_v1.tar  21%[===>                ]  17.65M  28.7KB/s    eta 12m 42s

    
ets/aclImdb_v1.tar.  22%[===>                ]  17.65M  29.8KB/s    eta 12m 42s

    
ts/aclImdb_v1.tar.g  22%[===>                ]  17.67M  31.2KB/s    eta 12m 42s

    
s/aclImdb_v1.tar.gz  22%[===>                ]  17.68M  31.2KB/s    eta 12m 42s

    
/aclImdb_v1.tar.gz   22%[===>                ]  17.68M  30.8KB/s    eta 12m 44s

    
aclImdb_v1.tar.gz    22%[===>                ]  17.69M  30.6KB/s    eta 12m 44s

    
clImdb_v1.tar.gz     22%[===>                ]  17.70M  30.8KB/s    eta 12m 44s

    
lImdb_v1.tar.gz      22%[===>                ]  17.71M  30.8KB/s    eta 12m 44s

    
Imdb_v1.tar.gz       22%[===>                ]  17.71M  31.4KB/s    eta 12m 44s

    
mdb_v1.tar.gz        22%[===>                ]  17.72M  31.4KB/s    eta 12m 46s

    
db_v1.tar.gz         22%[===>                ]  17.73M  31.6KB/s    eta 12m 46s

    
b_v1.tar.gz          22%[===>                ]  17.74M  33.5KB/s    eta 12m 46s

    
_v1.tar.gz           22%[===>                ]  17.75M  33.8KB/s    eta 12m 46s

    
v1.tar.gz            22%[===>                ]  17.76M  34.4KB/s    eta 12m 46s

    
1.tar.gz             22%[===>                ]  17.78M  36.0KB/s    eta 12m 47s

    
.tar.gz              22%[===>                ]  17.79M  37.7KB/s    eta 12m 47s

    
tar.gz               22%[===>                ]  17.80M  37.4KB/s    eta 12m 47s

    
ar.gz                22%[===>                ]  17.83M  40.4KB/s    eta 12m 47s

    
r.gz                 22%[===>                ]  17.85M  42.6KB/s    eta 12m 47s

    
.gz                  22%[===>                ]  17.86M  43.7KB/s    eta 12m 47s

    
gz                   22%[===>                ]  17.88M  46.6KB/s    eta 12m 47s

    
z                    22%[===>                ]  17.90M  48.1KB/s    eta 12m 47s

    
<div class="k-default-codeblock">
```
                 22%[===>                ]  17.91M  49.3KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
              d  22%[===>                ]  17.93M  51.8KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
             da  22%[===>                ]  17.95M  54.0KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  22%[===>                ]  17.97M  54.7KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
           data  22%[===>                ]  18.00M  57.7KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  22%[===>                ]  18.01M  58.5KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  22%[===>                ]  18.03M  60.1KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  22%[===>                ]  18.04M  61.7KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  22%[===>                ]  18.06M  63.5KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  22%[===>                ]  18.08M  65.4KB/s    eta 12m 47s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  22%[===>                ]  18.10M  67.0KB/s    eta 12m 46s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  22%[===>                ]  18.12M  68.6KB/s    eta 12m 46s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  22%[===>                ]  18.14M  69.4KB/s    eta 12m 46s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  22%[===>                ]  18.16M  69.8KB/s    eta 12m 46s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  22%[===>                ]  18.18M  74.8KB/s    eta 12m 45s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  22%[===>                ]  18.20M  72.6KB/s    eta 12m 45s

```
</div>
    
   datasets/aclImdb  22%[===>                ]  18.21M  72.4KB/s    eta 12m 45s

    
  datasets/aclImdb_  22%[===>                ]  18.23M  73.1KB/s    eta 12m 45s

    
 datasets/aclImdb_v  22%[===>                ]  18.25M  73.1KB/s    eta 12m 45s

    
datasets/aclImdb_v1  22%[===>                ]  18.27M  73.8KB/s    eta 12m 45s

    
atasets/aclImdb_v1.  22%[===>                ]  18.29M  74.5KB/s    eta 12m 45s

    
tasets/aclImdb_v1.t  22%[===>                ]  18.32M  74.9KB/s    eta 12m 45s

    
asets/aclImdb_v1.ta  22%[===>                ]  18.33M  69.8KB/s    eta 12m 45s

    
sets/aclImdb_v1.tar  22%[===>                ]  18.35M  73.0KB/s    eta 12m 45s

    
ets/aclImdb_v1.tar.  22%[===>                ]  18.36M  70.2KB/s    eta 12m 45s

    
ts/aclImdb_v1.tar.g  22%[===>                ]  18.38M  73.0KB/s    eta 12m 45s

    
s/aclImdb_v1.tar.gz  22%[===>                ]  18.40M  72.9KB/s    eta 12m 45s

    
/aclImdb_v1.tar.gz   22%[===>                ]  18.42M  72.3KB/s    eta 12m 44s

    
aclImdb_v1.tar.gz    22%[===>                ]  18.43M  72.1KB/s    eta 12m 44s

    
clImdb_v1.tar.gz     22%[===>                ]  18.44M  70.3KB/s    eta 12m 44s

    
lImdb_v1.tar.gz      23%[===>                ]  18.46M  68.8KB/s    eta 12m 44s

    
Imdb_v1.tar.gz       23%[===>                ]  18.47M  67.0KB/s    eta 12m 45s

    
mdb_v1.tar.gz        23%[===>                ]  18.49M  65.7KB/s    eta 12m 45s

    
db_v1.tar.gz         23%[===>                ]  18.50M  64.2KB/s    eta 12m 45s

    
b_v1.tar.gz          23%[===>                ]  18.51M  61.5KB/s    eta 12m 45s

    
_v1.tar.gz           23%[===>                ]  18.52M  60.3KB/s    eta 12m 46s

    
v1.tar.gz            23%[===>                ]  18.54M  60.1KB/s    eta 12m 46s

    
1.tar.gz             23%[===>                ]  18.56M  59.6KB/s    eta 12m 46s

    
.tar.gz              23%[===>                ]  18.57M  57.5KB/s    eta 12m 46s

    
tar.gz               23%[===>                ]  18.59M  57.0KB/s    eta 12m 45s

    
ar.gz                23%[===>                ]  18.61M  62.4KB/s    eta 12m 45s

    
r.gz                 23%[===>                ]  18.62M  58.8KB/s    eta 12m 45s

    
.gz                  23%[===>                ]  18.64M  62.6KB/s    eta 12m 45s

    
gz                   23%[===>                ]  18.67M  61.6KB/s    eta 12m 45s

    
z                    23%[===>                ]  18.68M  61.9KB/s    eta 12m 44s

    
<div class="k-default-codeblock">
```
                 23%[===>                ]  18.70M  63.0KB/s    eta 12m 44s

```
</div>
    
<div class="k-default-codeblock">
```
              d  23%[===>                ]  18.73M  64.8KB/s    eta 12m 44s

```
</div>
    
<div class="k-default-codeblock">
```
             da  23%[===>                ]  18.75M  66.1KB/s    eta 12m 44s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  23%[===>                ]  18.77M  68.6KB/s    eta 12m 43s

```
</div>
    
<div class="k-default-codeblock">
```
           data  23%[===>                ]  18.79M  70.6KB/s    eta 12m 43s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  23%[===>                ]  18.81M  72.3KB/s    eta 12m 43s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  23%[===>                ]  18.82M  73.1KB/s    eta 12m 43s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  23%[===>                ]  18.84M  74.6KB/s    eta 12m 43s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  23%[===>                ]  18.87M  78.5KB/s    eta 12m 42s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  23%[===>                ]  18.89M  79.6KB/s    eta 12m 42s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  23%[===>                ]  18.91M  82.7KB/s    eta 12m 42s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  23%[===>                ]  18.93M  83.6KB/s    eta 12m 42s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  23%[===>                ]  18.96M  84.6KB/s    eta 12m 42s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  23%[===>                ]  18.98M  86.8KB/s    eta 12m 40s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  23%[===>                ]  19.00M  87.2KB/s    eta 12m 40s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  23%[===>                ]  19.03M  89.6KB/s    eta 12m 40s

```
</div>
    
   datasets/aclImdb  23%[===>                ]  19.05M  89.4KB/s    eta 12m 40s

    
  datasets/aclImdb_  23%[===>                ]  19.08M  91.3KB/s    eta 12m 40s

    
 datasets/aclImdb_v  23%[===>                ]  19.11M  95.8KB/s    eta 12m 37s

    
datasets/aclImdb_v1  23%[===>                ]  19.15M  99.0KB/s    eta 12m 37s

    
atasets/aclImdb_v1.  23%[===>                ]  19.19M   105KB/s    eta 12m 37s

    
tasets/aclImdb_v1.t  23%[===>                ]  19.23M   112KB/s    eta 12m 37s

    
asets/aclImdb_v1.ta  24%[===>                ]  19.28M   120KB/s    eta 12m 31s

    
sets/aclImdb_v1.tar  24%[===>                ]  19.33M   128KB/s    eta 12m 31s

    
ets/aclImdb_v1.tar.  24%[===>                ]  19.35M   121KB/s    eta 12m 31s

    
ts/aclImdb_v1.tar.g  24%[===>                ]  19.43M   137KB/s    eta 12m 31s

    
s/aclImdb_v1.tar.gz  24%[===>                ]  19.47M   141KB/s    eta 12m 25s

    
/aclImdb_v1.tar.gz   24%[===>                ]  19.52M   149KB/s    eta 12m 25s

    
aclImdb_v1.tar.gz    24%[===>                ]  19.57M   155KB/s    eta 12m 25s

    
clImdb_v1.tar.gz     24%[===>                ]  19.62M   161KB/s    eta 12m 25s

    
lImdb_v1.tar.gz      24%[===>                ]  19.63M   153KB/s    eta 12m 21s

    
Imdb_v1.tar.gz       24%[===>                ]  19.71M   162KB/s    eta 12m 21s

    
mdb_v1.tar.gz        24%[===>                ]  19.76M   169KB/s    eta 12m 21s

    
db_v1.tar.gz         24%[===>                ]  19.77M   161KB/s    eta 12m 17s

    
b_v1.tar.gz          24%[===>                ]  19.81M   159KB/s    eta 12m 17s

    
_v1.tar.gz           24%[===>                ]  19.82M   149KB/s    eta 12m 17s

    
v1.tar.gz            24%[===>                ]  19.86M   148KB/s    eta 12m 17s

    
1.tar.gz             24%[===>                ]  19.87M   144KB/s    eta 12m 15s

    
.tar.gz              24%[===>                ]  19.89M   139KB/s    eta 12m 15s

    
tar.gz               24%[===>                ]  19.91M   132KB/s    eta 12m 15s

    
ar.gz                24%[===>                ]  19.92M   126KB/s    eta 12m 15s

    
r.gz                 24%[===>                ]  19.94M   128KB/s    eta 12m 15s

    
.gz                  24%[===>                ]  19.96M   120KB/s    eta 12m 15s

    
gz                   24%[===>                ]  19.97M   109KB/s    eta 12m 15s

    
z                    24%[===>                ]  19.99M   103KB/s    eta 12m 15s

    
<div class="k-default-codeblock">
```
                 24%[===>                ]  20.00M  97.1KB/s    eta 12m 15s

```
</div>
    
<div class="k-default-codeblock">
```
              d  24%[===>                ]  20.02M  91.6KB/s    eta 12m 15s

```
</div>
    
<div class="k-default-codeblock">
```
             da  24%[===>                ]  20.04M  86.7KB/s    eta 12m 15s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  25%[====>               ]  20.06M  87.0KB/s    eta 12m 15s

```
</div>
    
<div class="k-default-codeblock">
```
           data  25%[====>               ]  20.08M  83.1KB/s    eta 12m 15s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  25%[====>               ]  20.10M  78.7KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  25%[====>               ]  20.13M  73.4KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  25%[====>               ]  20.16M  74.9KB/s    eta 12m 14s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  25%[====>               ]  20.19M  74.3KB/s    eta 12m 13s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  25%[====>               ]  20.22M  81.5KB/s    eta 12m 13s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  25%[====>               ]  20.25M  78.3KB/s    eta 12m 13s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  25%[====>               ]  20.28M  80.6KB/s    eta 12m 13s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  25%[====>               ]  20.30M  83.1KB/s    eta 12m 13s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  25%[====>               ]  20.33M  85.3KB/s    eta 12m 10s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  25%[====>               ]  20.36M  88.5KB/s    eta 12m 10s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  25%[====>               ]  20.38M  91.0KB/s    eta 12m 10s

```
</div>
    
   datasets/aclImdb  25%[====>               ]  20.41M  94.1KB/s    eta 12m 10s

    
  datasets/aclImdb_  25%[====>               ]  20.44M  99.3KB/s    eta 12m 10s

    
 datasets/aclImdb_v  25%[====>               ]  20.47M   103KB/s    eta 12m 7s 

    
datasets/aclImdb_v1  25%[====>               ]  20.50M   109KB/s    eta 12m 7s 

    
atasets/aclImdb_v1.  25%[====>               ]  20.53M   111KB/s    eta 12m 7s 

    
tasets/aclImdb_v1.t  25%[====>               ]  20.56M   113KB/s    eta 12m 7s 

    
asets/aclImdb_v1.ta  25%[====>               ]  20.59M   121KB/s    eta 12m 4s 

    
sets/aclImdb_v1.tar  25%[====>               ]  20.61M   122KB/s    eta 12m 4s 

    
ets/aclImdb_v1.tar.  25%[====>               ]  20.64M   121KB/s    eta 12m 4s 

    
ts/aclImdb_v1.tar.g  25%[====>               ]  20.66M   120KB/s    eta 12m 4s 

    
s/aclImdb_v1.tar.gz  25%[====>               ]  20.68M   119KB/s    eta 12m 4s 

    
/aclImdb_v1.tar.gz   25%[====>               ]  20.71M   121KB/s    eta 12m 1s 

    
aclImdb_v1.tar.gz    25%[====>               ]  20.74M   123KB/s    eta 12m 1s 

    
clImdb_v1.tar.gz     25%[====>               ]  20.78M   126KB/s    eta 12m 1s 

    
lImdb_v1.tar.gz      25%[====>               ]  20.80M   126KB/s    eta 12m 1s 

    
Imdb_v1.tar.gz       25%[====>               ]  20.83M   126KB/s    eta 12m 1s 

    
mdb_v1.tar.gz        26%[====>               ]  20.86M   119KB/s    eta 11m 58s

    
db_v1.tar.gz         26%[====>               ]  20.93M   128KB/s    eta 11m 58s

    
b_v1.tar.gz          26%[====>               ]  20.95M   130KB/s    eta 11m 58s

    
_v1.tar.gz           26%[====>               ]  20.99M   131KB/s    eta 11m 58s

    
v1.tar.gz            26%[====>               ]  21.01M   123KB/s    eta 11m 54s

    
1.tar.gz             26%[====>               ]  21.07M   130KB/s    eta 11m 54s

    
.tar.gz              26%[====>               ]  21.09M   128KB/s    eta 11m 54s

    
tar.gz               26%[====>               ]  21.12M   130KB/s    eta 11m 54s

    
ar.gz                26%[====>               ]  21.15M   131KB/s    eta 11m 54s

    
r.gz                 26%[====>               ]  21.18M   133KB/s    eta 11m 50s

    
.gz                  26%[====>               ]  21.19M   125KB/s    eta 11m 50s

    
gz                   26%[====>               ]  21.24M   131KB/s    eta 11m 50s

    
z                    26%[====>               ]  21.26M   130KB/s    eta 11m 50s

    
<div class="k-default-codeblock">
```
                 26%[====>               ]  21.28M   127KB/s    eta 11m 50s

```
</div>
    
<div class="k-default-codeblock">
```
              d  26%[====>               ]  21.31M   126KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
             da  26%[====>               ]  21.34M   124KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  26%[====>               ]  21.36M   124KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
           data  26%[====>               ]  21.39M   120KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  26%[====>               ]  21.41M   120KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  26%[====>               ]  21.44M   120KB/s    eta 11m 44s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  26%[====>               ]  21.48M   129KB/s    eta 11m 44s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  26%[====>               ]  21.51M   122KB/s    eta 11m 44s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  26%[====>               ]  21.54M   123KB/s    eta 11m 44s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  26%[====>               ]  21.57M   124KB/s    eta 11m 44s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  26%[====>               ]  21.60M   123KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  26%[====>               ]  21.62M   122KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  26%[====>               ]  21.65M   115KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  27%[====>               ]  21.67M   117KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  27%[====>               ]  21.68M   113KB/s    eta 11m 40s

```
</div>
    
   datasets/aclImdb  27%[====>               ]  21.72M   114KB/s    eta 11m 40s

    
  datasets/aclImdb_  27%[====>               ]  21.74M   113KB/s    eta 11m 40s

    
 datasets/aclImdb_v  27%[====>               ]  21.76M   110KB/s    eta 11m 40s

    
datasets/aclImdb_v1  27%[====>               ]  21.77M   107KB/s    eta 11m 40s

    
atasets/aclImdb_v1.  27%[====>               ]  21.79M   104KB/s    eta 11m 38s

    
tasets/aclImdb_v1.t  27%[====>               ]  21.81M  98.6KB/s    eta 11m 38s

    
asets/aclImdb_v1.ta  27%[====>               ]  21.82M  96.2KB/s    eta 11m 38s

    
sets/aclImdb_v1.tar  27%[====>               ]  21.84M  91.2KB/s    eta 11m 38s

    
ets/aclImdb_v1.tar.  27%[====>               ]  21.86M  86.6KB/s    eta 11m 38s

    
ts/aclImdb_v1.tar.g  27%[====>               ]  21.87M  84.2KB/s    eta 11m 38s

    
s/aclImdb_v1.tar.gz  27%[====>               ]  21.89M  82.5KB/s    eta 11m 38s

    
/aclImdb_v1.tar.gz   27%[====>               ]  21.91M  80.4KB/s    eta 11m 38s

    
aclImdb_v1.tar.gz    27%[====>               ]  21.93M  79.6KB/s    eta 11m 38s

    
clImdb_v1.tar.gz     27%[====>               ]  21.95M  77.1KB/s    eta 11m 37s

    
lImdb_v1.tar.gz      27%[====>               ]  21.97M  75.9KB/s    eta 11m 37s

    
Imdb_v1.tar.gz       27%[====>               ]  21.99M  80.3KB/s    eta 11m 37s

    
mdb_v1.tar.gz        27%[====>               ]  22.02M  79.6KB/s    eta 11m 37s

    
db_v1.tar.gz         27%[====>               ]  22.05M  83.5KB/s    eta 11m 37s

    
b_v1.tar.gz          27%[====>               ]  22.06M  76.6KB/s    eta 11m 36s

    
_v1.tar.gz           27%[====>               ]  22.09M  77.6KB/s    eta 11m 36s

    
v1.tar.gz            27%[====>               ]  22.12M  79.9KB/s    eta 11m 36s

    
1.tar.gz             27%[====>               ]  22.13M  78.5KB/s    eta 11m 36s

    
.tar.gz              27%[====>               ]  22.15M  79.6KB/s    eta 11m 35s

    
tar.gz               27%[====>               ]  22.17M  79.6KB/s    eta 11m 35s

    
ar.gz                27%[====>               ]  22.18M  78.7KB/s    eta 11m 35s

    
r.gz                 27%[====>               ]  22.20M  75.7KB/s    eta 11m 35s

    
.gz                  27%[====>               ]  22.22M  77.3KB/s    eta 11m 35s

    
gz                   27%[====>               ]  22.24M  76.5KB/s    eta 11m 35s

    
z                    27%[====>               ]  22.24M  71.6KB/s    eta 11m 35s

    
<div class="k-default-codeblock">
```
                 27%[====>               ]  22.27M  72.7KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
              d  27%[====>               ]  22.28M  71.0KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
             da  27%[====>               ]  22.29M  69.4KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  27%[====>               ]  22.30M  67.9KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
           data  27%[====>               ]  22.31M  65.8KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  27%[====>               ]  22.32M  63.6KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  27%[====>               ]  22.33M  60.7KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  27%[====>               ]  22.34M  57.4KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  27%[====>               ]  22.35M  52.9KB/s    eta 11m 36s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  27%[====>               ]  22.36M  54.6KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  27%[====>               ]  22.37M  51.6KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  27%[====>               ]  22.38M  48.7KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  27%[====>               ]  22.40M  49.1KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  27%[====>               ]  22.41M  47.9KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  27%[====>               ]  22.42M  46.7KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  27%[====>               ]  22.42M  44.3KB/s    eta 11m 38s

```
</div>
    
   datasets/aclImdb  27%[====>               ]  22.43M  41.2KB/s    eta 11m 40s

    
  datasets/aclImdb_  27%[====>               ]  22.45M  40.4KB/s    eta 11m 40s

    
 datasets/aclImdb_v  27%[====>               ]  22.46M  39.7KB/s    eta 11m 40s

    
datasets/aclImdb_v1  28%[====>               ]  22.47M  41.9KB/s    eta 11m 40s

    
atasets/aclImdb_v1.  28%[====>               ]  22.48M  39.9KB/s    eta 11m 40s

    
tasets/aclImdb_v1.t  28%[====>               ]  22.49M  39.9KB/s    eta 11m 41s

    
asets/aclImdb_v1.ta  28%[====>               ]  22.50M  40.0KB/s    eta 11m 41s

    
sets/aclImdb_v1.tar  28%[====>               ]  22.51M  39.6KB/s    eta 11m 41s

    
ets/aclImdb_v1.tar.  28%[====>               ]  22.53M  39.9KB/s    eta 11m 41s

    
ts/aclImdb_v1.tar.g  28%[====>               ]  22.54M  39.7KB/s    eta 11m 41s

    
s/aclImdb_v1.tar.gz  28%[====>               ]  22.55M  39.9KB/s    eta 11m 42s

    
/aclImdb_v1.tar.gz   28%[====>               ]  22.56M  40.2KB/s    eta 11m 42s

    
aclImdb_v1.tar.gz    28%[====>               ]  22.57M  41.7KB/s    eta 11m 42s

    
clImdb_v1.tar.gz     28%[====>               ]  22.58M  41.5KB/s    eta 11m 42s

    
lImdb_v1.tar.gz      28%[====>               ]  22.59M  42.3KB/s    eta 11m 42s

    
Imdb_v1.tar.gz       28%[====>               ]  22.61M  42.7KB/s    eta 11m 42s

    
mdb_v1.tar.gz        28%[====>               ]  22.62M  41.4KB/s    eta 11m 42s

    
db_v1.tar.gz         28%[====>               ]  22.64M  43.4KB/s    eta 11m 42s

    
b_v1.tar.gz          28%[====>               ]  22.65M  42.7KB/s    eta 11m 43s

    
_v1.tar.gz           28%[====>               ]  22.67M  45.4KB/s    eta 11m 43s

    
v1.tar.gz            28%[====>               ]  22.69M  51.8KB/s    eta 11m 43s

    
1.tar.gz             28%[====>               ]  22.71M  52.1KB/s    eta 11m 43s

    
.tar.gz              28%[====>               ]  22.72M  53.7KB/s    eta 11m 43s

    
tar.gz               28%[====>               ]  22.74M  54.8KB/s    eta 11m 42s

    
ar.gz                28%[====>               ]  22.76M  56.5KB/s    eta 11m 42s

    
r.gz                 28%[====>               ]  22.78M  58.4KB/s    eta 11m 42s

    
.gz                  28%[====>               ]  22.80M  60.3KB/s    eta 11m 42s

    
gz                   28%[====>               ]  22.82M  62.2KB/s    eta 11m 42s

    
z                    28%[====>               ]  22.84M  63.7KB/s    eta 11m 41s

    
<div class="k-default-codeblock">
```
                 28%[====>               ]  22.85M  64.4KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
              d  28%[====>               ]  22.87M  65.7KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
             da  28%[====>               ]  22.89M  67.7KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  28%[====>               ]  22.91M  70.0KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
           data  28%[====>               ]  22.93M  71.5KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  28%[====>               ]  22.95M  73.6KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  28%[====>               ]  22.95M  69.3KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  28%[====>               ]  22.98M  76.8KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  28%[====>               ]  23.00M  74.9KB/s    eta 11m 39s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  28%[====>               ]  23.02M  76.2KB/s    eta 11m 39s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  28%[====>               ]  23.04M  75.6KB/s    eta 11m 39s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  28%[====>               ]  23.06M  76.6KB/s    eta 11m 39s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  28%[====>               ]  23.08M  77.6KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  28%[====>               ]  23.11M  78.1KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  28%[====>               ]  23.13M  78.8KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  28%[====>               ]  23.15M  79.3KB/s    eta 11m 38s

```
</div>
    
   datasets/aclImdb  28%[====>               ]  23.17M  79.8KB/s    eta 11m 38s

    
  datasets/aclImdb_  28%[====>               ]  23.19M  80.2KB/s    eta 11m 37s

    
 datasets/aclImdb_v  28%[====>               ]  23.22M  80.6KB/s    eta 11m 37s

    
datasets/aclImdb_v1  28%[====>               ]  23.24M  80.9KB/s    eta 11m 37s

    
atasets/aclImdb_v1.  28%[====>               ]  23.26M  81.2KB/s    eta 11m 37s

    
tasets/aclImdb_v1.t  29%[====>               ]  23.28M  76.8KB/s    eta 11m 36s

    
asets/aclImdb_v1.ta  29%[====>               ]  23.29M  74.3KB/s    eta 11m 36s

    
sets/aclImdb_v1.tar  29%[====>               ]  23.34M  81.6KB/s    eta 11m 36s

    
ets/aclImdb_v1.tar.  29%[====>               ]  23.35M  81.1KB/s    eta 11m 36s

    
ts/aclImdb_v1.tar.g  29%[====>               ]  23.37M  80.4KB/s    eta 11m 36s

    
s/aclImdb_v1.tar.gz  29%[====>               ]  23.38M  83.9KB/s    eta 11m 35s

    
/aclImdb_v1.tar.gz   29%[====>               ]  23.41M  81.5KB/s    eta 11m 35s

    
aclImdb_v1.tar.gz    29%[====>               ]  23.43M  82.7KB/s    eta 11m 35s

    
clImdb_v1.tar.gz     29%[====>               ]  23.45M  82.9KB/s    eta 11m 35s

    
lImdb_v1.tar.gz      29%[====>               ]  23.46M  79.9KB/s    eta 11m 34s

    
Imdb_v1.tar.gz       29%[====>               ]  23.48M  79.6KB/s    eta 11m 34s

    
mdb_v1.tar.gz        29%[====>               ]  23.50M  78.0KB/s    eta 11m 34s

    
db_v1.tar.gz         29%[====>               ]  23.51M  77.4KB/s    eta 11m 34s

    
b_v1.tar.gz          29%[====>               ]  23.53M  75.3KB/s    eta 11m 34s

    
_v1.tar.gz           29%[====>               ]  23.54M  73.0KB/s    eta 11m 34s

    
v1.tar.gz            29%[====>               ]  23.54M  68.2KB/s    eta 11m 34s

    
1.tar.gz             29%[====>               ]  23.55M  65.2KB/s    eta 11m 34s

    
.tar.gz              29%[====>               ]  23.56M  63.0KB/s    eta 11m 36s

    
tar.gz               29%[====>               ]  23.57M  61.3KB/s    eta 11m 36s

    
ar.gz                29%[====>               ]  23.58M  59.1KB/s    eta 11m 36s

    
r.gz                 29%[====>               ]  23.58M  60.5KB/s    eta 11m 36s

    
.gz                  29%[====>               ]  23.59M  60.8KB/s    eta 11m 36s

    
gz                   29%[====>               ]  23.60M  53.0KB/s    eta 11m 37s

    
z                    29%[====>               ]  23.61M  49.5KB/s    eta 11m 37s

    
<div class="k-default-codeblock">
```
                 29%[====>               ]  23.62M  45.7KB/s    eta 11m 37s

```
</div>
    
<div class="k-default-codeblock">
```
              d  29%[====>               ]  23.63M  42.6KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
             da  29%[====>               ]  23.64M  39.4KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  29%[====>               ]  23.64M  36.1KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
           data  29%[====>               ]  23.65M  36.2KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  29%[====>               ]  23.65M  33.5KB/s    eta 11m 38s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  29%[====>               ]  23.66M  31.7KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  29%[====>               ]  23.66M  29.9KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  29%[====>               ]  23.67M  29.3KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  29%[====>               ]  23.68M  28.0KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  29%[====>               ]  23.68M  28.3KB/s    eta 11m 40s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  29%[====>               ]  23.69M  29.0KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  29%[====>               ]  23.71M  28.6KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  29%[====>               ]  23.71M  27.6KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  29%[====>               ]  23.72M  27.1KB/s    eta 11m 41s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  29%[====>               ]  23.72M  27.0KB/s    eta 11m 43s

```
</div>
    
   datasets/aclImdb  29%[====>               ]  23.73M  27.3KB/s    eta 11m 43s

    
  datasets/aclImdb_  29%[====>               ]  23.74M  27.6KB/s    eta 11m 43s

    
 datasets/aclImdb_v  29%[====>               ]  23.75M  28.2KB/s    eta 11m 43s

    
datasets/aclImdb_v1  29%[====>               ]  23.76M  28.0KB/s    eta 11m 43s

    
atasets/aclImdb_v1.  29%[====>               ]  23.77M  28.1KB/s    eta 11m 44s

    
tasets/aclImdb_v1.t  29%[====>               ]  23.78M  27.8KB/s    eta 11m 44s

    
asets/aclImdb_v1.ta  29%[====>               ]  23.78M  27.4KB/s    eta 11m 44s

    
sets/aclImdb_v1.tar  29%[====>               ]  23.79M  26.9KB/s    eta 11m 46s

    
ets/aclImdb_v1.tar.  29%[====>               ]  23.79M  26.2KB/s    eta 11m 46s

    
ts/aclImdb_v1.tar.g  29%[====>               ]  23.80M  25.9KB/s    eta 11m 46s

    
s/aclImdb_v1.tar.gz  29%[====>               ]  23.80M  25.6KB/s    eta 11m 46s

    
/aclImdb_v1.tar.gz   29%[====>               ]  23.81M  25.4KB/s    eta 11m 48s

    
aclImdb_v1.tar.gz    29%[====>               ]  23.81M  24.7KB/s    eta 11m 48s

    
clImdb_v1.tar.gz     29%[====>               ]  23.82M  24.6KB/s    eta 11m 48s

    
lImdb_v1.tar.gz      29%[====>               ]  23.82M  23.6KB/s    eta 11m 48s

    
Imdb_v1.tar.gz       29%[====>               ]  23.82M  22.5KB/s    eta 11m 50s

    
mdb_v1.tar.gz        29%[====>               ]  23.83M  21.8KB/s    eta 11m 50s

    
db_v1.tar.gz         29%[====>               ]  23.83M  21.5KB/s    eta 11m 50s

    
b_v1.tar.gz          29%[====>               ]  23.84M  21.5KB/s    eta 11m 50s

    
_v1.tar.gz           29%[====>               ]  23.85M  21.0KB/s    eta 11m 52s

    
v1.tar.gz            29%[====>               ]  23.85M  20.5KB/s    eta 11m 52s

    
1.tar.gz             29%[====>               ]  23.86M  20.1KB/s    eta 11m 52s

    
.tar.gz              29%[====>               ]  23.86M  19.6KB/s    eta 11m 52s

    
tar.gz               29%[====>               ]  23.87M  18.9KB/s    eta 11m 52s

    
ar.gz                29%[====>               ]  23.87M  17.8KB/s    eta 11m 54s

    
r.gz                 29%[====>               ]  23.87M  17.1KB/s    eta 11m 54s

    
.gz                  29%[====>               ]  23.88M  17.0KB/s    eta 11m 54s

    
gz                   29%[====>               ]  23.88M  17.2KB/s    eta 11m 54s

    
z                    29%[====>               ]  23.89M  17.3KB/s    eta 11m 56s

    
<div class="k-default-codeblock">
```
                 29%[====>               ]  23.90M  17.6KB/s    eta 11m 56s

```
</div>
    
<div class="k-default-codeblock">
```
              d  29%[====>               ]  23.90M  17.8KB/s    eta 11m 56s

```
</div>
    
<div class="k-default-codeblock">
```
             da  29%[====>               ]  23.91M  18.1KB/s    eta 11m 56s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  29%[====>               ]  23.91M  17.8KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
           data  29%[====>               ]  23.92M  17.6KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  29%[====>               ]  23.92M  18.0KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  29%[====>               ]  23.92M  17.4KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  29%[====>               ]  23.93M  18.5KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  29%[====>               ]  23.93M  18.5KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  29%[====>               ]  23.94M  18.5KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  29%[====>               ]  23.95M  18.5KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  29%[====>               ]  23.95M  18.4KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  29%[====>               ]  23.96M  18.5KB/s    eta 12m 0s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  29%[====>               ]  23.96M  18.7KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  29%[====>               ]  23.97M  19.6KB/s    eta 12m 1s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  29%[====>               ]  23.97M  20.2KB/s    eta 12m 1s 

```
</div>
    
   datasets/aclImdb  29%[====>               ]  23.98M  20.3KB/s    eta 12m 1s 

    
  datasets/aclImdb_  29%[====>               ]  23.98M  20.6KB/s    eta 12m 1s 

    
 datasets/aclImdb_v  29%[====>               ]  23.99M  20.9KB/s    eta 12m 3s 

    
datasets/aclImdb_v1  29%[====>               ]  23.99M  20.8KB/s    eta 12m 3s 

    
atasets/aclImdb_v1.  29%[====>               ]  24.00M  20.8KB/s    eta 12m 3s 

    
tasets/aclImdb_v1.t  29%[====>               ]  24.00M  20.7KB/s    eta 12m 3s 

    
asets/aclImdb_v1.ta  29%[====>               ]  24.01M  20.9KB/s    eta 12m 5s 

    
sets/aclImdb_v1.tar  29%[====>               ]  24.02M  22.0KB/s    eta 12m 5s 

    
ets/aclImdb_v1.tar.  29%[====>               ]  24.03M  23.4KB/s    eta 12m 5s 

    
ts/aclImdb_v1.tar.g  29%[====>               ]  24.03M  24.3KB/s    eta 12m 5s 

    
s/aclImdb_v1.tar.gz  29%[====>               ]  24.04M  25.6KB/s    eta 12m 6s 

    
/aclImdb_v1.tar.gz   29%[====>               ]  24.05M  25.3KB/s    eta 12m 6s 

    
aclImdb_v1.tar.gz    29%[====>               ]  24.06M  25.9KB/s    eta 12m 6s 

    
clImdb_v1.tar.gz     30%[=====>              ]  24.07M  27.1KB/s    eta 12m 6s 

    
lImdb_v1.tar.gz      30%[=====>              ]  24.08M  28.2KB/s    eta 12m 6s 

    
Imdb_v1.tar.gz       30%[=====>              ]  24.09M  30.0KB/s    eta 12m 6s 

    
mdb_v1.tar.gz        30%[=====>              ]  24.11M  32.3KB/s    eta 12m 6s 

    
db_v1.tar.gz         30%[=====>              ]  24.13M  35.2KB/s    eta 12m 6s 

    
b_v1.tar.gz          30%[=====>              ]  24.15M  38.1KB/s    eta 12m 6s 

    
_v1.tar.gz           30%[=====>              ]  24.17M  41.9KB/s    eta 12m 6s 

    
v1.tar.gz            30%[=====>              ]  24.20M  46.3KB/s    eta 12m 4s 

    
1.tar.gz             30%[=====>              ]  24.22M  50.6KB/s    eta 12m 4s 

    
.tar.gz              30%[=====>              ]  24.25M  55.6KB/s    eta 12m 4s 

    
tar.gz               30%[=====>              ]  24.29M  61.2KB/s    eta 12m 4s 

    
ar.gz                30%[=====>              ]  24.33M  67.9KB/s    eta 12m 1s 

    
r.gz                 30%[=====>              ]  24.34M  70.0KB/s    eta 12m 1s 

    
.gz                  30%[=====>              ]  24.40M  81.4KB/s    eta 12m 1s 

    
gz                   30%[=====>              ]  24.43M  86.8KB/s    eta 12m 1s 

    
z                    30%[=====>              ]  24.47M  92.9KB/s    eta 11m 58s

    
<div class="k-default-codeblock">
```
                 30%[=====>              ]  24.51M  99.9KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
              d  30%[=====>              ]  24.55M   110KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
             da  30%[=====>              ]  24.59M   116KB/s    eta 11m 58s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  30%[=====>              ]  24.64M   127KB/s    eta 11m 53s

```
</div>
    
<div class="k-default-codeblock">
```
           data  30%[=====>              ]  24.68M   134KB/s    eta 11m 53s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  30%[=====>              ]  24.72M   140KB/s    eta 11m 53s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  30%[=====>              ]  24.77M   149KB/s    eta 11m 53s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  30%[=====>              ]  24.81M   154KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  30%[=====>              ]  24.83M   148KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  30%[=====>              ]  24.86M   143KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  31%[=====>              ]  24.88M   142KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  31%[=====>              ]  24.95M   147KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  31%[=====>              ]  24.98M   147KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  31%[=====>              ]  24.99M   148KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  31%[=====>              ]  25.00M   138KB/s    eta 11m 47s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  31%[=====>              ]  25.02M   134KB/s    eta 11m 44s

```
</div>
    
   datasets/aclImdb  31%[=====>              ]  25.03M   129KB/s    eta 11m 44s

    
  datasets/aclImdb_  31%[=====>              ]  25.05M   123KB/s    eta 11m 44s

    
 datasets/aclImdb_v  31%[=====>              ]  25.07M   118KB/s    eta 11m 44s

    
datasets/aclImdb_v1  31%[=====>              ]  25.08M   116KB/s    eta 11m 44s

    
atasets/aclImdb_v1.  31%[=====>              ]  25.10M   110KB/s    eta 11m 44s

    
tasets/aclImdb_v1.t  31%[=====>              ]  25.12M   105KB/s    eta 11m 44s

    
asets/aclImdb_v1.ta  31%[=====>              ]  25.13M   103KB/s    eta 11m 44s

    
sets/aclImdb_v1.tar  31%[=====>              ]  25.14M  89.3KB/s    eta 11m 44s

    
ets/aclImdb_v1.tar.  31%[=====>              ]  25.16M  84.5KB/s    eta 11m 44s

    
ts/aclImdb_v1.tar.g  31%[=====>              ]  25.17M  83.0KB/s    eta 11m 44s

    
s/aclImdb_v1.tar.gz  31%[=====>              ]  25.18M  77.5KB/s    eta 11m 44s

    
/aclImdb_v1.tar.gz   31%[=====>              ]  25.19M  71.6KB/s    eta 11m 44s

    
aclImdb_v1.tar.gz    31%[=====>              ]  25.21M  69.2KB/s    eta 11m 45s

    
clImdb_v1.tar.gz     31%[=====>              ]  25.21M  66.1KB/s    eta 11m 45s

    
lImdb_v1.tar.gz      31%[=====>              ]  25.22M  63.2KB/s    eta 11m 45s

    
Imdb_v1.tar.gz       31%[=====>              ]  25.23M  53.1KB/s    eta 11m 45s

    
mdb_v1.tar.gz        31%[=====>              ]  25.24M  50.2KB/s    eta 11m 46s

    
db_v1.tar.gz         31%[=====>              ]  25.24M  48.9KB/s    eta 11m 46s

    
b_v1.tar.gz          31%[=====>              ]  25.25M  47.3KB/s    eta 11m 46s

    
_v1.tar.gz           31%[=====>              ]  25.26M  45.7KB/s    eta 11m 46s

    
v1.tar.gz            31%[=====>              ]  25.27M  44.3KB/s    eta 11m 47s

    
1.tar.gz             31%[=====>              ]  25.28M  42.2KB/s    eta 11m 47s

    
.tar.gz              31%[=====>              ]  25.29M  41.0KB/s    eta 11m 47s

    
tar.gz               31%[=====>              ]  25.29M  39.5KB/s    eta 11m 47s

    
ar.gz                31%[=====>              ]  25.30M  37.8KB/s    eta 11m 47s

    
r.gz                 31%[=====>              ]  25.31M  36.0KB/s    eta 11m 48s

    
.gz                  31%[=====>              ]  25.32M  35.9KB/s    eta 11m 48s

    
gz                   31%[=====>              ]  25.33M  36.7KB/s    eta 11m 48s

    
z                    31%[=====>              ]  25.35M  35.7KB/s    eta 11m 48s

    
<div class="k-default-codeblock">
```
                 31%[=====>              ]  25.36M  35.8KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
              d  31%[=====>              ]  25.37M  35.9KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
             da  31%[=====>              ]  25.38M  35.3KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  31%[=====>              ]  25.39M  37.1KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
           data  31%[=====>              ]  25.40M  38.1KB/s    eta 11m 48s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  31%[=====>              ]  25.42M  39.0KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  31%[=====>              ]  25.42M  39.3KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  31%[=====>              ]  25.44M  39.9KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  31%[=====>              ]  25.45M  40.3KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  31%[=====>              ]  25.46M  40.7KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  31%[=====>              ]  25.47M  41.0KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  31%[=====>              ]  25.48M  41.4KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  31%[=====>              ]  25.49M  42.3KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  31%[=====>              ]  25.50M  42.5KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  31%[=====>              ]  25.51M  43.0KB/s    eta 11m 49s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  31%[=====>              ]  25.53M  43.4KB/s    eta 11m 49s

```
</div>
    
   datasets/aclImdb  31%[=====>              ]  25.54M  44.4KB/s    eta 11m 49s

    
  datasets/aclImdb_  31%[=====>              ]  25.55M  43.7KB/s    eta 11m 50s

    
 datasets/aclImdb_v  31%[=====>              ]  25.56M  46.1KB/s    eta 11m 50s

    
datasets/aclImdb_v1  31%[=====>              ]  25.58M  47.0KB/s    eta 11m 50s

    
atasets/aclImdb_v1.  31%[=====>              ]  25.60M  47.9KB/s    eta 11m 50s

    
tasets/aclImdb_v1.t  31%[=====>              ]  25.62M  50.8KB/s    eta 11m 50s

    
asets/aclImdb_v1.ta  31%[=====>              ]  25.65M  53.7KB/s    eta 11m 48s

    
sets/aclImdb_v1.tar  32%[=====>              ]  25.67M  57.0KB/s    eta 11m 48s

    
ets/aclImdb_v1.tar.  32%[=====>              ]  25.71M  61.2KB/s    eta 11m 48s

    
ts/aclImdb_v1.tar.g  32%[=====>              ]  25.74M  65.0KB/s    eta 11m 48s

    
s/aclImdb_v1.tar.gz  32%[=====>              ]  25.78M  71.1KB/s    eta 11m 45s

    
/aclImdb_v1.tar.gz   32%[=====>              ]  25.83M  77.9KB/s    eta 11m 45s

    
aclImdb_v1.tar.gz    32%[=====>              ]  25.88M  86.1KB/s    eta 11m 45s

    
clImdb_v1.tar.gz     32%[=====>              ]  25.94M  95.8KB/s    eta 11m 45s

    
lImdb_v1.tar.gz      32%[=====>              ]  26.00M   107KB/s    eta 11m 39s

    
Imdb_v1.tar.gz       32%[=====>              ]  26.02M   108KB/s    eta 11m 39s

    
mdb_v1.tar.gz        32%[=====>              ]  26.06M   112KB/s    eta 11m 39s

    
db_v1.tar.gz         32%[=====>              ]  26.21M   140KB/s    eta 11m 39s

    
b_v1.tar.gz          32%[=====>              ]  26.27M   150KB/s    eta 11m 30s

    
_v1.tar.gz           32%[=====>              ]  26.33M   161KB/s    eta 11m 30s

    
v1.tar.gz            32%[=====>              ]  26.37M   167KB/s    eta 11m 30s

    
1.tar.gz             32%[=====>              ]  26.41M   174KB/s    eta 11m 30s

    
.tar.gz              32%[=====>              ]  26.44M   176KB/s    eta 11m 26s

    
tar.gz               33%[=====>              ]  26.52M   189KB/s    eta 11m 26s

    
ar.gz                33%[=====>              ]  26.56M   193KB/s    eta 11m 26s

    
r.gz                 33%[=====>              ]  26.58M   191KB/s    eta 11m 26s

    
.gz                  33%[=====>              ]  26.59M   188KB/s    eta 11m 26s

    
gz                   33%[=====>              ]  26.62M   174KB/s    eta 11m 22s

    
z                    33%[=====>              ]  26.65M   175KB/s    eta 11m 22s

    
<div class="k-default-codeblock">
```
                 33%[=====>              ]  26.69M   174KB/s    eta 11m 22s

```
</div>
    
<div class="k-default-codeblock">
```
              d  33%[=====>              ]  26.70M   168KB/s    eta 11m 22s

```
</div>
    
<div class="k-default-codeblock">
```
             da  33%[=====>              ]  26.72M   161KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  33%[=====>              ]  26.73M   154KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
           data  33%[=====>              ]  26.74M   144KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  33%[=====>              ]  26.75M   144KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  33%[=====>              ]  26.77M   141KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  33%[=====>              ]  26.78M   116KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  33%[=====>              ]  26.80M   109KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  33%[=====>              ]  26.81M  92.2KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  33%[=====>              ]  26.83M  89.6KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  33%[=====>              ]  26.83M  83.1KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  33%[=====>              ]  26.85M  79.9KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  33%[=====>              ]  26.85M  64.9KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  33%[=====>              ]  26.86M  59.6KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  33%[=====>              ]  26.87M  57.5KB/s    eta 11m 21s

```
</div>
    
   datasets/aclImdb  33%[=====>              ]  26.88M  56.9KB/s    eta 11m 22s

    
  datasets/aclImdb_  33%[=====>              ]  26.89M  54.3KB/s    eta 11m 22s

    
 datasets/aclImdb_v  33%[=====>              ]  26.90M  49.8KB/s    eta 11m 22s

    
datasets/aclImdb_v1  33%[=====>              ]  26.91M  44.7KB/s    eta 11m 22s

    
atasets/aclImdb_v1.  33%[=====>              ]  26.92M  43.7KB/s    eta 11m 23s

    
tasets/aclImdb_v1.t  33%[=====>              ]  26.92M  42.5KB/s    eta 11m 23s

    
asets/aclImdb_v1.ta  33%[=====>              ]  26.93M  41.4KB/s    eta 11m 23s

    
sets/aclImdb_v1.tar  33%[=====>              ]  26.94M  40.2KB/s    eta 11m 23s

    
ets/aclImdb_v1.tar.  33%[=====>              ]  26.95M  39.1KB/s    eta 11m 23s

    
ts/aclImdb_v1.tar.g  33%[=====>              ]  26.96M  36.2KB/s    eta 11m 24s

    
s/aclImdb_v1.tar.gz  33%[=====>              ]  26.96M  35.9KB/s    eta 11m 24s

    
/aclImdb_v1.tar.gz   33%[=====>              ]  26.97M  33.3KB/s    eta 11m 24s

    
aclImdb_v1.tar.gz    33%[=====>              ]  26.98M  33.3KB/s    eta 11m 24s

    
clImdb_v1.tar.gz     33%[=====>              ]  26.98M  32.9KB/s    eta 11m 24s

    
lImdb_v1.tar.gz      33%[=====>              ]  26.99M  32.4KB/s    eta 11m 25s

    
Imdb_v1.tar.gz       33%[=====>              ]  27.00M  32.5KB/s    eta 11m 25s

    
mdb_v1.tar.gz        33%[=====>              ]  27.01M  31.7KB/s    eta 11m 25s

    
db_v1.tar.gz         33%[=====>              ]  27.02M  32.3KB/s    eta 11m 25s

    
b_v1.tar.gz          33%[=====>              ]  27.03M  32.4KB/s    eta 11m 25s

    
_v1.tar.gz           33%[=====>              ]  27.03M  31.8KB/s    eta 11m 26s

    
v1.tar.gz            33%[=====>              ]  27.04M  31.7KB/s    eta 11m 26s

    
1.tar.gz             33%[=====>              ]  27.05M  32.3KB/s    eta 11m 26s

    
.tar.gz              33%[=====>              ]  27.06M  31.9KB/s    eta 11m 26s

    
tar.gz               33%[=====>              ]  27.06M  31.7KB/s    eta 11m 26s

    
ar.gz                33%[=====>              ]  27.08M  31.8KB/s    eta 11m 26s

    
r.gz                 33%[=====>              ]  27.08M  31.8KB/s    eta 11m 26s

    
.gz                  33%[=====>              ]  27.09M  31.8KB/s    eta 11m 26s

    
gz                   33%[=====>              ]  27.10M  31.9KB/s    eta 11m 27s

    
z                    33%[=====>              ]  27.11M  31.9KB/s    eta 11m 27s

    
<div class="k-default-codeblock">
```
                 33%[=====>              ]  27.12M  33.5KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
              d  33%[=====>              ]  27.12M  34.0KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
             da  33%[=====>              ]  27.13M  33.7KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  33%[=====>              ]  27.14M  34.5KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
           data  33%[=====>              ]  27.15M  34.5KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  33%[=====>              ]  27.16M  34.4KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  33%[=====>              ]  27.16M  33.3KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  33%[=====>              ]  27.18M  34.6KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  33%[=====>              ]  27.19M  34.9KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  33%[=====>              ]  27.20M  35.8KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  33%[=====>              ]  27.21M  36.8KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  33%[=====>              ]  27.23M  37.9KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  33%[=====>              ]  27.24M  40.2KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  33%[=====>              ]  27.26M  42.2KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  33%[=====>              ]  27.27M  42.7KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  34%[=====>              ]  27.28M  43.0KB/s    eta 11m 28s

```
</div>
    
   datasets/aclImdb  34%[=====>              ]  27.31M  44.5KB/s    eta 11m 28s

    
  datasets/aclImdb_  34%[=====>              ]  27.33M  47.4KB/s    eta 11m 28s

    
 datasets/aclImdb_v  34%[=====>              ]  27.34M  48.0KB/s    eta 11m 28s

    
datasets/aclImdb_v1  34%[=====>              ]  27.35M  48.0KB/s    eta 11m 28s

    
atasets/aclImdb_v1.  34%[=====>              ]  27.36M  48.2KB/s    eta 11m 28s

    
tasets/aclImdb_v1.t  34%[=====>              ]  27.37M  49.0KB/s    eta 11m 28s

    
asets/aclImdb_v1.ta  34%[=====>              ]  27.38M  49.6KB/s    eta 11m 28s

    
sets/aclImdb_v1.tar  34%[=====>              ]  27.39M  50.2KB/s    eta 11m 28s

    
ets/aclImdb_v1.tar.  34%[=====>              ]  27.40M  50.9KB/s    eta 11m 28s

    
ts/aclImdb_v1.tar.g  34%[=====>              ]  27.42M  52.1KB/s    eta 11m 28s

    
s/aclImdb_v1.tar.gz  34%[=====>              ]  27.42M  49.0KB/s    eta 11m 28s

    
/aclImdb_v1.tar.gz   34%[=====>              ]  27.44M  49.9KB/s    eta 11m 28s

    
aclImdb_v1.tar.gz    34%[=====>              ]  27.46M  50.4KB/s    eta 11m 29s

    
clImdb_v1.tar.gz     34%[=====>              ]  27.47M  50.2KB/s    eta 11m 29s

    
lImdb_v1.tar.gz      34%[=====>              ]  27.48M  49.7KB/s    eta 11m 29s

    
Imdb_v1.tar.gz       34%[=====>              ]  27.49M  48.5KB/s    eta 11m 29s

    
mdb_v1.tar.gz        34%[=====>              ]  27.50M  47.3KB/s    eta 11m 29s

    
db_v1.tar.gz         34%[=====>              ]  27.51M  47.5KB/s    eta 11m 29s

    
b_v1.tar.gz          34%[=====>              ]  27.52M  47.7KB/s    eta 11m 29s

    
_v1.tar.gz           34%[=====>              ]  27.53M  46.0KB/s    eta 11m 29s

    
v1.tar.gz            34%[=====>              ]  27.54M  47.5KB/s    eta 11m 29s

    
1.tar.gz             34%[=====>              ]  27.55M  45.3KB/s    eta 11m 29s

    
.tar.gz              34%[=====>              ]  27.56M  44.3KB/s    eta 11m 29s

    
tar.gz               34%[=====>              ]  27.58M  45.8KB/s    eta 11m 29s

    
ar.gz                34%[=====>              ]  27.59M  46.8KB/s    eta 11m 29s

    
r.gz                 34%[=====>              ]  27.60M  47.1KB/s    eta 11m 29s

    
.gz                  34%[=====>              ]  27.62M  47.4KB/s    eta 11m 29s

    
gz                   34%[=====>              ]  27.63M  48.4KB/s    eta 11m 29s

    
z                    34%[=====>              ]  27.65M  49.7KB/s    eta 11m 29s

    
<div class="k-default-codeblock">
```
                 34%[=====>              ]  27.67M  51.3KB/s    eta 11m 29s

```
</div>
    
<div class="k-default-codeblock">
```
              d  34%[=====>              ]  27.69M  56.0KB/s    eta 11m 29s

```
</div>
    
<div class="k-default-codeblock">
```
             da  34%[=====>              ]  27.71M  57.9KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  34%[=====>              ]  27.74M  60.0KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
           data  34%[=====>              ]  27.77M  63.4KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  34%[=====>              ]  27.80M  67.7KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  34%[=====>              ]  27.84M  73.5KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  34%[=====>              ]  27.88M  80.2KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  34%[=====>              ]  27.92M  86.5KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  34%[=====>              ]  27.94M  89.7KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  34%[=====>              ]  27.96M  90.0KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  34%[=====>              ]  28.00M  95.3KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  35%[======>             ]  28.08M   111KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  35%[======>             ]  28.10M   115KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  35%[======>             ]  28.12M   116KB/s    eta 11m 21s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  35%[======>             ]  28.13M   115KB/s    eta 11m 21s

```
</div>
    
   datasets/aclImdb  35%[======>             ]  28.16M   119KB/s    eta 11m 17s

    
  datasets/aclImdb_  35%[======>             ]  28.18M   121KB/s    eta 11m 17s

    
 datasets/aclImdb_v  35%[======>             ]  28.20M   116KB/s    eta 11m 17s

    
datasets/aclImdb_v1  35%[======>             ]  28.24M   120KB/s    eta 11m 17s

    
atasets/aclImdb_v1.  35%[======>             ]  28.26M   120KB/s    eta 11m 16s

    
tasets/aclImdb_v1.t  35%[======>             ]  28.28M   119KB/s    eta 11m 16s

    
asets/aclImdb_v1.ta  35%[======>             ]  28.30M   118KB/s    eta 11m 16s

    
sets/aclImdb_v1.tar  35%[======>             ]  28.32M   118KB/s    eta 11m 16s

    
ets/aclImdb_v1.tar.  35%[======>             ]  28.34M   117KB/s    eta 11m 16s

    
ts/aclImdb_v1.tar.g  35%[======>             ]  28.36M   114KB/s    eta 11m 14s

    
s/aclImdb_v1.tar.gz  35%[======>             ]  28.39M   111KB/s    eta 11m 14s

    
/aclImdb_v1.tar.gz   35%[======>             ]  28.41M   107KB/s    eta 11m 14s

    
aclImdb_v1.tar.gz    35%[======>             ]  28.43M   103KB/s    eta 11m 14s

    
clImdb_v1.tar.gz     35%[======>             ]  28.45M   102KB/s    eta 11m 13s

    
lImdb_v1.tar.gz      35%[======>             ]  28.47M   103KB/s    eta 11m 13s

    
Imdb_v1.tar.gz       35%[======>             ]  28.48M  95.9KB/s    eta 11m 13s

    
mdb_v1.tar.gz        35%[======>             ]  28.51M  82.4KB/s    eta 11m 13s

    
db_v1.tar.gz         35%[======>             ]  28.53M  80.0KB/s    eta 11m 13s

    
b_v1.tar.gz          35%[======>             ]  28.54M  77.6KB/s    eta 11m 13s

    
_v1.tar.gz           35%[======>             ]  28.56M  73.7KB/s    eta 11m 13s

    
v1.tar.gz            35%[======>             ]  28.57M  71.3KB/s    eta 11m 13s

    
1.tar.gz             35%[======>             ]  28.57M  72.1KB/s    eta 11m 13s

    
.tar.gz              35%[======>             ]  28.58M  66.4KB/s    eta 11m 13s

    
tar.gz               35%[======>             ]  28.59M  64.6KB/s    eta 11m 13s

    
ar.gz                35%[======>             ]  28.60M  62.4KB/s    eta 11m 13s

    
r.gz                 35%[======>             ]  28.61M  60.2KB/s    eta 11m 14s

    
.gz                  35%[======>             ]  28.62M  57.6KB/s    eta 11m 14s

    
gz                   35%[======>             ]  28.63M  55.5KB/s    eta 11m 14s

    
z                    35%[======>             ]  28.64M  54.8KB/s    eta 11m 14s

    
<div class="k-default-codeblock">
```
                 35%[======>             ]  28.65M  52.5KB/s    eta 11m 14s

```
</div>
    
<div class="k-default-codeblock">
```
              d  35%[======>             ]  28.67M  50.8KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
             da  35%[======>             ]  28.68M  47.0KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  35%[======>             ]  28.71M  48.2KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
           data  35%[======>             ]  28.72M  46.8KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  35%[======>             ]  28.73M  48.7KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  35%[======>             ]  28.75M  45.9KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  35%[======>             ]  28.76M  47.9KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  35%[======>             ]  28.78M  47.8KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  35%[======>             ]  28.79M  48.6KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  35%[======>             ]  28.81M  49.2KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  35%[======>             ]  28.83M  50.3KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  35%[======>             ]  28.84M  52.3KB/s    eta 11m 13s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  35%[======>             ]  28.86M  53.5KB/s    eta 11m 12s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  35%[======>             ]  28.88M  55.3KB/s    eta 11m 12s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  36%[======>             ]  28.89M  56.7KB/s    eta 11m 12s

```
</div>
    
   datasets/aclImdb  36%[======>             ]  28.91M  58.7KB/s    eta 11m 12s

    
  datasets/aclImdb_  36%[======>             ]  28.93M  60.5KB/s    eta 11m 12s

    
 datasets/aclImdb_v  36%[======>             ]  28.95M  61.8KB/s    eta 11m 11s

    
datasets/aclImdb_v1  36%[======>             ]  28.97M  62.5KB/s    eta 11m 11s

    
atasets/aclImdb_v1.  36%[======>             ]  28.98M  61.0KB/s    eta 11m 11s

    
tasets/aclImdb_v1.t  36%[======>             ]  29.01M  64.7KB/s    eta 11m 11s

    
asets/aclImdb_v1.ta  36%[======>             ]  29.03M  68.2KB/s    eta 11m 11s

    
sets/aclImdb_v1.tar  36%[======>             ]  29.05M  66.7KB/s    eta 11m 11s

    
ets/aclImdb_v1.tar.  36%[======>             ]  29.07M  68.1KB/s    eta 11m 11s

    
ts/aclImdb_v1.tar.g  36%[======>             ]  29.09M  69.6KB/s    eta 11m 11s

    
s/aclImdb_v1.tar.gz  36%[======>             ]  29.11M  71.5KB/s    eta 11m 11s

    
/aclImdb_v1.tar.gz   36%[======>             ]  29.13M  72.9KB/s    eta 11m 9s 

    
aclImdb_v1.tar.gz    36%[======>             ]  29.15M  72.3KB/s    eta 11m 9s 

    
clImdb_v1.tar.gz     36%[======>             ]  29.15M  70.4KB/s    eta 11m 9s 

    
lImdb_v1.tar.gz      36%[======>             ]  29.18M  73.3KB/s    eta 11m 9s 

    
Imdb_v1.tar.gz       36%[======>             ]  29.19M  73.1KB/s    eta 11m 9s 

    
mdb_v1.tar.gz        36%[======>             ]  29.21M  73.6KB/s    eta 11m 9s 

    
db_v1.tar.gz         36%[======>             ]  29.23M  73.9KB/s    eta 11m 9s 

    
b_v1.tar.gz          36%[======>             ]  29.25M  74.7KB/s    eta 11m 9s 

    
_v1.tar.gz           36%[======>             ]  29.27M  75.7KB/s    eta 11m 9s 

    
v1.tar.gz            36%[======>             ]  29.30M  76.4KB/s    eta 11m 7s 

    
1.tar.gz             36%[======>             ]  29.32M  77.4KB/s    eta 11m 7s 

    
.tar.gz              36%[======>             ]  29.33M  76.0KB/s    eta 11m 7s 

    
tar.gz               36%[======>             ]  29.35M  80.1KB/s    eta 11m 7s 

    
ar.gz                36%[======>             ]  29.37M  78.0KB/s    eta 11m 7s 

    
r.gz                 36%[======>             ]  29.39M  78.4KB/s    eta 11m 6s 

    
.gz                  36%[======>             ]  29.41M  78.9KB/s    eta 11m 6s 

    
gz                   36%[======>             ]  29.43M  78.8KB/s    eta 11m 6s 

    
z                    36%[======>             ]  29.45M  79.0KB/s    eta 11m 6s 

    
<div class="k-default-codeblock">
```
                 36%[======>             ]  29.47M  78.7KB/s    eta 11m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  36%[======>             ]  29.50M  78.4KB/s    eta 11m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  36%[======>             ]  29.52M  78.4KB/s    eta 11m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  36%[======>             ]  29.53M  78.2KB/s    eta 11m 5s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  36%[======>             ]  29.55M  81.9KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  36%[======>             ]  29.56M  78.1KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  36%[======>             ]  29.57M  78.0KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  36%[======>             ]  29.58M  76.7KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  36%[======>             ]  29.60M  75.7KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  36%[======>             ]  29.61M  74.5KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  36%[======>             ]  29.63M  72.6KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  36%[======>             ]  29.64M  70.6KB/s    eta 11m 4s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  36%[======>             ]  29.65M  68.4KB/s    eta 11m 3s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  36%[======>             ]  29.67M  68.7KB/s    eta 11m 3s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  36%[======>             ]  29.68M  66.4KB/s    eta 11m 3s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  37%[======>             ]  29.70M  64.7KB/s    eta 11m 3s 

```
</div>
    
   datasets/aclImdb  37%[======>             ]  29.71M  63.9KB/s    eta 11m 3s 

    
  datasets/aclImdb_  37%[======>             ]  29.73M  62.9KB/s    eta 11m 3s 

    
 datasets/aclImdb_v  37%[======>             ]  29.73M  59.3KB/s    eta 11m 3s 

    
datasets/aclImdb_v1  37%[======>             ]  29.76M  56.2KB/s    eta 11m 3s 

    
atasets/aclImdb_v1.  37%[======>             ]  29.77M  54.5KB/s    eta 11m 3s 

    
tasets/aclImdb_v1.t  37%[======>             ]  29.77M  51.2KB/s    eta 11m 3s 

    
asets/aclImdb_v1.ta  37%[======>             ]  29.78M  51.2KB/s    eta 11m 3s 

    
sets/aclImdb_v1.tar  37%[======>             ]  29.79M  48.0KB/s    eta 11m 3s 

    
ets/aclImdb_v1.tar.  37%[======>             ]  29.79M  47.1KB/s    eta 11m 4s 

    
ts/aclImdb_v1.tar.g  37%[======>             ]  29.80M  45.5KB/s    eta 11m 4s 

    
s/aclImdb_v1.tar.gz  37%[======>             ]  29.81M  44.5KB/s    eta 11m 4s 

    
/aclImdb_v1.tar.gz   37%[======>             ]  29.81M  43.4KB/s    eta 11m 4s 

    
aclImdb_v1.tar.gz    37%[======>             ]  29.83M  40.9KB/s    eta 11m 5s 

    
clImdb_v1.tar.gz     37%[======>             ]  29.83M  39.5KB/s    eta 11m 5s 

    
lImdb_v1.tar.gz      37%[======>             ]  29.84M  38.6KB/s    eta 11m 5s 

    
Imdb_v1.tar.gz       37%[======>             ]  29.85M  37.7KB/s    eta 11m 5s 

    
mdb_v1.tar.gz        37%[======>             ]  29.86M  36.9KB/s    eta 11m 6s 

    
db_v1.tar.gz         37%[======>             ]  29.86M  35.3KB/s    eta 11m 6s 

    
b_v1.tar.gz          37%[======>             ]  29.88M  35.1KB/s    eta 11m 6s 

    
_v1.tar.gz           37%[======>             ]  29.88M  33.8KB/s    eta 11m 6s 

    
v1.tar.gz            37%[======>             ]  29.89M  32.3KB/s    eta 11m 6s 

    
1.tar.gz             37%[======>             ]  29.90M  33.5KB/s    eta 11m 6s 

    
.tar.gz              37%[======>             ]  29.91M  33.1KB/s    eta 11m 6s 

    
tar.gz               37%[======>             ]  29.91M  29.9KB/s    eta 11m 6s 

    
ar.gz                37%[======>             ]  29.92M  30.4KB/s    eta 11m 6s 

    
r.gz                 37%[======>             ]  29.93M  30.9KB/s    eta 11m 7s 

    
.gz                  37%[======>             ]  29.94M  31.0KB/s    eta 11m 7s 

    
gz                   37%[======>             ]  29.94M  30.6KB/s    eta 11m 7s 

    
z                    37%[======>             ]  29.94M  29.6KB/s    eta 11m 7s 

    
<div class="k-default-codeblock">
```
                 37%[======>             ]  29.95M  30.0KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  37%[======>             ]  29.96M  30.1KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  37%[======>             ]  29.97M  29.5KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  37%[======>             ]  29.97M  30.1KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  37%[======>             ]  29.98M  30.4KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  37%[======>             ]  29.99M  30.0KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  37%[======>             ]  30.00M  30.0KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  37%[======>             ]  30.01M  29.9KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  37%[======>             ]  30.02M  30.5KB/s    eta 11m 9s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  37%[======>             ]  30.02M  29.6KB/s    eta 11m 9s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  37%[======>             ]  30.03M  28.1KB/s    eta 11m 9s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  37%[======>             ]  30.04M  29.2KB/s    eta 11m 9s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  37%[======>             ]  30.05M  29.2KB/s    eta 11m 10s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  37%[======>             ]  30.06M  29.3KB/s    eta 11m 10s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  37%[======>             ]  30.07M  30.7KB/s    eta 11m 10s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  37%[======>             ]  30.07M  29.3KB/s    eta 11m 10s

```
</div>
    
   datasets/aclImdb  37%[======>             ]  30.08M  28.1KB/s    eta 11m 11s

    
  datasets/aclImdb_  37%[======>             ]  30.08M  26.7KB/s    eta 11m 11s

    
 datasets/aclImdb_v  37%[======>             ]  30.09M  27.6KB/s    eta 11m 11s

    
datasets/aclImdb_v1  37%[======>             ]  30.10M  28.5KB/s    eta 11m 11s

    
atasets/aclImdb_v1.  37%[======>             ]  30.10M  27.6KB/s    eta 11m 11s

    
tasets/aclImdb_v1.t  37%[======>             ]  30.11M  26.1KB/s    eta 11m 11s

    
asets/aclImdb_v1.ta  37%[======>             ]  30.12M  26.5KB/s    eta 11m 12s

    
sets/aclImdb_v1.tar  37%[======>             ]  30.12M  26.2KB/s    eta 11m 12s

    
ets/aclImdb_v1.tar.  37%[======>             ]  30.13M  25.7KB/s    eta 11m 12s

    
ts/aclImdb_v1.tar.g  37%[======>             ]  30.14M  25.8KB/s    eta 11m 12s

    
s/aclImdb_v1.tar.gz  37%[======>             ]  30.14M  25.3KB/s    eta 11m 13s

    
/aclImdb_v1.tar.gz   37%[======>             ]  30.15M  24.7KB/s    eta 11m 13s

    
aclImdb_v1.tar.gz    37%[======>             ]  30.15M  24.2KB/s    eta 11m 13s

    
clImdb_v1.tar.gz     37%[======>             ]  30.16M  24.1KB/s    eta 11m 13s

    
lImdb_v1.tar.gz      37%[======>             ]  30.17M  25.2KB/s    eta 11m 14s

    
Imdb_v1.tar.gz       37%[======>             ]  30.18M  24.1KB/s    eta 11m 14s

    
mdb_v1.tar.gz        37%[======>             ]  30.18M  23.7KB/s    eta 11m 14s

    
db_v1.tar.gz         37%[======>             ]  30.19M  23.7KB/s    eta 11m 14s

    
b_v1.tar.gz          37%[======>             ]  30.20M  23.7KB/s    eta 11m 15s

    
_v1.tar.gz           37%[======>             ]  30.21M  24.3KB/s    eta 11m 15s

    
v1.tar.gz            37%[======>             ]  30.22M  25.7KB/s    eta 11m 15s

    
1.tar.gz             37%[======>             ]  30.23M  27.4KB/s    eta 11m 15s

    
.tar.gz              37%[======>             ]  30.24M  27.3KB/s    eta 11m 15s

    
tar.gz               37%[======>             ]  30.25M  27.5KB/s    eta 11m 15s

    
ar.gz                37%[======>             ]  30.26M  30.1KB/s    eta 11m 15s

    
r.gz                 37%[======>             ]  30.27M  31.4KB/s    eta 11m 15s

    
.gz                  37%[======>             ]  30.28M  31.4KB/s    eta 11m 15s

    
gz                   37%[======>             ]  30.29M  32.0KB/s    eta 11m 15s

    
z                    37%[======>             ]  30.29M  32.1KB/s    eta 11m 15s

    
<div class="k-default-codeblock">
```
                 37%[======>             ]  30.31M  33.6KB/s    eta 11m 15s

```
</div>
    
<div class="k-default-codeblock">
```
              d  37%[======>             ]  30.32M  33.9KB/s    eta 11m 16s

```
</div>
    
<div class="k-default-codeblock">
```
             da  37%[======>             ]  30.33M  34.7KB/s    eta 11m 16s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  37%[======>             ]  30.33M  34.7KB/s    eta 11m 16s

```
</div>
    
<div class="k-default-codeblock">
```
           data  37%[======>             ]  30.34M  34.2KB/s    eta 11m 16s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  37%[======>             ]  30.34M  33.6KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  37%[======>             ]  30.35M  33.3KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  37%[======>             ]  30.35M  33.3KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  37%[======>             ]  30.36M  32.6KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  37%[======>             ]  30.36M  31.9KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  37%[======>             ]  30.37M  31.2KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  37%[======>             ]  30.38M  30.0KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  37%[======>             ]  30.38M  29.1KB/s    eta 11m 17s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  37%[======>             ]  30.39M  29.2KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  37%[======>             ]  30.39M  28.6KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  37%[======>             ]  30.40M  26.3KB/s    eta 11m 18s

```
</div>
    
   datasets/aclImdb  37%[======>             ]  30.40M  25.2KB/s    eta 11m 18s

    
  datasets/aclImdb_  37%[======>             ]  30.41M  24.0KB/s    eta 11m 20s

    
 datasets/aclImdb_v  37%[======>             ]  30.41M  23.5KB/s    eta 11m 20s

    
datasets/aclImdb_v1  37%[======>             ]  30.42M  23.2KB/s    eta 11m 20s

    
atasets/aclImdb_v1.  37%[======>             ]  30.42M  21.0KB/s    eta 11m 20s

    
tasets/aclImdb_v1.t  37%[======>             ]  30.42M  20.4KB/s    eta 11m 21s

    
asets/aclImdb_v1.ta  37%[======>             ]  30.43M  19.6KB/s    eta 11m 21s

    
sets/aclImdb_v1.tar  37%[======>             ]  30.43M  19.5KB/s    eta 11m 21s

    
ets/aclImdb_v1.tar.  37%[======>             ]  30.44M  19.6KB/s    eta 11m 21s

    
ts/aclImdb_v1.tar.g  37%[======>             ]  30.44M  19.7KB/s    eta 11m 22s

    
s/aclImdb_v1.tar.gz  37%[======>             ]  30.45M  19.6KB/s    eta 11m 22s

    
/aclImdb_v1.tar.gz   37%[======>             ]  30.46M  19.4KB/s    eta 11m 22s

    
aclImdb_v1.tar.gz    37%[======>             ]  30.46M  19.4KB/s    eta 11m 22s

    
clImdb_v1.tar.gz     37%[======>             ]  30.47M  19.5KB/s    eta 11m 23s

    
lImdb_v1.tar.gz      37%[======>             ]  30.47M  19.7KB/s    eta 11m 23s

    
Imdb_v1.tar.gz       37%[======>             ]  30.48M  19.8KB/s    eta 11m 23s

    
mdb_v1.tar.gz        37%[======>             ]  30.48M  19.8KB/s    eta 11m 23s

    
db_v1.tar.gz         38%[======>             ]  30.49M  19.2KB/s    eta 11m 23s

    
b_v1.tar.gz          38%[======>             ]  30.49M  19.5KB/s    eta 11m 23s

    
_v1.tar.gz           38%[======>             ]  30.50M  20.0KB/s    eta 11m 23s

    
v1.tar.gz            38%[======>             ]  30.50M  20.2KB/s    eta 11m 23s

    
1.tar.gz             38%[======>             ]  30.51M  21.0KB/s    eta 11m 24s

    
.tar.gz              38%[======>             ]  30.52M  20.8KB/s    eta 11m 24s

    
tar.gz               38%[======>             ]  30.52M  21.0KB/s    eta 11m 24s

    
ar.gz                38%[======>             ]  30.53M  22.0KB/s    eta 11m 24s

    
r.gz                 38%[======>             ]  30.54M  22.5KB/s    eta 11m 25s

    
.gz                  38%[======>             ]  30.54M  22.4KB/s    eta 11m 25s

    
gz                   38%[======>             ]  30.55M  23.9KB/s    eta 11m 25s

    
z                    38%[======>             ]  30.56M  24.4KB/s    eta 11m 25s

    
<div class="k-default-codeblock">
```
                 38%[======>             ]  30.57M  25.1KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
              d  38%[======>             ]  30.58M  25.8KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
             da  38%[======>             ]  30.59M  27.0KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  38%[======>             ]  30.60M  28.2KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
           data  38%[======>             ]  30.61M  29.2KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  38%[======>             ]  30.62M  30.6KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  38%[======>             ]  30.64M  31.5KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  38%[======>             ]  30.65M  33.3KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  38%[======>             ]  30.67M  35.7KB/s    eta 11m 26s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  38%[======>             ]  30.68M  36.2KB/s    eta 11m 25s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  38%[======>             ]  30.70M  40.4KB/s    eta 11m 25s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  38%[======>             ]  30.72M  42.2KB/s    eta 11m 25s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  38%[======>             ]  30.73M  44.3KB/s    eta 11m 25s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  38%[======>             ]  30.75M  47.2KB/s    eta 11m 25s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  38%[======>             ]  30.77M  50.1KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  38%[======>             ]  30.79M  52.0KB/s    eta 11m 24s

```
</div>
    
   datasets/aclImdb  38%[======>             ]  30.81M  53.7KB/s    eta 11m 24s

    
  datasets/aclImdb_  38%[======>             ]  30.83M  57.5KB/s    eta 11m 24s

    
 datasets/aclImdb_v  38%[======>             ]  30.85M  58.7KB/s    eta 11m 23s

    
datasets/aclImdb_v1  38%[======>             ]  30.86M  58.9KB/s    eta 11m 23s

    
atasets/aclImdb_v1.  38%[======>             ]  30.89M  62.2KB/s    eta 11m 23s

    
tasets/aclImdb_v1.t  38%[======>             ]  30.91M  63.5KB/s    eta 11m 23s

    
asets/aclImdb_v1.ta  38%[======>             ]  30.92M  64.1KB/s    eta 11m 22s

    
sets/aclImdb_v1.tar  38%[======>             ]  30.94M  65.5KB/s    eta 11m 22s

    
ets/aclImdb_v1.tar.  38%[======>             ]  30.96M  66.6KB/s    eta 11m 22s

    
ts/aclImdb_v1.tar.g  38%[======>             ]  30.98M  67.7KB/s    eta 11m 22s

    
s/aclImdb_v1.tar.gz  38%[======>             ]  31.00M  69.0KB/s    eta 11m 21s

    
/aclImdb_v1.tar.gz   38%[======>             ]  31.02M  65.5KB/s    eta 11m 21s

    
aclImdb_v1.tar.gz    38%[======>             ]  31.04M  67.8KB/s    eta 11m 21s

    
clImdb_v1.tar.gz     38%[======>             ]  31.05M  65.0KB/s    eta 11m 21s

    
lImdb_v1.tar.gz      38%[======>             ]  31.06M  63.9KB/s    eta 11m 21s

    
Imdb_v1.tar.gz       38%[======>             ]  31.07M  62.5KB/s    eta 11m 21s

    
mdb_v1.tar.gz        38%[======>             ]  31.08M  59.1KB/s    eta 11m 21s

    
db_v1.tar.gz         38%[======>             ]  31.09M  58.1KB/s    eta 11m 21s

    
b_v1.tar.gz          38%[======>             ]  31.10M  56.5KB/s    eta 11m 21s

    
_v1.tar.gz           38%[======>             ]  31.12M  54.6KB/s    eta 11m 21s

    
v1.tar.gz            38%[======>             ]  31.13M  51.5KB/s    eta 11m 22s

    
1.tar.gz             38%[======>             ]  31.13M  50.8KB/s    eta 11m 22s

    
.tar.gz              38%[======>             ]  31.14M  48.3KB/s    eta 11m 22s

    
tar.gz               38%[======>             ]  31.15M  47.1KB/s    eta 11m 22s

    
ar.gz                38%[======>             ]  31.15M  45.9KB/s    eta 11m 22s

    
r.gz                 38%[======>             ]  31.16M  44.2KB/s    eta 11m 22s

    
.gz                  38%[======>             ]  31.17M  41.9KB/s    eta 11m 22s

    
gz                   38%[======>             ]  31.17M  39.6KB/s    eta 11m 22s

    
z                    38%[======>             ]  31.18M  37.7KB/s    eta 11m 22s

    
<div class="k-default-codeblock">
```
                 38%[======>             ]  31.18M  35.2KB/s    eta 11m 22s

```
</div>
    
<div class="k-default-codeblock">
```
              d  38%[======>             ]  31.19M  35.2KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
             da  38%[======>             ]  31.20M  33.5KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  38%[======>             ]  31.21M  33.2KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
           data  38%[======>             ]  31.22M  32.3KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  38%[======>             ]  31.23M  33.0KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  38%[======>             ]  31.25M  34.3KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  38%[======>             ]  31.26M  33.8KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  38%[======>             ]  31.27M  33.9KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  38%[======>             ]  31.28M  33.9KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  39%[======>             ]  31.29M  35.1KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  39%[======>             ]  31.30M  36.3KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  39%[======>             ]  31.31M  36.4KB/s    eta 11m 23s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  39%[======>             ]  31.32M  35.6KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  39%[======>             ]  31.33M  37.2KB/s    eta 11m 24s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  39%[======>             ]  31.34M  38.1KB/s    eta 11m 24s

```
</div>
    
   datasets/aclImdb  39%[======>             ]  31.34M  36.9KB/s    eta 11m 24s

    
  datasets/aclImdb_  39%[======>             ]  31.36M  38.4KB/s    eta 11m 24s

    
 datasets/aclImdb_v  39%[======>             ]  31.37M  38.6KB/s    eta 11m 24s

    
datasets/aclImdb_v1  39%[======>             ]  31.38M  37.1KB/s    eta 11m 24s

    
atasets/aclImdb_v1.  39%[======>             ]  31.39M  36.9KB/s    eta 11m 25s

    
tasets/aclImdb_v1.t  39%[======>             ]  31.39M  36.1KB/s    eta 11m 25s

    
asets/aclImdb_v1.ta  39%[======>             ]  31.40M  36.1KB/s    eta 11m 25s

    
sets/aclImdb_v1.tar  39%[======>             ]  31.41M  35.4KB/s    eta 11m 25s

    
ets/aclImdb_v1.tar.  39%[======>             ]  31.41M  34.9KB/s    eta 11m 25s

    
ts/aclImdb_v1.tar.g  39%[======>             ]  31.42M  34.2KB/s    eta 11m 25s

    
s/aclImdb_v1.tar.gz  39%[======>             ]  31.43M  33.4KB/s    eta 11m 25s

    
/aclImdb_v1.tar.gz   39%[======>             ]  31.44M  32.6KB/s    eta 11m 25s

    
aclImdb_v1.tar.gz    39%[======>             ]  31.45M  31.9KB/s    eta 11m 26s

    
clImdb_v1.tar.gz     39%[======>             ]  31.45M  30.7KB/s    eta 11m 26s

    
lImdb_v1.tar.gz      39%[======>             ]  31.46M  28.8KB/s    eta 11m 26s

    
Imdb_v1.tar.gz       39%[======>             ]  31.47M  28.6KB/s    eta 11m 27s

    
mdb_v1.tar.gz        39%[======>             ]  31.47M  29.0KB/s    eta 11m 27s

    
db_v1.tar.gz         39%[======>             ]  31.48M  27.9KB/s    eta 11m 27s

    
b_v1.tar.gz          39%[======>             ]  31.49M  27.6KB/s    eta 11m 27s

    
_v1.tar.gz           39%[======>             ]  31.50M  28.5KB/s    eta 11m 27s

    
v1.tar.gz            39%[======>             ]  31.51M  29.8KB/s    eta 11m 27s

    
1.tar.gz             39%[======>             ]  31.52M  29.0KB/s    eta 11m 27s

    
.tar.gz              39%[======>             ]  31.52M  28.7KB/s    eta 11m 27s

    
tar.gz               39%[======>             ]  31.53M  30.1KB/s    eta 11m 27s

    
ar.gz                39%[======>             ]  31.54M  30.0KB/s    eta 11m 27s

    
r.gz                 39%[======>             ]  31.55M  30.6KB/s    eta 11m 27s

    
.gz                  39%[======>             ]  31.56M  31.1KB/s    eta 11m 27s

    
gz                   39%[======>             ]  31.57M  31.2KB/s    eta 11m 27s

    
z                    39%[======>             ]  31.58M  31.9KB/s    eta 11m 27s

    
<div class="k-default-codeblock">
```
                 39%[======>             ]  31.58M  32.1KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
              d  39%[======>             ]  31.60M  32.2KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
             da  39%[======>             ]  31.61M  32.9KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  39%[======>             ]  31.62M  33.5KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
           data  39%[======>             ]  31.63M  35.3KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  39%[======>             ]  31.65M  37.8KB/s    eta 11m 28s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  39%[======>             ]  31.66M  39.1KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  39%[======>             ]  31.67M  40.4KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  39%[======>             ]  31.68M  40.6KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  39%[======>             ]  31.70M  42.5KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  39%[======>             ]  31.71M  43.4KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  39%[======>             ]  31.73M  44.3KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  39%[======>             ]  31.74M  45.1KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  39%[======>             ]  31.76M  45.9KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  39%[======>             ]  31.77M  46.8KB/s    eta 11m 27s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  39%[======>             ]  31.78M  47.5KB/s    eta 11m 27s

```
</div>
    
   datasets/aclImdb  39%[======>             ]  31.80M  48.9KB/s    eta 11m 27s

    
  datasets/aclImdb_  39%[======>             ]  31.82M  50.5KB/s    eta 11m 27s

    
 datasets/aclImdb_v  39%[======>             ]  31.83M  52.0KB/s    eta 11m 26s

    
datasets/aclImdb_v1  39%[======>             ]  31.85M  53.3KB/s    eta 11m 26s

    
atasets/aclImdb_v1.  39%[======>             ]  31.87M  54.5KB/s    eta 11m 26s

    
tasets/aclImdb_v1.t  39%[======>             ]  31.89M  58.2KB/s    eta 11m 26s

    
asets/aclImdb_v1.ta  39%[======>             ]  31.91M  60.7KB/s    eta 11m 25s

    
sets/aclImdb_v1.tar  39%[======>             ]  31.92M  58.7KB/s    eta 11m 25s

    
ets/aclImdb_v1.tar.  39%[======>             ]  31.92M  57.3KB/s    eta 11m 25s

    
ts/aclImdb_v1.tar.g  39%[======>             ]  31.96M  61.5KB/s    eta 11m 25s

    
s/aclImdb_v1.tar.gz  39%[======>             ]  31.99M  66.7KB/s    eta 11m 25s

    
/aclImdb_v1.tar.gz   39%[======>             ]  32.01M  67.2KB/s    eta 11m 23s

    
aclImdb_v1.tar.gz    39%[======>             ]  32.02M  68.5KB/s    eta 11m 23s

    
clImdb_v1.tar.gz     39%[======>             ]  32.04M  68.0KB/s    eta 11m 23s

    
lImdb_v1.tar.gz      39%[======>             ]  32.06M  68.4KB/s    eta 11m 23s

    
Imdb_v1.tar.gz       39%[======>             ]  32.07M  68.5KB/s    eta 11m 22s

    
mdb_v1.tar.gz        39%[======>             ]  32.09M  68.8KB/s    eta 11m 22s

    
db_v1.tar.gz         40%[=======>            ]  32.10M  69.8KB/s    eta 11m 22s

    
b_v1.tar.gz          40%[=======>            ]  32.13M  71.7KB/s    eta 11m 22s

    
_v1.tar.gz           40%[=======>            ]  32.14M  72.8KB/s    eta 11m 21s

    
v1.tar.gz            40%[=======>            ]  32.16M  73.0KB/s    eta 11m 21s

    
1.tar.gz             40%[=======>            ]  32.18M  72.9KB/s    eta 11m 21s

    
.tar.gz              40%[=======>            ]  32.19M  73.1KB/s    eta 11m 21s

    
tar.gz               40%[=======>            ]  32.21M  73.0KB/s    eta 11m 21s

    
ar.gz                40%[=======>            ]  32.23M  74.1KB/s    eta 11m 20s

    
r.gz                 40%[=======>            ]  32.25M  73.9KB/s    eta 11m 20s

    
.gz                  40%[=======>            ]  32.27M  73.4KB/s    eta 11m 20s

    
gz                   40%[=======>            ]  32.29M  78.3KB/s    eta 11m 20s

    
z                    40%[=======>            ]  32.32M  82.8KB/s    eta 11m 20s

    
<div class="k-default-codeblock">
```
                 40%[=======>            ]  32.35M  82.4KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
              d  40%[=======>            ]  32.38M  80.3KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
             da  40%[=======>            ]  32.39M  78.7KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  40%[=======>            ]  32.44M  85.9KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
           data  40%[=======>            ]  32.47M  88.7KB/s    eta 11m 18s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  40%[=======>            ]  32.50M  92.2KB/s    eta 11m 14s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  40%[=======>            ]  32.53M  96.4KB/s    eta 11m 14s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  40%[=======>            ]  32.57M   101KB/s    eta 11m 14s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  40%[=======>            ]  32.59M   103KB/s    eta 11m 14s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  40%[=======>            ]  32.63M   105KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  40%[=======>            ]  32.67M   109KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  40%[=======>            ]  32.71M   112KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  40%[=======>            ]  32.74M   114KB/s    eta 11m 11s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  40%[=======>            ]  32.78M   119KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  40%[=======>            ]  32.81M   121KB/s    eta 11m 8s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  40%[=======>            ]  32.85M   124KB/s    eta 11m 8s 

```
</div>
    
   datasets/aclImdb  40%[=======>            ]  32.89M   127KB/s    eta 11m 8s 

    
  datasets/aclImdb_  41%[=======>            ]  32.92M   130KB/s    eta 11m 4s 

    
 datasets/aclImdb_v  41%[=======>            ]  32.96M   132KB/s    eta 11m 4s 

    
datasets/aclImdb_v1  41%[=======>            ]  32.99M   134KB/s    eta 11m 4s 

    
atasets/aclImdb_v1.  41%[=======>            ]  33.02M   131KB/s    eta 11m 4s 

    
tasets/aclImdb_v1.t  41%[=======>            ]  33.06M   135KB/s    eta 11m 1s 

    
asets/aclImdb_v1.ta  41%[=======>            ]  33.09M   139KB/s    eta 11m 1s 

    
sets/aclImdb_v1.tar  41%[=======>            ]  33.12M   134KB/s    eta 11m 1s 

    
ets/aclImdb_v1.tar.  41%[=======>            ]  33.16M   134KB/s    eta 11m 1s 

    
ts/aclImdb_v1.tar.g  41%[=======>            ]  33.18M   130KB/s    eta 10m 59s

    
s/aclImdb_v1.tar.gz  41%[=======>            ]  33.22M   133KB/s    eta 10m 59s

    
/aclImdb_v1.tar.gz   41%[=======>            ]  33.24M   131KB/s    eta 10m 59s

    
aclImdb_v1.tar.gz    41%[=======>            ]  33.26M   124KB/s    eta 10m 59s

    
clImdb_v1.tar.gz     41%[=======>            ]  33.30M   127KB/s    eta 10m 56s

    
lImdb_v1.tar.gz      41%[=======>            ]  33.33M   125KB/s    eta 10m 56s

    
Imdb_v1.tar.gz       41%[=======>            ]  33.35M   122KB/s    eta 10m 56s

    
mdb_v1.tar.gz        41%[=======>            ]  33.37M   120KB/s    eta 10m 56s

    
db_v1.tar.gz         41%[=======>            ]  33.39M   116KB/s    eta 10m 55s

    
b_v1.tar.gz          41%[=======>            ]  33.42M   114KB/s    eta 10m 55s

    
_v1.tar.gz           41%[=======>            ]  33.44M   112KB/s    eta 10m 55s

    
v1.tar.gz            41%[=======>            ]  33.46M   109KB/s    eta 10m 55s

    
1.tar.gz             41%[=======>            ]  33.47M  98.7KB/s    eta 10m 54s

    
.tar.gz              41%[=======>            ]  33.51M  99.2KB/s    eta 10m 54s

    
tar.gz               41%[=======>            ]  33.53M  96.9KB/s    eta 10m 54s

    
ar.gz                41%[=======>            ]  33.55M  97.0KB/s    eta 10m 54s

    
r.gz                 41%[=======>            ]  33.57M  92.0KB/s    eta 10m 54s

    
.gz                  41%[=======>            ]  33.59M  90.8KB/s    eta 10m 51s

    
gz                   41%[=======>            ]  33.61M  88.7KB/s    eta 10m 51s

    
z                    41%[=======>            ]  33.62M  84.5KB/s    eta 10m 51s

    
<div class="k-default-codeblock">
```
                 41%[=======>            ]  33.65M  87.5KB/s    eta 10m 51s

```
</div>
    
<div class="k-default-codeblock">
```
              d  41%[=======>            ]  33.67M  88.4KB/s    eta 10m 51s

```
</div>
    
<div class="k-default-codeblock">
```
             da  41%[=======>            ]  33.68M  81.8KB/s    eta 10m 50s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  42%[=======>            ]  33.70M  85.0KB/s    eta 10m 50s

```
</div>
    
<div class="k-default-codeblock">
```
           data  42%[=======>            ]  33.73M  80.8KB/s    eta 10m 50s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  42%[=======>            ]  33.75M  80.0KB/s    eta 10m 50s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  42%[=======>            ]  33.75M  77.5KB/s    eta 10m 50s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  42%[=======>            ]  33.78M  79.0KB/s    eta 10m 49s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  42%[=======>            ]  33.79M  77.8KB/s    eta 10m 49s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  42%[=======>            ]  33.81M  77.9KB/s    eta 10m 49s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  42%[=======>            ]  33.83M  77.1KB/s    eta 10m 49s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  42%[=======>            ]  33.85M  77.9KB/s    eta 10m 49s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  42%[=======>            ]  33.87M  84.6KB/s    eta 10m 47s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  42%[=======>            ]  33.89M  78.6KB/s    eta 10m 47s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  42%[=======>            ]  33.92M  80.4KB/s    eta 10m 47s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  42%[=======>            ]  33.93M  78.9KB/s    eta 10m 47s

```
</div>
    
   datasets/aclImdb  42%[=======>            ]  33.95M  78.8KB/s    eta 10m 47s

    
  datasets/aclImdb_  42%[=======>            ]  33.97M  78.2KB/s    eta 10m 46s

    
 datasets/aclImdb_v  42%[=======>            ]  33.99M  79.6KB/s    eta 10m 46s

    
datasets/aclImdb_v1  42%[=======>            ]  34.01M  81.6KB/s    eta 10m 46s

    
atasets/aclImdb_v1.  42%[=======>            ]  34.03M  79.4KB/s    eta 10m 46s

    
tasets/aclImdb_v1.t  42%[=======>            ]  34.05M  78.9KB/s    eta 10m 45s

    
asets/aclImdb_v1.ta  42%[=======>            ]  34.07M  79.7KB/s    eta 10m 45s

    
sets/aclImdb_v1.tar  42%[=======>            ]  34.10M  81.4KB/s    eta 10m 45s

    
ets/aclImdb_v1.tar.  42%[=======>            ]  34.12M  81.0KB/s    eta 10m 45s

    
ts/aclImdb_v1.tar.g  42%[=======>            ]  34.14M  80.9KB/s    eta 10m 43s

    
s/aclImdb_v1.tar.gz  42%[=======>            ]  34.16M  83.3KB/s    eta 10m 43s

    
/aclImdb_v1.tar.gz   42%[=======>            ]  34.17M  78.6KB/s    eta 10m 43s

    
aclImdb_v1.tar.gz    42%[=======>            ]  34.19M  81.0KB/s    eta 10m 43s

    
clImdb_v1.tar.gz     42%[=======>            ]  34.21M  79.3KB/s    eta 10m 42s

    
lImdb_v1.tar.gz      42%[=======>            ]  34.22M  78.3KB/s    eta 10m 42s

    
Imdb_v1.tar.gz       42%[=======>            ]  34.23M  75.7KB/s    eta 10m 42s

    
mdb_v1.tar.gz        42%[=======>            ]  34.25M  73.6KB/s    eta 10m 42s

    
db_v1.tar.gz         42%[=======>            ]  34.26M  74.5KB/s    eta 10m 42s

    
b_v1.tar.gz          42%[=======>            ]  34.28M  71.9KB/s    eta 10m 42s

    
_v1.tar.gz           42%[=======>            ]  34.29M  70.2KB/s    eta 10m 42s

    
v1.tar.gz            42%[=======>            ]  34.31M  67.8KB/s    eta 10m 42s

    
1.tar.gz             42%[=======>            ]  34.32M  66.0KB/s    eta 10m 42s

    
.tar.gz              42%[=======>            ]  34.33M  64.7KB/s    eta 10m 42s

    
tar.gz               42%[=======>            ]  34.35M  63.5KB/s    eta 10m 42s

    
ar.gz                42%[=======>            ]  34.36M  63.2KB/s    eta 10m 42s

    
r.gz                 42%[=======>            ]  34.37M  61.2KB/s    eta 10m 42s

    
.gz                  42%[=======>            ]  34.39M  58.1KB/s    eta 10m 41s

    
gz                   42%[=======>            ]  34.40M  56.8KB/s    eta 10m 41s

    
z                    42%[=======>            ]  34.41M  54.7KB/s    eta 10m 41s

    
<div class="k-default-codeblock">
```
                 42%[=======>            ]  34.43M  54.9KB/s    eta 10m 41s

```
</div>
    
<div class="k-default-codeblock">
```
              d  42%[=======>            ]  34.44M  56.3KB/s    eta 10m 41s

```
</div>
    
<div class="k-default-codeblock">
```
             da  42%[=======>            ]  34.45M  53.5KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  42%[=======>            ]  34.47M  53.5KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
           data  42%[=======>            ]  34.48M  53.2KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  43%[=======>            ]  34.50M  54.5KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  43%[=======>            ]  34.51M  51.7KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  43%[=======>            ]  34.54M  54.2KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  43%[=======>            ]  34.56M  54.5KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  43%[=======>            ]  34.57M  54.0KB/s    eta 10m 40s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  43%[=======>            ]  34.58M  54.6KB/s    eta 10m 39s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  43%[=======>            ]  34.59M  54.9KB/s    eta 10m 39s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  43%[=======>            ]  34.60M  54.1KB/s    eta 10m 39s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  43%[=======>            ]  34.61M  53.8KB/s    eta 10m 39s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  43%[=======>            ]  34.62M  52.5KB/s    eta 10m 39s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  43%[=======>            ]  34.63M  51.5KB/s    eta 10m 39s

```
</div>
    
   datasets/aclImdb  43%[=======>            ]  34.64M  50.7KB/s    eta 10m 39s

    
  datasets/aclImdb_  43%[=======>            ]  34.65M  50.1KB/s    eta 10m 39s

    
 datasets/aclImdb_v  43%[=======>            ]  34.65M  48.7KB/s    eta 10m 39s

    
datasets/aclImdb_v1  43%[=======>            ]  34.66M  48.8KB/s    eta 10m 39s

    
atasets/aclImdb_v1.  43%[=======>            ]  34.67M  47.0KB/s    eta 10m 40s

    
tasets/aclImdb_v1.t  43%[=======>            ]  34.68M  46.6KB/s    eta 10m 40s

    
asets/aclImdb_v1.ta  43%[=======>            ]  34.69M  46.3KB/s    eta 10m 40s

    
sets/aclImdb_v1.tar  43%[=======>            ]  34.71M  46.6KB/s    eta 10m 40s

    
ets/aclImdb_v1.tar.  43%[=======>            ]  34.72M  47.1KB/s    eta 10m 40s

    
ts/aclImdb_v1.tar.g  43%[=======>            ]  34.72M  43.6KB/s    eta 10m 40s

    
s/aclImdb_v1.tar.gz  43%[=======>            ]  34.75M  49.2KB/s    eta 10m 40s

    
/aclImdb_v1.tar.gz   43%[=======>            ]  34.76M  46.3KB/s    eta 10m 40s

    
aclImdb_v1.tar.gz    43%[=======>            ]  34.78M  46.6KB/s    eta 10m 40s

    
clImdb_v1.tar.gz     43%[=======>            ]  34.80M  48.5KB/s    eta 10m 40s

    
lImdb_v1.tar.gz      43%[=======>            ]  34.81M  47.9KB/s    eta 10m 38s

    
Imdb_v1.tar.gz       43%[=======>            ]  34.83M  48.6KB/s    eta 10m 38s

    
mdb_v1.tar.gz        43%[=======>            ]  34.84M  47.6KB/s    eta 10m 38s

    
db_v1.tar.gz         43%[=======>            ]  34.85M  48.1KB/s    eta 10m 38s

    
b_v1.tar.gz          43%[=======>            ]  34.87M  50.1KB/s    eta 10m 38s

    
_v1.tar.gz           43%[=======>            ]  34.88M  51.2KB/s    eta 10m 38s

    
v1.tar.gz            43%[=======>            ]  34.90M  52.3KB/s    eta 10m 38s

    
1.tar.gz             43%[=======>            ]  34.92M  54.4KB/s    eta 10m 38s

    
.tar.gz              43%[=======>            ]  34.93M  56.0KB/s    eta 10m 38s

    
tar.gz               43%[=======>            ]  34.94M  56.4KB/s    eta 10m 37s

    
ar.gz                43%[=======>            ]  34.96M  57.2KB/s    eta 10m 37s

    
r.gz                 43%[=======>            ]  34.98M  58.8KB/s    eta 10m 37s

    
.gz                  43%[=======>            ]  35.00M  59.4KB/s    eta 10m 37s

    
gz                   43%[=======>            ]  35.01M  59.4KB/s    eta 10m 36s

    
z                    43%[=======>            ]  35.03M  59.3KB/s    eta 10m 36s

    
<div class="k-default-codeblock">
```
                 43%[=======>            ]  35.04M  62.4KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
              d  43%[=======>            ]  35.06M  62.4KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
             da  43%[=======>            ]  35.08M  60.7KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  43%[=======>            ]  35.10M  61.5KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
           data  43%[=======>            ]  35.12M  62.6KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  43%[=======>            ]  35.14M  63.7KB/s    eta 10m 36s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  43%[=======>            ]  35.17M  65.2KB/s    eta 10m 34s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  43%[=======>            ]  35.19M  70.2KB/s    eta 10m 34s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  43%[=======>            ]  35.20M  67.4KB/s    eta 10m 34s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  43%[=======>            ]  35.24M  71.9KB/s    eta 10m 34s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  43%[=======>            ]  35.26M  73.4KB/s    eta 10m 33s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  43%[=======>            ]  35.28M  75.5KB/s    eta 10m 33s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  44%[=======>            ]  35.31M  77.5KB/s    eta 10m 33s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  44%[=======>            ]  35.34M  82.3KB/s    eta 10m 33s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  44%[=======>            ]  35.36M  84.0KB/s    eta 10m 30s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  44%[=======>            ]  35.39M  89.1KB/s    eta 10m 30s

```
</div>
    
   datasets/aclImdb  44%[=======>            ]  35.42M  92.8KB/s    eta 10m 30s

    
  datasets/aclImdb_  44%[=======>            ]  35.45M  98.0KB/s    eta 10m 30s

    
 datasets/aclImdb_v  44%[=======>            ]  35.48M   101KB/s    eta 10m 28s

    
datasets/aclImdb_v1  44%[=======>            ]  35.51M   104KB/s    eta 10m 28s

    
atasets/aclImdb_v1.  44%[=======>            ]  35.55M   106KB/s    eta 10m 28s

    
tasets/aclImdb_v1.t  44%[=======>            ]  35.58M   107KB/s    eta 10m 28s

    
asets/aclImdb_v1.ta  44%[=======>            ]  35.61M   116KB/s    eta 10m 25s

    
sets/aclImdb_v1.tar  44%[=======>            ]  35.63M   105KB/s    eta 10m 25s

    
ets/aclImdb_v1.tar.  44%[=======>            ]  35.68M   110KB/s    eta 10m 25s

    
ts/aclImdb_v1.tar.g  44%[=======>            ]  35.70M   111KB/s    eta 10m 25s

    
s/aclImdb_v1.tar.gz  44%[=======>            ]  35.73M   110KB/s    eta 10m 23s

    
/aclImdb_v1.tar.gz   44%[=======>            ]  35.75M   110KB/s    eta 10m 23s

    
aclImdb_v1.tar.gz    44%[=======>            ]  35.76M   108KB/s    eta 10m 23s

    
clImdb_v1.tar.gz     44%[=======>            ]  35.79M   107KB/s    eta 10m 23s

    
lImdb_v1.tar.gz      44%[=======>            ]  35.82M   107KB/s    eta 10m 23s

    
Imdb_v1.tar.gz       44%[=======>            ]  35.85M   109KB/s    eta 10m 21s

    
mdb_v1.tar.gz        44%[=======>            ]  35.88M   110KB/s    eta 10m 21s

    
db_v1.tar.gz         44%[=======>            ]  35.90M   110KB/s    eta 10m 21s

    
b_v1.tar.gz          44%[=======>            ]  35.93M   111KB/s    eta 10m 21s

    
_v1.tar.gz           44%[=======>            ]  35.94M   108KB/s    eta 10m 21s

    
v1.tar.gz            44%[=======>            ]  35.97M   109KB/s    eta 10m 18s

    
1.tar.gz             44%[=======>            ]  36.00M   110KB/s    eta 10m 18s

    
.tar.gz              44%[=======>            ]  36.03M   104KB/s    eta 10m 18s

    
tar.gz               44%[=======>            ]  36.07M   103KB/s    eta 10m 17s

    
ar.gz                44%[=======>            ]  36.10M   107KB/s    eta 10m 17s

    
r.gz                 45%[========>           ]  36.11M   104KB/s    eta 10m 17s

    
.gz                  45%[========>           ]  36.13M   103KB/s    eta 10m 17s

    
gz                   45%[========>           ]  36.15M  98.9KB/s    eta 10m 17s

    
z                    45%[========>           ]  36.16M  95.8KB/s    eta 10m 15s

    
<div class="k-default-codeblock">
```
                 45%[========>           ]  36.18M  91.0KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
              d  45%[========>           ]  36.19M  82.1KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
             da  45%[========>           ]  36.21M  80.2KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
            dat  45%[========>           ]  36.22M  75.9KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
           data  45%[========>           ]  36.24M  77.2KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
          datas  45%[========>           ]  36.25M  73.8KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
         datase  45%[========>           ]  36.27M  70.1KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  45%[========>           ]  36.28M  68.0KB/s    eta 10m 15s

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  45%[========>           ]  36.30M  66.7KB/s    eta 10m 14s

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  45%[========>           ]  36.32M  70.6KB/s    eta 10m 14s

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  45%[========>           ]  36.33M  64.7KB/s    eta 10m 14s

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  45%[========>           ]  36.35M  66.0KB/s    eta 10m 14s

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  45%[========>           ]  36.37M  62.3KB/s    eta 10m 13s

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  45%[========>           ]  36.38M  61.2KB/s    eta 10m 13s

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  45%[========>           ]  36.40M  63.3KB/s    eta 10m 13s

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  45%[========>           ]  36.42M  62.8KB/s    eta 10m 13s

```
</div>
    
   datasets/aclImdb  45%[========>           ]  36.44M  62.5KB/s    eta 10m 12s

    
  datasets/aclImdb_  45%[========>           ]  36.46M  63.8KB/s    eta 10m 12s

    
 datasets/aclImdb_v  45%[========>           ]  36.48M  63.9KB/s    eta 10m 12s

    
datasets/aclImdb_v1  45%[========>           ]  36.50M  64.8KB/s    eta 10m 12s

    
atasets/aclImdb_v1.  45%[========>           ]  36.52M  66.9KB/s    eta 10m 11s

    
tasets/aclImdb_v1.t  45%[========>           ]  36.55M  72.6KB/s    eta 10m 11s

    
asets/aclImdb_v1.ta  45%[========>           ]  36.58M  74.8KB/s    eta 10m 11s

    
sets/aclImdb_v1.tar  45%[========>           ]  36.61M  78.4KB/s    eta 10m 11s

    
ets/aclImdb_v1.tar.  45%[========>           ]  36.65M  81.4KB/s    eta 10m 8s 

    
ts/aclImdb_v1.tar.g  45%[========>           ]  36.66M  79.8KB/s    eta 10m 8s 

    
s/aclImdb_v1.tar.gz  45%[========>           ]  36.72M  88.2KB/s    eta 10m 8s 

    
/aclImdb_v1.tar.gz   45%[========>           ]  36.73M  87.5KB/s    eta 10m 8s 

    
aclImdb_v1.tar.gz    45%[========>           ]  36.75M  87.8KB/s    eta 10m 8s 

    
clImdb_v1.tar.gz     45%[========>           ]  36.79M  93.6KB/s    eta 10m 5s 

    
lImdb_v1.tar.gz      45%[========>           ]  36.81M  95.6KB/s    eta 10m 5s 

    
Imdb_v1.tar.gz       45%[========>           ]  36.83M  97.0KB/s    eta 10m 5s 

    
mdb_v1.tar.gz        45%[========>           ]  36.86M  99.5KB/s    eta 10m 5s 

    
db_v1.tar.gz         45%[========>           ]  36.89M   102KB/s    eta 10m 5s 

    
b_v1.tar.gz          46%[========>           ]  36.91M   103KB/s    eta 10m 3s 

    
_v1.tar.gz           46%[========>           ]  36.94M   105KB/s    eta 10m 3s 

    
v1.tar.gz            46%[========>           ]  36.96M   106KB/s    eta 10m 3s 

    
1.tar.gz             46%[========>           ]  36.99M   108KB/s    eta 10m 3s 

    
.tar.gz              46%[========>           ]  37.01M   108KB/s    eta 10m 1s 

    
tar.gz               46%[========>           ]  37.04M   108KB/s    eta 10m 1s 

    
ar.gz                46%[========>           ]  37.06M   109KB/s    eta 10m 1s 

    
r.gz                 46%[========>           ]  37.09M   109KB/s    eta 10m 1s 

    
.gz                  46%[========>           ]  37.12M   108KB/s    eta 9m 59s 

    
gz                   46%[========>           ]  37.14M   107KB/s    eta 9m 59s 

    
z                    46%[========>           ]  37.17M   105KB/s    eta 9m 59s 

    
<div class="k-default-codeblock">
```
                 46%[========>           ]  37.21M   110KB/s    eta 9m 59s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  46%[========>           ]  37.24M   104KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  46%[========>           ]  37.27M   110KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  46%[========>           ]  37.30M   109KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  46%[========>           ]  37.35M   112KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  46%[========>           ]  37.38M   114KB/s    eta 9m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  46%[========>           ]  37.41M   116KB/s    eta 9m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  46%[========>           ]  37.45M   117KB/s    eta 9m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  46%[========>           ]  37.49M   119KB/s    eta 9m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  46%[========>           ]  37.53M   122KB/s    eta 9m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  46%[========>           ]  37.56M   125KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  46%[========>           ]  37.61M   129KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  46%[========>           ]  37.65M   133KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  46%[========>           ]  37.69M   136KB/s    eta 9m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  47%[========>           ]  37.73M   138KB/s    eta 9m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  47%[========>           ]  37.74M   134KB/s    eta 9m 46s 

```
</div>
    
   datasets/aclImdb  47%[========>           ]  37.80M   142KB/s    eta 9m 46s 

    
  datasets/aclImdb_  47%[========>           ]  37.83M   143KB/s    eta 9m 46s 

    
 datasets/aclImdb_v  47%[========>           ]  37.86M   144KB/s    eta 9m 46s 

    
datasets/aclImdb_v1  47%[========>           ]  37.90M   145KB/s    eta 9m 43s 

    
atasets/aclImdb_v1.  47%[========>           ]  37.93M   146KB/s    eta 9m 43s 

    
tasets/aclImdb_v1.t  47%[========>           ]  37.96M   146KB/s    eta 9m 43s 

    
asets/aclImdb_v1.ta  47%[========>           ]  38.00M   146KB/s    eta 9m 43s 

    
sets/aclImdb_v1.tar  47%[========>           ]  38.03M   146KB/s    eta 9m 40s 

    
ets/aclImdb_v1.tar.  47%[========>           ]  38.07M   141KB/s    eta 9m 40s 

    
ts/aclImdb_v1.tar.g  47%[========>           ]  38.10M   141KB/s    eta 9m 40s 

    
s/aclImdb_v1.tar.gz  47%[========>           ]  38.13M   140KB/s    eta 9m 40s 

    
/aclImdb_v1.tar.gz   47%[========>           ]  38.17M   140KB/s    eta 9m 37s 

    
aclImdb_v1.tar.gz    47%[========>           ]  38.20M   140KB/s    eta 9m 37s 

    
clImdb_v1.tar.gz     47%[========>           ]  38.23M   139KB/s    eta 9m 37s 

    
lImdb_v1.tar.gz      47%[========>           ]  38.27M   139KB/s    eta 9m 37s 

    
Imdb_v1.tar.gz       47%[========>           ]  38.31M   136KB/s    eta 9m 34s 

    
mdb_v1.tar.gz        47%[========>           ]  38.34M   135KB/s    eta 9m 34s 

    
db_v1.tar.gz         47%[========>           ]  38.39M   136KB/s    eta 9m 34s 

    
b_v1.tar.gz          47%[========>           ]  38.43M   144KB/s    eta 9m 34s 

    
_v1.tar.gz           47%[========>           ]  38.48M   140KB/s    eta 9m 31s 

    
v1.tar.gz            48%[========>           ]  38.53M   144KB/s    eta 9m 31s 

    
1.tar.gz             48%[========>           ]  38.58M   148KB/s    eta 9m 31s 

    
.tar.gz              48%[========>           ]  38.64M   152KB/s    eta 9m 31s 

    
tar.gz               48%[========>           ]  38.70M   158KB/s    eta 9m 25s 

    
ar.gz                48%[========>           ]  38.78M   167KB/s    eta 9m 25s 

    
r.gz                 48%[========>           ]  38.86M   180KB/s    eta 9m 25s 

    
.gz                  48%[========>           ]  38.94M   196KB/s    eta 9m 25s 

    
gz                   48%[========>           ]  39.04M   213KB/s    eta 9m 17s 

    
z                    48%[========>           ]  39.15M   236KB/s    eta 9m 17s 

    
<div class="k-default-codeblock">
```
                 48%[========>           ]  39.27M   255KB/s    eta 9m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  49%[========>           ]  39.40M   290KB/s    eta 9m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  49%[========>           ]  39.47M   288KB/s    eta 9m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  49%[========>           ]  39.62M   314KB/s    eta 9m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  49%[========>           ]  39.66M   314KB/s    eta 9m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  49%[========>           ]  39.88M   358KB/s    eta 9m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  49%[========>           ]  39.95M   367KB/s    eta 9m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  49%[========>           ]  40.00M   370KB/s    eta 8m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  49%[========>           ]  40.08M   392KB/s    eta 8m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  50%[=========>          ]  40.16M   390KB/s    eta 8m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  50%[=========>          ]  40.25M   411KB/s    eta 8m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  50%[=========>          ]  40.34M   405KB/s    eta 8m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  50%[=========>          ]  40.35M   381KB/s    eta 8m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  50%[=========>          ]  40.44M   374KB/s    eta 8m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  50%[=========>          ]  40.46M   358KB/s    eta 8m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  50%[=========>          ]  40.58M   340KB/s    eta 8m 45s 

```
</div>
    
   datasets/aclImdb  50%[=========>          ]  40.66M   335KB/s    eta 8m 38s 

    
  datasets/aclImdb_  50%[=========>          ]  40.69M   328KB/s    eta 8m 38s 

    
 datasets/aclImdb_v  50%[=========>          ]  40.69M   291KB/s    eta 8m 38s 

    
datasets/aclImdb_v1  50%[=========>          ]  40.75M   289KB/s    eta 8m 38s 

    
atasets/aclImdb_v1.  50%[=========>          ]  40.77M   224KB/s    eta 8m 37s 

    
tasets/aclImdb_v1.t  50%[=========>          ]  40.81M   206KB/s    eta 8m 37s 

    
asets/aclImdb_v1.ta  50%[=========>          ]  40.84M   202KB/s    eta 8m 37s 

    
sets/aclImdb_v1.tar  50%[=========>          ]  40.86M   186KB/s    eta 8m 37s 

    
ets/aclImdb_v1.tar.  50%[=========>          ]  40.87M   163KB/s    eta 8m 35s 

    
ts/aclImdb_v1.tar.g  50%[=========>          ]  40.88M   147KB/s    eta 8m 35s 

    
s/aclImdb_v1.tar.gz  50%[=========>          ]  40.90M   133KB/s    eta 8m 35s 

    
/aclImdb_v1.tar.gz   50%[=========>          ]  40.91M   127KB/s    eta 8m 35s 

    
aclImdb_v1.tar.gz    51%[=========>          ]  40.93M   129KB/s    eta 8m 35s 

    
clImdb_v1.tar.gz     51%[=========>          ]  40.94M   113KB/s    eta 8m 35s 

    
lImdb_v1.tar.gz      51%[=========>          ]  40.96M   111KB/s    eta 8m 35s 

    
Imdb_v1.tar.gz       51%[=========>          ]  40.98M  93.7KB/s    eta 8m 35s 

    
mdb_v1.tar.gz        51%[=========>          ]  40.99M  91.3KB/s    eta 8m 35s 

    
db_v1.tar.gz         51%[=========>          ]  41.01M  77.1KB/s    eta 8m 34s 

    
b_v1.tar.gz          51%[=========>          ]  41.03M  75.1KB/s    eta 8m 34s 

    
_v1.tar.gz           51%[=========>          ]  41.04M  73.0KB/s    eta 8m 34s 

    
v1.tar.gz            51%[=========>          ]  41.06M  76.0KB/s    eta 8m 34s 

    
1.tar.gz             51%[=========>          ]  41.07M  66.6KB/s    eta 8m 33s 

    
.tar.gz              51%[=========>          ]  41.09M  65.3KB/s    eta 8m 33s 

    
tar.gz               51%[=========>          ]  41.11M  67.6KB/s    eta 8m 33s 

    
ar.gz                51%[=========>          ]  41.13M  65.8KB/s    eta 8m 33s 

    
r.gz                 51%[=========>          ]  41.15M  63.1KB/s    eta 8m 32s 

    
.gz                  51%[=========>          ]  41.16M  61.7KB/s    eta 8m 32s 

    
gz                   51%[=========>          ]  41.18M  62.6KB/s    eta 8m 32s 

    
z                    51%[=========>          ]  41.20M  66.8KB/s    eta 8m 32s 

    
<div class="k-default-codeblock">
```
                 51%[=========>          ]  41.22M  66.5KB/s    eta 8m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  51%[=========>          ]  41.24M  67.5KB/s    eta 8m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  51%[=========>          ]  41.26M  68.6KB/s    eta 8m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  51%[=========>          ]  41.28M  70.2KB/s    eta 8m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  51%[=========>          ]  41.30M  71.0KB/s    eta 8m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  51%[=========>          ]  41.32M  71.9KB/s    eta 8m 30s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  51%[=========>          ]  41.35M  72.6KB/s    eta 8m 30s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  51%[=========>          ]  41.37M  73.3KB/s    eta 8m 30s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  51%[=========>          ]  41.39M  74.6KB/s    eta 8m 30s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  51%[=========>          ]  41.41M  76.1KB/s    eta 8m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  51%[=========>          ]  41.43M  77.6KB/s    eta 8m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  51%[=========>          ]  41.46M  79.1KB/s    eta 8m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  51%[=========>          ]  41.48M  81.1KB/s    eta 8m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  51%[=========>          ]  41.50M  82.4KB/s    eta 8m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  51%[=========>          ]  41.52M  83.4KB/s    eta 8m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  51%[=========>          ]  41.54M  84.7KB/s    eta 8m 27s 

```
</div>
    
   datasets/aclImdb  51%[=========>          ]  41.55M  81.9KB/s    eta 8m 27s 

    
  datasets/aclImdb_  51%[=========>          ]  41.59M  88.2KB/s    eta 8m 27s 

    
 datasets/aclImdb_v  51%[=========>          ]  41.60M  88.8KB/s    eta 8m 27s 

    
datasets/aclImdb_v1  51%[=========>          ]  41.63M  90.3KB/s    eta 8m 25s 

    
atasets/aclImdb_v1.  51%[=========>          ]  41.65M  88.9KB/s    eta 8m 25s 

    
tasets/aclImdb_v1.t  51%[=========>          ]  41.67M  90.4KB/s    eta 8m 25s 

    
asets/aclImdb_v1.ta  51%[=========>          ]  41.68M  88.9KB/s    eta 8m 25s 

    
sets/aclImdb_v1.tar  51%[=========>          ]  41.70M  88.8KB/s    eta 8m 25s 

    
ets/aclImdb_v1.tar.  52%[=========>          ]  41.73M  89.4KB/s    eta 8m 24s 

    
ts/aclImdb_v1.tar.g  52%[=========>          ]  41.76M  89.6KB/s    eta 8m 24s 

    
s/aclImdb_v1.tar.gz  52%[=========>          ]  41.78M  90.0KB/s    eta 8m 24s 

    
/aclImdb_v1.tar.gz   52%[=========>          ]  41.81M  91.3KB/s    eta 8m 24s 

    
aclImdb_v1.tar.gz    52%[=========>          ]  41.83M  92.0KB/s    eta 8m 23s 

    
clImdb_v1.tar.gz     52%[=========>          ]  41.86M  92.7KB/s    eta 8m 23s 

    
lImdb_v1.tar.gz      52%[=========>          ]  41.88M  84.5KB/s    eta 8m 22s 

    
Imdb_v1.tar.gz       52%[=========>          ]  41.90M  84.9KB/s    eta 8m 22s 

    
mdb_v1.tar.gz        52%[=========>          ]  41.94M  86.5KB/s    eta 8m 22s 

    
db_v1.tar.gz         52%[=========>          ]  41.95M  84.8KB/s    eta 8m 22s 

    
b_v1.tar.gz          52%[=========>          ]  41.96M  83.5KB/s    eta 8m 22s 

    
_v1.tar.gz           52%[=========>          ]  41.98M  81.7KB/s    eta 8m 21s 

    
v1.tar.gz            52%[=========>          ]  41.99M  82.6KB/s    eta 8m 21s 

    
1.tar.gz             52%[=========>          ]  42.01M  78.9KB/s    eta 8m 21s 

    
.tar.gz              52%[=========>          ]  42.02M  78.9KB/s    eta 8m 21s 

    
tar.gz               52%[=========>          ]  42.03M  77.3KB/s    eta 8m 21s 

    
ar.gz                52%[=========>          ]  42.05M  75.9KB/s    eta 8m 20s 

    
r.gz                 52%[=========>          ]  42.06M  74.5KB/s    eta 8m 20s 

    
.gz                  52%[=========>          ]  42.07M  68.4KB/s    eta 8m 20s 

    
gz                   52%[=========>          ]  42.09M  68.9KB/s    eta 8m 20s 

    
z                    52%[=========>          ]  42.10M  66.3KB/s    eta 8m 20s 

    
<div class="k-default-codeblock">
```
                 52%[=========>          ]  42.12M  64.5KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  52%[=========>          ]  42.13M  62.4KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  52%[=========>          ]  42.14M  59.8KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  52%[=========>          ]  42.15M  57.5KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  52%[=========>          ]  42.16M  54.7KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  52%[=========>          ]  42.17M  58.0KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  52%[=========>          ]  42.19M  56.1KB/s    eta 8m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  52%[=========>          ]  42.20M  51.6KB/s    eta 8m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  52%[=========>          ]  42.22M  52.0KB/s    eta 8m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  52%[=========>          ]  42.23M  52.4KB/s    eta 8m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  52%[=========>          ]  42.25M  53.3KB/s    eta 8m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  52%[=========>          ]  42.28M  55.1KB/s    eta 8m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  52%[=========>          ]  42.30M  57.2KB/s    eta 8m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  52%[=========>          ]  42.33M  59.3KB/s    eta 8m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  52%[=========>          ]  42.35M  61.6KB/s    eta 8m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  52%[=========>          ]  42.38M  64.2KB/s    eta 8m 17s 

```
</div>
    
   datasets/aclImdb  52%[=========>          ]  42.42M  68.2KB/s    eta 8m 17s 

    
  datasets/aclImdb_  52%[=========>          ]  42.46M  78.0KB/s    eta 8m 17s 

    
 datasets/aclImdb_v  52%[=========>          ]  42.50M  82.0KB/s    eta 8m 17s 

    
datasets/aclImdb_v1  53%[=========>          ]  42.53M  84.7KB/s    eta 8m 14s 

    
atasets/aclImdb_v1.  53%[=========>          ]  42.55M  85.2KB/s    eta 8m 14s 

    
tasets/aclImdb_v1.t  53%[=========>          ]  42.62M  98.2KB/s    eta 8m 14s 

    
asets/aclImdb_v1.ta  53%[=========>          ]  42.66M   104KB/s    eta 8m 14s 

    
sets/aclImdb_v1.tar  53%[=========>          ]  42.69M   106KB/s    eta 8m 11s 

    
ets/aclImdb_v1.tar.  53%[=========>          ]  42.71M   109KB/s    eta 8m 11s 

    
ts/aclImdb_v1.tar.g  53%[=========>          ]  42.75M   115KB/s    eta 8m 11s 

    
s/aclImdb_v1.tar.gz  53%[=========>          ]  42.76M   114KB/s    eta 8m 11s 

    
/aclImdb_v1.tar.gz   53%[=========>          ]  42.81M   122KB/s    eta 8m 11s 

    
aclImdb_v1.tar.gz    53%[=========>          ]  42.83M   123KB/s    eta 8m 9s  

    
clImdb_v1.tar.gz     53%[=========>          ]  42.85M   125KB/s    eta 8m 9s  

    
lImdb_v1.tar.gz      53%[=========>          ]  42.88M   126KB/s    eta 8m 9s  

    
Imdb_v1.tar.gz       53%[=========>          ]  42.90M   125KB/s    eta 8m 9s  

    
mdb_v1.tar.gz        53%[=========>          ]  42.93M   125KB/s    eta 8m 7s  

    
db_v1.tar.gz         53%[=========>          ]  42.95M   124KB/s    eta 8m 7s  

    
b_v1.tar.gz          53%[=========>          ]  42.98M   124KB/s    eta 8m 7s  

    
_v1.tar.gz           53%[=========>          ]  43.00M   116KB/s    eta 8m 6s  

    
v1.tar.gz            53%[=========>          ]  43.03M   117KB/s    eta 8m 6s  

    
1.tar.gz             53%[=========>          ]  43.05M   113KB/s    eta 8m 6s  

    
.tar.gz              53%[=========>          ]  43.08M   109KB/s    eta 8m 6s  

    
tar.gz               53%[=========>          ]  43.10M   109KB/s    eta 8m 5s  

    
ar.gz                53%[=========>          ]  43.13M   110KB/s    eta 8m 5s  

    
r.gz                 53%[=========>          ]  43.15M  99.9KB/s    eta 8m 5s  

    
.gz                  53%[=========>          ]  43.18M  97.2KB/s    eta 8m 5s  

    
gz                   53%[=========>          ]  43.20M  97.6KB/s    eta 8m 3s  

    
z                    53%[=========>          ]  43.23M  97.3KB/s    eta 8m 3s  

    
<div class="k-default-codeblock">
```
                 53%[=========>          ]  43.25M  93.3KB/s    eta 8m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
              d  53%[=========>          ]  43.28M  95.4KB/s    eta 8m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
             da  53%[=========>          ]  43.30M  90.6KB/s    eta 8m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  54%[=========>          ]  43.33M  91.4KB/s    eta 8m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  54%[=========>          ]  43.35M  90.7KB/s    eta 8m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  54%[=========>          ]  43.38M  91.7KB/s    eta 8m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  54%[=========>          ]  43.41M  92.6KB/s    eta 8m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  54%[=========>          ]  43.43M  93.2KB/s    eta 8m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  54%[=========>          ]  43.46M  93.9KB/s    eta 8m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  54%[=========>          ]  43.50M  96.4KB/s    eta 8m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  54%[=========>          ]  43.52M   103KB/s    eta 7m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  54%[=========>          ]  43.55M  96.8KB/s    eta 7m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  54%[=========>          ]  43.62M   105KB/s    eta 7m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  54%[=========>          ]  43.65M   108KB/s    eta 7m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  54%[=========>          ]  43.68M   109KB/s    eta 7m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  54%[=========>          ]  43.72M   112KB/s    eta 7m 55s 

```
</div>
    
   datasets/aclImdb  54%[=========>          ]  43.75M   114KB/s    eta 7m 55s 

    
  datasets/aclImdb_  54%[=========>          ]  43.79M   116KB/s    eta 7m 55s 

    
 datasets/aclImdb_v  54%[=========>          ]  43.82M   119KB/s    eta 7m 55s 

    
datasets/aclImdb_v1  54%[=========>          ]  43.86M   123KB/s    eta 7m 52s 

    
atasets/aclImdb_v1.  54%[=========>          ]  43.90M   126KB/s    eta 7m 52s 

    
tasets/aclImdb_v1.t  54%[=========>          ]  43.94M   130KB/s    eta 7m 52s 

    
asets/aclImdb_v1.ta  54%[=========>          ]  43.98M   131KB/s    eta 7m 52s 

    
sets/aclImdb_v1.tar  54%[=========>          ]  44.02M   135KB/s    eta 7m 49s 

    
ets/aclImdb_v1.tar.  54%[=========>          ]  44.06M   138KB/s    eta 7m 49s 

    
ts/aclImdb_v1.tar.g  54%[=========>          ]  44.10M   140KB/s    eta 7m 49s 

    
s/aclImdb_v1.tar.gz  55%[==========>         ]  44.14M   142KB/s    eta 7m 49s 

    
/aclImdb_v1.tar.gz   55%[==========>         ]  44.18M   145KB/s    eta 7m 46s 

    
aclImdb_v1.tar.gz    55%[==========>         ]  44.21M   147KB/s    eta 7m 46s 

    
clImdb_v1.tar.gz     55%[==========>         ]  44.23M   142KB/s    eta 7m 46s 

    
lImdb_v1.tar.gz      55%[==========>         ]  44.25M   142KB/s    eta 7m 46s 

    
Imdb_v1.tar.gz       55%[==========>         ]  44.31M   156KB/s    eta 7m 44s 

    
mdb_v1.tar.gz        55%[==========>         ]  44.34M   149KB/s    eta 7m 44s 

    
db_v1.tar.gz         55%[==========>         ]  44.38M   150KB/s    eta 7m 44s 

    
b_v1.tar.gz          55%[==========>         ]  44.41M   150KB/s    eta 7m 44s 

    
_v1.tar.gz           55%[==========>         ]  44.45M   151KB/s    eta 7m 44s 

    
v1.tar.gz            55%[==========>         ]  44.49M   152KB/s    eta 7m 41s 

    
1.tar.gz             55%[==========>         ]  44.52M   151KB/s    eta 7m 41s 

    
.tar.gz              55%[==========>         ]  44.56M   153KB/s    eta 7m 41s 

    
tar.gz               55%[==========>         ]  44.60M   151KB/s    eta 7m 41s 

    
ar.gz                55%[==========>         ]  44.63M   150KB/s    eta 7m 41s 

    
r.gz                 55%[==========>         ]  44.67M   150KB/s    eta 7m 38s 

    
.gz                  55%[==========>         ]  44.70M   151KB/s    eta 7m 38s 

    
gz                   55%[==========>         ]  44.74M   151KB/s    eta 7m 38s 

    
z                    55%[==========>         ]  44.78M   151KB/s    eta 7m 38s 

    
<div class="k-default-codeblock">
```
                 55%[==========>         ]  44.82M   151KB/s    eta 7m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  55%[==========>         ]  44.86M   152KB/s    eta 7m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  55%[==========>         ]  44.91M   153KB/s    eta 7m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  56%[==========>         ]  44.94M   153KB/s    eta 7m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  56%[==========>         ]  44.99M   160KB/s    eta 7m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  56%[==========>         ]  45.03M   162KB/s    eta 7m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  56%[==========>         ]  45.06M   153KB/s    eta 7m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  56%[==========>         ]  45.11M   157KB/s    eta 7m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  56%[==========>         ]  45.13M   155KB/s    eta 7m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  56%[==========>         ]  45.17M   155KB/s    eta 7m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  56%[==========>         ]  45.20M   153KB/s    eta 7m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  56%[==========>         ]  45.24M   151KB/s    eta 7m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  56%[==========>         ]  45.26M   149KB/s    eta 7m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  56%[==========>         ]  45.29M   145KB/s    eta 7m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  56%[==========>         ]  45.34M   142KB/s    eta 7m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  56%[==========>         ]  45.37M   142KB/s    eta 7m 27s 

```
</div>
    
   datasets/aclImdb  56%[==========>         ]  45.40M   141KB/s    eta 7m 25s 

    
  datasets/aclImdb_  56%[==========>         ]  45.43M   139KB/s    eta 7m 25s 

    
 datasets/aclImdb_v  56%[==========>         ]  45.45M   137KB/s    eta 7m 25s 

    
datasets/aclImdb_v1  56%[==========>         ]  45.48M   135KB/s    eta 7m 25s 

    
atasets/aclImdb_v1.  56%[==========>         ]  45.50M   131KB/s    eta 7m 25s 

    
tasets/aclImdb_v1.t  56%[==========>         ]  45.50M   122KB/s    eta 7m 24s 

    
asets/aclImdb_v1.ta  56%[==========>         ]  45.54M   114KB/s    eta 7m 24s 

    
sets/aclImdb_v1.tar  56%[==========>         ]  45.57M   112KB/s    eta 7m 24s 

    
ets/aclImdb_v1.tar.  56%[==========>         ]  45.59M   108KB/s    eta 7m 23s 

    
ts/aclImdb_v1.tar.g  56%[==========>         ]  45.60M   105KB/s    eta 7m 23s 

    
s/aclImdb_v1.tar.gz  56%[==========>         ]  45.61M  98.4KB/s    eta 7m 23s 

    
/aclImdb_v1.tar.gz   56%[==========>         ]  45.63M  95.7KB/s    eta 7m 23s 

    
aclImdb_v1.tar.gz    56%[==========>         ]  45.64M  91.6KB/s    eta 7m 23s 

    
clImdb_v1.tar.gz     56%[==========>         ]  45.65M  88.0KB/s    eta 7m 22s 

    
lImdb_v1.tar.gz      56%[==========>         ]  45.67M  83.0KB/s    eta 7m 22s 

    
Imdb_v1.tar.gz       56%[==========>         ]  45.68M  84.1KB/s    eta 7m 22s 

    
mdb_v1.tar.gz        56%[==========>         ]  45.70M  80.4KB/s    eta 7m 22s 

    
db_v1.tar.gz         56%[==========>         ]  45.71M  76.7KB/s    eta 7m 21s 

    
b_v1.tar.gz          56%[==========>         ]  45.73M  73.0KB/s    eta 7m 21s 

    
_v1.tar.gz           57%[==========>         ]  45.74M  67.7KB/s    eta 7m 21s 

    
v1.tar.gz            57%[==========>         ]  45.76M  67.1KB/s    eta 7m 21s 

    
1.tar.gz             57%[==========>         ]  45.77M  64.4KB/s    eta 7m 21s 

    
.tar.gz              57%[==========>         ]  45.79M  63.5KB/s    eta 7m 21s 

    
tar.gz               57%[==========>         ]  45.82M  63.1KB/s    eta 7m 21s 

    
ar.gz                57%[==========>         ]  45.84M  68.1KB/s    eta 7m 21s 

    
r.gz                 57%[==========>         ]  45.85M  61.6KB/s    eta 7m 20s 

    
.gz                  57%[==========>         ]  45.89M  69.9KB/s    eta 7m 20s 

    
gz                   57%[==========>         ]  45.91M  67.8KB/s    eta 7m 20s 

    
z                    57%[==========>         ]  45.93M  69.9KB/s    eta 7m 20s 

    
<div class="k-default-codeblock">
```
                 57%[==========>         ]  45.96M  71.7KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  57%[==========>         ]  45.98M  73.3KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  57%[==========>         ]  45.99M  69.2KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  57%[==========>         ]  46.02M  72.6KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  57%[==========>         ]  46.05M  75.8KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  57%[==========>         ]  46.07M  76.0KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  57%[==========>         ]  46.08M  75.9KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  57%[==========>         ]  46.10M  76.2KB/s    eta 7m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  57%[==========>         ]  46.11M  75.9KB/s    eta 7m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  57%[==========>         ]  46.12M  76.2KB/s    eta 7m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  57%[==========>         ]  46.13M  76.1KB/s    eta 7m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  57%[==========>         ]  46.15M  75.9KB/s    eta 7m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  57%[==========>         ]  46.16M  74.7KB/s    eta 7m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  57%[==========>         ]  46.18M  73.2KB/s    eta 7m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  57%[==========>         ]  46.19M  71.9KB/s    eta 7m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  57%[==========>         ]  46.20M  70.3KB/s    eta 7m 16s 

```
</div>
    
   datasets/aclImdb  57%[==========>         ]  46.22M  72.9KB/s    eta 7m 15s 

    
  datasets/aclImdb_  57%[==========>         ]  46.24M  68.7KB/s    eta 7m 15s 

    
 datasets/aclImdb_v  57%[==========>         ]  46.26M  68.7KB/s    eta 7m 15s 

    
datasets/aclImdb_v1  57%[==========>         ]  46.28M  68.9KB/s    eta 7m 15s 

    
atasets/aclImdb_v1.  57%[==========>         ]  46.30M  68.2KB/s    eta 7m 15s 

    
tasets/aclImdb_v1.t  57%[==========>         ]  46.31M  65.7KB/s    eta 7m 14s 

    
asets/aclImdb_v1.ta  57%[==========>         ]  46.33M  71.1KB/s    eta 7m 14s 

    
sets/aclImdb_v1.tar  57%[==========>         ]  46.36M  69.4KB/s    eta 7m 14s 

    
ets/aclImdb_v1.tar.  57%[==========>         ]  46.39M  73.4KB/s    eta 7m 14s 

    
ts/aclImdb_v1.tar.g  57%[==========>         ]  46.43M  75.0KB/s    eta 7m 12s 

    
s/aclImdb_v1.tar.gz  57%[==========>         ]  46.48M  80.8KB/s    eta 7m 12s 

    
/aclImdb_v1.tar.gz   57%[==========>         ]  46.53M  87.5KB/s    eta 7m 12s 

    
aclImdb_v1.tar.gz    58%[==========>         ]  46.58M  95.0KB/s    eta 7m 12s 

    
clImdb_v1.tar.gz     58%[==========>         ]  46.62M  98.0KB/s    eta 7m 9s  

    
lImdb_v1.tar.gz      58%[==========>         ]  46.68M   107KB/s    eta 7m 9s  

    
Imdb_v1.tar.gz       58%[==========>         ]  46.75M   119KB/s    eta 7m 9s  

    
mdb_v1.tar.gz        58%[==========>         ]  46.80M   126KB/s    eta 7m 9s  

    
db_v1.tar.gz         58%[==========>         ]  46.85M   135KB/s    eta 7m 9s  

    
b_v1.tar.gz          58%[==========>         ]  46.89M   141KB/s    eta 7m 4s  

    
_v1.tar.gz           58%[==========>         ]  46.93M   139KB/s    eta 7m 4s  

    
v1.tar.gz            58%[==========>         ]  47.01M   148KB/s    eta 7m 4s  

    
1.tar.gz             58%[==========>         ]  47.07M   156KB/s    eta 7m 1s  

    
.tar.gz              58%[==========>         ]  47.08M   154KB/s    eta 7m 1s  

    
tar.gz               58%[==========>         ]  47.12M   159KB/s    eta 7m 1s  

    
ar.gz                58%[==========>         ]  47.14M   160KB/s    eta 7m 1s  

    
r.gz                 58%[==========>         ]  47.17M   166KB/s    eta 7m 1s  

    
.gz                  58%[==========>         ]  47.19M   166KB/s    eta 6m 59s 

    
gz                   58%[==========>         ]  47.22M   165KB/s    eta 6m 59s 

    
z                    58%[==========>         ]  47.24M   161KB/s    eta 6m 59s 

    
<div class="k-default-codeblock">
```
                 58%[==========>         ]  47.27M   157KB/s    eta 6m 59s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  58%[==========>         ]  47.29M   152KB/s    eta 6m 59s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  58%[==========>         ]  47.32M   154KB/s    eta 6m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  59%[==========>         ]  47.34M   139KB/s    eta 6m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  59%[==========>         ]  47.37M   131KB/s    eta 6m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  59%[==========>         ]  47.39M   125KB/s    eta 6m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  59%[==========>         ]  47.41M   126KB/s    eta 6m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  59%[==========>         ]  47.43M   105KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  59%[==========>         ]  47.46M   107KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  59%[==========>         ]  47.49M   104KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  59%[==========>         ]  47.51M   104KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  59%[==========>         ]  47.54M   105KB/s    eta 6m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  59%[==========>         ]  47.55M  97.9KB/s    eta 6m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  59%[==========>         ]  47.60M   105KB/s    eta 6m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  59%[==========>         ]  47.63M   106KB/s    eta 6m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  59%[==========>         ]  47.65M   106KB/s    eta 6m 54s 

```
</div>
    
   datasets/aclImdb  59%[==========>         ]  47.68M   109KB/s    eta 6m 54s 

    
  datasets/aclImdb_  59%[==========>         ]  47.71M   108KB/s    eta 6m 52s 

    
 datasets/aclImdb_v  59%[==========>         ]  47.73M   101KB/s    eta 6m 52s 

    
datasets/aclImdb_v1  59%[==========>         ]  47.77M   101KB/s    eta 6m 52s 

    
atasets/aclImdb_v1.  59%[==========>         ]  47.81M   104KB/s    eta 6m 50s 

    
tasets/aclImdb_v1.t  59%[==========>         ]  47.82M   101KB/s    eta 6m 50s 

    
asets/aclImdb_v1.ta  59%[==========>         ]  47.83M  99.4KB/s    eta 6m 50s 

    
sets/aclImdb_v1.tar  59%[==========>         ]  47.85M  96.6KB/s    eta 6m 50s 

    
ets/aclImdb_v1.tar.  59%[==========>         ]  47.87M  94.5KB/s    eta 6m 50s 

    
ts/aclImdb_v1.tar.g  59%[==========>         ]  47.89M  89.3KB/s    eta 6m 50s 

    
s/aclImdb_v1.tar.gz  59%[==========>         ]  47.92M  85.2KB/s    eta 6m 50s 

    
/aclImdb_v1.tar.gz   59%[==========>         ]  47.94M  83.4KB/s    eta 6m 49s 

    
aclImdb_v1.tar.gz    59%[==========>         ]  47.95M  81.7KB/s    eta 6m 49s 

    
clImdb_v1.tar.gz     59%[==========>         ]  47.96M  83.6KB/s    eta 6m 49s 

    
lImdb_v1.tar.gz      59%[==========>         ]  47.97M  75.5KB/s    eta 6m 49s 

    
Imdb_v1.tar.gz       59%[==========>         ]  47.98M  72.4KB/s    eta 6m 49s 

    
mdb_v1.tar.gz        59%[==========>         ]  47.99M  70.3KB/s    eta 6m 49s 

    
db_v1.tar.gz         59%[==========>         ]  48.00M  68.2KB/s    eta 6m 49s 

    
b_v1.tar.gz          59%[==========>         ]  48.02M  65.2KB/s    eta 6m 49s 

    
_v1.tar.gz           59%[==========>         ]  48.03M  64.7KB/s    eta 6m 49s 

    
v1.tar.gz            59%[==========>         ]  48.05M  62.8KB/s    eta 6m 49s 

    
1.tar.gz             59%[==========>         ]  48.06M  61.7KB/s    eta 6m 48s 

    
.tar.gz              59%[==========>         ]  48.08M  64.9KB/s    eta 6m 48s 

    
tar.gz               59%[==========>         ]  48.11M  64.3KB/s    eta 6m 48s 

    
ar.gz                59%[==========>         ]  48.13M  61.6KB/s    eta 6m 48s 

    
r.gz                 60%[===========>        ]  48.16M  64.8KB/s    eta 6m 48s 

    
.gz                  60%[===========>        ]  48.19M  68.3KB/s    eta 6m 46s 

    
gz                   60%[===========>        ]  48.20M  66.3KB/s    eta 6m 46s 

    
z                    60%[===========>        ]  48.26M  74.2KB/s    eta 6m 46s 

    
<div class="k-default-codeblock">
```
                 60%[===========>        ]  48.29M  80.1KB/s    eta 6m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  60%[===========>        ]  48.32M  84.0KB/s    eta 6m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  60%[===========>        ]  48.34M  88.1KB/s    eta 6m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  60%[===========>        ]  48.38M  92.8KB/s    eta 6m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  60%[===========>        ]  48.38M  90.1KB/s    eta 6m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  60%[===========>        ]  48.43M   100KB/s    eta 6m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  60%[===========>        ]  48.46M   104KB/s    eta 6m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  60%[===========>        ]  48.48M   107KB/s    eta 6m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  60%[===========>        ]  48.51M   113KB/s    eta 6m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  60%[===========>        ]  48.54M   116KB/s    eta 6m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  60%[===========>        ]  48.57M   121KB/s    eta 6m 40s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  60%[===========>        ]  48.60M   122KB/s    eta 6m 40s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  60%[===========>        ]  48.63M   122KB/s    eta 6m 40s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  60%[===========>        ]  48.66M   119KB/s    eta 6m 40s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  60%[===========>        ]  48.69M   120KB/s    eta 6m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  60%[===========>        ]  48.72M   119KB/s    eta 6m 38s 

```
</div>
    
   datasets/aclImdb  60%[===========>        ]  48.75M   120KB/s    eta 6m 38s 

    
  datasets/aclImdb_  60%[===========>        ]  48.79M   119KB/s    eta 6m 38s 

    
 datasets/aclImdb_v  60%[===========>        ]  48.82M   125KB/s    eta 6m 36s 

    
datasets/aclImdb_v1  60%[===========>        ]  48.82M   106KB/s    eta 6m 36s 

    
atasets/aclImdb_v1.  60%[===========>        ]  48.87M   112KB/s    eta 6m 36s 

    
tasets/aclImdb_v1.t  60%[===========>        ]  48.89M   113KB/s    eta 6m 36s 

    
asets/aclImdb_v1.ta  60%[===========>        ]  48.92M   113KB/s    eta 6m 35s 

    
sets/aclImdb_v1.tar  61%[===========>        ]  48.95M   109KB/s    eta 6m 35s 

    
ets/aclImdb_v1.tar.  61%[===========>        ]  48.98M   111KB/s    eta 6m 35s 

    
ts/aclImdb_v1.tar.g  61%[===========>        ]  49.01M   109KB/s    eta 6m 35s 

    
s/aclImdb_v1.tar.gz  61%[===========>        ]  49.04M   112KB/s    eta 6m 33s 

    
/aclImdb_v1.tar.gz   61%[===========>        ]  49.07M   112KB/s    eta 6m 33s 

    
aclImdb_v1.tar.gz    61%[===========>        ]  49.08M   107KB/s    eta 6m 33s 

    
clImdb_v1.tar.gz     61%[===========>        ]  49.12M   107KB/s    eta 6m 33s 

    
lImdb_v1.tar.gz      61%[===========>        ]  49.15M   107KB/s    eta 6m 32s 

    
Imdb_v1.tar.gz       61%[===========>        ]  49.16M   105KB/s    eta 6m 32s 

    
mdb_v1.tar.gz        61%[===========>        ]  49.18M   103KB/s    eta 6m 32s 

    
db_v1.tar.gz         61%[===========>        ]  49.20M  98.4KB/s    eta 6m 32s 

    
b_v1.tar.gz          61%[===========>        ]  49.22M  99.5KB/s    eta 6m 32s 

    
_v1.tar.gz           61%[===========>        ]  49.24M  97.0KB/s    eta 6m 31s 

    
v1.tar.gz            61%[===========>        ]  49.26M  96.3KB/s    eta 6m 31s 

    
1.tar.gz             61%[===========>        ]  49.28M  93.7KB/s    eta 6m 31s 

    
.tar.gz              61%[===========>        ]  49.30M  92.0KB/s    eta 6m 31s 

    
tar.gz               61%[===========>        ]  49.32M  89.6KB/s    eta 6m 30s 

    
ar.gz                61%[===========>        ]  49.33M  86.7KB/s    eta 6m 30s 

    
r.gz                 61%[===========>        ]  49.35M  84.5KB/s    eta 6m 30s 

    
.gz                  61%[===========>        ]  49.37M  83.6KB/s    eta 6m 30s 

    
gz                   61%[===========>        ]  49.39M  80.1KB/s    eta 6m 30s 

    
z                    61%[===========>        ]  49.39M  77.3KB/s    eta 6m 29s 

    
<div class="k-default-codeblock">
```
                 61%[===========>        ]  49.41M  75.3KB/s    eta 6m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  61%[===========>        ]  49.43M  72.6KB/s    eta 6m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  61%[===========>        ]  49.45M  72.7KB/s    eta 6m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  61%[===========>        ]  49.47M  73.3KB/s    eta 6m 29s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  61%[===========>        ]  49.49M  73.1KB/s    eta 6m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  61%[===========>        ]  49.51M  73.6KB/s    eta 6m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  61%[===========>        ]  49.52M  69.4KB/s    eta 6m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  61%[===========>        ]  49.54M  68.2KB/s    eta 6m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  61%[===========>        ]  49.56M  68.8KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  61%[===========>        ]  49.56M  65.1KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  61%[===========>        ]  49.59M  63.9KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  61%[===========>        ]  49.61M  64.6KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  61%[===========>        ]  49.61M  63.1KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  61%[===========>        ]  49.63M  61.7KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  61%[===========>        ]  49.64M  57.2KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  61%[===========>        ]  49.65M  56.5KB/s    eta 6m 26s 

```
</div>
    
   datasets/aclImdb  61%[===========>        ]  49.66M  55.0KB/s    eta 6m 26s 

    
  datasets/aclImdb_  61%[===========>        ]  49.67M  56.7KB/s    eta 6m 26s 

    
 datasets/aclImdb_v  61%[===========>        ]  49.68M  54.0KB/s    eta 6m 26s 

    
datasets/aclImdb_v1  61%[===========>        ]  49.68M  50.9KB/s    eta 6m 26s 

    
atasets/aclImdb_v1.  61%[===========>        ]  49.70M  50.3KB/s    eta 6m 26s 

    
tasets/aclImdb_v1.t  61%[===========>        ]  49.71M  49.3KB/s    eta 6m 26s 

    
asets/aclImdb_v1.ta  61%[===========>        ]  49.72M  48.6KB/s    eta 6m 26s 

    
sets/aclImdb_v1.tar  61%[===========>        ]  49.73M  47.3KB/s    eta 6m 26s 

    
ets/aclImdb_v1.tar.  61%[===========>        ]  49.74M  44.4KB/s    eta 6m 26s 

    
ts/aclImdb_v1.tar.g  62%[===========>        ]  49.75M  43.7KB/s    eta 6m 26s 

    
s/aclImdb_v1.tar.gz  62%[===========>        ]  49.76M  42.7KB/s    eta 6m 26s 

    
/aclImdb_v1.tar.gz   62%[===========>        ]  49.77M  40.3KB/s    eta 6m 26s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  49.78M  39.1KB/s    eta 6m 26s 

    
clImdb_v1.tar.gz     62%[===========>        ]  49.78M  39.3KB/s    eta 6m 26s 

    
lImdb_v1.tar.gz      62%[===========>        ]  49.79M  36.8KB/s    eta 6m 26s 

    
Imdb_v1.tar.gz       62%[===========>        ]  49.80M  34.9KB/s    eta 6m 26s 

    
mdb_v1.tar.gz        62%[===========>        ]  49.80M  34.5KB/s    eta 6m 26s 

    
db_v1.tar.gz         62%[===========>        ]  49.81M  32.8KB/s    eta 6m 26s 

    
b_v1.tar.gz          62%[===========>        ]  49.81M  33.3KB/s    eta 6m 26s 

    
_v1.tar.gz           62%[===========>        ]  49.82M  32.1KB/s    eta 6m 26s 

    
v1.tar.gz            62%[===========>        ]  49.82M  29.7KB/s    eta 6m 26s 

    
1.tar.gz             62%[===========>        ]  49.83M  28.3KB/s    eta 6m 26s 

    
.tar.gz              62%[===========>        ]  49.83M  27.6KB/s    eta 6m 26s 

    
tar.gz               62%[===========>        ]  49.83M  27.2KB/s    eta 6m 26s 

    
ar.gz                62%[===========>        ]  49.83M  24.7KB/s    eta 6m 26s 

    
r.gz                 62%[===========>        ]  49.84M  23.8KB/s    eta 6m 27s 

    
.gz                  62%[===========>        ]  49.85M  22.5KB/s    eta 6m 27s 

    
gz                   62%[===========>        ]  49.85M  21.3KB/s    eta 6m 27s 

    
z                    62%[===========>        ]  49.85M  21.2KB/s    eta 6m 27s 

    
<div class="k-default-codeblock">
```
                 62%[===========>        ]  49.86M  19.9KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  62%[===========>        ]  49.87M  19.8KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  62%[===========>        ]  49.87M  19.6KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  62%[===========>        ]  49.88M  19.2KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  62%[===========>        ]  49.88M  19.8KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  62%[===========>        ]  49.89M  19.8KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  62%[===========>        ]  49.89M  19.3KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  62%[===========>        ]  49.90M  19.4KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  62%[===========>        ]  49.90M  18.8KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  62%[===========>        ]  49.91M  19.3KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  62%[===========>        ]  49.91M  19.0KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  62%[===========>        ]  49.92M  20.2KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  62%[===========>        ]  49.93M  20.9KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  62%[===========>        ]  49.93M  21.5KB/s    eta 6m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  62%[===========>        ]  49.94M  22.1KB/s    eta 6m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  62%[===========>        ]  49.95M  22.8KB/s    eta 6m 28s 

```
</div>
    
   datasets/aclImdb  62%[===========>        ]  49.96M  23.9KB/s    eta 6m 28s 

    
  datasets/aclImdb_  62%[===========>        ]  49.97M  24.7KB/s    eta 6m 28s 

    
 datasets/aclImdb_v  62%[===========>        ]  49.98M  25.4KB/s    eta 6m 27s 

    
datasets/aclImdb_v1  62%[===========>        ]  49.98M  26.0KB/s    eta 6m 27s 

    
atasets/aclImdb_v1.  62%[===========>        ]  49.99M  27.2KB/s    eta 6m 27s 

    
tasets/aclImdb_v1.t  62%[===========>        ]  49.99M  25.6KB/s    eta 6m 27s 

    
asets/aclImdb_v1.ta  62%[===========>        ]  50.01M  27.5KB/s    eta 6m 27s 

    
sets/aclImdb_v1.tar  62%[===========>        ]  50.01M  27.4KB/s    eta 6m 27s 

    
ets/aclImdb_v1.tar.  62%[===========>        ]  50.02M  27.8KB/s    eta 6m 27s 

    
ts/aclImdb_v1.tar.g  62%[===========>        ]  50.03M  27.8KB/s    eta 6m 27s 

    
s/aclImdb_v1.tar.gz  62%[===========>        ]  50.04M  29.0KB/s    eta 6m 27s 

    
/aclImdb_v1.tar.gz   62%[===========>        ]  50.05M  30.1KB/s    eta 6m 27s 

    
aclImdb_v1.tar.gz    62%[===========>        ]  50.07M  32.7KB/s    eta 6m 27s 

    
clImdb_v1.tar.gz     62%[===========>        ]  50.09M  34.6KB/s    eta 6m 27s 

    
lImdb_v1.tar.gz      62%[===========>        ]  50.10M  35.2KB/s    eta 6m 27s 

    
Imdb_v1.tar.gz       62%[===========>        ]  50.14M  39.9KB/s    eta 6m 27s 

    
mdb_v1.tar.gz        62%[===========>        ]  50.16M  43.7KB/s    eta 6m 27s 

    
db_v1.tar.gz         62%[===========>        ]  50.18M  46.1KB/s    eta 6m 27s 

    
b_v1.tar.gz          62%[===========>        ]  50.20M  48.5KB/s    eta 6m 27s 

    
_v1.tar.gz           62%[===========>        ]  50.22M  51.0KB/s    eta 6m 25s 

    
v1.tar.gz            62%[===========>        ]  50.24M  52.8KB/s    eta 6m 25s 

    
1.tar.gz             62%[===========>        ]  50.26M  55.1KB/s    eta 6m 25s 

    
.tar.gz              62%[===========>        ]  50.29M  57.7KB/s    eta 6m 25s 

    
tar.gz               62%[===========>        ]  50.31M  60.4KB/s    eta 6m 24s 

    
ar.gz                62%[===========>        ]  50.33M  63.1KB/s    eta 6m 24s 

    
r.gz                 62%[===========>        ]  50.35M  66.5KB/s    eta 6m 24s 

    
.gz                  62%[===========>        ]  50.37M  67.1KB/s    eta 6m 24s 

    
gz                   62%[===========>        ]  50.40M  70.2KB/s    eta 6m 23s 

    
z                    62%[===========>        ]  50.42M  73.3KB/s    eta 6m 23s 

    
<div class="k-default-codeblock">
```
                 62%[===========>        ]  50.44M  76.6KB/s    eta 6m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  62%[===========>        ]  50.46M  78.5KB/s    eta 6m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  62%[===========>        ]  50.49M  81.9KB/s    eta 6m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  62%[===========>        ]  50.52M  84.0KB/s    eta 6m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  62%[===========>        ]  50.52M  82.6KB/s    eta 6m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  63%[===========>        ]  50.56M  91.8KB/s    eta 6m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  63%[===========>        ]  50.57M  87.1KB/s    eta 6m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  63%[===========>        ]  50.58M  83.9KB/s    eta 6m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  63%[===========>        ]  50.61M  86.4KB/s    eta 6m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  63%[===========>        ]  50.62M  84.1KB/s    eta 6m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  63%[===========>        ]  50.63M  82.8KB/s    eta 6m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  63%[===========>        ]  50.65M  82.4KB/s    eta 6m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  63%[===========>        ]  50.66M  80.8KB/s    eta 6m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  63%[===========>        ]  50.67M  79.6KB/s    eta 6m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  63%[===========>        ]  50.69M  78.4KB/s    eta 6m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  63%[===========>        ]  50.70M  72.6KB/s    eta 6m 19s 

```
</div>
    
   datasets/aclImdb  63%[===========>        ]  50.73M  71.3KB/s    eta 6m 19s 

    
  datasets/aclImdb_  63%[===========>        ]  50.74M  71.9KB/s    eta 6m 19s 

    
 datasets/aclImdb_v  63%[===========>        ]  50.75M  69.0KB/s    eta 6m 19s 

    
datasets/aclImdb_v1  63%[===========>        ]  50.76M  66.3KB/s    eta 6m 19s 

    
atasets/aclImdb_v1.  63%[===========>        ]  50.77M  63.7KB/s    eta 6m 19s 

    
tasets/aclImdb_v1.t  63%[===========>        ]  50.78M  61.5KB/s    eta 6m 19s 

    
asets/aclImdb_v1.ta  63%[===========>        ]  50.79M  57.9KB/s    eta 6m 19s 

    
sets/aclImdb_v1.tar  63%[===========>        ]  50.80M  54.9KB/s    eta 6m 19s 

    
ets/aclImdb_v1.tar.  63%[===========>        ]  50.81M  55.4KB/s    eta 6m 19s 

    
ts/aclImdb_v1.tar.g  63%[===========>        ]  50.82M  50.1KB/s    eta 6m 19s 

    
s/aclImdb_v1.tar.gz  63%[===========>        ]  50.83M  50.9KB/s    eta 6m 18s 

    
/aclImdb_v1.tar.gz   63%[===========>        ]  50.85M  52.0KB/s    eta 6m 18s 

    
aclImdb_v1.tar.gz    63%[===========>        ]  50.86M  48.9KB/s    eta 6m 18s 

    
clImdb_v1.tar.gz     63%[===========>        ]  50.88M  51.0KB/s    eta 6m 18s 

    
lImdb_v1.tar.gz      63%[===========>        ]  50.90M  53.1KB/s    eta 6m 18s 

    
Imdb_v1.tar.gz       63%[===========>        ]  50.93M  53.7KB/s    eta 6m 17s 

    
mdb_v1.tar.gz        63%[===========>        ]  50.95M  56.1KB/s    eta 6m 17s 

    
db_v1.tar.gz         63%[===========>        ]  50.98M  58.3KB/s    eta 6m 17s 

    
b_v1.tar.gz          63%[===========>        ]  51.01M  61.3KB/s    eta 6m 17s 

    
_v1.tar.gz           63%[===========>        ]  51.04M  68.0KB/s    eta 6m 15s 

    
v1.tar.gz            63%[===========>        ]  51.07M  69.4KB/s    eta 6m 15s 

    
1.tar.gz             63%[===========>        ]  51.11M  73.6KB/s    eta 6m 15s 

    
.tar.gz              63%[===========>        ]  51.15M  80.3KB/s    eta 6m 15s 

    
tar.gz               63%[===========>        ]  51.17M  82.8KB/s    eta 6m 13s 

    
ar.gz                63%[===========>        ]  51.19M  85.7KB/s    eta 6m 13s 

    
r.gz                 63%[===========>        ]  51.21M  88.1KB/s    eta 6m 13s 

    
.gz                  63%[===========>        ]  51.23M  89.8KB/s    eta 6m 13s 

    
gz                   63%[===========>        ]  51.26M  92.7KB/s    eta 6m 13s 

    
z                    63%[===========>        ]  51.28M  95.2KB/s    eta 6m 12s 

    
<div class="k-default-codeblock">
```
                 63%[===========>        ]  51.30M  96.8KB/s    eta 6m 12s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  63%[===========>        ]  51.32M  98.2KB/s    eta 6m 12s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  64%[===========>        ]  51.35M  99.4KB/s    eta 6m 12s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  64%[===========>        ]  51.37M  99.9KB/s    eta 6m 11s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  64%[===========>        ]  51.39M  99.5KB/s    eta 6m 11s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  64%[===========>        ]  51.41M  99.1KB/s    eta 6m 11s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  64%[===========>        ]  51.43M  98.6KB/s    eta 6m 11s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  64%[===========>        ]  51.46M  98.4KB/s    eta 6m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  64%[===========>        ]  51.48M  97.3KB/s    eta 6m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  64%[===========>        ]  51.51M  96.2KB/s    eta 6m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  64%[===========>        ]  51.54M  97.7KB/s    eta 6m 9s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  64%[===========>        ]  51.55M  93.3KB/s    eta 6m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  64%[===========>        ]  51.60M  98.4KB/s    eta 6m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  64%[===========>        ]  51.61M  94.2KB/s    eta 6m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  64%[===========>        ]  51.64M  95.0KB/s    eta 6m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  64%[===========>        ]  51.66M  95.4KB/s    eta 6m 8s  

```
</div>
    
   datasets/aclImdb  64%[===========>        ]  51.69M  96.3KB/s    eta 6m 6s  

    
  datasets/aclImdb_  64%[===========>        ]  51.72M  97.8KB/s    eta 6m 6s  

    
 datasets/aclImdb_v  64%[===========>        ]  51.75M  98.8KB/s    eta 6m 6s  

    
datasets/aclImdb_v1  64%[===========>        ]  51.79M   100KB/s    eta 6m 6s  

    
atasets/aclImdb_v1.  64%[===========>        ]  51.82M   103KB/s    eta 6m 4s  

    
tasets/aclImdb_v1.t  64%[===========>        ]  51.85M   105KB/s    eta 6m 4s  

    
asets/aclImdb_v1.ta  64%[===========>        ]  51.89M   107KB/s    eta 6m 4s  

    
sets/aclImdb_v1.tar  64%[===========>        ]  51.92M   109KB/s    eta 6m 4s  

    
ets/aclImdb_v1.tar.  64%[===========>        ]  51.95M   111KB/s    eta 6m 2s  

    
ts/aclImdb_v1.tar.g  64%[===========>        ]  51.99M   113KB/s    eta 6m 2s  

    
s/aclImdb_v1.tar.gz  64%[===========>        ]  51.99M   110KB/s    eta 6m 2s  

    
/aclImdb_v1.tar.gz   64%[===========>        ]  52.04M   115KB/s    eta 6m 2s  

    
aclImdb_v1.tar.gz    64%[===========>        ]  52.06M   115KB/s    eta 6m 0s  

    
clImdb_v1.tar.gz     64%[===========>        ]  52.09M   116KB/s    eta 6m 0s  

    
lImdb_v1.tar.gz      64%[===========>        ]  52.12M   121KB/s    eta 6m 0s  

    
Imdb_v1.tar.gz       65%[============>       ]  52.15M   117KB/s    eta 6m 0s  

    
mdb_v1.tar.gz        65%[============>       ]  52.19M   119KB/s    eta 5m 59s 

    
db_v1.tar.gz         65%[============>       ]  52.22M   121KB/s    eta 5m 59s 

    
b_v1.tar.gz          65%[============>       ]  52.25M   122KB/s    eta 5m 59s 

    
_v1.tar.gz           65%[============>       ]  52.28M   123KB/s    eta 5m 59s 

    
v1.tar.gz            65%[============>       ]  52.32M   123KB/s    eta 5m 57s 

    
1.tar.gz             65%[============>       ]  52.35M   123KB/s    eta 5m 57s 

    
.tar.gz              65%[============>       ]  52.38M   121KB/s    eta 5m 57s 

    
tar.gz               65%[============>       ]  52.41M   113KB/s    eta 5m 55s 

    
ar.gz                65%[============>       ]  52.47M   119KB/s    eta 5m 55s 

    
r.gz                 65%[============>       ]  52.49M   118KB/s    eta 5m 55s 

    
.gz                  65%[============>       ]  52.52M   124KB/s    eta 5m 55s 

    
gz                   65%[============>       ]  52.54M   119KB/s    eta 5m 55s 

    
z                    65%[============>       ]  52.57M   119KB/s    eta 5m 53s 

    
<div class="k-default-codeblock">
```
                 65%[============>       ]  52.61M   119KB/s    eta 5m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  65%[============>       ]  52.61M   113KB/s    eta 5m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  65%[============>       ]  52.66M   119KB/s    eta 5m 53s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  65%[============>       ]  52.69M   117KB/s    eta 5m 51s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  65%[============>       ]  52.71M   113KB/s    eta 5m 51s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  65%[============>       ]  52.74M   113KB/s    eta 5m 51s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  65%[============>       ]  52.75M   106KB/s    eta 5m 51s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  65%[============>       ]  52.79M   109KB/s    eta 5m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  65%[============>       ]  52.81M   105KB/s    eta 5m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  65%[============>       ]  52.84M   106KB/s    eta 5m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  65%[============>       ]  52.87M   112KB/s    eta 5m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  65%[============>       ]  52.89M   104KB/s    eta 5m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  65%[============>       ]  52.91M   103KB/s    eta 5m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  65%[============>       ]  52.94M   101KB/s    eta 5m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  66%[============>       ]  52.96M   100KB/s    eta 5m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  66%[============>       ]  52.99M  98.9KB/s    eta 5m 47s 

```
</div>
    
   datasets/aclImdb  66%[============>       ]  53.01M  97.6KB/s    eta 5m 47s 

    
  datasets/aclImdb_  66%[============>       ]  53.04M   103KB/s    eta 5m 47s 

    
 datasets/aclImdb_v  66%[============>       ]  53.06M  96.8KB/s    eta 5m 47s 

    
datasets/aclImdb_v1  66%[============>       ]  53.09M  97.6KB/s    eta 5m 46s 

    
atasets/aclImdb_v1.  66%[============>       ]  53.12M  98.1KB/s    eta 5m 46s 

    
tasets/aclImdb_v1.t  66%[============>       ]  53.14M   104KB/s    eta 5m 46s 

    
asets/aclImdb_v1.ta  66%[============>       ]  53.17M  99.8KB/s    eta 5m 46s 

    
sets/aclImdb_v1.tar  66%[============>       ]  53.20M   101KB/s    eta 5m 44s 

    
ets/aclImdb_v1.tar.  66%[============>       ]  53.23M   103KB/s    eta 5m 44s 

    
ts/aclImdb_v1.tar.g  66%[============>       ]  53.25M   103KB/s    eta 5m 44s 

    
s/aclImdb_v1.tar.gz  66%[============>       ]  53.29M   108KB/s    eta 5m 44s 

    
/aclImdb_v1.tar.gz   66%[============>       ]  53.32M  95.7KB/s    eta 5m 42s 

    
aclImdb_v1.tar.gz    66%[============>       ]  53.37M   105KB/s    eta 5m 42s 

    
clImdb_v1.tar.gz     66%[============>       ]  53.43M   114KB/s    eta 5m 42s 

    
lImdb_v1.tar.gz      66%[============>       ]  53.45M   102KB/s    eta 5m 41s 

    
Imdb_v1.tar.gz       66%[============>       ]  53.47M   101KB/s    eta 5m 41s 

    
mdb_v1.tar.gz        66%[============>       ]  53.50M   101KB/s    eta 5m 41s 

    
db_v1.tar.gz         66%[============>       ]  53.52M   101KB/s    eta 5m 41s 

    
b_v1.tar.gz          66%[============>       ]  53.53M  98.5KB/s    eta 5m 39s 

    
_v1.tar.gz           66%[============>       ]  53.54M  94.0KB/s    eta 5m 39s 

    
v1.tar.gz            66%[============>       ]  53.56M  93.8KB/s    eta 5m 39s 

    
1.tar.gz             66%[============>       ]  53.57M  90.4KB/s    eta 5m 39s 

    
.tar.gz              66%[============>       ]  53.59M  88.8KB/s    eta 5m 39s 

    
tar.gz               66%[============>       ]  53.60M  87.9KB/s    eta 5m 39s 

    
ar.gz                66%[============>       ]  53.62M  84.7KB/s    eta 5m 39s 

    
r.gz                 66%[============>       ]  53.63M  82.6KB/s    eta 5m 39s 

    
.gz                  66%[============>       ]  53.64M  81.7KB/s    eta 5m 39s 

    
gz                   66%[============>       ]  53.66M  78.6KB/s    eta 5m 39s 

    
z                    66%[============>       ]  53.66M  73.7KB/s    eta 5m 38s 

    
<div class="k-default-codeblock">
```
                 66%[============>       ]  53.67M  69.8KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  66%[============>       ]  53.68M  71.9KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  66%[============>       ]  53.69M  63.7KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  66%[============>       ]  53.70M  54.1KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  66%[============>       ]  53.71M  53.7KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  66%[============>       ]  53.72M  56.3KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  66%[============>       ]  53.72M  53.2KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  66%[============>       ]  53.74M  49.6KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  66%[============>       ]  53.75M  48.6KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  67%[============>       ]  53.75M  47.2KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  67%[============>       ]  53.76M  45.0KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  67%[============>       ]  53.77M  43.9KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  67%[============>       ]  53.78M  43.0KB/s    eta 5m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  67%[============>       ]  53.79M  42.1KB/s    eta 5m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  67%[============>       ]  53.80M  41.7KB/s    eta 5m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  67%[============>       ]  53.82M  41.5KB/s    eta 5m 37s 

```
</div>
    
   datasets/aclImdb  67%[============>       ]  53.83M  41.3KB/s    eta 5m 37s 

    
  datasets/aclImdb_  67%[============>       ]  53.84M  41.4KB/s    eta 5m 37s 

    
 datasets/aclImdb_v  67%[============>       ]  53.87M  42.9KB/s    eta 5m 37s 

    
datasets/aclImdb_v1  67%[============>       ]  53.89M  43.9KB/s    eta 5m 37s 

    
atasets/aclImdb_v1.  67%[============>       ]  53.90M  45.9KB/s    eta 5m 37s 

    
tasets/aclImdb_v1.t  67%[============>       ]  53.92M  46.2KB/s    eta 5m 36s 

    
asets/aclImdb_v1.ta  67%[============>       ]  53.95M  50.5KB/s    eta 5m 36s 

    
sets/aclImdb_v1.tar  67%[============>       ]  53.97M  51.5KB/s    eta 5m 36s 

    
ets/aclImdb_v1.tar.  67%[============>       ]  53.99M  53.4KB/s    eta 5m 36s 

    
ts/aclImdb_v1.tar.g  67%[============>       ]  54.01M  55.4KB/s    eta 5m 36s 

    
s/aclImdb_v1.tar.gz  67%[============>       ]  54.03M  58.6KB/s    eta 5m 35s 

    
/aclImdb_v1.tar.gz   67%[============>       ]  54.05M  60.5KB/s    eta 5m 35s 

    
aclImdb_v1.tar.gz    67%[============>       ]  54.08M  64.0KB/s    eta 5m 35s 

    
clImdb_v1.tar.gz     67%[============>       ]  54.10M  67.5KB/s    eta 5m 35s 

    
lImdb_v1.tar.gz      67%[============>       ]  54.13M  71.8KB/s    eta 5m 35s 

    
Imdb_v1.tar.gz       67%[============>       ]  54.15M  75.3KB/s    eta 5m 33s 

    
mdb_v1.tar.gz        67%[============>       ]  54.18M  78.4KB/s    eta 5m 33s 

    
db_v1.tar.gz         67%[============>       ]  54.20M  81.6KB/s    eta 5m 33s 

    
b_v1.tar.gz          67%[============>       ]  54.23M  84.3KB/s    eta 5m 33s 

    
_v1.tar.gz           67%[============>       ]  54.25M  82.7KB/s    eta 5m 32s 

    
v1.tar.gz            67%[============>       ]  54.29M  86.4KB/s    eta 5m 32s 

    
1.tar.gz             67%[============>       ]  54.31M  87.6KB/s    eta 5m 32s 

    
.tar.gz              67%[============>       ]  54.33M  88.0KB/s    eta 5m 32s 

    
tar.gz               67%[============>       ]  54.35M  88.9KB/s    eta 5m 32s 

    
ar.gz                67%[============>       ]  54.38M  91.1KB/s    eta 5m 30s 

    
r.gz                 67%[============>       ]  54.40M  96.3KB/s    eta 5m 30s 

    
.gz                  67%[============>       ]  54.42M  94.4KB/s    eta 5m 30s 

    
gz                   67%[============>       ]  54.44M  95.9KB/s    eta 5m 30s 

    
z                    67%[============>       ]  54.47M  96.6KB/s    eta 5m 30s 

    
<div class="k-default-codeblock">
```
                 67%[============>       ]  54.49M  97.5KB/s    eta 5m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  67%[============>       ]  54.52M  97.7KB/s    eta 5m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  67%[============>       ]  54.54M  97.8KB/s    eta 5m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  68%[============>       ]  54.57M  96.8KB/s    eta 5m 28s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  68%[============>       ]  54.59M  96.2KB/s    eta 5m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  68%[============>       ]  54.62M  95.7KB/s    eta 5m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  68%[============>       ]  54.64M  95.1KB/s    eta 5m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  68%[============>       ]  54.67M  95.8KB/s    eta 5m 27s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  68%[============>       ]  54.70M  96.4KB/s    eta 5m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  68%[============>       ]  54.73M  97.1KB/s    eta 5m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  68%[============>       ]  54.76M  97.4KB/s    eta 5m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  68%[============>       ]  54.81M   102KB/s    eta 5m 24s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  68%[============>       ]  54.83M   102KB/s    eta 5m 24s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  68%[============>       ]  54.86M   103KB/s    eta 5m 24s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  68%[============>       ]  54.89M   103KB/s    eta 5m 24s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  68%[============>       ]  54.91M   102KB/s    eta 5m 22s 

```
</div>
    
   datasets/aclImdb  68%[============>       ]  54.94M   105KB/s    eta 5m 22s 

    
  datasets/aclImdb_  68%[============>       ]  54.97M   105KB/s    eta 5m 22s 

    
 datasets/aclImdb_v  68%[============>       ]  54.99M   105KB/s    eta 5m 22s 

    
datasets/aclImdb_v1  68%[============>       ]  55.01M   105KB/s    eta 5m 22s 

    
atasets/aclImdb_v1.  68%[============>       ]  55.03M   105KB/s    eta 5m 21s 

    
tasets/aclImdb_v1.t  68%[============>       ]  55.06M   105KB/s    eta 5m 21s 

    
asets/aclImdb_v1.ta  68%[============>       ]  55.08M   105KB/s    eta 5m 21s 

    
sets/aclImdb_v1.tar  68%[============>       ]  55.09M   101KB/s    eta 5m 21s 

    
ets/aclImdb_v1.tar.  68%[============>       ]  55.10M  99.5KB/s    eta 5m 20s 

    
ts/aclImdb_v1.tar.g  68%[============>       ]  55.11M  96.6KB/s    eta 5m 20s 

    
s/aclImdb_v1.tar.gz  68%[============>       ]  55.12M  92.8KB/s    eta 5m 20s 

    
/aclImdb_v1.tar.gz   68%[============>       ]  55.15M  92.9KB/s    eta 5m 20s 

    
aclImdb_v1.tar.gz    68%[============>       ]  55.16M  90.2KB/s    eta 5m 20s 

    
clImdb_v1.tar.gz     68%[============>       ]  55.17M  92.5KB/s    eta 5m 19s 

    
lImdb_v1.tar.gz      68%[============>       ]  55.19M  83.7KB/s    eta 5m 19s 

    
Imdb_v1.tar.gz       68%[============>       ]  55.20M  80.9KB/s    eta 5m 19s 

    
mdb_v1.tar.gz        68%[============>       ]  55.21M  77.7KB/s    eta 5m 19s 

    
db_v1.tar.gz         68%[============>       ]  55.23M  74.1KB/s    eta 5m 19s 

    
b_v1.tar.gz          68%[============>       ]  55.24M  75.5KB/s    eta 5m 18s 

    
_v1.tar.gz           68%[============>       ]  55.25M  73.8KB/s    eta 5m 18s 

    
v1.tar.gz            68%[============>       ]  55.27M  69.7KB/s    eta 5m 18s 

    
1.tar.gz             68%[============>       ]  55.28M  67.5KB/s    eta 5m 18s 

    
.tar.gz              68%[============>       ]  55.29M  65.8KB/s    eta 5m 18s 

    
tar.gz               68%[============>       ]  55.31M  64.1KB/s    eta 5m 18s 

    
ar.gz                68%[============>       ]  55.33M  62.4KB/s    eta 5m 18s 

    
r.gz                 68%[============>       ]  55.34M  60.8KB/s    eta 5m 18s 

    
.gz                  69%[============>       ]  55.36M  57.0KB/s    eta 5m 18s 

    
gz                   69%[============>       ]  55.39M  61.2KB/s    eta 5m 17s 

    
z                    69%[============>       ]  55.40M  60.8KB/s    eta 5m 17s 

    
<div class="k-default-codeblock">
```
                 69%[============>       ]  55.42M  62.8KB/s    eta 5m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  69%[============>       ]  55.43M  63.8KB/s    eta 5m 17s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  69%[============>       ]  55.46M  62.7KB/s    eta 5m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  69%[============>       ]  55.48M  63.7KB/s    eta 5m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  69%[============>       ]  55.49M  63.7KB/s    eta 5m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  69%[============>       ]  55.50M  63.1KB/s    eta 5m 16s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  69%[============>       ]  55.53M  64.6KB/s    eta 5m 15s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  69%[============>       ]  55.54M  64.7KB/s    eta 5m 15s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  69%[============>       ]  55.55M  64.7KB/s    eta 5m 15s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  69%[============>       ]  55.57M  64.7KB/s    eta 5m 15s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  69%[============>       ]  55.59M  65.1KB/s    eta 5m 14s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  69%[============>       ]  55.61M  66.0KB/s    eta 5m 14s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  69%[============>       ]  55.62M  65.8KB/s    eta 5m 14s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  69%[============>       ]  55.65M  67.5KB/s    eta 5m 14s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  69%[============>       ]  55.66M  66.4KB/s    eta 5m 14s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  69%[============>       ]  55.68M  68.0KB/s    eta 5m 14s 

```
</div>
    
   datasets/aclImdb  69%[============>       ]  55.70M  66.8KB/s    eta 5m 14s 

    
  datasets/aclImdb_  69%[============>       ]  55.72M  71.4KB/s    eta 5m 14s 

    
 datasets/aclImdb_v  69%[============>       ]  55.74M  68.3KB/s    eta 5m 13s 

    
datasets/aclImdb_v1  69%[============>       ]  55.76M  70.5KB/s    eta 5m 13s 

    
atasets/aclImdb_v1.  69%[============>       ]  55.78M  71.0KB/s    eta 5m 13s 

    
tasets/aclImdb_v1.t  69%[============>       ]  55.80M  71.9KB/s    eta 5m 13s 

    
asets/aclImdb_v1.ta  69%[============>       ]  55.82M  70.8KB/s    eta 5m 11s 

    
sets/aclImdb_v1.tar  69%[============>       ]  55.84M  70.9KB/s    eta 5m 11s 

    
ets/aclImdb_v1.tar.  69%[============>       ]  55.85M  69.1KB/s    eta 5m 11s 

    
ts/aclImdb_v1.tar.g  69%[============>       ]  55.87M  71.7KB/s    eta 5m 11s 

    
s/aclImdb_v1.tar.gz  69%[============>       ]  55.88M  69.4KB/s    eta 5m 11s 

    
/aclImdb_v1.tar.gz   69%[============>       ]  55.91M  72.6KB/s    eta 5m 11s 

    
aclImdb_v1.tar.gz    69%[============>       ]  55.92M  72.4KB/s    eta 5m 11s 

    
clImdb_v1.tar.gz     69%[============>       ]  55.93M  71.9KB/s    eta 5m 11s 

    
lImdb_v1.tar.gz      69%[============>       ]  55.95M  72.0KB/s    eta 5m 11s 

    
Imdb_v1.tar.gz       69%[============>       ]  55.96M  70.7KB/s    eta 5m 10s 

    
mdb_v1.tar.gz        69%[============>       ]  55.98M  71.4KB/s    eta 5m 10s 

    
db_v1.tar.gz         69%[============>       ]  55.99M  69.2KB/s    eta 5m 10s 

    
b_v1.tar.gz          69%[============>       ]  56.00M  70.2KB/s    eta 5m 10s 

    
_v1.tar.gz           69%[============>       ]  56.02M  68.0KB/s    eta 5m 10s 

    
v1.tar.gz            69%[============>       ]  56.03M  68.3KB/s    eta 5m 9s  

    
1.tar.gz             69%[============>       ]  56.04M  66.5KB/s    eta 5m 9s  

    
.tar.gz              69%[============>       ]  56.05M  66.2KB/s    eta 5m 9s  

    
tar.gz               69%[============>       ]  56.07M  65.4KB/s    eta 5m 9s  

    
ar.gz                69%[============>       ]  56.09M  64.7KB/s    eta 5m 8s  

    
r.gz                 69%[============>       ]  56.11M  63.8KB/s    eta 5m 8s  

    
.gz                  69%[============>       ]  56.13M  63.8KB/s    eta 5m 8s  

    
gz                   69%[============>       ]  56.15M  63.9KB/s    eta 5m 8s  

    
z                    69%[============>       ]  56.16M  64.2KB/s    eta 5m 8s  

    
<div class="k-default-codeblock">
```
                 70%[=============>      ]  56.19M  63.0KB/s    eta 5m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
              d  70%[=============>      ]  56.22M  66.0KB/s    eta 5m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
             da  70%[=============>      ]  56.23M  62.7KB/s    eta 5m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  70%[=============>      ]  56.24M  62.6KB/s    eta 5m 7s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  70%[=============>      ]  56.26M  62.9KB/s    eta 5m 7s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  70%[=============>      ]  56.27M  62.3KB/s    eta 5m 7s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  70%[=============>      ]  56.28M  61.2KB/s    eta 5m 7s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  70%[=============>      ]  56.29M  61.1KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  70%[=============>      ]  56.31M  58.6KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  70%[=============>      ]  56.33M  58.3KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  70%[=============>      ]  56.34M  58.2KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  70%[=============>      ]  56.35M  57.9KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  70%[=============>      ]  56.36M  57.7KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  70%[=============>      ]  56.38M  56.5KB/s    eta 5m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  70%[=============>      ]  56.39M  55.8KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  70%[=============>      ]  56.40M  51.6KB/s    eta 5m 5s  

```
</div>
    
   datasets/aclImdb  70%[=============>      ]  56.41M  51.2KB/s    eta 5m 5s  

    
  datasets/aclImdb_  70%[=============>      ]  56.42M  50.1KB/s    eta 5m 5s  

    
 datasets/aclImdb_v  70%[=============>      ]  56.43M  47.6KB/s    eta 5m 5s  

    
datasets/aclImdb_v1  70%[=============>      ]  56.43M  47.3KB/s    eta 5m 5s  

    
atasets/aclImdb_v1.  70%[=============>      ]  56.44M  44.4KB/s    eta 5m 5s  

    
tasets/aclImdb_v1.t  70%[=============>      ]  56.44M  41.2KB/s    eta 5m 5s  

    
asets/aclImdb_v1.ta  70%[=============>      ]  56.45M  40.0KB/s    eta 5m 5s  

    
sets/aclImdb_v1.tar  70%[=============>      ]  56.46M  38.8KB/s    eta 5m 5s  

    
ets/aclImdb_v1.tar.  70%[=============>      ]  56.46M  36.1KB/s    eta 5m 5s  

    
ts/aclImdb_v1.tar.g  70%[=============>      ]  56.47M  34.8KB/s    eta 5m 5s  

    
s/aclImdb_v1.tar.gz  70%[=============>      ]  56.47M  34.6KB/s    eta 5m 5s  

    
/aclImdb_v1.tar.gz   70%[=============>      ]  56.48M  32.2KB/s    eta 5m 5s  

    
aclImdb_v1.tar.gz    70%[=============>      ]  56.49M  32.6KB/s    eta 5m 5s  

    
clImdb_v1.tar.gz     70%[=============>      ]  56.49M  30.9KB/s    eta 5m 5s  

    
lImdb_v1.tar.gz      70%[=============>      ]  56.50M  29.6KB/s    eta 5m 5s  

    
Imdb_v1.tar.gz       70%[=============>      ]  56.50M  28.6KB/s    eta 5m 5s  

    
mdb_v1.tar.gz        70%[=============>      ]  56.51M  27.3KB/s    eta 5m 5s  

    
db_v1.tar.gz         70%[=============>      ]  56.51M  26.5KB/s    eta 5m 5s  

    
b_v1.tar.gz          70%[=============>      ]  56.52M  24.5KB/s    eta 5m 5s  

    
_v1.tar.gz           70%[=============>      ]  56.52M  25.7KB/s    eta 5m 5s  

    
v1.tar.gz            70%[=============>      ]  56.53M  24.2KB/s    eta 5m 5s  

    
1.tar.gz             70%[=============>      ]  56.53M  22.8KB/s    eta 5m 5s  

    
.tar.gz              70%[=============>      ]  56.54M  23.1KB/s    eta 5m 5s  

    
tar.gz               70%[=============>      ]  56.54M  22.8KB/s    eta 5m 5s  

    
ar.gz                70%[=============>      ]  56.55M  23.1KB/s    eta 5m 5s  

    
r.gz                 70%[=============>      ]  56.56M  23.7KB/s    eta 5m 5s  

    
.gz                  70%[=============>      ]  56.57M  24.2KB/s    eta 5m 5s  

    
gz                   70%[=============>      ]  56.58M  24.8KB/s    eta 5m 5s  

    
z                    70%[=============>      ]  56.58M  26.1KB/s    eta 5m 5s  

    
<div class="k-default-codeblock">
```
                 70%[=============>      ]  56.59M  26.0KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
              d  70%[=============>      ]  56.60M  26.6KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
             da  70%[=============>      ]  56.61M  27.7KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  70%[=============>      ]  56.62M  28.2KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  70%[=============>      ]  56.63M  29.2KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  70%[=============>      ]  56.64M  30.3KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  70%[=============>      ]  56.65M  31.5KB/s    eta 5m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  70%[=============>      ]  56.66M  33.1KB/s    eta 5m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  70%[=============>      ]  56.67M  34.2KB/s    eta 5m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  70%[=============>      ]  56.70M  37.3KB/s    eta 5m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  70%[=============>      ]  56.71M  39.4KB/s    eta 5m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  70%[=============>      ]  56.73M  42.6KB/s    eta 5m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  70%[=============>      ]  56.75M  46.4KB/s    eta 5m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  70%[=============>      ]  56.78M  49.8KB/s    eta 5m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  70%[=============>      ]  56.80M  54.1KB/s    eta 5m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  70%[=============>      ]  56.83M  57.8KB/s    eta 5m 3s  

```
</div>
    
   datasets/aclImdb  70%[=============>      ]  56.86M  63.7KB/s    eta 5m 2s  

    
  datasets/aclImdb_  70%[=============>      ]  56.90M  69.0KB/s    eta 5m 2s  

    
 datasets/aclImdb_v  70%[=============>      ]  56.94M  74.8KB/s    eta 5m 2s  

    
datasets/aclImdb_v1  71%[=============>      ]  56.97M  77.6KB/s    eta 5m 2s  

    
atasets/aclImdb_v1.  71%[=============>      ]  57.02M  86.9KB/s    eta 4m 59s 

    
tasets/aclImdb_v1.t  71%[=============>      ]  57.05M  92.0KB/s    eta 4m 59s 

    
asets/aclImdb_v1.ta  71%[=============>      ]  57.08M  94.8KB/s    eta 4m 59s 

    
sets/aclImdb_v1.tar  71%[=============>      ]  57.10M  97.7KB/s    eta 4m 59s 

    
ets/aclImdb_v1.tar.  71%[=============>      ]  57.14M   105KB/s    eta 4m 59s 

    
ts/aclImdb_v1.tar.g  71%[=============>      ]  57.17M   110KB/s    eta 4m 57s 

    
s/aclImdb_v1.tar.gz  71%[=============>      ]  57.21M   114KB/s    eta 4m 57s 

    
/aclImdb_v1.tar.gz   71%[=============>      ]  57.24M   118KB/s    eta 4m 57s 

    
aclImdb_v1.tar.gz    71%[=============>      ]  57.27M   122KB/s    eta 4m 57s 

    
clImdb_v1.tar.gz     71%[=============>      ]  57.31M   124KB/s    eta 4m 55s 

    
lImdb_v1.tar.gz      71%[=============>      ]  57.34M   128KB/s    eta 4m 55s 

    
Imdb_v1.tar.gz       71%[=============>      ]  57.37M   130KB/s    eta 4m 55s 

    
mdb_v1.tar.gz        71%[=============>      ]  57.40M   132KB/s    eta 4m 55s 

    
db_v1.tar.gz         71%[=============>      ]  57.43M   133KB/s    eta 4m 53s 

    
b_v1.tar.gz          71%[=============>      ]  57.47M   134KB/s    eta 4m 53s 

    
_v1.tar.gz           71%[=============>      ]  57.48M   130KB/s    eta 4m 53s 

    
v1.tar.gz            71%[=============>      ]  57.49M   123KB/s    eta 4m 53s 

    
1.tar.gz             71%[=============>      ]  57.55M   130KB/s    eta 4m 51s 

    
.tar.gz              71%[=============>      ]  57.57M   127KB/s    eta 4m 51s 

    
tar.gz               71%[=============>      ]  57.60M   129KB/s    eta 4m 51s 

    
ar.gz                71%[=============>      ]  57.62M   124KB/s    eta 4m 51s 

    
r.gz                 71%[=============>      ]  57.66M   124KB/s    eta 4m 51s 

    
.gz                  71%[=============>      ]  57.69M   126KB/s    eta 4m 49s 

    
gz                   71%[=============>      ]  57.72M   128KB/s    eta 4m 49s 

    
z                    71%[=============>      ]  57.75M   125KB/s    eta 4m 49s 

    
<div class="k-default-codeblock">
```
                 71%[=============>      ]  57.76M   122KB/s    eta 4m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  72%[=============>      ]  57.80M   122KB/s    eta 4m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  72%[=============>      ]  57.83M   123KB/s    eta 4m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  72%[=============>      ]  57.87M   122KB/s    eta 4m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  72%[=============>      ]  57.90M   122KB/s    eta 4m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  72%[=============>      ]  57.93M   122KB/s    eta 4m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  72%[=============>      ]  57.96M   122KB/s    eta 4m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  72%[=============>      ]  58.00M   121KB/s    eta 4m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  72%[=============>      ]  58.03M   122KB/s    eta 4m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  72%[=============>      ]  58.04M   116KB/s    eta 4m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  72%[=============>      ]  58.07M   121KB/s    eta 4m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  72%[=============>      ]  58.10M   125KB/s    eta 4m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  72%[=============>      ]  58.13M   116KB/s    eta 4m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  72%[=============>      ]  58.15M   116KB/s    eta 4m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  72%[=============>      ]  58.16M   113KB/s    eta 4m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  72%[=============>      ]  58.18M   110KB/s    eta 4m 43s 

```
</div>
    
   datasets/aclImdb  72%[=============>      ]  58.19M   106KB/s    eta 4m 43s 

    
  datasets/aclImdb_  72%[=============>      ]  58.20M   102KB/s    eta 4m 43s 

    
 datasets/aclImdb_v  72%[=============>      ]  58.22M  98.2KB/s    eta 4m 43s 

    
datasets/aclImdb_v1  72%[=============>      ]  58.23M  95.5KB/s    eta 4m 42s 

    
atasets/aclImdb_v1.  72%[=============>      ]  58.25M  90.9KB/s    eta 4m 42s 

    
tasets/aclImdb_v1.t  72%[=============>      ]  58.26M  87.1KB/s    eta 4m 42s 

    
asets/aclImdb_v1.ta  72%[=============>      ]  58.27M  83.5KB/s    eta 4m 42s 

    
sets/aclImdb_v1.tar  72%[=============>      ]  58.29M  80.7KB/s    eta 4m 42s 

    
ets/aclImdb_v1.tar.  72%[=============>      ]  58.30M  73.7KB/s    eta 4m 41s 

    
ts/aclImdb_v1.tar.g  72%[=============>      ]  58.32M  69.8KB/s    eta 4m 41s 

    
s/aclImdb_v1.tar.gz  72%[=============>      ]  58.33M  66.3KB/s    eta 4m 41s 

    
/aclImdb_v1.tar.gz   72%[=============>      ]  58.34M  62.5KB/s    eta 4m 41s 

    
aclImdb_v1.tar.gz    72%[=============>      ]  58.36M  64.0KB/s    eta 4m 41s 

    
clImdb_v1.tar.gz     72%[=============>      ]  58.37M  60.8KB/s    eta 4m 41s 

    
lImdb_v1.tar.gz      72%[=============>      ]  58.39M  59.0KB/s    eta 4m 41s 

    
Imdb_v1.tar.gz       72%[=============>      ]  58.40M  54.8KB/s    eta 4m 41s 

    
mdb_v1.tar.gz        72%[=============>      ]  58.42M  58.2KB/s    eta 4m 40s 

    
db_v1.tar.gz         72%[=============>      ]  58.43M  55.9KB/s    eta 4m 40s 

    
b_v1.tar.gz          72%[=============>      ]  58.44M  55.6KB/s    eta 4m 40s 

    
_v1.tar.gz           72%[=============>      ]  58.46M  55.3KB/s    eta 4m 40s 

    
v1.tar.gz            72%[=============>      ]  58.47M  55.5KB/s    eta 4m 40s 

    
1.tar.gz             72%[=============>      ]  58.49M  56.1KB/s    eta 4m 39s 

    
.tar.gz              72%[=============>      ]  58.50M  56.8KB/s    eta 4m 39s 

    
tar.gz               72%[=============>      ]  58.52M  55.3KB/s    eta 4m 39s 

    
ar.gz                72%[=============>      ]  58.54M  56.2KB/s    eta 4m 39s 

    
r.gz                 72%[=============>      ]  58.55M  55.6KB/s    eta 4m 38s 

    
.gz                  73%[=============>      ]  58.57M  55.0KB/s    eta 4m 38s 

    
gz                   73%[=============>      ]  58.58M  57.5KB/s    eta 4m 38s 

    
z                    73%[=============>      ]  58.60M  58.3KB/s    eta 4m 38s 

    
<div class="k-default-codeblock">
```
                 73%[=============>      ]  58.61M  59.3KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  73%[=============>      ]  58.63M  59.8KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  73%[=============>      ]  58.65M  60.9KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  73%[=============>      ]  58.66M  61.7KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  73%[=============>      ]  58.68M  61.4KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  73%[=============>      ]  58.69M  63.0KB/s    eta 4m 38s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  73%[=============>      ]  58.71M  65.1KB/s    eta 4m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  73%[=============>      ]  58.73M  63.8KB/s    eta 4m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  73%[=============>      ]  58.74M  63.4KB/s    eta 4m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  73%[=============>      ]  58.76M  64.9KB/s    eta 4m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  73%[=============>      ]  58.78M  65.8KB/s    eta 4m 37s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  73%[=============>      ]  58.80M  66.4KB/s    eta 4m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  73%[=============>      ]  58.82M  67.0KB/s    eta 4m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  73%[=============>      ]  58.83M  69.3KB/s    eta 4m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  73%[=============>      ]  58.86M  70.4KB/s    eta 4m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  73%[=============>      ]  58.89M  72.9KB/s    eta 4m 34s 

```
</div>
    
   datasets/aclImdb  73%[=============>      ]  58.91M  75.1KB/s    eta 4m 34s 

    
  datasets/aclImdb_  73%[=============>      ]  58.94M  78.2KB/s    eta 4m 34s 

    
 datasets/aclImdb_v  73%[=============>      ]  58.98M  81.5KB/s    eta 4m 34s 

    
datasets/aclImdb_v1  73%[=============>      ]  59.01M  84.9KB/s    eta 4m 33s 

    
atasets/aclImdb_v1.  73%[=============>      ]  59.05M  88.9KB/s    eta 4m 33s 

    
tasets/aclImdb_v1.t  73%[=============>      ]  59.06M  87.2KB/s    eta 4m 33s 

    
asets/aclImdb_v1.ta  73%[=============>      ]  59.11M  93.1KB/s    eta 4m 33s 

    
sets/aclImdb_v1.tar  73%[=============>      ]  59.15M  97.6KB/s    eta 4m 31s 

    
ets/aclImdb_v1.tar.  73%[=============>      ]  59.17M   100KB/s    eta 4m 31s 

    
ts/aclImdb_v1.tar.g  73%[=============>      ]  59.20M   103KB/s    eta 4m 31s 

    
s/aclImdb_v1.tar.gz  73%[=============>      ]  59.22M   103KB/s    eta 4m 31s 

    
/aclImdb_v1.tar.gz   73%[=============>      ]  59.25M   104KB/s    eta 4m 29s 

    
aclImdb_v1.tar.gz    73%[=============>      ]  59.27M   107KB/s    eta 4m 29s 

    
clImdb_v1.tar.gz     73%[=============>      ]  59.30M   108KB/s    eta 4m 29s 

    
lImdb_v1.tar.gz      73%[=============>      ]  59.32M   109KB/s    eta 4m 29s 

    
Imdb_v1.tar.gz       73%[=============>      ]  59.35M   110KB/s    eta 4m 28s 

    
mdb_v1.tar.gz        74%[=============>      ]  59.37M   110KB/s    eta 4m 28s 

    
db_v1.tar.gz         74%[=============>      ]  59.40M   111KB/s    eta 4m 28s 

    
b_v1.tar.gz          74%[=============>      ]  59.42M   110KB/s    eta 4m 28s 

    
_v1.tar.gz           74%[=============>      ]  59.44M   110KB/s    eta 4m 26s 

    
v1.tar.gz            74%[=============>      ]  59.48M   113KB/s    eta 4m 26s 

    
1.tar.gz             74%[=============>      ]  59.50M   112KB/s    eta 4m 26s 

    
.tar.gz              74%[=============>      ]  59.53M   111KB/s    eta 4m 26s 

    
tar.gz               74%[=============>      ]  59.56M   111KB/s    eta 4m 25s 

    
ar.gz                74%[=============>      ]  59.60M   109KB/s    eta 4m 25s 

    
r.gz                 74%[=============>      ]  59.62M   111KB/s    eta 4m 25s 

    
.gz                  74%[=============>      ]  59.66M   110KB/s    eta 4m 25s 

    
gz                   74%[=============>      ]  59.67M   105KB/s    eta 4m 23s 

    
z                    74%[=============>      ]  59.69M   103KB/s    eta 4m 23s 

    
<div class="k-default-codeblock">
```
                 74%[=============>      ]  59.72M   104KB/s    eta 4m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  74%[=============>      ]  59.74M   104KB/s    eta 4m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  74%[=============>      ]  59.76M   104KB/s    eta 4m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  74%[=============>      ]  59.79M   102KB/s    eta 4m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  74%[=============>      ]  59.81M   102KB/s    eta 4m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  74%[=============>      ]  59.83M   102KB/s    eta 4m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  74%[=============>      ]  59.86M   101KB/s    eta 4m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  74%[=============>      ]  59.88M   102KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  74%[=============>      ]  59.90M   101KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  74%[=============>      ]  59.92M   100KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  74%[=============>      ]  59.94M  95.5KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  74%[=============>      ]  59.98M  95.7KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  74%[=============>      ]  60.00M  93.9KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  74%[=============>      ]  60.02M  92.3KB/s    eta 4m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  74%[=============>      ]  60.03M  89.6KB/s    eta 4m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  74%[=============>      ]  60.05M  83.3KB/s    eta 4m 19s 

```
</div>
    
   datasets/aclImdb  74%[=============>      ]  60.09M  87.2KB/s    eta 4m 19s 

    
  datasets/aclImdb_  74%[=============>      ]  60.11M  82.4KB/s    eta 4m 19s 

    
 datasets/aclImdb_v  74%[=============>      ]  60.13M  83.4KB/s    eta 4m 17s 

    
datasets/aclImdb_v1  74%[=============>      ]  60.15M  83.9KB/s    eta 4m 17s 

    
atasets/aclImdb_v1.  74%[=============>      ]  60.16M  81.3KB/s    eta 4m 17s 

    
tasets/aclImdb_v1.t  75%[==============>     ]  60.19M  81.6KB/s    eta 4m 17s 

    
asets/aclImdb_v1.ta  75%[==============>     ]  60.21M  81.5KB/s    eta 4m 16s 

    
sets/aclImdb_v1.tar  75%[==============>     ]  60.23M  81.7KB/s    eta 4m 16s 

    
ets/aclImdb_v1.tar.  75%[==============>     ]  60.25M  79.8KB/s    eta 4m 16s 

    
ts/aclImdb_v1.tar.g  75%[==============>     ]  60.28M  80.7KB/s    eta 4m 16s 

    
s/aclImdb_v1.tar.gz  75%[==============>     ]  60.30M  80.4KB/s    eta 4m 15s 

    
/aclImdb_v1.tar.gz   75%[==============>     ]  60.32M  79.8KB/s    eta 4m 15s 

    
aclImdb_v1.tar.gz    75%[==============>     ]  60.34M  80.1KB/s    eta 4m 15s 

    
clImdb_v1.tar.gz     75%[==============>     ]  60.37M  79.9KB/s    eta 4m 15s 

    
lImdb_v1.tar.gz      75%[==============>     ]  60.39M  84.2KB/s    eta 4m 14s 

    
Imdb_v1.tar.gz       75%[==============>     ]  60.41M  82.2KB/s    eta 4m 14s 

    
mdb_v1.tar.gz        75%[==============>     ]  60.44M  83.9KB/s    eta 4m 14s 

    
db_v1.tar.gz         75%[==============>     ]  60.47M  84.9KB/s    eta 4m 14s 

    
b_v1.tar.gz          75%[==============>     ]  60.49M  85.9KB/s    eta 4m 13s 

    
_v1.tar.gz           75%[==============>     ]  60.52M  88.5KB/s    eta 4m 13s 

    
v1.tar.gz            75%[==============>     ]  60.58M  91.8KB/s    eta 4m 13s 

    
1.tar.gz             75%[==============>     ]  60.60M  92.9KB/s    eta 4m 11s 

    
.tar.gz              75%[==============>     ]  60.63M  93.9KB/s    eta 4m 11s 

    
tar.gz               75%[==============>     ]  60.66M  94.9KB/s    eta 4m 11s 

    
ar.gz                75%[==============>     ]  60.69M  96.7KB/s    eta 4m 11s 

    
r.gz                 75%[==============>     ]  60.70M  93.8KB/s    eta 4m 10s 

    
.gz                  75%[==============>     ]  60.75M  98.9KB/s    eta 4m 10s 

    
gz                   75%[==============>     ]  60.76M  98.8KB/s    eta 4m 10s 

    
z                    75%[==============>     ]  60.79M   101KB/s    eta 4m 10s 

    
<div class="k-default-codeblock">
```
                 75%[==============>     ]  60.82M   101KB/s    eta 4m 10s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  75%[==============>     ]  60.83M  98.1KB/s    eta 4m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
             da  75%[==============>     ]  60.87M   103KB/s    eta 4m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  75%[==============>     ]  60.89M   103KB/s    eta 4m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  75%[==============>     ]  60.91M   103KB/s    eta 4m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  75%[==============>     ]  60.94M   104KB/s    eta 4m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  75%[==============>     ]  60.96M  96.5KB/s    eta 4m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  76%[==============>     ]  61.00M   104KB/s    eta 4m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  76%[==============>     ]  61.02M  96.0KB/s    eta 4m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  76%[==============>     ]  61.04M  95.4KB/s    eta 4m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  76%[==============>     ]  61.06M  94.2KB/s    eta 4m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  76%[==============>     ]  61.09M  93.5KB/s    eta 4m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  76%[==============>     ]  61.11M  92.3KB/s    eta 4m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  76%[==============>     ]  61.14M  95.4KB/s    eta 4m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  76%[==============>     ]  61.16M  90.0KB/s    eta 4m 4s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  76%[==============>     ]  61.18M  88.9KB/s    eta 4m 4s  

```
</div>
    
   datasets/aclImdb  76%[==============>     ]  61.19M  83.4KB/s    eta 4m 4s  

    
  datasets/aclImdb_  76%[==============>     ]  61.22M  87.0KB/s    eta 4m 4s  

    
 datasets/aclImdb_v  76%[==============>     ]  61.24M  84.9KB/s    eta 4m 3s  

    
datasets/aclImdb_v1  76%[==============>     ]  61.25M  87.8KB/s    eta 4m 3s  

    
atasets/aclImdb_v1.  76%[==============>     ]  61.27M  79.4KB/s    eta 4m 3s  

    
tasets/aclImdb_v1.t  76%[==============>     ]  61.29M  79.0KB/s    eta 4m 3s  

    
asets/aclImdb_v1.ta  76%[==============>     ]  61.30M  78.2KB/s    eta 4m 2s  

    
sets/aclImdb_v1.tar  76%[==============>     ]  61.30M  75.6KB/s    eta 4m 2s  

    
ets/aclImdb_v1.tar.  76%[==============>     ]  61.32M  73.5KB/s    eta 4m 2s  

    
ts/aclImdb_v1.tar.g  76%[==============>     ]  61.34M  74.2KB/s    eta 4m 2s  

    
s/aclImdb_v1.tar.gz  76%[==============>     ]  61.35M  77.2KB/s    eta 4m 1s  

    
/aclImdb_v1.tar.gz   76%[==============>     ]  61.37M  72.1KB/s    eta 4m 1s  

    
aclImdb_v1.tar.gz    76%[==============>     ]  61.39M  71.5KB/s    eta 4m 1s  

    
clImdb_v1.tar.gz     76%[==============>     ]  61.40M  70.6KB/s    eta 4m 1s  

    
lImdb_v1.tar.gz      76%[==============>     ]  61.42M  69.7KB/s    eta 4m 0s  

    
Imdb_v1.tar.gz       76%[==============>     ]  61.44M  68.2KB/s    eta 4m 0s  

    
mdb_v1.tar.gz        76%[==============>     ]  61.45M  66.5KB/s    eta 4m 0s  

    
db_v1.tar.gz         76%[==============>     ]  61.47M  64.9KB/s    eta 4m 0s  

    
b_v1.tar.gz          76%[==============>     ]  61.49M  64.9KB/s    eta 4m 0s  

    
_v1.tar.gz           76%[==============>     ]  61.51M  63.8KB/s    eta 4m 0s  

    
v1.tar.gz            76%[==============>     ]  61.51M  63.5KB/s    eta 4m 0s  

    
1.tar.gz             76%[==============>     ]  61.53M  60.7KB/s    eta 4m 0s  

    
.tar.gz              76%[==============>     ]  61.55M  62.1KB/s    eta 3m 59s 

    
tar.gz               76%[==============>     ]  61.57M  62.2KB/s    eta 3m 59s 

    
ar.gz                76%[==============>     ]  61.59M  64.5KB/s    eta 3m 59s 

    
r.gz                 76%[==============>     ]  61.61M  63.8KB/s    eta 3m 59s 

    
.gz                  76%[==============>     ]  61.63M  65.0KB/s    eta 3m 58s 

    
gz                   76%[==============>     ]  61.63M  61.5KB/s    eta 3m 58s 

    
z                    76%[==============>     ]  61.66M  63.9KB/s    eta 3m 58s 

    
<div class="k-default-codeblock">
```
                 76%[==============>     ]  61.67M  60.2KB/s    eta 3m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  76%[==============>     ]  61.70M  62.7KB/s    eta 3m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  76%[==============>     ]  61.71M  62.6KB/s    eta 3m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  76%[==============>     ]  61.73M  62.1KB/s    eta 3m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  76%[==============>     ]  61.74M  61.6KB/s    eta 3m 58s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  76%[==============>     ]  61.75M  61.1KB/s    eta 3m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  76%[==============>     ]  61.77M  60.7KB/s    eta 3m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  77%[==============>     ]  61.78M  60.3KB/s    eta 3m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  77%[==============>     ]  61.80M  60.3KB/s    eta 3m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  77%[==============>     ]  61.81M  59.3KB/s    eta 3m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  77%[==============>     ]  61.82M  58.8KB/s    eta 3m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  77%[==============>     ]  61.84M  60.8KB/s    eta 3m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  77%[==============>     ]  61.85M  60.6KB/s    eta 3m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  77%[==============>     ]  61.87M  59.6KB/s    eta 3m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  77%[==============>     ]  61.88M  58.8KB/s    eta 3m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  77%[==============>     ]  61.90M  59.0KB/s    eta 3m 55s 

```
</div>
    
   datasets/aclImdb  77%[==============>     ]  61.91M  55.5KB/s    eta 3m 55s 

    
  datasets/aclImdb_  77%[==============>     ]  61.94M  59.3KB/s    eta 3m 54s 

    
 datasets/aclImdb_v  77%[==============>     ]  61.96M  63.2KB/s    eta 3m 54s 

    
datasets/aclImdb_v1  77%[==============>     ]  61.98M  61.3KB/s    eta 3m 54s 

    
atasets/aclImdb_v1.  77%[==============>     ]  62.00M  66.3KB/s    eta 3m 54s 

    
tasets/aclImdb_v1.t  77%[==============>     ]  62.02M  65.2KB/s    eta 3m 54s 

    
asets/aclImdb_v1.ta  77%[==============>     ]  62.04M  66.9KB/s    eta 3m 53s 

    
sets/aclImdb_v1.tar  77%[==============>     ]  62.07M  68.5KB/s    eta 3m 53s 

    
ets/aclImdb_v1.tar.  77%[==============>     ]  62.09M  70.2KB/s    eta 3m 53s 

    
ts/aclImdb_v1.tar.g  77%[==============>     ]  62.11M  72.0KB/s    eta 3m 53s 

    
s/aclImdb_v1.tar.gz  77%[==============>     ]  62.13M  73.9KB/s    eta 3m 53s 

    
/aclImdb_v1.tar.gz   77%[==============>     ]  62.16M  75.7KB/s    eta 3m 52s 

    
aclImdb_v1.tar.gz    77%[==============>     ]  62.17M  76.8KB/s    eta 3m 52s 

    
clImdb_v1.tar.gz     77%[==============>     ]  62.18M  76.1KB/s    eta 3m 52s 

    
lImdb_v1.tar.gz      77%[==============>     ]  62.21M  78.0KB/s    eta 3m 52s 

    
Imdb_v1.tar.gz       77%[==============>     ]  62.23M  79.1KB/s    eta 3m 51s 

    
mdb_v1.tar.gz        77%[==============>     ]  62.25M  80.0KB/s    eta 3m 51s 

    
db_v1.tar.gz         77%[==============>     ]  62.28M  81.9KB/s    eta 3m 51s 

    
b_v1.tar.gz          77%[==============>     ]  62.30M  84.2KB/s    eta 3m 51s 

    
_v1.tar.gz           77%[==============>     ]  62.33M  85.7KB/s    eta 3m 49s 

    
v1.tar.gz            77%[==============>     ]  62.36M  91.7KB/s    eta 3m 49s 

    
1.tar.gz             77%[==============>     ]  62.38M  86.1KB/s    eta 3m 49s 

    
.tar.gz              77%[==============>     ]  62.43M  93.8KB/s    eta 3m 49s 

    
tar.gz               77%[==============>     ]  62.45M  94.5KB/s    eta 3m 48s 

    
ar.gz                77%[==============>     ]  62.47M  94.2KB/s    eta 3m 48s 

    
r.gz                 77%[==============>     ]  62.49M  94.9KB/s    eta 3m 48s 

    
.gz                  77%[==============>     ]  62.52M  94.5KB/s    eta 3m 48s 

    
gz                   77%[==============>     ]  62.55M  96.2KB/s    eta 3m 46s 

    
z                    78%[==============>     ]  62.58M  97.4KB/s    eta 3m 46s 

    
<div class="k-default-codeblock">
```
                 78%[==============>     ]  62.61M  98.5KB/s    eta 3m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  78%[==============>     ]  62.65M   101KB/s    eta 3m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  78%[==============>     ]  62.67M  98.2KB/s    eta 3m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  78%[==============>     ]  62.72M   105KB/s    eta 3m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  78%[==============>     ]  62.74M   105KB/s    eta 3m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  78%[==============>     ]  62.75M   104KB/s    eta 3m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  78%[==============>     ]  62.77M   105KB/s    eta 3m 45s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  78%[==============>     ]  62.80M  99.3KB/s    eta 3m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  78%[==============>     ]  62.83M   101KB/s    eta 3m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  78%[==============>     ]  62.84M  98.4KB/s    eta 3m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  78%[==============>     ]  62.86M  98.2KB/s    eta 3m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  78%[==============>     ]  62.88M  96.6KB/s    eta 3m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  78%[==============>     ]  62.89M  97.9KB/s    eta 3m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  78%[==============>     ]  62.90M  90.9KB/s    eta 3m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  78%[==============>     ]  62.92M  89.0KB/s    eta 3m 42s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  78%[==============>     ]  62.93M  86.8KB/s    eta 3m 42s 

```
</div>
    
   datasets/aclImdb  78%[==============>     ]  62.95M  84.5KB/s    eta 3m 41s 

    
  datasets/aclImdb_  78%[==============>     ]  62.96M  82.9KB/s    eta 3m 41s 

    
 datasets/aclImdb_v  78%[==============>     ]  62.97M  79.6KB/s    eta 3m 41s 

    
datasets/aclImdb_v1  78%[==============>     ]  62.99M  76.8KB/s    eta 3m 41s 

    
atasets/aclImdb_v1.  78%[==============>     ]  63.00M  73.7KB/s    eta 3m 41s 

    
tasets/aclImdb_v1.t  78%[==============>     ]  63.02M  70.0KB/s    eta 3m 41s 

    
asets/aclImdb_v1.ta  78%[==============>     ]  63.03M  70.6KB/s    eta 3m 41s 

    
sets/aclImdb_v1.tar  78%[==============>     ]  63.05M  66.0KB/s    eta 3m 41s 

    
ets/aclImdb_v1.tar.  78%[==============>     ]  63.07M  66.1KB/s    eta 3m 40s 

    
ts/aclImdb_v1.tar.g  78%[==============>     ]  63.09M  65.6KB/s    eta 3m 40s 

    
s/aclImdb_v1.tar.gz  78%[==============>     ]  63.12M  65.8KB/s    eta 3m 40s 

    
/aclImdb_v1.tar.gz   78%[==============>     ]  63.14M  68.5KB/s    eta 3m 40s 

    
aclImdb_v1.tar.gz    78%[==============>     ]  63.15M  62.2KB/s    eta 3m 39s 

    
clImdb_v1.tar.gz     78%[==============>     ]  63.20M  70.5KB/s    eta 3m 39s 

    
lImdb_v1.tar.gz      78%[==============>     ]  63.23M  71.0KB/s    eta 3m 39s 

    
Imdb_v1.tar.gz       78%[==============>     ]  63.25M  72.9KB/s    eta 3m 39s 

    
mdb_v1.tar.gz        78%[==============>     ]  63.27M  74.5KB/s    eta 3m 39s 

    
db_v1.tar.gz         78%[==============>     ]  63.28M  69.8KB/s    eta 3m 37s 

    
b_v1.tar.gz          78%[==============>     ]  63.33M  74.8KB/s    eta 3m 37s 

    
_v1.tar.gz           78%[==============>     ]  63.35M  73.8KB/s    eta 3m 37s 

    
v1.tar.gz            78%[==============>     ]  63.36M  74.6KB/s    eta 3m 36s 

    
1.tar.gz             78%[==============>     ]  63.38M  74.3KB/s    eta 3m 36s 

    
.tar.gz              79%[==============>     ]  63.39M  73.9KB/s    eta 3m 36s 

    
tar.gz               79%[==============>     ]  63.40M  73.6KB/s    eta 3m 36s 

    
ar.gz                79%[==============>     ]  63.41M  72.9KB/s    eta 3m 36s 

    
r.gz                 79%[==============>     ]  63.42M  72.1KB/s    eta 3m 35s 

    
.gz                  79%[==============>     ]  63.43M  70.8KB/s    eta 3m 35s 

    
gz                   79%[==============>     ]  63.45M  69.8KB/s    eta 3m 35s 

    
z                    79%[==============>     ]  63.46M  68.3KB/s    eta 3m 35s 

    
<div class="k-default-codeblock">
```
                 79%[==============>     ]  63.48M  67.0KB/s    eta 3m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  79%[==============>     ]  63.49M  66.4KB/s    eta 3m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  79%[==============>     ]  63.51M  69.2KB/s    eta 3m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  79%[==============>     ]  63.53M  63.3KB/s    eta 3m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  79%[==============>     ]  63.56M  65.2KB/s    eta 3m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  79%[==============>     ]  63.58M  62.9KB/s    eta 3m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  79%[==============>     ]  63.62M  65.5KB/s    eta 3m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  79%[==============>     ]  63.64M  70.3KB/s    eta 3m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  79%[==============>     ]  63.65M  62.6KB/s    eta 3m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  79%[==============>     ]  63.68M  68.0KB/s    eta 3m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  79%[==============>     ]  63.70M  70.1KB/s    eta 3m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  79%[==============>     ]  63.72M  70.6KB/s    eta 3m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  79%[==============>     ]  63.74M  72.1KB/s    eta 3m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  79%[==============>     ]  63.76M  73.7KB/s    eta 3m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  79%[==============>     ]  63.78M  75.5KB/s    eta 3m 31s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  79%[==============>     ]  63.80M  77.4KB/s    eta 3m 31s 

```
</div>
    
   datasets/aclImdb  79%[==============>     ]  63.81M  79.1KB/s    eta 3m 31s 

    
  datasets/aclImdb_  79%[==============>     ]  63.83M  82.1KB/s    eta 3m 31s 

    
 datasets/aclImdb_v  79%[==============>     ]  63.85M  82.7KB/s    eta 3m 31s 

    
datasets/aclImdb_v1  79%[==============>     ]  63.86M  81.5KB/s    eta 3m 30s 

    
atasets/aclImdb_v1.  79%[==============>     ]  63.88M  82.4KB/s    eta 3m 30s 

    
tasets/aclImdb_v1.t  79%[==============>     ]  63.90M  81.7KB/s    eta 3m 30s 

    
asets/aclImdb_v1.ta  79%[==============>     ]  63.92M  79.0KB/s    eta 3m 29s 

    
sets/aclImdb_v1.tar  79%[==============>     ]  63.95M  76.9KB/s    eta 3m 29s 

    
ets/aclImdb_v1.tar.  79%[==============>     ]  63.97M  77.0KB/s    eta 3m 29s 

    
ts/aclImdb_v1.tar.g  79%[==============>     ]  63.99M  79.7KB/s    eta 3m 29s 

    
s/aclImdb_v1.tar.gz  79%[==============>     ]  64.01M  76.0KB/s    eta 3m 29s 

    
/aclImdb_v1.tar.gz   79%[==============>     ]  64.03M  75.9KB/s    eta 3m 28s 

    
aclImdb_v1.tar.gz    79%[==============>     ]  64.05M  78.8KB/s    eta 3m 28s 

    
clImdb_v1.tar.gz     79%[==============>     ]  64.08M  78.9KB/s    eta 3m 28s 

    
lImdb_v1.tar.gz      79%[==============>     ]  64.09M  77.7KB/s    eta 3m 28s 

    
Imdb_v1.tar.gz       79%[==============>     ]  64.11M  77.6KB/s    eta 3m 27s 

    
mdb_v1.tar.gz        79%[==============>     ]  64.14M  77.8KB/s    eta 3m 27s 

    
db_v1.tar.gz         79%[==============>     ]  64.16M  78.1KB/s    eta 3m 27s 

    
b_v1.tar.gz          79%[==============>     ]  64.18M  77.4KB/s    eta 3m 27s 

    
_v1.tar.gz           80%[===============>    ]  64.20M  72.1KB/s    eta 3m 25s 

    
v1.tar.gz            80%[===============>    ]  64.24M  77.7KB/s    eta 3m 25s 

    
1.tar.gz             80%[===============>    ]  64.25M  75.8KB/s    eta 3m 25s 

    
.tar.gz              80%[===============>    ]  64.27M  75.9KB/s    eta 3m 25s 

    
tar.gz               80%[===============>    ]  64.29M  77.4KB/s    eta 3m 24s 

    
ar.gz                80%[===============>    ]  64.32M  76.9KB/s    eta 3m 24s 

    
r.gz                 80%[===============>    ]  64.34M  77.1KB/s    eta 3m 24s 

    
.gz                  80%[===============>    ]  64.36M  82.6KB/s    eta 3m 24s 

    
gz                   80%[===============>    ]  64.38M  80.3KB/s    eta 3m 23s 

    
z                    80%[===============>    ]  64.40M  80.0KB/s    eta 3m 23s 

    
<div class="k-default-codeblock">
```
                 80%[===============>    ]  64.43M  80.8KB/s    eta 3m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  80%[===============>    ]  64.45M  80.7KB/s    eta 3m 23s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  80%[===============>    ]  64.47M  81.5KB/s    eta 3m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  80%[===============>    ]  64.49M  80.4KB/s    eta 3m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  80%[===============>    ]  64.51M  81.0KB/s    eta 3m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  80%[===============>    ]  64.53M  82.1KB/s    eta 3m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  80%[===============>    ]  64.56M  83.1KB/s    eta 3m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  80%[===============>    ]  64.58M  81.5KB/s    eta 3m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  80%[===============>    ]  64.61M  83.3KB/s    eta 3m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  80%[===============>    ]  64.63M  82.6KB/s    eta 3m 21s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  80%[===============>    ]  64.65M  87.8KB/s    eta 3m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  80%[===============>    ]  64.67M  84.8KB/s    eta 3m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  80%[===============>    ]  64.69M  86.8KB/s    eta 3m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  80%[===============>    ]  64.72M  86.8KB/s    eta 3m 20s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  80%[===============>    ]  64.75M  88.8KB/s    eta 3m 18s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  80%[===============>    ]  64.77M  89.7KB/s    eta 3m 18s 

```
</div>
    
   datasets/aclImdb  80%[===============>    ]  64.79M  88.3KB/s    eta 3m 18s 

    
  datasets/aclImdb_  80%[===============>    ]  64.81M  88.2KB/s    eta 3m 18s 

    
 datasets/aclImdb_v  80%[===============>    ]  64.84M  91.2KB/s    eta 3m 17s 

    
datasets/aclImdb_v1  80%[===============>    ]  64.87M  91.3KB/s    eta 3m 17s 

    
atasets/aclImdb_v1.  80%[===============>    ]  64.89M  91.9KB/s    eta 3m 17s 

    
tasets/aclImdb_v1.t  80%[===============>    ]  64.92M  93.0KB/s    eta 3m 17s 

    
asets/aclImdb_v1.ta  80%[===============>    ]  64.94M  92.6KB/s    eta 3m 16s 

    
sets/aclImdb_v1.tar  80%[===============>    ]  64.97M  93.5KB/s    eta 3m 16s 

    
ets/aclImdb_v1.tar.  81%[===============>    ]  65.00M  96.5KB/s    eta 3m 16s 

    
ts/aclImdb_v1.tar.g  81%[===============>    ]  65.00M  89.7KB/s    eta 3m 16s 

    
s/aclImdb_v1.tar.gz  81%[===============>    ]  65.04M  95.2KB/s    eta 3m 14s 

    
/aclImdb_v1.tar.gz   81%[===============>    ]  65.07M  96.7KB/s    eta 3m 14s 

    
aclImdb_v1.tar.gz    81%[===============>    ]  65.08M  90.9KB/s    eta 3m 14s 

    
clImdb_v1.tar.gz     81%[===============>    ]  65.12M  94.8KB/s    eta 3m 14s 

    
lImdb_v1.tar.gz      81%[===============>    ]  65.14M  93.7KB/s    eta 3m 13s 

    
Imdb_v1.tar.gz       81%[===============>    ]  65.16M  92.5KB/s    eta 3m 13s 

    
mdb_v1.tar.gz        81%[===============>    ]  65.18M  92.1KB/s    eta 3m 13s 

    
db_v1.tar.gz         81%[===============>    ]  65.20M  91.4KB/s    eta 3m 13s 

    
b_v1.tar.gz          81%[===============>    ]  65.22M  86.3KB/s    eta 3m 12s 

    
_v1.tar.gz           81%[===============>    ]  65.25M  88.4KB/s    eta 3m 12s 

    
v1.tar.gz            81%[===============>    ]  65.27M  87.6KB/s    eta 3m 12s 

    
1.tar.gz             81%[===============>    ]  65.29M  86.8KB/s    eta 3m 12s 

    
.tar.gz              81%[===============>    ]  65.30M  85.2KB/s    eta 3m 12s 

    
tar.gz               81%[===============>    ]  65.33M  84.6KB/s    eta 3m 11s 

    
ar.gz                81%[===============>    ]  65.35M  85.9KB/s    eta 3m 11s 

    
r.gz                 81%[===============>    ]  65.37M  84.5KB/s    eta 3m 11s 

    
.gz                  81%[===============>    ]  65.39M  82.1KB/s    eta 3m 11s 

    
gz                   81%[===============>    ]  65.42M  80.0KB/s    eta 3m 10s 

    
z                    81%[===============>    ]  65.45M  80.1KB/s    eta 3m 10s 

    
<div class="k-default-codeblock">
```
                 81%[===============>    ]  65.46M  83.2KB/s    eta 3m 10s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  81%[===============>    ]  65.48M  79.4KB/s    eta 3m 10s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  81%[===============>    ]  65.50M  77.7KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  81%[===============>    ]  65.53M  80.8KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  81%[===============>    ]  65.55M  77.1KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  81%[===============>    ]  65.56M  74.2KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  81%[===============>    ]  65.58M  71.8KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  81%[===============>    ]  65.61M  73.2KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  81%[===============>    ]  65.62M  75.5KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  81%[===============>    ]  65.64M  72.6KB/s    eta 3m 8s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  81%[===============>    ]  65.67M  73.5KB/s    eta 3m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  81%[===============>    ]  65.68M  72.9KB/s    eta 3m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  81%[===============>    ]  65.71M  74.1KB/s    eta 3m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  81%[===============>    ]  65.73M  73.8KB/s    eta 3m 6s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  81%[===============>    ]  65.75M  73.5KB/s    eta 3m 5s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  81%[===============>    ]  65.75M  69.0KB/s    eta 3m 5s  

```
</div>
    
   datasets/aclImdb  81%[===============>    ]  65.78M  72.4KB/s    eta 3m 5s  

    
  datasets/aclImdb_  82%[===============>    ]  65.80M  74.8KB/s    eta 3m 5s  

    
 datasets/aclImdb_v  82%[===============>    ]  65.82M  72.8KB/s    eta 3m 5s  

    
datasets/aclImdb_v1  82%[===============>    ]  65.84M  73.1KB/s    eta 3m 5s  

    
atasets/aclImdb_v1.  82%[===============>    ]  65.86M  73.3KB/s    eta 3m 5s  

    
tasets/aclImdb_v1.t  82%[===============>    ]  65.88M  73.5KB/s    eta 3m 5s  

    
asets/aclImdb_v1.ta  82%[===============>    ]  65.90M  73.8KB/s    eta 3m 3s  

    
sets/aclImdb_v1.tar  82%[===============>    ]  65.93M  74.4KB/s    eta 3m 3s  

    
ets/aclImdb_v1.tar.  82%[===============>    ]  65.95M  78.1KB/s    eta 3m 3s  

    
ts/aclImdb_v1.tar.g  82%[===============>    ]  65.97M  75.5KB/s    eta 3m 3s  

    
s/aclImdb_v1.tar.gz  82%[===============>    ]  66.00M  78.8KB/s    eta 3m 3s  

    
/aclImdb_v1.tar.gz   82%[===============>    ]  66.02M  77.0KB/s    eta 3m 3s  

    
aclImdb_v1.tar.gz    82%[===============>    ]  66.04M  78.2KB/s    eta 3m 3s  

    
clImdb_v1.tar.gz     82%[===============>    ]  66.06M  77.9KB/s    eta 3m 1s  

    
lImdb_v1.tar.gz      82%[===============>    ]  66.07M  73.6KB/s    eta 3m 1s  

    
Imdb_v1.tar.gz       82%[===============>    ]  66.10M  77.0KB/s    eta 3m 1s  

    
mdb_v1.tar.gz        82%[===============>    ]  66.12M  75.6KB/s    eta 3m 1s  

    
db_v1.tar.gz         82%[===============>    ]  66.14M  75.5KB/s    eta 3m 0s  

    
b_v1.tar.gz          82%[===============>    ]  66.16M  75.6KB/s    eta 3m 0s  

    
_v1.tar.gz           82%[===============>    ]  66.17M  78.5KB/s    eta 3m 0s  

    
v1.tar.gz            82%[===============>    ]  66.19M  75.8KB/s    eta 3m 0s  

    
1.tar.gz             82%[===============>    ]  66.21M  76.3KB/s    eta 3m 0s  

    
.tar.gz              82%[===============>    ]  66.23M  76.9KB/s    eta 3m 0s  

    
tar.gz               82%[===============>    ]  66.25M  77.8KB/s    eta 3m 0s  

    
ar.gz                82%[===============>    ]  66.28M  78.2KB/s    eta 3m 0s  

    
r.gz                 82%[===============>    ]  66.30M  78.3KB/s    eta 2m 58s 

    
.gz                  82%[===============>    ]  66.32M  77.7KB/s    eta 2m 58s 

    
gz                   82%[===============>    ]  66.34M  76.8KB/s    eta 2m 58s 

    
z                    82%[===============>    ]  66.36M  77.0KB/s    eta 2m 58s 

    
<div class="k-default-codeblock">
```
                 82%[===============>    ]  66.39M  81.9KB/s    eta 2m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  82%[===============>    ]  66.41M  80.1KB/s    eta 2m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  82%[===============>    ]  66.43M  81.1KB/s    eta 2m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  82%[===============>    ]  66.46M  81.8KB/s    eta 2m 57s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  82%[===============>    ]  66.46M  77.8KB/s    eta 2m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  82%[===============>    ]  66.50M  86.2KB/s    eta 2m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  82%[===============>    ]  66.52M  84.6KB/s    eta 2m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  82%[===============>    ]  66.54M  86.4KB/s    eta 2m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  82%[===============>    ]  66.57M  87.2KB/s    eta 2m 56s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  83%[===============>    ]  66.59M  88.2KB/s    eta 2m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  83%[===============>    ]  66.62M  89.5KB/s    eta 2m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  83%[===============>    ]  66.64M  88.0KB/s    eta 2m 55s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  83%[===============>    ]  66.67M  86.8KB/s    eta 2m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  83%[===============>    ]  66.69M  87.9KB/s    eta 2m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  83%[===============>    ]  66.71M  86.7KB/s    eta 2m 54s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  83%[===============>    ]  66.73M  85.9KB/s    eta 2m 54s 

```
</div>
    
   datasets/aclImdb  83%[===============>    ]  66.75M  85.5KB/s    eta 2m 53s 

    
  datasets/aclImdb_  83%[===============>    ]  66.75M  83.0KB/s    eta 2m 53s 

    
 datasets/aclImdb_v  83%[===============>    ]  66.76M  81.2KB/s    eta 2m 53s 

    
datasets/aclImdb_v1  83%[===============>    ]  66.77M  78.8KB/s    eta 2m 53s 

    
atasets/aclImdb_v1.  83%[===============>    ]  66.78M  76.2KB/s    eta 2m 53s 

    
tasets/aclImdb_v1.t  83%[===============>    ]  66.79M  74.0KB/s    eta 2m 52s 

    
asets/aclImdb_v1.ta  83%[===============>    ]  66.80M  71.7KB/s    eta 2m 52s 

    
sets/aclImdb_v1.tar  83%[===============>    ]  66.81M  69.6KB/s    eta 2m 52s 

    
ets/aclImdb_v1.tar.  83%[===============>    ]  66.82M  72.3KB/s    eta 2m 52s 

    
ts/aclImdb_v1.tar.g  83%[===============>    ]  66.83M  67.1KB/s    eta 2m 52s 

    
s/aclImdb_v1.tar.gz  83%[===============>    ]  66.84M  63.9KB/s    eta 2m 52s 

    
/aclImdb_v1.tar.gz   83%[===============>    ]  66.86M  61.9KB/s    eta 2m 52s 

    
aclImdb_v1.tar.gz    83%[===============>    ]  66.87M  59.2KB/s    eta 2m 52s 

    
clImdb_v1.tar.gz     83%[===============>    ]  66.88M  56.8KB/s    eta 2m 52s 

    
lImdb_v1.tar.gz      83%[===============>    ]  66.89M  54.4KB/s    eta 2m 52s 

    
Imdb_v1.tar.gz       83%[===============>    ]  66.91M  54.3KB/s    eta 2m 51s 

    
mdb_v1.tar.gz        83%[===============>    ]  66.92M  53.2KB/s    eta 2m 51s 

    
db_v1.tar.gz         83%[===============>    ]  66.93M  49.9KB/s    eta 2m 51s 

    
b_v1.tar.gz          83%[===============>    ]  66.95M  49.3KB/s    eta 2m 51s 

    
_v1.tar.gz           83%[===============>    ]  66.97M  49.0KB/s    eta 2m 50s 

    
v1.tar.gz            83%[===============>    ]  66.99M  49.5KB/s    eta 2m 50s 

    
1.tar.gz             83%[===============>    ]  67.00M  52.1KB/s    eta 2m 50s 

    
.tar.gz              83%[===============>    ]  67.03M  54.8KB/s    eta 2m 50s 

    
tar.gz               83%[===============>    ]  67.05M  56.6KB/s    eta 2m 49s 

    
ar.gz                83%[===============>    ]  67.06M  56.9KB/s    eta 2m 49s 

    
r.gz                 83%[===============>    ]  67.10M  62.4KB/s    eta 2m 49s 

    
.gz                  83%[===============>    ]  67.12M  64.6KB/s    eta 2m 49s 

    
gz                   83%[===============>    ]  67.14M  66.5KB/s    eta 2m 48s 

    
z                    83%[===============>    ]  67.16M  67.9KB/s    eta 2m 48s 

    
<div class="k-default-codeblock">
```
                 83%[===============>    ]  67.17M  63.7KB/s    eta 2m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  83%[===============>    ]  67.22M  71.0KB/s    eta 2m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  83%[===============>    ]  67.24M  72.2KB/s    eta 2m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  83%[===============>    ]  67.26M  74.4KB/s    eta 2m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  83%[===============>    ]  67.27M  75.3KB/s    eta 2m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  83%[===============>    ]  67.30M  77.5KB/s    eta 2m 47s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  83%[===============>    ]  67.32M  79.5KB/s    eta 2m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  83%[===============>    ]  67.34M  80.2KB/s    eta 2m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  83%[===============>    ]  67.37M  82.5KB/s    eta 2m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  83%[===============>    ]  67.39M  84.2KB/s    eta 2m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  84%[===============>    ]  67.41M  85.3KB/s    eta 2m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  84%[===============>    ]  67.43M  85.4KB/s    eta 2m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  84%[===============>    ]  67.46M  85.5KB/s    eta 2m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  84%[===============>    ]  67.48M  85.4KB/s    eta 2m 44s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  84%[===============>    ]  67.50M  86.3KB/s    eta 2m 43s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  84%[===============>    ]  67.53M  88.8KB/s    eta 2m 43s 

```
</div>
    
   datasets/aclImdb  84%[===============>    ]  67.55M  86.4KB/s    eta 2m 43s 

    
  datasets/aclImdb_  84%[===============>    ]  67.58M  87.0KB/s    eta 2m 43s 

    
 datasets/aclImdb_v  84%[===============>    ]  67.60M  87.4KB/s    eta 2m 42s 

    
datasets/aclImdb_v1  84%[===============>    ]  67.61M  84.3KB/s    eta 2m 42s 

    
atasets/aclImdb_v1.  84%[===============>    ]  67.65M  95.6KB/s    eta 2m 42s 

    
tasets/aclImdb_v1.t  84%[===============>    ]  67.67M  89.8KB/s    eta 2m 42s 

    
asets/aclImdb_v1.ta  84%[===============>    ]  67.69M  90.3KB/s    eta 2m 41s 

    
sets/aclImdb_v1.tar  84%[===============>    ]  67.71M  89.9KB/s    eta 2m 41s 

    
ets/aclImdb_v1.tar.  84%[===============>    ]  67.74M  90.9KB/s    eta 2m 41s 

    
ts/aclImdb_v1.tar.g  84%[===============>    ]  67.76M  90.1KB/s    eta 2m 41s 

    
s/aclImdb_v1.tar.gz  84%[===============>    ]  67.79M  90.1KB/s    eta 2m 39s 

    
/aclImdb_v1.tar.gz   84%[===============>    ]  67.81M  92.1KB/s    eta 2m 39s 

    
aclImdb_v1.tar.gz    84%[===============>    ]  67.84M  92.2KB/s    eta 2m 39s 

    
clImdb_v1.tar.gz     84%[===============>    ]  67.86M  92.2KB/s    eta 2m 39s 

    
lImdb_v1.tar.gz      84%[===============>    ]  67.88M  90.3KB/s    eta 2m 38s 

    
Imdb_v1.tar.gz       84%[===============>    ]  67.90M  91.6KB/s    eta 2m 38s 

    
mdb_v1.tar.gz        84%[===============>    ]  67.92M  91.2KB/s    eta 2m 38s 

    
db_v1.tar.gz         84%[===============>    ]  67.93M  90.8KB/s    eta 2m 38s 

    
b_v1.tar.gz          84%[===============>    ]  67.95M  89.4KB/s    eta 2m 38s 

    
_v1.tar.gz           84%[===============>    ]  67.97M  87.3KB/s    eta 2m 37s 

    
v1.tar.gz            84%[===============>    ]  67.99M  81.1KB/s    eta 2m 37s 

    
1.tar.gz             84%[===============>    ]  68.01M  80.9KB/s    eta 2m 37s 

    
.tar.gz              84%[===============>    ]  68.02M  79.4KB/s    eta 2m 36s 

    
tar.gz               84%[===============>    ]  68.04M  81.9KB/s    eta 2m 36s 

    
ar.gz                84%[===============>    ]  68.06M  76.9KB/s    eta 2m 36s 

    
r.gz                 84%[===============>    ]  68.07M  75.9KB/s    eta 2m 36s 

    
.gz                  84%[===============>    ]  68.09M  74.4KB/s    eta 2m 36s 

    
gz                   84%[===============>    ]  68.11M  73.3KB/s    eta 2m 36s 

    
z                    84%[===============>    ]  68.12M  71.3KB/s    eta 2m 36s 

    
<div class="k-default-codeblock">
```
                 84%[===============>    ]  68.14M  70.4KB/s    eta 2m 36s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  84%[===============>    ]  68.16M  69.2KB/s    eta 2m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  84%[===============>    ]  68.17M  66.6KB/s    eta 2m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  84%[===============>    ]  68.19M  65.9KB/s    eta 2m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  85%[================>   ]  68.21M  64.9KB/s    eta 2m 35s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  85%[================>   ]  68.22M  64.9KB/s    eta 2m 34s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  85%[================>   ]  68.23M  61.1KB/s    eta 2m 34s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  85%[================>   ]  68.25M  61.7KB/s    eta 2m 34s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  85%[================>   ]  68.26M  61.2KB/s    eta 2m 34s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  85%[================>   ]  68.28M  61.3KB/s    eta 2m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  85%[================>   ]  68.30M  62.0KB/s    eta 2m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  85%[================>   ]  68.32M  65.4KB/s    eta 2m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  85%[================>   ]  68.34M  63.9KB/s    eta 2m 33s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  85%[================>   ]  68.36M  64.9KB/s    eta 2m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  85%[================>   ]  68.38M  63.3KB/s    eta 2m 32s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  85%[================>   ]  68.41M  65.8KB/s    eta 2m 32s 

```
</div>
    
   datasets/aclImdb  85%[================>   ]  68.42M  64.1KB/s    eta 2m 32s 

    
  datasets/aclImdb_  85%[================>   ]  68.44M  65.6KB/s    eta 2m 32s 

    
 datasets/aclImdb_v  85%[================>   ]  68.46M  65.6KB/s    eta 2m 32s 

    
datasets/aclImdb_v1  85%[================>   ]  68.47M  65.8KB/s    eta 2m 32s 

    
atasets/aclImdb_v1.  85%[================>   ]  68.48M  64.9KB/s    eta 2m 32s 

    
tasets/aclImdb_v1.t  85%[================>   ]  68.50M  64.1KB/s    eta 2m 31s 

    
asets/aclImdb_v1.ta  85%[================>   ]  68.51M  63.8KB/s    eta 2m 31s 

    
sets/aclImdb_v1.tar  85%[================>   ]  68.52M  62.2KB/s    eta 2m 31s 

    
ets/aclImdb_v1.tar.  85%[================>   ]  68.54M  61.5KB/s    eta 2m 31s 

    
ts/aclImdb_v1.tar.g  85%[================>   ]  68.55M  61.9KB/s    eta 2m 30s 

    
s/aclImdb_v1.tar.gz  85%[================>   ]  68.57M  63.4KB/s    eta 2m 30s 

    
/aclImdb_v1.tar.gz   85%[================>   ]  68.58M  61.8KB/s    eta 2m 30s 

    
aclImdb_v1.tar.gz    85%[================>   ]  68.59M  58.9KB/s    eta 2m 30s 

    
clImdb_v1.tar.gz     85%[================>   ]  68.61M  59.5KB/s    eta 2m 29s 

    
lImdb_v1.tar.gz      85%[================>   ]  68.62M  58.2KB/s    eta 2m 29s 

    
Imdb_v1.tar.gz       85%[================>   ]  68.64M  58.6KB/s    eta 2m 29s 

    
mdb_v1.tar.gz        85%[================>   ]  68.66M  57.8KB/s    eta 2m 29s 

    
db_v1.tar.gz         85%[================>   ]  68.68M  59.0KB/s    eta 2m 29s 

    
b_v1.tar.gz          85%[================>   ]  68.69M  56.3KB/s    eta 2m 29s 

    
_v1.tar.gz           85%[================>   ]  68.71M  58.4KB/s    eta 2m 29s 

    
v1.tar.gz            85%[================>   ]  68.73M  57.5KB/s    eta 2m 29s 

    
1.tar.gz             85%[================>   ]  68.74M  57.6KB/s    eta 2m 28s 

    
.tar.gz              85%[================>   ]  68.76M  58.8KB/s    eta 2m 28s 

    
tar.gz               85%[================>   ]  68.77M  58.9KB/s    eta 2m 28s 

    
ar.gz                85%[================>   ]  68.79M  58.9KB/s    eta 2m 28s 

    
r.gz                 85%[================>   ]  68.81M  61.4KB/s    eta 2m 27s 

    
.gz                  85%[================>   ]  68.82M  62.1KB/s    eta 2m 27s 

    
gz                   85%[================>   ]  68.84M  63.2KB/s    eta 2m 27s 

    
z                    85%[================>   ]  68.86M  63.8KB/s    eta 2m 27s 

    
<div class="k-default-codeblock">
```
                 85%[================>   ]  68.88M  65.1KB/s    eta 2m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  85%[================>   ]  68.91M  67.1KB/s    eta 2m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  85%[================>   ]  68.93M  69.6KB/s    eta 2m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  85%[================>   ]  68.96M  73.8KB/s    eta 2m 26s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  85%[================>   ]  68.99M  75.0KB/s    eta 2m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  86%[================>   ]  69.02M  79.4KB/s    eta 2m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  86%[================>   ]  69.06M  83.2KB/s    eta 2m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  86%[================>   ]  69.10M  87.3KB/s    eta 2m 25s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  86%[================>   ]  69.15M  95.0KB/s    eta 2m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  86%[================>   ]  69.20M   102KB/s    eta 2m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  86%[================>   ]  69.26M   109KB/s    eta 2m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  86%[================>   ]  69.32M   118KB/s    eta 2m 22s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  86%[================>   ]  69.37M   122KB/s    eta 2m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  86%[================>   ]  69.46M   137KB/s    eta 2m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  86%[================>   ]  69.51M   144KB/s    eta 2m 19s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  86%[================>   ]  69.56M   152KB/s    eta 2m 19s 

```
</div>
    
   datasets/aclImdb  86%[================>   ]  69.61M   154KB/s    eta 2m 16s 

    
  datasets/aclImdb_  86%[================>   ]  69.69M   168KB/s    eta 2m 16s 

    
 datasets/aclImdb_v  86%[================>   ]  69.73M   172KB/s    eta 2m 16s 

    
datasets/aclImdb_v1  86%[================>   ]  69.78M   179KB/s    eta 2m 16s 

    
atasets/aclImdb_v1.  87%[================>   ]  69.83M   185KB/s    eta 2m 16s 

    
tasets/aclImdb_v1.t  87%[================>   ]  69.88M   189KB/s    eta 2m 12s 

    
asets/aclImdb_v1.ta  87%[================>   ]  69.90M   186KB/s    eta 2m 12s 

    
sets/aclImdb_v1.tar  87%[================>   ]  69.97M   188KB/s    eta 2m 12s 

    
ets/aclImdb_v1.tar.  87%[================>   ]  70.03M   194KB/s    eta 2m 10s 

    
ts/aclImdb_v1.tar.g  87%[================>   ]  70.06M   194KB/s    eta 2m 10s 

    
s/aclImdb_v1.tar.gz  87%[================>   ]  70.09M   191KB/s    eta 2m 10s 

    
/aclImdb_v1.tar.gz   87%[================>   ]  70.10M   184KB/s    eta 2m 10s 

    
aclImdb_v1.tar.gz    87%[================>   ]  70.15M   184KB/s    eta 2m 8s  

    
clImdb_v1.tar.gz     87%[================>   ]  70.16M   179KB/s    eta 2m 8s  

    
lImdb_v1.tar.gz      87%[================>   ]  70.19M   175KB/s    eta 2m 8s  

    
Imdb_v1.tar.gz       87%[================>   ]  70.22M   168KB/s    eta 2m 8s  

    
mdb_v1.tar.gz        87%[================>   ]  70.25M   166KB/s    eta 2m 7s  

    
db_v1.tar.gz         87%[================>   ]  70.27M   153KB/s    eta 2m 7s  

    
b_v1.tar.gz          87%[================>   ]  70.31M   149KB/s    eta 2m 7s  

    
_v1.tar.gz           87%[================>   ]  70.34M   144KB/s    eta 2m 7s  

    
v1.tar.gz            87%[================>   ]  70.36M   143KB/s    eta 2m 6s  

    
1.tar.gz             87%[================>   ]  70.40M   135KB/s    eta 2m 6s  

    
.tar.gz              87%[================>   ]  70.43M   126KB/s    eta 2m 6s  

    
tar.gz               87%[================>   ]  70.46M   123KB/s    eta 2m 4s  

    
ar.gz                87%[================>   ]  70.50M   121KB/s    eta 2m 4s  

    
r.gz                 87%[================>   ]  70.51M   115KB/s    eta 2m 4s  

    
.gz                  87%[================>   ]  70.53M   116KB/s    eta 2m 4s  

    
gz                   87%[================>   ]  70.55M   110KB/s    eta 2m 4s  

    
z                    87%[================>   ]  70.56M   102KB/s    eta 2m 3s  

    
<div class="k-default-codeblock">
```
                 87%[================>   ]  70.58M  99.3KB/s    eta 2m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
              d  87%[================>   ]  70.60M  97.0KB/s    eta 2m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
             da  88%[================>   ]  70.62M  99.7KB/s    eta 2m 3s  

```
</div>
    
<div class="k-default-codeblock">
```
            dat  88%[================>   ]  70.64M  95.1KB/s    eta 2m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
           data  88%[================>   ]  70.66M  94.1KB/s    eta 2m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
          datas  88%[================>   ]  70.68M  92.4KB/s    eta 2m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
         datase  88%[================>   ]  70.70M  90.9KB/s    eta 2m 2s  

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  88%[================>   ]  70.72M  89.2KB/s    eta 2m 1s  

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  88%[================>   ]  70.74M  88.6KB/s    eta 2m 1s  

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  88%[================>   ]  70.76M  85.8KB/s    eta 2m 1s  

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  88%[================>   ]  70.78M  82.6KB/s    eta 2m 1s  

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  88%[================>   ]  70.81M  84.2KB/s    eta 2m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  88%[================>   ]  70.83M  81.3KB/s    eta 2m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  88%[================>   ]  70.85M  82.9KB/s    eta 2m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  88%[================>   ]  70.87M  80.5KB/s    eta 2m 0s  

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  88%[================>   ]  70.89M  79.0KB/s    eta 2m 0s  

```
</div>
    
   datasets/aclImdb  88%[================>   ]  70.90M  78.6KB/s    eta 1m 59s 

    
  datasets/aclImdb_  88%[================>   ]  70.92M  78.6KB/s    eta 1m 59s 

    
 datasets/aclImdb_v  88%[================>   ]  70.95M  79.5KB/s    eta 1m 59s 

    
datasets/aclImdb_v1  88%[================>   ]  70.97M  81.6KB/s    eta 1m 59s 

    
atasets/aclImdb_v1.  88%[================>   ]  71.00M  82.8KB/s    eta 1m 57s 

    
tasets/aclImdb_v1.t  88%[================>   ]  71.02M  83.9KB/s    eta 1m 57s 

    
asets/aclImdb_v1.ta  88%[================>   ]  71.05M  84.9KB/s    eta 1m 57s 

    
sets/aclImdb_v1.tar  88%[================>   ]  71.07M  86.0KB/s    eta 1m 57s 

    
ets/aclImdb_v1.tar.  88%[================>   ]  71.10M  87.2KB/s    eta 1m 56s 

    
ts/aclImdb_v1.tar.g  88%[================>   ]  71.12M  88.4KB/s    eta 1m 56s 

    
s/aclImdb_v1.tar.gz  88%[================>   ]  71.15M  89.7KB/s    eta 1m 56s 

    
/aclImdb_v1.tar.gz   88%[================>   ]  71.17M  91.0KB/s    eta 1m 56s 

    
aclImdb_v1.tar.gz    88%[================>   ]  71.20M  92.3KB/s    eta 1m 55s 

    
clImdb_v1.tar.gz     88%[================>   ]  71.23M  93.4KB/s    eta 1m 55s 

    
lImdb_v1.tar.gz      88%[================>   ]  71.26M  91.8KB/s    eta 1m 55s 

    
Imdb_v1.tar.gz       88%[================>   ]  71.30M  97.1KB/s    eta 1m 53s 

    
mdb_v1.tar.gz        88%[================>   ]  71.32M  92.8KB/s    eta 1m 53s 

    
db_v1.tar.gz         88%[================>   ]  71.35M  95.9KB/s    eta 1m 53s 

    
b_v1.tar.gz          88%[================>   ]  71.36M  95.6KB/s    eta 1m 53s 

    
_v1.tar.gz           88%[================>   ]  71.38M  95.3KB/s    eta 1m 53s 

    
v1.tar.gz            88%[================>   ]  71.40M  94.9KB/s    eta 1m 52s 

    
1.tar.gz             89%[================>   ]  71.42M  94.7KB/s    eta 1m 52s 

    
.tar.gz              89%[================>   ]  71.44M  93.5KB/s    eta 1m 52s 

    
tar.gz               89%[================>   ]  71.46M  92.3KB/s    eta 1m 52s 

    
ar.gz                89%[================>   ]  71.48M  91.6KB/s    eta 1m 51s 

    
r.gz                 89%[================>   ]  71.50M  91.3KB/s    eta 1m 51s 

    
.gz                  89%[================>   ]  71.52M  90.2KB/s    eta 1m 51s 

    
gz                   89%[================>   ]  71.54M  88.9KB/s    eta 1m 51s 

    
z                    89%[================>   ]  71.55M  86.7KB/s    eta 1m 50s 

    
<div class="k-default-codeblock">
```
                 89%[================>   ]  71.58M  85.8KB/s    eta 1m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
              d  89%[================>   ]  71.60M  85.0KB/s    eta 1m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
             da  89%[================>   ]  71.62M  85.3KB/s    eta 1m 50s 

```
</div>
    
<div class="k-default-codeblock">
```
            dat  89%[================>   ]  71.65M  85.2KB/s    eta 1m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
           data  89%[================>   ]  71.67M  84.0KB/s    eta 1m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
          datas  89%[================>   ]  71.70M  88.1KB/s    eta 1m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
         datase  89%[================>   ]  71.73M  84.4KB/s    eta 1m 49s 

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  89%[================>   ]  71.76M  89.2KB/s    eta 1m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  89%[================>   ]  71.80M  89.5KB/s    eta 1m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  89%[================>   ]  71.81M  88.4KB/s    eta 1m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  89%[================>   ]  71.87M  96.6KB/s    eta 1m 48s 

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  89%[================>   ]  71.91M  99.0KB/s    eta 1m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  89%[================>   ]  71.94M   103KB/s    eta 1m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  89%[================>   ]  71.98M   106KB/s    eta 1m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  89%[================>   ]  72.02M   111KB/s    eta 1m 46s 

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  89%[================>   ]  72.06M   109KB/s    eta 1m 44s 

```
</div>
    
   datasets/aclImdb  89%[================>   ]  72.13M   118KB/s    eta 1m 44s 

    
  datasets/aclImdb_  89%[================>   ]  72.17M   121KB/s    eta 1m 44s 

    
 datasets/aclImdb_v  89%[================>   ]  72.20M   124KB/s    eta 1m 44s 

    
datasets/aclImdb_v1  90%[=================>  ]  72.24M   127KB/s    eta 1m 41s 

    
atasets/aclImdb_v1.  90%[=================>  ]  72.26M   126KB/s    eta 1m 41s 

    
tasets/aclImdb_v1.t  90%[=================>  ]  72.31M   133KB/s    eta 1m 41s 

    
asets/aclImdb_v1.ta  90%[=================>  ]  72.33M   133KB/s    eta 1m 41s 

    
sets/aclImdb_v1.tar  90%[=================>  ]  72.36M   134KB/s    eta 1m 41s 

    
ets/aclImdb_v1.tar.  90%[=================>  ]  72.38M   134KB/s    eta 99s    

    
ts/aclImdb_v1.tar.g  90%[=================>  ]  72.41M   135KB/s    eta 99s    

    
s/aclImdb_v1.tar.gz  90%[=================>  ]  72.44M   135KB/s    eta 99s    

    
/aclImdb_v1.tar.gz   90%[=================>  ]  72.48M   136KB/s    eta 99s    

    
aclImdb_v1.tar.gz    90%[=================>  ]  72.51M   135KB/s    eta 98s    

    
clImdb_v1.tar.gz     90%[=================>  ]  72.55M   140KB/s    eta 98s    

    
lImdb_v1.tar.gz      90%[=================>  ]  72.58M   131KB/s    eta 98s    

    
Imdb_v1.tar.gz       90%[=================>  ]  72.63M   128KB/s    eta 96s    

    
mdb_v1.tar.gz        90%[=================>  ]  72.66M   124KB/s    eta 96s    

    
db_v1.tar.gz         90%[=================>  ]  72.68M   121KB/s    eta 96s    

    
b_v1.tar.gz          90%[=================>  ]  72.69M   123KB/s    eta 96s    

    
_v1.tar.gz           90%[=================>  ]  72.71M   112KB/s    eta 95s    

    
v1.tar.gz            90%[=================>  ]  72.72M   108KB/s    eta 95s    

    
1.tar.gz             90%[=================>  ]  72.73M   104KB/s    eta 95s    

    
.tar.gz              90%[=================>  ]  72.75M  99.6KB/s    eta 95s    

    
tar.gz               90%[=================>  ]  72.76M  98.7KB/s    eta 95s    

    
ar.gz                90%[=================>  ]  72.78M  95.0KB/s    eta 95s    

    
r.gz                 90%[=================>  ]  72.79M  90.1KB/s    eta 95s    

    
.gz                  90%[=================>  ]  72.80M  87.7KB/s    eta 95s    

    
gz                   90%[=================>  ]  72.81M  83.9KB/s    eta 94s    

    
z                    90%[=================>  ]  72.83M  80.9KB/s    eta 94s    

    
<div class="k-default-codeblock">
```
                 90%[=================>  ]  72.85M  77.6KB/s    eta 94s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  90%[=================>  ]  72.86M  73.3KB/s    eta 94s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  90%[=================>  ]  72.86M  66.7KB/s    eta 93s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  90%[=================>  ]  72.89M  65.6KB/s    eta 93s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  90%[=================>  ]  72.90M  63.2KB/s    eta 93s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  90%[=================>  ]  72.91M  60.2KB/s    eta 93s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  90%[=================>  ]  72.93M  58.3KB/s    eta 93s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  90%[=================>  ]  72.95M  57.3KB/s    eta 92s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  90%[=================>  ]  72.97M  57.9KB/s    eta 92s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  90%[=================>  ]  72.99M  59.3KB/s    eta 92s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  91%[=================>  ]  73.01M  61.4KB/s    eta 92s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  91%[=================>  ]  73.02M  62.1KB/s    eta 91s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  91%[=================>  ]  73.05M  65.2KB/s    eta 91s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  91%[=================>  ]  73.07M  66.6KB/s    eta 91s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  91%[=================>  ]  73.09M  69.1KB/s    eta 91s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  91%[=================>  ]  73.10M  69.6KB/s    eta 91s    

```
</div>
    
   datasets/aclImdb  91%[=================>  ]  73.12M  71.1KB/s    eta 90s    

    
  datasets/aclImdb_  91%[=================>  ]  73.14M  69.4KB/s    eta 90s    

    
 datasets/aclImdb_v  91%[=================>  ]  73.15M  70.4KB/s    eta 90s    

    
datasets/aclImdb_v1  91%[=================>  ]  73.18M  65.6KB/s    eta 89s    

    
atasets/aclImdb_v1.  91%[=================>  ]  73.19M  65.5KB/s    eta 89s    

    
tasets/aclImdb_v1.t  91%[=================>  ]  73.20M  64.3KB/s    eta 89s    

    
asets/aclImdb_v1.ta  91%[=================>  ]  73.21M  62.6KB/s    eta 89s    

    
sets/aclImdb_v1.tar  91%[=================>  ]  73.22M  62.3KB/s    eta 89s    

    
ets/aclImdb_v1.tar.  91%[=================>  ]  73.23M  60.0KB/s    eta 89s    

    
ts/aclImdb_v1.tar.g  91%[=================>  ]  73.24M  59.7KB/s    eta 89s    

    
s/aclImdb_v1.tar.gz  91%[=================>  ]  73.25M  58.6KB/s    eta 89s    

    
/aclImdb_v1.tar.gz   91%[=================>  ]  73.26M  57.0KB/s    eta 89s    

    
aclImdb_v1.tar.gz    91%[=================>  ]  73.27M  55.1KB/s    eta 89s    

    
clImdb_v1.tar.gz     91%[=================>  ]  73.28M  54.3KB/s    eta 88s    

    
lImdb_v1.tar.gz      91%[=================>  ]  73.29M  51.3KB/s    eta 88s    

    
Imdb_v1.tar.gz       91%[=================>  ]  73.31M  50.0KB/s    eta 88s    

    
mdb_v1.tar.gz        91%[=================>  ]  73.32M  48.8KB/s    eta 88s    

    
db_v1.tar.gz         91%[=================>  ]  73.33M  45.8KB/s    eta 88s    

    
b_v1.tar.gz          91%[=================>  ]  73.34M  44.6KB/s    eta 88s    

    
_v1.tar.gz           91%[=================>  ]  73.36M  45.4KB/s    eta 88s    

    
v1.tar.gz            91%[=================>  ]  73.37M  47.2KB/s    eta 88s    

    
1.tar.gz             91%[=================>  ]  73.39M  48.0KB/s    eta 87s    

    
.tar.gz              91%[=================>  ]  73.41M  49.4KB/s    eta 87s    

    
tar.gz               91%[=================>  ]  73.43M  52.6KB/s    eta 87s    

    
ar.gz                91%[=================>  ]  73.46M  55.8KB/s    eta 87s    

    
r.gz                 91%[=================>  ]  73.48M  55.2KB/s    eta 86s    

    
.gz                  91%[=================>  ]  73.52M  63.4KB/s    eta 86s    

    
gz                   91%[=================>  ]  73.55M  65.5KB/s    eta 86s    

    
z                    91%[=================>  ]  73.57M  68.3KB/s    eta 86s    

    
<div class="k-default-codeblock">
```
                 91%[=================>  ]  73.60M  71.8KB/s    eta 86s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  91%[=================>  ]  73.63M  71.7KB/s    eta 84s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  91%[=================>  ]  73.67M  75.9KB/s    eta 84s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  91%[=================>  ]  73.70M  79.6KB/s    eta 84s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  91%[=================>  ]  73.71M  80.4KB/s    eta 84s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  91%[=================>  ]  73.73M  80.6KB/s    eta 83s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  91%[=================>  ]  73.74M  81.7KB/s    eta 83s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  91%[=================>  ]  73.75M  82.6KB/s    eta 83s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  91%[=================>  ]  73.77M  79.1KB/s    eta 82s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  91%[=================>  ]  73.80M  81.5KB/s    eta 82s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  92%[=================>  ]  73.81M  81.3KB/s    eta 82s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  92%[=================>  ]  73.83M  80.3KB/s    eta 82s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  92%[=================>  ]  73.84M  80.6KB/s    eta 81s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  92%[=================>  ]  73.86M  79.0KB/s    eta 81s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  92%[=================>  ]  73.88M  79.9KB/s    eta 81s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  92%[=================>  ]  73.89M  74.3KB/s    eta 81s    

```
</div>
    
   datasets/aclImdb  92%[=================>  ]  73.91M  72.6KB/s    eta 80s    

    
  datasets/aclImdb_  92%[=================>  ]  73.92M  68.4KB/s    eta 80s    

    
 datasets/aclImdb_v  92%[=================>  ]  73.94M  67.8KB/s    eta 80s    

    
datasets/aclImdb_v1  92%[=================>  ]  73.95M  67.4KB/s    eta 80s    

    
atasets/aclImdb_v1.  92%[=================>  ]  73.97M  63.6KB/s    eta 80s    

    
tasets/aclImdb_v1.t  92%[=================>  ]  73.98M  61.8KB/s    eta 80s    

    
asets/aclImdb_v1.ta  92%[=================>  ]  74.00M  62.0KB/s    eta 80s    

    
sets/aclImdb_v1.tar  92%[=================>  ]  74.02M  61.8KB/s    eta 80s    

    
ets/aclImdb_v1.tar.  92%[=================>  ]  74.03M  60.7KB/s    eta 80s    

    
ts/aclImdb_v1.tar.g  92%[=================>  ]  74.04M  60.8KB/s    eta 79s    

    
s/aclImdb_v1.tar.gz  92%[=================>  ]  74.06M  63.8KB/s    eta 79s    

    
/aclImdb_v1.tar.gz   92%[=================>  ]  74.08M  61.3KB/s    eta 79s    

    
aclImdb_v1.tar.gz    92%[=================>  ]  74.09M  62.0KB/s    eta 79s    

    
clImdb_v1.tar.gz     92%[=================>  ]  74.11M  61.2KB/s    eta 78s    

    
lImdb_v1.tar.gz      92%[=================>  ]  74.13M  62.3KB/s    eta 78s    

    
Imdb_v1.tar.gz       92%[=================>  ]  74.14M  61.5KB/s    eta 78s    

    
mdb_v1.tar.gz        92%[=================>  ]  74.16M  59.1KB/s    eta 77s    

    
db_v1.tar.gz         92%[=================>  ]  74.18M  61.1KB/s    eta 77s    

    
b_v1.tar.gz          92%[=================>  ]  74.19M  60.0KB/s    eta 77s    

    
_v1.tar.gz           92%[=================>  ]  74.21M  54.5KB/s    eta 77s    

    
v1.tar.gz            92%[=================>  ]  74.23M  54.8KB/s    eta 77s    

    
1.tar.gz             92%[=================>  ]  74.24M  56.5KB/s    eta 77s    

    
.tar.gz              92%[=================>  ]  74.26M  56.3KB/s    eta 77s    

    
tar.gz               92%[=================>  ]  74.27M  55.1KB/s    eta 77s    

    
ar.gz                92%[=================>  ]  74.28M  54.4KB/s    eta 76s    

    
r.gz                 92%[=================>  ]  74.29M  54.1KB/s    eta 76s    

    
.gz                  92%[=================>  ]  74.31M  53.3KB/s    eta 76s    

    
gz                   92%[=================>  ]  74.32M  52.6KB/s    eta 76s    

    
z                    92%[=================>  ]  74.33M  51.1KB/s    eta 75s    

    
<div class="k-default-codeblock">
```
                 92%[=================>  ]  74.35M  51.5KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  92%[=================>  ]  74.36M  50.7KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  92%[=================>  ]  74.37M  49.9KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  92%[=================>  ]  74.38M  49.0KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  92%[=================>  ]  74.40M  48.3KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  92%[=================>  ]  74.40M  45.4KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  92%[=================>  ]  74.42M  46.7KB/s    eta 75s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  92%[=================>  ]  74.43M  46.8KB/s    eta 74s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  92%[=================>  ]  74.44M  44.9KB/s    eta 74s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  92%[=================>  ]  74.45M  45.1KB/s    eta 74s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  92%[=================>  ]  74.47M  49.2KB/s    eta 74s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  92%[=================>  ]  74.48M  47.7KB/s    eta 74s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  92%[=================>  ]  74.49M  45.5KB/s    eta 73s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  92%[=================>  ]  74.50M  46.9KB/s    eta 73s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  92%[=================>  ]  74.52M  46.9KB/s    eta 73s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  92%[=================>  ]  74.53M  46.3KB/s    eta 73s    

```
</div>
    
   datasets/aclImdb  92%[=================>  ]  74.54M  47.1KB/s    eta 73s    

    
  datasets/aclImdb_  92%[=================>  ]  74.56M  47.5KB/s    eta 73s    

    
 datasets/aclImdb_v  92%[=================>  ]  74.57M  46.9KB/s    eta 73s    

    
datasets/aclImdb_v1  92%[=================>  ]  74.58M  49.5KB/s    eta 73s    

    
atasets/aclImdb_v1.  92%[=================>  ]  74.59M  48.0KB/s    eta 72s    

    
tasets/aclImdb_v1.t  92%[=================>  ]  74.61M  48.6KB/s    eta 72s    

    
asets/aclImdb_v1.ta  93%[=================>  ]  74.63M  49.8KB/s    eta 72s    

    
sets/aclImdb_v1.tar  93%[=================>  ]  74.63M  48.1KB/s    eta 72s    

    
ets/aclImdb_v1.tar.  93%[=================>  ]  74.65M  48.6KB/s    eta 71s    

    
ts/aclImdb_v1.tar.g  93%[=================>  ]  74.66M  51.9KB/s    eta 71s    

    
s/aclImdb_v1.tar.gz  93%[=================>  ]  74.67M  50.5KB/s    eta 71s    

    
/aclImdb_v1.tar.gz   93%[=================>  ]  74.69M  50.9KB/s    eta 71s    

    
aclImdb_v1.tar.gz    93%[=================>  ]  74.70M  51.3KB/s    eta 71s    

    
clImdb_v1.tar.gz     93%[=================>  ]  74.71M  51.6KB/s    eta 71s    

    
lImdb_v1.tar.gz      93%[=================>  ]  74.73M  51.3KB/s    eta 71s    

    
Imdb_v1.tar.gz       93%[=================>  ]  74.74M  50.8KB/s    eta 71s    

    
mdb_v1.tar.gz        93%[=================>  ]  74.75M  53.7KB/s    eta 70s    

    
db_v1.tar.gz         93%[=================>  ]  74.77M  52.7KB/s    eta 70s    

    
b_v1.tar.gz          93%[=================>  ]  74.78M  53.0KB/s    eta 70s    

    
_v1.tar.gz           93%[=================>  ]  74.79M  53.6KB/s    eta 70s    

    
v1.tar.gz            93%[=================>  ]  74.81M  52.4KB/s    eta 69s    

    
1.tar.gz             93%[=================>  ]  74.82M  52.9KB/s    eta 69s    

    
.tar.gz              93%[=================>  ]  74.84M  52.9KB/s    eta 69s    

    
tar.gz               93%[=================>  ]  74.85M  54.3KB/s    eta 69s    

    
ar.gz                93%[=================>  ]  74.87M  55.2KB/s    eta 69s    

    
r.gz                 93%[=================>  ]  74.89M  55.5KB/s    eta 68s    

    
.gz                  93%[=================>  ]  74.92M  59.1KB/s    eta 68s    

    
gz                   93%[=================>  ]  74.94M  60.2KB/s    eta 68s    

    
z                    93%[=================>  ]  74.94M  59.6KB/s    eta 68s    

    
<div class="k-default-codeblock">
```
                 93%[=================>  ]  74.97M  60.5KB/s    eta 67s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  93%[=================>  ]  75.00M  64.9KB/s    eta 67s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  93%[=================>  ]  75.02M  65.6KB/s    eta 67s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  93%[=================>  ]  75.03M  66.6KB/s    eta 67s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  93%[=================>  ]  75.05M  68.5KB/s    eta 67s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  93%[=================>  ]  75.07M  67.6KB/s    eta 66s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  93%[=================>  ]  75.08M  68.9KB/s    eta 66s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  93%[=================>  ]  75.10M  69.5KB/s    eta 66s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  93%[=================>  ]  75.12M  70.9KB/s    eta 66s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  93%[=================>  ]  75.13M  70.9KB/s    eta 65s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  93%[=================>  ]  75.15M  70.6KB/s    eta 65s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  93%[=================>  ]  75.17M  70.3KB/s    eta 65s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  93%[=================>  ]  75.19M  71.4KB/s    eta 65s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  93%[=================>  ]  75.21M  69.8KB/s    eta 64s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  93%[=================>  ]  75.23M  70.4KB/s    eta 64s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  93%[=================>  ]  75.25M  73.4KB/s    eta 64s    

```
</div>
    
   datasets/aclImdb  93%[=================>  ]  75.27M  71.1KB/s    eta 64s    

    
  datasets/aclImdb_  93%[=================>  ]  75.30M  74.0KB/s    eta 63s    

    
 datasets/aclImdb_v  93%[=================>  ]  75.33M  77.6KB/s    eta 63s    

    
datasets/aclImdb_v1  93%[=================>  ]  75.37M  82.3KB/s    eta 63s    

    
atasets/aclImdb_v1.  93%[=================>  ]  75.39M  81.3KB/s    eta 63s    

    
tasets/aclImdb_v1.t  94%[=================>  ]  75.45M  91.1KB/s    eta 61s    

    
asets/aclImdb_v1.ta  94%[=================>  ]  75.47M  93.5KB/s    eta 61s    

    
sets/aclImdb_v1.tar  94%[=================>  ]  75.50M  98.0KB/s    eta 61s    

    
ets/aclImdb_v1.tar.  94%[=================>  ]  75.53M   103KB/s    eta 61s    

    
ts/aclImdb_v1.tar.g  94%[=================>  ]  75.56M   108KB/s    eta 61s    

    
s/aclImdb_v1.tar.gz  94%[=================>  ]  75.60M   113KB/s    eta 59s    

    
/aclImdb_v1.tar.gz   94%[=================>  ]  75.62M   111KB/s    eta 59s    

    
aclImdb_v1.tar.gz    94%[=================>  ]  75.67M   115KB/s    eta 59s    

    
clImdb_v1.tar.gz     94%[=================>  ]  75.71M   120KB/s    eta 58s    

    
lImdb_v1.tar.gz      94%[=================>  ]  75.73M   120KB/s    eta 58s    

    
Imdb_v1.tar.gz       94%[=================>  ]  75.75M   119KB/s    eta 58s    

    
mdb_v1.tar.gz        94%[=================>  ]  75.77M   117KB/s    eta 58s    

    
db_v1.tar.gz         94%[=================>  ]  75.79M   114KB/s    eta 57s    

    
b_v1.tar.gz          94%[=================>  ]  75.82M   118KB/s    eta 57s    

    
_v1.tar.gz           94%[=================>  ]  75.84M   109KB/s    eta 57s    

    
v1.tar.gz            94%[=================>  ]  75.86M   106KB/s    eta 57s    

    
1.tar.gz             94%[=================>  ]  75.88M   105KB/s    eta 56s    

    
.tar.gz              94%[=================>  ]  75.90M   101KB/s    eta 56s    

    
tar.gz               94%[=================>  ]  75.93M  97.0KB/s    eta 56s    

    
ar.gz                94%[=================>  ]  75.95M  91.8KB/s    eta 56s    

    
r.gz                 94%[=================>  ]  75.97M  97.7KB/s    eta 54s    

    
.gz                  94%[=================>  ]  75.98M  83.8KB/s    eta 54s    

    
gz                   94%[=================>  ]  76.02M  83.7KB/s    eta 54s    

    
z                    94%[=================>  ]  76.04M  85.7KB/s    eta 54s    

    
<div class="k-default-codeblock">
```
                 94%[=================>  ]  76.07M  87.3KB/s    eta 53s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  94%[=================>  ]  76.09M  86.8KB/s    eta 53s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  94%[=================>  ]  76.11M  87.8KB/s    eta 53s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  94%[=================>  ]  76.13M  88.1KB/s    eta 53s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  94%[=================>  ]  76.16M  88.6KB/s    eta 53s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  94%[=================>  ]  76.18M  88.5KB/s    eta 52s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  94%[=================>  ]  76.21M  82.1KB/s    eta 52s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  95%[==================> ]  76.25M  87.2KB/s    eta 52s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  95%[==================> ]  76.26M  83.1KB/s    eta 51s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  95%[==================> ]  76.29M  85.8KB/s    eta 51s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  95%[==================> ]  76.30M  84.0KB/s    eta 51s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  95%[==================> ]  76.32M  83.6KB/s    eta 51s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  95%[==================> ]  76.33M  81.3KB/s    eta 50s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  95%[==================> ]  76.35M  86.6KB/s    eta 50s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  95%[==================> ]  76.37M  80.2KB/s    eta 50s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  95%[==================> ]  76.38M  78.8KB/s    eta 50s    

```
</div>
    
   datasets/aclImdb  95%[==================> ]  76.40M  76.8KB/s    eta 49s    

    
  datasets/aclImdb_  95%[==================> ]  76.40M  70.4KB/s    eta 49s    

    
 datasets/aclImdb_v  95%[==================> ]  76.43M  70.6KB/s    eta 49s    

    
datasets/aclImdb_v1  95%[==================> ]  76.44M  69.4KB/s    eta 49s    

    
atasets/aclImdb_v1.  95%[==================> ]  76.46M  69.0KB/s    eta 48s    

    
tasets/aclImdb_v1.t  95%[==================> ]  76.48M  66.5KB/s    eta 48s    

    
asets/aclImdb_v1.ta  95%[==================> ]  76.49M  65.5KB/s    eta 48s    

    
sets/aclImdb_v1.tar  95%[==================> ]  76.51M  65.5KB/s    eta 48s    

    
ets/aclImdb_v1.tar.  95%[==================> ]  76.53M  63.3KB/s    eta 47s    

    
ts/aclImdb_v1.tar.g  95%[==================> ]  76.54M  68.5KB/s    eta 47s    

    
s/aclImdb_v1.tar.gz  95%[==================> ]  76.56M  62.3KB/s    eta 47s    

    
/aclImdb_v1.tar.gz   95%[==================> ]  76.58M  64.3KB/s    eta 47s    

    
aclImdb_v1.tar.gz    95%[==================> ]  76.59M  61.7KB/s    eta 46s    

    
clImdb_v1.tar.gz     95%[==================> ]  76.61M  61.9KB/s    eta 46s    

    
lImdb_v1.tar.gz      95%[==================> ]  76.62M  61.4KB/s    eta 46s    

    
Imdb_v1.tar.gz       95%[==================> ]  76.64M  63.0KB/s    eta 46s    

    
mdb_v1.tar.gz        95%[==================> ]  76.66M  63.7KB/s    eta 46s    

    
db_v1.tar.gz         95%[==================> ]  76.68M  64.1KB/s    eta 46s    

    
b_v1.tar.gz          95%[==================> ]  76.71M  66.1KB/s    eta 46s    

    
_v1.tar.gz           95%[==================> ]  76.73M  67.2KB/s    eta 46s    

    
v1.tar.gz            95%[==================> ]  76.75M  69.6KB/s    eta 46s    

    
1.tar.gz             95%[==================> ]  76.78M  75.6KB/s    eta 44s    

    
.tar.gz              95%[==================> ]  76.81M  75.5KB/s    eta 44s    

    
tar.gz               95%[==================> ]  76.84M  79.1KB/s    eta 44s    

    
ar.gz                95%[==================> ]  76.88M  83.4KB/s    eta 44s    

    
r.gz                 95%[==================> ]  76.92M  88.9KB/s    eta 42s    

    
.gz                  95%[==================> ]  76.97M  95.8KB/s    eta 42s    

    
gz                   96%[==================> ]  77.03M   103KB/s    eta 42s    

    
z                    96%[==================> ]  77.09M   112KB/s    eta 42s    

    
<div class="k-default-codeblock">
```
                 96%[==================> ]  77.16M   123KB/s    eta 39s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  96%[==================> ]  77.24M   135KB/s    eta 39s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  96%[==================> ]  77.25M   134KB/s    eta 39s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  96%[==================> ]  77.38M   156KB/s    eta 39s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  96%[==================> ]  77.45M   166KB/s    eta 35s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  96%[==================> ]  77.51M   178KB/s    eta 35s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  96%[==================> ]  77.59M   189KB/s    eta 35s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  96%[==================> ]  77.66M   199KB/s    eta 35s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  96%[==================> ]  77.68M   198KB/s    eta 32s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  96%[==================> ]  77.73M   203KB/s    eta 32s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  97%[==================> ]  77.84M   221KB/s    eta 32s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  97%[==================> ]  77.90M   227KB/s    eta 32s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  97%[==================> ]  77.96M   238KB/s    eta 32s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  97%[==================> ]  78.02M   245KB/s    eta 28s    

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  97%[==================> ]  78.03M   235KB/s    eta 28s    

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  97%[==================> ]  78.12M   246KB/s    eta 28s    

```
</div>
    
   datasets/aclImdb  97%[==================> ]  78.16M   246KB/s    eta 28s    

    
  datasets/aclImdb_  97%[==================> ]  78.21M   246KB/s    eta 25s    

    
 datasets/aclImdb_v  97%[==================> ]  78.25M   242KB/s    eta 25s    

    
datasets/aclImdb_v1  97%[==================> ]  78.29M   225KB/s    eta 25s    

    
atasets/aclImdb_v1.  97%[==================> ]  78.35M   241KB/s    eta 25s    

    
tasets/aclImdb_v1.t  97%[==================> ]  78.38M   221KB/s    eta 23s    

    
asets/aclImdb_v1.ta  97%[==================> ]  78.42M   213KB/s    eta 23s    

    
sets/aclImdb_v1.tar  97%[==================> ]  78.45M   204KB/s    eta 23s    

    
ets/aclImdb_v1.tar.  97%[==================> ]  78.47M   189KB/s    eta 23s    

    
ts/aclImdb_v1.tar.g  97%[==================> ]  78.53M   190KB/s    eta 21s    

    
s/aclImdb_v1.tar.gz  97%[==================> ]  78.55M   192KB/s    eta 21s    

    
/aclImdb_v1.tar.gz   97%[==================> ]  78.58M   178KB/s    eta 21s    

    
aclImdb_v1.tar.gz    98%[==================> ]  78.63M   163KB/s    eta 20s    

    
clImdb_v1.tar.gz     98%[==================> ]  78.65M   149KB/s    eta 20s    

    
lImdb_v1.tar.gz      98%[==================> ]  78.66M   140KB/s    eta 20s    

    
Imdb_v1.tar.gz       98%[==================> ]  78.67M   136KB/s    eta 20s    

    
mdb_v1.tar.gz        98%[==================> ]  78.68M   126KB/s    eta 20s    

    
db_v1.tar.gz         98%[==================> ]  78.69M   128KB/s    eta 20s    

    
b_v1.tar.gz          98%[==================> ]  78.70M   112KB/s    eta 20s    

    
_v1.tar.gz           98%[==================> ]  78.71M   105KB/s    eta 20s    

    
v1.tar.gz            98%[==================> ]  78.72M   100KB/s    eta 19s    

    
1.tar.gz             98%[==================> ]  78.73M  94.3KB/s    eta 19s    

    
.tar.gz              98%[==================> ]  78.74M  89.0KB/s    eta 19s    

    
tar.gz               98%[==================> ]  78.76M  88.3KB/s    eta 19s    

    
ar.gz                98%[==================> ]  78.77M  86.3KB/s    eta 18s    

    
r.gz                 98%[==================> ]  78.79M  73.9KB/s    eta 18s    

    
.gz                  98%[==================> ]  78.82M  71.0KB/s    eta 18s    

    
gz                   98%[==================> ]  78.84M  64.0KB/s    eta 18s    

    
z                    98%[==================> ]  78.87M  64.0KB/s    eta 17s    

    
<div class="k-default-codeblock">
```
                 98%[==================> ]  78.90M  61.4KB/s    eta 17s    

```
</div>
    
<div class="k-default-codeblock">
```
              d  98%[==================> ]  78.93M  67.0KB/s    eta 17s    

```
</div>
    
<div class="k-default-codeblock">
```
             da  98%[==================> ]  78.96M  71.9KB/s    eta 17s    

```
</div>
    
<div class="k-default-codeblock">
```
            dat  98%[==================> ]  79.01M  81.8KB/s    eta 17s    

```
</div>
    
<div class="k-default-codeblock">
```
           data  98%[==================> ]  79.05M  90.4KB/s    eta 15s    

```
</div>
    
<div class="k-default-codeblock">
```
          datas  98%[==================> ]  79.09M   101KB/s    eta 15s    

```
</div>
    
<div class="k-default-codeblock">
```
         datase  98%[==================> ]  79.12M   106KB/s    eta 15s    

```
</div>
    
<div class="k-default-codeblock">
```
        dataset  98%[==================> ]  79.16M   117KB/s    eta 15s    

```
</div>
    
<div class="k-default-codeblock">
```
       datasets  98%[==================> ]  79.22M   134KB/s    eta 15s    

```
</div>
    
<div class="k-default-codeblock">
```
      datasets/  98%[==================> ]  79.29M   148KB/s    eta 12s    

```
</div>
    
<div class="k-default-codeblock">
```
     datasets/a  98%[==================> ]  79.36M   166KB/s    eta 12s    

```
</div>
    
<div class="k-default-codeblock">
```
    datasets/ac  98%[==================> ]  79.40M   162KB/s    eta 12s    

```
</div>
    
<div class="k-default-codeblock">
```
   datasets/acl  99%[==================> ]  79.55M   193KB/s    eta 12s    

```
</div>
    
<div class="k-default-codeblock">
```
  datasets/aclI  99%[==================> ]  79.61M   209KB/s    eta 8s     

```
</div>
    
<div class="k-default-codeblock">
```
 datasets/aclIm  99%[==================> ]  79.67M   218KB/s    eta 8s     

```
</div>
    
<div class="k-default-codeblock">
```
datasets/aclImd  99%[==================> ]  79.73M   226KB/s    eta 8s     

```
</div>
    
   datasets/aclImdb  99%[==================> ]  79.74M   216KB/s    eta 8s     

    
  datasets/aclImdb_  99%[==================> ]  79.86M   228KB/s    eta 5s     

    
 datasets/aclImdb_v  99%[==================> ]  79.95M   240KB/s    eta 5s     

    
datasets/aclImdb_v1  99%[==================> ]  79.99M   244KB/s    eta 5s     

    
atasets/aclImdb_v1.  99%[==================> ]  80.04M   244KB/s    eta 5s     

    
tasets/aclImdb_v1.t  99%[==================> ]  80.08M   246KB/s    eta 5s     

    
asets/aclImdb_v1.ta  99%[==================> ]  80.12M   241KB/s    eta 1s     

    
sets/aclImdb_v1.tar  99%[==================> ]  80.17M   233KB/s    eta 1s     

    
ets/aclImdb_v1.tar.  99%[==================> ]  80.22M   230KB/s    eta 1s     
datasets/aclImdb_v1 100%[===================>]  80.23M   231KB/s    in 16m 42s 
    
<div class="k-default-codeblock">
```
2024-05-20 10:38:01 (82.0 KB/s) - ‘datasets/aclImdb_v1.tar.gz’ saved [84125825/84125825]
```
</div>
    


We'll use the `keras.utils.text_dataset_from_directory` utility to generate our
labelled `tf.data.Dataset` dataset from text files.


```python
train_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/test", batch_size=BATCH_SIZE
)
```

<div class="k-default-codeblock">
```
Found 25000 files belonging to 2 classes.

Using 20000 files for training.

Found 25000 files belonging to 2 classes.

Using 5000 files for validation.

Found 25000 files belonging to 2 classes.

```
</div>
We will now convert the text to lowercase.


```python
train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))
```

Let's print a few samples.


```python
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f"Text: {text_batch.numpy()[i]}")
        print(f"Label: {label_batch.numpy()[i]}")
```

<div class="k-default-codeblock">
```
Text: b'"pandemonium" is a horror movie spoof that comes off more stupid than funny. believe me when i tell you, i love comedies. especially comedy spoofs. "airplane", "the naked gun" trilogy, "blazing saddles", "high anxiety", and "spaceballs" are some of my favorite comedies that spoof a particular genre. "pandemonium" is not up there with those films. most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\'t all that funny. there are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\'s all this film has going for it. geez, "scream" had more laughs than this film and that was more of a horror film. how bizarre is that?<br /><br />*1/2 (out of four)'
Label: 0
Text: b"david mamet is a very interesting and a very un-equal director. his first movie 'house of games' was the one i liked best, and it set a series of films with characters whose perspective of life changes as they get into complicated situations, and so does the perspective of the viewer.<br /><br />so is 'homicide' which from the title tries to set the mind of the viewer to the usual crime drama. the principal characters are two cops, one jewish and one irish who deal with a racially charged area. the murder of an old jewish shop owner who proves to be an ancient veteran of the israeli independence war triggers the jewish identity in the mind and heart of the jewish detective.<br /><br />this is were the flaws of the film are the more obvious. the process of awakening is theatrical and hard to believe, the group of jewish militants is operatic, and the way the detective eventually walks to the final violent confrontation is pathetic. the end of the film itself is mamet-like smart, but disappoints from a human emotional perspective.<br /><br />joe mantegna and william macy give strong performances, but the flaws of the story are too evident to be easily compensated."
Label: 0
Text: b'great documentary about the lives of ny firefighters during the worst terrorist attack of all time.. that reason alone is why this should be a must see collectors item.. what shocked me was not only the attacks, but the"high fat diet" and physical appearance of some of these firefighters. i think a lot of doctors would agree with me that,in the physical shape they were in, some of these firefighters would not of made it to the 79th floor carrying over 60 lbs of gear. having said that i now have a greater respect for firefighters and i realize becoming a firefighter is a life altering job. the french have a history of making great documentary\'s and that is what this is, a great documentary.....'
Label: 1

```
</div>
### Tokenizing the data

We'll be using the `keras_nlp.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_nlp.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization
algorithm; training it on a corpus gives us a vocabulary of subwords. A subword
tokenizer is a compromise between word tokenizers (word tokenizers need very
large vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, KerasNLP
makes it very simple to train WordPiece on a corpus with the
`keras_nlp.tokenizers.compute_word_piece_vocabulary` utility.


```python

def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

```

Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence
length when the input sequence length is shorter than the maximum sequence
length.
- `"[UNK]"` - Unknown token.


```python
reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
vocab = train_word_piece(train_ds, VOCABULARY_SIZE, reserved_tokens)
```

Let's see some tokens!


```python
print("Tokens: ", vocab[100:110])
```

<div class="k-default-codeblock">
```
Tokens:  ['à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é']

```
</div>
Now, let's define the tokenizer. We will configure the tokenizer with the
the vocabularies trained above. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.


```python
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)
```

Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.


```python
input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))
```

<div class="k-default-codeblock">
```
Sentence:  tf.Tensor(b'great movie - especially the music - etta james - "at last". this speaks volumes when you have finally found that special someone.', shape=(), dtype=string)
Tokens:  

[  218   150    14   393   137   356    14  4917  2941   719    14     3
   164   370     3    15   145  2705 11670   186   155   160   557   391
   146   452   416    15     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0]
Recovered text after detokenizing:  tf.Tensor(b'great movie - especially the music - etta james - " at last " . this speaks volumes when you have finally found that special someone . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]', shape=(), dtype=string)

```
</div>
---
## Formatting the dataset

Next, we'll format our datasets in the form that will be fed to the models. We
need to tokenize the text.


```python

def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(tf.data.AUTOTUNE).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)

```

---
## Model

Let's build a simple Transformer model. We will use `TokenAndPositionEmbedding`
and `TransformerDecoder` from KerasNLP library. `TokenAndPositionEmbedding`
represents words and their order in a sentence, while `TransformerDecoder`
outputs one vector for each time step of our input sequence. Here, we take the
mean across all time steps and use a feedforward network on top of it to
classify text.


```python

def build_model(
    vocabulary_size=20000,
    max_sequence_length=200,
    hidden_dim=32,
    num_heads=2,
    intermediate_dim=32,
    dropout=0.1,
):
    token_id_input = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        embedding_dim=hidden_dim,
    )(token_id_input)
    x = keras.layers.Dropout(rate=dropout)(x)
    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=dropout,
    )(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(intermediate_dim, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=token_id_input, outputs=outputs)

```

---
## Training and evaluating our model

First, we train and evaluate the model with mixed precision
(`"mixed_bfloat16"`). Afterward, we compare the results with FP8
training/inference.


```python
model = build_model(**MODEL_KWARGS)
model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (mixed_bfloat16): {result[1]:.2%}")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ token_and_position_embedding    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">646,400</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbedding</span>)     │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ transformer_decoder             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">6,464</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerDecoder</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">653,953</span> (2.49 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">653,953</span> (2.49 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  26:04 3s/step - accuracy: 0.5938 - loss: 0.7035

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  11:06 1s/step - accuracy: 0.5078 - loss: 0.7196

<div class="k-default-codeblock">
```

```
</div>
  56/625 ━[37m━━━━━━━━━━━━━━━━━━━  11s 20ms/step - accuracy: 0.5148 - loss: 0.6960

<div class="k-default-codeblock">
```

```
</div>
 113/625 ━━━[37m━━━━━━━━━━━━━━━━━  5s 10ms/step - accuracy: 0.5447 - loss: 0.6829 

<div class="k-default-codeblock">
```

```
</div>
 172/625 ━━━━━[37m━━━━━━━━━━━━━━━  3s 7ms/step - accuracy: 0.5725 - loss: 0.6647 

<div class="k-default-codeblock">
```

```
</div>
 232/625 ━━━━━━━[37m━━━━━━━━━━━━━  2s 6ms/step - accuracy: 0.5983 - loss: 0.6437

<div class="k-default-codeblock">
```

```
</div>
 296/625 ━━━━━━━━━[37m━━━━━━━━━━━  1s 4ms/step - accuracy: 0.6207 - loss: 0.6231

<div class="k-default-codeblock">
```

```
</div>
 361/625 ━━━━━━━━━━━[37m━━━━━━━━━  1s 4ms/step - accuracy: 0.6386 - loss: 0.6056

<div class="k-default-codeblock">
```

```
</div>
 426/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 3ms/step - accuracy: 0.6533 - loss: 0.5908

<div class="k-default-codeblock">
```

```
</div>
 490/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - accuracy: 0.6655 - loss: 0.5780

<div class="k-default-codeblock">
```

```
</div>
 552/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 3ms/step - accuracy: 0.6759 - loss: 0.5668

<div class="k-default-codeblock">
```

```
</div>
 618/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - accuracy: 0.6855 - loss: 0.5562

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - accuracy: 0.6866 - loss: 0.5550 - val_accuracy: 0.8204 - val_loss: 0.3901


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  11:15 1s/step - accuracy: 0.8750 - loss: 0.2920

<div class="k-default-codeblock">
```

```
</div>
  59/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 870us/step - accuracy: 0.8755 - loss: 0.3043

<div class="k-default-codeblock">
```

```
</div>
 119/625 ━━━[37m━━━━━━━━━━━━━━━━━  0s 857us/step - accuracy: 0.8755 - loss: 0.3023

<div class="k-default-codeblock">
```

```
</div>
 180/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 848us/step - accuracy: 0.8731 - loss: 0.3042

<div class="k-default-codeblock">
```

```
</div>
 241/625 ━━━━━━━[37m━━━━━━━━━━━━━  0s 842us/step - accuracy: 0.8739 - loss: 0.3013

<div class="k-default-codeblock">
```

```
</div>
 302/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 840us/step - accuracy: 0.8760 - loss: 0.2970

<div class="k-default-codeblock">
```

```
</div>
 367/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 829us/step - accuracy: 0.8781 - loss: 0.2932

<div class="k-default-codeblock">
```

```
</div>
 432/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 820us/step - accuracy: 0.8800 - loss: 0.2899

<div class="k-default-codeblock">
```

```
</div>
 496/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 815us/step - accuracy: 0.8815 - loss: 0.2875

<div class="k-default-codeblock">
```

```
</div>
 561/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 810us/step - accuracy: 0.8828 - loss: 0.2854

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 923us/step - accuracy: 0.8840 - loss: 0.2835 - val_accuracy: 0.8078 - val_loss: 0.4478


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 0.9062 - loss: 0.1977

<div class="k-default-codeblock">
```

```
</div>
  66/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 779us/step - accuracy: 0.9125 - loss: 0.2358

<div class="k-default-codeblock">
```

```
</div>
 130/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 783us/step - accuracy: 0.9128 - loss: 0.2306

<div class="k-default-codeblock">
```

```
</div>
 194/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 784us/step - accuracy: 0.9094 - loss: 0.2338

<div class="k-default-codeblock">
```

```
</div>
 258/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 785us/step - accuracy: 0.9096 - loss: 0.2315

<div class="k-default-codeblock">
```

```
</div>
 322/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 785us/step - accuracy: 0.9108 - loss: 0.2284

<div class="k-default-codeblock">
```

```
</div>
 386/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 785us/step - accuracy: 0.9124 - loss: 0.2253

<div class="k-default-codeblock">
```

```
</div>
 452/625 ━━━━━━━━━━━━━━[37m━━━━━━  0s 783us/step - accuracy: 0.9140 - loss: 0.2223

<div class="k-default-codeblock">
```

```
</div>
 517/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 781us/step - accuracy: 0.9151 - loss: 0.2203

<div class="k-default-codeblock">
```

```
</div>
 583/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 780us/step - accuracy: 0.9159 - loss: 0.2188

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 890us/step - accuracy: 0.9163 - loss: 0.2179 - val_accuracy: 0.8388 - val_loss: 0.4165


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  5:01 386ms/step - accuracy: 0.8750 - loss: 0.4444

<div class="k-default-codeblock">
```

```
</div>
  85/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 597us/step - accuracy: 0.8026 - loss: 0.5145  

<div class="k-default-codeblock">
```

```
</div>
 171/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 592us/step - accuracy: 0.8048 - loss: 0.5078

<div class="k-default-codeblock">
```

```
</div>
 253/782 ━━━━━━[37m━━━━━━━━━━━━━━  0s 598us/step - accuracy: 0.8066 - loss: 0.5026

<div class="k-default-codeblock">
```

```
</div>
 370/782 ━━━━━━━━━[37m━━━━━━━━━━━  0s 544us/step - accuracy: 0.8082 - loss: 0.4982

<div class="k-default-codeblock">
```

```
</div>
 483/782 ━━━━━━━━━━━━[37m━━━━━━━━  0s 520us/step - accuracy: 0.8094 - loss: 0.4955

<div class="k-default-codeblock">
```

```
</div>
 600/782 ━━━━━━━━━━━━━━━[37m━━━━━  0s 503us/step - accuracy: 0.8101 - loss: 0.4936

<div class="k-default-codeblock">
```

```
</div>
 712/782 ━━━━━━━━━━━━━━━━━━[37m━━  0s 494us/step - accuracy: 0.8104 - loss: 0.4934

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 1s 491us/step - accuracy: 0.8105 - loss: 0.4934


<div class="k-default-codeblock">
```
Accuracy (mixed_bfloat16): 81.16%

```
</div>
We can enable FP8 training/inference with a one-line API:
`model.quantize("float8")`.


```python
model = build_model(**MODEL_KWARGS)
model.quantize("float8")
```

<div class="k-default-codeblock">
```
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer InputLayer does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer PositionEmbedding does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer ReversibleEmbedding does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer Dropout does not have a `quantize()` method implemented.
  warnings.warn(str(e))
/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer LayerNormalization does not have a `quantize()` method implemented.
  warnings.warn(str(e))

/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer Softmax does not have a `quantize()` method implemented.
  warnings.warn(str(e))

/home/hongyu/miniconda3/envs/kimm/lib/python3.11/site-packages/keras/src/models/model.py:381: UserWarning: Layer GlobalAveragePooling1D does not have a `quantize()` method implemented.
  warnings.warn(str(e))

```
</div>
To inspect that FP8 training takes place, we can print out some variables
related to FP8 training:

- `*_scale`: The scaling factor that shift the distribution of inputs, weights
    and gradients into the representable range of FP8. Defaults to `1.0`
- `*_amax_history`: The amax history window used for scaling factor computation.
    Defaults to `0.0` with the length of 1024.


```python
pattern = r"(transformer).+(multi_head).+(query).+(scale|amax_history)"
for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))
```

The dtype policies of FP8 layers have also been modified.


```python
for layer in model._flatten_layers(recursive=True):
    if "float8" in str(layer.dtype_policy):
        print(f"{layer.name}: {layer.dtype_policy}")
```

<div class="k-default-codeblock">
```
feedforward_output_dense: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
feedforward_intermediate_dense: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
attention_output: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
value: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
key: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
query: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
dense_2: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">
dense_3: <QuantizedFloat8DTypePolicy "float8_from_mixed_bfloat16">

```
</div>
Let's train the model and see the results. We can verify that the accuracy
doesn't decrease with FP8 training that the variables containing FP8 information
change after fitting.


```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (float8): {result[1]:.2%}")

for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))
```

<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  25:46 2s/step - accuracy: 0.4062 - loss: 0.7810

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  16:26 2s/step - accuracy: 0.4844 - loss: 0.7454

<div class="k-default-codeblock">
```

```
</div>
  43/625 ━[37m━━━━━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.5172 - loss: 0.7026

<div class="k-default-codeblock">
```

```
</div>
  85/625 ━━[37m━━━━━━━━━━━━━━━━━━  10s 20ms/step - accuracy: 0.5261 - loss: 0.6979

<div class="k-default-codeblock">
```

```
</div>
 127/625 ━━━━[37m━━━━━━━━━━━━━━━━  6s 14ms/step - accuracy: 0.5379 - loss: 0.6923 

<div class="k-default-codeblock">
```

```
</div>
 169/625 ━━━━━[37m━━━━━━━━━━━━━━━  4s 11ms/step - accuracy: 0.5548 - loss: 0.6818

<div class="k-default-codeblock">
```

```
</div>
 211/625 ━━━━━━[37m━━━━━━━━━━━━━━  3s 9ms/step - accuracy: 0.5726 - loss: 0.6683 

<div class="k-default-codeblock">
```

```
</div>
 253/625 ━━━━━━━━[37m━━━━━━━━━━━━  2s 7ms/step - accuracy: 0.5894 - loss: 0.6536

<div class="k-default-codeblock">
```

```
</div>
 297/625 ━━━━━━━━━[37m━━━━━━━━━━━  2s 7ms/step - accuracy: 0.6045 - loss: 0.6398

<div class="k-default-codeblock">
```

```
</div>
 342/625 ━━━━━━━━━━[37m━━━━━━━━━━  1s 6ms/step - accuracy: 0.6179 - loss: 0.6270

<div class="k-default-codeblock">
```

```
</div>
 386/625 ━━━━━━━━━━━━[37m━━━━━━━━  1s 5ms/step - accuracy: 0.6294 - loss: 0.6157

<div class="k-default-codeblock">
```

```
</div>
 431/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 5ms/step - accuracy: 0.6398 - loss: 0.6052

<div class="k-default-codeblock">
```

```
</div>
 475/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - accuracy: 0.6489 - loss: 0.5958

<div class="k-default-codeblock">
```

```
</div>
 520/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 4ms/step - accuracy: 0.6574 - loss: 0.5869

<div class="k-default-codeblock">
```

```
</div>
 565/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - accuracy: 0.6651 - loss: 0.5786

<div class="k-default-codeblock">
```

```
</div>
 610/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 4ms/step - accuracy: 0.6722 - loss: 0.5709

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 6s 5ms/step - accuracy: 0.6745 - loss: 0.5684 - val_accuracy: 0.8140 - val_loss: 0.4036


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  16:12 2s/step - accuracy: 0.7812 - loss: 0.3850

<div class="k-default-codeblock">
```

```
</div>
  41/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8619 - loss: 0.3272  

<div class="k-default-codeblock">
```

```
</div>
  83/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8637 - loss: 0.3236

<div class="k-default-codeblock">
```

```
</div>
 125/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8636 - loss: 0.3212

<div class="k-default-codeblock">
```

```
</div>
 167/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8641 - loss: 0.3186

<div class="k-default-codeblock">
```

```
</div>
 209/625 ━━━━━━[37m━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8659 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 251/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8686 - loss: 0.3091

<div class="k-default-codeblock">
```

```
</div>
 295/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.8710 - loss: 0.3045

<div class="k-default-codeblock">
```

```
</div>
 340/625 ━━━━━━━━━━[37m━━━━━━━━━━  0s 1ms/step - accuracy: 0.8732 - loss: 0.3009

<div class="k-default-codeblock">
```

```
</div>
 385/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.8752 - loss: 0.2976

<div class="k-default-codeblock">
```

```
</div>
 430/625 ━━━━━━━━━━━━━[37m━━━━━━━  0s 1ms/step - accuracy: 0.8769 - loss: 0.2948

<div class="k-default-codeblock">
```

```
</div>
 475/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.8784 - loss: 0.2924

<div class="k-default-codeblock">
```

```
</div>
 518/625 ━━━━━━━━━━━━━━━━[37m━━━━  0s 1ms/step - accuracy: 0.8795 - loss: 0.2906

<div class="k-default-codeblock">
```

```
</div>
 562/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.8805 - loss: 0.2890

<div class="k-default-codeblock">
```

```
</div>
 607/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 1ms/step - accuracy: 0.8814 - loss: 0.2874

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.8818 - loss: 0.2869 - val_accuracy: 0.8300 - val_loss: 0.3832


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  2s 3ms/step - accuracy: 0.9062 - loss: 0.2026

<div class="k-default-codeblock">
```

```
</div>
  45/625 ━[37m━━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9078 - loss: 0.2419

<div class="k-default-codeblock">
```

```
</div>
  89/625 ━━[37m━━━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9122 - loss: 0.2290

<div class="k-default-codeblock">
```

```
</div>
 133/625 ━━━━[37m━━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9123 - loss: 0.2268

<div class="k-default-codeblock">
```

```
</div>
 177/625 ━━━━━[37m━━━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9115 - loss: 0.2264

<div class="k-default-codeblock">
```

```
</div>
 222/625 ━━━━━━━[37m━━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9110 - loss: 0.2257

<div class="k-default-codeblock">
```

```
</div>
 267/625 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9117 - loss: 0.2227

<div class="k-default-codeblock">
```

```
</div>
 312/625 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - accuracy: 0.9126 - loss: 0.2203

<div class="k-default-codeblock">
```

```
</div>
 357/625 ━━━━━━━━━━━[37m━━━━━━━━━  0s 1ms/step - accuracy: 0.9136 - loss: 0.2181

<div class="k-default-codeblock">
```

```
</div>
 402/625 ━━━━━━━━━━━━[37m━━━━━━━━  0s 1ms/step - accuracy: 0.9148 - loss: 0.2157

<div class="k-default-codeblock">
```

```
</div>
 447/625 ━━━━━━━━━━━━━━[37m━━━━━━  0s 1ms/step - accuracy: 0.9159 - loss: 0.2135

<div class="k-default-codeblock">
```

```
</div>
 492/625 ━━━━━━━━━━━━━━━[37m━━━━━  0s 1ms/step - accuracy: 0.9168 - loss: 0.2119

<div class="k-default-codeblock">
```

```
</div>
 537/625 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - accuracy: 0.9173 - loss: 0.2109

<div class="k-default-codeblock">
```

```
</div>
 582/625 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - accuracy: 0.9177 - loss: 0.2101

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9181 - loss: 0.2094 - val_accuracy: 0.8400 - val_loss: 0.4063


    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  1s 2ms/step - accuracy: 0.9062 - loss: 0.4228

<div class="k-default-codeblock">
```

```
</div>
  80/782 ━━[37m━━━━━━━━━━━━━━━━━━  0s 638us/step - accuracy: 0.8027 - loss: 0.5005

<div class="k-default-codeblock">
```

```
</div>
 161/782 ━━━━[37m━━━━━━━━━━━━━━━━  0s 630us/step - accuracy: 0.8043 - loss: 0.4972

<div class="k-default-codeblock">
```

```
</div>
 240/782 ━━━━━━[37m━━━━━━━━━━━━━━  0s 633us/step - accuracy: 0.8066 - loss: 0.4921

<div class="k-default-codeblock">
```

```
</div>
 321/782 ━━━━━━━━[37m━━━━━━━━━━━━  0s 631us/step - accuracy: 0.8084 - loss: 0.4878

<div class="k-default-codeblock">
```

```
</div>
 401/782 ━━━━━━━━━━[37m━━━━━━━━━━  0s 630us/step - accuracy: 0.8093 - loss: 0.4852

<div class="k-default-codeblock">
```

```
</div>
 482/782 ━━━━━━━━━━━━[37m━━━━━━━━  0s 629us/step - accuracy: 0.8100 - loss: 0.4832

<div class="k-default-codeblock">
```

```
</div>
 562/782 ━━━━━━━━━━━━━━[37m━━━━━━  0s 629us/step - accuracy: 0.8105 - loss: 0.4815

<div class="k-default-codeblock">
```

```
</div>
 637/782 ━━━━━━━━━━━━━━━━[37m━━━━  0s 634us/step - accuracy: 0.8106 - loss: 0.4809

<div class="k-default-codeblock">
```

```
</div>
 709/782 ━━━━━━━━━━━━━━━━━━[37m━━  0s 641us/step - accuracy: 0.8106 - loss: 0.4807

<div class="k-default-codeblock">
```

```
</div>
 780/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 647us/step - accuracy: 0.8105 - loss: 0.4807

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 1s 650us/step - accuracy: 0.8105 - loss: 0.4807


<div class="k-default-codeblock">
```
Accuracy (float8): 80.98%

```
</div>
---
## Recipes

- The improvements in training speed are relatively small if the model is not
sufficiently large. The recommendation is to train with a model containing
parameters >5B.
- You will need hardware such as NVIDIA H100 that supports FP8 Tensor Cores to
gain the speedups.

---
## References
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Flax - fp8_ops.py](https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py)
