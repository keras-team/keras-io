"""
Title: Sampling Techniques using KerasNLP
Author: Usha Rengaraju
Date created: 2023/07/10
Last modified: 2023/08/02
Description: Discussion of various sampling techniques using KerasNLP
"""
"""
## Overview

KerasNLP offers a wide range of sampling techniques already implemented and very easy to
use. Samplers are basically the methods of selecting the next token based on the
distribution of the previous tokens. There are various sampling techniques like `Beam
Sampler` , `Contrastive Sampler`, `Random Sampler` and many more.

This guide will show you how to use the various samplers and will also show the various
hyperparameters of these samplers and how the output will be affected by these.

"""

"""
## Imports & setup

This tutorial requires you to have KerasNLP installed:

```shell
pip install keras-nlp
```

We begin by importing all required packages:
"""

"""
## Model Loading

For this tutorial we will be using a pretrained `OPT Language Model` which is also
directed loaded from the kerasnlp.

For the output generation we take the sample input as `I recently ate`.
"""

import keras_nlp

opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_1.3b_en")

"""
By default the model uses a greedy sampler
"""

opt_lm.generate("I recently ate", max_length=50)

"""
#Beam Sampler

At each time-step, beam search keeps the beams (sequences) of the top num_beams highest
accumulated probabilities, and uses each one of the beams to predict candidate next
tokens.
"""

"""
### setting num_beams=2

"""

opt_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
opt_lm.generate("I recently ate", max_length=50)

"""
### num_beams=10
"""

opt_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=10))
opt_lm.generate("I recently ate", max_length=50)

"""
### num_beams=30

"""

opt_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=30))
opt_lm.generate("I recently ate", max_length=50)

"""
#Greedy Sampler

This sampler is implemented on greedy search, i.e., always picking up the token of the
largest probability as the next token.
"""

opt_lm.compile(sampler=keras_nlp.samplers.GreedySampler())
opt_lm.generate("I recently ate", max_length=50)

"""
#Contrastive Sampler

This sampler implements contrastive search algorithm. In short, the sampler chooses the
token having the max "score" as the next token. The "score" is a weighted sum between
token's probability and max similarity against previous tokens. By using this joint
score, contrastive sampler reduces the behavior of duplicating seen tokens.

**k**: int, the k value of top-k. Next token will be chosen from k tokens.<br>
**alpha**: float, the weight of minus max similarity in joint score computation. The
larger the value of alpha, the score relies more on the similarity than the token
probability.<br>
**seed**: int, defaults to None. The random seed.
"""

"""
### Large alpha small k
"""

opt_lm.compile(sampler=keras_nlp.samplers.ContrastiveSampler(k=2, alpha=0.9, seed=45))
opt_lm.generate("I recently ate", max_length=50)

"""
### Large alpha large k
"""

opt_lm.compile(sampler=keras_nlp.samplers.ContrastiveSampler(k=10, alpha=0.9, seed=45))
opt_lm.generate("I recently ate", max_length=50)

"""
### small alpha large k
"""

opt_lm.compile(sampler=keras_nlp.samplers.ContrastiveSampler(k=2, alpha=0.1, seed=45))
opt_lm.generate("I recently ate", max_length=50)

"""
### Small alpha small k
"""

opt_lm.compile(sampler=keras_nlp.samplers.ContrastiveSampler(k=2, alpha=0.1, seed=45))
opt_lm.generate("I recently ate", max_length=50)

"""
#Random Sampler

This sampler implements random sampling. Briefly, random sampler randomly selects a token
from the entire distribution of the tokens, with selection chance determined by the
probability of each token.

**seed**: int, defaults to None. The random seed.
"""

opt_lm.compile(sampler=keras_nlp.samplers.RandomSampler(seed=51))
opt_lm.generate("I recently ate", max_length=50)

"""
#Top K Sampler

This sampler implements top-k search algorithm. Briefly, top-k algorithm randomly selects
a token from the tokens of top K probability, with selection chance determined by the
probability.

**k**: int, the k value of top-k. Next token will be chosen from k tokens.<br>
**seed**: int, defaults to None. The random seed.
"""

"""
### k=2
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=5, seed=30))
opt_lm.generate("I recently ate", max_length=50)

"""
### k=10
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=10, seed=30))
opt_lm.generate("I recently ate", max_length=50)

"""
### k=100
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=100, seed=30))
opt_lm.generate("I recently ate", max_length=50)

"""
#Top-P Sampler

This sampler implements top-p search algorithm. Top-p search selects tokens from the
smallest subset of output probabilities that sum to greater than p. Put in another way,
top-p will first order token predictions by likelihood, and ignore all tokens after the
cumulative probability of selected tokens exceeds p, then select a token from the
remaining tokens.
<br>
<br>
**p**: float, the p value of top-p.<br>
**k**: int, defaults to None. If set, this argument defines a heuristic "top-k" cutoff
applied before the "top-p" sampling. All logits not in the top k will be discarded, and
the remaining logits will be sorted to find a cutoff point for p. Setting this arg can
significantly speed sampling up by reducing the number of tokens to sort.<br>
**seed**: int, defaults to None. The random seed.
"""

"""
### Low p and no top-k
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopPSampler(p=0.1, k=None, seed=22))
opt_lm.generate("I recently ate", max_length=50)

"""
### Low p and top-k
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopPSampler(p=0.1, k=5, seed=22))
opt_lm.generate("I recently ate", max_length=50)

"""
### High p and no top-k
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopPSampler(p=0.8, k=None, seed=22))
opt_lm.generate("I recently ate", max_length=50)

"""
### High p and top-k
"""

opt_lm.compile(sampler=keras_nlp.samplers.TopPSampler(p=0.8, k=5, seed=22))
opt_lm.generate("I recently ate", max_length=50)


"""
## References

https://keras.io/api/keras_nlp/samplers/
"""
