"""
Title: OPT2 Text Generation with KerasNLP and Keras Distribution API
Author: Qianli (Scott) Zhu
Date created: 01/11/2024
Last modified: 01/11/2024
Description: Use OPT2 model and Keras distribution API to do text generation.
Accelerator: GPU
"""

"""
In this tutorial, you will learn to use [KerasNLP](https://keras.io/keras_nlp/)
to load a pre-trained Large Language Model (LLM) - [OPT-2 model](https://arxiv.org/abs/2205.01068)
(originally invented by Meta), finetune and generate with a distribute hardware
setting.
"""

"""
##  Before we begin

Colab offers different kinds of runtimes. Make sure to go to **Runtime ->
Change runtime type** and choose the GPU Hardware Accelerator runtime
(which should have >12G host RAM and ~16G GPU RAM) since you will finetune the
OPT-2 model. Running this tutorial on CPU runtime will take hours.

Also note that this example was originally created with 8 V100 GPUs, explicitly
to simulate how to do inference of large model with limited hardwares.
"""

"""
## Install KerasNLP, Choose Backend and Import Dependencies

This examples uses the latest distribution API from [Keras](https://keras.io/keras/).
The API is currently supporting JAX backend, and we are adding more backends
support in the coming future.
"""

import os

# This will allow JAX to scale more to fully leverage all the available GPU memory.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

# We have 8 V100 GPUs, and each of which has 16G of GPU memory.
# It will not be enough memory on a single device to host all the model weights
# and optimizer state.
# We are going to show case how to distribute the large model weights, so that
# a popular LLM model (7B param) can be finetuned on a previous generation of
# hardware.
print(jax.devices())

os.environ['KERAS_BACKEND'] = "jax"

import keras
print(keras.version())
print(keras.backend.backend())

keras.mixed_precision.set_global_policy("mixed_float16")

import keras_nlp


"""
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

- Pretrained models with `generate()` method, e.g., `keras_nlp.models.OPTCausalLM`.
- Sampler class that implements generation algorithms such as Top-K, Beam and
    contrastive search. These samplers can be used to generate text with
    custom models.
"""

def create_opt_model(model_spec):
  opt_model = keras_nlp.models.OPTCausalLM.from_preset(model_spec)
  opt_model.summary()
  return opt_model

"""
we are going to first try to create the 7B model without any distribution,
and it will error out with a OOM message from JAX. The 7B param would take
about 28G GPU memory, and the per-GPU memory limit 16G. This doesn't even
count other items like optimizer states, as well as forward and backward path.
"""
# model_spec = 'opt_6.7b_en'
# langauge_model = create_opt_model(model_spec)

"""
Now let's create a new with distributions. In Keras 3, we introduce a new
unified distribution API that allow you to do data and model parallel
trainings. You can find more details of the API in https://keras.io/api/distribution/.
"""

# Create a 2D mesh for model parallel, change the mesh shape to tune the
# ratio of data/model parallelism
_BATCH_DIM_NAME = "batch"
_MODEL_DIM_NAME = "model"

# Create mesh with (1, 8) shape so that the weights are sharded across all 8
# GPUs.
mesh = keras.distribution.DeviceMesh(
    (1, 8),
    [_BATCH_DIM_NAME, _MODEL_DIM_NAME],
    devices=keras.distribution.list_devices())

"""
The following code specifies how we would like to distribute the model weights.
The layout map is a dict like object, which maps the string key to a Layout.
The string key is used to indentify the variables in the Keras model, and the
corresponding Layout sharding will be applied to the weights. Note that the
key is like a regex, so it can be applied to both variables and its related
optimizer states.

You can find more details about the Layout Map in https://keras.io/api/distribution/layout_map/#layoutmap-class.
"""
unshard_dim = None
model_dim = _MODEL_DIM_NAME

layout_map = keras.distribution.LayoutMap(mesh)

layout_map[r"embeddings.*"] = (unshard_dim, model_dim)

# Transformer block sharding
layout_map[r"self_attention.*(query|key|value).*kernel.*"] = (
    unshard_dim, unshard_dim, model_dim)
layout_map[r"self_attention.*(query|key|value).*bias.*"] = (
    model_dim, unshard_dim)
layout_map[r"self_attention.*attention_output.*kernel.*"] = (
    unshard_dim, model_dim, unshard_dim)
layout_map[r"intermediate_dense.*kernel.*"] = (
    unshard_dim, model_dim)
layout_map[r"intermediate_dense.*bias.*"] = (
    model_dim,)
layout_map[r"output_dense.*kernel.*"] = (model_dim, unshard_dim)
layout_map[r"output_dense.*bias.*"] = (unshard_dim,)


"""
Next we will create a global distribut setting, and all the variables/data
created afterwards will be distributed according to this setting.

There is also a scope based API available with `model_parallel.scope()`.
"""
model_parallel = keras.distribution.ModelParallel(
    mesh, layout_map, batch_dim_name=_BATCH_DIM_NAME)
keras.distribution.set_distribution(model_parallel)

"""
Let's create the 2.7B model here, and with the model weights and forward path,
it won't be able to fit into GPU memory without any distribution.
"""
# Other avaiable model_spec are 'opt_125m_en', 'opt_1.3b_en' and 'opt_6.7b_en'
model_spec = 'opt_2.7b_en'
large_model = create_opt_model(model_spec)

"""
Inference

Note that the first run will take long time, since JAX need to compile the
generate function with XLA. The follow up runs will be much faster.
"""
prompt = "What is machine learning?"
print(large_model.generate(prompt))
