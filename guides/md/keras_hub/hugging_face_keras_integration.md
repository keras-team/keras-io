# Loading HuggingFace Transformers checkpoints into multi-backend KerasHub models

**Author:** [Laxma Reddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)<br><br>
**Date created:** 2025/06/17<br><br>
**Last modified:** 2025/06/17<br><br>
**Description:** How to load and run inference from KerasHub model checkpoints hosted on HuggingFace Hub.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/hugging_face_keras_integration.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/hugging_face_keras_integration.py)



---
## Introduction

KerasHub has built-in converters for HuggingFace's `.safetensors` models.
Loading model weights from HuggingFace is therefore no more difficult than
using KerasHub's own presets.

### KerasHub built-in HuggingFace transformers converters

KerasHub simplifies the use of HuggingFace Transformers models through its
built-in converters. These converters automatically handle the process of translating
HuggingFace model checkpoints into a format that's compatible with the Keras ecosystem.
This means you can seamlessly load a wide variety of pretrained models from the HuggingFace
Hub directly into KerasHub with just a few lines of code.

Key advantages of using KerasHub converters:

- **Ease of Use**: Load HuggingFace models without manual conversion steps.
- **Broad Compatibility**: Access a vast range of models available on the HuggingFace Hub.
- **Seamless Integration**: Work with these models using familiar Keras APIs for training,
evaluation, and inference.

Fortunately, all of this happens behind the scenes, so you can focus on using
the models rather than managing the conversion process!

---
## Setup

Before you begin, make sure you have the necessary libraries installed.
You'll primarily need `keras` and `keras_hub`.

**Note:** Changing the backend after Keras has been imported might not work as expected.
Ensure `KERAS_BACKEND` is set at the beginning of your script.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # "tensorflow" or  "torch"

import keras
import keras_hub
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1750320123.730040    8092 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1750320123.734497    8092 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1750320123.745803    8092 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1750320123.745816    8092 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1750320123.745818    8092 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1750320123.745819    8092 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

KerasHub allows you to easily load models from HuggingFace Transformers.
Here's an example of how to load a Gemma causal language model.
In this particular case, you will need to consent to Google's license on
HuggingFace for being able to download model weights, and provide your
`HF_TOKEN` as environment variable or as "Colab secret" when working with
Google Colab.


```python
# not a keras checkpoint, it is a HF transformer checkpoint

gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("hf://google/gemma-2b")
```

<div class="k-default-codeblock">
```
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
```
</div>

Let us try running some inference


```python
gemma_lm.generate("I want to say", max_length=30)
```




<div class="k-default-codeblock">
```
'I want to say thank you to the staff at the <strong><em><u><strong><em><u><strong><em><u><strong><em><u><strong><em><u><strong><em>'
```
</div>

### Fine-tuning a Gemma Transformer checkpoint using the Keras `model.fit(...)` API

Once you have loaded HuggingFace weights, you can use the instantiated model
just like any other KerasHub model. For instance, you might fine-tune the model
on your own data like so:


```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
gemma_lm.fit(x=features, batch_size=2)
```

    
<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 35s 35s/step - loss: 0.0342 - sparse_categorical_accuracy: 0.1538

<keras.src.callbacks.history.History at 0x7e970c14cbf0>
```
</div>

### Saving and uploading the new checkpoint

To store and share your fine-tuned model, KerasHub makes it easy to save or
upload it using standard methods. You can do this through familiar commands
such as:


```python
gemma_lm.save_to_preset("./gemma-2b-finetuned")
keras_hub.upload_preset("hf://laxmareddyp/gemma-2b-finetune", "./gemma-2b-finetuned")
```

By uploading your preset, you can then load it from anywhere using:
`loaded_model = keras_hub.models.GemmaCausalLM.from_preset("hf://YOUR_HF_USERNAME/gemma-2b-finetuned")`

For a comprehensive, step-by-step guide on uploading your model, refer to the official KerasHub upload documentation.
You can find all the details here: [KerasHub Upload Guide](https://keras.io/keras_hub/guides/upload/)

By integrating HuggingFace Transformers, KerasHub significantly expands your access to pretrained models.
The Hugging Face Hub now hosts well over 750k+ model checkpoints across various domains such as NLP,
Computer Vision, Audio, and more. Of these, approximately 400K models are currently compatible with KerasHub,
giving you access to a vast and diverse selection of state-of-the-art architectures for your projects.

With KerasHub, you can:

- **Tap into State-of-the-Art Models**: Easily experiment with the latest
architectures and pretrained weights from the research community and industry.
- **Reduce Development Time**: Leverage existing models instead of training from scratch,
saving significant time and computational resources.
- **Enhance Model Capabilities**: Find specialized models for a wide array of tasks,
from text generation and translation to image segmentation and object detection.

This seamless access empowers you to build more powerful and sophisticated AI applications with Keras.

---
## Use a wider range of frameworks

Keras 3, and by extension KerasHub, is designed for multi-framework compatibility.
This means you can run your models with different backend frameworks like JAX, TensorFlow, and PyTorch.
This flexibility allows you to:

- **Choose the Best Backend for Your Needs**: Select a backend based on performance characteristics,
hardware compatibility (e.g., TPUs with JAX), or existing team expertise.
- **Interoperability**: More easily integrate KerasHub models into existing
workflows that might be built on TensorFlow or PyTorch.
- **Future-Proofing**: Adapt to evolving framework landscapes without
rewriting your core model logic.

---
## Run transformer models in JAX backend and on TPUs

To experiment with a model using JAX, you can utilize Keras by setting its backend to JAX.
By switching Keras’s backend before model construction, and ensuring your environment is connected to a TPU runtime.
Keras will then automatically leverage JAX’s TPU support,
allowing your model to train efficiently on TPU hardware without further code changes.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("hf://google/gemma-2b")
```

---
## Additional Examples

### Generation

Here’s an example using Llama: Loading a PyTorch Hugging Face transformer checkpoint into KerasHub and running it on the JAX backend.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

from keras_hub.models import Llama3CausalLM

# Get the model
causal_lm = Llama3CausalLM.from_preset("hf://NousResearch/Hermes-2-Pro-Llama-3-8B")

prompts = [
    """<|im_start|>system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
<|im_start|>user
Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.<|im_end|>
<|im_start|>assistant""",
]

# Generate from the model
causal_lm.generate(prompts, max_length=30)[0]
```




<div class="k-default-codeblock">
```
'<tool_call>Csystem\nYou are a sentient, superintelligent artificial general intelligence, here to teach and assist me.\n<tool'
```
</div>

### Changing precision

You can adjust your model’s precision by configuring it through `keras.config` as follows


```python
import keras

keras.config.set_dtype_policy("bfloat16")

from keras_hub.models import Llama3CausalLM

causal_lm = Llama3CausalLM.from_preset("hf://NousResearch/Hermes-2-Pro-Llama-3-8B")
```

Go try loading other model weights! You can find more options on HuggingFace
and use them with `from_preset("hf://<namespace>/<model-name>")`.

Happy experimenting!
