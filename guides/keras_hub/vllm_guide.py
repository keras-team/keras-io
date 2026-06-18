"""
Title: Serving KerasHub models with vLLM
Author: [Dhiraj](https://github.com/Dhiraj099)
Date created: 2025/08/16
Last modified: 2026/06/17
Description: Export a KerasHub model to Hugging Face format and serve it with vLLM.
Accelerator: TPU
"""

"""
## Introduction

This guide shows how to take a
[Gemma 3](https://www.kaggle.com/models/keras/gemma3/) model from
KerasHub, export it to the Hugging Face safetensors format, and serve it with
[vLLM](https://docs.vllm.ai/) on a Cloud TPU for fast, high-throughput inference.

vLLM is an inference and serving engine for large language models. It uses
techniques such as PagedAttention and continuous batching to keep the
accelerator busy and to serve many requests at once. KerasHub models do not run
inside vLLM directly, but KerasHub can export a model to the standard Hugging
Face format that vLLM loads natively. The bridge is a single method call:
`export_to_transformers()`.

The whole workflow runs in one session :

- Export the Keras model to safetensors on the CPU.
- Serve those safetensors with vLLM on the TPU.

**Prerequisites.** Select a **TPU v5e (or newer)** runtime
(`Runtime > Change runtime type`); the TPU build of vLLM does not support the
older v2-8. Gemma is a gated model, so you also need a
[Kaggle account](https://www.kaggle.com/) and an API token.
"""

"""
## Setup

We need two pieces: KerasHub, to load and export the model, and the TPU build of
vLLM, `vllm-tpu`. `vllm-tpu` is the JAX/TPU distribution of vLLM and is the
dependency that must match your TPU runtime, so it is the one we pin. pip
resolves a compatible `transformers` and `numpy` automatically.
"""

"""shell
pip install -q vllm-tpu keras-hub keras
"""
import os
import sys
import logging
import warnings

# Suppress noisy C++ and backend compilation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TQDM_DISABLE"] = "1"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

"""
## Authenticate with Kaggle

Gemma's weights are gated, so KerasHub needs Kaggle credentials to download the
preset. In Colab, store `KAGGLE_USERNAME` and `KAGGLE_KEY` in the Secrets panel
(the key icon in the left sidebar) and load them into the environment. On other
platforms, set the same two environment variables directly. You can create a
token at `kaggle.com -> Settings -> API`.
"""

import os

try:
    from google.colab import userdata

    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
except Exception:
    pass

print("Kaggle credentials configured.")

"""
## Export the model to Hugging Face format

Keras reads the `KERAS_BACKEND` environment variable when it is first imported,
so we set it before importing `keras_hub`. We use the PyTorch backend here: on a
TPU VM it runs on the CPU, so the export never occupies the TPU and the device
stays free for vLLM in the next step. The export is otherwise
backend-independent: the safetensors it writes are identical whichever backend
you pick.

> **Important Note on TPU Memory Allocation:** While you can technically use the
> JAX or TensorFlow backends to export the model, doing so on a TPU runtime will
> cause JAX/TF to immediately reserve the majority of the TPU's memory. Because
> vLLM requires exclusive access to the TPU initialized in the same session,
> pre-allocating that memory might cause vLLM to crash with a device initialization
> or Out-Of-Memory error.

`export_to_transformers()` writes `config.json`, the tokenizer files, and a
`model.safetensors` file to the export directory.
"""

os.environ["KERAS_BACKEND"] = "torch"

import keras_hub

model_preset = "gemma3_1b"
export_path = "./gemma3_exported"

gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(model_preset)
gemma_lm.export_to_transformers(export_path)
print(f"Model exported to {export_path}.")

"""
The exported files on disk are now everything vLLM needs, so we release the Keras
model to free the host memory it occupies before starting the vllm server.
"""

import gc

del gemma_lm
gc.collect()
print("Released the Keras model from host memory.")

"""
## Serve the model with vLLM

We load the exported directory into vLLM and run inference in-process, so the
generated text prints directly in the notebook. Two settings are specific to
running vLLM inside a Colab notebook on a TPU:

- `VLLM_ENABLE_V1_MULTIPROCESSING` is set to `"0"` so vLLM's engine runs in the
current process. By default the engine is launched in a forked subprocess, which
is unsafe once JAX has initialized the TPU.
- vLLM internally calls `sys.stdout.fileno()`, which a Colab notebook stream does
not implement. We point `fileno()` at the real output descriptor so the call
succeeds; normal cell output is unaffected.

The first `generate()` call triggers a one-time XLA compilation for the TPU and
takes a couple of minutes. Later calls are fast.
"""

import sys
import logging
import warnings

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.stdout.fileno = lambda: 1
sys.stderr.fileno = lambda: 2

from vllm import LLM, SamplingParams

llm = LLM(
    model=export_path,
    load_format="safetensors",
    dtype="bfloat16",
    max_model_len=1024,
    tensor_parallel_size=1,
)
print("vLLM engine ready.")

"""
## Run inference

vLLM accepts a list of prompts and generates for all of them in a single batch,
which is the source of its throughput advantage. `SamplingParams` controls the
decoding behavior, such as the temperature and the maximum number of tokens. We
pass `use_tqdm=False` to keep the progress bars out of the output.
"""

prompts = [
    "The future of artificial intelligence will involve",
    "Write a one-sentence summary of how solar panels work.",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)

outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
for output in outputs:
    print("=" * 60)
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text.strip()}")

"""
## Conclusion

Congratulations! You exported a Gemma 3 model from KerasHub to the Hugging Face
safetensors format and served it with vLLM on a TPU, all in a single session.
The same pattern works across various supported KerasHub model architectures,
including Gemma, Qwen, and Mistral variants.

For a production deployment, run vLLM as a standalone server with
`vllm serve <export_path>`, which exposes an OpenAI-compatible HTTP API and
removes the notebook-specific settings used above. To scale up, move from a Colab
TPU to a Cloud TPU VM (v5e or v6e) and select a larger preset.
"""
