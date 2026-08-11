"""
Title: Serving KerasHub models with vLLM on TPU
Author: [Anthony Etim](https://github.com/anthony-etim), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2026/08/10
Last modified: 2026/08/10
Description: An introduction to serving KerasHub models with vLLM on TPU.
Accelerator: TPU
"""

"""
# Introduction

KerasHub models generate text with `CausalLM.generate()`, which decodes a fixed
batch of prompts in one static loop. That works well for experiments and for
scoring a dataset offline. It works less well for serving, where requests
arrive one at a time, have different lengths, and finish at different steps. A
static loop makes every request in the batch wait for the longest one.

[vLLM](https://docs.vllm.ai/) is built for that case. It schedules requests
continuously, keeps the KV cache in fixed-size pages so memory is not reserved
for padding, and runs attention with a kernel written for paged memory.

This guide shows how to serve a KerasHub `CausalLM` through vLLM's TPU backend.
You do not convert the model, export the weights, or reimplement the
architecture. The preset loads with `from_preset` the way it always does, and
the same KerasHub layers run the forward pass.

The entry point is one class:

```python
from keras_hub.vllm import KerasHubLLM

llm = KerasHubLLM("keras_hub:gemma3_instruct_1b")
```

`KerasHubLLM` is a `vllm.LLM` that understands the `keras_hub:<preset>` model
scheme. Very little of the model behaves differently while it serves. Attention
layers call vLLM's TPU paged-attention kernel instead of building a dense
attention matrix, and models with learned position embeddings read vLLM's
per-token positions instead of counting from the start of the sequence. Every
other layer is untouched, and outside of serving even those two run their
ordinary path, so `generate()` still works exactly as before.

Everything here uses vLLM's Python `LLM` API, which batches a list of prompts
in one process. That is the right shape for a notebook and for offline batch
work. Standing up an HTTP endpoint is a separate topic and is not covered.

This guide needs a TPU runtime. On Colab, pick one under
**Runtime > Change runtime type**.
"""

"""
# Setup

Install KerasHub and the TPU build of vLLM. The `vllm-tpu` wheel brings in
`tpu-inference`, which is the TPU backend vLLM runs on. The version of `flax` is pinned because current `tpu-inference` needs
`flax==0.12.8`.
"""

"""shell
pip uninstall -y torchaudio -q
pip install -q vllm-tpu
pip install -q keras-hub
pip install -q flax==0.12.8
pip install -q --no-deps --force-reinstall git+https://github.com/vllm-project/tpu-inference
"""

"""
A few environment variables have to be set before Keras or vLLM is imported.
The first two put Keras on the JAX backend with NNX enabled. NNX is not
optional here: it makes the backbone's variables NNX state, which is how the
TPU runner carries the weights without a conversion step. The next two point
JAX and vLLM at the TPU. The two after that stop JAX from preallocating the
whole device, which would leave nothing for the vLLM KV cache. The last one
keeps the vLLM engine in this process instead of a subprocess, so its errors
surface in the notebook.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"
os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

"""
Most presets download from Kaggle without credentials. The gated ones (Gemma
and Llama, among others) need you to accept the license on the model's Kaggle
page and to set `KAGGLE_USERNAME` and `KAGGLE_KEY`. On Colab, add them in the
Secrets panel (the key icon in the left sidebar) and enable notebook access.
"""

try:
    from google.colab import userdata

    for secret in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            os.environ[secret] = userdata.get(secret)
        except Exception:
            pass
except ImportError:
    pass

"""
One last piece of setup, and this one is worth doing. The first time a model
runs, XLA compiles it, which takes a few minutes and dominates everything else
on the clock. A compilation cache means you pay that once rather than once per
session.

The cache goes in a local directory here. A local directory lasts as long as
the runtime, so to keep the cache across sessions on Colab, mount Drive and
point the cache at it instead:

```python
from google.colab import drive

drive.mount("/content/drive")
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/drive/MyDrive/jax_cache"
```
"""

os.environ["JAX_COMPILATION_CACHE_DIR"] = "jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

"""
# Load a model

Pass any KerasHub `CausalLM` preset with the `keras_hub:` prefix. Behind that
call, `KerasHubLLM` writes a temporary model directory for the preset holding
a config file and the exported tokenizer, then starts a vLLM engine pointed at
it. That directory holds no weights and is deleted with the `KerasHubLLM`
object. The weights come from a normal `CausalLM.from_preset` call.

`max_model_len` caps prompt plus generated tokens. Keep it near what you
actually need: it sizes the KV cache and the shapes that get compiled, so a
large value costs memory and compile time you may not use.

vLLM logs a lot at startup, and its logger needs a real file behind stdout,
which a notebook kernel does not provide. Sending the startup logs to a file
covers both problems.
"""

from contextlib import redirect_stderr
from contextlib import redirect_stdout

from keras_hub.vllm import KerasHubLLM

PRESET = "gemma3_instruct_1b"

with (
    open("vllm_init.log", "w") as log,
    redirect_stdout(log),
    redirect_stderr(log),
):
    llm = KerasHubLLM(f"keras_hub:{PRESET}", max_model_len=512)

print("model loaded")

"""
If this raises, the reason is in `vllm_init.log`. The usual causes are a
missing Kaggle license acceptance and a `max_model_len` too large for the
memory left after the model is loaded.
"""

"""
# Generate

`generate()` takes a list of prompts and returns one `RequestOutput` per
prompt. vLLM batches them for you, so passing three prompts at once costs
much less than three separate calls.
"""

prompts = [
    "The future of artificial intelligence is",
    "In a shocking finding, scientists discovered",
    "The best way to learn programming is",
]

outputs = llm.generate(prompts)
for output in outputs:
    print("-" * 72)
    print(output.prompt + output.outputs[0].text)

"""
# Sampling

Called with no `SamplingParams`, `KerasHubLLM` samples the way the preset
would sample under `CausalLM.generate()`. Every KerasHub task compiles itself
with `CausalLM.compile`'s default sampler, which is `top_k`, and that setting
is translated into the equivalent vLLM `SamplingParams` for you. This is the
one place `KerasHubLLM` deliberately departs from `vllm.LLM`, which would
otherwise apply its own generic defaults and quietly change how the model
behaves.

Length is not part of a KerasHub sampler, so vLLM's default of 16 output
tokens applies. That is short. Pass `SamplingParams` when you want more.
"""

from vllm import SamplingParams

greedy = SamplingParams(temperature=0.0, max_tokens=48)

outputs = llm.generate(prompts, greedy)
for output in outputs:
    print("-" * 72)
    print(output.prompt + output.outputs[0].text)

"""
Explicit `SamplingParams` override the model's own settings completely. Use
them whenever you want reproducible output: `temperature=0.0` is greedy
decoding, which makes the model pick the highest-probability token every step.
"""

"""
# Serving many requests

The reason to serve through vLLM is what happens under load. Here 32 requests
are submitted at once, as a rough stand-in for concurrent traffic. vLLM
schedules them continuously, so a request that finishes early frees its slot
instead of idling until the whole batch is done.

The prompts are deliberately all different. vLLM enables prefix caching by
default, so 32 copies of one prompt would let 31 of them skip most of the
prefill and report a throughput you would never see on real traffic.
"""

import time

many_prompts = [
    f"Question {i}: explain in one paragraph how a computer works." for i in range(32)
]
params = SamplingParams(temperature=0.0, max_tokens=64)

llm.generate(many_prompts, params)

start = time.perf_counter()
outputs = llm.generate(many_prompts, params)
elapsed = time.perf_counter() - start

generated = sum(len(output.outputs[0].token_ids) for output in outputs)
print(f"{len(outputs)} requests, {generated} tokens in {elapsed:.2f}s")
print(f"{generated / elapsed:.0f} output tokens/s")

"""
The first `generate()` call is a warmup and is not timed. The engine
precompiles its shape buckets at startup, so most of the compile cost is
already paid by then, but the first call still touches things a timed run
should not have to pay for.
"""

"""
# Supported models

A KerasHub model can serve through vLLM once its attention layer has a serving
route. These families have one and have been run end to end on TPU:

| Family | Example preset |
|---|---|
| GPT-2 | `gpt2_base_en`, `gpt2_large_en` |
| Llama 3 | `llama3.2_instruct_1b` |
| Gemma 1 | `gemma_2b_en` |
| Gemma 2 | `gemma2_2b_en` |
| Gemma 3 (text) | `gemma3_instruct_1b` |
| Qwen 2.5 | `qwen2.5_coder_0.5b` |

Everything else about a model is already generic, so adding a family means
adding a route to one attention layer. A family without one fails loudly
instead of falling back to something slower or wrong: the serving wrapper
counts how many attention layers dispatched to the paged kernel, and raises
if that does not match the number of transformer layers.

Most `qwen2.5` presets other than the one above currently fail to download
from Kaggle for reasons unrelated to serving, so start with
`qwen2.5_coder_0.5b` for that family.

Two limits are worth knowing before you plan around this. Only `CausalLM`
presets are served, since this integration targets autoregressive text
generation. And vision-language models are not supported: the attention path
does not currently accept the custom mask a bidirectional image encoder needs.
"""

"""
# How it works

Three pieces make this work, and none of them require a converted checkpoint.

**A neutral config.** `KerasHubLLM` writes a config with its own `keras_hub`
model type rather than borrowing a model family from Hugging Face. vLLM sizes
the KV cache from the real backbone dimensions instead of guessing them from a
name.

**The preset's own tokenizer.** Serving tokenizes with the KerasHub tokenizer
that shipped with the preset, registered with vLLM under
`tokenizer_mode="keras_hub"`. There is no conversion step to a Hugging Face
tokenizer and no chance of the two disagreeing.

**A serving context.** While a request is in flight, a thread-local context is
active, carrying the paged KV caches and vLLM's per-token positions. Routed
attention layers check for it on entry: if it is set they call the
paged-attention kernel, and if not they run the ordinary dense path,
unchanged. Learned position embeddings read the same context for their
positions. This is why the same model still works with `generate()`, and why
the context is cleared when a step ends even if the forward raised.
"""

"""
# Next steps

- Try a different preset from the table above. Load one model per runtime
  session, since each engine holds the TPU for itself.
- Compare against `CausalLM.generate()` on the same weights. The gap is small
  for one request at a time and grows with concurrency and prompt length,
  which is exactly the workload vLLM is for.
- Read `vllm_init.log` once, even on a successful run. It shows the
  KV cache size vLLM settled on and how many shapes it compiled, which are
  the two numbers to adjust if you run out of memory.
"""
