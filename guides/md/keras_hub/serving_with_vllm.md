# Serving KerasHub models with vLLM on TPU

**Author:** [Anthony Etim](https://github.com/anthony-etim)<br>
**Date created:** 2026/08/10<br>
**Last modified:** 2026/08/10<br>
**Description:** An introduction to serving KerasHub models with vLLM on TPU.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/serving_with_vllm.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/serving_with_vllm.py)



---
## Introduction

This guide shows how to serve a KerasHub `CausalLM` through vLLM's TPU backend.
You do not convert the model, export the weights, or reimplement the
architecture. The preset loads with `from_preset` the way it always does, and
the same KerasHub layers run the forward pass.

The entry point is one class:

```python
from keras_hub.vllm import KerasHubLLM

llm = KerasHubLLM("keras_hub:gemma3_instruct_1b")
```

KerasHub models generate text with `CausalLM.generate()`, which decodes a fixed
batch of prompts in one static loop. That works well for experiments and for
scoring a dataset offline. It works less well for serving, where requests
arrive one at a time, have different lengths, and finish at different steps. A
static loop makes every request in the batch wait for the longest one.

[vLLM](https://docs.vllm.ai/) is built for that case. It schedules requests
continuously, keeps the KV cache in fixed-size pages so memory is not reserved
for padding, and runs attention with a kernel written for paged memory.

This guide needs a TPU runtime. On Colab, pick one under
**Runtime > Change runtime type**.

---
## Setup

Install KerasHub and the TPU build of vLLM, which brings in `tpu-inference`.
Both pins are needed: Colab's TPU image has no `keras-hub`, and its `flax`
0.11.2 predates what `tpu-inference` requires.


```python
!pip uninstall -y torchaudio -q
!pip install -q vllm-tpu
!pip install -q 'keras-hub>=0.31.0'
!pip install -q --no-deps --force-reinstall git+https://github.com/vllm-project/tpu-inference
!pip install -q flax==0.12.8
```

<div class="k-default-codeblock">
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tpu-inference 0.0.0 requires jax==0.11.0, but you have jax 0.10.2 which is incompatible.
tpu-inference 0.0.0 requires jaxlib==0.11.0, but you have jaxlib 0.10.2 which is incompatible.
tpu-inference 0.0.0 requires libtpu==0.0.44, but you have libtpu 0.0.43 which is incompatible.
vllm-tpu 0.26.0 requires tpu-inference==0.26.0, but you have tpu-inference 0.0.0 which is incompatible.
```
</div>

A few environment variables have to be set before Keras or vLLM is imported.
Set `KERAS_BACKEND` to `jax` and `KERAS_NNX_ENABLED` to `true` to run Keras on
the JAX backend with NNX enabled. NNX is not optional here: it makes the
backbone's variables NNX state, which is how the TPU runner carries the weights
without a conversion step. `JAX_PLATFORMS` and `VLLM_TARGET_DEVICE` point JAX
and vLLM at the TPU. `XLA_PYTHON_CLIENT_PREALLOCATE` and
`XLA_PYTHON_CLIENT_ALLOCATOR` stop JAX from preallocating the whole device,
which would leave nothing for the vLLM KV cache.
`VLLM_ENABLE_V1_MULTIPROCESSING` keeps the vLLM engine in this process instead
of a subprocess, so its errors surface in the notebook.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"
os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

Most presets download from Kaggle without credentials. The gated ones (Gemma
and Llama, among others) need you to accept the license on the model's Kaggle
page and to set `KAGGLE_USERNAME` and `KAGGLE_KEY`. On Colab, add them in the
Secrets panel (the key icon in the left sidebar) and enable notebook access.


```python
try:
    from google.colab import userdata

    for secret in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            os.environ[secret] = userdata.get(secret)
        except Exception:
            pass
except ImportError:
    pass
```

One last piece of setup. The first time a model runs, XLA compiles it, which
takes a few minutes and dominates everything else on the clock. A compilation
cache means you pay that once rather than once per session.

The cache goes in a local directory here. A local directory lasts as long as
the runtime, so to keep the cache across sessions on Colab, mount Drive and
point the cache at it instead:

```python
from google.colab import drive

drive.mount("/content/drive")
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/drive/MyDrive/jax_cache"
```


```python
os.environ["JAX_COMPILATION_CACHE_DIR"] = "jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
```

---
## Load a model

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


```python
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
```

<div class="k-default-codeblock">
```
/usr/local/lib/python3.12/dist-packages/jax/_src/cloud_tpu_init.py:88: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled")
  warnings.warn(

ERROR 08-16 20:22:58 [tpu_info.py:40] Unable to poll TPU GCE Metadata. Got status code: 404 and content: 

INFO 08-16 20:22:58 [__init__.py:76] TPU info: node_name=None | tpu_type=v5e-1 | worker_id=0 | num_chips=1 | num_cores_per_chip=2

WARNING 08-16 20:22:58 [tpu_platform.py:426] Pin memory is not supported on TPU.

Check failed with unknown exit code: -6.

INFO 08-16 20:23:02 [importing.py:53] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.

WARNING 08-16 20:23:02 [importing.py:73] Triton is installed, but `triton.backends` could not be imported. Disabling Triton.

INFO 08-16 20:23:02 [importing.py:88] Triton not installed or not compatible; certain GPU-related functions will not be available.

INFO 08-16 20:23:18 [attention_interface.py:58] Using default RPA kernel

WARNING 08-16 20:23:18 [interface.py:368] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")

WARNING 08-16 20:23:18 [interface.py:368] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")

WARNING 08-16 20:23:18 [interface.py:368] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")

WARNING 08-16 20:23:18 [interface.py:368] Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")

INFO 08-16 20:23:18 [__init__.py:112] Registered model loader `<class 'tpu_inference.models.jax.utils.weight_utils.JaxDummyModelLoader'>` with load format `jax_dummy`

INFO 08-16 20:23:18 [__init__.py:112] Registered model loader `<class 'tpu_inference.models.common.pathways_dummy_loader.PathwaysDummyModelLoader'>` with load format `pathways_dummy`

INFO 08-16 20:23:18 [__init__.py:112] Registered model loader `<class 'tpu_inference.models.vllm.vllm_model_loader.IncrementalModelLoader'>` with load format `tpu_streaming_loader`

WARNING 08-16 20:23:18 [__init__.py:101] Load format `runai_streamer` is already registered, and will be overwritten by the new loader class `<class 'tpu_inference.models.vllm.vllm_model_loader.RunaiIncrementalModelLoader'>`.

INFO 08-16 20:23:18 [__init__.py:112] Registered model loader `<class 'tpu_inference.models.vllm.vllm_model_loader.RunaiIncrementalModelLoader'>` with load format `runai_streamer`

INFO 08-16 20:23:18 [model_loader.py:866] Registered JAX model KerasHubForCausalLM with tpu_inference and vLLM registries.

WARNING 08-16 20:23:18 [registry.py:62] keras_hub.src.vllm.tokenizer.KerasHubTokenizer is already registered for tokenizer_mode='keras_hub'. It is overwritten by the new one.

WARNING 08-16 20:23:18 [registry.py:40] vllm.renderers.hf.HfRenderer is already registered for renderer_mode='keras_hub'. It is overwritten by the new one.

INFO 08-16 20:23:18 [api_utils.py:273] non-default args: {'tokenizer': '/tmp/keras_hub_vllm_ns1sy0k7', 'tokenizer_mode': 'keras_hub', 'dtype': 'bfloat16', 'max_model_len': 512, 'disable_log_stats': True, 'model': '/tmp/keras_hub_vllm_ns1sy0k7'}

WARNING 08-16 20:23:18 [arg_utils.py:1628] The global random seed is set to 0. Since VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may affect the random state of the Python process that launched vLLM.

INFO 08-16 20:23:18 [model.py:627] Resolved architecture: KerasHubForCausalLM

INFO 08-16 20:23:18 [model.py:1799] Using max model len 512

INFO 08-16 20:23:18 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.

INFO 08-16 20:23:18 [vllm.py:1114] Asynchronous scheduling is enabled.

WARNING 08-16 20:23:18 [vllm.py:1218] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.

INFO 08-16 20:23:18 [kernel.py:303] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])

INFO 08-16 20:23:18 [tpu_platform.py:233] Initialized sharding configuration: ShardingConfigManager(total_devices=1, sharding_strategy=ShardingStrategy(tensor_parallelism=1, expert_parallelism=1, sequence_parallelism=1, data_parallelism=1, attention_data_parallelism=1, attention_data_expert_parallelism=1, decode_context_parallelism=1, prefill_context_parallelism=1), mm_encoder_tp_mode=weights, device_indexes=None)

INFO 08-16 20:23:18 [tpu_platform.py:342] Force using UniProcExecutor for JAX on single host without pipeline parallelism.

WARNING 08-16 20:23:18 [vllm.py:582] Model Runner V2 requires Triton; using the V1 model runner instead.

INFO 08-16 20:23:18 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant

WARNING 08-16 20:23:23 [registry.py:249] Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.

INFO 08-16 20:23:23 [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='/tmp/keras_hub_vllm_ns1sy0k7', speculative_config=None, tokenizer='/tmp/keras_hub_vllm_ns1sy0k7', skip_tokenizer_init=False, tokenizer_mode=keras_hub, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=512, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=None, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=/tmp/keras_hub_vllm_ns1sy0k7, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'eager', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::qwen_gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::hpc_rope_norm_forward', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': None, 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')

INFO 08-16 20:23:24 [parallel_state.py:1612] world_size=1 rank=0 local_rank=0 distributed_init_method=file:///tmp/tmp82x_gdbv backend=gloo

INFO 08-16 20:23:24 [parallel_state.py:1943] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A

INFO 08-16 20:23:24 [tpu_runner.py:885] Init mesh | mesh=Mesh('data': 1, 'model': 1, axis_types=(Auto, Auto))

INFO 08-16 20:23:24 [utils.py:211] Prepared token paddings: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

INFO 08-16 20:23:24 [utils.py:157] Prepared request paddings: [8, 16, 32, 64, 128, 256]

INFO 08-16 20:23:24 [utils.py:176] Prepared attn request paddings: [256]

INFO 08-16 20:23:24 [compilation_manager.py:66] Enabling JAX compile cache.

INFO 08-16 20:23:24 [tpu_worker.py:471] Init worker | rank=0 | is_first_rank=True | is_last_rank=True | topology_order_id=0 | is_driver_worker=True | hbm=[(0.0, 15.75)]GiB |self.devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)] | total devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)] | local_devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]

INFO 08-16 20:23:24 [model_loader.py:656] Loading model with MODEL_IMPL_TYPE=auto

INFO 08-16 20:23:24 [model_loader.py:659] Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'

INFO 08-16 20:23:52 [tpu_runner.py:1262] Cleared JIT caches after weight loading

INFO 08-16 20:23:52 [tpu_runner.py:1264] Init model | hbm=[(1.86, 15.75)]GiB

INFO 08-16 20:23:52 [tpu_platform.py:417] Using cache_config.block_size: 32 instead of overriding with _align_hybrid_block_size() since we set mamba_page_size_padded in kv_cache_manager.py

INFO 08-16 20:23:52 [tpu_worker.py:557] Memory statistics | total_hbm_limit_gb=15.75GiB | total_hbm_limit_cap_gb=14.49GiB | total_hbm_used_gb=1.86GiB | total_hbm_avail_gb=12.63GiB

INFO 08-16 20:23:52 [kv_cache_utils.py:2214] GPU KV cache size: 509,184 tokens

INFO 08-16 20:23:52 [kv_cache_utils.py:2215] Maximum concurrency for 512 tokens per request: 994.50x

INFO 08-16 20:23:52 [compilation_manager.py:949] Compiling sampling with different input shapes.

INFO 08-16 20:23:52 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 8, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:23:53 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 8, 'do_sampling': True, 'logprobs': True} finished in 0.70 [secs].

INFO 08-16 20:23:53 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 8, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:23:53 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 8, 'do_sampling': True, 'logprobs': False} finished in 0.67 [secs].

INFO 08-16 20:23:53 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 8, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:23:54 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 8, 'do_sampling': False, 'logprobs': True} finished in 0.41 [secs].

INFO 08-16 20:23:54 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 8, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:23:54 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 8, 'do_sampling': False, 'logprobs': False} finished in 0.41 [secs].

INFO 08-16 20:23:54 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 16, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:23:55 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 16, 'do_sampling': True, 'logprobs': True} finished in 0.75 [secs].

INFO 08-16 20:23:55 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 16, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:23:56 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 16, 'do_sampling': True, 'logprobs': False} finished in 0.72 [secs].

INFO 08-16 20:23:56 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 16, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:23:56 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 16, 'do_sampling': False, 'logprobs': True} finished in 0.44 [secs].

INFO 08-16 20:23:56 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 16, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:23:57 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 16, 'do_sampling': False, 'logprobs': False} finished in 0.43 [secs].

INFO 08-16 20:23:57 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 32, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:23:58 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 32, 'do_sampling': True, 'logprobs': True} finished in 0.71 [secs].

INFO 08-16 20:23:58 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 32, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:23:58 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 32, 'do_sampling': True, 'logprobs': False} finished in 0.72 [secs].

INFO 08-16 20:23:58 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 32, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:23:59 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 32, 'do_sampling': False, 'logprobs': True} finished in 0.42 [secs].

INFO 08-16 20:23:59 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 32, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:23:59 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 32, 'do_sampling': False, 'logprobs': False} finished in 0.42 [secs].

INFO 08-16 20:23:59 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 64, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:24:00 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 64, 'do_sampling': True, 'logprobs': True} finished in 0.82 [secs].

INFO 08-16 20:24:00 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 64, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:24:01 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 64, 'do_sampling': True, 'logprobs': False} finished in 0.77 [secs].

INFO 08-16 20:24:01 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 64, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:24:01 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 64, 'do_sampling': False, 'logprobs': True} finished in 0.42 [secs].

INFO 08-16 20:24:01 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 64, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:24:02 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 64, 'do_sampling': False, 'logprobs': False} finished in 0.41 [secs].

INFO 08-16 20:24:02 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 128, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:24:03 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 128, 'do_sampling': True, 'logprobs': True} finished in 0.84 [secs].

INFO 08-16 20:24:03 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 128, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:24:04 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 128, 'do_sampling': True, 'logprobs': False} finished in 0.85 [secs].

INFO 08-16 20:24:04 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 128, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:24:04 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 128, 'do_sampling': False, 'logprobs': True} finished in 0.42 [secs].

INFO 08-16 20:24:04 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 128, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:24:05 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 128, 'do_sampling': False, 'logprobs': False} finished in 0.42 [secs].

INFO 08-16 20:24:05 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 256, 'do_sampling': True, 'logprobs': True}

INFO 08-16 20:24:06 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 256, 'do_sampling': True, 'logprobs': True} finished in 0.93 [secs].

INFO 08-16 20:24:06 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 256, 'do_sampling': True, 'logprobs': False}

INFO 08-16 20:24:07 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 256, 'do_sampling': True, 'logprobs': False} finished in 0.88 [secs].

INFO 08-16 20:24:07 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 256, 'do_sampling': False, 'logprobs': True}

INFO 08-16 20:24:07 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 256, 'do_sampling': False, 'logprobs': True} finished in 0.42 [secs].

INFO 08-16 20:24:07 [compilation_manager.py:137] Precompile worker0 sample --> {'num_reqs': 256, 'do_sampling': False, 'logprobs': False}

INFO 08-16 20:24:08 [compilation_manager.py:188] Compilation of worker0 sample --> {'num_reqs': 256, 'do_sampling': False, 'logprobs': False} finished in 0.43 [secs].

INFO 08-16 20:24:08 [compilation_manager.py:1043] Compiling gather_logprobs with different input shapes.

INFO 08-16 20:24:08 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 8}

INFO 08-16 20:24:08 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 8} finished in 0.70 [secs].

INFO 08-16 20:24:08 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 16}

INFO 08-16 20:24:09 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 16} finished in 0.72 [secs].

INFO 08-16 20:24:09 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 32}

INFO 08-16 20:24:14 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 32} finished in 4.54 [secs].

INFO 08-16 20:24:14 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 64}

INFO 08-16 20:24:18 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 64} finished in 4.57 [secs].

INFO 08-16 20:24:18 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 128}

INFO 08-16 20:24:23 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 128} finished in 4.52 [secs].

INFO 08-16 20:24:23 [compilation_manager.py:137] Precompile worker0 gather_logprobs --> {'num_reqs': 256}

INFO 08-16 20:24:27 [compilation_manager.py:188] Compilation of worker0 gather_logprobs --> {'num_reqs': 256} finished in 4.50 [secs].

INFO 08-16 20:24:27 [compilation_manager.py:1099] Compiling compute_and_gather_prompt_logprobs with different input shapes.

INFO 08-16 20:24:27 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 16}

INFO 08-16 20:24:28 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 16} finished in 0.74 [secs].

INFO 08-16 20:24:28 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 32}

INFO 08-16 20:24:32 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 32} finished in 4.53 [secs].

INFO 08-16 20:24:32 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 64}

INFO 08-16 20:24:37 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 64} finished in 4.46 [secs].

INFO 08-16 20:24:37 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 128}

INFO 08-16 20:24:42 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 128} finished in 4.54 [secs].

INFO 08-16 20:24:42 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 256}

INFO 08-16 20:24:46 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 256} finished in 4.56 [secs].

INFO 08-16 20:24:46 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 512}

INFO 08-16 20:24:51 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 512} finished in 4.57 [secs].

INFO 08-16 20:24:51 [compilation_manager.py:137] Precompile worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 1024}

INFO 08-16 20:24:55 [compilation_manager.py:188] Compilation of worker0 compute_and_gather_prompt_logprobs --> {'num_tokens': 1024} finished in 4.54 [secs].

INFO 08-16 20:24:55 [compilation_manager.py:1108] Skipping precompilation of compute_and_gather_prompt_logprobs for num_tokens=2048, as it exceeds the MAX_PRECOMPILE_PROMPT_TOKENS=1024 limit to avoid redundant host CPU JAX tracing for long sequence lengths.

INFO 08-16 20:24:55 [compilation_manager.py:1108] Skipping precompilation of compute_and_gather_prompt_logprobs for num_tokens=4096, as it exceeds the MAX_PRECOMPILE_PROMPT_TOKENS=1024 limit to avoid redundant host CPU JAX tracing for long sequence lengths.

INFO 08-16 20:24:55 [compilation_manager.py:1108] Skipping precompilation of compute_and_gather_prompt_logprobs for num_tokens=8192, as it exceeds the MAX_PRECOMPILE_PROMPT_TOKENS=1024 limit to avoid redundant host CPU JAX tracing for long sequence lengths.

INFO 08-16 20:24:55 [compilation_manager.py:228] Warm-up call pass finished in 0.16 [secs] over 24 tasks.

INFO 08-16 20:24:55 [kv_cache_manager.py:958] Hybrid KV cache layout: num_kv_cache_groups=1, num_kv_cache_tensors=26, kv_cache_config.num_blocks=15912, duplicate_shared_layers=False

INFO 08-16 20:24:55 [kv_cache_manager.py:986] Init kv-cache | num_total_layers=26 | num_blocks=[15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912, 15912] | regular_attn_layers=26 | regular_attn_shape=(num_blocks, (32, 1, 2, 256)) | regular_attn_sharding=NamedSharding(mesh=Mesh('data': 1, 'model': 1, axis_types=(Auto, Auto)), spec=P('data', None, 'model'), memory_kind=device) | regular_attn_dtype=bfloat16 | hbm=[(14.5, 15.75)]Gb

INFO 08-16 20:24:55 [compilation_manager.py:236] Precompile all the subgraphs with possible input shapes.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 16, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 32, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 64, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 128, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 256, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 512, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 1024, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, ..., 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 2048, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, ..., 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 4096, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, ..., 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:24:56 [compilation_manager.py:137] Precompile worker0 backbone --> {'num_tokens': 8192, 'num_reqs': 256, 'shared_attention_metadata': SharedAttentionMetadata(input_positions=Array([1, 1, 1, ..., 1, 1, 1], dtype=int32), seq_lens=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), query_start_loc=Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
INFO 08-16 20:24:56 [compilation_manager.py:137]        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32), request_distribution=Array([0, 0, 0], dtype=int32), mamba_state_indices=None, padded_num_reqs=256)}

INFO 08-16 20:24:56 [compilation_manager.py:160] AOT lower skipped for worker0 backbone (not a jit); will compile in warmup.

INFO 08-16 20:27:01 [compilation_manager.py:228] Warm-up call pass finished in 124.80 [secs] over 10 tasks.

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 8}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 8} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 8}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 8} finished in 0.00 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 16}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 16} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 32}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 32} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 64}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 64} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 128}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 128} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 256}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8, 'next_tokens_size': 256} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 16}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 16} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 8}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 8} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 16}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 16} finished in 0.00 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 32}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 32} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 64}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 64} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 128}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 128} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 256}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 16, 'next_tokens_size': 256} finished in 0.05 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 32}

INFO 08-16 20:27:01 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 32} finished in 0.06 [secs].

INFO 08-16 20:27:01 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 8}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 8} finished in 0.06 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 16}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 16} finished in 0.07 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 32}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 32} finished in 0.00 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 64}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 64} finished in 0.07 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 128}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 128} finished in 0.06 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 256}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 32, 'next_tokens_size': 256} finished in 0.06 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 64}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 64} finished in 0.08 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 8}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 8} finished in 0.09 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 16}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 16} finished in 0.08 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 32}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 32} finished in 0.08 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 64}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 64} finished in 0.00 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 128}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 128} finished in 0.09 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 256}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 64, 'next_tokens_size': 256} finished in 0.07 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 128}

INFO 08-16 20:27:02 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 128} finished in 0.10 [secs].

INFO 08-16 20:27:02 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 8}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 8} finished in 0.10 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 16}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 16} finished in 0.11 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 32}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 32} finished in 0.10 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 64}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 64} finished in 0.10 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 128}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 128} finished in 0.00 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 256}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 128, 'next_tokens_size': 256} finished in 0.10 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 256}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 256} finished in 0.11 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 8}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 8} finished in 0.11 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 16}

INFO 08-16 20:27:03 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:03 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 32}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 32} finished in 0.19 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 64}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 64} finished in 0.13 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 128}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 128} finished in 0.11 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 256}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 256, 'next_tokens_size': 256} finished in 0.00 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 512}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 512} finished in 0.11 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 8}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 8} finished in 0.13 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 16}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 32}

INFO 08-16 20:27:04 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 32} finished in 0.16 [secs].

INFO 08-16 20:27:04 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 64}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 64} finished in 0.24 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 128}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 128} finished in 0.12 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 256}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 512, 'next_tokens_size': 256} finished in 0.12 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 1024}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 1024} finished in 0.11 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 8}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 8} finished in 0.12 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 16}

INFO 08-16 20:27:05 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:05 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 32}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 32} finished in 0.15 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 64}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 64} finished in 0.24 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 128}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 128} finished in 0.11 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 256}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 1024, 'next_tokens_size': 256} finished in 0.12 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 2048}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 2048} finished in 0.11 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 8}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 8} finished in 0.12 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 16}

INFO 08-16 20:27:06 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:06 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 32}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 32} finished in 0.15 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 64}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 64} finished in 0.24 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 128}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 128} finished in 0.11 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 256}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 2048, 'next_tokens_size': 256} finished in 0.13 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 4096}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 4096} finished in 0.11 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 8}

INFO 08-16 20:27:07 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 8} finished in 0.12 [secs].

INFO 08-16 20:27:07 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 16}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 16} finished in 0.15 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 32}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 32} finished in 0.16 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 64}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 64} finished in 0.25 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 128}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 128} finished in 0.11 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 256}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 4096, 'next_tokens_size': 256} finished in 0.11 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 8192}

INFO 08-16 20:27:08 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 8192} finished in 0.13 [secs].

INFO 08-16 20:27:08 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 8}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 8} finished in 0.12 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 16}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 32}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 32} finished in 0.15 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 64}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 64} finished in 0.25 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 128}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 128} finished in 0.11 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 256}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of _substitute_placeholder_token_fn --> {'num_tokens': 8192, 'next_tokens_size': 256} finished in 0.11 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:228] Warm-up call pass finished in 0.04 [secs] over 77 tasks.

INFO 08-16 20:27:09 [compilation_manager.py:862] Compiling select_from_array with different input shapes.

INFO 08-16 20:27:09 [compilation_manager.py:829] Compiling select_from_array for worker0 select all logits.

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 16, 'index_size': 8}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 16, 'index_size': 8} finished in 0.03 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 16, 'index_size': 16}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 16, 'index_size': 16} finished in 0.04 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 8}

INFO 08-16 20:27:09 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 8} finished in 0.03 [secs].

INFO 08-16 20:27:09 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 16}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 16} finished in 0.04 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 32}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 32, 'index_size': 32} finished in 0.05 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 8}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 8} finished in 0.04 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 16}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 16} finished in 0.04 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 32}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 32} finished in 0.05 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 64}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 64, 'index_size': 64} finished in 0.06 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 8}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 8} finished in 0.04 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 16}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 16} finished in 0.05 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 32}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 32} finished in 0.06 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 64}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 64} finished in 0.07 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 128}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 128, 'index_size': 128} finished in 0.08 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 8}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 8} finished in 0.05 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 16}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 16} finished in 0.06 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 32}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 32} finished in 0.08 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 64}

INFO 08-16 20:27:10 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 64} finished in 0.08 [secs].

INFO 08-16 20:27:10 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 128}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 128} finished in 0.08 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 256}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 256, 'index_size': 256} finished in 0.08 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 8}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 8} finished in 0.08 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 16}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 16} finished in 0.08 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 32}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 32} finished in 0.09 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 64}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 64} finished in 0.10 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 128}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 128} finished in 0.09 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 256}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 512, 'index_size': 256} finished in 0.09 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 8}

INFO 08-16 20:27:11 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 8} finished in 0.14 [secs].

INFO 08-16 20:27:11 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 16}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 16} finished in 0.13 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 32}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 32} finished in 0.14 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 64}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 64} finished in 0.13 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 128}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 128} finished in 0.13 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 256}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 1024, 'index_size': 256} finished in 0.13 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 8}

INFO 08-16 20:27:12 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 8} finished in 0.22 [secs].

INFO 08-16 20:27:12 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 16}

INFO 08-16 20:27:13 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 16} finished in 0.23 [secs].

INFO 08-16 20:27:13 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 32}

INFO 08-16 20:27:13 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 32} finished in 0.23 [secs].

INFO 08-16 20:27:13 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 64}

INFO 08-16 20:27:13 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 64} finished in 0.20 [secs].

INFO 08-16 20:27:13 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 128}

INFO 08-16 20:27:13 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 128} finished in 0.20 [secs].

INFO 08-16 20:27:13 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 256}

INFO 08-16 20:27:13 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 2048, 'index_size': 256} finished in 0.19 [secs].

INFO 08-16 20:27:13 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 8}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 8} finished in 0.03 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 16}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 16} finished in 0.04 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 32}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 32} finished in 0.04 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 64}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 64} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 128}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 128} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 256}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 4096, 'index_size': 256} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 8}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 8} finished in 0.03 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 16}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 16} finished in 0.04 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 32}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 32} finished in 0.04 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 64}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 64} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 128}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 128} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 256}

INFO 08-16 20:27:14 [compilation_manager.py:188] Compilation of select_from_array [worker0 select all logits] --> {'array_size': 8192, 'index_size': 256} finished in 0.08 [secs].

INFO 08-16 20:27:14 [compilation_manager.py:909] Compiling compute_logits with different input shapes.

INFO 08-16 20:27:14 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 8}

INFO 08-16 20:27:15 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 8} finished in 0.17 [secs].

INFO 08-16 20:27:15 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 16}

INFO 08-16 20:27:15 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 16} finished in 0.18 [secs].

INFO 08-16 20:27:15 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 32}

INFO 08-16 20:27:15 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 32} finished in 0.21 [secs].

INFO 08-16 20:27:15 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 64}

INFO 08-16 20:27:15 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 64} finished in 0.24 [secs].

INFO 08-16 20:27:15 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 128}

INFO 08-16 20:27:16 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 128} finished in 0.31 [secs].

INFO 08-16 20:27:16 [compilation_manager.py:137] Precompile worker0 compute_logits --> {'num_reqs': 256}

INFO 08-16 20:27:16 [compilation_manager.py:188] Compilation of worker0 compute_logits --> {'num_reqs': 256} finished in 0.43 [secs].

INFO 08-16 20:27:16 [compilation_manager.py:1885] Compiling structured_decoding with different input shapes.

INFO 08-16 20:27:16 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 8}

INFO 08-16 20:27:21 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 8} finished in 5.21 [secs].

INFO 08-16 20:27:21 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 16}

INFO 08-16 20:27:26 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 16} finished in 4.86 [secs].

INFO 08-16 20:27:26 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 32}

INFO 08-16 20:27:28 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 32} finished in 1.22 [secs].

INFO 08-16 20:27:28 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 64}

INFO 08-16 20:27:28 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 64} finished in 0.42 [secs].

INFO 08-16 20:27:28 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 128}

INFO 08-16 20:27:28 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 128} finished in 0.45 [secs].

INFO 08-16 20:27:29 [compilation_manager.py:137] Precompile structured_decode --> {'num_reqs': 256}

INFO 08-16 20:27:29 [compilation_manager.py:188] Compilation of structured_decode --> {'num_reqs': 256} finished in 0.38 [secs].

INFO 08-16 20:27:29 [compilation_manager.py:228] Warm-up call pass finished in 0.01 [secs] over 6 tasks.

INFO 08-16 20:27:29 [core.py:340] init engine (profile, create kv cache, warmup model) took 217.29 s (compilation: 153.49 s)

model loaded
```
</div>

If this raises, the reason is in `vllm_init.log`. The usual causes are a
missing Kaggle license acceptance and a `max_model_len` too large for the
memory left after the model is loaded.

---
## Generate

`generate()` returns one `RequestOutput` per prompt. Pass a list to batch
several at once; vLLM schedules them together.


```python
prompt = "The future of artificial intelligence is"

output = llm.generate(prompt)[0]
print(output.prompt + output.outputs[0].text)
```


<div class="k-default-codeblock">
```
Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
```
</div>

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]

<div class="k-default-codeblock">
```
WARNING 08-16 20:27:30 [tpu_runner.py:1568] Should not schedule a request that does nothing!
```
</div>

Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.21it/s, est. speed input: 22.48 toks/s, output: 51.39 toks/s]

    
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.21it/s, est. speed input: 22.48 toks/s, output: 51.39 toks/s]

    
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  3.20it/s, est. speed input: 22.48 toks/s, output: 51.39 toks/s]

<div class="k-default-codeblock">
```
The future of artificial intelligence is being shaped by a complex interplay of factors - technological advancements, ethical considerations, and
```
</div>

---
## Sampling

Called with no `SamplingParams`, `KerasHubLLM` samples the way the preset
would sample under `CausalLM.generate()`. Every KerasHub task compiles itself
with `CausalLM.compile`'s default sampler, which is `top_k`, and that setting
is translated into the equivalent vLLM `SamplingParams` for you. This is the
one place `KerasHubLLM` deliberately departs from `vllm.LLM`, which would
otherwise apply its own generic defaults and quietly change how the model
behaves.

Length is not part of a KerasHub sampler, so vLLM's default of 16 output
tokens applies. That is short. Pass `SamplingParams` when you want more.


```python
from vllm import SamplingParams

greedy = SamplingParams(temperature=0.0, max_tokens=48)

output = llm.generate(prompt, greedy)[0]
print(output.prompt + output.outputs[0].text)
```


<div class="k-default-codeblock">
```
Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
```
</div>

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]

<div class="k-default-codeblock">
```
WARNING 08-16 20:27:30 [tpu_runner.py:1568] Should not schedule a request that does nothing!
```
</div>

Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.80it/s, est. speed input: 47.63 toks/s, output: 326.54 toks/s]

    
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.80it/s, est. speed input: 47.63 toks/s, output: 326.54 toks/s]

    
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  6.75it/s, est. speed input: 47.63 toks/s, output: 326.54 toks/s]

<div class="k-default-codeblock">
```
The future of artificial intelligence is a topic of intense debate and speculation. While many see it as a transformative force for good, others worry about its potential dangers.  

Here's a breakdown of key aspects of AI's future:

* **Narrow AI:**
```
</div>

Explicit `SamplingParams` override the model's own settings completely. Use
them whenever you want reproducible output: `temperature=0.0` is greedy
decoding, which makes the model pick the highest-probability token every step.

---
## Serving many requests

The reason to serve through vLLM is what happens under load. Here 32 requests
are submitted at once, as a rough stand-in for concurrent traffic. vLLM
schedules them continuously, so a request that finishes early frees its slot
instead of idling until the whole batch is done.

The prompts are deliberately all different. vLLM enables prefix caching by
default, so 32 copies of one prompt would let 31 of them skip most of the
prefill and report a throughput you would never see on real traffic.


```python
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
```


<div class="k-default-codeblock">
```
Rendering prompts:   0%|          | 0/32 [00:00<?, ?it/s]
```
</div>

Processed prompts:   0%|          | 0/32 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]

<div class="k-default-codeblock">
```
WARNING 08-16 20:27:31 [tpu_runner.py:1568] Should not schedule a request that does nothing!
```
</div>

Processed prompts:   3%|▎         | 1/32 [00:00<00:12,  2.41it/s, est. speed input: 33.79 toks/s, output: 154.45 toks/s]

    
Processed prompts: 100%|██████████| 32/32 [00:00<00:00,  2.41it/s, est. speed input: 1131.44 toks/s, output: 4930.16 toks/s]

    
Processed prompts: 100%|██████████| 32/32 [00:00<00:00, 76.95it/s, est. speed input: 1131.44 toks/s, output: 4930.16 toks/s]

    



<div class="k-default-codeblock">
```
Rendering prompts:   0%|          | 0/32 [00:00<?, ?it/s]
```
</div>

Processed prompts:   0%|          | 0/32 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]

<div class="k-default-codeblock">
```
WARNING 08-16 20:27:31 [tpu_runner.py:1568] Should not schedule a request that does nothing!
```
</div>

Processed prompts:   3%|▎         | 1/32 [00:00<00:09,  3.12it/s, est. speed input: 43.70 toks/s, output: 199.75 toks/s]

    
Processed prompts: 100%|██████████| 32/32 [00:00<00:00,  3.12it/s, est. speed input: 1463.72 toks/s, output: 6378.02 toks/s]

    
Processed prompts: 100%|██████████| 32/32 [00:00<00:00, 99.52it/s, est. speed input: 1463.72 toks/s, output: 6378.02 toks/s]

<div class="k-default-codeblock">
```
32 requests, 2048 tokens in 0.34s
5993 output tokens/s
```
</div>

The first `generate()` call is a warmup and is not timed. The engine
precompiles its shape buckets at startup, so most of the compile cost is
already paid by then, but the first call still touches things a timed run
should not have to pay for.

---
## Supported models

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

Two limits are worth knowing before you plan around this. Only `CausalLM`
presets are served, since this integration targets autoregressive text
generation. And vision-language models are not supported: the attention path
does not currently accept the custom mask a bidirectional image encoder needs.

---
## Next steps

- Try a different preset from the table above. Load one model per runtime
  session, since each engine holds the TPU for itself.
- Compare against `CausalLM.generate()` on the same weights. The gap is small
  for one request at a time and grows with concurrency and prompt length,
  which is exactly the workload vLLM is for.
- Read `vllm_init.log` once, even on a successful run. It shows the
  KV cache size vLLM settled on and how many shapes it compiled, which are
  the two numbers to adjust if you run out of memory.
