"""
Title: Serving Gemma with vLLM 
Author: Dhiraj 
Date created: 2025/08/16
Last modified: 2025/08/18
Description: Export Gemma models from KerasHub to Hugging Face and serve with vLLM for fast inference.
Accelerator: TPU and GPU
"""

"""
## Introduction

This guide demonstrates how to export Gemma models from KerasHub to the Hugging Face format and serve them using vLLM for efficient, high-throughput inference. We'll walk through the process step-by-step, from loading a pre-trained Gemma model in KerasHub to running inferences with vLLM in a Google Colab environment.

vLLM is an optimized serving engine for large language models that leverages techniques like PagedAttention to enable continuous batching and high GPU utilization. By exporting KerasHub models to a compatible format, you can take advantage of vLLM's performance benefits while starting from the Keras ecosystem

At present, this is supported only for Gemma 2 and its presets. In the future, there will be more coverage of the models in KerasHub.

**Note:** We'll perform the model export on a TPU runtime (for efficiency with larger models) and then switch to a GPU runtime for serving with vLLM, as vLLM [does not support TPU v2 on Colab](https://docs.vllm.ai/en/v0.5.5/getting_started/tpu-installation.html)
"""

"""
## Setup

First, install the required libraries. Select a TPU runtime in Colab before running these cells.

"""

"""shell
!pip install -q --upgrade keras-hub huggingface-hub
"""

import keras_hub
from huggingface_hub import snapshot_download
import os
import shutil
import json
"""
## Loading and Exporting the Model

Load a pre-trained Gemma 2 model from KerasHub using the 'gemma2_instruct_2b_en' preset. This is an instruction-tuned variant suitable for conversational tasks.

**Note:** The export method needs to map the weights from Keras to safetensors, hence requiring double the RAM needed to load a preset. This is also the reason why we are running on a TPU instance in Colab as it offers more VRAM instead of GPU.
"""

# Load the pre-trained Gemma model
model_preset = 'gemma2_instruct_2b_en'
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(model_preset)
print("✅ Gemma model loaded successfully")

# Set export path
export_path = "./gemma_exported"

# Export to Hugging Face format
gemma_lm.export_to_transformers(export_path)
print(f"Model exported successfully to {export_path}")

"""
## Downloading Additional Metadata

vLLM requires complete Hugging Face model configuration files. Download these from the original Gemma repository on Hugging Face.
"""

SERVABLE_CKPT_DIR = "./gemma_exported"

# Download metadata files
snapshot_download(repo_id="google/gemma-2-2b-it", allow_patterns="*.json", local_dir=SERVABLE_CKPT_DIR)
print("✅ Metadata files downloaded")

"""
## Updating the Model Index

The exported model uses a single safetensors file, unlike the original which may have multiple shards. Update the index file to reflect this.
"""

index_path = os.path.join(SERVABLE_CKPT_DIR, "model.safetensors.index.json")
with open(index_path, "r") as f:
    index_data = json.load(f)

# Replace shard references with single file
for key in index_data.get("weight_map", {}):
    index_data["weight_map"][key] = "model.safetensors"

with open(index_path, "w") as f:
    json.dump(index_data, f, indent=2)

print("✅ Model index updated")

# Verify the directory contents
print("Directory contents:")
for file in os.listdir(SERVABLE_CKPT_DIR):
    size = os.path.getsize(os.path.join(SERVABLE_CKPT_DIR, file)) / (1024 * 1024)
    print(f"{file}: {size:.2f} MB")

"""
## Saving to Google Drive

Save the files to Google Drive. This is needed because vLLM currently [does not support TPU v2 on Colab](https://docs.vllm.ai/en/v0.5.5/getting_started/tpu-installation.html) and cannot dynamically switch the backend to CPU. Switch to a different Colab GPU instance for serving after saving. If you are using Cloud TPU or GPU from the start, you may skip this step.

**Note:** the `model.safetensors` file is ~9.5GB for Gemma 2B, so ensure you have enough space in your Google Drive.
"""

from google.colab import drive

drive.mount("/content/drive")

drive_dir = "/content/drive/MyDrive/gemma_exported"

# Remove any existing exports with the same name
if os.path.exists(drive_dir):
    shutil.rmtree(drive_dir)
    print("✅ Existing export removed")

# Copy the exported model to Google Drive
shutil.copytree(SERVABLE_CKPT_DIR, drive_dir)
print("✅ Model copied to Google Drive")
"""
Verify the file sizes to ensure no corruption during copy. Here's how they should appear:
"""

print("Drive directory contents:")
for file in os.listdir(drive_dir):
    size = os.path.getsize(os.path.join(drive_dir, file)) / (1024 * 1024)
    print(f"{file}: {size:.2f} MB")

"""
Disconnect TPU runtime (if applicable) and re-connect with a T4 GPU runtime before proceeding.
"""

from google.colab import drive

drive.mount("/content/drive")

SERVABLE_CKPT_DIR = "/content/drive/MyDrive/gemma_exported"

print("Drive directory contents:")
for file in os.listdir(SERVABLE_CKPT_DIR):
    size = os.path.getsize(os.path.join(SERVABLE_CKPT_DIR, file)) / (1024 * 1024)
    print(f"{file}: {size:.2f} MB")


"""
## Install vLLM
"""

"""shell
!pip install -q vllm
"""

"""
## Instantiating vLLM

Load the exported model into vLLM for serving.
"""
from vllm import LLM, SamplingParams

llm = LLM(model=SERVABLE_CKPT_DIR, load_format="safetensors", dtype="float32")
print("✅ vLLM engine initialized")

"""
## Generating with vLLM

First, test with a simple prompt to verify the setup.
"""
simple_prompt = "Hello, what is vLLM?"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
outputs = llm.generate(simple_prompt, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}\nGenerated text: {output.outputs[0].text}")
    
"""
As we have loaded the weights of the Gemma instruct model, let's use a formatted example with the chat template.
"""
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start} and {solution_end}"""

TEMPLATE = """
<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


question = (
    "Trevor and two of his neighborhood friends go to the toy shop every year "
    "to buy toys. Trevor always spends $20 more than his friend Reed on toys, "
    "and Reed spends 2 times as much money as their friend Quinn on the toys. "
    "If Trevor spends $80 every year to buy his toys, calculate how much money "
    "in total the three spend in 4 years."
)
prompts = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)]

sampling_params = SamplingParams(temperature=0.9, top_p=0.92, max_tokens=768)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print("===============================")
    print(f"Prompt: {output.prompt}\nGenerated text: {output.outputs[0].text}")

"""
## Conclusion

You've now successfully exported a KerasHub Gemma model to Hugging Face format and served it with vLLM for efficient inference. This setup enables high-throughput generation, suitable for production or batch processing.

Experiment with different prompts, sampling parameters, or larger Gemma variants (ensure sufficient GPU memory). For deployment beyond Colab, consider Docker containers or cloud instances.

Happy experimenting!
"""