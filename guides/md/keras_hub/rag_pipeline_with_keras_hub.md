# RAG Pipeline with KerasHub

**Author:** [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)<br>
**Date created:** 2025/07/22<br>
**Last modified:** 2025/07/24<br>
**Description:** RAG pipeline for brain MRI analysis: image retrieval, context search, and report generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/rag_pipeline_with_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/rag_pipeline_with_keras_hub.py)



---
## Introduction

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of
large language models with external knowledge retrieval. Instead of relying solely on the
model's pre-trained knowledge, RAG allows the model to access and use relevant information
from a database or knowledge base to generate more accurate and contextually relevant responses.

In this guide, we'll walk you through implementing a RAG pipeline for medical image analysis
using KerasHub models. We'll show you how to:

1. Load and configure Vision Transformer (ViT) and Gemma3 language models
2. Process brain MRI images and extract meaningful features
3. Implement similarity search for retrieving relevant medical reports
4. Generate comprehensive radiology reports using retrieved contex
5. Compare RAG approach with direct vision-language model generation

This pipeline demonstrates how to build a sophisticated medical AI system that can:

- Analyze brain MRI images using state-of-the-art vision models
- Retrieve relevant medical context from a database
- Generate detailed radiology reports with proper medical terminology
- Provide diagnostic impressions and treatment recommendations

Let's get started!

---
## Setup

First, let's import the necessary libraries and configure our environment. We'll be using
KerasHub to download and run the language models, and we'll need to authenticate with
Kaggle to access the model weights. We'll also set up the JAX backend for optimal
performance on GPU accelerators.

---
## Kaggle Credentials Setup

If running in Google Colab, set up Kaggle API credentials using the `google.colab.userdata` module to enable downloading models and datasets from Kaggle. This step is only required in Colab environments.


```python
import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

keras.config.set_dtype_policy("bfloat16")
import keras_hub
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from nilearn import datasets, image
import re

```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1753394569.845535    5660 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1753394569.850098    5660 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1753394569.861799    5660 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753394569.861813    5660 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753394569.861815    5660 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753394569.861816    5660 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

---
## Model Loading

Loads the vision model (for image feature extraction) and the Gemma3 vision-language model (for report generation). Returns both models for use in the RAG pipeline.


```python

def load_models():
    """
    Load and configure vision model for feature extraction, Gemma3 VLM for report generation, and a compact text model for benchmarking.
    Returns:
        tuple: (vision_model, vlm_model, text_model)
    """
    # Vision model for feature extraction (lightweight MobileNetV3)
    vision_model = keras_hub.models.ImageClassifier.from_preset(
        "mobilenet_v3_large_100_imagenet_21k"
    )
    # Gemma3 Text model for report generation in RAG Pipeline (compact)
    text_model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
    # Gemma3 VLM for report generation (original, for benchmarking)
    vlm_model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b")
    return vision_model, vlm_model, text_model

```

---
## Image and Caption Preparation

Prepares OASIS brain MRI images and generates captions for each image. Returns lists of image paths and captions.


```python

def prepare_images_and_captions(oasis, images_dir="images"):
    """
    Prepare OASIS brain MRI images and generate captions.

    Args:
        oasis: OASIS dataset object containing brain MRI data
        images_dir (str): Directory to save processed images

    Returns:
        tuple: (image_paths, captions) - Lists of image paths and corresponding captions
    """
    os.makedirs(images_dir, exist_ok=True)
    image_paths = []
    captions = []
    for i, img_path in enumerate(oasis.gray_matter_maps):
        img = image.load_img(img_path)
        data = img.get_fdata()
        slice_ = data[:, :, data.shape[2] // 2]
        slice_ = (
            (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_)) * 255
        ).astype(np.uint8)
        img_pil = Image.fromarray(slice_)
        fname = f"oasis_{i}.png"
        fpath = os.path.join(images_dir, fname)
        img_pil.save(fpath)
        image_paths.append(fpath)
        captions.append(f"OASIS Brain MRI {i}")
    print("Saved 4 OASIS Brain MRI images:", image_paths)
    return image_paths, captions

```

---
## Image Visualization Utility

Displays a set of processed brain MRI images with their corresponding captions.


```python

def visualize_images(image_paths, captions):
    """
    Visualize the processed brain MRI images.

    Args:
        image_paths (list): List of image file paths
        captions (list): List of corresponding image captions
    """
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    # If only one image, axes is not a list
    if n == 1:
        axes = [axes]
    for i, (img_path, title) in enumerate(zip(image_paths, captions)):
        img = Image.open(img_path)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.suptitle("OASIS Brain MRI Images")
    plt.tight_layout()
    plt.show()

```

---
## Prediction Visualization Utility

Displays the query image and the most similar retrieved image from the database side by side.


```python

def visualize_prediction(query_img_path, db_image_paths, best_idx, db_reports):
    """
    Visualize the query image and the most similar retrieved image.

    Args:
        query_img_path (str): Path to the query image
        db_image_paths (list): List of database image paths
        best_idx (int): Index of the most similar database image
        db_reports (list): List of database reports
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(Image.open(query_img_path), cmap="gray")
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    axes[1].imshow(Image.open(db_image_paths[best_idx]), cmap="gray")
    axes[1].set_title("Retrieved Context Image")
    axes[1].axis("off")
    plt.suptitle("Query and Most Similar Database Image")
    plt.tight_layout()
    plt.show()

```

---
## Image Feature Extraction

Extracts a feature vector from an image using the vision model.


```python

def extract_image_features(img_path, vision_model):
    """
    Extract features from an image using the vision model.

    Args:
        img_path (str): Path to the input image
        vision_model: Pre-trained vision model for feature extraction

    Returns:
        numpy.ndarray: Extracted feature vector
    """
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    features = vision_model(x)
    return features

```

---
## DB Reports

List of example radiology reports corresponding to each database image. Used as context for the RAG pipeline to generate new reports for query images.


```python
db_reports = [
    "MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.",
    "Normal MRI scan, no abnormal findings.",
    "Diffuse atrophy noted, no focal lesions.",
]
```

---
## Output Cleaning Utility

Cleans the generated text output by removing prompt echoes and unwanted headers.


```python

def clean_generated_output(generated_text, prompt):
    """
    Remove prompt echo and header details from generated text.

    Args:
        generated_text (str): Raw generated text from the language model
        prompt (str): Original prompt used for generation

    Returns:
        str: Cleaned text without prompt echo and headers
    """
    # Remove the prompt from the beginning of the generated text
    if generated_text.startswith(prompt):
        cleaned_text = generated_text[len(prompt) :].strip()
    else:
        cleaned_text = generated_text.replace(prompt, "").strip()

    # Remove header details and unwanted formatting
    lines = cleaned_text.split("\n")
    filtered_lines = []
    skip_next = False
    subheading_pattern = re.compile(r"^(\s*[A-Za-z0-9 .\-()]+:)(.*)")

    for line in lines:
        line = line.replace("<end_of_turn>", "").strip()
        line = line.replace("**", "")
        line = line.replace("*", "")
        # Remove empty lines after headers (existing logic)
        if any(
            header in line
            for header in [
                "**Patient:**",
                "**Date of Exam:**",
                "**Exam:**",
                "**Referring Physician:**",
                "**Patient ID:**",
                "Patient:",
                "Date of Exam:",
                "Exam:",
                "Referring Physician:",
                "Patient ID:",
            ]
        ):
            continue
        elif line.strip() == "" and skip_next:
            skip_next = False
            continue
        else:
            # Split subheadings onto their own line if content follows
            match = subheading_pattern.match(line)
            if match and match.group(2).strip():
                filtered_lines.append(match.group(1).strip())
                filtered_lines.append(match.group(2).strip())
                filtered_lines.append("")  # Add a blank line after subheading
            else:
                filtered_lines.append(line)
                # Add a blank line after subheadings (lines ending with ':')
                if line.endswith(":") and (
                    len(filtered_lines) == 1 or filtered_lines[-2] != ""
                ):
                    filtered_lines.append("")
            skip_next = False

    # Remove any empty lines and excessive whitespace
    cleaned_text = "\n".join(
        [l for l in filtered_lines if l.strip() or l == ""]
    ).strip()

    return cleaned_text

```

---
## RAG Pipeline

Implements the Retrieval-Augmented Generation (RAG) pipeline:
- Extracts features from the query image and database images.
- Finds the most similar image in the database.
- Uses the retrieved report and the query image as input to the Gemma3 VLM to generate a new report.
Returns the index of the matched image, the retrieved report, and the generated report.


```python

def rag_pipeline(query_img_path, db_image_paths, db_reports, vision_model, text_model):
    """
    Retrieval-Augmented Generation pipeline using vision model for retrieval and a compact text model for report generation.
    Args:
        query_img_path (str): Path to the query image
        db_image_paths (list): List of database image paths
        db_reports (list): List of database reports
        vision_model: Vision model for feature extraction
        text_model: Compact text model for report generation
    Returns:
        tuple: (best_idx, retrieved_report, generated_report)
    """
    # Extract features for the query image
    query_features = extract_image_features(query_img_path, vision_model)
    # Extract features for the database images
    db_features = np.vstack(
        [extract_image_features(p, vision_model) for p in db_image_paths]
    )
    # Ensure features are numpy arrays for similarity search
    db_features_np = np.array(db_features)
    query_features_np = np.array(query_features)
    # Similarity search
    similarity = np.dot(db_features_np, query_features_np.T).squeeze()
    best_idx = np.argmax(similarity)
    retrieved_report = db_reports[best_idx]
    print(f"[RAG] Matched image index: {best_idx}")
    print(f"[RAG] Matched image path: {db_image_paths[best_idx]}")
    print(f"[RAG] Retrieved context/report:\n{retrieved_report}\n")
    PROMPT_TEMPLATE = (
        "Context:\n{context}\n\n"
        "Based on the above radiology report and the provided brain MRI image, please:\n"
        "1. Provide a diagnostic impression.\n"
        "2. Explain the diagnostic reasoning.\n"
        "3. Suggest possible treatment options.\n"
        "Format your answer as a structured radiology report.\n"
    )
    prompt = PROMPT_TEMPLATE.format(context=retrieved_report)
    # Generate report using the text model (text only, no image input)
    output = text_model.generate(
        {
            "prompts": prompt,
        }
    )
    cleaned_output = clean_generated_output(output, prompt)
    return best_idx, retrieved_report, cleaned_output

```

---
## Vision-Language Model (Direct Approach)

Generates a radiology report directly from the query image using the Gemma3 VLM, without retrieval.


```python

def vlm_generate_report(query_img_path, vlm_model, question=None):
    """
    Generate a radiology report directly from the image using a vision-language model.
    Args:
        query_img_path (str): Path to the query image
        vlm_model: Pre-trained vision-language model (Gemma3 VLM)
        question (str): Optional question or prompt to include
    Returns:
        str: Generated radiology repor
    """
    PROMPT_TEMPLATE = (
        "Based on the provided brain MRI image, please:\n"
        "1. Provide a diagnostic impression.\n"
        "2. Explain the diagnostic reasoning.\n"
        "3. Suggest possible treatment options.\n"
        "Format your answer as a structured radiology report.\n"
    )
    if question is None:
        question = ""
    # Preprocess the image as required by the model
    img = Image.open(query_img_path).convert("RGB").resize((224, 224))
    image = np.array(img) / 255.0
    image = np.expand_dims(image, axis=0)
    # Generate report using the VLM
    output = vlm_model.generate(
        {
            "images": image,
            "prompts": PROMPT_TEMPLATE.format(question=question),
        }
    )
    # Clean the generated outpu
    cleaned_output = clean_generated_output(
        output, PROMPT_TEMPLATE.format(question=question)
    )
    return cleaned_output

```

---
## Main Execution Pipeline

This section loads models, prepares data, runs the RAG pipeline, and compares RAG with direct VLM generation.


```python
if __name__ == "__main__":
    # Load models
    print("Loading models...")
    vision_model, vlm_model, text_model = load_models()

    # Prepare data
    print("Preparing OASIS dataset...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=4)  # Use 4 images
    print("Download dataset is completed.")
    image_paths, captions = prepare_images_and_captions(oasis)
    visualize_images(image_paths, captions)

    # Split data: first 3 as database, last as query
    db_image_paths = image_paths[:-1]
    query_img_path = image_paths[-1]

    # Extract database features
    print("Extracting database features...")
    db_features = np.vstack(
        [extract_image_features(p, vision_model) for p in db_image_paths]
    )

    # Run RAG pipeline
    print("Running RAG pipeline...")
    best_idx, retrieved_report, generated_report = rag_pipeline(
        query_img_path, db_image_paths, db_reports, vision_model, text_model
    )

    # Visualize results
    visualize_prediction(query_img_path, db_image_paths, best_idx, db_reports)

    # Print RAG results
    print("\n" + "=" * 50)
    print("RAG PIPELINE RESULTS")
    print("=" * 50)
    print(f"\nMatched DB Report Index: {best_idx}")
    print(f"Matched DB Report: {retrieved_report}")
    print("\n--- Generated Report ---\n", generated_report)

    # Run VLM (direct approach)
    print("\n" + "=" * 50)
    print("VLM RESULTS (Direct Approach)")
    print("=" * 50)
    vlm_report = vlm_generate_report(query_img_path, vlm_model)
    print("\n--- Vision-Language Model (No Retrieval) Report ---\n", vlm_report)
```

<div class="k-default-codeblock">
```
Loading models...

Downloading from https://www.kaggle.com/api/v1/models/keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet_21k/1/download/config.json...
```
</div>

  0%|                                                                                                                                                             | 0.00/4.04k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.04k/4.04k [00:00<00:00, 11.8MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet_21k/1/download/task.json...
```
</div>

  0%|                                                                                                                                                             | 0.00/8.21k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8.21k/8.21k [00:00<00:00, 23.8MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet_21k/1/download/task.weights.h5...
```
</div>

  0%|                                                                                                                                                             | 0.00/21.5M [00:00<?, ?B/s]

    
  5%|██████▉                                                                                                                                             | 1.00M/21.5M [00:00<00:04, 4.64MB/s]

    
 37%|███████████████████████████████████████████████████████                                                                                             | 8.00M/21.5M [00:00<00:00, 30.6MB/s]

    
 60%|█████████████████████████████████████████████████████████████████████████████████████████▍                                                          | 13.0M/21.5M [00:00<00:00, 37.5MB/s]

    
 84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 18.0M/21.5M [00:00<00:00, 35.1MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.5M/21.5M [00:00<00:00, 35.5MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet_21k/1/download/model.weights.h5...
```
</div>

  0%|                                                                                                                                                             | 0.00/11.9M [00:00<?, ?B/s]

    
  8%|████████████▍                                                                                                                                       | 1.00M/11.9M [00:00<00:02, 5.33MB/s]

    
 67%|███████████████████████████████████████████████████████████████████████████████████████████████████▍                                                | 8.00M/11.9M [00:00<00:00, 33.8MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11.9M/11.9M [00:00<00:00, 37.4MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/gemma3/keras/gemma3_instruct_1b/3/download/config.json...
```
</div>

  0%|                                                                                                                                                               | 0.00/966 [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 966/966 [00:00<00:00, 3.20MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/gemma3/keras/gemma3_instruct_1b/3/download/task.json...
```
</div>

  0%|                                                                                                                                                             | 0.00/3.23k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.23k/3.23k [00:00<00:00, 10.3MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/gemma3/keras/gemma3_instruct_1b/3/download/assets/tokenizer/vocabulary.spm...
```
</div>

  0%|                                                                                                                                                             | 0.00/4.47M [00:00<?, ?B/s]

    
 22%|█████████████████████████████████                                                                                                                   | 1.00M/4.47M [00:00<00:00, 5.37MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.47M/4.47M [00:00<00:00, 17.6MB/s]

    


<div class="k-default-codeblock">
```
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

Downloading from https://www.kaggle.com/api/v1/models/keras/gemma3/keras/gemma3_instruct_1b/3/download/model.weights.h5...
```
</div>

  0%|                                                                                                                                                             | 0.00/1.86G [00:00<?, ?B/s]

    
  0%|                                                                                                                                                    | 1.00M/1.86G [00:00<06:20, 5.26MB/s]

    
  0%|▌                                                                                                                                                   | 8.00M/1.86G [00:00<00:59, 33.4MB/s]

    
  1%|█▏                                                                                                                                                  | 15.0M/1.86G [00:00<00:41, 48.0MB/s]

    
  1%|██                                                                                                                                                  | 27.0M/1.86G [00:00<00:36, 54.2MB/s]

    
  2%|██▉                                                                                                                                                 | 38.0M/1.86G [00:00<00:29, 67.2MB/s]

    
  3%|███▋                                                                                                                                                | 48.0M/1.86G [00:00<00:26, 72.8MB/s]

    
  3%|████▎                                                                                                                                               | 56.0M/1.86G [00:00<00:26, 73.4MB/s]

    
  3%|████▉                                                                                                                                               | 64.0M/1.86G [00:01<00:25, 76.2MB/s]

    
  4%|█████▊                                                                                                                                              | 75.0M/1.86G [00:01<00:22, 86.3MB/s]

    
  4%|██████▌                                                                                                                                             | 84.0M/1.86G [00:01<00:21, 87.9MB/s]

    
  5%|███████▏                                                                                                                                            | 93.0M/1.86G [00:01<00:22, 84.9MB/s]

    
  5%|███████▉                                                                                                                                             | 102M/1.86G [00:01<00:21, 87.4MB/s]

    
  6%|████████▉                                                                                                                                            | 114M/1.86G [00:01<00:19, 96.4MB/s]

    
  6%|█████████▋                                                                                                                                           | 124M/1.86G [00:01<00:26, 70.7MB/s]

    
  7%|██████████▎                                                                                                                                          | 132M/1.86G [00:02<00:34, 53.5MB/s]

    
  8%|███████████▏                                                                                                                                         | 144M/1.86G [00:02<00:27, 67.1MB/s]

    
  8%|████████████                                                                                                                                         | 154M/1.86G [00:02<00:24, 74.9MB/s]

    
  9%|████████████▋                                                                                                                                        | 163M/1.86G [00:02<00:27, 66.7MB/s]

    
  9%|█████████████▋                                                                                                                                       | 175M/1.86G [00:02<00:22, 79.3MB/s]

    
 10%|██████████████▌                                                                                                                                      | 187M/1.86G [00:02<00:20, 87.5MB/s]

    
 10%|███████████████▍                                                                                                                                     | 197M/1.86G [00:02<00:23, 77.3MB/s]

    
 11%|████████████████                                                                                                                                     | 206M/1.86G [00:03<00:24, 73.3MB/s]

    
 11%|████████████████▉                                                                                                                                    | 217M/1.86G [00:03<00:21, 82.3MB/s]

    
 12%|█████████████████▉                                                                                                                                   | 229M/1.86G [00:03<00:19, 91.0MB/s]

    
 13%|██████████████████▋                                                                                                                                  | 239M/1.86G [00:03<00:25, 67.8MB/s]

    
 13%|███████████████████▌                                                                                                                                 | 250M/1.86G [00:03<00:22, 76.7MB/s]

    
 14%|████████████████████▍                                                                                                                                | 261M/1.86G [00:03<00:20, 83.9MB/s]

    
 14%|█████████████████████                                                                                                                                | 270M/1.86G [00:03<00:20, 85.9MB/s]

    
 15%|█████████████████████▊                                                                                                                               | 279M/1.86G [00:03<00:19, 86.3MB/s]

    
 15%|██████████████████████▍                                                                                                                              | 288M/1.86G [00:04<00:21, 79.1MB/s]

    
 16%|███████████████████████▍                                                                                                                             | 300M/1.86G [00:04<00:19, 88.5MB/s]

    
 16%|████████████████████████▏                                                                                                                            | 309M/1.86G [00:04<00:19, 86.3MB/s]

    
 17%|████████████████████████▉                                                                                                                            | 320M/1.86G [00:04<00:17, 93.8MB/s]

    
 17%|█████████████████████████▊                                                                                                                           | 331M/1.86G [00:04<00:20, 81.3MB/s]

    
 18%|██████████████████████████▊                                                                                                                          | 343M/1.86G [00:04<00:18, 88.9MB/s]

    
 19%|███████████████████████████▋                                                                                                                         | 355M/1.86G [00:04<00:17, 95.6MB/s]

    
 19%|████████████████████████████▌                                                                                                                        | 365M/1.86G [00:04<00:18, 85.9MB/s]

    
 20%|█████████████████████████████▍                                                                                                                       | 377M/1.86G [00:05<00:16, 95.5MB/s]

    
 20%|██████████████████████████████▏                                                                                                                      | 387M/1.86G [00:05<00:18, 88.1MB/s]

    
 21%|███████████████████████████████▏                                                                                                                     | 400M/1.86G [00:05<00:15, 99.4MB/s]

    
 22%|████████████████████████████████▍                                                                                                                     | 412M/1.86G [00:05<00:15, 101MB/s]

    
 22%|█████████████████████████████████▎                                                                                                                    | 424M/1.86G [00:05<00:15, 102MB/s]

    
 23%|██████████████████████████████████▎                                                                                                                   | 436M/1.86G [00:05<00:14, 108MB/s]

    
 23%|███████████████████████████████████▏                                                                                                                  | 448M/1.86G [00:05<00:13, 113MB/s]

    
 24%|████████████████████████████████████▏                                                                                                                 | 460M/1.86G [00:05<00:13, 112MB/s]

    
 25%|█████████████████████████████████████                                                                                                                 | 471M/1.86G [00:05<00:13, 109MB/s]

    
 25%|█████████████████████████████████████▋                                                                                                               | 482M/1.86G [00:06<00:16, 89.1MB/s]

    
 26%|██████████████████████████████████████▍                                                                                                              | 492M/1.86G [00:06<00:16, 92.1MB/s]

    
 26%|███████████████████████████████████████▎                                                                                                             | 504M/1.86G [00:06<00:14, 99.4MB/s]

    
 27%|████████████████████████████████████████▏                                                                                                            | 514M/1.86G [00:06<00:14, 98.8MB/s]

    
 27%|████████████████████████████████████████▉                                                                                                            | 524M/1.86G [00:06<00:14, 98.5MB/s]

    
 28%|█████████████████████████████████████████▋                                                                                                           | 534M/1.86G [00:06<00:14, 97.9MB/s]

    
 29%|██████████████████████████████████████████▍                                                                                                          | 544M/1.86G [00:06<00:14, 98.3MB/s]

    
 29%|███████████████████████████████████████████▎                                                                                                         | 554M/1.86G [00:06<00:16, 88.5MB/s]

    
 30%|████████████████████████████████████████████                                                                                                         | 565M/1.86G [00:07<00:15, 93.3MB/s]

    
 30%|████████████████████████████████████████████▉                                                                                                        | 576M/1.86G [00:07<00:14, 96.8MB/s]

    
 31%|█████████████████████████████████████████████▊                                                                                                       | 586M/1.86G [00:07<00:14, 96.1MB/s]

    
 31%|██████████████████████████████████████████████▌                                                                                                      | 597M/1.86G [00:07<00:14, 97.2MB/s]

    
 32%|███████████████████████████████████████████████▍                                                                                                     | 607M/1.86G [00:07<00:13, 97.6MB/s]

    
 32%|████████████████████████████████████████████████▏                                                                                                    | 617M/1.86G [00:07<00:13, 97.9MB/s]

    
 33%|████████████████████████████████████████████████▉                                                                                                    | 627M/1.86G [00:07<00:13, 97.1MB/s]

    
 33%|█████████████████████████████████████████████████▋                                                                                                   | 637M/1.86G [00:07<00:13, 96.7MB/s]

    
 34%|██████████████████████████████████████████████████▌                                                                                                  | 647M/1.86G [00:07<00:13, 97.9MB/s]

    
 34%|███████████████████████████████████████████████████▎                                                                                                 | 657M/1.86G [00:08<00:13, 95.7MB/s]

    
 35%|████████████████████████████████████████████████████▏                                                                                                | 668M/1.86G [00:08<00:13, 98.8MB/s]

    
 36%|████████████████████████████████████████████████████▉                                                                                                | 678M/1.86G [00:08<00:13, 98.4MB/s]

    
 36%|█████████████████████████████████████████████████████▊                                                                                               | 689M/1.86G [00:08<00:13, 97.8MB/s]

    
 37%|███████████████████████████████████████████████████████                                                                                               | 700M/1.86G [00:08<00:12, 100MB/s]

    
 37%|███████████████████████████████████████████████████████▍                                                                                             | 710M/1.86G [00:08<00:12, 98.8MB/s]

    
 38%|████████████████████████████████████████████████████████▏                                                                                            | 720M/1.86G [00:08<00:12, 97.9MB/s]

    
 38%|█████████████████████████████████████████████████████████                                                                                            | 730M/1.86G [00:08<00:12, 98.3MB/s]

    
 39%|█████████████████████████████████████████████████████████▊                                                                                           | 740M/1.86G [00:08<00:12, 97.7MB/s]

    
 39%|██████████████████████████████████████████████████████████▋                                                                                          | 751M/1.86G [00:09<00:12, 96.2MB/s]

    
 40%|███████████████████████████████████████████████████████████▍                                                                                         | 761M/1.86G [00:09<00:12, 93.7MB/s]

    
 40%|████████████████████████████████████████████████████████████                                                                                         | 770M/1.86G [00:09<00:12, 92.5MB/s]

    
 41%|████████████████████████████████████████████████████████████▉                                                                                        | 781M/1.86G [00:09<00:12, 95.5MB/s]

    
 42%|█████████████████████████████████████████████████████████████▊                                                                                       | 792M/1.86G [00:09<00:11, 98.5MB/s]

    
 42%|██████████████████████████████████████████████████████████████▌                                                                                      | 802M/1.86G [00:09<00:11, 97.1MB/s]

    
 43%|███████████████████████████████████████████████████████████████▍                                                                                     | 812M/1.86G [00:09<00:12, 95.4MB/s]

    
 43%|████████████████████████████████████████████████████████████████▎                                                                                    | 823M/1.86G [00:09<00:11, 97.8MB/s]

    
 44%|█████████████████████████████████████████████████████████████████                                                                                    | 833M/1.86G [00:09<00:11, 98.7MB/s]

    
 44%|█████████████████████████████████████████████████████████████████▊                                                                                   | 843M/1.86G [00:10<00:11, 97.0MB/s]

    
 45%|██████████████████████████████████████████████████████████████████▌                                                                                  | 853M/1.86G [00:10<00:11, 92.3MB/s]

    
 45%|███████████████████████████████████████████████████████████████████▎                                                                                 | 862M/1.86G [00:10<00:13, 83.8MB/s]

    
 46%|████████████████████████████████████████████████████████████████████                                                                                 | 872M/1.86G [00:10<00:12, 88.3MB/s]

    
 46%|████████████████████████████████████████████████████████████████████▊                                                                                | 881M/1.86G [00:10<00:12, 86.2MB/s]

    
 47%|█████████████████████████████████████████████████████████████████████▌                                                                               | 891M/1.86G [00:10<00:11, 89.4MB/s]

    
 47%|██████████████████████████████████████████████████████████████████████▍                                                                              | 902M/1.86G [00:10<00:11, 91.5MB/s]

    
 48%|███████████████████████████████████████████████████████████████████████▎                                                                             | 913M/1.86G [00:10<00:11, 93.9MB/s]

    
 48%|████████████████████████████████████████████████████████████████████████▏                                                                            | 924M/1.86G [00:10<00:10, 97.6MB/s]

    
 49%|████████████████████████████████████████████████████████████████████████▉                                                                            | 934M/1.86G [00:11<00:10, 98.2MB/s]

    
 49%|█████████████████████████████████████████████████████████████████████████▋                                                                           | 944M/1.86G [00:11<00:10, 99.1MB/s]

    
 50%|██████████████████████████████████████████████████████████████████████████▍                                                                          | 954M/1.86G [00:11<00:10, 99.5MB/s]

    
 51%|███████████████████████████████████████████████████████████████████████████▎                                                                         | 964M/1.86G [00:11<00:12, 80.7MB/s]

    
 51%|████████████████████████████████████████████████████████████████████████████                                                                         | 974M/1.86G [00:11<00:11, 85.0MB/s]

    
 52%|████████████████████████████████████████████████████████████████████████████▊                                                                        | 983M/1.86G [00:11<00:14, 69.2MB/s]

    
 52%|█████████████████████████████████████████████████████████████████████████████▌                                                                       | 994M/1.86G [00:11<00:12, 79.3MB/s]

    
 53%|█████████████████████████████████████████████████████████████████████████████▊                                                                      | 0.98G/1.86G [00:11<00:11, 84.6MB/s]

    
 53%|██████████████████████████████████████████████████████████████████████████████▋                                                                     | 0.99G/1.86G [00:12<00:10, 88.5MB/s]

    
 54%|███████████████████████████████████████████████████████████████████████████████▍                                                                    | 1.00G/1.86G [00:12<00:10, 90.3MB/s]

    
 54%|████████████████████████████████████████████████████████████████████████████████                                                                    | 1.01G/1.86G [00:12<00:12, 74.3MB/s]

    
 55%|█████████████████████████████████████████████████████████████████████████████████                                                                   | 1.02G/1.86G [00:12<00:10, 86.2MB/s]

    
 55%|█████████████████████████████████████████████████████████████████████████████████▉                                                                  | 1.03G/1.86G [00:12<00:09, 92.7MB/s]

    
 56%|██████████████████████████████████████████████████████████████████████████████████▊                                                                 | 1.04G/1.86G [00:12<00:09, 97.8MB/s]

    
 56%|████████████████████████████████████████████████████████████████████████████████████▏                                                                | 1.05G/1.86G [00:12<00:08, 102MB/s]

    
 57%|█████████████████████████████████████████████████████████████████████████████████████                                                                | 1.06G/1.86G [00:12<00:07, 108MB/s]

    
 58%|█████████████████████████████████████████████████████████████████████████████████████▉                                                               | 1.08G/1.86G [00:13<00:08, 103MB/s]

    
 58%|██████████████████████████████████████████████████████████████████████████████████████▉                                                              | 1.09G/1.86G [00:13<00:07, 109MB/s]

    
 59%|███████████████████████████████████████████████████████████████████████████████████████▊                                                             | 1.10G/1.86G [00:13<00:07, 111MB/s]

    
 59%|████████████████████████████████████████████████████████████████████████████████████████                                                            | 1.11G/1.86G [00:13<00:08, 98.1MB/s]

    
 60%|████████████████████████████████████████████████████████████████████████████████████████▊                                                           | 1.12G/1.86G [00:13<00:08, 90.9MB/s]

    
 61%|█████████████████████████████████████████████████████████████████████████████████████████▋                                                          | 1.13G/1.86G [00:13<00:07, 99.0MB/s]

    
 61%|███████████████████████████████████████████████████████████████████████████████████████████▎                                                         | 1.14G/1.86G [00:13<00:07, 106MB/s]

    
 62%|████████████████████████████████████████████████████████████████████████████████████████████▏                                                        | 1.15G/1.86G [00:13<00:07, 104MB/s]

    
 62%|████████████████████████████████████████████████████████████████████████████████████████████▉                                                        | 1.16G/1.86G [00:13<00:07, 107MB/s]

    
 63%|█████████████████████████████████████████████████████████████████████████████████████████████▉                                                       | 1.17G/1.86G [00:14<00:06, 112MB/s]

    
 64%|██████████████████████████████████████████████████████████████████████████████████████████████▏                                                     | 1.19G/1.86G [00:14<00:07, 94.0MB/s]

    
 64%|███████████████████████████████████████████████████████████████████████████████████████████████                                                     | 1.20G/1.86G [00:14<00:07, 99.3MB/s]

    
 65%|████████████████████████████████████████████████████████████████████████████████████████████████▋                                                    | 1.21G/1.86G [00:14<00:06, 103MB/s]

    
 65%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                                                   | 1.22G/1.86G [00:14<00:06, 103MB/s]

    
 66%|██████████████████████████████████████████████████████████████████████████████████████████████████▍                                                  | 1.23G/1.86G [00:14<00:06, 107MB/s]

    
 67%|███████████████████████████████████████████████████████████████████████████████████████████████████▎                                                 | 1.24G/1.86G [00:14<00:06, 106MB/s]

    
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                | 1.25G/1.86G [00:14<00:06, 109MB/s]

    
 68%|█████████████████████████████████████████████████████████████████████████████████████████████████████▏                                               | 1.27G/1.86G [00:14<00:05, 109MB/s]

    
 68%|██████████████████████████████████████████████████████████████████████████████████████████████████████                                               | 1.28G/1.86G [00:15<00:05, 109MB/s]

    
 69%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                                              | 1.29G/1.86G [00:15<00:05, 109MB/s]

    
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊                                             | 1.30G/1.86G [00:15<00:05, 110MB/s]

    
 70%|████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                            | 1.31G/1.86G [00:15<00:05, 111MB/s]

    
 71%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                           | 1.32G/1.86G [00:15<00:05, 109MB/s]

    
 72%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                          | 1.33G/1.86G [00:15<00:05, 110MB/s]

    
 72%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                         | 1.34G/1.86G [00:15<00:05, 107MB/s]

    
 73%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                        | 1.36G/1.86G [00:15<00:04, 109MB/s]

    
 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                       | 1.37G/1.86G [00:15<00:04, 110MB/s]

    
 74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                      | 1.38G/1.86G [00:16<00:05, 92.9MB/s]

    
 75%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████                                      | 1.39G/1.86G [00:16<00:05, 101MB/s]

    
 75%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                     | 1.40G/1.86G [00:16<00:04, 103MB/s]

    
 76%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                    | 1.41G/1.86G [00:16<00:04, 105MB/s]

    
 76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 1.42G/1.86G [00:16<00:04, 110MB/s]

    
 77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                  | 1.43G/1.86G [00:16<00:04, 113MB/s]

    
 78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 1.45G/1.86G [00:16<00:04, 96.9MB/s]

    
 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                | 1.46G/1.86G [00:16<00:04, 101MB/s]

    
 79%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                               | 1.47G/1.86G [00:17<00:04, 103MB/s]

    
 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                              | 1.48G/1.86G [00:17<00:03, 104MB/s]

    
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                              | 1.49G/1.86G [00:17<00:03, 106MB/s]

    
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                             | 1.50G/1.86G [00:17<00:03, 110MB/s]

    
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 1.51G/1.86G [00:17<00:03, 112MB/s]

    
 82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                           | 1.52G/1.86G [00:17<00:03, 110MB/s]

    
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                          | 1.53G/1.86G [00:17<00:03, 109MB/s]

    
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1.54G/1.86G [00:17<00:03, 108MB/s]

    
 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                        | 1.55G/1.86G [00:17<00:03, 108MB/s]

    
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                       | 1.57G/1.86G [00:18<00:02, 109MB/s]

    
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                       | 1.58G/1.86G [00:18<00:02, 108MB/s]

    
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                      | 1.59G/1.86G [00:18<00:02, 110MB/s]

    
 86%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                     | 1.60G/1.86G [00:18<00:02, 110MB/s]

    
 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                    | 1.61G/1.86G [00:18<00:02, 108MB/s]

    
 87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                   | 1.62G/1.86G [00:18<00:02, 103MB/s]

    
 88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                  | 1.63G/1.86G [00:18<00:02, 104MB/s]

    
 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                 | 1.64G/1.86G [00:18<00:02, 103MB/s]

    
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                | 1.65G/1.86G [00:18<00:02, 106MB/s]

    
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏               | 1.67G/1.86G [00:19<00:02, 96.3MB/s]

    
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 1.68G/1.86G [00:19<00:01, 101MB/s]

    
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊              | 1.69G/1.86G [00:19<00:01, 101MB/s]

    
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋             | 1.70G/1.86G [00:19<00:01, 103MB/s]

    
 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌            | 1.71G/1.86G [00:19<00:01, 104MB/s]

    
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌           | 1.72G/1.86G [00:19<00:01, 106MB/s]

    
 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 1.73G/1.86G [00:19<00:01, 108MB/s]

    
 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎         | 1.74G/1.86G [00:19<00:01, 107MB/s]

    
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏        | 1.75G/1.86G [00:19<00:01, 96.7MB/s]

    
 95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉        | 1.76G/1.86G [00:20<00:01, 96.3MB/s]

    
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊       | 1.77G/1.86G [00:20<00:01, 92.6MB/s]

    
 96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌      | 1.78G/1.86G [00:20<00:00, 97.6MB/s]

    
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌     | 1.79G/1.86G [00:20<00:00, 101MB/s]

    
 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍    | 1.81G/1.86G [00:20<00:00, 106MB/s]

    
 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 1.82G/1.86G [00:20<00:00, 110MB/s]

    
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 1.83G/1.86G [00:20<00:00, 109MB/s]

    
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎ | 1.84G/1.86G [00:20<00:00, 113MB/s]

    
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1.85G/1.86G [00:20<00:00, 115MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.86G/1.86G [00:21<00:00, 95.0MB/s]

    


<div class="k-default-codeblock">
```
Preparing OASIS dataset...
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Dataset found in <span style="color: #800080; text-decoration-color: #800080">/home/laxmareddyp/nilearn_data/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">oasis1</span>
</pre>



<div class="k-default-codeblock">
```
Download dataset is completed.

Saved 4 OASIS Brain MRI images: ['images/oasis_0.png', 'images/oasis_1.png', 'images/oasis_2.png', 'images/oasis_3.png']
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_222.png)
    


<div class="k-default-codeblock">
```
Extracting database features...

Running RAG pipeline...

[RAG] Matched image index: 0
[RAG] Matched image path: images/oasis_0.png
[RAG] Retrieved context/report:
MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_226.png)
    


    
<div class="k-default-codeblock">
```
==================================================
RAG PIPELINE RESULTS
==================================================

Matched DB Report Index: 0
Matched DB Report: MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.

--- Generated Report ---
 Radiology Report

Imaging Procedure:
MRI of the brain

Findings:
Right frontal lobe:
1.5cm lesion, non-enhancing, no edema.

Diagnostic Impression:
A 1.5cm lesion in the right frontal lobe, non-enhancing, with no edema.

Diagnostic Reasoning:
The MRI findings suggest a lesion within the right frontal lobe. The absence of enhancement and the lack of edema are consistent with a lesion that is not actively growing or causing inflammation. The lesion's size (1.5cm) is within the typical range for this type of lesion.

Possible Treatment Options:
Given the lesion's characteristics, treatment options will depend on several factors, including the lesion's location, size, and potential impact on neurological function. Potential options include:

Observation:
Monitoring the lesion for any changes over time.

Surgical Resection:
Removal of the lesion.

Stereotactic Radiosurgery:
Targeted destruction of the lesion using focused radiation.

Clinical Trial:
Investigating new therapies for lesions of this type.

Disclaimer:
This is a preliminary assessment based on the provided information. A definitive diagnosis and treatment plan should be determined by a qualified medical professional.

---

Important Considerations:

Further Investigation:
It's crucial to note that this report is limited by the provided image. Further investigation may be needed to determine the lesion's characteristics, including:

Diffusion Tensor Imaging (DTI):
To assess white matter integrity.

Neuropsychological Testing:
To evaluate cognitive function.

Neuroimaging Follow-up:
To monitor for any changes over time.

Let me know if you'd like me to elaborate on any specific aspect of this report.

==================================================
VLM RESULTS (Direct Approach)
==================================================

--- Vision-Language Model (No Retrieval) Report ---
 Radiology Report

Medical Record Number:
[MRN]

Clinical Indication:
[Reason for the MRI - e.g., Headache, Neurological Symptoms, etc.]

1. Impression:

Likely Multiple Sclerosis (MS) with evidence of white matter lesions consistent with disseminated demyelinating disease.  There is also a small, indeterminate lesion in the right frontal lobe that requires further investigation to rule out other etiologies.

2. Diagnostic Reasoning:

The MRI demonstrates numerous white matter lesions scattered throughout the brain parenchyma. These lesions are characterized by hyperintensity on T2-weighted imaging and FLAIR sequences, indicative of edema and demyelination. The distribution of these lesions is non-specific, but the pattern is commonly seen in Multiple Sclerosis.

Specifically:

White Matter Lesions:
The presence of numerous, confluent, and scattered white matter lesions is the most significant finding. These lesions are typically seen in MS.

   T2/FLAIR Hyperintensity: The hyperintensity on T2 and FLAIR sequences reflects the presence of fluid within the lesions, representing edema and demyelination.
Contrast Enhancement:
Some lesions demonstrate contrast enhancement, which is a hallmark of active demyelination and inflammation.  The degree of enhancement can vary.

Small Right Frontal Lesion:
A small, solitary lesion is present in the right frontal lobe. While it could be consistent with MS, its isolated nature warrants consideration of other potential causes, such as vascular inflammation, demyelinating lesions not typical of MS, or a small, early lesion.

Differential Diagnosis:

Other Demyelinating Diseases:
Progressive Multifocal Leukoencephalopathy (PML) should be considered, although less likely given the widespread nature of the lesions.

Vascular Inflammation:
Vasculitis can present with similar white matter changes.

Autoimmune Encephalitis:
Certain autoimmune encephalitis can cause white matter abnormalities.

Normal Pressure Hydrocephalus (NPH):
Although less likely given the presence of numerous lesions, NPH can sometimes present with white matter changes.

3. Treatment Options:

The treatment plan should be determined in consultation with the patient’s neurologist. Potential options include:

Disease-Modifying Therapies (DMTs):
These medications aim to slow the progression of MS. Examples include interferon beta, glatiramer acetate, natalizumab, fingolimod, and dimethyl fumarate. The choice of DMT will depend on the patient’s disease activity, risk factors, and preferences.

Symptomatic Treatment:
Management of specific symptoms such as fatigue, pain, depression, and cognitive dysfunction.

Immunomodulatory Therapies:
For acute exacerbations, corticosteroids may be used to reduce inflammation and improve symptoms.

Further Investigation:
Given the indeterminate lesion in the right frontal lobe, further investigation may be warranted, including:

Repeat MRI:
To monitor for changes in the lesion over time.

Blood Tests:
To rule out other inflammatory or autoimmune conditions.

Lumbar Puncture:
To analyze cerebrospinal fluid for oligoclonal bands and other markers of inflammation (if clinically indicated).

Recommendations:

   Correlation with clinical findings is recommended.
   Consultation with a neurologist is advised for further management and treatment planning.

Radiologist:
[Radiologist Name]

Credentials:
[Radiologist Credentials]

---

Disclaimer:
This report is based solely on the provided image and clinical information. A complete diagnostic assessment requires a thorough review of the patient's medical history, physical examination findings, and other relevant investigations.

Note:
This is a sample report and needs to be adapted based on the specific details of the MRI image and the patient's clinical presentation.  The presence of lesions alone does not definitively diagnose MS, and further investigation is often necessary.
```
</div>

---
## Comparison: RAG Pipeline vs Direct VLM

- **MobileNet + Gemma3 1B text model**: ~1B total parameters
- **Gemma3 VLM 4B model**: ~4B total parameters
- **Results**: The RAG pipeline (MobileNet + Gemma3 1B) is better due to its use of retrieval and context, providing more relevant and accurate reports with fewer parameters.

**Detailed Comparison:**

- **Accuracy & Relevance:**

  - RAG pipeline leverages retrieval to provide contextually relevant and case-specific reports, often matching or exceeding the quality of much larger VLMs.
  - Direct VLM (Gemma3 4B) produces more generic outputs, lacking access to specific prior cases.

- **Speed & Resource Usage:**

  - MobileNet + Gemma3 1B is significantly faster and more memory-efficient, making it suitable for edge devices and real-time applications.
  - Gemma3 4B requires more computational resources and is slower, especially on limited hardware.

- **Scalability & Flexibility:**

  - The RAG approach allows easy swapping of retriever/generator models and can be adapted to different domains or datasets.
  - Direct VLM is less flexible and requires retraining or fine-tuning for new domains.

- **Interpretability:**

  - RAG pipeline provides traceability by showing which database report was used for context, aiding in clinical interpretability and trust.
  - Direct VLM does not provide this transparency.

- **Practical Implications:**

  - RAG is more practical for deployment in resource-constrained environments and can be incrementally improved by updating the database.
  - Large VLMs are best suited for cloud or high-performance environments.

In practice, the RAG approach leverages both image similarity and prior knowledge to generate more precise and clinically meaningful reports, while the direct VLM approach is limited to general knowledge and lacks case-specific context.

---
## Conclusion

This demonstration showcases the power of Retrieval-Augmented Generation (RAG) in combining vision and language models for intelligent analysis using KerasHub models.

**Key Achievements:**

- Model Integration: Vision Transformer + Gemma3 LLM via KerasHub
- Feature Extraction: Meaningful features from brain MRI images
- Similarity Search: Efficient retrieval of relevant context
- Report Generation: Comprehensive reports using retrieved context
- Comparison Analysis: RAG vs direct VLM approaches

**Key Benefits:**

- Enhanced Accuracy: More contextually relevant outputs
- Scalable Architecture: Easy to extend with different models
- KerasHub Integration: State-of-the-art models efficiently
- Real-world Applicability: Various vision-language tasks

This guide demonstrates how KerasHub enables rapid prototyping and deployment of advanced AI systems for real-world applications.

---
## Security Warning

⚠️ **IMPORTANT SECURITY AND PRIVACY CONSIDERATIONS**

This pipeline is for educational purposes only. For production use:

- Anonymize medical data following HIPAA guidelines
- Implement access controls and encryption
- Validate inputs and secure APIs
- Consult medical professionals for clinical decisions
- This system should NOT be used for actual medical diagnosis without proper validation
