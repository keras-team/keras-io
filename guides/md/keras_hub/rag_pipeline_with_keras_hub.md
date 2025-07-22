# RAG Pipeline with KerasHub

**Author:** [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)<br>
**Date created:** 2025/07/22<br>
**Last modified:** 2025/07/22<br>
**Description:** RAG pipeline for MRI: image retrieval, context search, and report generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/rag_pipeline_with_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/rag_pipeline_with_keras_hub.py)




```python
# =============================================================================
# INTRODUCTION
# =============================================================================
```

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
4. Generate comprehensive radiology reports using retrieved context
5. Compare RAG approach with direct vision-language model generation

This pipeline demonstrates how to build a sophisticated medical AI system that can:
- Analyze brain MRI images using state-of-the-art vision models
- Retrieve relevant medical context from a database
- Generate detailed radiology reports with proper medical terminology
- Provide diagnostic impressions and treatment recommendations

Let's get started!


```python
# =============================================================================
# SETUP
# =============================================================================
```

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

os.environ["KERAS_BACKEND"] = "jax"
import keras
import json
import numpy as np

keras.config.set_dtype_policy("bfloat16")
import keras_hub
import kagglehub
import requests
from PIL import Image
import matplotlib.pyplot as plt

# from medmnist import PathMNIST
from nilearn import datasets, image

# =============================================================================
# MODEL LOADING AND CONFIGURATION
# =============================================================================
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1753170679.520139   30164 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1753170679.524906   30164 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1753170679.536639   30164 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753170679.536652   30164 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753170679.536653   30164 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753170679.536655   30164 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

Model Loading

Loads the vision model (for image feature extraction) and the Gemma3 vision-language model (for report generation). Returns both models for use in the RAG pipeline.


```python

def load_models():
    """
    Load and configure vision model for feature extraction and Gemma3 VLM for report generation.
    Returns:
        tuple: (vision_model, vlm_model)
    """
    # Vision model for feature extraction
    backbone = keras_hub.models.Backbone.from_preset("vit_large_patch32_384_imagenet")
    preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
        "vit_large_patch32_384_imagenet"
    )
    vision_model = keras_hub.models.ViTImageClassifier(
        backbone=backbone, num_classes=2, preprocessor=preprocessor
    )
    # Gemma3 VLM for report generation
    vlm_model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b")
    return vision_model, vlm_model


# =============================================================================
# DATA PREPARATION AND PROCESSING
# =============================================================================
```

Image and Caption Preparation

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


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================
```

Image Visualization Utility

Displays a set of processed brain MRI images with their corresponding captions.


```python

def visualize_images(image_paths, captions):
    """
    Visualize the processed brain MRI images.

    Args:
        image_paths (list): List of image file paths
        captions (list): List of corresponding image captions
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (img_path, title) in enumerate(zip(image_paths, captions)):
        img = Image.open(img_path)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.suptitle("OASIS Brain MRI Images")
    plt.tight_layout()
    plt.show()

```

Prediction Visualization Utility

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


# =============================================================================
# FEATURE EXTRACTION AND PROCESSING
# =============================================================================
```

Image Feature Extraction

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


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Example radiology reports for each DB image (replace with real reports if available)
db_reports = [
    "MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.",
    "Normal MRI scan, no abnormal findings.",
    "Diffuse atrophy noted, no focal lesions.",
]

# =============================================================================
# TEXT PROCESSING AND CLEANING
# =============================================================================
```

Output Cleaning Utility

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
        # If prompt is not at the beginning, try to find and remove it
        cleaned_text = generated_text.replace(prompt, "").strip()

    # Remove header details
    lines = cleaned_text.split("\n")
    filtered_lines = []
    skip_next = False

    for line in lines:
        # Skip lines with these headers
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
        # Skip empty lines after headers
        elif line.strip() == "" and skip_next:
            skip_next = False
            continue
        else:
            filtered_lines.append(line)
            skip_next = False

    # Join the lines back together
    cleaned_text = "\n".join(filtered_lines).strip()

    return cleaned_text


# =============================================================================
# RAG PIPELINE IMPLEMENTATION
# =============================================================================
```

RAG Pipeline

Implements the Retrieval-Augmented Generation (RAG) pipeline:
- Extracts features from the query image and database images.
- Finds the most similar image in the database.
- Uses the retrieved report and the query image as input to the Gemma3 VLM to generate a new report.
Returns the index of the matched image, the retrieved report, and the generated report.


```python

def rag_pipeline(query_img_path, db_image_paths, db_reports, vision_model, vlm_model):
    """
    Retrieval-Augmented Generation pipeline using vision model for retrieval and Gemma3 VLM for report generation.
    Args:
        query_img_path (str): Path to the query image
        db_image_paths (list): List of database image paths
        db_reports (list): List of database reports
        vision_model: Vision model for feature extraction
        vlm_model: Gemma3 VLM for report generation
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
    # Print matched image and context for debugging/inspection
    print(f"[RAG] Matched image index: {best_idx}")
    print(f"[RAG] Matched image path: {db_image_paths[best_idx]}")
    print(f"[RAG] Retrieved context/report:\n{retrieved_report}\n")
    # Prepare the prompt
    PROMPT_TEMPLATE = (
        "Context:\n{context}\n\n"
        "Based on the above radiology report and the provided brain MRI image, please:\n"
        "1. Provide a diagnostic impression.\n"
        "2. Explain the diagnostic reasoning.\n"
        "3. Suggest possible treatment options.\n"
        "Format your answer as a structured radiology report.\n"
    )
    prompt = PROMPT_TEMPLATE.format(context=retrieved_report)
    # Preprocess the query image for the VLM
    img = Image.open(query_img_path).convert("RGB").resize((224, 224))
    image = np.array(img) / 255.0
    image = np.expand_dims(image, axis=0)
    # Generate report using the VLM
    output = vlm_model.generate(
        {
            "images": image,
            "prompts": prompt,
        }
    )
    # Clean the generated output
    cleaned_output = clean_generated_output(output, prompt)
    return best_idx, retrieved_report, cleaned_output


# =============================================================================
# VISION-LANGUAGE MODEL (DIRECT APPROACH)
# =============================================================================
```

Vision-Language Model (Direct Approach)

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
        str: Generated radiology report
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
    # Clean the generated output
    cleaned_output = clean_generated_output(
        output, PROMPT_TEMPLATE.format(question=question)
    )
    return cleaned_output


# =============================================================================
# MAIN EXECUTION
# =============================================================================
```

Main Execution

Runs the RAG pipeline: loads models, prepares data, and displays results.


```python
if __name__ == "__main__":
    """
    Main execution pipeline for RAG-based medical image analysis.

    This script demonstrates:
    1. Loading pre-trained vision and language models
    2. Processing OASIS brain MRI dataset
    3. Implementing RAG pipeline with retrieval and generation
    4. Comparing RAG approach with direct VLM approach
    """

    # Load models
    print("Loading models...")
    vision_model, vlm_model = load_models()

    # Prepare data
    print("Preparing OASIS dataset...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=4)
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
        query_img_path, db_image_paths, db_reports, vision_model, vlm_model
    )

    # Visualize results
    visualize_prediction(query_img_path, db_image_paths, best_idx, db_reports)

    # Print RAG results
    print("\n" + "=" * 50)
    print("RAG PIPELINE RESULTS")
    print("=" * 50)
    print("\n--- Retrieved Report ---\n", retrieved_report)
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

Downloading from https://www.kaggle.com/api/v1/models/keras/vit/keras/vit_large_patch32_384_imagenet/2/download/config.json...
```
</div>

  0%|                                                                                                                                                               | 0.00/699 [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 699/699 [00:00<00:00, 2.17MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/vit/keras/vit_large_patch32_384_imagenet/2/download/model.weights.h5...
```
</div>

  0%|                                                                                                                                                             | 0.00/1.14G [00:00<?, ?B/s]

    
  0%|▏                                                                                                                                                   | 1.00M/1.14G [00:00<03:40, 5.54MB/s]

    
  1%|█▏                                                                                                                                                  | 9.00M/1.14G [00:00<00:31, 38.0MB/s]

    
  2%|██▍                                                                                                                                                 | 19.0M/1.14G [00:00<00:20, 60.1MB/s]

    
  2%|███▎                                                                                                                                                | 26.0M/1.14G [00:00<00:27, 42.8MB/s]

    
  3%|████                                                                                                                                                | 32.0M/1.14G [00:00<00:29, 40.3MB/s]

    
  3%|████▋                                                                                                                                               | 37.0M/1.14G [00:00<00:31, 37.8MB/s]

    
  4%|█████▎                                                                                                                                              | 42.0M/1.14G [00:01<00:34, 34.7MB/s]

    
  4%|██████                                                                                                                                              | 48.0M/1.14G [00:01<00:34, 34.0MB/s]

    
  5%|██████▊                                                                                                                                             | 54.0M/1.14G [00:01<00:32, 35.5MB/s]

    
  5%|███████▎                                                                                                                                            | 58.0M/1.14G [00:01<00:34, 33.8MB/s]

    
  6%|████████▋                                                                                                                                           | 68.0M/1.14G [00:01<00:23, 48.3MB/s]

    
  6%|█████████▌                                                                                                                                          | 75.0M/1.14G [00:01<00:22, 51.4MB/s]

    
  7%|██████████▎                                                                                                                                         | 81.0M/1.14G [00:02<00:23, 49.2MB/s]

    
  7%|███████████                                                                                                                                         | 87.0M/1.14G [00:02<00:22, 51.1MB/s]

    
  8%|███████████▊                                                                                                                                        | 93.0M/1.14G [00:02<00:24, 45.6MB/s]

    
  9%|█████████████▎                                                                                                                                       | 104M/1.14G [00:02<00:18, 61.1MB/s]

    
 10%|██████████████▏                                                                                                                                      | 111M/1.14G [00:02<00:24, 44.9MB/s]

    
 11%|███████████████▋                                                                                                                                     | 123M/1.14G [00:02<00:18, 60.6MB/s]

    
 11%|████████████████▋                                                                                                                                    | 131M/1.14G [00:02<00:19, 57.0MB/s]

    
 12%|█████████████████▊                                                                                                                                   | 139M/1.14G [00:03<00:23, 46.3MB/s]

    
 13%|███████████████████                                                                                                                                  | 149M/1.14G [00:03<00:19, 55.3MB/s]

    
 13%|███████████████████▉                                                                                                                                 | 156M/1.14G [00:03<00:20, 51.8MB/s]

    
 14%|█████████████████████▏                                                                                                                               | 166M/1.14G [00:03<00:17, 59.7MB/s]

    
 15%|██████████████████████▌                                                                                                                              | 177M/1.14G [00:03<00:14, 71.0MB/s]

    
 16%|████████████████████████                                                                                                                             | 188M/1.14G [00:03<00:12, 80.6MB/s]

    
 17%|█████████████████████████▌                                                                                                                           | 200M/1.14G [00:03<00:11, 90.6MB/s]

    
 18%|███████████████████████████                                                                                                                          | 212M/1.14G [00:04<00:10, 98.6MB/s]

    
 19%|████████████████████████████▋                                                                                                                         | 223M/1.14G [00:04<00:09, 101MB/s]

    
 20%|██████████████████████████████                                                                                                                       | 235M/1.14G [00:04<00:09, 98.4MB/s]

    
 21%|███████████████████████████████▎                                                                                                                     | 245M/1.14G [00:04<00:09, 97.6MB/s]

    
 22%|████████████████████████████████▌                                                                                                                    | 255M/1.14G [00:04<00:09, 96.4MB/s]

    
 23%|██████████████████████████████████▏                                                                                                                   | 266M/1.14G [00:04<00:09, 101MB/s]

    
 24%|███████████████████████████████████▋                                                                                                                  | 278M/1.14G [00:04<00:08, 105MB/s]

    
 25%|█████████████████████████████████████▎                                                                                                                | 290M/1.14G [00:04<00:08, 109MB/s]

    
 26%|██████████████████████████████████████▋                                                                                                               | 301M/1.14G [00:04<00:08, 111MB/s]

    
 27%|████████████████████████████████████████                                                                                                              | 312M/1.14G [00:05<00:08, 112MB/s]

    
 28%|█████████████████████████████████████████▎                                                                                                           | 323M/1.14G [00:05<00:10, 83.5MB/s]

    
 28%|██████████████████████████████████████████▍                                                                                                          | 332M/1.14G [00:05<00:10, 79.7MB/s]

    
 29%|███████████████████████████████████████████▌                                                                                                         | 341M/1.14G [00:05<00:13, 64.5MB/s]

    
 30%|█████████████████████████████████████████████                                                                                                        | 353M/1.14G [00:05<00:11, 77.0MB/s]

    
 31%|██████████████████████████████████████████████▏                                                                                                      | 362M/1.14G [00:05<00:12, 67.6MB/s]

    
 32%|███████████████████████████████████████████████▋                                                                                                     | 373M/1.14G [00:06<00:11, 75.2MB/s]

    
 33%|████████████████████████████████████████████████▋                                                                                                    | 381M/1.14G [00:06<00:11, 71.2MB/s]

    
 33%|█████████████████████████████████████████████████▋                                                                                                   | 389M/1.14G [00:06<00:12, 66.3MB/s]

    
 34%|██████████████████████████████████████████████████▊                                                                                                  | 398M/1.14G [00:06<00:12, 64.4MB/s]

    
 35%|███████████████████████████████████████████████████▋                                                                                                 | 405M/1.14G [00:06<00:14, 54.9MB/s]

    
 35%|████████████████████████████████████████████████████▋                                                                                                | 413M/1.14G [00:06<00:13, 60.8MB/s]

    
 36%|█████████████████████████████████████████████████████▋                                                                                               | 420M/1.14G [00:06<00:15, 51.6MB/s]

    
 37%|███████████████████████████████████████████████████████▏                                                                                             | 432M/1.14G [00:07<00:11, 64.8MB/s]

    
 38%|████████████████████████████████████████████████████████▍                                                                                            | 442M/1.14G [00:07<00:10, 72.4MB/s]

    
 39%|█████████████████████████████████████████████████████████▍                                                                                           | 450M/1.14G [00:07<00:10, 72.1MB/s]

    
 40%|██████████████████████████████████████████████████████████▉                                                                                          | 461M/1.14G [00:07<00:08, 82.5MB/s]

    
 40%|████████████████████████████████████████████████████████████                                                                                         | 470M/1.14G [00:07<00:08, 85.2MB/s]

    
 41%|█████████████████████████████████████████████████████████████▏                                                                                       | 479M/1.14G [00:07<00:09, 75.1MB/s]

    
 42%|██████████████████████████████████████████████████████████████▏                                                                                      | 487M/1.14G [00:07<00:12, 58.4MB/s]

    
 43%|███████████████████████████████████████████████████████████████▋                                                                                     | 499M/1.14G [00:08<00:09, 72.4MB/s]

    
 43%|████████████████████████████████████████████████████████████████▊                                                                                    | 507M/1.14G [00:08<00:10, 69.1MB/s]

    
 44%|██████████████████████████████████████████████████████████████████▎                                                                                  | 519M/1.14G [00:08<00:08, 78.5MB/s]

    
 46%|███████████████████████████████████████████████████████████████████▊                                                                                 | 531M/1.14G [00:08<00:07, 88.9MB/s]

    
 46%|█████████████████████████████████████████████████████████████████████▏                                                                               | 542M/1.14G [00:08<00:07, 93.3MB/s]

    
 47%|███████████████████████████████████████████████████████████████████████▏                                                                              | 554M/1.14G [00:08<00:06, 101MB/s]

    
 48%|████████████████████████████████████████████████████████████████████████▏                                                                            | 565M/1.14G [00:08<00:06, 96.5MB/s]

    
 49%|██████████████████████████████████████████████████████████████████████████                                                                            | 576M/1.14G [00:08<00:06, 101MB/s]

    
 50%|███████████████████████████████████████████████████████████████████████████▌                                                                          | 588M/1.14G [00:08<00:05, 107MB/s]

    
 51%|█████████████████████████████████████████████████████████████████████████████▏                                                                        | 600M/1.14G [00:09<00:05, 107MB/s]

    
 52%|██████████████████████████████████████████████████████████████████████████████                                                                       | 611M/1.14G [00:09<00:06, 86.0MB/s]

    
 53%|███████████████████████████████████████████████████████████████████████████████▍                                                                     | 622M/1.14G [00:09<00:06, 92.8MB/s]

    
 54%|████████████████████████████████████████████████████████████████████████████████▋                                                                    | 632M/1.14G [00:09<00:05, 94.6MB/s]

    
 55%|██████████████████████████████████████████████████████████████████████████████████                                                                   | 643M/1.14G [00:09<00:05, 97.2MB/s]

    
 56%|████████████████████████████████████████████████████████████████████████████████████                                                                  | 654M/1.14G [00:09<00:05, 100MB/s]

    
 57%|████████████████████████████████████████████████████████████████████████████████████▊                                                                | 664M/1.14G [00:09<00:05, 99.7MB/s]

    
 58%|██████████████████████████████████████████████████████████████████████████████████████                                                               | 674M/1.14G [00:09<00:05, 89.8MB/s]

    
 59%|███████████████████████████████████████████████████████████████████████████████████████▎                                                             | 684M/1.14G [00:10<00:05, 89.1MB/s]

    
 59%|████████████████████████████████████████████████████████████████████████████████████████▌                                                            | 693M/1.14G [00:10<00:05, 89.2MB/s]

    
 60%|█████████████████████████████████████████████████████████████████████████████████████████▋                                                           | 702M/1.14G [00:10<00:05, 84.4MB/s]

    
 61%|██████████████████████████████████████████████████████████████████████████████████████████▉                                                          | 712M/1.14G [00:10<00:05, 87.8MB/s]

    
 62%|████████████████████████████████████████████████████████████████████████████████████████████▎                                                        | 723M/1.14G [00:10<00:04, 95.1MB/s]

    
 63%|█████████████████████████████████████████████████████████████████████████████████████████████▋                                                       | 734M/1.14G [00:10<00:04, 99.9MB/s]

    
 64%|███████████████████████████████████████████████████████████████████████████████████████████████▋                                                      | 744M/1.14G [00:10<00:04, 101MB/s]

    
 65%|████████████████████████████████████████████████████████████████████████████████████████████████▎                                                    | 754M/1.14G [00:10<00:04, 97.5MB/s]

    
 65%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                                                   | 764M/1.14G [00:10<00:04, 96.3MB/s]

    
 66%|███████████████████████████████████████████████████████████████████████████████████████████████████▋                                                  | 775M/1.14G [00:11<00:04, 101MB/s]

    
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                | 785M/1.14G [00:11<00:04, 99.4MB/s]

    
 68%|█████████████████████████████████████████████████████████████████████████████████████████████████████▌                                               | 795M/1.14G [00:11<00:04, 96.8MB/s]

    
 69%|███████████████████████████████████████████████████████████████████████████████████████████████████████▋                                              | 806M/1.14G [00:11<00:03, 100MB/s]

    
 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████                                             | 817M/1.14G [00:11<00:03, 101MB/s]

    
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                           | 827M/1.14G [00:11<00:03, 100MB/s]

    
 72%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                          | 837M/1.14G [00:11<00:03, 96.6MB/s]

    
 73%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                        | 847M/1.14G [00:11<00:03, 87.5MB/s]

    
 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                       | 856M/1.14G [00:11<00:03, 84.7MB/s]

    
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                      | 866M/1.14G [00:12<00:03, 86.6MB/s]

    
 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                     | 877M/1.14G [00:12<00:03, 93.1MB/s]

    
 76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                   | 888M/1.14G [00:12<00:03, 94.4MB/s]

    
 77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                  | 899M/1.14G [00:12<00:02, 98.2MB/s]

    
 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                 | 909M/1.14G [00:12<00:02, 98.7MB/s]

    
 79%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                               | 919M/1.14G [00:12<00:02, 98.1MB/s]

    
 80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                              | 929M/1.14G [00:12<00:03, 78.6MB/s]

    
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                             | 938M/1.14G [00:12<00:03, 79.3MB/s]

    
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                            | 947M/1.14G [00:13<00:02, 81.0MB/s]

    
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 957M/1.14G [00:13<00:02, 86.9MB/s]

    
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                         | 967M/1.14G [00:13<00:02, 72.9MB/s]

    
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                        | 976M/1.14G [00:13<00:02, 77.0MB/s]

    
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 985M/1.14G [00:13<00:02, 80.8MB/s]

    
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                      | 995M/1.14G [00:13<00:02, 85.8MB/s]

    
 86%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                    | 0.98G/1.14G [00:13<00:02, 65.5MB/s]

    
 87%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                   | 0.99G/1.14G [00:13<00:02, 72.4MB/s]

    
 88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 1.00G/1.14G [00:14<00:01, 81.2MB/s]

    
 89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                | 1.01G/1.14G [00:14<00:01, 88.1MB/s]

    
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊               | 1.02G/1.14G [00:14<00:01, 93.6MB/s]

    
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████              | 1.03G/1.14G [00:14<00:01, 69.1MB/s]

    
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏            | 1.04G/1.14G [00:14<00:01, 73.7MB/s]

    
 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍           | 1.05G/1.14G [00:14<00:01, 60.9MB/s]

    
 93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋          | 1.06G/1.14G [00:14<00:01, 69.2MB/s]

    
 94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊         | 1.07G/1.14G [00:15<00:01, 73.6MB/s]

    
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊        | 1.08G/1.14G [00:15<00:00, 75.4MB/s]

    
 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏      | 1.09G/1.14G [00:15<00:00, 82.3MB/s]

    
 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍     | 1.10G/1.14G [00:15<00:00, 86.8MB/s]

    
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌    | 1.11G/1.14G [00:15<00:00, 87.9MB/s]

    
 98%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋   | 1.11G/1.14G [00:15<00:00, 85.3MB/s]

    
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉  | 1.12G/1.14G [00:15<00:00, 80.0MB/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 1.13G/1.14G [00:15<00:00, 87.1MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.14G/1.14G [00:16<00:00, 76.1MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/vit/keras/vit_large_patch32_384_imagenet/2/download/preprocessor.json...
```
</div>

  0%|                                                                                                                                                             | 0.00/1.74k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.74k/1.74k [00:00<00:00, 5.42MB/s]

    


<div class="k-default-codeblock">
```
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

Preparing OASIS dataset...
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Added README.md to <span style="color: #800080; text-decoration-color: #800080">/home/laxmareddyp/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">nilearn_data</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Dataset created in <span style="color: #800080; text-decoration-color: #800080">/home/laxmareddyp/nilearn_data/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">oasis1</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloading data from <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://www.nitrc.org/frs/download.php/6364/archive_dartel.tgz</span> <span style="color: #808000; text-decoration-color: #808000">...</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">88514560</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.8</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.</span>2s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">194355200</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">21.5</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.</span>3s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">298713088</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">33.0</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.</span>1s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">401899520</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">44.4</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.</span>0s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">508862464</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">56.2</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.</span>9s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">616652800</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">68.1</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.</span>8s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">717389824</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">79.3</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.</span>8s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Downloaded <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">820551680</span> of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">905208634</span> bytes <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90.6</span>%%,    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.</span>8s remaining<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span>  <span style="color: #808000; text-decoration-color: #808000">...</span>done. <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span> seconds, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> min<span style="font-weight: bold">)</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Extracting data from 
<span style="color: #800080; text-decoration-color: #800080">/home/laxmareddyp/nilearn_data/oasis1/60f82260d880dec0385d5f2ba9df2b83/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">archive_dartel.tgz...</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> .. done.

</pre>



<div class="k-default-codeblock">
```
Saved 4 OASIS Brain MRI images: ['images/oasis_0.png', 'images/oasis_1.png', 'images/oasis_2.png', 'images/oasis_3.png']
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_151.png)
    


<div class="k-default-codeblock">
```
Extracting database features...

Running RAG pipeline...

[RAG] Matched image index: 1
[RAG] Matched image path: images/oasis_1.png
[RAG] Retrieved context/report:
Normal MRI scan, no abnormal findings.
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_155.png)
    


    
<div class="k-default-codeblock">
```
==================================================
RAG PIPELINE RESULTS
==================================================

--- Retrieved Report ---
 Normal MRI scan, no abnormal findings.

--- Generated Report ---
 ---

**Radiology Report**

**Clinical Indication:** [Clinical Indication - Placeholder - e.g., Headache, Neurological Symptoms]

**1. Diagnostic Impression:**

Normal Brain MRI. No evidence of acute or chronic abnormalities detected.

**2. Diagnostic Reasoning:**

The MRI scan demonstrates normal signal intensity and morphology of the visualized brain structures, including the cerebrum, cerebellum, brainstem, and white matter. There is no evidence of mass effect, edema, hemorrhage, or significant vascular abnormalities. Contrast enhancement is unremarkable. The ventricles are of normal size and shape. The orbits and sella turcica appear unremarkable. No significant findings are noted in the visualized portions of the brain.

**3. Suggested Treatment Options:**

Given the normal findings, no specific treatment is indicated at this time.  The patient should be monitored for any new or worsening symptoms.  If the clinical indication for the MRI remains present (e.g., headache), further investigation may be warranted based on the patient's clinical presentation and response to initial management.  This may include further neurological examination, neurocognitive testing, or consideration of alternative diagnostic modalities depending on the specific clinical concern.

**Radiologist:** [Radiologist Name - Placeholder]
**Date:** [Date - Placeholder]

---

**Important Disclaimer:** *This report is based solely on the provided information (normal MRI scan and a general description). A complete and accurate diagnosis requires a thorough clinical evaluation by a qualified physician, considering the patient's history, physical examination, and other relevant investigations. This report is for informational purposes only and should not be considered a substitute for professional medical advice.*
<end_of_turn>

==================================================
VLM RESULTS (Direct Approach)
==================================================

--- Vision-Language Model (No Retrieval) Report ---
 **Radiology Report**

**Medical Record Number:** [MRN]
**Clinical Indication:** [Reason for the MRI - e.g., Headache, Neurological Symptoms, etc.]

**1. Impression:**

Likely Multiple Sclerosis (MS) with evidence of white matter lesions consistent with disseminated demyelinating disease.  There is also a small, indeterminate lesion in the right frontal lobe that requires further investigation to rule out other etiologies.

**2. Diagnostic Reasoning:**

The MRI demonstrates numerous white matter lesions scattered throughout the brain parenchyma. These lesions are characterized by hyperintensity on T2-weighted imaging and FLAIR sequences, indicative of edema and demyelination. The distribution of these lesions is non-specific, but the pattern is commonly seen in Multiple Sclerosis.

Specifically:

*   **White Matter Lesions:** The presence of numerous, confluent, and scattered white matter lesions is the most significant finding. These lesions are typically seen in MS.
*   **T2/FLAIR Hyperintensity:** The hyperintensity on T2 and FLAIR sequences reflects the presence of fluid within the lesions, representing edema and demyelination.
*   **Contrast Enhancement:**  Some lesions demonstrate contrast enhancement, which is a hallmark of active demyelination and inflammation.  The degree of enhancement can vary.
*   **Small Right Frontal Lesion:** A small, solitary lesion is present in the right frontal lobe. While it could be consistent with MS, its isolated nature warrants consideration of other potential causes, such as vascular inflammation, demyelinating lesions not typical of MS, or a small, early lesion.

**Differential Diagnosis:**

*   **Other Demyelinating Diseases:**  Progressive Multifocal Leukoencephalopathy (PML) should be considered, although less likely given the widespread nature of the lesions.
*   **Vascular Inflammation:**  Vasculitis can present with similar white matter changes.
*   **Autoimmune Encephalitis:**  Certain autoimmune encephalitis can cause white matter abnormalities.
*   **Normal Pressure Hydrocephalus (NPH):**  Although less likely given the presence of numerous lesions, NPH can sometimes present with white matter changes.

**3. Treatment Options:**

The treatment plan should be determined in consultation with the patient’s neurologist. Potential options include:

*   **Disease-Modifying Therapies (DMTs):**  These medications aim to slow the progression of MS. Examples include interferon beta, glatiramer acetate, natalizumab, fingolimod, and dimethyl fumarate. The choice of DMT will depend on the patient’s disease activity, risk factors, and preferences.
*   **Symptomatic Treatment:**  Management of specific symptoms such as fatigue, pain, depression, and cognitive dysfunction.
*   **Immunomodulatory Therapies:**  For acute exacerbations, corticosteroids may be used to reduce inflammation and improve symptoms.
*   **Further Investigation:**  Given the indeterminate lesion in the right frontal lobe, further investigation may be warranted, including:
    *   **Repeat MRI:**  To monitor for changes in the lesion over time.
    *   **Blood Tests:**  To rule out other inflammatory or autoimmune conditions.
    *   **Lumbar Puncture:**  To analyze cerebrospinal fluid for oligoclonal bands and other markers of inflammation (if clinically indicated).

**Recommendations:**

*   Correlation with clinical findings is recommended.
*   Consultation with a neurologist is advised for further management and treatment planning.

**Radiologist:** [Radiologist Name]
**Credentials:** [Radiologist Credentials]

---

**Disclaimer:** *This report is based solely on the provided image and clinical information. A complete diagnostic assessment requires a thorough review of the patient's medical history, physical examination findings, and other relevant investigations.*

**Note:** This is a sample report and needs to be adapted based on the specific details of the MRI image and the patient's clinical presentation.  The presence of lesions alone does not definitively diagnose MS, and further investigation is often necessary.<end_of_turn>
```
</div>

Comparison: RAG Pipeline vs. Direct VLM

- RAG pipeline (ViT + Gemma3 model): Produces more accurate and contextually relevant outputs by retrieving the most similar case and using its report as context for generation.
- Gemma3 VLM (direct, no retrieval): Produces more generic and often less accurate outputs, as the model does not have access to relevant context from similar cases.

In practice, the RAG approach leverages both image similarity and prior knowledge to generate more precise and clinically meaningful reports, while the direct VLM approach is limited to general knowledge and lacks case-specific context.


```python
# =============================================================================
# CONCLUSION
# =============================================================================
```

---
## Conclusion

This demonstration showcases the power of Retrieval-Augmented Generation (RAG) in
combining vision and language models for intelligent analysis using kerashub models.

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

This guide demonstrates how KerasHub enables rapid prototyping and deployment
of advanced AI systems for real-world applications.


```python
# =============================================================================
# SECURITY WARNING
# =============================================================================
```

---
## Security Warning

⚠️ **IMPORTANT SECURITY AND PRIVACY CONSIDERATIONS**

This pipeline is for educational purposes only. For production use:
- Anonymize medical data following HIPAA guidelines
- Implement access controls and encryption
- Validate inputs and secure APIs
- Consult medical professionals for clinical decisions
- This system should NOT be used for actual medical diagnosis without proper validation
