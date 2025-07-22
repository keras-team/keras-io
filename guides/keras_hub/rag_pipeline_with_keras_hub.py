"""
Title: RAG Pipeline with KerasHub
Author: [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/07/22
Last modified: 2025/07/22
Description: RAG pipeline for brain MRI analysis: image retrieval, context search, and report generation.
Accelerator: GPU

"""

"""
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
"""

"""
## Setup

First, let's import the necessary libraries and configure our environment. We'll be using
KerasHub to download and run the language models, and we'll need to authenticate with
Kaggle to access the model weights. We'll also set up the JAX backend for optimal
performance on GPU accelerators.
"""

"""
## Kaggle Credentials Setup

If running in Google Colab, set up Kaggle API credentials using the `google.colab.userdata` module to enable downloading models and datasets from Kaggle. This step is only required in Colab environments.

"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

keras.config.set_dtype_policy("bfloat16")
import keras_hub
from PIL import Image
import matplotlib.pyplot as plt
from nilearn import datasets, image


"""
## Model Loading

Loads the vision model (for image feature extraction) and the Gemma3 vision-language model (for report generation). Returns both models for use in the RAG pipeline.
"""


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


"""
## Image and Caption Preparation

Prepares OASIS brain MRI images and generates captions for each image. Returns lists of image paths and captions.
"""


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


"""
## Image Visualization Utility

Displays a set of processed brain MRI images with their corresponding captions.
"""


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


"""
##Prediction Visualization Utility

Displays the query image and the most similar retrieved image from the database side by side.
"""


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


"""
##Image Feature Extraction

Extracts a feature vector from an image using the vision model.
"""


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


"""
## DB Reports

List of example radiology reports corresponding to each database image. Used as context for the RAG pipeline to generate new reports for query images.
"""
db_reports = [
    "MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.",
    "Normal MRI scan, no abnormal findings.",
    "Diffuse atrophy noted, no focal lesions.",
]

"""
##Output Cleaning Utility

Cleans the generated text output by removing prompt echoes and unwanted headers.
"""


def clean_generated_output(generated_text, prompt):
    """
    Remove prompt echo and header details from generated text.

    Args:
        generated_text (str): Raw generated text from the language model
        prompt (str): Original prompt used for generation

    Returns:
        str: Cleaned text without prompt echo and headers
    """
    # Remove the prompt from the beginning of the generated tex
    if generated_text.startswith(prompt):
        cleaned_text = generated_text[len(prompt) :].strip()
    else:
        # If prompt is not at the beginning, try to find and remove i
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


"""
## RAG Pipeline

Implements the Retrieval-Augmented Generation (RAG) pipeline:
- Extracts features from the query image and database images.
- Finds the most similar image in the database.
- Uses the retrieved report and the query image as input to the Gemma3 VLM to generate a new report.
Returns the index of the matched image, the retrieved report, and the generated report.
"""


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
    # Prepare the promp
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
    # Clean the generated outpu
    cleaned_output = clean_generated_output(output, prompt)
    return best_idx, retrieved_report, cleaned_output


"""
##Vision-Language Model (Direct Approach)

Generates a radiology report directly from the query image using the Gemma3 VLM, without retrieval.
"""


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


"""
##Main Execution

Runs the RAG pipeline: loads models, prepares data, and displays results.
"""
if __name__ == "__main__":
    """
    Main execution pipeline for RAG-based medical image analysis.

    This script demonstrates:
    1. Loading pre-trained vision and language models
    2. Processing OASIS brain MRI datase
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

"""
##Comparison: RAG Pipeline vs Direct VLM

- RAG pipeline (ViT + Gemma3 model): Produces more accurate and contextually relevant outputs by retrieving the most similar case and using its report as context for generation.
- Gemma3 VLM (direct, no retrieval): Produces more generic and often less accurate outputs, as the model does not have access to relevant context from similar cases.

In practice, the RAG approach leverages both image similarity and prior knowledge to generate more precise and clinically meaningful reports, while the direct VLM approach is limited to general knowledge and lacks case-specific context.
"""

"""
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
"""

"""
## Security Warning

⚠️ **IMPORTANT SECURITY AND PRIVACY CONSIDERATIONS**

This pipeline is for educational purposes only. For production use:

- Anonymize medical data following HIPAA guidelines
- Implement access controls and encryption
- Validate inputs and secure APIs
- Consult medical professionals for clinical decisions
- This system should NOT be used for actual medical diagnosis without proper validation
"""
