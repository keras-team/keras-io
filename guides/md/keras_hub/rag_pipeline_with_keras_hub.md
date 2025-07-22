# RAG Pipeline with KerasHub

**Author:** [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)<br>
**Date created:** 2025/07/22<br>
**Last modified:** 2025/07/22<br>
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

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

keras.config.set_dtype_policy("bfloat16")
import keras_hub
from PIL import Image
import matplotlib.pyplot as plt
from nilearn import datasets, image

```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1753176247.504811   34999 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1753176247.509156   34999 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1753176247.520234   34999 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753176247.520247   34999 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753176247.520249   34999 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753176247.520250   34999 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

---
## Model Loading

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

##Prediction Visualization Utility

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

##Image Feature Extraction

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

##Output Cleaning Utility

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

```

---
## RAG Pipeline

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

```

##Vision-Language Model (Direct Approach)

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

##Main Execution

Runs the RAG pipeline: loads models, prepares data, and displays results.


```python
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
```

<div class="k-default-codeblock">
```
Loading models...

normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

Preparing OASIS dataset...
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[</span><span style="color: #000080; text-decoration-color: #000080">fetch_oasis_vbm</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">]</span> Dataset found in <span style="color: #800080; text-decoration-color: #800080">/home/laxmareddyp/nilearn_data/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">oasis1</span>
</pre>



<div class="k-default-codeblock">
```
Saved 4 OASIS Brain MRI images: ['images/oasis_0.png', 'images/oasis_1.png', 'images/oasis_2.png', 'images/oasis_3.png']
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_5.png)
    


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

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_24_9.png)
    


    
<div class="k-default-codeblock">
```
==================================================
RAG PIPELINE RESULTS
==================================================

--- Retrieved Report ---
 MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.

--- Generated Report ---
 ---

**Radiology Report**

**Clinical Indication:** [Clinical Indication - Placeholder - e.g., Headache, Seizures, Cognitive Changes]

**Findings:**

*   **Brain:** The brain appears grossly normal in size and signal intensity. There is a 1.5 cm lesion located in the right frontal lobe. The lesion is non-enhancing on post-contrast imaging. No surrounding edema is identified. The lesion demonstrates a homogenous appearance.
*   **White Matter:** No evidence of white matter abnormalities is seen.
*   **Cerebellum:** The cerebellum appears normal in size and signal intensity.
*   **Brainstem:** The brainstem appears normal in size and signal intensity.
*   **Ventricles:** The ventricles are of normal size and morphology.

**Impression:**

1.  **Diagnostic Impression:**  The most likely diagnosis is a small, benign, non-enhancing lesion in the right frontal lobe.  Differential diagnoses include a small metastatic lesion, a benign hamartoma, or a small, early-stage low-grade tumor.

**Diagnostic Reasoning:**

The key features supporting this impression are:

*   **Size:** The lesion is relatively small (1.5 cm).
*   **Non-enhancing:** The absence of enhancement with contrast is highly suggestive of a benign process.  Contrast enhancement is typically seen in active tumor growth or inflammation.
*   **No Edema:** The lack of surrounding edema indicates that the lesion is not causing significant inflammation or mass effect.
*   **Homogenous Appearance:** A homogenous appearance on MRI suggests a relatively uniform tissue composition, further reducing the likelihood of a highly aggressive tumor.

However, it is crucial to acknowledge that the differential diagnosis remains broad.  While the findings are reassuring, further investigation may be warranted to definitively rule out other possibilities, particularly if the patient presents with concerning symptoms.

**Possible Treatment Options:**

Given the benign appearance of the lesion, immediate treatment is generally not indicated.  However, the following options should be considered:

1.  **Observation:**  Serial MRI scans (e.g., every 6-12 months) to monitor for any changes in size or signal characteristics. This is the most common approach for small, stable lesions.
2.  **Neuropsychological Testing:** If the patient is experiencing cognitive changes, neuropsychological testing can be performed to assess for any subtle functional deficits.
3.  **Further Investigation (if indicated):** If the lesion demonstrates any change on follow-up imaging, or if the patient develops new or worsening symptoms, further investigation such as a biopsy or additional imaging (e.g., PET/CT) may be considered to determine the exact nature of the lesion.

**Recommendations:**

*   Discuss these findings with the patient and their referring physician.
*   Recommend serial MRI surveillance to monitor the lesion.
*   Consider neuropsychological testing if clinically indicated.

**Radiologist:** [Radiologist Name - Placeholder]
**Date:** [Date - Placeholder]

---

**Disclaimer:** *This report is based solely on the provided information and should not be considered a definitive diagnosis.  A complete clinical evaluation and correlation with the patient's history and symptoms are necessary for accurate diagnosis and treatment planning.*
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

##Comparison: RAG Pipeline vs Direct VLM

- RAG pipeline (ViT + Gemma3 model): Produces more accurate and contextually relevant outputs by retrieving the most similar case and using its report as context for generation.
- Gemma3 VLM (direct, no retrieval): Produces more generic and often less accurate outputs, as the model does not have access to relevant context from similar cases.

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
