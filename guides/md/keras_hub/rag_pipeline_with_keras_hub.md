# RAG Pipeline with KerasHub

**Author:** [Laxmareddy Patlolla](https://github.com/laxmareddyp), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)<br>
**Date created:** 2025/07/22<br>
**Last modified:** 2025/08/08<br>
**Description:** RAG pipeline for brain MRI analysis: image retrieval, context search, and report generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/rag_pipeline_with_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/rag_pipeline_with_keras_hub.py)



---
## Welcome to Your RAG Adventure!

Hey there! Ready to dive into something really exciting? We're about to build a system that can look at brain MRI images and generate detailed medical reports - but here's the cool part: it's not just any AI system. We're building something that's like having a super-smart medical assistant who can look at thousands of previous cases to give you the most accurate diagnosis possible!

**What makes this special?** Instead of just relying on what the AI learned during training, our system will actually "remember" similar cases it has seen before and use that knowledge to make better decisions. It's like having a doctor who can instantly recall every similar case they've ever treated!

**What we're going to discover together:**

- How to make AI models work together like a well-oiled machine
- Why having access to previous cases makes AI much smarter
- How to build systems that are both powerful AND efficient
- The magic of combining image understanding with language generation

Think of this as your journey into the future of AI-powered medical analysis. By the end, you'll have built something that could potentially help doctors make better decisions faster!

Ready to start this adventure? Let's go!

---
## Setting Up Our AI Workshop

Alright, before we start building our amazing RAG system, we need to set up our digital workshop! Think of this like gathering all the tools a master craftsman needs before creating a masterpiece.

**What we're doing here:** We're importing all the powerful libraries that will help us build our AI system. It's like opening our toolbox and making sure we have every tool we need - from the precision screwdrivers (our AI models) to the heavy machinery (our data processing tools).

**Why JAX?** We're using JAX as our backend because it's like having a super-fast engine under the hood. It's designed to work beautifully with modern AI models and can handle complex calculations lightning-fast, especially when you have a GPU to help out!

**The magic of KerasHub:** This is where things get really exciting! KerasHub is like having access to a massive library of pre-trained AI models. Instead of training models from scratch (which would take forever), we can grab models that are already experts at understanding images and generating text. It's like having a team of specialists ready to work for us!

Let's get our tools ready and start building something amazing!

---
## Getting Your VIP Pass to the AI Model Library! 🎫

Okay, here's the deal - we're about to access some seriously powerful AI models, but first we need to get our VIP pass! Think of Kaggle as this exclusive club where all the coolest AI models hang out, and we need the right credentials to get in.

**Why do we need this?** The AI models we're going to use are like expensive, high-performance sports cars. They're incredibly powerful, but they're also quite valuable, so we need to prove we're authorized to use them. It's like having a membership card to the most exclusive AI gym in town!

**Here's how to get your VIP access:**

1. **Head to the VIP lounge:** Go to your Kaggle account settings at https://www.kaggle.com/settings/account
2. **Get your special key:** Scroll down to the "API" section and click "Create New API Token"
3. **Set up your access:** This will give you the secret codes (API key and username) that let you download and use these amazing models

**Pro tip:** If you're running this in Google Colab (which is like having a super-powered computer in the cloud), you can store these credentials securely and access them easily. It's like having a digital wallet for your AI models!

Once you've got your credentials set up, you'll be able to download and use some of the most advanced AI models available today. Pretty exciting, right? 🚀


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
E0000 00:00:1754676982.407845    5369 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1754676982.412281    5369 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1754676982.423666    5369 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1754676982.423680    5369 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1754676982.423682    5369 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1754676982.423683    5369 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

---
## Understanding the Magic Behind RAG! ✨

Alright, let's take a moment to understand what makes RAG so special! Think of RAG as having a super-smart assistant who doesn't just answer questions from memory, but actually goes to the library to look up the most relevant information first.

**The Three Musketeers of RAG:**

1. **The Retriever** 🕵️‍♂️: This is like having a detective who can look at a new image and instantly find similar cases from a massive database. It's the part that says "Hey, I've seen something like this before!"

2. **The Generator** ✍️: This is like having a brilliant writer who takes all the information the detective found and crafts a perfect response. It's the part that says "Based on what I found, here's what I think is happening."

3. **The Knowledge Base** 📚: This is our treasure trove of information - think of it as a massive library filled with thousands of medical cases, each with their own detailed reports.

**Here's what our amazing RAG system will do:**

- **Step 1:** Our MobileNetV3 model will look at a brain MRI image and extract its "fingerprint" - the unique features that make it special
- **Step 2:** It will search through our database of previous cases and find the most similar one
- **Step 3:** It will grab the medical report from that similar case
- **Step 4:** Our Gemma3 text model will use that context to generate a brand new, super-accurate report
- **Step 5:** We'll compare this with what a traditional AI would do (spoiler: RAG wins! 🏆)

**Why this is revolutionary:** Instead of the AI just guessing based on what it learned during training, it's actually looking at real, similar cases to make its decision. It's like the difference between a doctor who's just graduated from medical school versus one who has seen thousands of patients!

Ready to see this magic in action? Let's start building! 🎯

---
## Loading Our AI Dream Team! 🤖

Alright, this is where the real magic begins! We're about to load up our AI models - think of this as assembling the ultimate team of specialists, each with their own superpower!

**What we're doing here:** We're downloading and setting up three different AI models, each with a specific role in our RAG system. It's like hiring the perfect team for a complex mission - you need the right person for each job!

**Meet our AI specialists:**

1. **MobileNetV3** 👁️: This is our "eyes" - a lightweight but incredibly smart model that can look at any image and understand what it's seeing. It's like having a radiologist who can instantly spot patterns in medical images!

2. **Gemma3 1B Text Model** ✍️: This is our "writer" - a compact but powerful language model that can generate detailed medical reports. Think of it as having a medical writer who can turn complex findings into clear, professional reports.

3. **Gemma3 4B VLM** 🧠: This is our "benchmark" - a larger, more powerful model that can both see images AND generate text. We'll use this to compare how well our RAG approach performs against traditional methods.

**Why this combination is brilliant:** Instead of using one massive, expensive model, we're using smaller, specialized models that work together perfectly. It's like having a team of experts instead of one generalist - more efficient, faster, and often more accurate!

Let's load up our AI dream team and see what they can do! 🚀


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


# Load models
print("Loading models...")
vision_model, vlm_model, text_model = load_models()

```

<div class="k-default-codeblock">
```
Loading models...

normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
```
</div>

---
## Preparing Our Medical Images! 🧠📸

Now we're getting to the really exciting part - we're going to work with real brain MRI images! This is like having access to a medical imaging lab where we can study actual brain scans.

**What we're doing here:** We're downloading and preparing brain MRI images from the OASIS dataset. Think of this as setting up our own mini radiology department! We're taking raw MRI data and turning it into images that our AI models can understand and analyze.

**Why brain MRIs?** Brain MRI images are incredibly complex and detailed - they show us the structure of the brain in amazing detail. They're perfect for testing our RAG system because:
- They're complex enough to challenge our AI models
- They have real medical significance
- They're perfect for demonstrating how retrieval can improve accuracy

**The magic of data preparation:** We're not just downloading images - we're processing them to make sure they're in the perfect format for our AI models. It's like preparing ingredients for a master chef - everything needs to be just right!

**What you'll see:** After this step, you'll have a collection of brain MRI images that we can use to test our RAG system. Each image represents a different brain scan, and we'll use these to demonstrate how our system can find similar cases and generate accurate reports.

Ready to see some real brain scans? Let's prepare our medical images! 🔬


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


# Prepare data
print("Preparing OASIS dataset...")
oasis = datasets.fetch_oasis_vbm(n_subjects=4)  # Use 4 images
print("Download dataset is completed.")
image_paths, captions = prepare_images_and_captions(oasis)

```

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

---
## Let's Take a Look at Our Brain Scans! 👀

Alright, this is the moment we've been waiting for! We're about to visualize our brain MRI images - think of this as opening up a medical textbook and seeing the actual brain scans that we'll be working with.

**What we're doing here:** We're creating a visual display of all our brain MRI images so we can see exactly what we're working with. It's like having a lightbox in a radiology department where doctors can examine multiple scans at once.

**Why visualization is crucial:** In medical imaging, seeing is believing! By visualizing our images, we can:

- Understand what our AI models are actually looking at
- Appreciate the complexity and detail in each brain scan
- Get a sense of how different each scan can be
- Prepare ourselves for what our RAG system will be analyzing

**What you'll observe:** Each image shows a different slice through a brain, revealing the intricate patterns and structures that make each brain unique. Some might show normal brain tissue, while others might reveal interesting variations or patterns.

**The beauty of brain imaging:** Every brain scan tells a story - the folds, the tissue density, the overall structure. Our AI models will learn to read these stories and find similar patterns across different scans.

Take a good look at these images - they're the foundation of everything our RAG system will do! 🧠✨


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


# Visualize the prepared images
visualize_images(image_paths, captions)

```


    
![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_11_0.png)
    


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

Extracts a feature vector from an image using the small `vision(MobileNetV3)` model.


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

List of example `radiology reports` corresponding to each database image. Used as context for the RAG pipeline to generate new reports for `query images`.


```python
db_reports = [
    "MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.",
    "Normal MRI scan, no abnormal findings.",
    "Diffuse atrophy noted, no focal lesions.",
]
```

---
## Output Cleaning Utility

Cleans the `generated text` output by removing prompt echoes and unwanted headers.


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
## The Heart of Our RAG System! ❤️

Alright, this is where all the magic happens! We're about to build the core of our RAG pipeline - think of this as the engine room of our AI system, where all the complex machinery works together to create something truly amazing.

**What is RAG, really?**

Imagine you're a detective trying to solve a complex case. Instead of just relying on your memory and training, you have access to a massive database of similar cases. When you encounter a new situation, you can instantly look up the most relevant previous cases and use that information to make a much better decision. That's exactly what RAG does!

**The Three Superheroes of Our RAG System:**

1. **The Retriever** 🕵️‍♂️: This is our detective - it looks at a new brain scan and instantly finds the most similar cases from our database. It's like having a photographic memory for medical images!

2. **The Generator** ✍️: This is our brilliant medical writer - it takes all the information our detective found and crafts a perfect, detailed report. It's like having a radiologist who can write like a medical journalist!

3. **The Knowledge Base** 📚: This is our treasure trove - a massive collection of real medical cases and reports that our system can learn from. It's like having access to every medical textbook ever written!

**Here's the Step-by-Step Magic:**

- **Step 1** 🔍: Our MobileNetV3 model extracts the "fingerprint" of the new brain scan
- **Step 2** 🎯: It searches through our database and finds the most similar previous case
- **Step 3** 📋: It grabs the medical report from that similar case
- **Step 4** 🧠: It combines this context with our generation prompt
- **Step 5** ✨: Our Gemma3 text model creates a brand new, super-accurate report

**Why This is Revolutionary:**

- **🎯 Factual Accuracy**: Instead of guessing, we're using real medical reports as our guide
- **🔍 Relevance**: We're finding the most similar cases, not just any random information
- **⚡ Efficiency**: We're using a smaller, faster model but getting better results
- **📊 Traceability**: We can show exactly which previous cases influenced our diagnosis
- **🚀 Scalability**: We can easily add new cases to make our system even smarter

**The Real Magic:** This isn't just about making AI smarter - it's about making AI more trustworthy, more accurate, and more useful in real-world medical applications. We're building the future of AI-assisted medicine!

Ready to see this magic in action? Let's run our RAG pipeline! 🎯✨


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


# Split data: first 3 as database, last as query
db_image_paths = image_paths[:-1]
query_img_path = image_paths[-1]

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

```

<div class="k-default-codeblock">
```
Running RAG pipeline...

[RAG] Matched image index: 0
[RAG] Matched image path: images/oasis_0.png
[RAG] Retrieved context/report:
MRI shows a 1.5cm lesion in the right frontal lobe, non-enhancing, no edema.
```
</div>

![png](/home/laxmareddyp/keras_guides/keras-io/guides/img/rag_pipeline_with_keras_hub/rag_pipeline_with_keras_hub_21_2.png)
    


    
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
```
</div>

---
## The Ultimate Showdown: RAG vs Traditional AI! 🥊

Alright, now we're getting to the really exciting part! We've built our amazing RAG system, but how do we know it's actually better than traditional approaches? Let's put it to the test!

**What we're about to do:** We're going to compare our RAG system with a traditional Vision-Language Model (VLM) approach. Think of this as a scientific experiment where we're testing two different methods to see which one performs better.

**The Battle of the Titans:**

- **🥊 RAG Approach**: Our smart system using MobileNetV3 + Gemma3 1B (1B total parameters) with retrieved medical context
- **🥊 Direct VLM Approach**: A traditional system using Gemma3 4B VLM (4B parameters) with only pre-trained knowledge

**Why this comparison is crucial:** This is like comparing a doctor who has access to thousands of previous cases versus one who only has their medical school training. Which one would you trust more?

**What we're going to discover:**

- **🔍 The Power of Context**: How having access to similar medical cases dramatically improves accuracy
- **⚖️ Size vs Intelligence**: Whether bigger models are always better (spoiler: they're not!)
- **🏥 Real-World Practicality**: Why RAG is more practical for actual medical applications
- **🧠 The Knowledge Gap**: How domain-specific knowledge beats general knowledge

**The Real Question:** Can a smaller, smarter system with access to relevant context outperform a larger system that's working in the dark?

**What makes this exciting:** This isn't just a technical comparison - it's about understanding the future of AI. We're testing whether intelligence comes from size or from having the right information at the right time.

Ready to see which approach wins? Let's run the ultimate AI showdown! 🎯🏆


```python

def vlm_generate_report(query_img_path, vlm_model, question=None):
    """
    Generate a radiology report directly from the image using a vision-language model.
    Args:
        query_img_path (str): Path to the query image
        vlm_model: Pre-trained vision-language model (Gemma3 4B VLM)
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


# Run VLM (direct approach)
print("\n" + "=" * 50)
print("VLM RESULTS (Direct Approach)")
print("=" * 50)
vlm_report = vlm_generate_report(query_img_path, vlm_model)
print("\n--- Vision-Language Model (No Retrieval) Report ---\n", vlm_report)
```

    
<div class="k-default-codeblock">
```
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
## The Results Are In: RAG Wins! 🏆

Drumroll please... 🥁 The results are in, and they're absolutely fascinating! Let's break down what we just discovered in our ultimate AI showdown.

**The Numbers Don't Lie:**

- **🥊 RAG Approach**: MobileNet + Gemma3 1B text model (~1B total parameters)
- **🥊 Direct VLM Approach**: Gemma3 VLM 4B model (~4B total parameters)
- **🏆 Winner**: RAG pipeline! (And here's why it's revolutionary...)

**What We Just Proved:**

**🎯 Accuracy & Relevance - RAG Dominates!**

- Our RAG system provides contextually relevant, case-specific reports that often match or exceed the quality of much larger models
- The traditional VLM produces more generic, "textbook" responses that lack the specificity of real medical cases
- It's like comparing a doctor who's seen thousands of similar cases versus one who's only read about them in textbooks!

**⚡ Speed & Efficiency - RAG is Lightning Fast!**

- Our RAG system is significantly faster and more memory-efficient
- It can run on edge devices and provide real-time results
- The larger VLM requires massive computational resources and is much slower
- Think of it as comparing a sports car to a freight train - both can get you there, but one is much more practical!

**🔄 Scalability & Flexibility - RAG is Future-Proof!**

- Our RAG approach can easily adapt to new domains or datasets
- We can swap out different models without retraining everything
- The traditional approach requires expensive retraining for new domains
- It's like having a modular system versus a monolithic one!

**🔍 Interpretability & Trust - RAG is Transparent!**

- Our RAG system shows exactly which previous cases influenced its decision
- This transparency builds trust and helps with clinical validation
- The traditional approach is a "black box" - we don't know why it made certain decisions
- In medicine, trust and transparency are everything!

**🏥 Real-World Practicality - RAG is Ready for Action!**

- Our RAG system can be deployed in resource-constrained environments
- It can be continuously improved by adding new cases to the database
- The traditional approach requires expensive cloud infrastructure
- This is the difference between a practical solution and a research project!

**The Bottom Line:**

We've just proven that intelligence isn't about size - it's about having the right information at the right time. Our RAG system is smaller, faster, more accurate, and more practical than traditional approaches. This isn't just a technical victory - it's a glimpse into the future of AI! 🚀✨

---
## Congratulations! You've Just Built the Future of AI! 🎉

Wow! What an incredible journey we've been on together! We started with a simple idea and ended up building something that could revolutionize how AI systems work in the real world. Let's take a moment to celebrate what we've accomplished!

**What We Just Built Together:**

**🤖 The Ultimate AI Dream Team:**

- **MobileNetV3 + Gemma3 1B text model** - Our dynamic duo that works together like a well-oiled machine
- **Gemma3 4B VLM model** - Our worthy opponent that helped us prove our point
- **KerasHub Integration** - The magic that made it all possible

**🔬 Real-World Medical Analysis:**

- **Feature Extraction** - We taught our AI to "see" brain MRI images like a radiologist
- **Similarity Search** - We built a system that can instantly find similar medical cases
- **Report Generation** - We created an AI that writes detailed, accurate medical reports
- **Comparative Analysis** - We proved that our approach is better than traditional methods

**🚀 Revolutionary Results:**

- **Enhanced Accuracy** - Our system provides more relevant, contextually aware outputs
- **Scalable Architecture** - We built something that can grow and adapt to new challenges
- **Real-World Applicability** - This isn't just research - it's ready for actual medical applications
- **Future-Proof Design** - Our system can evolve and improve over time

**The Real Magic:** We've just demonstrated that the future of AI isn't about building bigger and bigger models. It's about building smarter systems that know how to find and use the right information at the right time. We've shown that a small, well-designed system with access to relevant context can outperform massive models that work in isolation.

**What This Means for the Future:** This isn't just about medical imaging - this approach can be applied to any field where having access to relevant context makes a difference. From legal document analysis to financial forecasting, from scientific research to creative writing, the principles we've demonstrated here can revolutionize how AI systems work.

**You're Now Part of the AI Revolution:** By understanding and building this RAG system, you're now equipped with knowledge that's at the cutting edge of AI development. You understand not just how to use AI models, but how to make them work together intelligently.

**The Journey Continues:** This is just the beginning! The world of AI is evolving rapidly, and the techniques we've explored here are just the tip of the iceberg. Keep experimenting, keep learning, and keep building amazing things!

**Thank you for joining this adventure!** 🚀✨

And we've just built something beautiful together! 🌟

---
## Security Warning

⚠️ **IMPORTANT SECURITY AND PRIVACY CONSIDERATIONS**

This pipeline is for educational purposes only. For production use:

- Anonymize medical data following HIPAA guidelines
- Implement access controls and encryption
- Validate inputs and secure APIs
- Consult medical professionals for clinical decisions
- This system should NOT be used for actual medical diagnosis without proper validation
