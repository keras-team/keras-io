import os
import json

MAPPING = {
    "Chapter 8: Image classification": {
        "url": "https://deeplearningwithpython.io/chapters/chapter08_image-classification",
        "files": [
            "image_classification_from_scratch",
            "image_classification_efficientnet_fine_tuning",
            "mnist_convnet",
            "3D_image_classification",
            "bit",
            "cct",
            "cait",
            "convmixer",
            "deit",
            "focal_modulation_network",
            "image_classification_using_global_context_vision_transformer",
            "image_classification_with_vision_transformer",
            "mlp_image_classification",
            "mobilevit",
            "perceiver_image_classification",
            "shiftvit",
            "swin_transformers",
            "token_learner",
            "vit_small_ds",
            "xray_classification_with_tpus",
            "attention_mil_classification",
        ]
    },
    "Chapter 9: ConvNet architecture patterns": {
        "url": "https://deeplearningwithpython.io/chapters/chapter09_convnet-architecture-patterns",
        "files": [
            "involution",
            "patch_convnet",
            "learnable_resizer",
        ]
    },
    "Chapter 10: Interpreting what ConvNets learn": {
        "url": "https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn",
        "files": [
            "grad_cam",
            "integrated_gradients",
            "visualizing_what_convnets_learn",
        ]
    },
    "Chapter 11: Image segmentation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter11_image-segmentation",
        "files": [
            "deeplabv3_plus",
            "basnet_segmentation",
            "fully_convolutional_network",
            "oxford_pets_image_segmentation",
            "pointnet_segmentation",
        ]
    },
    "Chapter 12: Object detection": {
        "url": "https://deeplearningwithpython.io/chapters/chapter12_object-detection",
        "files": [
            "retinanet",
            "yolov8",
            "object_detection_using_vision_transformer",
            "keypoint_detection",
        ]
    },
    "Chapter 16: Text generation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter16_text-generation",
        "files": [
            "image_captioning",
        ]
    },
     "Chapter 17: Image generation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter17_image-generation",
        "files": [
            "autoencoder",
        ]
    },
}

SECONDARY_MAPPING = {
    "image_classification_with_vision_transformer": "Chapter 15: Language models and the Transformer",
    "mobilevit": "Chapter 15: Language models and the Transformer",
    "swin_transformers": "Chapter 15: Language models and the Transformer",
    "vit_small_ds": "Chapter 15: Language models and the Transformer",
    "cct": "Chapter 15: Language models and the Transformer",
    "cait": "Chapter 15: Language models and the Transformer",
    "deit": "Chapter 15: Language models and the Transformer",
    "image_captioning": "Chapter 15: Language models and the Transformer",
}

CHAPTER_15_URL = "https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer"

BASE_DIR_MD = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/vision/md"
BASE_DIR_IPYNB = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/vision/ipynb"

def get_chapter_link(chapter_name):
    if chapter_name == "Chapter 15: Language models and the Transformer":
        return CHAPTER_15_URL
    for key, val in MAPPING.items():
        if key == chapter_name:
            return val["url"]
    return None

def process_files():
    # Build mapping from filename (no ext) to list of (chapter_name, url)
    file_to_chapters = {}
    
    for chapter, data in MAPPING.items():
        for fname_no_ext in data["files"]:
            if fname_no_ext not in file_to_chapters:
                file_to_chapters[fname_no_ext] = []
            file_to_chapters[fname_no_ext].append((chapter, data["url"]))

    for fname_no_ext, chapter in SECONDARY_MAPPING.items():
        if fname_no_ext not in file_to_chapters:
             file_to_chapters[fname_no_ext] = []
        exists = any(c == chapter for c, u in file_to_chapters[fname_no_ext])
        if not exists:
            file_to_chapters[fname_no_ext].append((chapter, get_chapter_link(chapter)))

    # Sort chapters
    for fname_no_ext in file_to_chapters:
        def get_chapter_num(c_tuple):
            try:
                return int(c_tuple[0].split(":")[0].replace("Chapter ", ""))
            except:
                return 999
        file_to_chapters[fname_no_ext].sort(key=get_chapter_num)

    # Process files
    for fname_no_ext, chapters in file_to_chapters.items():
        # Update .md
        md_path = os.path.join(BASE_DIR_MD, fname_no_ext + ".md")
        if os.path.exists(md_path):
            update_md(md_path, chapters)
        
        # Update .ipynb
        ipynb_path = os.path.join(BASE_DIR_IPYNB, fname_no_ext + ".ipynb")
        if os.path.exists(ipynb_path):
            update_ipynb(ipynb_path, chapters)

def update_md(fpath, chapters):
    with open(fpath, "r") as f:
        content = f.read()

    if "Relevant Chapters from Deep Learning with Python" in content:
        # Already added
        return

    # Create block
    # Note: user example showed "---" before the section in .md
    block = "\n---\n## Relevant Chapters from Deep Learning with Python\n"
    for chap_name, chap_url in chapters:
        block += f"- [{chap_name}]({chap_url})\n"
    
    # Append
    # Check if file ends with newline
    if not content.endswith("\n"):
        content += "\n"
    
    with open(fpath, "w") as f:
        f.write(content + block)
    print(f"Updated MD: {os.path.basename(fpath)}")

def update_ipynb(fpath, chapters):
    with open(fpath, "r") as f:
        try:
            notebook = json.load(f)
        except Exception as e:
            print(f"Error reading notebook {fpath}: {e}")
            return

    # Check if last cell is already the relevant chapters
    cells = notebook.get("cells", [])
    if not cells:
        return # Empty notebook?

    last_cell = cells[-1]
    if last_cell["cell_type"] == "markdown":
        source_str = "".join(last_cell["source"])
        if "Relevant Chapters from Deep Learning with Python" in source_str:
            return

    # Create new cell
    source_lines = [
        "## Relevant Chapters from Deep Learning with Python\n"
    ]
    for chap_name, chap_url in chapters:
        source_lines.append(f"- [{chap_name}]({chap_url})\n")

    new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

    cells.append(new_cell)
    notebook["cells"] = cells

    with open(fpath, "w") as f:
        json.dump(notebook, f, indent=1) # Keras notebooks usually use indent=1? or 2? Most are 1 or 2.
        # Let's check indentation of original file by reading partially or just use default json dump.
        # Default json dump usually puts no space after separator, but standard python json.dump with indent adds it.
        # Actually simplest is to just write it out.
        # Note: indent=1 is common in some tools, but let's see.
        # I'll use indent=1 based on the viewed file in step 88 which seemed compact. 
        # Actually step 88 output:
        # 1: {
        # 2:  "cells": [
        # It uses 1 space indent.
    
    print(f"Updated IPYNB: {os.path.basename(fpath)}")

if __name__ == "__main__":
    process_files()
