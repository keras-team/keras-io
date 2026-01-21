import os

MAPPING = {
    "Chapter 8: Image classification": {
        "url": "https://deeplearningwithpython.io/chapters/chapter08_image-classification",
        "files": [
            "image_classification_from_scratch.py",
            "image_classification_efficientnet_fine_tuning.py",
            "mnist_convnet.py",
            "3D_image_classification.py",
            "bit.py",
            "cct.py",
            "cait.py",
            "convmixer.py",
            "deit.py",
            "focal_modulation_network.py",
            "image_classification_using_global_context_vision_transformer.py",
            "image_classification_with_vision_transformer.py",
            "mlp_image_classification.py",
            "mobilevit.py",
            "perceiver_image_classification.py",
            "shiftvit.py",
            "swin_transformers.py",
            "token_learner.py",
            "vit_small_ds.py",
            "xray_classification_with_tpus.py",
            "attention_mil_classification.py",
        ]
    },
    "Chapter 9: ConvNet architecture patterns": {
        "url": "https://deeplearningwithpython.io/chapters/chapter09_convnet-architecture-patterns",
        "files": [
            "involution.py",
            "patch_convnet.py",
            "learnable_resizer.py",
        ]
    },
    "Chapter 10: Interpreting what ConvNets learn": {
        "url": "https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn",
        "files": [
            "grad_cam.py",
            "integrated_gradients.py",
            "visualizing_what_convnets_learn.py",
        ]
    },
    "Chapter 11: Image segmentation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter11_image-segmentation",
        "files": [
            "deeplabv3_plus.py",
            "basnet_segmentation.py",
            "fully_convolutional_network.py",
            "oxford_pets_image_segmentation.py",
            "pointnet_segmentation.py",
        ]
    },
    "Chapter 12: Object detection": {
        "url": "https://deeplearningwithpython.io/chapters/chapter12_object-detection",
        "files": [
            "retinanet.py",
            "yolov8.py",
            "object_detection_using_vision_transformer.py",
            "keypoint_detection.py",
        ]
    },
    "Chapter 16: Text generation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter16_text-generation",
        "files": [
            "image_captioning.py",
        ]
    },
     "Chapter 17: Image generation": {
        "url": "https://deeplearningwithpython.io/chapters/chapter17_image-generation",
        "files": [
            "autoencoder.py",
        ]
    },
}

# Secondary mappings for files that need multiple chapters
SECONDARY_MAPPING = {
    "image_classification_with_vision_transformer.py": "Chapter 15: Language models and the Transformer",
    "mobilevit.py": "Chapter 15: Language models and the Transformer",
    "swin_transformers.py": "Chapter 15: Language models and the Transformer",
    "vit_small_ds.py": "Chapter 15: Language models and the Transformer",
    "cct.py": "Chapter 15: Language models and the Transformer",
    "cait.py": "Chapter 15: Language models and the Transformer",
    "deit.py": "Chapter 15: Language models and the Transformer",
    "image_captioning.py": "Chapter 15: Language models and the Transformer",
}

CHAPTER_15_URL = "https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer"

BASE_DIR = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/vision"

def get_chapter_link(chapter_name):
    if chapter_name == "Chapter 15: Language models and the Transformer":
        return CHAPTER_15_URL
    for key, val in MAPPING.items():
        if key == chapter_name:
            return val["url"]
    return None

def process_files():
    # Invert mapping for easier lookup
    file_to_chapters = {}
    
    for chapter, data in MAPPING.items():
        for fname in data["files"]:
            if fname not in file_to_chapters:
                file_to_chapters[fname] = []
            file_to_chapters[fname].append((chapter, data["url"]))

    # Add secondary mappings
    for fname, chapter in SECONDARY_MAPPING.items():
        if fname not in file_to_chapters:
             # Should be covered by primary mapping usually, but just in case
             file_to_chapters[fname] = []
        
        # Check if already present
        exists = any(c == chapter for c, u in file_to_chapters[fname])
        if not exists:
            file_to_chapters[fname].append((chapter, get_chapter_link(chapter)))

    for fname, chapters in file_to_chapters.items():
        # Sort chapters by chapter number to look nice
        # Extract number from "Chapter N:"
        def get_chapter_num(c_tuple):
            try:
                return int(c_tuple[0].split(":")[0].replace("Chapter ", ""))
            except:
                return 999
        
        chapters.sort(key=get_chapter_num)
        
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Skipping missing file: {fname}")
            continue
            
        with open(fpath, "r") as f:
            content = f.read()
            
        if "Relevant Chapters" in content:
            print(f"Skipping {fname}, already has Relevant Chapters section")
            continue
            
        # Find the end of the first docstring
        # We assume standard Keras example format where the file starts with """
        if not content.strip().startswith('"""'):
            print(f"Skipping {fname}, does not start with docstring")
            continue
            
        # Find the second """ (end of first docstring)
        # We start searching from index 3 to skip the first """
        end_docstring_idx = content.find('"""', 3)
        
        if end_docstring_idx == -1:
            print(f"Skipping {fname}, could not find end of docstring")
            continue
            
        # Check if there is a newline before the closing quotes
        # We want to insert before the closing quotes
        
        insertion = "\nRelevant Chapters\n"
        for chap_name, chap_url in chapters:
            insertion += f"- [{chap_name}]({chap_url})\n"
        
        # Construct new content
        # Check indentation of the closing quotes? Usually it is 0 for file-level docstring
        
        new_content = content[:end_docstring_idx] + insertion + content[end_docstring_idx:]
        
        with open(fpath, "w") as f:
            f.write(new_content)
        print(f"Updated {fname}")

if __name__ == "__main__":
    process_files()
