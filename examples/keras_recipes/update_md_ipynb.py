
import os
import json

md_dir = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/keras_recipes/md"
ipynb_dir = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/keras_recipes/ipynb"

chapters = {
    "04": ("Chapter 4: Classification and regression", "https://deeplearningwithpython.io/chapters/chapter04_classification-and-regression"),
    "05": ("Chapter 5: Fundamentals of machine learning", "https://deeplearningwithpython.io/chapters/chapter05_fundamentals-of-ml"),
    "06": ("Chapter 6: The universal workflow of machine learning", "https://deeplearningwithpython.io/chapters/chapter06_universal-workflow-of-ml"),
    "07": ("Chapter 7: A deep dive on Keras", "https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras"),
    "08": ("Chapter 8: Image classification", "https://deeplearningwithpython.io/chapters/chapter08_image-classification"),
    "14": ("Chapter 14: Text classification", "https://deeplearningwithpython.io/chapters/chapter14_text-classification"),
    "15": ("Chapter 15: Language models and the Transformer", "https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer"),
    "16": ("Chapter 16: Text generation", "https://deeplearningwithpython.io/chapters/chapter16_text-generation"),
    "18": ("Chapter 18: Best practices for the real world", "https://deeplearningwithpython.io/chapters/chapter18_best-practices-for-the-real-world"),
}

file_map = {
    "antirectifier": ["07"],
    "approximating_non_function_mappings": ["04"],
    "bayesian_neural_networks": ["05"],
    "better_knowledge_distillation": ["18"],
    "creating_tfrecords": ["06"],
    "debugging_tips": ["07"],
    "endpoint_layer_pattern": ["07"],
    "float8_training_and_inference_with_transformer": ["15", "18"],
    "memory_efficient_embeddings": ["14"],
    "packaging_keras_models_for_wide_distribution": ["18"],
    "parameter_efficient_finetuning_of_gemma_with_lora_and_qlora": ["16"],
    "reproducibility_recipes": ["06"],
    "sample_size_estimate": ["05"],
    "sklearn_metric_callbacks": ["07"],
    "subclassing_conv_layers": ["07", "08"],
    "tensorflow_numpy_models": ["07"],
    "tf_serving": ["18"],
    "tfrecord": ["06"],
    "trainer_pattern": ["07"],
}

def get_links_md(chapter_ids):
    links = []
    for cid in chapter_ids:
        title, url = chapters[cid]
        links.append(f"- [{title}]({url})")
    return "\n## Relevant Chapters from Deep Learning with Python\n" + "\n".join(links) + "\n"


# Update MD files
for basename, chapter_ids in file_map.items():
    md_path = os.path.join(md_dir, basename + ".md")
    if not os.path.exists(md_path):
        print(f"MD file not found: {md_path}")
        continue
        
    with open(md_path, "r") as f:
        content = f.read()
    
    section_title = "## Relevant Chapters from Deep Learning with Python"
    
    if section_title in content:
        # Check if --- is present before the section
        # We expect "\n---\n## Relevant Chapters..."
        # But my previous script wrote "\n\n## Relevant Chapters..." (or single newline)
        
        if f"\n---\n{section_title}" in content:
            print(f"Skipping MD {basename}, already has horizontal rule.")
        else:
            # Replace the section title with --- \n title
            # Use strict replacement of the exact string added previously if possible, 
            # or just regex replace or simple string replace
            # My previous script added: "\n" + '"""' ... wait, that was for py files.
            # IN this file I used: "\n## Relevant Chapters..." plus links
            
            # Use replace with a check
            if f"\n{section_title}" in content:
                 new_content = content.replace(f"\n{section_title}", f"\n---\n{section_title}")
                 with open(md_path, "w") as f:
                     f.write(new_content)
                 print(f"Fixed MD {basename} (added ---)")
            else:
                 print(f"Could not automatically fix {basename}, please check manually.")
    else:
        # This part shouldn't be reached if previous run worked, but just in case
        links_block = get_links_md(chapter_ids)
        # Add --- for new additions
        links_block = "\n---\n" + links_block.strip() + "\n"
        
        if not content.endswith("\n"):
            links_block = "\n" + links_block
        with open(md_path, "w") as f:
            f.write(content + links_block)
        print(f"Updated MD {basename} (fresh add)")
