import os
import json

base_path = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/nlp"
md_base_path = os.path.join(base_path, "md")
ipynb_base_path = os.path.join(base_path, "ipynb")

chapters = {
    1: ("What is deep learning?", "https://deeplearningwithpython.io/chapters/chapter01_what-is-deep-learning"),
    2: ("The mathematical building blocks of neural networks", "https://deeplearningwithpython.io/chapters/chapter02_mathematical-building-blocks"),
    3: ("Introduction to TensorFlow, PyTorch, JAX, and Keras", "https://deeplearningwithpython.io/chapters/chapter03_introduction-to-ml-frameworks"),
    4: ("Classification and regression", "https://deeplearningwithpython.io/chapters/chapter04_classification-and-regression"),
    5: ("Fundamentals of machine learning", "https://deeplearningwithpython.io/chapters/chapter05_fundamentals-of-ml"),
    6: ("The universal workflow of machine learning", "https://deeplearningwithpython.io/chapters/chapter06_universal-workflow-of-ml"),
    7: ("A deep dive on Keras", "https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras"),
    8: ("Image classification", "https://deeplearningwithpython.io/chapters/chapter08_image-classification"),
    9: ("ConvNet architecture patterns", "https://deeplearningwithpython.io/chapters/chapter09_convnet-architecture-patterns"),
    10: ("Interpreting what ConvNets learn", "https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn"),
    11: ("Image segmentation", "https://deeplearningwithpython.io/chapters/chapter11_image-segmentation"),
    12: ("Object detection", "https://deeplearningwithpython.io/chapters/chapter12_object-detection"),
    13: ("Timeseries forecasting", "https://deeplearningwithpython.io/chapters/chapter13_timeseries-forecasting"),
    14: ("Text classification", "https://deeplearningwithpython.io/chapters/chapter14_text-classification"),
    15: ("Language models and the Transformer", "https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer"),
    16: ("Text generation", "https://deeplearningwithpython.io/chapters/chapter16_text-generation"),
    17: ("Image generation", "https://deeplearningwithpython.io/chapters/chapter17_image-generation"),
    18: ("Best practices for the real world", "https://deeplearningwithpython.io/chapters/chapter18_best-practices-for-the-real-world"),
    19: ("The future of AI", "https://deeplearningwithpython.io/chapters/chapter19_future_of_ai"),
    20: ("Conclusions", "https://deeplearningwithpython.io/chapters/chapter20_conclusion"),
}

file_mappings = {
    "abstractive_summarization_with_bart.py": [15, 16],
    "active_learning_review_classification.py": [14],
    "addition_rnn.py": [15, 16],
    "bidirectional_lstm_imdb.py": [14],
    "data_parallel_training_with_keras_hub.py": [7, 18],
    "fnet_classification_with_keras_hub.py": [14, 15],
    "lstm_seq2seq.py": [15, 16],
    "masked_language_modeling.py": [15],
    "multi_label_classification.py": [14],
    "multimodal_entailment.py": [9, 14],
    "multiple_choice_task_with_transfer_learning.py": [14],
    "ner_transformers.py": [14, 15],
    "neural_machine_translation_with_keras_hub.py": [15, 16],
    "neural_machine_translation_with_transformer.py": [15, 16],
    "parameter_efficient_finetuning_of_gpt2_with_lora.py": [16, 18],
    "pretrained_word_embeddings.py": [14],
    "semantic_similarity_with_bert.py": [14, 15],
    "semantic_similarity_with_keras_hub.py": [14, 15],
    "sentence_embeddings_with_sbert.py": [14, 15],
    "text_classification_from_scratch.py": [14],
    "text_classification_with_switch_transformer.py": [14, 15],
    "text_classification_with_transformer.py": [14, 15],
    "text_extraction_with_bert.py": [15],
    "tweet-classification-using-tfdf.py": [14],
}

def get_links_md(chapter_ids):
    links = []
    for cid in chapter_ids:
        title, url = chapters[cid]
        links.append(f"- [Chapter {cid}: {title}]({url})")
    return "\n".join(links)

def update_md(filename, chapter_ids):
    md_filename = filename.replace(".py", ".md")
    file_path = os.path.join(md_base_path, md_filename)
    if not os.path.exists(file_path):
        print(f"Skipping {md_filename}: File not found")
        return

    with open(file_path, "r") as f:
        content = f.read()

    if "Relevant Chapters" in content:
        print(f"Skipping {md_filename}: Already has 'Relevant Chapters'")
        return

    links_str = get_links_md(chapter_ids)
    append_block = f'\n\n## Relevant Chapters\n{links_str}\n'
    
    with open(file_path, "a") as f:
        f.write(append_block)
    print(f"Updated {md_filename}")

def update_ipynb(filename, chapter_ids):
    ipynb_filename = filename.replace(".py", ".ipynb")
    file_path = os.path.join(ipynb_base_path, ipynb_filename)
    if not os.path.exists(file_path):
        print(f"Skipping {ipynb_filename}: File not found")
        return

    with open(file_path, "r") as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {ipynb_filename}: Invalid JSON")
            return

    # Check if already added
    if notebook["cells"] and notebook["cells"][-1]["source"]:
        last_source = "".join(notebook["cells"][-1]["source"])
        if "Relevant Chapters" in last_source:
             print(f"Skipping {ipynb_filename}: Already has 'Relevant Chapters'")
             return

    links_str = get_links_md(chapter_ids)
    new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Relevant Chapters\n",
            links_str
        ]
    }
    
    notebook["cells"].append(new_cell)
    
    with open(file_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Updated {ipynb_filename}")

for filename, chapter_ids in file_mappings.items():
    update_md(filename, chapter_ids)
    update_ipynb(filename, chapter_ids)
