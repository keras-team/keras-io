import json
import os

chapter_links = {
    "Chapter 7": "[Chapter 7: A deep dive on Keras](https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras)",
    "Chapter 14": "[Chapter 14: Text classification](https://deeplearningwithpython.io/chapters/chapter14_text-classification)",
    "Chapter 15": "[Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)",
}

updates = {
    "gat_node_classification": ["Chapter 7", "Chapter 15"],
    "mpnn-molecular-graphs": ["Chapter 7", "Chapter 15"],
    "gnn_citations": ["Chapter 7"],
    "node2vec_movielens": ["Chapter 14"],
}

base_path = "/usr/local/google/home/divyasreepat/Desktop/keras-io/examples/graph"

def create_cell(chapters):
    source = [
        "## Relevant Chapters from Deep Learning with Python\n"
    ]
    for chap in chapters:
        source.append(f"- {chapter_links[chap]}\n")
    
    return {
        "cell_type": "markdown",
        "metadata": {
            "colab_type": "text"
        },
        "source": source
    }

def update_ipynb(file_name, chapters):
    file_path = os.path.join(base_path, "ipynb", f"{file_name}.ipynb")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Check checks
    if len(data["cells"]) > 0:
        last_cell = data["cells"][-1]
        last_source = last_cell.get("source", [])
        if isinstance(last_source, list) and len(last_source) > 0:
            if "Relevant Chapters" in last_source[0]:
                print(f"Skipping {file_name} IPYNB: already updated.")
                return

    new_cell = create_cell(chapters)
    data["cells"].append(new_cell)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"Updated {file_name} IPYNB")

def update_md(file_name, chapters):
    file_path = os.path.join(base_path, "md", f"{file_name}.md")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        content = f.read()
        
    if "Relevant Chapters from Deep Learning with Python" in content:
        print(f"Skipping {file_name} MD: already updated.")
        return

    append_text = "\n\n## Relevant Chapters from Deep Learning with Python\n"
    for chap in chapters:
        append_text += f"- {chapter_links[chap]}\n"
    
    with open(file_path, "a") as f:
        f.write(append_text)
    print(f"Updated {file_name} MD")

if __name__ == "__main__":
    for name, chapters in updates.items():
        update_ipynb(name, chapters)
        update_md(name, chapters)
