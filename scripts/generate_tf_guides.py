from pathlib import Path
import copy
import json
import random
import string
import re

CONFIG = [
    {
        "title": "The Functional API",
        "source_name": "functional_api",
        "target_name": "functional",
    },
    {
        "title": "Training & evaluation with the built-in methods",
        "source_name": "training_with_built_in_methods",
        "target_name": "train_and_evaluate",
    },
    {
        "title": "Making new Layers & Models via subclassing",
        "source_name": "making_new_layers_and_models_via_subclassing",
        "target_name": "custom_layers_and_models",
    },
    {
        "title": "Recurrent Neural Networks (RNN) with Keras",
        "source_name": "working_with_rnns",
        "target_name": "rnn",
    },
    {
        "title": "Masking and padding with Keras",
        "source_name": "understanding_masking_and_padding",
        "target_name": "masking_and_padding",
    },
    {
        "title": "Save and load Keras models",
        "source_name": "serialization_and_saving",
        "target_name": "save_and_serialize",
    },
    {
        "title": "Writing your own callbacks",
        "source_name": "writing_your_own_callbacks",
        "target_name": "custom_callback",
    },
    {
        "title": "Writing a training loop from scratch",
        "source_name": "writing_a_training_loop_from_scratch",
        "target_name": "writing_a_training_loop_from_scratch",
    },
    {
        "title": "Transfer learning & fine-tuning",
        "source_name": "transfer_learning",
        "target_name": "transfer_learning",
    },
    {
        "title": "The Sequential model",
        "source_name": "sequential_model",
        "target_name": "sequential_model",
    },
    {
        "title": "Customizing what happens in `fit()`",
        "source_name": "customizing_what_happens_in_fit",
        "target_name": "customizing_what_happens_in_fit",
    },
]


TF_BUTTONS_TEMPLATE = {
    "cell_type": "markdown",
    "metadata": {"colab_type": "text",},
    "source": [
        '<table class="tfo-notebook-buttons" align="left">\n',
        "  <td>\n",
        '    <a target="_blank" href="https://www.tensorflow.org/guide/keras/TARGET_NAME"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>\n',
        "  </td>\n",
        "  <td>\n",
        '    <a target="_blank" href="https://colab.research.google.com/github/keras-team/keras-io/blob/master/tf/TARGET_NAME.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>\n',
        "  </td>\n",
        "  <td>\n",
        '    <a target="_blank" href="https://github.com/keras-team/keras-io/blob/master/guides/SOURCE_NAME.py"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>\n',
        "  </td>\n",
        "  <td>\n",
        '    <a href="https://storage.googleapis.com/tensorflow_docs/keras-io/tf/TARGET_NAME.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>\n',
        "  </td>\n",
        "</table>",
    ],
}


TF_IPYNB_CELLS_TEMPLATE = [
    {
        "cell_type": "markdown",
        "metadata": {"colab_type": "text",},
        "source": ["##### Copyright 2020 The TensorFlow Authors."],
    },
    {
        "cell_type": "code",
        "execution_count": 0,
        "metadata": {"cellView": "form", "colab": {}, "colab_type": "code",},
        "outputs": [],
        "source": [
            '#@title Licensed under the Apache License, Version 2.0 (the "License");\n',
            "# you may not use this file except in compliance with the License.\n",
            "# You may obtain a copy of the License at\n",
            "#\n",
            "# https://www.apache.org/licenses/LICENSE-2.0\n",
            "#\n",
            "# Unless required by applicable law or agreed to in writing, software\n",
            '# distributed under the License is distributed on an "AS IS" BASIS,\n',
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
            "# See the License for the specific language governing permissions and\n",
            "# limitations under the License.",
        ],
    },
    # Then: title
    # Then: buttons
]

TF_IPYNB_BASE = {
    "metadata": {
        "colab": {
            "collapsed_sections": [],
            "name": "",  # FILL ME
            "private_outputs": True,
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def generate_single_tf_guide(source_dir, target_dir, title, source_name, target_name):
    nb = (Path(source_dir) / source_name).with_suffix(".ipynb")
    original_ipynb = json.loads(nb.read_text())

    # Skip first title cell
    cells = original_ipynb["cells"][1:]
    # Strip Keras tags
    for cell in cells:
        if cell["cell_type"] == "markdown":
            new_lines = []
            lines = cell["source"]
            num_lines = len(lines)
            for i in range(num_lines - 1):
                if lines[i].startswith('<div class="k-default-codeblock">') and lines[
                    i + 1
                ].startswith("```"):
                    continue
                elif lines[i].startswith("</div>") and lines[i - 1].startswith("```"):
                    continue
                else:
                    new_lines.append(lines[i])
            if len(lines) >= 2 and not (
                lines[-1].startswith("</div>") and lines[-2].startswith("```")
            ):
                new_lines.append(lines[-1])
            if len(lines) < 2:
                new_lines.append(lines[-1])
            cell["source"] = new_lines
        elif cell["cell_type"] == "code":
          lines = cell["source"]
          if not lines[0].strip():
            lines = lines[1:]
          if not lines[-1].strip():
            lines = lines[:-1]
          cell["source"] = lines

    # Add header cells
    header_cells = copy.deepcopy(TF_IPYNB_CELLS_TEMPLATE)
    # Add title cell
    header_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {"colab_type": "text"},
            "source": ["# " + title],
        }
    )
    buttons = copy.deepcopy(TF_BUTTONS_TEMPLATE)
    for i in range(len(buttons["source"])):
        buttons["source"][i] = buttons["source"][i].replace("TARGET_NAME", target_name)
        buttons["source"][i] = buttons["source"][i].replace("SOURCE_NAME", source_name)
    header_cells.append(buttons)
    cells = header_cells + cells
    for cell in cells:
        cell['metadata']['id'] = random_id()

    notebook = {}
    for key in TF_IPYNB_BASE.keys():
        notebook[key] = TF_IPYNB_BASE[key]
    notebook["metadata"]["colab"]["name"] = target_name + '.ipynb'
    notebook["cells"] = cells

    f = open(Path(target_dir) / (target_name + ".ipynb"), "w")
    json_st = json.dumps(notebook, indent=1, sort_keys=True)

    # Apply link conversion
    json_st = json_st.replace(
        "(/api/callbacks/",
        "(https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/",
    )
    json_st = json_st.replace(
        "keras.io/api/layers/recurrent_layers/rnn/",
        "https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN/",
    )
    json_st = json_st.replace(
        "https://keras.io/api/layers/recurrent_layers/gru/",
        "https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU/",
    )
    json_st = json_st.replace(
        "https://keras.io/api/layers/recurrent_layers/lstm/",
        "https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM/",
    )
    json_st = json_st.replace(
        "https://keras.io/api/layers/recurrent_layers/bidirectional/",
        "https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional/",
    )
    json_st = json_st.replace(
        "https://keras.io/api/callbacks/",
        "https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/",
    )
    for entry in CONFIG:
        src = entry["source_name"]
        dst = entry["target_name"]
        json_st = re.sub(
            r"(?is)]\((\s*)/guides/" + src,
            "](https://www.tensorflow.org/guide/keras/" + dst,
            json_st,
        )
        json_st = re.sub(
            r"(?is)(\s+)/guides/" + src,
            "https://www.tensorflow.org/guide/keras/" + dst,
            json_st,
        )
    f.write(json_st)
    f.close()


def random_id():
    length = 12
    letters = string.ascii_lowercase + string.ascii_uppercase + '0123456789'
    return ''.join(random.choice(letters) for i in range(length))


def generate_tf_guides():
    random.seed(1337)
    for entry in CONFIG:
        generate_single_tf_guide(
            source_dir="../guides/ipynb/",
            target_dir="../tf/",
            title=entry["title"],
            source_name=entry["source_name"],
            target_name=entry["target_name"],
        )
