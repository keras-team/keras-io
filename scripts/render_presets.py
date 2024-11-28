"""Custom rendering code for keras_hub presets.

The model metadata is pulled from the library, each preset has a
metadata dictionary as follows:

{
    'description': Description of the model,
    'params': Parameter count of the model,
    'official_name': Name of the model,
    'path': Relative path of the model on keras.io,
}
"""

from hub_master import MODELS_MASTER

try:
    import keras_hub
except Exception as e:
    print(f"Could not import KerasHub. Exception: {e}")
    keras_hub = None


TABLE_HEADER = (
    "Preset | Model API | Parameters | Description\n"
    "-------|-----------|------------|------------\n"
)

TABLE_HEADER_PER_MODEL = (
    "Preset | Parameters | Description\n"
    "-------|------------|------------\n"
)


def format_param_count(metadata):
    """Format a parameter count for the table."""
    try:
        count = metadata["params"]
    except KeyError:
        return "Unknown"
    if count >= 1e9:
        return f"{(count / 1e9):.2f}B"
    if count >= 1e6:
        return f"{(count / 1e6):.2f}M"
    if count >= 1e3:
        return f"{(count / 1e3):.2f}K"
    return f"{count}"


def format_path(metadata):
    """Returns Path for the given preset"""
    for child in MODELS_MASTER["children"]:
        path = child["path"].strip("/")
        if metadata["path"] == path:
            text = child["title"]
            link = f"/keras_hub/api/models/{path}"
            return f"[{text}]({link})"
    return "-"


def format_preset_link(preset, handle):
    url = handle.replace("kaggle://", "https://www.kaggle.com/models/")
    return f"[{preset}]({url})"


def is_base_class(symbol):
    return symbol in (
        keras_hub.models.Backbone,
        keras_hub.models.Tokenizer,
        keras_hub.models.Preprocessor,
        keras_hub.models.Task,
    )


def sort_presets(presets):
    # Sort by path and then by parameter count.
    return sorted(
        presets.keys(),
        key=lambda x: (
            presets[x]["metadata"]["path"],
            presets[x]["metadata"]["params"],
        )
    )


def render_row(preset, data, add_doc_link=False):
    """Renders a row for a preset in a markdown table."""
    metadata = data["metadata"]
    url = data["kaggle_handle"]
    url = url.replace("kaggle://", "https://www.kaggle.com/models/")
    cols = []
    cols.append(format_preset_link(preset, data["kaggle_handle"]))
    if add_doc_link:
        cols.append(format_path(metadata))
    cols.append(format_param_count(metadata))
    cols.append(metadata["description"])
    return " | ".join(cols) + "\n"


def render_all_presets():
    """Renders the markdown table for backbone presets as a string."""
    table = TABLE_HEADER
    symbol = keras_hub.models.Backbone
    for preset in sort_presets(symbol.presets):
        data = symbol.presets[preset]
        table += render_row(preset, data, add_doc_link=True)
    return table


def render_table(symbol):
    if keras_hub is None:
        return ""

    table = TABLE_HEADER_PER_MODEL
    if is_base_class(symbol) or len(symbol.presets) == 0:
        return None
    for preset in sort_presets(symbol.presets):
        data = symbol.presets[preset]
        table += render_row(preset, data)
    return table


def render_tags(template):
    """Replaces all custom KerasHub tags with rendered content."""
    if keras_hub is None:
        return template

    if "{{presets_table}}" in template:
        template = template.replace("{{presets_table}}", render_all_presets())
    return template
