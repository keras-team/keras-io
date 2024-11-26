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
    try:
        return f"[{metadata['official_name']}]({metadata['path']})"
    except KeyError:
        return "Unknown"


def is_base_class(symbol):
    return symbol in (
        keras_hub.models.Backbone,
        keras_hub.models.Tokenizer,
        keras_hub.models.Preprocessor,
        keras_hub.models.Task,
    )


def render_all_presets(symbols):
    """Renders the markdown table for backbone presets as a string."""

    table = TABLE_HEADER

    # Backbones has alias, which duplicates some presets.
    # Use a set to keep them unique.
    added_presets = set()
    # Bakcbone presets
    for name, symbol in symbols:
        if is_base_class(symbol) or "Backbone" not in name:
            continue
        presets = symbol.presets
        # Only keep the ones with pretrained weights for KerasCV Backbones.
        for preset in presets:
            if preset in added_presets:
                continue
            else:
                added_presets.add(preset)
            metadata = presets[preset]["metadata"]
            url = presets[preset]["kaggle_handle"]
            url = url.replace("kaggle://", "https://www.kaggle.com/models/")
            table += (
                f"[{preset}]({url}) | "
                f"{format_path(metadata)} | "
                f"{format_param_count(metadata)} | "
                f"{metadata['description']}"
            )
            table += "\n"
    return table


def render_table(symbol):
    if keras_hub is None:
        return ""

    table = TABLE_HEADER_PER_MODEL
    if is_base_class(symbol) or len(symbol.presets) == 0:
        return None
    for preset in symbol.presets:
        metadata = symbol.presets[preset]["metadata"]
        url = symbol.presets[preset]["kaggle_handle"]
        url = url.replace("kaggle://", "https://www.kaggle.com/models/")
        table += (
            f"[{preset}]({url}) | "
            f"{format_param_count(metadata)} | "
            f"{metadata['description']} \n"
        )
    return table


def render_tags(template):
    """Replaces all custom KerasHub tags with rendered content."""
    if keras_hub is None:
        return template

    symbols = keras_hub.models.__dict__.items()
    if "{{presets_table}}" in template:
        template = template.replace(
            "{{presets_table}}", render_all_presets(symbols)
        )
    return template
