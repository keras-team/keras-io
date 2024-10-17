"""Custom rendering code for the /api/{keras_hub|keras_cv}/models page.

The model metadata is pulled from the library, each preset has a
metadata dictionary as follows:

{
    'description': Description of the model,
    'params': Parameter count of the model,
    'official_name': Name of the model,
    'path': Relative path of the model on keras.io,
}
"""

import inspect

try:
    import keras_cv
except Exception as e:
    print(f"Could not import Keras CV. Exception: {e}")
    keras_cv = None


TABLE_HEADER = (
    "Preset name | Model | Parameters | Description\n"
    "------------|-------|------------|------------\n"
)

TABLE_HEADER_PER_MODEL = (
    "Preset name | Parameters | Description\n"
    "------------|------------|------------\n"
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
    import keras_hub

    return symbol in (
        keras_hub.models.Backbone,
        keras_hub.models.Tokenizer,
        keras_hub.models.Preprocessor,
        keras_hub.models.Task,
        keras_hub.models.Classifier,
        keras_hub.models.CausalLM,
        keras_hub.models.MaskedLM,
        keras_hub.models.Seq2SeqLM,
    )


def render_backbone_table(symbols):
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
            table += (
                f"{preset} | "
                f"{format_path(metadata)} | "
                f"{format_param_count(metadata)} | "
                f"{metadata['description']}"
            )
            if "model_card" in metadata:
                table += f" [Model Card]({metadata['model_card']})"
            table += "\n"
    return table


def render_table(symbol):
    table = TABLE_HEADER_PER_MODEL
    if is_base_class(symbol) or len(symbol.presets) == 0:
        return None
    for preset in symbol.presets:
        # Do not print all backbone presets for a task
        if (
            issubclass(symbol, keras_cv.models.Task)
            and preset
            in keras_cv.src.models.backbones.backbone_presets.backbone_presets
        ):
            continue

        metadata = symbol.presets[preset]["metadata"]
        table += (
            f"{preset} | "
            f"{format_param_count(metadata)} | "
            f"{metadata['description']} \n"
        )
    return table


def render_tags(template, lib):
    """Replaces all custom KerasHub/KerasCV tags with rendered content."""
    symbols = lib.models.__dict__.items()
    if "{{backbone_presets_table}}" in template:
        template = template.replace(
            "{{backbone_presets_table}}", render_backbone_table(symbols)
        )
    return template
