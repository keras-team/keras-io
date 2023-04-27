"""Custom rendering code for the /api/keras_nlp/models page.

The model metadata is pulled from the keras_nlp library, each preset has a
metadata dictionary as follows:

{
    'description': Description of the model,
    'params': Parameter count of the model,
    'official_name': Name of the model,
    'path': Relative path of the model on keras.io,
}
"""


TABLE_HEADER = (
    "Preset ID | Model | Parameters | Description\n"
    "----------|-------|------------|------------\n"
)


def format_param_count(count):
    """Format a parameter count for the table."""
    if count >= 1e9:
        return f"{int(count / 1e9)}B"
    if count >= 1e6:
        return f"{int(count / 1e6)}M"
    if count >= 1e3:
        return f"{int(count / 1e3)}K"
    return f"{count}"


def format_path(metadata):
    """Returns Path for the given preset"""
    return f"[{metadata['official_name']}]({metadata['path']})"


def render_backbone_table(template):
    """Renders the markdown table for backbone presets as a string."""
    # Import KerasNLP
    import keras_nlp

    table = TABLE_HEADER

    # Bakcbone presets
    for name, symbol in keras_nlp.models.__dict__.items():
        if "Backbone" not in name:
            continue
        for preset in symbol.presets:
            metadata = symbol.presets[preset]["metadata"]
            table += (
                f"{preset} | "
                f"{format_path(metadata)} | "
                f"{format_param_count(metadata['params'])} | "
                f"{metadata['description']} \n"
            )
    return template.replace("{{backbone_presets_table}}", table)


def render_classifier_table(template):
    """Renders the markdown table for classifier presets as a string."""
    import keras_nlp

    table = TABLE_HEADER

    # Classifier presets
    for name, symbol in keras_nlp.models.__dict__.items():
        if "Classifier" not in name:
            continue
        for preset in symbol.presets:
            if preset not in symbol.backbone_cls.presets:
                metadata = symbol.presets[preset]["metadata"]
                table += (
                    f"{preset} | "
                    f"{format_path(metadata)} | "
                    f"{format_param_count(metadata['params'])} | "
                    f"{metadata['description']} \n"
                )
    return template.replace("{{classifier_presets_table}}", table)


def render_keras_nlp_tags(template):
    """Replaces all custom KerasNLP tags with rendered content."""
    if "{{backbone_presets_table}}" in template:
        template = render_backbone_table(template)
    if "{{classifier_presets_table}}" in template:
        template = render_classifier_table(template)
    return template
