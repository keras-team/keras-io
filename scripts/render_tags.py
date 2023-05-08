"""Custom rendering code for the /api/{keras_nlp|keras_cv}/models page.

The model metadata is pulled from the library, each preset has a
metadata dictionary as follows:

{
    'description': Description of the model,
    'params': Parameter count of the model,
    'official_name': Name of the model,
    'path': Relative path of the model on keras.io,
}
"""

import keras_cv
import inspect


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


def render_backbone_table(symbols):
    """Renders the markdown table for backbone presets as a string."""

    table = TABLE_HEADER

    # Backbones has alias, which duplicates some presets.
    # Use a set to keep them unique.
    added_presets = set()
    # Bakcbone presets
    for name, symbol in symbols:
        if "Backbone" not in name:
            continue
        presets = symbol.presets
        # Only keep the ones with pretrained weights for KerasCV Backbones.
        if issubclass(symbol, keras_cv.models.backbones.backbone.Backbone):
            presets = symbol.presets_with_weights
        for preset in presets:
            if preset in added_presets:
                continue
            else:
                added_presets.add(preset)
            metadata = presets[preset]["metadata"]
            # KerasCV backbones docs' URL has a "backbones/" path.
            if (
                issubclass(symbol, keras_cv.models.backbones.backbone.Backbone)
                and "path" in metadata
            ):
                metadata["path"] = "backbones/" + metadata["path"]
            table += (
                f"{preset} | "
                f"{format_path(metadata)} | "
                f"{format_param_count(metadata)} | "
                f"{metadata['description']} \n"
            )
    return table


def render_classifier_table(symbols):
    """Renders the markdown table for classifier presets as a string."""

    table = TABLE_HEADER

    # Classifier presets
    for name, symbol in symbols:
        if "Classifier" not in name:
            continue
        for preset in symbol.presets:
            if preset not in symbol.backbone_cls.presets:
                metadata = symbol.presets[preset]["metadata"]
                table += (
                    f"{preset} | "
                    f"{format_path(metadata)} | "
                    f"{format_param_count(metadata)} | "
                    f"{metadata['description']} \n"
                )
    return table


def render_task_table(symbols):
    """Renders the markdown table for Task presets as a string."""
    table = TABLE_HEADER

    for name, symbol in symbols:
        if not inspect.isclass(symbol):
            continue
        if not issubclass(symbol, keras_cv.models.task.Task):
            continue
        for preset in symbol.presets:
            # Do not print all backbone presets for a task
            if preset in keras_cv.models.backbones.backbone_presets.backbone_presets:
                continue
            # Only render the ones with pretrained_weights for KerasCV.
            metadata = symbol.presets_with_weights[preset]["metadata"]
            # KerasCV tasks docs' URL has a "tasks/" path.
            metadata["path"] = "tasks/" + metadata["path"]
            table += (
                f"{preset} | "
                f"{format_path(metadata)} | "
                f"{format_param_count(metadata)} | "
                f"{metadata['description']} \n"
            )
    return table


def render_table(symbol):
    table = TABLE_HEADER_PER_MODEL
    if len(symbol.presets) == 0:
        return None
    for preset in symbol.presets:
        # Do not print all backbone presets for a task
        if (
            issubclass(symbol, keras_cv.models.task.Task)
            and preset in keras_cv.models.backbones.backbone_presets.backbone_presets
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
    """Replaces all custom KerasNLP/KerasCV tags with rendered content."""
    symbols = lib.models.__dict__.items()
    if "{{backbone_presets_table}}" in template:
        template = template.replace(
            "{{backbone_presets_table}}", render_backbone_table(symbols)
        )
    if "{{classifier_presets_table}}" in template:
        template = template.replace(
            "{{classifier_presets_table}}", render_classifier_table(symbols)
        )
    if "{{task_presets_table}}" in template:
        template = template.replace(
            "{{task_presets_table}}", render_task_table(symbols)
        )
    return template
