"""Keras Generate Presets Table Implementation

Presets table contains details about the model information such as Name, Preset ID, parameters, Description.

This code is used to render preset table at https://keras.io/api/keras_nlp/models/.

It uses metadata present in keras-nlp/models/xx/xx_presets.py.

The model metadata is present as a dict form as
```py
metdata{
    'description': tuple of string
    'params': parameter count of model
    'official_name': Name of model
    'path': Relative path of model at keras.io
}
```

NOTE It generated presets table for those models which has path in it.
"""

def render_keras_nlp_tags(template):

    def param_count(count:int):
        if count>=1e9:
            return f"{int(count / 1e9)}B"
        if count >= 1e6:
            return f"{int(count / 1e6)}M"
        if count >= 1e3:
            return f"{int(count / 1e3)}K"
        return f"{count}"

    if "{{backbone_presets_table}}" in template:
        # Import KerasNLP
        import keras_nlp

        # Table Header
        table = "Preset ID | Model | Parameters | Description  \n"

        # Column alignment
        table += "-------|--------|-------|------\n"

        # Classifier presets
        for name, symbol in keras_nlp.models.__dict__.items():
            if "Classifier" not in name:
                continue
            for preset in symbol.presets:
                if preset in symbol.backbone_cls.presets:
                    # Generating table for only those which has path in metadata
                    if 'path' in symbol.presets[preset]['metadata']:
                        table += (f"{preset} | [{symbol.presets[preset]['metadata']['official_name']}]({symbol.presets[preset]['metadata']['path']}) | {param_count(symbol.presets[preset]['metadata']['params'])} | {symbol.presets[preset]['metadata']['description']}  \n")

        template = template.replace(
            "{{backbone_presets_table}}", table
        )
    if "{{classifier_presets_table}}" in template:
        # Import KerasNLP and do some stuff.
        from keras_nlp.models.bert import bert_presets

        # Table Header
        table = "Preset ID | Model | Parameters | Description  \n"

        # Column alignment
        table += "-------|--------|-------|------\n"

        # Classifier presets
        for name, symbol in keras_nlp.models.__dict__.items():
            if "Classifier" not in name:
                continue
            for preset in symbol.presets:
                if not preset in symbol.backbone_cls.presets:
                    table += (f"{preset} | [{symbol.presets[preset]['metadata']['official_name']}]({symbol.presets[preset]['metadata']['path']}) | {param_count(symbol.presets[preset]['metadata']['params'])} | {symbol.presets[preset]['metadata']['description']}  \n")

        template = template.replace(
            "{{classifier_presets_table}}", table
        )
    return template