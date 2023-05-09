SAVING_MASTER = {
    "path": "model_saving_apis/",
    "title": "Saving & serialization",
    "toc": True,
    "children": [
        {
            "path": "model_saving_and_loading",
            "title": "Whole model saving & loading",
            "generate": [
                "tensorflow.keras.Model.save",
                "tensorflow.keras.saving.save_model",
                "tensorflow.keras.saving.load_model",
            ],
        },
        {
            "path": "weights_saving_and_loading",
            "title": "Weights-only saving & loading",
            "generate": [
                "tensorflow.keras.Model.get_weights",
                "tensorflow.keras.Model.set_weights",
                "tensorflow.keras.Model.save_weights",
                "tensorflow.keras.Model.load_weights",
            ],
        },
        {
            "path": "model_config_serialization",
            "title": "Model config serialization",
            "generate": [
                "tensorflow.keras.Model.get_config",
                "tensorflow.keras.Model.from_config",
                "tensorflow.keras.models.clone_model",  # TODO: move somewhere else?
            ],
        },
        {
            "path": "export",
            "title": "Model export for inference",
            "generate": [
                "tensorflow.keras.export.ExportArchive",
            ],
        },
        {
            "path": "serialization_utils",
            "title": "Serialization utilities",
            "generate": [
                "tensorflow.keras.utils.serialize_keras_object",  # TODO: move to saving
                "tensorflow.keras.utils.deserialize_keras_object",  # TODO: move to saving
                "tensorflow.keras.saving.custom_object_scope",
                "tensorflow.keras.saving.get_custom_objects",
                "tensorflow.keras.saving.register_keras_serializable",
            ],
        },
    ],
}
