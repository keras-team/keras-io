MODELS_MASTER = {
    "path": "models/",
    "title": "Models API",  # TODO
    "toc": True,
    "children": [
        {
            "path": "model",
            "title": "The Model class",  # TODO
            "generate": [
                "tensorflow.keras.Model",
                "tensorflow.keras.Model.summary",
                "tensorflow.keras.Model.get_layer",
            ],
        },
        {
            "path": "sequential",
            "title": "The Sequential class",  # TODO
            "generate": [
                "tensorflow.keras.Sequential",
                "tensorflow.keras.Sequential.add",
                "tensorflow.keras.Sequential.pop",
            ],
        },
        {
            "path": "model_training_apis",
            "title": "Model training APIs",  # TODO
            "generate": [
                "tensorflow.keras.Model.compile",
                "tensorflow.keras.Model.fit",
                "tensorflow.keras.Model.evaluate",
                "tensorflow.keras.Model.predict",
                "tensorflow.keras.Model.train_on_batch",
                "tensorflow.keras.Model.test_on_batch",
                "tensorflow.keras.Model.predict_on_batch",
                "tensorflow.keras.Model.run_eagerly",
            ],
        },
        {
            "path": "model_saving_apis",
            "title": "Model saving & serialization APIs",  # TODO
            "generate": [
                "tensorflow.keras.Model.save",
                "tensorflow.keras.models.save_model",
                "tensorflow.keras.models.load_model",
                "tensorflow.keras.Model.get_weights",
                "tensorflow.keras.Model.set_weights",
                "tensorflow.keras.Model.save_weights",
                "tensorflow.keras.Model.load_weights",
                "tensorflow.keras.Model.get_config",
                "tensorflow.keras.Model.from_config",
                "tensorflow.keras.models.model_from_config",
                "tensorflow.keras.Model.to_json",
                "tensorflow.keras.models.model_from_json",
                "tensorflow.keras.models.clone_model",
            ],
        },
    ],
}
