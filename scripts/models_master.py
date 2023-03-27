from saving_master import SAVING_MASTER

MODELS_MASTER = {
    "path": "models/",
    "title": "Models API",
    "toc": True,
    "children": [
        {
            "path": "model",
            "title": "The Model class",
            "generate": [
                "tensorflow.keras.Model",
                "tensorflow.keras.Model.summary",
                "tensorflow.keras.Model.get_layer",
            ],
        },
        {
            "path": "sequential",
            "title": "The Sequential class",
            "generate": [
                "tensorflow.keras.Sequential",
                "tensorflow.keras.Sequential.add",
                "tensorflow.keras.Sequential.pop",
            ],
        },
        {
            "path": "model_training_apis",
            "title": "Model training APIs",
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
        SAVING_MASTER,
    ],
}
