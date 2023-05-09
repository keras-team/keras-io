UTILS_MASTER = {
    "path": "utils/",
    "title": "Utilities",
    "toc": True,
    "children": [
        {
            "path": "model_plotting_utils",
            "title": "Model plotting utilities",
            "generate": [
                "tensorflow.keras.utils.plot_model",
                "tensorflow.keras.utils.model_to_dot",
            ],
        },
        {
            "path": "feature_space",
            "title": "Structured data preprocessing utilities",
            "generate": [
                "tensorflow.keras.utils.FeatureSpace",
            ],
        },
        {
            "path": "python_utils",
            "title": "Python & NumPy utilities",
            "generate": [
                "tensorflow.keras.utils.set_random_seed",
                "tensorflow.keras.utils.split_dataset",
                "tensorflow.keras.utils.get_file",
                "tensorflow.keras.utils.Progbar",
                "tensorflow.keras.utils.Sequence",
                "tensorflow.keras.utils.to_categorical",
                "tensorflow.keras.utils.to_ordinal",
                "tensorflow.keras.utils.normalize",
            ],
        },
        {
            "path": "backend_utils",
            "title": "Backend utilities",
            "generate": [
                "tensorflow.keras.backend.clear_session",
                "tensorflow.keras.backend.floatx",
                "tensorflow.keras.backend.set_floatx",
                "tensorflow.keras.backend.image_data_format",
                "tensorflow.keras.backend.set_image_data_format",
                "tensorflow.keras.backend.epsilon",
                "tensorflow.keras.backend.set_epsilon",
                "tensorflow.keras.backend.is_keras_tensor",
                "tensorflow.keras.backend.get_uid",
                "tensorflow.keras.backend.rnn",
            ],
        },
    ],
}
