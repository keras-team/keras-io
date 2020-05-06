
UTILS_MASTER = {
    'path': 'utils/',
    'title': 'Utilities',
    'toc': True,
    'children': [
        {
            'path': 'model_plotting_utils',
            'title': 'Model plotting utilities',
            'generate': [
                'tensorflow.keras.utils.plot_model',
                'tensorflow.keras.utils.model_to_dot',
            ]
        },
        {
            'path': 'serialization_utils',
            'title': 'Serialization utilities',
            'generate': [
                'tensorflow.keras.utils.custom_object_scope',
                'tensorflow.keras.utils.get_custom_objects',
                'tensorflow.keras.utils.register_keras_serializable',
                'tensorflow.keras.utils.serialize_keras_object',
                'tensorflow.keras.utils.deserialize_keras_object',
            ]
        },
        {
            'path': 'python_utils',
            'title': 'Python & NumPy utilities',
            'generate': [
                'tensorflow.keras.utils.to_categorical',
                'tensorflow.keras.utils.normalize',
                'tensorflow.keras.utils.get_file',
                'tensorflow.keras.utils.Progbar',
                'tensorflow.keras.utils.Sequence',
            ]
        },
    ]
}
