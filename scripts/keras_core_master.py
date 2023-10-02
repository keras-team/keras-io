# The line `from keras_core_api_master import KERAS_CORE_API_MASTER` is importing the variable
# `KERAS_CORE_API_MASTER` from the module `keras_core_api_master`.
from keras_core_api_master import KERAS_CORE_API_MASTER

# The code is defining a dictionary named `KERAS_CORE_MASTER`. This dictionary contains information
# about the Keras Core module, such as its path, title, and a list of children. The `path` key
# specifies the path to the Keras Core module, the `title` key provides a title for the module, and
# the `children` key contains a list of sub-modules or sections within the Keras Core module.
KERAS_CORE_MASTER = {
    "path": "keras_core/", 
    "title": "Keras Core: Keras for TensorFlow, JAX, and PyTorch",
    "children": [
        {
            "path": "guides/",
            "title": "Keras Core developer guides",
            "toc": True,
            "children": [
                {
                    "path": "getting_started_with_keras_core",
                    "title": "Getting started with Keras Core",
                },
                {
                    "path": "functional_api",
                    "title": "The Functional API",
                },
                {
                    "path": "sequential_model",
                    "title": "The Sequential model",
                },
                {
                    "path": "making_new_layers_and_models_via_subclassing",
                    "title": "Making new layers & models via subclassing",
                },
                {
                    "path": "training_with_built_in_methods",
                    "title": "Training & evaluation with the built-in methods",
                },
                # {
                #     "path": "serialization_and_saving",
                #     "title": "Serialization & saving",
                # },
                {
                    "path": "writing_your_own_callbacks",
                    "title": "Writing your own callbacks",
                },
                {
                    "path": "transfer_learning",
                    "title": "Transfer learning",
                },
                {
                    "path": "understanding_masking_and_padding",
                    "title": "Understanding masking & padding",
                },
                # Backend-specific guides
                {
                    "path": "custom_train_step_in_tensorflow",
                    "title": "Customizing what happens in `fit()` with TensorFlow",
                },
                {
                    "path": "custom_train_step_in_jax",
                    "title": "Customizing what happens in `fit()` with JAX",
                },
                {
                    "path": "custom_train_step_in_torch",
                    "title": "Customizing what happens in `fit()` with PyTorch",
                },
                {
                    "path": "writing_a_custom_training_loop_in_tensorflow",
                    "title": "Writing a custom training loop with TensorFlow",
                },
                {
                    "path": "writing_a_custom_training_loop_in_jax",
                    "title": "Writing a custom training loop with JAX",
                },
                {
                    "path": "writing_a_custom_training_loop_in_torch",
                    "title": "Writing a custom training loop with PyTorch",
                },
                {
                    "path": "distributed_training_with_tensorflow",
                    "title": "Distributed training with TensorFlow",
                },
                {
                    "path": "distributed_training_with_jax",
                    "title": "Distributed training with JAX",
                },
                {
                    "path": "distributed_training_with_torch",
                    "title": "Distributed training with PyTorch",
                },
            ],
        },
        KERAS_CORE_API_MASTER,
    ],
}
