CV_GUIDES_MASTER = {
    "path": "keras_cv/",
    "title": "KerasCV",
    "toc": True,
    "children": [
        {
            "path": "cut_mix_mix_up_and_rand_augment",
            "title": "CutMix, MixUp, and RandAugment image augmentation with KerasCV",
        },
        {
            "path": "custom_image_augmentations",
            "title": "Custom Image Augmentations with BaseImageAugmentationLayer",
        },
        {"path": "coco_metrics", "title": "Using KerasCV COCO Metrics"},
    ],
}

NLP_GUIDES_MASTER = {
    "path": "keras_nlp/",
    "title": "KerasNLP",
    "toc": True,
    "children": [
        {
            "path": "transformer_pretraining",
            "title": "Pretraining a Transformer from scratch with KerasNLP",
        },
    ],
}

KT_GUIDES_MASTER = {
    "path": "keras_tuner/",
    "title": "Hyperparameter Tuning",
    "toc": True,
    "children": [
        {
            "path": "getting_started",
            "title": "Getting started with KerasTuner",
        },
        {
            "path": "distributed_tuning",
            "title": "Distributed hyperparameter tuning with KerasTuner",
        },
        {
            "path": "custom_tuner",
            "title": "Tune hyperparameters in your custom training loop",
        },
        {
            "path": "visualize_tuning",
            "title": "Visualize the hyperparameter tuning process",
        },
        {
            "path": "tailor_the_search_space",
            "title": "Tailor the search space",
        },
    ],
}

GUIDES_MASTER = {
    "path": "guides/",
    "title": "Developer guides",
    "toc": True,
    "children": [
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
        {
            "path": "customizing_what_happens_in_fit",
            "title": "Customizing what happens in `fit()`",
        },
        {
            "path": "writing_a_training_loop_from_scratch",
            "title": "Writing a training loop from scratch",
        },
        {
            "path": "serialization_and_saving",
            "title": "Serialization & saving",
        },
        {
            "path": "writing_your_own_callbacks",
            "title": "Writing your own callbacks",
        },
        # {
        #     'path': 'writing_your_own_metrics',
        #     'title': 'Writing your own Metrics',
        # },
        # {
        #     'path': 'writing_your_own_losses',
        #     'title': 'Writing your own Losses',
        # },
        {
            "path": "preprocessing_layers",
            "title": "Working with preprocessing layers",
        },
        {
            "path": "working_with_rnns",
            "title": "Working with recurrent neural networks",
        },
        {
            "path": "understanding_masking_and_padding",
            "title": "Understanding masking & padding",
        },
        {
            "path": "distributed_training",
            "title": "Multi-GPU & distributed training",
        },
        # {
        #     'path': 'tpu_training',
        #     'title': 'Training Keras models on TPU',
        # },
        {
            "path": "transfer_learning",
            "title": "Transfer learning & fine-tuning",
        },
        # {
        #     'path': 'hyperparameter_optimization',
        #     'title': 'Hyperparameter optimization',
        # },
        KT_GUIDES_MASTER,
        CV_GUIDES_MASTER,
        NLP_GUIDES_MASTER,
        # TODO: mixed precision
    ],
}
