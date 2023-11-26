from guides_master import GUIDES_MASTER
from examples_master import EXAMPLES_MASTER
from api_master import API_MASTER
from keras2_api_master import KERAS2_API_MASTER

MASTER = {
    "path": "/",
    "title": "Keras: the Python Deep Learning library",
    "children": [
        {
            "path": "about",
            "title": "About Keras",
        },
        {
            "path": "getting_started/",
            "title": "Getting started",
            "children": [
                {
                    "path": "intro_to_keras_for_engineers",
                    "title": "Introduction to Keras for engineers",
                },
                {
                    "path": "ecosystem",
                    "title": "The Keras ecosystem",
                },
                {
                    "path": "faq",
                    "title": "Frequently Asked Questions",
                    "outline": False,
                },
            ],
        },
        GUIDES_MASTER,
        API_MASTER,
        KERAS2_API_MASTER,
        EXAMPLES_MASTER,
        {
            "path": "keras_tuner/",
            "title": "KerasTuner: Hyperparameter Tuning",
        },
        {
            "path": "keras_cv/",
            "title": "KerasCV: Computer Vision Workflows",
        },
        {
            "path": "keras_nlp/",
            "title": "KerasNLP: Natural Language Workflows",
        },
    ],
}
