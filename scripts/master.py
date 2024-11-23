from guides_master import GUIDES_MASTER
from examples_master import EXAMPLES_MASTER
from api_master import API_MASTER
from tuner_master import TUNER_MASTER
from hub_master import HUB_MASTER
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
                    "path": "benchmarks",
                    "title": "Keras 3 benchmarks",
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
        EXAMPLES_MASTER,
        API_MASTER,
        KERAS2_API_MASTER,
        TUNER_MASTER,
        HUB_MASTER,
    ],
}
