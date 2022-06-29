PREPROCESSING_MASTER = {
    "path": "preprocessing/",
    "title": "Preprocessing layers",
    "toc": True,
    "children": [
        {
            "path": "auto_contrast",
            "title": "AutoContrast layer",
            "generate": ["keras_cv.layers.AutoContrast"],
        },
        {
            "path": "aug_mix",
            "title": "AugMix layer",
            "generate": ["keras_cv.layers.AugMix"],
        },
        {
            "path": "channel_shuffle",
            "title": "ChannelShuffle layer",
            "generate": ["keras_cv.layers.ChannelShuffle"],
        },
        {
            "path": "cut_mix",
            "title": "CutMix layer",
            "generate": ["keras_cv.layers.CutMix"],
        },
        {
            "path": "equalization",
            "title": "Equalization layer",
            "generate": ["keras_cv.layers.Equalization"],
        },
        {
            "path": "fourier_mix",
            "title": "FourierMix layer",
            "generate": ["keras_cv.layers.FourierMix"],
        },
        {
            "path": "grayscale",
            "title": "Grayscale layer",
            "generate": ["keras_cv.layers.Grayscale"],
        },
        {
            "path": "grid_mask",
            "title": "GridMask layer",
            "generate": ["keras_cv.layers.GridMask"],
        },
        {
            "path": "mix_up",
            "title": "MixUp layer",
            "generate": ["keras_cv.layers.MixUp"],
        },
        {
            "path": "posterization",
            "title": "Posterization layer",
            "generate": ["keras_cv.layers.Posterization"],
        },
        {
            "path": "rand_augment",
            "title": "RandAugment layer",
            "generate": ["keras_cv.layers.RandAugment"],
        },
        {
            "path": "random_augmentation_pipeline",
            "title": "RandomAugmentationPipeline layer",
            "generate": ["keras_cv.layers.RandomAugmentationPipeline"],
        },
        {
            "path": "random_channel_shift",
            "title": "RandomChannelShift layer",
            "generate": ["keras_cv.layers.RandomChannelShift"],
        },
        {
            "path": "random_color_degeneration",
            "title": "RandomColorDegeneration layer",
            "generate": ["keras_cv.layers.RandomColorDegeneration"],
        },
        {
            "path": "random_cutout",
            "title": "RandomCutout layer",
            "generate": ["keras_cv.layers.RandomCutout"],
        },
        {
            "path": "random_hue",
            "title": "RandomHue layer",
            "generate": ["keras_cv.layers.RandomHue"],
        },
        {
            "path": "random_saturation",
            "title": "RandomSaturation layer",
            "generate": ["keras_cv.layers.RandomSaturation"],
        },
        {
            "path": "random_sharpness",
            "title": "RandomSharpness layer",
            "generate": ["keras_cv.layers.RandomSharpness"],
        },
        {
            "path": "random_shear",
            "title": "RandomShear layer",
            "generate": ["keras_cv.layers.RandomShear"],
        },
        {
            "path": "solarization",
            "title": "Solarization layer",
            "generate": ["keras_cv.layers.Solarization"],
        },
    ],
}

BOUNDING_BOX_FORMATS = {
    "path": "formats",
    "title": "Bounding box formats",
    "generate": [
        "keras_cv.bounding_box.CENTER_XYWH",
        "keras_cv.bounding_box.XYWH",
        "keras_cv.bounding_box.XYXY",
        "keras_cv.bounding_box.REL_XYXY",
    ],
}

BOUNDING_BOX_UTILS = {
    "path": "utils/",
    "title": "Bounding box utilities",
    "toc": True,
    "children": [
        {
            "path": "convert_format",
            "title": "Convert bounding box formats",
            "generate": ["keras_cv.bounding_box.convert_format"],
        },
    ],
}

BOUNDING_BOX_MASTER = {
    "path": "bounding_box/",
    "title": "Bounding box formats and utilities",
    "toc": True,
    "children": [BOUNDING_BOX_FORMATS, BOUNDING_BOX_UTILS],
}

REGULARIZATION_MASTER = {
    "path": "regularization/",
    "title": "Regularization layers",
    "toc": True,
    "children": [
        {
            "path": "dropblock2d",
            "title": "DropBlock2D layer",
            "generate": [
                "keras_cv.layers.DropBlock2D",
            ],
        }
    ],
}

LAYERS_MASTER = {
    "path": "layers/",
    "title": "Layers",
    "toc": True,
    "children": [PREPROCESSING_MASTER, REGULARIZATION_MASTER],
}


METRICS_MASTER = {
    "path": "metrics/",
    "title": "Metrics",
    "toc": True,
    "children": [
        {
            "path": "coco_mean_average_precision",
            "title": "COCOMeanAveragePrecision metric",
            "generate": [
                "keras_cv.metrics.COCOMeanAveragePrecision",
            ],
        },
        {
            "path": "coco_recall",
            "title": "COCORecall metric",
            "generate": [
                "keras_cv.metrics.COCORecall",
            ],
        },
    ],
}

MODELS_MASTER = {
    "path": "models/",
    "title": "Models",
    "toc": True,
    "children": [
        {
            "path": "densenet",
            "title": "DenseNet models",
            "generate": [
                "keras_cv.models.DenseNet121",
                "keras_cv.models.DenseNet169",
                "keras_cv.models.DenseNet201",
            ],
        },
    ],
}

CV_API_MASTER = {
    "path": "keras_cv/",
    "title": "KerasCV",
    "toc": True,
    "children": [LAYERS_MASTER, METRICS_MASTER, MODELS_MASTER, BOUNDING_BOX_MASTER],
}
