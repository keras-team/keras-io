AUGMENTATION_MASTER = {
    "path": "augmentation/",
    "title": "Augmentation layers",
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
            "path": "fourier_mix",
            "title": "FourierMix layer",
            "generate": ["keras_cv.layers.FourierMix"],
        },
        {
            "path": "grid_mask",
            "title": "GridMask layer",
            "generate": ["keras_cv.layers.GridMask"],
        },
        {
            "path": "jittered_resize",
            "title": "JitteredResize layer",
            "generate": ["keras_cv.layers.JitteredResize"],
        },
        {
            "path": "mix_up",
            "title": "MixUp layer",
            "generate": ["keras_cv.layers.MixUp"],
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

PREPROCESSING_MASTER = {
    "path": "preprocessing/",
    "title": "Preprocessing layers",
    "toc": True,
    "children": [
        {
            "path": "resizing",
            "title": "Resizing layer",
            "generate": ["keras_cv.layers.Resizing"],
        },
        {
            "path": "grayscale",
            "title": "Grayscale layer",
            "generate": ["keras_cv.layers.Grayscale"],
        },
        {
            "path": "equalization",
            "title": "Equalization layer",
            "generate": ["keras_cv.layers.Equalization"],
        },
        {
            "path": "posterization",
            "title": "Posterization layer",
            "generate": ["keras_cv.layers.Posterization"],
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
        {
            "path": "compute_iou",
            "title": "Compute intersection over union of bounding boxes",
            "generate": ["keras_cv.bounding_box.compute_iou"],
        },
        {
            "path": "clip_to_image",
            "title": "Clip bounding boxes to be within the bounds of provided images",
            "generate": ["keras_cv.bounding_box.clip_to_image"],
        },
        {
            "path": "to_dense",
            "title": "Convert a bounding box dictionary to -1 padded Dense tensors",
            "generate": ["keras_cv.bounding_box.to_dense"],
        },
        {
            "path": "to_ragged",
            "title": "Convert a bounding box dictionary batched Ragged tensors",
            "generate": ["keras_cv.bounding_box.to_ragged"],
        },
        {
            "path": "validate_format",
            "title": "Ensure that your bounding boxes comply with the bounding box spec",
            "generate": ["keras_cv.bounding_box.validate_format"],
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
        },
        {
            "path": "drop_path",
            "title": "DropPath layer",
            "generate": [
                "keras_cv.layers.DropPath",
            ],
        },
        {
            "path": "squeeze_and_excite_2d",
            "title": "SqueezeAndExcite2D layer",
            "generate": [
                "keras_cv.layers.SqueezeAndExcite2D",
            ],
        },
        {
            "path": "squeeze_and_excite_2d",
            "title": "SqueezeAndExcite2D layer",
            "generate": [
                "keras_cv.layers.SqueezeAndExcite2D",
            ],
        },
        {
            "path": "stochastic_depth",
            "title": "StochasticDepth layer",
            "generate": [
                "keras_cv.layers.StochasticDepth",
            ],
        },
    ],
}

LAYERS_MASTER = {
    "path": "layers/",
    "title": "Layers",
    "toc": True,
    "children": [AUGMENTATION_MASTER, PREPROCESSING_MASTER, REGULARIZATION_MASTER],
}

#
# METRICS_MASTER = {
#     "path": "metrics/",
#     "title": "Metrics",
#     "toc": True,
#     "children": [
#         # Temporarily remove COCO metrics
#         # {
#         #     "path": "coco_mean_average_precision",
#         #     "title": "COCOMeanAveragePrecision metric",
#         #     "generate": [
#         #         "keras_cv.metrics.COCOMeanAveragePrecision",
#         #     ],
#         # },
#         # {
#         #     "path": "coco_recall",
#         #     "title": "COCORecall metric",
#         #     "generate": [
#         #         "keras_cv.metrics.COCORecall",
#         #     ],
#         # },
#     ],
# }

MODELS_MASTER = {
    "path": "models/",
    "title": "Models",
    "toc": True,
    "children": [
        {
            "path": "stable_diffusion",
            "title": "StableDiffusion image-generation model",
            "generate": [
                "keras_cv.models.StableDiffusion",
            ],
        },
        {
            "path": "retinanet",
            "title": "The RetinaNet model",
            "generate": [
                "keras_cv.models.RetinaNet",
            ],
        },
        {
            "path": "faster_rcnn",
            "title": "The FasterRCNN model",
            "generate": ["keras_cv.models.FasterRCNN"],
        },
        {
            "path": "efficientnetv2",
            "title": "EfficientNetV2 models",
            "generate": [
                "keras_cv.models.EfficientNetV2B0",
                "keras_cv.models.EfficientNetV2B1",
                "keras_cv.models.EfficientNetV2B2",
                "keras_cv.models.EfficientNetV2B3",
                "keras_cv.models.EfficientNetV2S",
                "keras_cv.models.EfficientNetV2M",
                "keras_cv.models.EfficientNetV2L",
            ],
        },
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
    "children": [LAYERS_MASTER, MODELS_MASTER, BOUNDING_BOX_MASTER],
}
