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
        # {
        #     "path": "to_dense",
        #     "title": "Convert a bounding box dictionary to -1 padded Dense tensors",
        #     "generate": ["keras_cv.bounding_box.to_dense"],
        # },
        # {
        #     "path": "to_ragged",
        #     "title": "Convert a bounding box dictionary batched Ragged tensors",
        #     "generate": ["keras_cv.bounding_box.to_ragged"],
        # },
        # {
        #     "path": "validate_format",
        #     "title": "Ensure that your bounding boxes comply with the bounding box spec",
        #     "generate": ["keras_cv.bounding_box.validate_format"],
        # },
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


BACKBONES_MASTER = {
    "path": "backbones/",
    "title": "Backbones",
    "toc": True,
    "children": [
        {
            "path": "resnet_v1",
            "title": "ResNetV1 backbones",
            "generate": [
                "keras_cv.models.ResNetBackbone",
                "keras_cv.models.ResNetBackbone.from_preset",
                "keras_cv.models.ResNet18Backbone",
                "keras_cv.models.ResNet34Backbone",
                "keras_cv.models.ResNet50Backbone",
                "keras_cv.models.ResNet101Backbone",
                "keras_cv.models.ResNet152Backbone",
            ],
        },
        {
            "path": "resnet_v2",
            "title": "ResNetV2 backbones",
            "generate": [
                "keras_cv.models.ResNetV2Backbone",
                "keras_cv.models.ResNetV2Backbone.from_preset",
                "keras_cv.models.ResNet18V2Backbone",
                "keras_cv.models.ResNet34V2Backbone",
                "keras_cv.models.ResNet50V2Backbone",
                "keras_cv.models.ResNet101V2Backbone",
                "keras_cv.models.ResNet152V2Backbone",
            ],
        },
        {
            "path": "csp_darknet",
            "title": "CSPDarkNet backbones",
            "generate": [
                "keras_cv.models.CSPDarkNetBackbone",
                "keras_cv.models.CSPDarkNetBackbone.from_preset",
                "keras_cv.models.CSPDarkNetTinyBackbone",
                "keras_cv.models.CSPDarkNetSBackbone",
                "keras_cv.models.CSPDarkNetMBackbone",
                "keras_cv.models.CSPDarkNetLBackbone",
                "keras_cv.models.CSPDarkNetXLBackbone",
            ],
        },
        {
            "path": "yolo_v8",
            "title": "YOLOV8 backbones",
            "generate": [
                "keras_cv.models.YOLOV8Backbone",
                "keras_cv.models.YOLOV8Backbone.from_preset",
            ],
        },
        {
            "path": "mobilenetv3",
            "title": "MobileNetV3 backbones",
            "generate": [
                "keras_cv.models.MobileNetV3Backbone",
                "keras_cv.models.MobileNetV3Backbone.from_preset",
                "keras_cv.models.MobileNetV3SmallBackbone",
                "keras_cv.models.MobileNetV3LargeBackbone",
            ],
        },
        {
            "path": "efficientnetv2",
            "title": "EfficientNetV2 models",
            "generate": [
                "keras_cv.models.EfficientNetV2Backbone",
                "keras_cv.models.EfficientNetV2Backbone.from_preset",
                "keras_cv.models.EfficientNetV2B0Backbone",
                "keras_cv.models.EfficientNetV2B1Backbone",
                "keras_cv.models.EfficientNetV2B2Backbone",
                "keras_cv.models.EfficientNetV2B3Backbone",
                "keras_cv.models.EfficientNetV2SBackbone",
                "keras_cv.models.EfficientNetV2MBackbone",
                "keras_cv.models.EfficientNetV2LBackbone",
            ],
        },
    ],
}

TASKS_MASKTER = {
    "path": "tasks/",
    "title": "Tasks",
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
                "keras_cv.models.RetinaNet.from_preset",
            ],
        },
        {
            "path": "image_classifier",
            "title": "The ImageClassifier model",
            "generate": [
                "keras_cv.models.ImageClassifier",
                "keras_cv.models.ImageClassifier.from_preset",
            ],
        },
        {
            "path": "yolo_v8_detector",
            "title": "The YOLOV8Detector model",
            "generate": [
                "keras_cv.models.YOLOV8Detector",
                "keras_cv.models.YOLOV8Detector.from_preset",
            ],
        },
    ],
}

MODELS_MASTER = {
    "path": "models/",
    "title": "Models",
    "toc": True,
    "children": [TASKS_MASKTER, BACKBONES_MASTER],
}

CV_API_MASTER = {
    "path": "keras_cv/",
    "title": "KerasCV",
    "toc": True,
    "children": [LAYERS_MASTER, MODELS_MASTER, BOUNDING_BOX_MASTER],
}
