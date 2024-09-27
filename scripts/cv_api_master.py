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
        "keras_cv.bounding_box.REL_XYWH",
        "keras_cv.bounding_box.XYXY",
        "keras_cv.bounding_box.REL_XYXY",
        "keras_cv.bounding_box.YXYX",
        "keras_cv.bounding_box.REL_YXYX",
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

LOSSES_MASTER = {
    "path": "losses/",
    "title": "Losses",
    "toc": True,
    "children": [
        {
            "path": "binary_focal_crossentropy",
            "title": "Binary Penalty Reduced Focal CrossEntropy",
            "generate": ["keras_cv.losses.BinaryPenaltyReducedFocalCrossEntropy"],
        },
        {
            "path": "ciou_loss",
            "title": "CIoU Loss",
            "generate": ["keras_cv.losses.CIoULoss"],
        },
        {
            "path": "focal_loss",
            "title": "Focal Loss",
            "generate": ["keras_cv.losses.FocalLoss"],
        },
        {
            "path": "giou_loss",
            "title": "GIoU Loss",
            "generate": ["keras_cv.losses.GIoULoss"],
        },
        {
            "path": "iou_loss",
            "title": "IoU Loss",
            "generate": ["keras_cv.losses.IoULoss"],
        },
        {
            "path": "simclr_loss",
            "title": "SimCLR Loss",
            "generate": ["keras_cv.losses.SimCLRLoss"],
        },
        {
            "path": "smoothl1_loss",
            "title": "SmoothL1Loss Loss",
            "generate": ["keras_cv.losses.SmoothL1Loss"],
        },
    ],
}


BACKBONES_MASTER = {
    "path": "backbones/",
    "title": "Backbones",
    "toc": True,
    "children": [
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
            "path": "densenet",
            "title": "DenseNet backbones",
            "generate": [
                "keras_cv.models.DenseNetBackbone",
                "keras_cv.models.DenseNetBackbone.from_preset",
                "keras_cv.models.DenseNet121Backbone",
                "keras_cv.models.DenseNet169Backbone",
                "keras_cv.models.DenseNet201Backbone",
            ],
        },
        {
            "path": "efficientnet_v1",
            "title": "EfficientNetV1 models",
            "generate": [
                "keras_cv.models.EfficientNetV1Backbone",
                "keras_cv.models.EfficientNetV1Backbone.from_preset",
                "keras_cv.models.EfficientNetV1B0Backbone",
                "keras_cv.models.EfficientNetV1B1Backbone",
                "keras_cv.models.EfficientNetV1B2Backbone",
                "keras_cv.models.EfficientNetV1B3Backbone",
                "keras_cv.models.EfficientNetV1B4Backbone",
                "keras_cv.models.EfficientNetV1B5Backbone",
                "keras_cv.models.EfficientNetV1B6Backbone",
                "keras_cv.models.EfficientNetV1B7Backbone",
            ],
        },
        {
            "path": "efficientnet_v2",
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
        {
            "path": "efficientnet_lite",
            "title": "EfficientNet Lite backbones",
            "generate": [
                "keras_cv.models.EfficientNetLiteBackbone",
                "keras_cv.models.EfficientNetLiteBackbone.from_preset",
                "keras_cv.models.EfficientNetLiteB0Backbone",
                "keras_cv.models.EfficientNetLiteB1Backbone",
                "keras_cv.models.EfficientNetLiteB2Backbone",
                "keras_cv.models.EfficientNetLiteB3Backbone",
                "keras_cv.models.EfficientNetLiteB4Backbone",
            ],
        },
        {
            "path": "mix_transformer",
            "title": "MixTransformer backbones",
            "generate": [
                "keras_cv.models.MiTBackbone",
                "keras_cv.models.MiTBackbone.from_preset",
                "keras_cv.models.MiTB0Backbone",
                "keras_cv.models.MiTB1Backbone",
                "keras_cv.models.MiTB2Backbone",
                "keras_cv.models.MiTB3Backbone",
                "keras_cv.models.MiTB4Backbone",
                "keras_cv.models.MiTB5Backbone",
            ],
        },
        {
            "path": "mobilenet_v3",
            "title": "MobileNetV3 backbones",
            "generate": [
                "keras_cv.models.MobileNetV3Backbone",
                "keras_cv.models.MobileNetV3Backbone.from_preset",
                "keras_cv.models.MobileNetV3SmallBackbone",
                "keras_cv.models.MobileNetV3LargeBackbone",
            ],
        },
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
            "path": "vgg16",
            "title": "VGG16 backbones",
            "generate": [
                "keras_cv.models.VGG16Backbone",
                "keras_cv.models.VGG16Backbone.from_preset",
            ],
        },
        {
            "path": "vitdet",
            "title": "ViTDet backbones",
            "generate": [
                "keras_cv.models.ViTDetBackbone",
                "keras_cv.models.ViTDetBackbone.from_preset",
                "keras_cv.models.ViTDetBBackbone",
                "keras_cv.models.ViTDetLBackbone",
                "keras_cv.models.ViTDetHBackbone",
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
    ],
}

TASKS_MASKTER = {
    "path": "tasks/",
    "title": "Tasks",
    "toc": True,
    "children": [
        {
            "path": "basnet_segmentation",
            "title": "BASNet Segmentation",
            "generate": [
                "keras_cv.models.BASNet",
                "keras_cv.models.BASNet.from_preset",
            ],
        },
        {
            "path": "deeplab_v3_segmentation",
            "title": "DeepLabV3Plus Segmentation",
            "generate": [
                "keras_cv.models.DeepLabV3Plus",
                "keras_cv.models.DeepLabV3Plus.from_preset",
            ],
        },
        {
            "path": "segformer_segmentation",
            "title": "SegFormer Segmentation",
            "generate": [
                "keras_cv.models.SegFormer",
                "keras_cv.models.SegFormer.from_preset",
                "keras_cv.models.SegFormerB0",
                "keras_cv.models.SegFormerB1",
                "keras_cv.models.SegFormerB2",
                "keras_cv.models.SegFormerB3",
                "keras_cv.models.SegFormerB4",
                "keras_cv.models.SegFormerB5",
            ],
        },
        {
            "path": "segment_anything",
            "title": "Segment Anything",
            "generate": [
                "keras_cv.models.SegmentAnythingModel",
                "keras_cv.models.SegmentAnythingModel.from_preset",
                "keras_cv.models.SAMMaskDecoder",
                "keras_cv.models.SAMPromptEncoder",
                "keras_cv.models.TwoWayTransformer",
                "keras_cv.layers.MultiHeadAttentionWithDownsampling",
                "keras_cv.layers.TwoWayMultiHeadAttention",
                "keras_cv.layers.RandomFrequencyPositionalEmbeddings",
            ],
        },
        {
            "path": "feature_extractor",
            "title": "CLIP Feature extractor",
            "generate": [
                "keras_cv.models.CLIP",
                "keras_cv.models.CLIP.from_preset",
                "keras_cv.models.feature_extractor.CLIPAttention",
                "keras_cv.models.feature_extractor.CLIPEncoder",
                "keras_cv.models.feature_extractor.CLIPImageEncoder",
                # "keras_cv.models.feature_extractor.CLIPPatchingAndEmbedding",
                "keras_cv.models.feature_extractor.CLIPProcessor",
                "keras_cv.models.feature_extractor.CLIPTextEncoder",
                "keras_cv.models.feature_extractor.QuickGELU",
                "keras_cv.models.feature_extractor.ResidualAttention",
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
            "path": "retinanet",
            "title": "The RetinaNet model",
            "generate": [
                "keras_cv.models.RetinaNet",
                "keras_cv.models.RetinaNet.from_preset",
                "keras_cv.models.retinanet.PredictionHead"
            ],
        },
        {
            "path": "stable_diffusion",
            "title": "StableDiffusion image-generation model",
            "generate": [
                "keras_cv.models.StableDiffusion",
                "keras_cv.models.StableDiffusionV2",
                "keras_cv.models.stable_diffusion.Decoder",
                "keras_cv.models.stable_diffusion.DiffusionModel",
                "keras_cv.models.stable_diffusion.ImageEncoder",
                "keras_cv.models.stable_diffusion.NoiseScheduler",
                "keras_cv.models.stable_diffusion.SimpleTokenizer",
                "keras_cv.models.stable_diffusion.TextEncoder",
                "keras_cv.models.stable_diffusion.TextEncoderV2",
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
    "children": [LAYERS_MASTER, MODELS_MASTER, BOUNDING_BOX_MASTER, LOSSES_MASTER],
}
