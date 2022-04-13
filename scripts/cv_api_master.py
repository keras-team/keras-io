PREPROCESSING_MASTER = {
    'path': 'preprocessing/',
    'title': 'Preprocessing Layers',
    'toc': True,
    'children': [
        {
            'path': 'auto_contrast',
            'title': 'The AutoContrast class',
            'generate': ['keras_cv.layers.AutoContrast']
        },
        {
            'path': 'channel_shuffle',
            'title': 'The ChannelShuffle class',
            'generate': ['keras_cv.layers.ChannelShuffle']
        },
        {
            'path': 'cut_mix',
            'title': 'The CutMix class',
            'generate': ['keras_cv.layers.CutMix']
        },
        {
            'path': 'equalization',
            'title': 'The Equalization class',
            'generate': ['keras_cv.layers.Equalization']
        },
        {
            'path': 'grayscale',
            'title': 'The Grayscale class',
            'generate': ['keras_cv.layers.Grayscale']
        },
        {
            'path': 'grid_mask',
            'title': 'The GridMask class',
            'generate': ['keras_cv.layers.GridMask']
        },
        {
            'path': 'mix_up',
            'title': 'The MixUp class',
            'generate': ['keras_cv.layers.MixUp']
        },
        {
            'path': 'posterization',
            'title': 'The Posterization class',
            'generate': ['keras_cv.layers.Posterization']
        },
        {
            'path': 'rand_augment',
            'title': 'The RandAugment class',
            'generate': ['keras_cv.layers.RandAugment']
        },
        {
            'path': 'random_augmentation_pipeline',
            'title': 'The RandomAugmentationPipeline class',
            'generate': ['keras_cv.layers.RandomAugmentationPipeline']
        },
        {
            'path': 'random_channel_shift',
            'title': 'The RandomChannelShift class',
            'generate': ['keras_cv.layers.RandomChannelShift']
        },
        {
            'path': 'rancom_color_degeneration',
            'title': 'The RandomColorDegeneration class',
            'generate': ['keras_cv.layers.RandomColorDegeneration']
        },
        {
            'path': 'random_cutout',
            'title': 'The RandomCutout class',
            'generate': ['keras_cv.layers.RandomCutout']
        },
        {
            'path': 'random_hue',
            'title': 'The RandomHue class',
            'generate': ['keras_cv.layers.RandomHue']
        },
        {
            'path': 'random_saturation',
            'title': 'The RandomSaturation class',
            'generate': ['keras_cv.layers.RandomSaturation']
        },
        {
            'path': 'random_sharpness',
            'title': 'The RandomSharpness class',
            'generate': ['keras_cv.layers.RandomSharpness']
        },
        {
            'path': 'random_shear',
            'title': 'The RandomShear class',
            'generate': ['keras_cv.layers.RandomShear']
        },
        {
            'path': 'solarization',
            'title': 'The Solarization class',
            'generate': ['keras_cv.layers.Solarization']
        },
    ]
}

REGULARIZATION_MASTER = {
    'path': 'regularization/',
    'title': 'Regularization Layers',
    'toc': True,
    'children': [
        {
            'path': 'dropblock2d',
            'title': 'The DropBlock2D class',
            'generate': [
                'keras_cv.layers.DropBlock2D',
            ]
        }
    ]
}

LAYERS_MASTER = {
    'path': 'layers/',
    'title': 'Layers',
    'toc': True,
    'children': [
        PREPROCESSING_MASTER,
        REGULARIZATION_MASTER
    ]
}


METRICS_MASTER = {
    'path': 'metrics/',
    'title': 'Metrics',
    'toc': True,
    'children': [
        {
            'path': 'coco_mean_average_precision',
            'title': 'The COCOMeanAveragePrecision class',
            'generate': [
                'keras_cv.metrics.COCOMeanAveragePrecision',
            ]
        },
        {
            'path': 'coco_recall',
            'title': 'The COCORecall class',
            'generate': [
                'keras_cv.metrics.COCORecall',
            ]
        },
    ]
}

CV_API_MASTER = {
    'path': 'keras_cv/',
    'title': 'KerasCV',
    'toc': True,
    'children': [
        LAYERS_MASTER,
        METRICS_MASTER
    ]
}
