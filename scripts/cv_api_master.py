PREPROCESSING_MASTER = {
    'path': 'preprocessing/',
    'title': 'Preprocessing layers',
    'toc': True,
    'children': [
        {
            'path': 'auto_contrast',
            'title': 'AutoContrast layer',
            'generate': ['keras_cv.layers.AutoContrast']
        },
        {
            'path': 'channel_shuffle',
            'title': 'ChannelShuffle layer',
            'generate': ['keras_cv.layers.ChannelShuffle']
        },
        {
            'path': 'cut_mix',
            'title': 'CutMix layer',
            'generate': ['keras_cv.layers.CutMix']
        },
        {
            'path': 'equalization',
            'title': 'Equalization layer',
            'generate': ['keras_cv.layers.Equalization']
        },
        {
            'path': 'grayscale',
            'title': 'Grayscale layer',
            'generate': ['keras_cv.layers.Grayscale']
        },
        {
            'path': 'grid_mask',
            'title': 'GridMask layer',
            'generate': ['keras_cv.layers.GridMask']
        },
        {
            'path': 'mix_up',
            'title': 'MixUp layer',
            'generate': ['keras_cv.layers.MixUp']
        },
        {
            'path': 'posterization',
            'title': 'Posterization layer',
            'generate': ['keras_cv.layers.Posterization']
        },
        {
            'path': 'rand_augment',
            'title': 'RandAugment layer',
            'generate': ['keras_cv.layers.RandAugment']
        },
        {
            'path': 'random_augmentation_pipeline',
            'title': 'RandomAugmentationPipeline layer',
            'generate': ['keras_cv.layers.RandomAugmentationPipeline']
        },
        {
            'path': 'random_channel_shift',
            'title': 'RandomChannelShift layer',
            'generate': ['keras_cv.layers.RandomChannelShift']
        },
        {
            'path': 'rancom_color_degeneration',
            'title': 'RandomColorDegeneration layer',
            'generate': ['keras_cv.layers.RandomColorDegeneration']
        },
        {
            'path': 'random_cutout',
            'title': 'RandomCutout layer',
            'generate': ['keras_cv.layers.RandomCutout']
        },
        {
            'path': 'random_hue',
            'title': 'RandomHue layer',
            'generate': ['keras_cv.layers.RandomHue']
        },
        {
            'path': 'random_saturation',
            'title': 'RandomSaturation layer',
            'generate': ['keras_cv.layers.RandomSaturation']
        },
        {
            'path': 'random_sharpness',
            'title': 'RandomSharpness layer',
            'generate': ['keras_cv.layers.RandomSharpness']
        },
        {
            'path': 'random_shear',
            'title': 'RandomShear layer',
            'generate': ['keras_cv.layers.RandomShear']
        },
        {
            'path': 'solarization',
            'title': 'Solarization layer',
            'generate': ['keras_cv.layers.Solarization']
        },
    ]
}

REGULARIZATION_MASTER = {
    'path': 'regularization/',
    'title': 'Regularization layers',
    'toc': True,
    'children': [
        {
            'path': 'dropblock2d',
            'title': 'DropBlock2D layer',
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
            'title': 'COCOMeanAveragePrecision metric',
            'generate': [
                'keras_cv.metrics.COCOMeanAveragePrecision',
            ]
        },
        {
            'path': 'coco_recall',
            'title': 'COCORecall metric',
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
