PREPROCESSING_MASTER = {
    'path': 'preprocessing/',
    'title': 'KerasCV Preprocessing Layers',
    'toc': True,
    'children': [
        {
            'path': 'auto_contrast',
            'title': 'The AutoContrast class',
            'generate': [
                'keras_cv.layers.AutoContrast',
            ]
        },
    ]
}

REGULARIZATION_MASTER = {

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
        #LAYERS_MASTER,
        METRICS_MASTER
    ]
}
