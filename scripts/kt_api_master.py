PAGES = {
    'documentation/oracles.md': [
        'kerastuner.oracles.BayesianOptimization',
        'kerastuner.oracles.Hyperband',
        'kerastuner.oracles.RandomSearch',
        'kerastuner.Oracle',
        'kerastuner.Oracle.create_trial',
        'kerastuner.Oracle.end_trial',
        'kerastuner.Oracle.get_best_trials',
        'kerastuner.Oracle.get_state',
        'kerastuner.Oracle.set_state',
        'kerastuner.Oracle.update_trial',
    ],
}

KT_API_MASTER = {
    'path': 'keras-tuner/',
    'title': 'Keras Tuner',
    'toc': True,
    'children': [
        {
            'path': 'hyperparameters',
            'title': 'The HyperParameters class',
            'generate': [
                'kerastuner.HyperParameters',
                'kerastuner.HyperParameters.Boolean',
                'kerastuner.HyperParameters.Choice',
                'kerastuner.HyperParameters.Fixed',
                'kerastuner.HyperParameters.Float',
                'kerastuner.HyperParameters.Int',
                'kerastuner.HyperParameters.conditional_scope',
                'kerastuner.HyperParameters.get',
            ]
        },
        {
            'path': 'tuners',
            'title': 'The Tuner class',
            'generate': [
                'kerastuner.BayesianOptimization',
                'kerastuner.Hyperband',
                'kerastuner.RandomSearch',
                'kerastuner.tuners.Sklearn',
                'kerastuner.Tuner',
                'kerastuner.Tuner.get_best_models',
                'kerastuner.Tuner.get_state',
                'kerastuner.Tuner.load_model',
                'kerastuner.Tuner.on_epoch_begin',
                'kerastuner.Tuner.on_batch_begin',
                'kerastuner.Tuner.on_batch_end',
                'kerastuner.Tuner.on_epoch_end',
                'kerastuner.Tuner.run_trial',
                'kerastuner.Tuner.save_model',
                'kerastuner.Tuner.search',
                'kerastuner.Tuner.set_state',
                'kerastuner.Tuner.get_best_hyperparameters',
                'kerastuner.Tuner.get_best_models',
                'kerastuner.Tuner.get_state',
                'kerastuner.Tuner.load_model',
                'kerastuner.Tuner.run_trial',
                'kerastuner.Tuner.save_model',
                'kerastuner.Tuner.search',
                'kerastuner.Tuner.set_state',
            ]
        },
        {
            'path': 'hypermodels',
            'title': 'The HyperModel class',
            'generate': [
                'kerastuner.HyperModel',
                'kerastuner.HyperModel.build',
                'kerastuner.applications.HyperXception',
                'kerastuner.applications.HyperResNet',
            ]
        },
        {
            'path': 'oracles',
            'title': 'The Oracle class',
            'generate': [
                'kerastuner.oracles.BayesianOptimization',
                'kerastuner.oracles.Hyperband',
                'kerastuner.oracles.RandomSearch',
                'kerastuner.Oracle',
                'kerastuner.Oracle.create_trial',
                'kerastuner.Oracle.end_trial',
                'kerastuner.Oracle.get_best_trials',
                'kerastuner.Oracle.get_state',
                'kerastuner.Oracle.set_state',
                'kerastuner.Oracle.update_trial',
            ]
        },
    ]
}
