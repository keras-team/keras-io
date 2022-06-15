ORACLE_MASTER = {
    "path": "oracles/",
    "title": "Oracles",
    "toc": True,
    "children": [
        {
            "path": "base_oracle",
            "title": "The base Oracle class",
            "generate": [
                "keras_tuner.Oracle",
                "keras_tuner.Oracle.create_trial",
                "keras_tuner.Oracle.end_trial",
                "keras_tuner.Oracle.get_best_trials",
                "keras_tuner.Oracle.get_state",
                "keras_tuner.Oracle.set_state",
                "keras_tuner.Oracle.score_trial",
                "keras_tuner.Oracle.populate_space",
                "keras_tuner.Oracle.update_trial",
            ],
        },
        {
            "path": "random",
            "title": "RandomSearch Oracle",
            "generate": [
                "keras_tuner.oracles.RandomSearchOracle",
            ],
        },
        {
            "path": "bayesian",
            "title": "BayesianOptimization Oracle",
            "generate": [
                "keras_tuner.oracles.BayesianOptimizationOracle",
            ],
        },
        {
            "path": "hyperband",
            "title": "Hyperband Oracle",
            "generate": [
                "keras_tuner.oracles.HyperbandOracle",
            ],
        },
    ],
}

HYPERMODEL_MASTER = {
    "path": "hypermodels/",
    "title": "HyperModels",
    "toc": True,
    "children": [
        {
            "path": "base_hypermodel",
            "title": "The base HyperModel class",
            "generate": [
                "keras_tuner.HyperModel",
                "keras_tuner.HyperModel.build",
            ],
        },
        {
            "path": "hyper_resnet",
            "title": "HyperResNet",
            "generate": [
                "keras_tuner.applications.HyperResNet",
            ],
        },
        {
            "path": "hyper_xception",
            "title": "HyperXception",
            "generate": [
                "keras_tuner.applications.HyperXception",
            ],
        },
    ],
}

TUNER_MASTER = {
    "path": "tuners/",
    "title": "Tuners",
    "toc": True,
    "children": [
        {
            "path": "base_tuner",
            "title": "The base Tuner class",
            "generate": [
                "keras_tuner.Tuner",
                "keras_tuner.Tuner.get_best_hyperparameters",
                "keras_tuner.Tuner.get_best_models",
                "keras_tuner.Tuner.get_state",
                "keras_tuner.Tuner.load_model",
                "keras_tuner.Tuner.on_epoch_begin",
                "keras_tuner.Tuner.on_batch_begin",
                "keras_tuner.Tuner.on_batch_end",
                "keras_tuner.Tuner.on_epoch_end",
                "keras_tuner.Tuner.run_trial",
                "keras_tuner.Tuner.results_summary",
                "keras_tuner.Tuner.save_model",
                "keras_tuner.Tuner.search",
                "keras_tuner.Tuner.search_space_summary",
                "keras_tuner.Tuner.set_state",
            ],
        },
        {
            "path": "random",
            "title": "RandomSearch Tuner",
            "generate": [
                "keras_tuner.RandomSearch",
            ],
        },
        {
            "path": "bayesian",
            "title": "BayesianOptimization Tuner",
            "generate": [
                "keras_tuner.BayesianOptimization",
            ],
        },
        {
            "path": "hyperband",
            "title": "Hyperband Tuner",
            "generate": [
                "keras_tuner.Hyperband",
            ],
        },
        {
            "path": "sklearn",
            "title": "Sklearn Tuner",
            "generate": [
                "keras_tuner.SklearnTuner",
            ],
        },
    ],
}

KT_API_MASTER = {
    "path": "keras_tuner/",
    "title": "KerasTuner",
    "toc": True,
    "children": [
        {
            "path": "hyperparameters",
            "title": "HyperParameters",
            "generate": [
                "keras_tuner.HyperParameters",
                "keras_tuner.HyperParameters.Boolean",
                "keras_tuner.HyperParameters.Choice",
                "keras_tuner.HyperParameters.Fixed",
                "keras_tuner.HyperParameters.Float",
                "keras_tuner.HyperParameters.Int",
                "keras_tuner.HyperParameters.conditional_scope",
                "keras_tuner.HyperParameters.get",
            ],
        },
        TUNER_MASTER,
        ORACLE_MASTER,
        HYPERMODEL_MASTER,
    ],
}
