FEATURE_INTERACTION_LAYERS_MASTER = {
    "path": "feature_interaction_layers/",
    "title": "Feature Interaction Layers",
    "toc": True,
    "children": [
        {
            "path": "dot_interaction",
            "title": "DotInteraction layer",
            "generate": [
                "keras_rs.layers.DotInteraction",
                "keras_rs.layers.DotInteraction.call",
            ],
        },
        {
            "path": "feature_cross",
            "title": "FeatureCross layer",
            "generate": [
                "keras_rs.layers.FeatureCross",
                "keras_rs.layers.FeatureCross.call",
            ],
        },
    ]
}

RETRIEVAL_LAYERS_MASTER = {
    "path": "retrieval_layers/",
    "title": "Retrieval Layers",
    "toc": True,
    "children": [
        {
            "path": "retrieval",
            "title": "Retrieval layer",
            "generate": [
                "keras_rs.layers.Retrieval",
                "keras_rs.layers.Retrieval.call",
            ],
        },
        {
            "path": "brute_force_retrieval",
            "title": "BruteForceRetrieval layer",
            "generate": [
                "keras_rs.layers.BruteForceRetrieval",
                "keras_rs.layers.BruteForceRetrieval.call",
            ],
        },
        # {
        #     "path": "tpu_approximate_retrieval",
        #     "title": "TPUApproximateRetrieval layer",
        #     "generate": [
        #         "keras_rs.layers.TPUApproximateRetrieval",
        #         "keras_rs.layers.TPUApproximateRetrieval.call",
        #     ],
        # },
        {
            "path": "hard_negative_mining",
            "title": "HardNegativeMining layer",
            "generate": [
                "keras_rs.layers.HardNegativeMining",
                "keras_rs.layers.HardNegativeMining.call",
            ],
        },
        {
            "path": "remove_accidental_hits",
            "title": "RemoveAccidentalHits layer",
            "generate": [
                "keras_rs.layers.RemoveAccidentalHits",
                "keras_rs.layers.RemoveAccidentalHits.call",
            ],
        },
        {
            "path": "sampling_probability_correction",
            "title": "SamplingProbabilityCorrection layer",
            "generate": [
                "keras_rs.layers.SamplingProbabilityCorrection",
                "keras_rs.layers.SamplingProbabilityCorrection.call",
            ],
        },
    ]
}

# LAYERS_MASTER = {
#     "path": "layers/",
#     "title": "Layers",
#     "toc": True,
#     "children": [
#         {
#             "path": "distributed_embedding",
#             "title": "DistributedEmbedding layer",
#             "generate": [
#                 "keras_rs.layers.DistributedEmbedding",
#                 "keras_rs.layers.DistributedEmbedding.call",
#             ],
#         },
#         {
#             "path": "frequency_estimator",
#             "title": "FrequencyEstimator layer",
#             "generate": [
#                 "keras_rs.layers.FrequencyEstimator",
#                 "keras_rs.layers.FrequencyEstimator.call",
#             ],
#         },
#     ]
# }

LOSSES_MASTER = {
    "path": "losses/",
    "title": "Losses",
    "toc": True,
    "children": [
        {
            "path": "pairwise_hinge_loss",
            "title": "PairwiseHingeLoss",
            "generate": [
                "keras_rs.losses.PairwiseHingeLoss",
                "keras_rs.losses.PairwiseHingeLoss.call",
            ],
        },
        {
            "path": "pairwise_logistic_loss",
            "title": "PairwiseLogisticLoss",
            "generate": [
                "keras_rs.losses.PairwiseLogisticLoss",
                "keras_rs.losses.PairwiseLogisticLoss.call",
            ],
        },
        {
            "path": "pairwise_mean_squared_error",
            "title": "PairwiseMeanSquaredError",
            "generate": [
                "keras_rs.losses.PairwiseMeanSquaredError",
                "keras_rs.losses.PairwiseMeanSquaredError.call",
            ],
        },
        {
            "path": "pairwise_soft_zero_one_loss",
            "title": "PairwiseSoftZeroOneLoss",
            "generate": [
                "keras_rs.losses.PairwiseSoftZeroOneLoss",
                "keras_rs.losses.PairwiseSoftZeroOneLoss.call",
            ],
        },
        # {
        #     "path": "list_mle_loss",
        #     "title": "ListMLELoss",
        #     "generate": [
        #         "keras_rs.losses.ListMLELoss",
        #         "keras_rs.losses.ListMLELoss.call",
        #     ],
        # },
    ]
}

METRICS_MASTER = {
    "path": "metrics/",
    "title": "Metrics",
    "toc": True,
    "children": [
        {
            "path": "mean_reciprocal_rank",
            "title": "MeanReciprocalRank metric",
            "generate": [
                "keras_rs.metrics.MeanReciprocalRank",
            ],
        },
        {
            "path": "mean_average_precision",
            "title": "MeanAveragePrecision metric",
            "generate": [
                "keras_rs.metrics.MeanAveragePrecision",
            ],
        },
        # {
        #     "path": "dcg",
        #     "title": "DCG metric",
        #     "generate": [
        #         "keras_rs.metrics.DCG",
        #     ],
        # },
        # {
        #     "path": "ndcg",
        #     "title": "nDCG metric",
        #     "generate": [
        #         "keras_rs.metrics.NDCG",
        #     ],
        # },
    ]
}

RS_API_MASTER = {
    "path": "api/",
    "title": "API documentation",
    "toc": True,
    "children": [
        # LAYERS_MASTER,
        FEATURE_INTERACTION_LAYERS_MASTER,
        RETRIEVAL_LAYERS_MASTER,
        LOSSES_MASTER,
        METRICS_MASTER,
    ],
}

RS_EXAMPLES_MASTER = {
    "path": "examples/",
    "title": "Examples",
    "toc": True,
    "children": [
        {
            "path": "basic_retrieval",
            "title": "Recommending movies: retrieval",
        },
        {
            "path": "basic_ranking",
            "title": "Recommending movies: ranking",
        },
        {
            "path": "data_parallel_retrieval",
            "title": "Recommending movies: retrieval with data parallel training",
        },
        {
            "path": "sequential_retrieval",
            "title": (
                "Recommending movies: retrieval using a sequential model "
                "[GRU4Rec]",
            )
        },
        {
            "path": "scann",
            "title": (
                "Faster retrieval with Scalable Nearest Neighbours (ScANN)"
            )
        },
        {
            "path": "multi_task",
            "title": "Multi-task recommenders: retrieval + ranking",
        },
        {
            "path": "deep_recommender",
            "title": "Deep Recommenders",
        },
        {
            "path": "listwise_ranking",
            "title": "List-wise ranking",
        },
        {
            "path": "dcn",
            "title": "Ranking with Deep and Cross Networks (DCN)",
        },
        {
            "path": "sas_rec",
            "title": (
                "Retrieval using a Transformer-based sequential model [SasRec]"
            )
        },
    ],
}

RS_MASTER = {
    "path": "keras_rs/",
    "title": "KerasRS",
    "children": [
        RS_API_MASTER,
        RS_EXAMPLES_MASTER,
    ],
}
