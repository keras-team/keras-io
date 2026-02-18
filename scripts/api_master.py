API_MASTER = {
    "path": "api/",
    "title": "Keras 3 API documentation",
    "toc": True,
    "children": [
        {
            "path": "models/",
            "title": "Models API",
            "toc": True,
            "children": [
                {
                    "path": "model",
                    "title": "The Model class",
                    "generate": [
                        "keras.Model",
                        "keras.Model.summary",
                        "keras.Model.get_layer",
                        "keras.Model.get_quantization_layer_structure",
                        "keras.Model.get_state_tree",
                        "keras.Model.set_state_tree",
                        "keras.Model.quantize",
                    ],
                    "aliases": {
                        "keras.Model": ["keras.models.Model"]
                    }
                },
                {
                    "path": "sequential",
                    "title": "The Sequential class",
                    "generate": [
                        "keras.Sequential",
                        "keras.Sequential.add",
                        "keras.Sequential.pop",
                    ],
                    "aliases": {
                        "keras.Sequential": ["keras.models.Sequential"]
                    }
                },
                {
                    "path": "model_training_apis",
                    "title": "Model training APIs",
                    "generate": [
                        "keras.Model.compile",
                        "keras.Model.fit",
                        "keras.Model.evaluate",
                        "keras.Model.predict",
                        "keras.Model.train_on_batch",
                        "keras.Model.test_on_batch",
                        "keras.Model.predict_on_batch",
                    ],
                },
                {
                    "path": "model_saving_apis/",
                    "title": "Saving & serialization",
                    "toc": True,
                    "children": [
                        {
                            "path": "model_saving_and_loading",
                            "title": "Whole model saving & loading",
                            "generate": [
                                "keras.Model.save",
                                "keras.saving.save_model",
                                "keras.saving.load_model",
                            ],
                            "aliases": {
                                "keras.saving.save_model": ["keras.models.save_model"],
                                "keras.saving.load_model": ["keras.models.load_model"]
                            }
                        },
                        {
                            "path": "weights_saving_and_loading",
                            "title": "Weights-only saving & loading",
                            "generate": [
                                "keras.Model.save_weights",
                                "keras.Model.load_weights",
                            ],
                        },
                        {
                            "path": "model_config_serialization",
                            "title": "Model config serialization",
                            "generate": [
                                "keras.Model.get_config",
                                "keras.Model.from_config",
                                "keras.models.clone_model",
                                "keras.models.model_from_json",
                            ],
                        },
                        {
                            "path": "export",
                            "title": "Model export for inference",
                            "generate": [
                                "keras.Model.export",
                                "keras.export.ExportArchive",
                                "keras.export.ExportArchive.add_endpoint",
                                "keras.export.ExportArchive.add_variable_collection",
                                "keras.export.ExportArchive.track",
                                "keras.export.ExportArchive.write_out",
                            ],
                        },
                        {
                            "path": "serialization_utils",
                            "title": "Serialization utilities",
                            "generate": [
                                "keras.saving.serialize_keras_object",
                                "keras.saving.deserialize_keras_object",
                                "keras.saving.custom_object_scope",
                                "keras.saving.get_custom_objects",
                                "keras.saving.register_keras_serializable",
                            ],
                        },
                        {
                            "path": "keras_file_editor",
                            "title": "Keras weights file editor",
                            "generate": [
                                "keras.saving.KerasFileEditor",
                                "keras.saving.KerasFileEditor.summary",
                                "keras.saving.KerasFileEditor.compare",
                                "keras.saving.KerasFileEditor.save",
                                "keras.saving.KerasFileEditor.rename_object",
                                "keras.saving.KerasFileEditor.delete_object",
                                "keras.saving.KerasFileEditor.add_object",
                                "keras.saving.KerasFileEditor.delete_weight",
                                "keras.saving.KerasFileEditor.add_weights",
                            ],
                        },
                    ],
                },
                {
                    "path": "distillation/",
                    "title": "Knowledge distillation",
                    "toc": True,
                    "children": [
                        {
                            "path": "distiller",
                            "title": "Distiller model",
                            "generate": [
                                "keras.distillation.Distiller",
                            ],
                        },
                        {
                            "path": "distillation_loss",
                            "title": "Base distillation loss",
                            "generate": [
                                "keras.distillation.DistillationLoss",
                            ],
                        },
                        {
                            "path": "logits_distillation",
                            "title": "Logits distillation loss",
                            "generate": [
                                "keras.distillation.LogitsDistillation",
                            ],
                        },
                        {
                            "path": "feature_distillation",
                            "title": "Feature distillation loss",
                            "generate": [
                                "keras.distillation.FeatureDistillation",
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "path": "core/",
            "title": "Core API",
            "toc": True,
            "children": [
                {
                    "path": "core_classes",
                    "title": "Core classes",
                    "generate": [
                        "keras.KerasTensor",
                        "keras.Operation",
                        "keras.Variable",
                    ],
                },
            ],
        },
        {
            "path": "layers/",
            "title": "Layers API",
            "toc": True,
            "children": [
                {
                    "path": "base_layer",
                    "title": "The base Layer class",
                    "generate": [
                        "keras.Layer",
                        "keras.layers.Layer.weights",
                        "keras.layers.Layer.trainable_weights",
                        "keras.layers.Layer.non_trainable_weights",
                        "keras.layers.Layer.add_weight",
                        "keras.layers.Layer.trainable",
                        "keras.layers.Layer.get_weights",
                        "keras.layers.Layer.set_weights",
                        "keras.layers.Layer.get_config",
                        "keras.layers.Layer.add_loss",
                        "keras.layers.Layer.losses",
                    ],
                    "aliases": {
                        "keras.Layer": ["keras.layers.Layer"]
                    }
                },
                {
                    "path": "activations",
                    "title": "Layer activations",
                    "generate": [
                        "keras.activations.celu",
                        "keras.activations.elu",
                        "keras.activations.exponential",
                        "keras.activations.gelu",
                        "keras.activations.glu",
                        "keras.activations.hard_shrink",
                        "keras.activations.hard_sigmoid",
                        "keras.activations.hard_silu",
                        "keras.activations.hard_swish",
                        "keras.activations.hard_tanh",
                        "keras.activations.leaky_relu",
                        "keras.activations.linear",
                        "keras.activations.log_sigmoid",
                        "keras.activations.log_softmax",
                        "keras.activations.mish",
                        "keras.activations.relu",
                        "keras.activations.relu6",
                        "keras.activations.selu",
                        "keras.activations.sigmoid",
                        "keras.activations.silu",
                        "keras.activations.softmax",
                        "keras.activations.soft_shrink",
                        "keras.activations.softplus",
                        "keras.activations.softsign",
                        "keras.activations.sparse_plus",
                        "keras.activations.sparse_sigmoid",
                        "keras.activations.sparsemax",
                        "keras.activations.squareplus",
                        "keras.activations.swish",
                        "keras.activations.tanh",
                        "keras.activations.tanh_shrink",
                        "keras.activations.threshold",
                    ],
                    "aliases": {
                        "keras.activations.silu": ["keras.activations.swish"],
                        "keras.activations.hard_silu": ["keras.activations.hard_swish"]
                    }
                },
                {
                    "path": "initializers",
                    "title": "Layer weight initializers",
                    "generate": [
                        "keras.Initializer",
                        "keras.initializers.RandomNormal",
                        "keras.initializers.RandomUniform",
                        "keras.initializers.TruncatedNormal",
                        "keras.initializers.Zeros",
                        "keras.initializers.Ones",
                        "keras.initializers.GlorotNormal",
                        "keras.initializers.GlorotUniform",
                        "keras.initializers.HeNormal",
                        "keras.initializers.HeUniform",
                        "keras.initializers.Orthogonal",
                        "keras.initializers.Constant",
                        "keras.initializers.VarianceScaling",
                        "keras.initializers.LecunNormal",
                        "keras.initializers.LecunUniform",
                        "keras.initializers.IdentityInitializer",
                    ],
                    "aliases": {
                        "keras.Initializer": ["keras.initializers.Initializer"],
                        "keras.initializers.Zeros": ["keras.initializers.zeros"],
                        "keras.initializers.Ones": ["keras.initializers.ones"],
                        "keras.initializers.Constant": ["keras.initializers.constant"],
                        "keras.initializers.HeNormal": ["keras.initializers.he_normal"],
                        "keras.initializers.HeUniform": ["keras.initializers.he_uniform"],
                        "keras.initializers.LecunNormal": ["keras.initializers.lecun_normal"],
                        "keras.initializers.LecunUniform": ["keras.initializers.lecun_uniform"],
                        "keras.initializers.GlorotNormal": ["keras.initializers.glorot_normal"],
                        "keras.initializers.GlorotUniform": ["keras.initializers.glorot_uniform"],
                        "keras.initializers.RandomNormal": ["keras.initializers.random_normal"],
                        "keras.initializers.RandomUniform": ["keras.initializers.random_uniform"],
                        "keras.initializers.TruncatedNormal": ["keras.initializers.truncated_normal"],
                        "keras.initializers.VarianceScaling": ["keras.initializers.variance_scaling"],
                        "keras.initializers.Orthogonal": ["keras.initializers.OrthogonalInitializer", "keras.initializers.orthogonal"],
                        "keras.initializers.IdentityInitializer": ["keras.initializers.Identity", "keras.initializers.identity"]
                    }
                },
                {
                    "path": "regularizers",
                    "title": "Layer weight regularizers",
                    "generate": [
                        "keras.Regularizer",
                        "keras.regularizers.L1",
                        "keras.regularizers.L2",
                        "keras.regularizers.L1L2",
                        "keras.regularizers.OrthogonalRegularizer",
                    ],
                    "aliases": {
                        "keras.Regularizer": ["keras.regularizers.Regularizer"],
                        "keras.regularizers.L1": ["keras.regularizers.l1"],
                        "keras.regularizers.L2": ["keras.regularizers.l2"],
                        "keras.regularizers.L1L2": ["keras.regularizers.l1_l2"],
                        "keras.regularizers.OrthogonalRegularizer": ["keras.regularizers.orthogonal_regularizer"]
                    }
                },
                {
                    "path": "constraints",
                    "title": "Layer weight constraints",
                    "generate": [
                        "keras.constraints.MaxNorm",
                        "keras.constraints.MinMaxNorm",
                        "keras.constraints.NonNeg",
                        "keras.constraints.UnitNorm",
                    ],
                    "aliases": {
                        "keras.constraints.MaxNorm": ["keras.constraints.max_norm"],
                        "keras.constraints.MinMaxNorm": ["keras.constraints.min_max_norm"],
                        "keras.constraints.NonNeg": ["keras.constraints.non_neg"],
                        "keras.constraints.UnitNorm": ["keras.constraints.unit_norm"]
                    }
                },
                {
                    "path": "core_layers/",
                    "title": "Core layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "input",
                            "title": "Input object",
                            "generate": [
                                "keras.Input",
                            ],
                            "aliases": {
                                "keras.Input": ["keras.layers.Input"]
                            }
                        },
                        {
                            "path": "input_spec",
                            "title": "InputSpec object",
                            "generate": [
                                "keras.InputSpec",
                            ],
                            "aliases": {
                                "keras.InputSpec": ["keras.layers.InputSpec"]
                            }
                        },
                        {
                            "path": "dense",
                            "title": "Dense layer",
                            "generate": ["keras.layers.Dense"],
                        },
                        {
                            "path": "einsum_dense",
                            "title": "EinsumDense layer",
                            "generate": ["keras.layers.EinsumDense"],
                        },
                        {
                            "path": "activation",
                            "title": "Activation layer",
                            "generate": ["keras.layers.Activation"],
                        },
                        {
                            "path": "embedding",
                            "title": "Embedding layer",
                            "generate": [
                                "keras.layers.Embedding",
                                "keras.layers.ReversibleEmbedding",
                            ],
                        },
                        {
                            "path": "masking",
                            "title": "Masking layer",
                            "generate": ["keras.layers.Masking"],
                        },
                        {
                            "path": "lambda",
                            "title": "Lambda layer",
                            "generate": ["keras.layers.Lambda"],
                        },
                        {
                            "path": "identity",
                            "title": "Identity layer",
                            "generate": ["keras.layers.Identity"],
                        },
                    ],
                },
                {
                    "path": "convolution_layers/",
                    "title": "Convolution layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "convolution1d",
                            "title": "Conv1D layer",
                            "generate": [
                                "keras.layers.Conv1D",
                            ],
                            "aliases": {
                                "keras.layers.Conv1D": ["keras.layers.Convolution1D"]
                            }
                        },
                        {
                            "path": "convolution2d",
                            "title": "Conv2D layer",
                            "generate": [
                                "keras.layers.Conv2D",
                            ],
                            "aliases": {
                                "keras.layers.Conv2D": ["keras.layers.Convolution2D"]
                            }
                        },
                        {
                            "path": "convolution3d",
                            "title": "Conv3D layer",
                            "generate": [
                                "keras.layers.Conv3D",
                            ],
                            "aliases": {
                                "keras.layers.Conv3D": ["keras.layers.Convolution3D"]
                            }
                        },
                        {
                            "path": "separable_convolution1d",
                            "title": "SeparableConv1D layer",
                            "generate": ["keras.layers.SeparableConv1D"],
                        },
                        {
                            "path": "separable_convolution2d",
                            "title": "SeparableConv2D layer",
                            "generate": ["keras.layers.SeparableConv2D"],
                        },
                        {
                            "path": "depthwise_convolution1d",
                            "title": "DepthwiseConv1D layer",
                            "generate": ["keras.layers.DepthwiseConv1D"],
                        },
                        {
                            "path": "depthwise_convolution2d",
                            "title": "DepthwiseConv2D layer",
                            "generate": ["keras.layers.DepthwiseConv2D"],
                        },
                        {
                            "path": "convolution1d_transpose",
                            "title": "Conv1DTranspose layer",
                            "generate": ["keras.layers.Conv1DTranspose"],
                        },
                        {
                            "path": "convolution2d_transpose",
                            "title": "Conv2DTranspose layer",
                            "generate": ["keras.layers.Conv2DTranspose"],
                        },
                        {
                            "path": "convolution3d_transpose",
                            "title": "Conv3DTranspose layer",
                            "generate": ["keras.layers.Conv3DTranspose"],
                        },
                    ],
                },
                {
                    "path": "pooling_layers/",
                    "title": "Pooling layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "max_pooling1d",
                            "title": "MaxPooling1D layer",
                            "generate": [
                                "keras.layers.MaxPooling1D",
                            ],
                            "aliases": {
                                "keras.layers.MaxPooling1D": ["keras.layers.MaxPool1D"]
                            }
                        },
                        {
                            "path": "max_pooling2d",
                            "title": "MaxPooling2D layer",
                            "generate": [
                                "keras.layers.MaxPooling2D",
                            ],
                            "aliases": {
                                "keras.layers.MaxPooling2D": ["keras.layers.MaxPool2D"]
                            }
                        },
                        {
                            "path": "max_pooling3d",
                            "title": "MaxPooling3D layer",
                            "generate": [
                                "keras.layers.MaxPooling3D",
                            ],
                            "aliases": {
                                "keras.layers.MaxPooling3D": ["keras.layers.MaxPool3D"]
                            }
                        },
                        {
                            "path": "average_pooling1d",
                            "title": "AveragePooling1D layer",
                            "generate": [
                                "keras.layers.AveragePooling1D",
                            ],
                            "aliases": {
                                "keras.layers.AveragePooling1D": ["keras.layers.AvgPool1D"]
                            }
                        },
                        {
                            "path": "average_pooling2d",
                            "title": "AveragePooling2D layer",
                            "generate": [
                                "keras.layers.AveragePooling2D",
                            ],
                            "aliases": {
                                "keras.layers.AveragePooling2D": ["keras.layers.AvgPool2D"]
                            }
                        },
                        {
                            "path": "average_pooling3d",
                            "title": "AveragePooling3D layer",
                            "generate": [
                                "keras.layers.AveragePooling3D",
                            ],
                            "aliases": {
                                "keras.layers.AveragePooling3D": ["keras.layers.AvgPool3D"]
                            }
                        },
                        {
                            "path": "global_max_pooling1d",
                            "title": "GlobalMaxPooling1D layer",
                            "generate": ["keras.layers.GlobalMaxPooling1D"],
                        },
                        {
                            "path": "global_max_pooling2d",
                            "title": "GlobalMaxPooling2D layer",
                            "generate": ["keras.layers.GlobalMaxPooling2D"],
                        },
                        {
                            "path": "global_max_pooling3d",
                            "title": "GlobalMaxPooling3D layer",
                            "generate": ["keras.layers.GlobalMaxPooling3D"],
                        },
                        {
                            "path": "global_average_pooling1d",
                            "title": "GlobalAveragePooling1D layer",
                            "generate": ["keras.layers.GlobalAveragePooling1D"],
                        },
                        {
                            "path": "global_average_pooling2d",
                            "title": "GlobalAveragePooling2D layer",
                            "generate": ["keras.layers.GlobalAveragePooling2D"],
                        },
                        {
                            "path": "global_average_pooling3d",
                            "title": "GlobalAveragePooling3D layer",
                            "generate": ["keras.layers.GlobalAveragePooling3D"],
                        },
                        {
                            "path": "adaptive_average_pooling1d",
                            "title": "AdaptiveAveragePooling1D layer",
                            "generate": ["keras.layers.AdaptiveAveragePooling1D"],
                        },
                        {
                            "path": "adaptive_average_pooling2d",
                            "title": "AdaptiveAveragePooling2D layer",
                            "generate": ["keras.layers.AdaptiveAveragePooling2D"],
                        },
                        {
                            "path": "adaptive_average_pooling3d",
                            "title": "AdaptiveAveragePooling3D layer",
                            "generate": ["keras.layers.AdaptiveAveragePooling3D"],
                        },
                        {
                            "path": "adaptive_max_pooling1d",
                            "title": "AdaptiveMaxPooling1D layer",
                            "generate": ["keras.layers.AdaptiveMaxPooling1D"],
                        },
                        {
                            "path": "adaptive_max_pooling2d",
                            "title": "AdaptiveMaxPooling2D layer",
                            "generate": ["keras.layers.AdaptiveMaxPooling2D"],
                        },
                        {
                            "path": "adaptive_max_pooling3d",
                            "title": "AdaptiveMaxPooling3D layer",
                            "generate": ["keras.layers.AdaptiveMaxPooling3D"],
                        },
                    ],
                },
                {
                    "path": "recurrent_layers/",
                    "title": "Recurrent layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "lstm",
                            "title": "LSTM layer",
                            "generate": ["keras.layers.LSTM"],
                        },
                        {
                            "path": "lstm_cell",
                            "title": "LSTM cell layer",
                            "generate": ["keras.layers.LSTMCell"],
                        },
                        {
                            "path": "gru",
                            "title": "GRU layer",
                            "generate": ["keras.layers.GRU"],
                        },
                        {
                            "path": "gru_cell",
                            "title": "GRU Cell layer",
                            "generate": ["keras.layers.GRUCell"],
                        },
                        {
                            "path": "simple_rnn",
                            "title": "SimpleRNN layer",
                            "generate": ["keras.layers.SimpleRNN"],
                        },
                        {
                            "path": "time_distributed",
                            "title": "TimeDistributed layer",
                            "generate": ["keras.layers.TimeDistributed"],
                        },
                        {
                            "path": "bidirectional",
                            "title": "Bidirectional layer",
                            "generate": ["keras.layers.Bidirectional"],
                        },
                        {
                            "path": "conv_lstm1d",
                            "title": "ConvLSTM1D layer",
                            "generate": ["keras.layers.ConvLSTM1D"],
                        },
                        {
                            "path": "conv_lstm2d",
                            "title": "ConvLSTM2D layer",
                            "generate": ["keras.layers.ConvLSTM2D"],
                        },
                        {
                            "path": "conv_lstm3d",
                            "title": "ConvLSTM3D layer",
                            "generate": ["keras.layers.ConvLSTM3D"],
                        },
                        {
                            "path": "rnn",
                            "title": "Base RNN layer",
                            "generate": ["keras.layers.RNN"],
                        },
                        {
                            "path": "simple_rnn_cell",
                            "title": "Simple RNN cell layer",
                            "generate": ["keras.layers.SimpleRNNCell"],
                        },
                        {
                            "path": "stacked_rnn_cell",
                            "title": "Stacked RNN cell layer",
                            "generate": ["keras.layers.StackedRNNCells"],
                        },
                    ],
                },
                {
                    "path": "preprocessing_layers/",
                    "title": "Preprocessing layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "text/",
                            "title": "Text preprocessing",
                            "toc": True,
                            "children": [
                                {
                                    "path": "text_vectorization",
                                    "title": "TextVectorization layer",
                                    "generate": ["keras.layers.TextVectorization"],
                                },
                            ],
                        },
                        {
                            "path": "numerical/",
                            "title": "Numerical features preprocessing layers",
                            "toc": True,
                            "children": [
                                {
                                    "path": "normalization",
                                    "title": "Normalization layer",
                                    "generate": ["keras.layers.Normalization"],
                                },
                                {
                                    "path": "spectral_normalization",
                                    "title": "Spectral Normalization layer",
                                    "generate": ["keras.layers.SpectralNormalization"],
                                },
                                {
                                    "path": "discretization",
                                    "title": "Discretization layer",
                                    "generate": ["keras.layers.Discretization"],
                                },
                            ],
                        },
                        {
                            "path": "categorical/",
                            "title": "Categorical features preprocessing layers",
                            "toc": True,
                            "children": [
                                {
                                    "path": "category_encoding",
                                    "title": "CategoryEncoding layer",
                                    "generate": ["keras.layers.CategoryEncoding"],
                                },
                                {
                                    "path": "hashing",
                                    "title": "Hashing layer",
                                    "generate": ["keras.layers.Hashing"],
                                },
                                {
                                    "path": "hashed_crossing",
                                    "title": "HashedCrossing layer",
                                    "generate": ["keras.layers.HashedCrossing"],
                                },
                                {
                                    "path": "string_lookup",
                                    "title": "StringLookup layer",
                                    "generate": ["keras.layers.StringLookup"],
                                },
                                {
                                    "path": "integer_lookup",
                                    "title": "IntegerLookup layer",
                                    "generate": ["keras.layers.IntegerLookup"],
                                },
                            ],
                        },
                        {
                            "path": "image_preprocessing/",
                            "title": "Image preprocessing layers",
                            "toc": True,
                            "children": [
                                {
                                    "path": "resizing",
                                    "title": "Resizing layer",
                                    "generate": ["keras.layers.Resizing"],
                                },
                                {
                                    "path": "rescaling",
                                    "title": "Rescaling layer",
                                    "generate": ["keras.layers.Rescaling"],
                                },
                                {
                                    "path": "center_crop",
                                    "title": "CenterCrop layer",
                                    "generate": ["keras.layers.CenterCrop"],
                                },
                                {
                                    "path": "auto_constrast",
                                    "title": "AutoContrast layer",
                                    "generate": ["keras.layers.AutoContrast"],
                                },
                            ],
                        },
                        {
                            "path": "image_augmentation/",
                            "title": "Image augmentation layers",
                            "toc": True,
                            "children": [
                                {
                                    "path": "aug_mix",
                                    "title": "AugMix layer",
                                    "generate": ["keras.layers.AugMix"],
                                },
                                {
                                    "path": "cut_mix",
                                    "title": "CutMix layer",
                                    "generate": ["keras.layers.CutMix"],
                                },
                                {
                                    "path": "equalization",
                                    "title": "Equalization layer",
                                    "generate": ["keras.layers.Equalization"],
                                },
                                {
                                    "path": "max_num_bounding_boxes",
                                    "title": "MaxNumBoundingBoxes layer",
                                    "generate": ["keras.layers.MaxNumBoundingBoxes"],
                                },
                                {
                                    "path": "mix_up",
                                    "title": "MixUp layer",
                                    "generate": ["keras.layers.MixUp"],
                                },
                                {
                                    "path": "pipeline",
                                    "title": "Pipeline layer",
                                    "generate": ["keras.layers.Pipeline"],
                                },
                                {
                                    "path": "rand_augment",
                                    "title": "RandAugment layer",
                                    "generate": ["keras.layers.RandAugment"],
                                },
                                {
                                    "path": "random_brightness",
                                    "title": "RandomBrightness layer",
                                    "generate": ["keras.layers.RandomBrightness"],
                                },
                                {
                                    "path": "random_color_degeneration",
                                    "title": "RandomColorDegeneration layer",
                                    "generate": [
                                        "keras.layers.RandomColorDegeneration"
                                    ],
                                },
                                {
                                    "path": "random_color_jitter",
                                    "title": "RandomColorJitter layer",
                                    "generate": ["keras.layers.RandomColorJitter"],
                                },
                                {
                                    "path": "random_contrast",
                                    "title": "RandomContrast layer",
                                    "generate": ["keras.layers.RandomContrast"],
                                },
                                {
                                    "path": "random_crop",
                                    "title": "RandomCrop layer",
                                    "generate": ["keras.layers.RandomCrop"],
                                },
                                {
                                    "path": "random_elastic_transform",
                                    "title": "RandomElasticTransform layer",
                                    "generate": ["keras.layers.RandomElasticTransform"],
                                },
                                {
                                    "path": "random_erasing",
                                    "title": "RandomErasing layer",
                                    "generate": ["keras.layers.RandomErasing"],
                                },
                                {
                                    "path": "random_flip",
                                    "title": "RandomFlip layer",
                                    "generate": ["keras.layers.RandomFlip"],
                                },
                                {
                                    "path": "random_gaussian_blur",
                                    "title": "RandomGaussianBlur layer",
                                    "generate": ["keras.layers.RandomGaussianBlur"],
                                },
                                {
                                    "path": "random_grayscale",
                                    "title": "RandomGrayscale layer",
                                    "generate": ["keras.layers.RandomGrayscale"],
                                },
                                {
                                    "path": "random_hue",
                                    "title": "RandomHue layer",
                                    "generate": ["keras.layers.RandomHue"],
                                },
                                {
                                    "path": "random_invert",
                                    "title": "RandomInvert layer",
                                    "generate": ["keras.layers.RandomInvert"],
                                },
                                {
                                    "path": "random_perspective",
                                    "title": "RandomPerspective layer",
                                    "generate": ["keras.layers.RandomPerspective"],
                                },
                                {
                                    "path": "random_posterization",
                                    "title": "RandomPosterization layer",
                                    "generate": ["keras.layers.RandomPosterization"],
                                },
                                {
                                    "path": "random_rotation",
                                    "title": "RandomRotation layer",
                                    "generate": ["keras.layers.RandomRotation"],
                                },
                                {
                                    "path": "random_saturation",
                                    "title": "RandomSaturation layer",
                                    "generate": ["keras.layers.RandomSaturation"],
                                },
                                {
                                    "path": "random_sharpness",
                                    "title": "RandomSharpness layer",
                                    "generate": ["keras.layers.RandomSharpness"],
                                },
                                {
                                    "path": "random_shear",
                                    "title": "RandomShear layer",
                                    "generate": ["keras.layers.RandomShear"],
                                },
                                {
                                    "path": "random_translation",
                                    "title": "RandomTranslation layer",
                                    "generate": ["keras.layers.RandomTranslation"],
                                },
                                {
                                    "path": "random_zoom",
                                    "title": "RandomZoom layer",
                                    "generate": ["keras.layers.RandomZoom"],
                                },
                                {
                                    "path": "solarization",
                                    "title": "Solarization layer",
                                    "generate": ["keras.layers.Solarization"],
                                },
                            ],
                        },
                        {
                            "path": "audio_preprocessing/",
                            "title": "Audio preprocessing layers",
                            "toc": True,
                            "children": [
                                {
                                    "path": "mel_spectrogram",
                                    "title": "MelSpectrogram layer",
                                    "generate": ["keras.layers.MelSpectrogram"],
                                },
                                {
                                    "path": "stft_spectrogram",
                                    "title": "STFTSpectrogram layer",
                                    "generate": ["keras.layers.STFTSpectrogram"],
                                },
                            ],
                        },
                    ],
                },
                {
                    "path": "normalization_layers/",
                    "title": "Normalization layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "batch_normalization",
                            "title": "BatchNormalization layer",
                            "generate": ["keras.layers.BatchNormalization"],
                        },
                        {
                            "path": "layer_normalization",
                            "title": "LayerNormalization layer",
                            "generate": ["keras.layers.LayerNormalization"],
                        },
                        {
                            "path": "unit_normalization",
                            "title": "UnitNormalization layer",
                            "generate": ["keras.layers.UnitNormalization"],
                        },
                        {
                            "path": "group_normalization",
                            "title": "GroupNormalization layer",
                            "generate": ["keras.layers.GroupNormalization"],
                        },
                        {
                            "path": "rms_normalization",
                            "title": "RMSNormalization layer",
                            "generate": ["keras.layers.RMSNormalization"],
                        },
                    ],
                },
                {
                    "path": "regularization_layers/",
                    "title": "Regularization layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "dropout",
                            "title": "Dropout layer",
                            "generate": ["keras.layers.Dropout"],
                        },
                        {
                            "path": "spatial_dropout1d",
                            "title": "SpatialDropout1D layer",
                            "generate": ["keras.layers.SpatialDropout1D"],
                        },
                        {
                            "path": "spatial_dropout2d",
                            "title": "SpatialDropout2D layer",
                            "generate": ["keras.layers.SpatialDropout2D"],
                        },
                        {
                            "path": "spatial_dropout3d",
                            "title": "SpatialDropout3D layer",
                            "generate": ["keras.layers.SpatialDropout3D"],
                        },
                        {
                            "path": "gaussian_dropout",
                            "title": "GaussianDropout layer",
                            "generate": ["keras.layers.GaussianDropout"],
                        },
                        {
                            "path": "alpha_dropout",
                            "title": "AlphaDropout layer",
                            "generate": ["keras.layers.AlphaDropout"],
                        },
                        {
                            "path": "gaussian_noise",
                            "title": "GaussianNoise layer",
                            "generate": ["keras.layers.GaussianNoise"],
                        },
                        {
                            "path": "activity_regularization",
                            "title": "ActivityRegularization layer",
                            "generate": ["keras.layers.ActivityRegularization"],
                        },
                    ],
                },
                {
                    "path": "attention_layers/",
                    "title": "Attention layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "group_query_attention",
                            "title": "GroupQueryAttention",
                            "generate": ["keras.layers.GroupQueryAttention"],
                        },
                        {
                            "path": "multi_head_attention",
                            "title": "MultiHeadAttention layer",
                            "generate": ["keras.layers.MultiHeadAttention"],
                        },
                        {
                            "path": "attention",
                            "title": "Attention layer",
                            "generate": ["keras.layers.Attention"],
                        },
                        {
                            "path": "additive_attention",
                            "title": "AdditiveAttention layer",
                            "generate": ["keras.layers.AdditiveAttention"],
                        },
                    ],
                },
                {
                    "path": "reshaping_layers/",
                    "title": "Reshaping layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "reshape",
                            "title": "Reshape layer",
                            "generate": ["keras.layers.Reshape"],
                        },
                        {
                            "path": "flatten",
                            "title": "Flatten layer",
                            "generate": ["keras.layers.Flatten"],
                        },
                        {
                            "path": "repeat_vector",
                            "title": "RepeatVector layer",
                            "generate": ["keras.layers.RepeatVector"],
                        },
                        {
                            "path": "permute",
                            "title": "Permute layer",
                            "generate": ["keras.layers.Permute"],
                        },
                        {
                            "path": "cropping1d",
                            "title": "Cropping1D layer",
                            "generate": ["keras.layers.Cropping1D"],
                        },
                        {
                            "path": "cropping2d",
                            "title": "Cropping2D layer",
                            "generate": ["keras.layers.Cropping2D"],
                        },
                        {
                            "path": "cropping3d",
                            "title": "Cropping3D layer",
                            "generate": ["keras.layers.Cropping3D"],
                        },
                        {
                            "path": "up_sampling1d",
                            "title": "UpSampling1D layer",
                            "generate": ["keras.layers.UpSampling1D"],
                        },
                        {
                            "path": "up_sampling2d",
                            "title": "UpSampling2D layer",
                            "generate": ["keras.layers.UpSampling2D"],
                        },
                        {
                            "path": "up_sampling3d",
                            "title": "UpSampling3D layer",
                            "generate": ["keras.layers.UpSampling3D"],
                        },
                        {
                            "path": "zero_padding1d",
                            "title": "ZeroPadding1D layer",
                            "generate": ["keras.layers.ZeroPadding1D"],
                        },
                        {
                            "path": "zero_padding2d",
                            "title": "ZeroPadding2D layer",
                            "generate": ["keras.layers.ZeroPadding2D"],
                        },
                        {
                            "path": "zero_padding3d",
                            "title": "ZeroPadding3D layer",
                            "generate": ["keras.layers.ZeroPadding3D"],
                        },
                    ],
                },
                {
                    "path": "merging_layers/",
                    "title": "Merging layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "concatenate",
                            "title": "Concatenate layer",
                            "generate": ["keras.layers.Concatenate"],
                        },
                        {
                            "path": "average",
                            "title": "Average layer",
                            "generate": ["keras.layers.Average"],
                        },
                        {
                            "path": "maximum",
                            "title": "Maximum layer",
                            "generate": ["keras.layers.Maximum"],
                        },
                        {
                            "path": "minimum",
                            "title": "Minimum layer",
                            "generate": ["keras.layers.Minimum"],
                        },
                        {
                            "path": "add",
                            "title": "Add layer",
                            "generate": ["keras.layers.Add"],
                        },
                        {
                            "path": "subtract",
                            "title": "Subtract layer",
                            "generate": ["keras.layers.Subtract"],
                        },
                        {
                            "path": "multiply",
                            "title": "Multiply layer",
                            "generate": ["keras.layers.Multiply"],
                        },
                        {
                            "path": "dot",
                            "title": "Dot layer",
                            "generate": ["keras.layers.Dot"],
                        },
                    ],
                },
                {
                    "path": "activation_layers/",
                    "title": "Activation layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "relu",
                            "title": "ReLU layer",
                            "generate": ["keras.layers.ReLU"],
                        },
                        {
                            "path": "softmax",
                            "title": "Softmax layer",
                            "generate": ["keras.layers.Softmax"],
                        },
                        {
                            "path": "leaky_relu",
                            "title": "LeakyReLU layer",
                            "generate": ["keras.layers.LeakyReLU"],
                        },
                        {
                            "path": "prelu",
                            "title": "PReLU layer",
                            "generate": ["keras.layers.PReLU"],
                        },
                        {
                            "path": "elu",
                            "title": "ELU layer",
                            "generate": ["keras.layers.ELU"],
                        },
                    ],
                },
                {
                    "path": "backend_specific_layers/",
                    "title": "Backend-specific layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "torch_module_wrapper",
                            "title": "TorchModuleWrapper layer",
                            "generate": ["keras.layers.TorchModuleWrapper"],
                        },
                        {
                            "path": "tfsm_layer",
                            "title": "Tensorflow SavedModel layer",
                            "generate": ["keras.layers.TFSMLayer"],
                        },
                        {
                            "path": "jax_layer",
                            "title": "JaxLayer",
                            "generate": ["keras.layers.JaxLayer"],
                        },
                        {
                            "path": "flax_layer",
                            "title": "FlaxLayer",
                            "generate": ["keras.layers.FlaxLayer"],
                        },
                    ],
                },
            ],
        },
        {
            "path": "callbacks/",
            "title": "Callbacks API",
            "toc": True,
            "children": [
                {
                    "path": "base_callback",
                    "title": "Base Callback class",
                    "generate": ["keras.callbacks.Callback"],
                },
                {
                    "path": "model_checkpoint",
                    "title": "ModelCheckpoint",
                    "generate": ["keras.callbacks.ModelCheckpoint"],
                },
                {
                    "path": "backup_and_restore",
                    "title": "BackupAndRestore",
                    "generate": ["keras.callbacks.BackupAndRestore"],
                },
                {
                    "path": "tensorboard",
                    "title": "TensorBoard",
                    "generate": ["keras.callbacks.TensorBoard"],
                },
                {
                    "path": "early_stopping",
                    "title": "EarlyStopping",
                    "generate": ["keras.callbacks.EarlyStopping"],
                },
                {
                    "path": "learning_rate_scheduler",
                    "title": "LearningRateScheduler",
                    "generate": ["keras.callbacks.LearningRateScheduler"],
                },
                {
                    "path": "reduce_lr_on_plateau",
                    "title": "ReduceLROnPlateau",
                    "generate": ["keras.callbacks.ReduceLROnPlateau"],
                },
                {
                    "path": "remote_monitor",
                    "title": "RemoteMonitor",
                    "generate": ["keras.callbacks.RemoteMonitor"],
                },
                {
                    "path": "lambda_callback",
                    "title": "LambdaCallback",
                    "generate": ["keras.callbacks.LambdaCallback"],
                },
                {
                    "path": "terminate_on_nan",
                    "title": "TerminateOnNaN",
                    "generate": ["keras.callbacks.TerminateOnNaN"],
                },
                {
                    "path": "csv_logger",
                    "title": "CSVLogger",
                    "generate": ["keras.callbacks.CSVLogger"],
                },
                {
                    "path": "progbar_logger",
                    "title": "ProgbarLogger",
                    "generate": ["keras.callbacks.ProgbarLogger"],
                },
                {
                    "path": "swap_ema_weights",
                    "title": "SwapEMAWeights",
                    "generate": ["keras.callbacks.SwapEMAWeights"],
                },
                {
                    "path": "callback_utilities",
                    "title": "Callback utilities",
                    "generate": [
                        "keras.callbacks.CallbackList",
                        "keras.callbacks.History",
                        # "keras.callbacks.OrbaxCheckpoint",
                    ],
                },
            ],
        },
        {
            "path": "ops/",  # TODO: improve
            "title": "Ops API",
            "toc": True,
            "children": [
                {
                    "path": "numpy/",
                    "title": "NumPy ops",
                    "generate": [
                        "keras.ops.abs",
                        "keras.ops.absolute",
                        "keras.ops.add",
                        "keras.ops.all",
                        "keras.ops.amax",
                        "keras.ops.amin",
                        "keras.ops.angle",
                        "keras.ops.any",
                        "keras.ops.append",
                        "keras.ops.arange",
                        "keras.ops.arccos",
                        "keras.ops.arccosh",
                        "keras.ops.arcsin",
                        "keras.ops.arcsinh",
                        "keras.ops.arctan",
                        "keras.ops.arctan2",
                        "keras.ops.arctanh",
                        "keras.ops.argmax",
                        "keras.ops.argmin",
                        "keras.ops.argpartition",
                        "keras.ops.argsort",
                        "keras.ops.array",
                        "keras.ops.array_split",
                        "keras.ops.average",
                        "keras.ops.bartlett",
                        "keras.ops.bincount",
                        "keras.ops.bitwise_and",
                        "keras.ops.bitwise_invert",
                        "keras.ops.bitwise_left_shift",
                        "keras.ops.bitwise_not",
                        "keras.ops.bitwise_or",
                        "keras.ops.bitwise_right_shift",
                        "keras.ops.bitwise_xor",
                        "keras.ops.blackman",
                        "keras.ops.broadcast_to",
                        "keras.ops.cbrt",
                        "keras.ops.ceil",
                        "keras.ops.clip",
                        "keras.ops.concatenate",
                        "keras.ops.conj",
                        "keras.ops.conjugate",
                        "keras.ops.copy",
                        "keras.ops.corrcoef",
                        "keras.ops.correlate",
                        "keras.ops.cos",
                        "keras.ops.cosh",
                        "keras.ops.count_nonzero",
                        "keras.ops.cross",
                        "keras.ops.cumprod",
                        "keras.ops.cumsum",
                        "keras.ops.deg2rad",
                        "keras.ops.diag",
                        "keras.ops.diagflat",
                        "keras.ops.diagonal",
                        "keras.ops.diff",
                        "keras.ops.digitize",
                        "keras.ops.divide",
                        "keras.ops.divide_no_nan",
                        "keras.ops.dot",
                        "keras.ops.einsum",
                        "keras.ops.empty",
                        "keras.ops.empty_like",
                        "keras.ops.equal",
                        "keras.ops.exp",
                        "keras.ops.exp2",
                        "keras.ops.expand_dims",
                        "keras.ops.expm1",
                        "keras.ops.eye",
                        "keras.ops.flip",
                        "keras.ops.floor",
                        "keras.ops.floor_divide",
                        "keras.ops.full",
                        "keras.ops.full_like",
                        "keras.ops.gcd",
                        "keras.ops.get_item",
                        "keras.ops.greater",
                        "keras.ops.greater_equal",
                        "keras.ops.hamming",
                        "keras.ops.hanning",
                        "keras.ops.heaviside",
                        "keras.ops.histogram",
                        "keras.ops.hstack",
                        "keras.ops.hypot",
                        "keras.ops.identity",
                        "keras.ops.imag",
                        "keras.ops.inner",
                        "keras.ops.isclose",
                        "keras.ops.isfinite",
                        "keras.ops.isinf",
                        "keras.ops.isin",
                        "keras.ops.isnan",
                        "keras.ops.isneginf",
                        "keras.ops.isposinf",
                        "keras.ops.isreal",
                        "keras.ops.kaiser",
                        "keras.ops.kron",
                        "keras.ops.lcm",
                        "keras.ops.ldexp",
                        "keras.ops.left_shift",
                        "keras.ops.less",
                        "keras.ops.less_equal",
                        "keras.ops.linspace",
                        "keras.ops.log",
                        "keras.ops.log10",
                        "keras.ops.log1p",
                        "keras.ops.log2",
                        "keras.ops.logaddexp",
                        "keras.ops.logaddexp2",
                        "keras.ops.logical_and",
                        "keras.ops.logical_not",
                        "keras.ops.logical_or",
                        "keras.ops.logical_xor",
                        "keras.ops.logspace",
                        "keras.ops.matmul",
                        "keras.ops.max",
                        "keras.ops.maximum",
                        "keras.ops.mean",
                        "keras.ops.median",
                        "keras.ops.meshgrid",
                        "keras.ops.min",
                        "keras.ops.minimum",
                        "keras.ops.mod",
                        "keras.ops.moveaxis",
                        "keras.ops.multiply",
                        "keras.ops.nan_to_num",
                        "keras.ops.ndim",
                        "keras.ops.negative",
                        "keras.ops.nonzero",
                        "keras.ops.norm",
                        "keras.ops.not_equal",
                        "keras.ops.ones",
                        "keras.ops.ones_like",
                        "keras.ops.outer",
                        "keras.ops.pad",
                        "keras.ops.power",
                        "keras.ops.prod",
                        "keras.ops.quantile",
                        "keras.ops.ravel",
                        "keras.ops.real",
                        "keras.ops.reciprocal",
                        "keras.ops.repeat",
                        "keras.ops.reshape",
                        "keras.ops.right_shift",
                        "keras.ops.roll",
                        "keras.ops.rot90",
                        "keras.ops.round",
                        "keras.ops.searchsorted",
                        "keras.ops.select",
                        "keras.ops.sign",
                        "keras.ops.signbit",
                        "keras.ops.sin",
                        "keras.ops.sinh",
                        "keras.ops.size",
                        "keras.ops.slogdet",
                        "keras.ops.sort",
                        "keras.ops.split",
                        "keras.ops.sqrt",
                        "keras.ops.square",
                        "keras.ops.squeeze",
                        "keras.ops.stack",
                        "keras.ops.std",
                        "keras.ops.subtract",
                        "keras.ops.sum",
                        "keras.ops.swapaxes",
                        "keras.ops.take",
                        "keras.ops.take_along_axis",
                        "keras.ops.tan",
                        "keras.ops.tanh",
                        "keras.ops.tensordot",
                        "keras.ops.tile",
                        "keras.ops.trace",
                        "keras.ops.trapezoid",
                        "keras.ops.transpose",
                        "keras.ops.tri",
                        "keras.ops.tril",
                        "keras.ops.triu",
                        "keras.ops.true_divide",
                        "keras.ops.trunc",
                        "keras.ops.unravel_index",
                        "keras.ops.vander",
                        "keras.ops.var",
                        "keras.ops.vdot",
                        "keras.ops.vectorize",
                        "keras.ops.view",
                        "keras.ops.vstack",
                        "keras.ops.where",
                        "keras.ops.zeros",
                        "keras.ops.zeros_like",
                    ],
                },
                {
                    "path": "nn/",
                    "title": "NN ops",
                    "generate": [
                        "keras.ops.adaptive_average_pool",
                        "keras.ops.adaptive_max_pool",
                        "keras.ops.average_pool",
                        "keras.ops.batch_normalization",
                        "keras.ops.binary_crossentropy",
                        "keras.ops.categorical_crossentropy",
                        "keras.ops.conv",
                        "keras.ops.conv_transpose",
                        "keras.ops.ctc_decode",
                        "keras.ops.ctc_loss",
                        "keras.ops.depthwise_conv",
                        "keras.ops.dot_product_attention",
                        "keras.ops.elu",
                        "keras.ops.gelu",
                        "keras.ops.hard_sigmoid",
                        "keras.ops.leaky_relu",
                        "keras.ops.log_sigmoid",
                        "keras.ops.log_softmax",
                        "keras.ops.max_pool",
                        "keras.ops.moments",
                        "keras.ops.multi_hot",
                        "keras.ops.normalize",
                        "keras.ops.one_hot",
                        "keras.ops.psnr",
                        "keras.ops.relu",
                        "keras.ops.relu6",
                        "keras.ops.selu",
                        "keras.ops.separable_conv",
                        "keras.ops.sigmoid",
                        "keras.ops.silu",
                        "keras.ops.hard_silu",
                        "keras.ops.softmax",
                        "keras.ops.softplus",
                        "keras.ops.softsign",
                        "keras.ops.sparse_categorical_crossentropy",
                        "keras.ops.swish",
                        "keras.ops.hard_swish",
                        "keras.ops.celu",
                        "keras.ops.sparsemax",
                        "keras.ops.squareplus",
                        "keras.ops.sparse_plus",
                        "keras.ops.soft_shrink",
                        "keras.ops.threshold",
                        "keras.ops.glu",
                        "keras.ops.tanh_shrink",
                        "keras.ops.hard_tanh",
                        "keras.ops.hard_shrink",
                    ],
                },
                {
                    "path": "linalg/",
                    "title": "Linear algebra ops",
                    "generate": [
                        "keras.ops.cholesky",
                        "keras.ops.det",
                        "keras.ops.eig",
                        "keras.ops.eigh",
                        "keras.ops.inv",
                        "keras.ops.logdet",
                        "keras.ops.lstsq",
                        "keras.ops.lu_factor",
                        "keras.ops.norm",
                        "keras.ops.qr",
                        "keras.ops.solve",
                        "keras.ops.solve_triangular",
                        "keras.ops.svd",
                    ],
                },
                {
                    "path": "core/",
                    "title": "Core ops",
                    "generate": [
                        "keras.ops.associative_scan",
                        "keras.ops.cast",
                        "keras.ops.cond",
                        "keras.ops.convert_to_numpy",
                        "keras.ops.convert_to_tensor",
                        "keras.ops.custom_gradient",
                        "keras.ops.dtype",
                        "keras.ops.erf",
                        "keras.ops.erfinv",
                        "keras.ops.extract_sequences",
                        "keras.ops.fori_loop",
                        "keras.ops.in_top_k",
                        "keras.ops.is_tensor",
                        "keras.ops.logsumexp",
                        "keras.ops.map",
                        "keras.ops.rsqrt",
                        "keras.ops.saturate_cast",
                        "keras.ops.scan",
                        "keras.ops.scatter",
                        "keras.ops.scatter_update",
                        "keras.ops.segment_max",
                        "keras.ops.segment_sum",
                        "keras.ops.shape",
                        "keras.ops.slice",
                        "keras.ops.slice_update",
                        "keras.ops.stop_gradient",
                        "keras.ops.switch",
                        "keras.ops.top_k",
                        "keras.ops.unstack",
                        "keras.ops.vectorized_map",
                        "keras.ops.while_loop",
                    ],
                },
                {
                    "path": "image/",
                    "title": "Image ops",
                    "generate": [
                        "keras.ops.image.affine_transform",
                        "keras.ops.image.crop_images",
                        "keras.ops.image.extract_patches",
                        "keras.ops.image.gaussian_blur",
                        "keras.ops.image.hsv_to_rgb",
                        "keras.ops.image.map_coordinates",
                        "keras.ops.image.pad_images",
                        "keras.ops.image.perspective_transform",
                        "keras.ops.image.resize",
                        "keras.ops.image.rgb_to_hsv",
                        "keras.ops.image.rgb_to_grayscale",
                        "keras.ops.image.elastic_transform",
                        "keras.ops.image.extract_patches_3d",
                        "keras.ops.image.scale_and_translate",
                    ],
                },
                {
                    "path": "fft/",
                    "title": "FFT ops",
                    "generate": [
                        "keras.ops.fft",
                        "keras.ops.fft2",
                        "keras.ops.ifft2",
                        "keras.ops.rfft",
                        "keras.ops.stft",
                        "keras.ops.irfft",
                        "keras.ops.istft",
                    ],
                },
                {
                    "path": "other_ops",
                    "title": "Other ops",
                    "generate": [
                        "keras.ops.jvp",
                        "keras.ops.polar",
                        "keras.ops.rearrange",
                        "keras.ops.sparse_sigmoid",
                        "keras.ops.unfold",
                        "keras.ops.view_as_complex",
                        "keras.ops.view_as_real",
                    ],
                },
            ],
        },
        {
            "path": "optimizers/",
            "title": "Optimizers",
            "toc": True,
            "generate": [
                "keras.Optimizer",
                "keras.optimizers.Optimizer.apply_gradients",
                "keras.optimizers.Optimizer.variables",
            ],
            "aliases": {
                "keras.Optimizer": ["keras.optimizers.Optimizer"]
            },
            "children": [
                {
                    "path": "sgd",
                    "title": "SGD",
                    "generate": ["keras.optimizers.SGD"],
                },
                {
                    "path": "rmsprop",
                    "title": "RMSprop",
                    "generate": ["keras.optimizers.RMSprop"],
                },
                {
                    "path": "adam",
                    "title": "Adam",
                    "generate": ["keras.optimizers.Adam"],
                },
                {
                    "path": "adamw",
                    "title": "AdamW",
                    "generate": ["keras.optimizers.AdamW"],
                },
                {
                    "path": "adadelta",
                    "title": "Adadelta",
                    "generate": ["keras.optimizers.Adadelta"],
                },
                {
                    "path": "adagrad",
                    "title": "Adagrad",
                    "generate": ["keras.optimizers.Adagrad"],
                },
                {
                    "path": "adamax",
                    "title": "Adamax",
                    "generate": ["keras.optimizers.Adamax"],
                },
                {
                    "path": "adafactor",
                    "title": "Adafactor",
                    "generate": ["keras.optimizers.Adafactor"],
                },
                {
                    "path": "Nadam",
                    "title": "Nadam",
                    "generate": ["keras.optimizers.Nadam"],
                },
                {
                    "path": "ftrl",
                    "title": "Ftrl",
                    "generate": ["keras.optimizers.Ftrl"],
                },
                {
                    "path": "lion",
                    "title": "Lion",
                    "generate": ["keras.optimizers.Lion"],
                },
                {
                    "path": "lamb",
                    "title": "Lamb",
                    "generate": ["keras.optimizers.Lamb"],
                },
                {
                    "path": "loss_scale_optimizer",
                    "title": "Loss Scale Optimizer",
                    "generate": ["keras.optimizers.LossScaleOptimizer"],
                },
                {
                    "path": "learning_rate_schedules/",
                    "title": "Learning rate schedules API",
                    "toc": True,
                    "skip_from_toc": True,
                    "children": [
                        {
                            "path": "learning_rate_schedule",
                            "title": "LearningRateSchedule",
                            "generate": [
                                "keras.optimizers.schedules.LearningRateSchedule"
                            ],
                        },
                        {
                            "path": "exponential_decay",
                            "title": "ExponentialDecay",
                            "generate": ["keras.optimizers.schedules.ExponentialDecay"],
                        },
                        {
                            "path": "piecewise_constant_decay",
                            "title": "PiecewiseConstantDecay",
                            "generate": [
                                "keras.optimizers.schedules.PiecewiseConstantDecay"
                            ],
                        },
                        {
                            "path": "polynomial_decay",
                            "title": "PolynomialDecay",
                            "generate": ["keras.optimizers.schedules.PolynomialDecay"],
                        },
                        {
                            "path": "inverse_time_decay",
                            "title": "InverseTimeDecay",
                            "generate": ["keras.optimizers.schedules.InverseTimeDecay"],
                        },
                        {
                            "path": "cosine_decay",
                            "title": "CosineDecay",
                            "generate": ["keras.optimizers.schedules.CosineDecay"],
                        },
                        {
                            "path": "cosine_decay_restarts",
                            "title": "CosineDecayRestarts",
                            "generate": [
                                "keras.optimizers.schedules.CosineDecayRestarts"
                            ],
                        },
                    ],
                },
                {
                    "path": "muon",
                    "title": "Muon",
                    "generate": ["keras.optimizers.Muon"],
                },
                {
                    "path": "optimizer_utilities",
                    "title": "Optimizer utilities",
                    "generate": [

                    ],
                },
            ],
        },
        {
            "path": "metrics/",
            "title": "Metrics",
            "toc": True,
            "children": [
                {
                    "path": "base_metric",
                    "title": "Base Metric class",
                    "generate": [
                        "keras.Metric",
                    ],
                    "aliases": {
                        "keras.Metric": ["keras.metrics.Metric"]
                    }
                },
                {
                    "path": "accuracy_metrics",
                    "title": "Accuracy metrics",
                    "generate": [
                        "keras.metrics.Accuracy",
                        "keras.metrics.BinaryAccuracy",
                        "keras.metrics.CategoricalAccuracy",
                        "keras.metrics.SparseCategoricalAccuracy",
                        "keras.metrics.TopKCategoricalAccuracy",
                        "keras.metrics.SparseTopKCategoricalAccuracy",
                        "keras.metrics.binary_accuracy",
                        "keras.metrics.categorical_accuracy",
                        "keras.metrics.sparse_categorical_accuracy",
                        "keras.metrics.sparse_top_k_categorical_accuracy",
                        "keras.metrics.top_k_categorical_accuracy",
                    ],
                },
                {
                    "path": "probabilistic_metrics",
                    "title": "Probabilistic metrics",
                    "generate": [
                        "keras.metrics.BinaryCrossentropy",
                        "keras.metrics.CategoricalCrossentropy",
                        "keras.metrics.SparseCategoricalCrossentropy",
                        "keras.metrics.KLDivergence",
                        "keras.metrics.Poisson",
                    ],
                },
                {
                    "path": "regression_metrics",
                    "title": "Regression metrics",
                    "generate": [
                        "keras.metrics.MeanSquaredError",
                        "keras.metrics.RootMeanSquaredError",
                        "keras.metrics.MeanAbsoluteError",
                        "keras.metrics.MeanAbsolutePercentageError",
                        "keras.metrics.MeanSquaredLogarithmicError",
                        "keras.metrics.CosineSimilarity",
                        "keras.metrics.LogCoshError",
                        "keras.metrics.R2Score",
                        "keras.metrics.huber",
                    ],
                },
                {
                    "path": "classification_metrics",
                    "title": "Classification metrics based on True/False positives & negatives",
                    "generate": [
                        "keras.metrics.AUC",
                        "keras.metrics.Precision",
                        "keras.metrics.Recall",
                        "keras.metrics.TruePositives",
                        "keras.metrics.TrueNegatives",
                        "keras.metrics.FalsePositives",
                        "keras.metrics.FalseNegatives",
                        "keras.metrics.PrecisionAtRecall",
                        "keras.metrics.RecallAtPrecision",
                        "keras.metrics.SensitivityAtSpecificity",
                        "keras.metrics.SpecificityAtSensitivity",
                        "keras.metrics.F1Score",
                        "keras.metrics.FBetaScore",
                        "keras.metrics.PearsonCorrelation",
                        "keras.metrics.ConcordanceCorrelation",
                        "keras.metrics.pearson_correlation",
                        "keras.metrics.concordance_correlation",
                    ],
                },
                {
                    "path": "segmentation_metrics",
                    "title": "Image segmentation metrics",
                    "generate": [
                        "keras.metrics.IoU",
                        "keras.metrics.BinaryIoU",
                        "keras.metrics.OneHotIoU",
                        "keras.metrics.OneHotMeanIoU",
                        "keras.metrics.MeanIoU",
                    ],
                },
                {
                    "path": "hinge_metrics",
                    "title": 'Hinge metrics for "maximum-margin" classification',
                    "generate": [
                        "keras.metrics.Hinge",
                        "keras.metrics.SquaredHinge",
                        "keras.metrics.CategoricalHinge",
                    ],
                },
                {
                    "path": "metrics_wrappers",
                    "title": "Metric wrappers and reduction metrics",
                    "generate": [
                        "keras.metrics.MeanMetricWrapper",
                        "keras.metrics.Mean",
                        "keras.metrics.Sum",

                    ],
                },
            ],
        },
        {
            "path": "losses/",
            "title": "Losses",
            "toc": True,
            "generate": [
                "keras.Loss",
            ],
            "aliases": {
                "keras.Loss": ["keras.losses.Loss"]
            },
            "children": [
                {
                    "path": "probabilistic_losses",
                    "title": "Probabilistic losses",
                    "generate": [
                        "keras.losses.BinaryCrossentropy",
                        "keras.losses.BinaryFocalCrossentropy",
                        "keras.losses.CategoricalCrossentropy",
                        "keras.losses.CategoricalFocalCrossentropy",
                        "keras.losses.SparseCategoricalCrossentropy",
                        "keras.losses.Poisson",
                        "keras.losses.CTC",
                        "keras.losses.KLDivergence",
                        "keras.losses.binary_crossentropy",
                        "keras.losses.categorical_crossentropy",
                        "keras.losses.sparse_categorical_crossentropy",
                        "keras.losses.poisson",
                        "keras.losses.ctc",
                        "keras.losses.kl_divergence",
                    ],
                },
                {
                    "path": "regression_losses",
                    "title": "Regression losses",
                    "generate": [
                        "keras.losses.MeanSquaredError",
                        "keras.losses.MeanAbsoluteError",
                        "keras.losses.MeanAbsolutePercentageError",
                        "keras.losses.MeanSquaredLogarithmicError",
                        "keras.losses.CosineSimilarity",
                        "keras.losses.Huber",
                        "keras.losses.LogCosh",
                        "keras.losses.Tversky",
                        "keras.losses.Dice",
                        "keras.losses.mean_squared_error",
                        "keras.losses.mean_absolute_error",
                        "keras.losses.mean_absolute_percentage_error",
                        "keras.losses.mean_squared_logarithmic_error",
                        "keras.losses.cosine_similarity",
                        "keras.losses.huber",
                        "keras.losses.log_cosh",
                        "keras.losses.tversky",
                        "keras.losses.dice",
                    ],
                },
                {
                    "path": "hinge_losses",
                    "title": 'Hinge losses for "maximum-margin" classification',
                    "generate": [
                        "keras.losses.Hinge",
                        "keras.losses.SquaredHinge",
                        "keras.losses.CategoricalHinge",
                        "keras.losses.hinge",
                        "keras.losses.squared_hinge",
                        "keras.losses.categorical_hinge",
                        "keras.losses.CategoricalGeneralizedCrossEntropy",
                        "keras.losses.Circle",
                        "keras.losses.categorical_generalized_cross_entropy",
                        "keras.losses.circle",

                    ],
                },
            ],
        },
        {
            "path": "data_loading/",
            "title": "Data loading",
            "toc": True,
            "children": [
                {
                    "path": "image",
                    "title": "Image data loading",
                    "generate": [
                        "keras.utils.image_dataset_from_directory",
                        "keras.utils.load_img",
                        "keras.utils.img_to_array",
                        "keras.utils.save_img",
                        "keras.utils.array_to_img",
                    ],
                },
                {
                    "path": "timeseries",
                    "title": "Timeseries data loading",
                    "generate": [
                        "keras.utils.timeseries_dataset_from_array",
                        "keras.utils.pad_sequences",  # LEGACY
                    ],
                },
                {
                    "path": "text",
                    "title": "Text data loading",
                    "generate": [
                        "keras.utils.text_dataset_from_directory",
                    ],
                },
                {
                    "path": "audio",
                    "title": "Audio data loading",
                    "generate": [
                        "keras.utils.audio_dataset_from_directory",
                    ],
                },
            ],
        },
        {
            "path": "tree/",
            "title": "Tree API",
            "toc": True,
            "children": [
                {
                    "path": "tree_utilities",
                    "title": "Tree utilities",
                    "generate": [
                        "keras.tree.MAP_TO_NONE",
                        "keras.tree.assert_same_paths",
                        "keras.tree.assert_same_structure",
                        "keras.tree.flatten",
                        "keras.tree.flatten_with_path",
                        "keras.tree.is_nested",
                        "keras.tree.lists_to_tuples",
                        "keras.tree.map_shape_structure",
                        "keras.tree.map_structure",
                        "keras.tree.map_structure_up_to",
                        "keras.tree.pack_sequence_as",
                        "keras.tree.traverse",
                    ],
                },
            ],
        },
        {
            "path": "datasets/",
            "title": "Built-in small datasets",
            "toc": True,
            "children": [
                {
                    "path": "mnist",
                    "title": "MNIST digits classification dataset",
                    "generate": ["keras.datasets.mnist.load_data"],
                },
                {
                    "path": "cifar10",
                    "title": "CIFAR10 small images classification dataset",
                    "generate": ["keras.datasets.cifar10.load_data"],
                },
                {
                    "path": "cifar100",
                    "title": "CIFAR100 small images classification dataset",
                    "generate": ["keras.datasets.cifar100.load_data"],
                },
                {
                    "path": "imdb",
                    "title": "IMDB movie review sentiment classification dataset",
                    "generate": [
                        "keras.datasets.imdb.load_data",
                        "keras.datasets.imdb.get_word_index",
                    ],
                },
                {
                    "path": "reuters",
                    "title": "Reuters newswire classification dataset",
                    "generate": [
                        "keras.datasets.reuters.load_data",
                        "keras.datasets.reuters.get_word_index",
                        "keras.datasets.reuters.get_label_names",
                    ],
                },
                {
                    "path": "fashion_mnist",
                    "title": "Fashion MNIST dataset, an alternative to MNIST",
                    "generate": ["keras.datasets.fashion_mnist.load_data"],
                },
                {
                    "path": "california_housing",
                    "title": "California Housing price regression dataset",
                    "generate": ["keras.datasets.california_housing.load_data"],
                },
                {
                    "path": "boston_housing",
                    "title": "Boston Housing price regression dataset",
                    "generate": ["keras.datasets.boston_housing.load_data"],
                },
            ],
        },
        {
            "path": "applications/",
            "title": "Keras Applications",
            "children": [
                {
                    "path": "xception",
                    "title": "Xception",
                    "toc": True,
                    "children": [
                        {
                            "path": "xception_model",
                            "title": "Xception model",
                            "generate": ["keras.applications.Xception"],
                        },
                        {
                            "path": "xception_preprocessing",
                            "title": "Xception preprocessing utilities",
                            "generate": [
                                "keras.applications.xception.decode_predictions",
                                "keras.applications.xception.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "efficientnet",
                    "title": "EfficientNet B0 to B7",
                    "toc": True,
                    "children": [
                        {
                            "path": "efficientnet_models",
                            "title": "EfficientNet models",
                            "generate": [
                                "keras.applications.EfficientNetB0",
                                "keras.applications.EfficientNetB1",
                                "keras.applications.EfficientNetB2",
                                "keras.applications.EfficientNetB3",
                                "keras.applications.EfficientNetB4",
                                "keras.applications.EfficientNetB5",
                                "keras.applications.EfficientNetB6",
                                "keras.applications.EfficientNetB7",
                            ],
                        },
                        {
                            "path": "efficientnet_preprocessing",
                            "title": "EfficientNet preprocessing utilities",
                            "generate": [
                                "keras.applications.efficientnet.decode_predictions",
                                "keras.applications.efficientnet.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "efficientnet_v2",
                    "title": "EfficientNetV2 B0 to B3 and S, M, L",
                    "toc": True,
                    "children": [
                        {
                            "path": "efficientnet_v2_models",
                            "title": "EfficientNetV2 models",
                            "generate": [
                                "keras.applications.EfficientNetV2B0",
                                "keras.applications.EfficientNetV2B1",
                                "keras.applications.EfficientNetV2B2",
                                "keras.applications.EfficientNetV2B3",
                                "keras.applications.EfficientNetV2S",
                                "keras.applications.EfficientNetV2M",
                                "keras.applications.EfficientNetV2L",
                            ],
                        },
                        {
                            "path": "efficientnet_v2_preprocessing",
                            "title": "EfficientNetV2 preprocessing utilities",
                            "generate": [
                                "keras.applications.efficientnet_v2.decode_predictions",
                                "keras.applications.efficientnet_v2.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "convnext",
                    "title": "ConvNeXt Tiny, Small, Base, Large, XLarge",
                    "toc": True,
                    "children": [
                        {
                            "path": "convnext_models",
                            "title": "ConvNeXt models",
                            "generate": [
                                "keras.applications.ConvNeXtTiny",
                                "keras.applications.ConvNeXtSmall",
                                "keras.applications.ConvNeXtBase",
                                "keras.applications.ConvNeXtLarge",
                                "keras.applications.ConvNeXtXLarge",
                            ],
                        },
                        {
                            "path": "convnext_preprocessing",
                            "title": "ConvNeXt preprocessing utilities",
                            "generate": [
                                "keras.applications.convnext.decode_predictions",
                                "keras.applications.convnext.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "vgg",
                    "title": "VGG16 and VGG19",
                    "toc": True,
                    "children": [
                        {
                            "path": "vgg_models",
                            "title": "VGG16 and VGG19 models",
                            "generate": [
                                "keras.applications.VGG16",
                                "keras.applications.VGG19",
                                "keras.applications.vgg16.VGG16",
                                "keras.applications.vgg19.VGG19",
                            ],
                        },
                        {
                            "path": "vgg_preprocessing",
                            "title": "VGG preprocessing utilities",
                            "generate": [
                                "keras.applications.vgg16.decode_predictions",
                                "keras.applications.vgg16.preprocess_input",
                                "keras.applications.vgg19.decode_predictions",
                                "keras.applications.vgg19.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "resnet",
                    "title": "ResNet and ResNetV2",
                    "toc": True,
                    "children": [
                        {
                            "path": "resnet_models",
                            "title": "ResNet models",
                            "generate": [
                                "keras.applications.ResNet50",
                                "keras.applications.ResNet101",
                                "keras.applications.ResNet152",
                                "keras.applications.ResNet50V2",
                                "keras.applications.ResNet101V2",
                                "keras.applications.ResNet152V2",
                            ],
                        },
                        {
                            "path": "resnet_preprocessing",
                            "title": "ResNet preprocessing utilities",
                            "generate": [
                                "keras.applications.resnet_v2.decode_predictions",
                                "keras.applications.resnet_v2.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "mobilenet",
                    "title": "MobileNet, MobileNetV2, and MobileNetV3",
                    "toc": True,
                    "children": [
                        {
                            "path": "mobilenet_models",
                            "title": "MobileNet models",
                            "generate": [
                                "keras.applications.MobileNet",
                                "keras.applications.MobileNetV2",
                                "keras.applications.MobileNetV3Small",
                                "keras.applications.MobileNetV3Large",
                            ],
                        },
                        {
                            "path": "mobilenet_preprocessing",
                            "title": "MobileNet preprocessing utilities",
                            "generate": [
                                "keras.applications.mobilenet.decode_predictions",
                                "keras.applications.mobilenet.preprocess_input",
                                "keras.applications.mobilenet_v2.decode_predictions",
                                "keras.applications.mobilenet_v2.preprocess_input",
                                "keras.applications.mobilenet_v3.decode_predictions",
                                "keras.applications.mobilenet_v3.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "densenet",
                    "title": "DenseNet",
                    "toc": True,
                    "children": [
                        {
                            "path": "densenet_models",
                            "title": "DenseNet models",
                            "generate": [
                                "keras.applications.DenseNet121",
                                "keras.applications.DenseNet169",
                                "keras.applications.DenseNet201",
                            ],
                        },
                        {
                            "path": "densenet_preprocessing",
                            "title": "DenseNet preprocessing utilities",
                            "generate": [
                                "keras.applications.densenet.decode_predictions",
                                "keras.applications.densenet.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "nasnet",
                    "title": "NasNetLarge and NasNetMobile",
                    "toc": True,
                    "children": [
                        {
                            "path": "nasnet_models",
                            "title": "NASNet models",
                            "generate": [
                                "keras.applications.NASNetLarge",
                                "keras.applications.NASNetMobile",
                            ],
                        },
                        {
                            "path": "nasnet_preprocessing",
                            "title": "NASNet preprocessing utilities",
                            "generate": [
                                "keras.applications.nasnet.decode_predictions",
                                "keras.applications.nasnet.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "inceptionv3",
                    "title": "InceptionV3",
                    "toc": True,
                    "children": [
                        {
                            "path": "inception_v3_model",
                            "title": "InceptionV3 model",
                            "generate": [
                                "keras.applications.InceptionV3",
                            ],
                        },
                        {
                            "path": "inception_v3_preprocessing",
                            "title": "InceptionV3 preprocessing utilities",
                            "generate": [
                                "keras.applications.inception_v3.decode_predictions",
                                "keras.applications.inception_v3.preprocess_input",
                            ],
                        },
                    ],
                },
                {
                    "path": "inceptionresnetv2",
                    "title": "InceptionResNetV2",
                    "toc": True,
                    "children": [
                        {
                            "path": "inception_resnet_v2_model",
                            "title": "InceptionResNetV2 model",
                            "generate": [
                                "keras.applications.InceptionResNetV2",
                            ],
                        },
                        {
                            "path": "inception_resnet_v2_preprocessing",
                            "title": "InceptionResNetV2 preprocessing utilities",
                            "generate": [
                                "keras.applications.inception_resnet_v2.decode_predictions",
                                "keras.applications.inception_resnet_v2.preprocess_input",
                            ],
                        },
                    ],
                },

            ],
        },
        {
            "path": "mixed_precision/",
            "title": "Mixed precision",
            "toc": True,
            "children": [
                {
                    "path": "policy",
                    "title": "Mixed precision policy API",
                    "generate": [
                        "keras.dtype_policies.DTypePolicy",
                        "keras.dtype_policies.DTypePolicyMap",
                        "keras.dtype_policies.FloatDTypePolicy",
                        "keras.dtype_policies.QuantizedDTypePolicy",
                        "keras.dtype_policies.QuantizedFloat8DTypePolicy",
                        "keras.dtype_policies.GPTQDTypePolicy",
                        "keras.config.dtype_policy",
                        "keras.config.set_dtype_policy",

                    ],
                },
            ],
        },
        {
            "path": "distribution/",
            "title": "Multi-device distribution",
            "toc": True,
            "children": [
                {
                    "path": "layout_map",
                    "title": "LayoutMap API",
                    "generate": [
                        "keras.distribution.LayoutMap",
                        "keras.distribution.DeviceMesh",
                        "keras.distribution.TensorLayout",
                        "keras.distribution.distribute_tensor",
                    ],
                },
                {
                    "path": "data_parallel",
                    "title": "DataParallel API",
                    "generate": [
                        "keras.distribution.DataParallel",
                    ],
                },
                {
                    "path": "model_parallel",
                    "title": "ModelParallel API",
                    "generate": [
                        "keras.distribution.ModelParallel",
                    ],
                },
                {
                    "path": "distribution_utils",
                    "title": "Distribution utilities",
                    "generate": [
                        "keras.distribution.set_distribution",
                        "keras.distribution.distribution",
                        "keras.distribution.list_devices",
                        "keras.distribution.initialize",
                        "keras.distribution.get_device_count",
                    ],
                },
            ],
        },
        {
            "path": "random/",
            "title": "RNG API",
            "toc": True,
            "children": [
                {
                    "path": "seed_generator",
                    "title": "SeedGenerator class",
                    "generate": ["keras.random.SeedGenerator"],
                },
                {
                    "path": "random_ops",
                    "title": "Random operations",
                    "generate": [
                        "keras.random.beta",
                        "keras.random.binomial",
                        "keras.random.categorical",
                        "keras.random.dropout",
                        "keras.random.gamma",
                        "keras.random.normal",
                        "keras.random.randint",
                        "keras.random.shuffle",
                        "keras.random.truncated_normal",
                        "keras.random.uniform",
                    ],
                },
            ],
        },
        {
            "path": "quantizers/",
            "title": "Quantizers",
            "toc": True,
            "generate": [
                "keras.Quantizer",
            ],
            "aliases": {
                "keras.Quantizer": ["keras.quantizers.Quantizer"]
            },
            "children": [
                {
                    "path": "quantizer_classes",
                    "title": "Quantizer classes",
                    "generate": [
                        "keras.quantizers.Quantizer",
                        "keras.quantizers.AbsMaxQuantizer",
                        "keras.quantizers.QuantizationConfig",
                        "keras.quantizers.Int8QuantizationConfig",
                        "keras.quantizers.Int4QuantizationConfig",
                        "keras.quantizers.Float8QuantizationConfig",
                        "keras.quantizers.GPTQConfig",
                    ],
                },
                {
                    "path": "quantizer_utils",
                    "title": "Quantizer utilities",
                    "generate": [
                        "keras.quantizers.abs_max_quantize",
                        "keras.quantizers.compute_float8_amax_history",
                        "keras.quantizers.compute_float8_scale",
                        "keras.quantizers.fake_quant_with_min_max_vars",
                        "keras.quantizers.pack_int4",
                        "keras.quantizers.quantize_and_dequantize",
                        "keras.quantizers.unpack_int4",
                    ],
                },
            ],
        },
        {
            "path": "scope/",
            "title": "Scope",
            "toc": True,
            "children": [
                {
                    "path": "scope_classes",
                    "title": "Scope classes",
                    "generate": [
                        "keras.SymbolicScope",
                        "keras.StatelessScope",
                    ],
                },
            ],
        },
        {
            "path": "rematerialization/",
            "title": "Rematerialization",
            "toc": True,
            "children": [
                {
                    "path": "remat_scope",
                    "title": "RematScope",
                    "generate": ["keras.RematScope"],
                },
                {
                    "path": "remat",
                    "title": "Remat",
                    "generate": ["keras.remat"],
                },
            ],
        },
        {
            "path": "utils/",
            "title": "Utilities",
            "toc": True,
            "children": [
                {
                    "path": "experiment_management_utils",
                    "title": "Experiment management utilities",
                    "generate": [
                        "keras.utils.Config",
                    ],
                },
                {
                    "path": "model_plotting_utils",
                    "title": "Model plotting utilities",
                    "generate": [
                        "keras.utils.plot_model",
                        "keras.utils.model_to_dot",
                    ],
                },
                {
                    "path": "feature_space",
                    "title": "Structured data preprocessing utilities",
                    "generate": [
                        "keras.utils.FeatureSpace",
                    ],
                },
                {
                    "path": "tensor_utils",
                    "title": "Tensor utilities",
                    "generate": [
                        "keras.utils.get_source_inputs",
                        "keras.utils.is_keras_tensor",
                        "keras.backend.is_float_dtype",
                        "keras.backend.is_int_dtype",
                        "keras.backend.standardize_dtype",
                    ],
                    "aliases": {
                        "keras.utils.is_keras_tensor": ["keras.backend.is_keras_tensor"]
                    }
                },
                {
                    "path": "bounding_boxes",
                    "title": "Bounding boxes",
                    "generate": [
                        "keras.utils.bounding_boxes.affine_transform",
                        "keras.utils.bounding_boxes.clip_to_image_size",
                        "keras.utils.bounding_boxes.compute_ciou",
                        "keras.utils.bounding_boxes.compute_iou",
                        "keras.utils.bounding_boxes.convert_format",
                        "keras.utils.bounding_boxes.crop",
                        "keras.utils.bounding_boxes.decode_deltas_to_boxes",
                        "keras.utils.bounding_boxes.encode_box_to_deltas",
                        "keras.utils.bounding_boxes.pad",
                    ],
                },
                {
                    "path": "python_utils",
                    "title": "Python & NumPy utilities",
                    "generate": [
                        "keras.utils.set_random_seed",
                        "keras.utils.split_dataset",
                        "keras.utils.pack_x_y_sample_weight",
                        "keras.utils.unpack_x_y_sample_weight",
                        "keras.utils.get_file",
                        "keras.utils.Progbar",
                        "keras.utils.to_categorical",
                        "keras.utils.normalize",
                    ],
                },
                {
                    "path": "bounding_boxes_utils",
                    "title": "Bounding boxes utilities",
                    "generate": [
                        "keras.utils.bounding_boxes.affine_transform",
                        "keras.utils.bounding_boxes.clip_to_image_size",
                        "keras.utils.bounding_boxes.compute_ciou",
                        "keras.utils.bounding_boxes.compute_iou",
                        "keras.utils.bounding_boxes.convert_format",
                        "keras.utils.bounding_boxes.crop",
                        "keras.utils.bounding_boxes.decode_deltas_to_boxes",
                        "keras.utils.bounding_boxes.encode_box_to_deltas",
                        "keras.utils.bounding_boxes.pad",
                    ],
                },
                {
                    "path": "visualization_utils",
                    "title": "Visualization utilities",
                    "generate": [
                        "keras.visualization.draw_bounding_boxes",
                        "keras.visualization.draw_segmentation_masks",
                        "keras.visualization.plot_bounding_box_gallery",
                        "keras.visualization.plot_image_gallery",
                        "keras.visualization.plot_segmentation_mask_gallery",
                    ],
                },
                {
                    "path": "saving_loading_utils",
                    "title": "Saving and loading utilities",
                    "generate": [
                        "keras.saving.load_weights",
                        "keras.saving.save_weights",
                    ],
                },
                {
                    "path": "preprocessing_utils",
                    "title": "Preprocessing utilities",
                    "generate": [
                        "keras.preprocessing.image.smart_resize",
                        "keras.preprocessing.image.load_img",
                        "keras.preprocessing.image.save_img",
                        "keras.utils.PyDataset",
                    ],
                    "aliases": {
                        "keras.utils.PyDataset": ["keras.utils.Sequence"]
                    }
                },
                {
                    "path": "backend_utils",
                    "title": "Backend utilities",
                    "generate": [
                        "keras.backend.get_uid",
                        "keras.backend.result_type",
                        "keras.backend.clear_session",
                        "keras.backend.epsilon",
                        "keras.backend.set_epsilon",
                        "keras.backend.floatx",
                        "keras.backend.set_floatx",
                    ],
                },
                {
                    "path": "sklearn_wrappers",
                    "title": "Scikit-Learn API wrappers",
                    "generate": [
                        "keras.wrappers.SKLearnClassifier",
                        "keras.wrappers.SKLearnRegressor",
                        "keras.wrappers.SKLearnTransformer",
                    ],
                },
                {
                    "path": "config_utils",
                    "title": "Keras configuration utilities",
                    "generate": [
                        "keras.version",
                        "keras.utils.clear_session",
                        "keras.config.enable_traceback_filtering",
                        "keras.config.disable_traceback_filtering",
                        "keras.config.is_traceback_filtering_enabled",
                        "keras.config.enable_interactive_logging",
                        "keras.config.disable_interactive_logging",
                        "keras.config.is_interactive_logging_enabled",
                        "keras.config.enable_unsafe_deserialization",
                        "keras.config.floatx",
                        "keras.config.set_floatx",
                        "keras.config.image_data_format",
                        "keras.config.set_image_data_format",
                        "keras.config.epsilon",
                        "keras.config.set_epsilon",
                        "keras.config.backend",
                        "keras.config.set_backend",
                        "keras.config.enable_flash_attention",
                        "keras.config.disable_flash_attention",
                        "keras.config.is_flash_attention_enabled",
                        "keras.config.is_nnx_enabled",
                        "keras.config.max_epochs",
                        "keras.config.set_max_epochs",
                        "keras.config.max_steps_per_epoch",
                        "keras.config.set_max_steps_per_epoch",
                        "keras.device",
                        "keras.name_scope",
                    ],
                },
            ],
        },
    ],
}
