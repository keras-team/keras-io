LAYERS_MASTER = {
    "path": "layers/",  # TODO
    "title": "Layers API",
    "toc": True,
    "children": [
        {
            "path": "base_layer",
            "title": "The base Layer class",
            "generate": [
                "tensorflow.keras.layers.Layer",
                "tensorflow.keras.layers.Layer.weights",
                "tensorflow.keras.layers.Layer.trainable_weights",
                "tensorflow.keras.layers.Layer.non_trainable_weights",
                "tensorflow.keras.layers.Layer.add_weight",
                "tensorflow.keras.layers.Layer.trainable",
                "tensorflow.keras.layers.Layer.get_weights",
                "tensorflow.keras.layers.Layer.set_weights",
                "tensorflow.keras.Model.get_config",
                "tensorflow.keras.layers.Layer.add_loss",
                "tensorflow.keras.layers.Layer.add_metric",
                "tensorflow.keras.layers.Layer.losses",
                "tensorflow.keras.layers.Layer.metrics",
                "tensorflow.keras.layers.Layer.dynamic",
            ],
        },
        {
            "path": "activations",
            "title": "Layer activations",
            "generate": [
                "tensorflow.keras.activations.relu",
                "tensorflow.keras.activations.sigmoid",
                "tensorflow.keras.activations.softmax",
                "tensorflow.keras.activations.softplus",
                "tensorflow.keras.activations.softsign",
                "tensorflow.keras.activations.tanh",
                "tensorflow.keras.activations.selu",
                "tensorflow.keras.activations.elu",
                "tensorflow.keras.activations.exponential",
            ],
        },
        {
            "path": "initializers",
            "title": "Layer weight initializers",
            "generate": [
                "tensorflow.keras.initializers.RandomNormal",
                "tensorflow.keras.initializers.RandomUniform",
                "tensorflow.keras.initializers.TruncatedNormal",
                "tensorflow.keras.initializers.Zeros",
                "tensorflow.keras.initializers.Ones",
                "tensorflow.keras.initializers.GlorotNormal",
                "tensorflow.keras.initializers.GlorotUniform",
                "tensorflow.keras.initializers.HeNormal",
                "tensorflow.keras.initializers.HeUniform",
                "tensorflow.keras.initializers.Identity",
                "tensorflow.keras.initializers.Orthogonal",
                "tensorflow.keras.initializers.Constant",
                "tensorflow.keras.initializers.VarianceScaling",
            ],
        },
        {
            "path": "regularizers",
            "title": "Layer weight regularizers",
            "generate": [
                "tensorflow.keras.regularizers.L1",
                "tensorflow.keras.regularizers.L2",
                "tensorflow.keras.regularizers.L1L2",
                "tensorflow.keras.regularizers.OrthogonalRegularizer",
            ],
        },
        {
            "path": "constraints",
            "title": "Layer weight constraints",
            "generate": [
                "tensorflow.keras.constraints.MaxNorm",
                "tensorflow.keras.constraints.MinMaxNorm",
                "tensorflow.keras.constraints.NonNeg",
                "tensorflow.keras.constraints.UnitNorm",
                "tensorflow.keras.constraints.RadialConstraint",
            ],
        },
        {
            "path": "core_layers/",
            "title": "Core layers",
            "toc": True,
            "children": [
                {
                    "path": "input",
                    "title": "Input object",
                    "generate": ["tensorflow.keras.Input"],
                },
                {
                    "path": "dense",
                    "title": "Dense layer",
                    "generate": ["tensorflow.keras.layers.Dense"],
                },
                {
                    "path": "activation",
                    "title": "Activation layer",
                    "generate": ["tensorflow.keras.layers.Activation"],
                },
                {
                    "path": "embedding",
                    "title": "Embedding layer",
                    "generate": ["tensorflow.keras.layers.Embedding"],
                },
                {
                    "path": "masking",
                    "title": "Masking layer",
                    "generate": ["tensorflow.keras.layers.Masking"],
                },
                {
                    "path": "lambda",
                    "title": "Lambda layer",
                    "generate": ["tensorflow.keras.layers.Lambda"],
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
                    "generate": ["tensorflow.keras.layers.Conv1D"],
                },
                {
                    "path": "convolution2d",
                    "title": "Conv2D layer",
                    "generate": ["tensorflow.keras.layers.Conv2D"],
                },
                {
                    "path": "convolution3d",
                    "title": "Conv3D layer",
                    "generate": ["tensorflow.keras.layers.Conv3D"],
                },
                {
                    "path": "separable_convolution1d",
                    "title": "SeparableConv1D layer",
                    "generate": ["tensorflow.keras.layers.SeparableConv1D"],
                },
                {
                    "path": "separable_convolution2d",
                    "title": "SeparableConv2D layer",
                    "generate": ["tensorflow.keras.layers.SeparableConv2D"],
                },
                {
                    "path": "depthwise_convolution2d",
                    "title": "DepthwiseConv2D layer",
                    "generate": ["tensorflow.keras.layers.DepthwiseConv2D"],
                },
                {
                    "path": "convolution1d_transpose",
                    "title": "Conv1DTranspose layer",
                    "generate": ["tensorflow.keras.layers.Conv1DTranspose"],
                },
                {
                    "path": "convolution2d_transpose",
                    "title": "Conv2DTranspose layer",
                    "generate": ["tensorflow.keras.layers.Conv2DTranspose"],
                },
                {
                    "path": "convolution3d_transpose",
                    "title": "Conv3DTranspose layer",
                    "generate": ["tensorflow.keras.layers.Conv3DTranspose"],
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
                    "generate": ["tensorflow.keras.layers.MaxPooling1D"],
                },
                {
                    "path": "max_pooling2d",
                    "title": "MaxPooling2D layer",
                    "generate": ["tensorflow.keras.layers.MaxPooling2D"],
                },
                {
                    "path": "max_pooling3d",
                    "title": "MaxPooling3D layer",
                    "generate": ["tensorflow.keras.layers.MaxPooling3D"],
                },
                {
                    "path": "average_pooling1d",
                    "title": "AveragePooling1D layer",
                    "generate": ["tensorflow.keras.layers.AveragePooling1D"],
                },
                {
                    "path": "average_pooling2d",
                    "title": "AveragePooling2D layer",
                    "generate": ["tensorflow.keras.layers.AveragePooling2D"],
                },
                {
                    "path": "average_pooling3d",
                    "title": "AveragePooling3D layer",
                    "generate": ["tensorflow.keras.layers.AveragePooling3D"],
                },
                {
                    "path": "global_max_pooling1d",
                    "title": "GlobalMaxPooling1D layer",
                    "generate": ["tensorflow.keras.layers.GlobalMaxPooling1D"],
                },
                {
                    "path": "global_max_pooling2d",
                    "title": "GlobalMaxPooling2D layer",
                    "generate": ["tensorflow.keras.layers.GlobalMaxPooling2D"],
                },
                {
                    "path": "global_max_pooling3d",
                    "title": "GlobalMaxPooling3D layer",
                    "generate": ["tensorflow.keras.layers.GlobalMaxPooling3D"],
                },
                {
                    "path": "global_average_pooling1d",
                    "title": "GlobalAveragePooling1D layer",
                    "generate": ["tensorflow.keras.layers.GlobalAveragePooling1D"],
                },
                {
                    "path": "global_average_pooling2d",
                    "title": "GlobalAveragePooling2D layer",
                    "generate": ["tensorflow.keras.layers.GlobalAveragePooling2D"],
                },
                {
                    "path": "global_average_pooling3d",
                    "title": "GlobalAveragePooling3D layer",
                    "generate": ["tensorflow.keras.layers.GlobalAveragePooling3D"],
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
                    "generate": ["tensorflow.keras.layers.LSTM"],
                },
                {
                    "path": "gru",
                    "title": "GRU layer",
                    "generate": ["tensorflow.keras.layers.GRU"],
                },
                {
                    "path": "simple_rnn",
                    "title": "SimpleRNN layer",
                    "generate": ["tensorflow.keras.layers.SimpleRNN"],
                },
                {
                    "path": "time_distributed",
                    "title": "TimeDistributed layer",
                    "generate": ["tensorflow.keras.layers.TimeDistributed"],
                },
                {
                    "path": "bidirectional",
                    "title": "Bidirectional layer",
                    "generate": ["tensorflow.keras.layers.Bidirectional"],
                },
                {
                    "path": "conv_lstm1d",
                    "title": "ConvLSTM1D layer",
                    "generate": ["tensorflow.keras.layers.ConvLSTM1D"],
                },
                {
                    "path": "conv_lstm2d",
                    "title": "ConvLSTM2D layer",
                    "generate": ["tensorflow.keras.layers.ConvLSTM2D"],
                },
                {
                    "path": "conv_lstm3d",
                    "title": "ConvLSTM3D layer",
                    "generate": ["tensorflow.keras.layers.ConvLSTM3D"],
                },
                {
                    "path": "rnn",
                    "title": "Base RNN layer",
                    "generate": ["tensorflow.keras.layers.RNN"],
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
                            "generate": ["tensorflow.keras.layers.TextVectorization"],
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
                            "generate": ["tensorflow.keras.layers.Normalization"],
                        },
                        {
                            "path": "discretization",
                            "title": "Discretization layer",
                            "generate": ["tensorflow.keras.layers.Discretization"],
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
                            "generate": ["tensorflow.keras.layers.CategoryEncoding"],
                        },
                        {
                            "path": "hashing",
                            "title": "Hashing layer",
                            "generate": ["tensorflow.keras.layers.Hashing"],
                        },
                        {
                            "path": "string_lookup",
                            "title": "StringLookup layer",
                            "generate": ["tensorflow.keras.layers.StringLookup"],
                        },
                        {
                            "path": "integer_lookup",
                            "title": "IntegerLookup layer",
                            "generate": ["tensorflow.keras.layers.IntegerLookup"],
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
                            "generate": ["tensorflow.keras.layers.Resizing"],
                        },
                        {
                            "path": "rescaling",
                            "title": "Rescaling layer",
                            "generate": ["tensorflow.keras.layers.Rescaling"],
                        },
                        {
                            "path": "center_crop",
                            "title": "CenterCrop layer",
                            "generate": ["tensorflow.keras.layers.CenterCrop"],
                        },
                    ],
                },
                {
                    "path": "image_augmentation/",
                    "title": "Image augmentation layers",
                    "toc": True,
                    "children": [
                        {
                            "path": "random_crop",
                            "title": "RandomCrop layer",
                            "generate": ["tensorflow.keras.layers.RandomCrop"],
                        },
                        {
                            "path": "random_flip",
                            "title": "RandomFlip layer",
                            "generate": ["tensorflow.keras.layers.RandomFlip"],
                        },
                        {
                            "path": "random_translation",
                            "title": "RandomTranslation layer",
                            "generate": ["tensorflow.keras.layers.RandomTranslation"],
                        },
                        {
                            "path": "random_rotation",
                            "title": "RandomRotation layer",
                            "generate": ["tensorflow.keras.layers.RandomRotation"],
                        },
                        {
                            "path": "random_zoom",
                            "title": "RandomZoom layer",
                            "generate": ["tensorflow.keras.layers.RandomZoom"],
                        },
                        {
                            "path": "random_height",
                            "title": "RandomHeight layer",
                            "generate": ["tensorflow.keras.layers.RandomHeight"],
                        },
                        {
                            "path": "random_width",
                            "title": "RandomWidth layer",
                            "generate": ["tensorflow.keras.layers.RandomWidth"],
                        },
                        {
                            "path": "random_contrast",
                            "title": "RandomContrast layer",
                            "generate": ["tensorflow.keras.layers.RandomContrast"],
                        },
                        {
                            "path": "random_brightness",
                            "title": "RandomBrightness layer",
                            "generate": ["tensorflow.keras.layers.RandomBrightness"],
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
                    "generate": ["tensorflow.keras.layers.BatchNormalization"],
                },
                {
                    "path": "layer_normalization",
                    "title": "LayerNormalization layer",
                    "generate": ["tensorflow.keras.layers.LayerNormalization"],
                },
                {
                    "path": "unit_normalization",
                    "title": "UnitNormalization layer",
                    "generate": ["tensorflow.keras.layers.UnitNormalization"],
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
                    "generate": ["tensorflow.keras.layers.Dropout"],
                },
                {
                    "path": "spatial_dropout1d",
                    "title": "SpatialDropout1D layer",
                    "generate": ["tensorflow.keras.layers.SpatialDropout1D"],
                },
                {
                    "path": "spatial_dropout2d",
                    "title": "SpatialDropout2D layer",
                    "generate": ["tensorflow.keras.layers.SpatialDropout2D"],
                },
                {
                    "path": "spatial_dropout3d",
                    "title": "SpatialDropout3D layer",
                    "generate": ["tensorflow.keras.layers.SpatialDropout3D"],
                },
                {
                    "path": "gaussian_dropout",
                    "title": "GaussianDropout layer",
                    "generate": ["tensorflow.keras.layers.GaussianDropout"],
                },
                {
                    "path": "gaussian_noise",
                    "title": "GaussianNoise layer",
                    "generate": ["tensorflow.keras.layers.GaussianNoise"],
                },
                {
                    "path": "activity_regularization",
                    "title": "ActivityRegularization layer",
                    "generate": ["tensorflow.keras.layers.ActivityRegularization"],
                },
                {
                    "path": "alpha_dropout",
                    "title": "AlphaDropout layer",
                    "generate": ["tensorflow.keras.layers.AlphaDropout"],
                },
            ],
        },
        {
            "path": "attention_layers/",
            "title": "Attention layers",
            "toc": True,
            "children": [
                {
                    "path": "multi_head_attention",
                    "title": "MultiHeadAttention layer",
                    "generate": ["tensorflow.keras.layers.MultiHeadAttention"],
                },
                {
                    "path": "attention",
                    "title": "Attention layer",
                    "generate": ["tensorflow.keras.layers.Attention"],
                },
                {
                    "path": "additive_attention",
                    "title": "AdditiveAttention layer",
                    "generate": ["tensorflow.keras.layers.AdditiveAttention"],
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
                    "generate": ["tensorflow.keras.layers.Reshape"],
                },
                {
                    "path": "flatten",
                    "title": "Flatten layer",
                    "generate": ["tensorflow.keras.layers.Flatten"],
                },
                {
                    "path": "repeat_vector",
                    "title": "RepeatVector layer",
                    "generate": ["tensorflow.keras.layers.RepeatVector"],
                },
                {
                    "path": "permute",
                    "title": "Permute layer",
                    "generate": ["tensorflow.keras.layers.Permute"],
                },
                {
                    "path": "cropping1d",
                    "title": "Cropping1D layer",
                    "generate": ["tensorflow.keras.layers.Cropping1D"],
                },
                {
                    "path": "cropping2d",
                    "title": "Cropping2D layer",
                    "generate": ["tensorflow.keras.layers.Cropping2D"],
                },
                {
                    "path": "cropping3d",
                    "title": "Cropping3D layer",
                    "generate": ["tensorflow.keras.layers.Cropping3D"],
                },
                {
                    "path": "up_sampling1d",
                    "title": "UpSampling1D layer",
                    "generate": ["tensorflow.keras.layers.UpSampling1D"],
                },
                {
                    "path": "up_sampling2d",
                    "title": "UpSampling2D layer",
                    "generate": ["tensorflow.keras.layers.UpSampling2D"],
                },
                {
                    "path": "up_sampling3d",
                    "title": "UpSampling3D layer",
                    "generate": ["tensorflow.keras.layers.UpSampling3D"],
                },
                {
                    "path": "zero_padding1d",
                    "title": "ZeroPadding1D layer",
                    "generate": ["tensorflow.keras.layers.ZeroPadding1D"],
                },
                {
                    "path": "zero_padding2d",
                    "title": "ZeroPadding2D layer",
                    "generate": ["tensorflow.keras.layers.ZeroPadding2D"],
                },
                {
                    "path": "zero_padding3d",
                    "title": "ZeroPadding3D layer",
                    "generate": ["tensorflow.keras.layers.ZeroPadding3D"],
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
                    "generate": ["tensorflow.keras.layers.Concatenate"],
                },
                {
                    "path": "average",
                    "title": "Average layer",
                    "generate": ["tensorflow.keras.layers.Average"],
                },
                {
                    "path": "maximum",
                    "title": "Maximum layer",
                    "generate": ["tensorflow.keras.layers.Maximum"],
                },
                {
                    "path": "minimum",
                    "title": "Minimum layer",
                    "generate": ["tensorflow.keras.layers.Minimum"],
                },
                {
                    "path": "add",
                    "title": "Add layer",
                    "generate": ["tensorflow.keras.layers.Add"],
                },
                {
                    "path": "subtract",
                    "title": "Subtract layer",
                    "generate": ["tensorflow.keras.layers.Subtract"],
                },
                {
                    "path": "multiply",
                    "title": "Multiply layer",
                    "generate": ["tensorflow.keras.layers.Multiply"],
                },
                {
                    "path": "dot",
                    "title": "Dot layer",
                    "generate": ["tensorflow.keras.layers.Dot"],
                },
            ],
        },
        {
            "path": "locally_connected_layers/",
            "title": "Locally-connected layers",
            "toc": True,
            "children": [
                {
                    "path": "locall_connected1d",
                    "title": "LocallyConnected1D layer",
                    "generate": ["tensorflow.keras.layers.LocallyConnected1D"],
                },
                {
                    "path": "locall_connected2d",
                    "title": "LocallyConnected2D layer",
                    "generate": ["tensorflow.keras.layers.LocallyConnected2D"],
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
                    "generate": ["tensorflow.keras.layers.ReLU"],
                },
                {
                    "path": "softmax",
                    "title": "Softmax layer",
                    "generate": ["tensorflow.keras.layers.Softmax"],
                },
                {
                    "path": "leaky_relu",
                    "title": "LeakyReLU layer",
                    "generate": ["tensorflow.keras.layers.LeakyReLU"],
                },
                {
                    "path": "prelu",
                    "title": "PReLU layer",
                    "generate": ["tensorflow.keras.layers.PReLU"],
                },
                {
                    "path": "elu",
                    "title": "ELU layer",
                    "generate": ["tensorflow.keras.layers.ELU"],
                },
                {
                    "path": "threshold_relu",
                    "title": "ThresholdedReLU layer",
                    "generate": ["tensorflow.keras.layers.ThresholdedReLU"],
                },
            ],
        },
    ],
}
