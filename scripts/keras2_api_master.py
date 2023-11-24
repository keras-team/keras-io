try:
    import tf_keras
except Exception as e:
    print(f"Could not import tf_keras. Exception: {e}")
    tf_keras = None

if tf_keras:
    parts = tf_keras.__version__.split(".")
    tf_keras_version = parts[0] + "." + parts[1]
else:
    tf_keras_version = "None"


# In order to refresh the pages for an old version (e.g. 2.14)
# You will need to re-run `python autogen.py make` after updating
# the tf_keras version in autogen.py and making sure to install
# the targeted keras_version locally. When you do this
# `/2.14/api/` (for instance) will be regenerated. You can then
# just reupload, which won't affect the directories for any other
# version number.
KERAS2_API_MASTER = {
    "path": tf_keras_version + "/api/",
    "title": "Keras 2 API documentation",
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
                        "tf_keras.Model",
                        "tf_keras.Model.summary",
                        "tf_keras.Model.get_layer",
                    ],
                },
                {
                    "path": "sequential",
                    "title": "The Sequential class",
                    "generate": [
                        "tf_keras.Sequential",
                        "tf_keras.Sequential.add",
                        "tf_keras.Sequential.pop",
                    ],
                },
                {
                    "path": "model_training_apis",
                    "title": "Model training APIs",
                    "generate": [
                        "tf_keras.Model.compile",
                        "tf_keras.Model.fit",
                        "tf_keras.Model.evaluate",
                        "tf_keras.Model.predict",
                        "tf_keras.Model.train_on_batch",
                        "tf_keras.Model.test_on_batch",
                        "tf_keras.Model.predict_on_batch",
                        "tf_keras.Model.run_eagerly",
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
                                "tf_keras.Model.save",
                                "tf_keras.saving.save_model",
                                "tf_keras.saving.load_model",
                            ],
                        },
                        {
                            "path": "weights_saving_and_loading",
                            "title": "Weights-only saving & loading",
                            "generate": [
                                "tf_keras.Model.get_weights",
                                "tf_keras.Model.set_weights",
                                "tf_keras.Model.save_weights",
                                "tf_keras.Model.load_weights",
                            ],
                        },
                        {
                            "path": "model_config_serialization",
                            "title": "Model config serialization",
                            "generate": [
                                "tf_keras.Model.get_config",
                                "tf_keras.Model.from_config",
                                "tf_keras.models.clone_model",
                            ],
                        },
                        {
                            "path": "export",
                            "title": "Model export for inference",
                            "generate": [
                                "tf_keras.export.ExportArchive",
                            ],
                        },
                        {
                            "path": "serialization_utils",
                            "title": "Serialization utilities",
                            "generate": [
                                "tf_keras.utils.serialize_keras_object",  # TODO: move to saving
                                "tf_keras.utils.deserialize_keras_object",  # TODO: move to saving
                                "tf_keras.saving.custom_object_scope",
                                "tf_keras.saving.get_custom_objects",
                                "tf_keras.saving.register_keras_serializable",
                            ],
                        },
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
                        "tf_keras.layers.Layer",
                        "tf_keras.layers.Layer.weights",
                        "tf_keras.layers.Layer.trainable_weights",
                        "tf_keras.layers.Layer.non_trainable_weights",
                        "tf_keras.layers.Layer.add_weight",
                        "tf_keras.layers.Layer.trainable",
                        "tf_keras.layers.Layer.get_weights",
                        "tf_keras.layers.Layer.set_weights",
                        "tf_keras.Model.get_config",
                        "tf_keras.layers.Layer.add_loss",
                        "tf_keras.layers.Layer.losses",
                    ],
                },
                {
                    "path": "activations",
                    "title": "Layer activations",
                    "generate": [
                        "tf_keras.activations.relu",
                        "tf_keras.activations.sigmoid",
                        "tf_keras.activations.softmax",
                        "tf_keras.activations.softplus",
                        "tf_keras.activations.softsign",
                        "tf_keras.activations.tanh",
                        "tf_keras.activations.selu",
                        "tf_keras.activations.elu",
                        "tf_keras.activations.exponential",
                    ],
                },
                {
                    "path": "initializers",
                    "title": "Layer weight initializers",
                    "generate": [
                        "tf_keras.initializers.RandomNormal",
                        "tf_keras.initializers.RandomUniform",
                        "tf_keras.initializers.TruncatedNormal",
                        "tf_keras.initializers.Zeros",
                        "tf_keras.initializers.Ones",
                        "tf_keras.initializers.GlorotNormal",
                        "tf_keras.initializers.GlorotUniform",
                        "tf_keras.initializers.HeNormal",
                        "tf_keras.initializers.HeUniform",
                        "tf_keras.initializers.Identity",
                        "tf_keras.initializers.Orthogonal",
                        "tf_keras.initializers.Constant",
                        "tf_keras.initializers.VarianceScaling",
                    ],
                },
                {
                    "path": "regularizers",
                    "title": "Layer weight regularizers",
                    "generate": [
                        "tf_keras.regularizers.L1",
                        "tf_keras.regularizers.L2",
                        "tf_keras.regularizers.L1L2",
                        "tf_keras.regularizers.OrthogonalRegularizer",
                    ],
                },
                {
                    "path": "constraints",
                    "title": "Layer weight constraints",
                    "generate": [
                        "tf_keras.constraints.MaxNorm",
                        "tf_keras.constraints.MinMaxNorm",
                        "tf_keras.constraints.NonNeg",
                        "tf_keras.constraints.UnitNorm",
                        "tf_keras.constraints.RadialConstraint",
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
                            "generate": ["tf_keras.Input"],
                        },
                        {
                            "path": "dense",
                            "title": "Dense layer",
                            "generate": ["tf_keras.layers.Dense"],
                        },
                        {
                            "path": "activation",
                            "title": "Activation layer",
                            "generate": ["tf_keras.layers.Activation"],
                        },
                        {
                            "path": "embedding",
                            "title": "Embedding layer",
                            "generate": ["tf_keras.layers.Embedding"],
                        },
                        {
                            "path": "masking",
                            "title": "Masking layer",
                            "generate": ["tf_keras.layers.Masking"],
                        },
                        {
                            "path": "lambda",
                            "title": "Lambda layer",
                            "generate": ["tf_keras.layers.Lambda"],
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
                            "generate": ["tf_keras.layers.Conv1D"],
                        },
                        {
                            "path": "convolution2d",
                            "title": "Conv2D layer",
                            "generate": ["tf_keras.layers.Conv2D"],
                        },
                        {
                            "path": "convolution3d",
                            "title": "Conv3D layer",
                            "generate": ["tf_keras.layers.Conv3D"],
                        },
                        {
                            "path": "separable_convolution1d",
                            "title": "SeparableConv1D layer",
                            "generate": ["tf_keras.layers.SeparableConv1D"],
                        },
                        {
                            "path": "separable_convolution2d",
                            "title": "SeparableConv2D layer",
                            "generate": ["tf_keras.layers.SeparableConv2D"],
                        },
                        {
                            "path": "depthwise_convolution2d",
                            "title": "DepthwiseConv2D layer",
                            "generate": ["tf_keras.layers.DepthwiseConv2D"],
                        },
                        {
                            "path": "convolution1d_transpose",
                            "title": "Conv1DTranspose layer",
                            "generate": ["tf_keras.layers.Conv1DTranspose"],
                        },
                        {
                            "path": "convolution2d_transpose",
                            "title": "Conv2DTranspose layer",
                            "generate": ["tf_keras.layers.Conv2DTranspose"],
                        },
                        {
                            "path": "convolution3d_transpose",
                            "title": "Conv3DTranspose layer",
                            "generate": ["tf_keras.layers.Conv3DTranspose"],
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
                            "generate": ["tf_keras.layers.MaxPooling1D"],
                        },
                        {
                            "path": "max_pooling2d",
                            "title": "MaxPooling2D layer",
                            "generate": ["tf_keras.layers.MaxPooling2D"],
                        },
                        {
                            "path": "max_pooling3d",
                            "title": "MaxPooling3D layer",
                            "generate": ["tf_keras.layers.MaxPooling3D"],
                        },
                        {
                            "path": "average_pooling1d",
                            "title": "AveragePooling1D layer",
                            "generate": ["tf_keras.layers.AveragePooling1D"],
                        },
                        {
                            "path": "average_pooling2d",
                            "title": "AveragePooling2D layer",
                            "generate": ["tf_keras.layers.AveragePooling2D"],
                        },
                        {
                            "path": "average_pooling3d",
                            "title": "AveragePooling3D layer",
                            "generate": ["tf_keras.layers.AveragePooling3D"],
                        },
                        {
                            "path": "global_max_pooling1d",
                            "title": "GlobalMaxPooling1D layer",
                            "generate": ["tf_keras.layers.GlobalMaxPooling1D"],
                        },
                        {
                            "path": "global_max_pooling2d",
                            "title": "GlobalMaxPooling2D layer",
                            "generate": ["tf_keras.layers.GlobalMaxPooling2D"],
                        },
                        {
                            "path": "global_max_pooling3d",
                            "title": "GlobalMaxPooling3D layer",
                            "generate": ["tf_keras.layers.GlobalMaxPooling3D"],
                        },
                        {
                            "path": "global_average_pooling1d",
                            "title": "GlobalAveragePooling1D layer",
                            "generate": ["tf_keras.layers.GlobalAveragePooling1D"],
                        },
                        {
                            "path": "global_average_pooling2d",
                            "title": "GlobalAveragePooling2D layer",
                            "generate": ["tf_keras.layers.GlobalAveragePooling2D"],
                        },
                        {
                            "path": "global_average_pooling3d",
                            "title": "GlobalAveragePooling3D layer",
                            "generate": ["tf_keras.layers.GlobalAveragePooling3D"],
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
                            "generate": ["tf_keras.layers.LSTM"],
                        },
                        {
                            "path": "gru",
                            "title": "GRU layer",
                            "generate": ["tf_keras.layers.GRU"],
                        },
                        {
                            "path": "simple_rnn",
                            "title": "SimpleRNN layer",
                            "generate": ["tf_keras.layers.SimpleRNN"],
                        },
                        {
                            "path": "time_distributed",
                            "title": "TimeDistributed layer",
                            "generate": ["tf_keras.layers.TimeDistributed"],
                        },
                        {
                            "path": "bidirectional",
                            "title": "Bidirectional layer",
                            "generate": ["tf_keras.layers.Bidirectional"],
                        },
                        {
                            "path": "conv_lstm1d",
                            "title": "ConvLSTM1D layer",
                            "generate": ["tf_keras.layers.ConvLSTM1D"],
                        },
                        {
                            "path": "conv_lstm2d",
                            "title": "ConvLSTM2D layer",
                            "generate": ["tf_keras.layers.ConvLSTM2D"],
                        },
                        {
                            "path": "conv_lstm3d",
                            "title": "ConvLSTM3D layer",
                            "generate": ["tf_keras.layers.ConvLSTM3D"],
                        },
                        {
                            "path": "rnn",
                            "title": "Base RNN layer",
                            "generate": ["tf_keras.layers.RNN"],
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
                                    "generate": ["tf_keras.layers.TextVectorization"],
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
                                    "generate": ["tf_keras.layers.Normalization"],
                                },
                                {
                                    "path": "discretization",
                                    "title": "Discretization layer",
                                    "generate": ["tf_keras.layers.Discretization"],
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
                                    "generate": ["tf_keras.layers.CategoryEncoding"],
                                },
                                {
                                    "path": "hashing",
                                    "title": "Hashing layer",
                                    "generate": ["tf_keras.layers.Hashing"],
                                },
                                {
                                    "path": "hashed_crossing",
                                    "title": "HashedCrossing layer",
                                    "generate": ["tf_keras.layers.HashedCrossing"],
                                },
                                {
                                    "path": "string_lookup",
                                    "title": "StringLookup layer",
                                    "generate": ["tf_keras.layers.StringLookup"],
                                },
                                {
                                    "path": "integer_lookup",
                                    "title": "IntegerLookup layer",
                                    "generate": ["tf_keras.layers.IntegerLookup"],
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
                                    "generate": ["tf_keras.layers.Resizing"],
                                },
                                {
                                    "path": "rescaling",
                                    "title": "Rescaling layer",
                                    "generate": ["tf_keras.layers.Rescaling"],
                                },
                                {
                                    "path": "center_crop",
                                    "title": "CenterCrop layer",
                                    "generate": ["tf_keras.layers.CenterCrop"],
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
                                    "generate": ["tf_keras.layers.RandomCrop"],
                                },
                                {
                                    "path": "random_flip",
                                    "title": "RandomFlip layer",
                                    "generate": ["tf_keras.layers.RandomFlip"],
                                },
                                {
                                    "path": "random_translation",
                                    "title": "RandomTranslation layer",
                                    "generate": ["tf_keras.layers.RandomTranslation"],
                                },
                                {
                                    "path": "random_rotation",
                                    "title": "RandomRotation layer",
                                    "generate": ["tf_keras.layers.RandomRotation"],
                                },
                                {
                                    "path": "random_zoom",
                                    "title": "RandomZoom layer",
                                    "generate": ["tf_keras.layers.RandomZoom"],
                                },
                                {
                                    "path": "random_contrast",
                                    "title": "RandomContrast layer",
                                    "generate": ["tf_keras.layers.RandomContrast"],
                                },
                                {
                                    "path": "random_brightness",
                                    "title": "RandomBrightness layer",
                                    "generate": ["tf_keras.layers.RandomBrightness"],
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
                            "generate": ["tf_keras.layers.BatchNormalization"],
                        },
                        {
                            "path": "layer_normalization",
                            "title": "LayerNormalization layer",
                            "generate": ["tf_keras.layers.LayerNormalization"],
                        },
                        {
                            "path": "unit_normalization",
                            "title": "UnitNormalization layer",
                            "generate": ["tf_keras.layers.UnitNormalization"],
                        },
                        {
                            "path": "group_normalization",
                            "title": "GroupNormalization layer",
                            "generate": ["tf_keras.layers.GroupNormalization"],
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
                            "generate": ["tf_keras.layers.Dropout"],
                        },
                        {
                            "path": "spatial_dropout1d",
                            "title": "SpatialDropout1D layer",
                            "generate": ["tf_keras.layers.SpatialDropout1D"],
                        },
                        {
                            "path": "spatial_dropout2d",
                            "title": "SpatialDropout2D layer",
                            "generate": ["tf_keras.layers.SpatialDropout2D"],
                        },
                        {
                            "path": "spatial_dropout3d",
                            "title": "SpatialDropout3D layer",
                            "generate": ["tf_keras.layers.SpatialDropout3D"],
                        },
                        {
                            "path": "gaussian_dropout",
                            "title": "GaussianDropout layer",
                            "generate": ["tf_keras.layers.GaussianDropout"],
                        },
                        {
                            "path": "gaussian_noise",
                            "title": "GaussianNoise layer",
                            "generate": ["tf_keras.layers.GaussianNoise"],
                        },
                        {
                            "path": "activity_regularization",
                            "title": "ActivityRegularization layer",
                            "generate": ["tf_keras.layers.ActivityRegularization"],
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
                            "generate": ["tf_keras.layers.MultiHeadAttention"],
                        },
                        {
                            "path": "attention",
                            "title": "Attention layer",
                            "generate": ["tf_keras.layers.Attention"],
                        },
                        {
                            "path": "additive_attention",
                            "title": "AdditiveAttention layer",
                            "generate": ["tf_keras.layers.AdditiveAttention"],
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
                            "generate": ["tf_keras.layers.Reshape"],
                        },
                        {
                            "path": "flatten",
                            "title": "Flatten layer",
                            "generate": ["tf_keras.layers.Flatten"],
                        },
                        {
                            "path": "repeat_vector",
                            "title": "RepeatVector layer",
                            "generate": ["tf_keras.layers.RepeatVector"],
                        },
                        {
                            "path": "permute",
                            "title": "Permute layer",
                            "generate": ["tf_keras.layers.Permute"],
                        },
                        {
                            "path": "cropping1d",
                            "title": "Cropping1D layer",
                            "generate": ["tf_keras.layers.Cropping1D"],
                        },
                        {
                            "path": "cropping2d",
                            "title": "Cropping2D layer",
                            "generate": ["tf_keras.layers.Cropping2D"],
                        },
                        {
                            "path": "cropping3d",
                            "title": "Cropping3D layer",
                            "generate": ["tf_keras.layers.Cropping3D"],
                        },
                        {
                            "path": "up_sampling1d",
                            "title": "UpSampling1D layer",
                            "generate": ["tf_keras.layers.UpSampling1D"],
                        },
                        {
                            "path": "up_sampling2d",
                            "title": "UpSampling2D layer",
                            "generate": ["tf_keras.layers.UpSampling2D"],
                        },
                        {
                            "path": "up_sampling3d",
                            "title": "UpSampling3D layer",
                            "generate": ["tf_keras.layers.UpSampling3D"],
                        },
                        {
                            "path": "zero_padding1d",
                            "title": "ZeroPadding1D layer",
                            "generate": ["tf_keras.layers.ZeroPadding1D"],
                        },
                        {
                            "path": "zero_padding2d",
                            "title": "ZeroPadding2D layer",
                            "generate": ["tf_keras.layers.ZeroPadding2D"],
                        },
                        {
                            "path": "zero_padding3d",
                            "title": "ZeroPadding3D layer",
                            "generate": ["tf_keras.layers.ZeroPadding3D"],
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
                            "generate": ["tf_keras.layers.Concatenate"],
                        },
                        {
                            "path": "average",
                            "title": "Average layer",
                            "generate": ["tf_keras.layers.Average"],
                        },
                        {
                            "path": "maximum",
                            "title": "Maximum layer",
                            "generate": ["tf_keras.layers.Maximum"],
                        },
                        {
                            "path": "minimum",
                            "title": "Minimum layer",
                            "generate": ["tf_keras.layers.Minimum"],
                        },
                        {
                            "path": "add",
                            "title": "Add layer",
                            "generate": ["tf_keras.layers.Add"],
                        },
                        {
                            "path": "subtract",
                            "title": "Subtract layer",
                            "generate": ["tf_keras.layers.Subtract"],
                        },
                        {
                            "path": "multiply",
                            "title": "Multiply layer",
                            "generate": ["tf_keras.layers.Multiply"],
                        },
                        {
                            "path": "dot",
                            "title": "Dot layer",
                            "generate": ["tf_keras.layers.Dot"],
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
                            "generate": ["tf_keras.layers.ReLU"],
                        },
                        {
                            "path": "softmax",
                            "title": "Softmax layer",
                            "generate": ["tf_keras.layers.Softmax"],
                        },
                        {
                            "path": "leaky_relu",
                            "title": "LeakyReLU layer",
                            "generate": ["tf_keras.layers.LeakyReLU"],
                        },
                        {
                            "path": "prelu",
                            "title": "PReLU layer",
                            "generate": ["tf_keras.layers.PReLU"],
                        },
                        {
                            "path": "elu",
                            "title": "ELU layer",
                            "generate": ["tf_keras.layers.ELU"],
                        },
                        {
                            "path": "threshold_relu",
                            "title": "ThresholdedReLU layer",
                            "generate": ["tf_keras.layers.ThresholdedReLU"],
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
                    "generate": ["tf_keras.callbacks.Callback"],
                },
                {
                    "path": "model_checkpoint",
                    "title": "ModelCheckpoint",
                    "generate": ["tf_keras.callbacks.ModelCheckpoint"],
                },
                {
                    "path": "backup_and_restore",
                    "title": "BackupAndRestore",
                    "generate": ["tf_keras.callbacks.BackupAndRestore"],
                },
                {
                    "path": "tensorboard",
                    "title": "TensorBoard",
                    "generate": ["tf_keras.callbacks.TensorBoard"],
                },
                {
                    "path": "early_stopping",
                    "title": "EarlyStopping",
                    "generate": ["tf_keras.callbacks.EarlyStopping"],
                },
                {  # LEGACY
                    "path": "learning_rate_scheduler",
                    "title": "LearningRateScheduler",
                    "generate": ["tf_keras.callbacks.LearningRateScheduler"],
                },
                {
                    "path": "reduce_lr_on_plateau",
                    "title": "ReduceLROnPlateau",
                    "generate": ["tf_keras.callbacks.ReduceLROnPlateau"],
                },
                {
                    "path": "remote_monitor",
                    "title": "RemoteMonitor",
                    "generate": ["tf_keras.callbacks.RemoteMonitor"],
                },
                {
                    "path": "lambda_callback",
                    "title": "LambdaCallback",
                    "generate": ["tf_keras.callbacks.LambdaCallback"],
                },
                {
                    "path": "terminate_on_nan",
                    "title": "TerminateOnNaN",
                    "generate": ["tf_keras.callbacks.TerminateOnNaN"],
                },
                {
                    "path": "csv_logger",
                    "title": "CSVLogger",
                    "generate": ["tf_keras.callbacks.CSVLogger"],
                },
                {
                    "path": "progbar_logger",
                    "title": "ProgbarLogger",
                    "generate": ["tf_keras.callbacks.ProgbarLogger"],
                },
            ],
        },
        {
            "path": "optimizers/",
            "title": "Optimizers",
            "toc": True,
            "generate": [
                "tf_keras.optimizers.Optimizer.apply_gradients",
                "tf_keras.optimizers.Optimizer.variables",
            ],
            "children": [
                {
                    "path": "sgd",
                    "title": "SGD",
                    "generate": ["tf_keras.optimizers.SGD"],
                },
                {
                    "path": "rmsprop",
                    "title": "RMSprop",
                    "generate": ["tf_keras.optimizers.RMSprop"],
                },
                {
                    "path": "adam",
                    "title": "Adam",
                    "generate": ["tf_keras.optimizers.Adam"],
                },
                {
                    "path": "adamw",
                    "title": "AdamW",
                    "generate": ["tf_keras.optimizers.AdamW"],
                },
                {
                    "path": "adadelta",
                    "title": "Adadelta",
                    "generate": ["tf_keras.optimizers.Adadelta"],
                },
                {
                    "path": "adagrad",
                    "title": "Adagrad",
                    "generate": ["tf_keras.optimizers.Adagrad"],
                },
                {
                    "path": "adamax",
                    "title": "Adamax",
                    "generate": ["tf_keras.optimizers.Adamax"],
                },
                {
                    "path": "adafactor",
                    "title": "Adafactor",
                    "generate": ["tf_keras.optimizers.Adafactor"],
                },
                {
                    "path": "Nadam",
                    "title": "Nadam",
                    "generate": ["tf_keras.optimizers.Nadam"],
                },
                {
                    "path": "ftrl",
                    "title": "Ftrl",
                    "generate": ["tf_keras.optimizers.Ftrl"],
                },
                {
                    "path": "learning_rate_schedules/",
                    "title": "Learning rate schedules API",
                    "toc": True,
                    "skip_from_toc": True,
                    "children": [
                        {
                            "path": "exponential_decay",
                            "title": "ExponentialDecay",
                            "generate": [
                                "tf_keras.optimizers.schedules.ExponentialDecay"
                            ],
                        },
                        {
                            "path": "piecewise_constant_decay",
                            "title": "PiecewiseConstantDecay",
                            "generate": [
                                "tf_keras.optimizers.schedules.PiecewiseConstantDecay"
                            ],
                        },
                        {
                            "path": "polynomial_decay",
                            "title": "PolynomialDecay",
                            "generate": [
                                "tf_keras.optimizers.schedules.PolynomialDecay"
                            ],
                        },
                        {
                            "path": "inverse_time_decay",
                            "title": "InverseTimeDecay",
                            "generate": [
                                "tf_keras.optimizers.schedules.InverseTimeDecay"
                            ],
                        },
                        {
                            "path": "cosine_decay",
                            "title": "CosineDecay",
                            "generate": ["tf_keras.optimizers.schedules.CosineDecay"],
                        },
                        {
                            "path": "cosine_decay_restarts",
                            "title": "CosineDecayRestarts",
                            "generate": [
                                "tf_keras.optimizers.schedules.CosineDecayRestarts"
                            ],
                        },
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
                    "path": "accuracy_metrics",
                    "title": "Accuracy metrics",
                    "generate": [
                        "tf_keras.metrics.Accuracy",
                        "tf_keras.metrics.BinaryAccuracy",
                        "tf_keras.metrics.CategoricalAccuracy",
                        "tf_keras.metrics.SparseCategoricalAccuracy",
                        "tf_keras.metrics.TopKCategoricalAccuracy",
                        "tf_keras.metrics.SparseTopKCategoricalAccuracy",
                    ],
                },
                {
                    "path": "probabilistic_metrics",
                    "title": "Probabilistic metrics",
                    "generate": [
                        "tf_keras.metrics.BinaryCrossentropy",
                        "tf_keras.metrics.CategoricalCrossentropy",
                        "tf_keras.metrics.SparseCategoricalCrossentropy",
                        "tf_keras.metrics.KLDivergence",
                        "tf_keras.metrics.Poisson",
                    ],
                },
                {
                    "path": "regression_metrics",
                    "title": "Regression metrics",
                    "generate": [
                        "tf_keras.metrics.MeanSquaredError",
                        "tf_keras.metrics.RootMeanSquaredError",
                        "tf_keras.metrics.MeanAbsoluteError",
                        "tf_keras.metrics.MeanAbsolutePercentageError",
                        "tf_keras.metrics.MeanSquaredLogarithmicError",
                        "tf_keras.metrics.CosineSimilarity",
                        "tf_keras.metrics.LogCoshError",
                    ],
                },
                {
                    "path": "classification_metrics",
                    "title": "Classification metrics based on True/False positives & negatives",
                    "generate": [
                        "tf_keras.metrics.AUC",
                        "tf_keras.metrics.Precision",
                        "tf_keras.metrics.Recall",
                        "tf_keras.metrics.TruePositives",
                        "tf_keras.metrics.TrueNegatives",
                        "tf_keras.metrics.FalsePositives",
                        "tf_keras.metrics.FalseNegatives",
                        "tf_keras.metrics.PrecisionAtRecall",
                        "tf_keras.metrics.SensitivityAtSpecificity",
                        "tf_keras.metrics.SpecificityAtSensitivity",
                    ],
                },
                {
                    "path": "segmentation_metrics",
                    "title": "Image segmentation metrics",
                    "generate": ["tf_keras.metrics.MeanIoU"],
                },
                {
                    "path": "hinge_metrics",
                    "title": 'Hinge metrics for "maximum-margin" classification',
                    "generate": [
                        "tf_keras.metrics.Hinge",
                        "tf_keras.metrics.SquaredHinge",
                        "tf_keras.metrics.CategoricalHinge",
                    ],
                },
            ],
        },
        {
            "path": "losses/",
            "title": "Losses",
            "toc": True,
            "children": [
                {
                    "path": "probabilistic_losses",
                    "title": "Probabilistic losses",
                    "generate": [
                        "tf_keras.losses.BinaryCrossentropy",
                        "tf_keras.losses.CategoricalCrossentropy",
                        "tf_keras.losses.SparseCategoricalCrossentropy",
                        "tf_keras.losses.Poisson",
                        "tf_keras.losses.binary_crossentropy",
                        "tf_keras.losses.categorical_crossentropy",
                        "tf_keras.losses.sparse_categorical_crossentropy",
                        "tf_keras.losses.poisson",
                        "tf_keras.losses.KLDivergence",
                        "tf_keras.losses.kl_divergence",
                    ],
                },
                {
                    "path": "regression_losses",
                    "title": "Regression losses",
                    "generate": [
                        "tf_keras.losses.MeanSquaredError",
                        "tf_keras.losses.MeanAbsoluteError",
                        "tf_keras.losses.MeanAbsolutePercentageError",
                        "tf_keras.losses.MeanSquaredLogarithmicError",
                        "tf_keras.losses.CosineSimilarity",
                        "tf_keras.losses.mean_squared_error",
                        "tf_keras.losses.mean_absolute_error",
                        "tf_keras.losses.mean_absolute_percentage_error",
                        "tf_keras.losses.mean_squared_logarithmic_error",
                        "tf_keras.losses.cosine_similarity",
                        "tf_keras.losses.Huber",
                        "tf_keras.losses.huber",
                        "tf_keras.losses.LogCosh",
                        "tf_keras.losses.log_cosh",
                    ],
                },
                {
                    "path": "hinge_losses",
                    "title": 'Hinge losses for "maximum-margin" classification',
                    "generate": [
                        "tf_keras.losses.Hinge",
                        "tf_keras.losses.SquaredHinge",
                        "tf_keras.losses.CategoricalHinge",
                        "tf_keras.losses.hinge",
                        "tf_keras.losses.squared_hinge",
                        "tf_keras.losses.categorical_hinge",
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
                        "tf_keras.utils.image_dataset_from_directory",
                        "tf_keras.utils.load_img",
                        "tf_keras.utils.img_to_array",
                        "tf_keras.utils.save_img",
                        # 'tf_keras.preprocessing.image.ImageDataGenerator',  # LEGACY
                        # 'tf_keras.preprocessing.image.ImageDataGenerator.flow',  # LEGACY
                        # 'tf_keras.preprocessing.image.ImageDataGenerator.flow_from_dataframe',  # LEGACY
                        # 'tf_keras.preprocessing.image.ImageDataGenerator.flow_from_directory',  # LEGACY
                    ],
                },
                {
                    "path": "timeseries",
                    "title": "Timeseries data loading",
                    "generate": [
                        "tf_keras.utils.timeseries_dataset_from_array",
                        # 'tf_keras.preprocessing.sequence.pad_sequences',  # LEGACY
                        # 'tf_keras.preprocessing.sequence.TimeseriesGenerator',  # LEGACY
                    ],
                },
                {
                    "path": "text",
                    "title": "Text data loading",
                    "generate": [
                        "tf_keras.utils.text_dataset_from_directory",
                        # 'tf_keras.preprocessing.text.Tokenizer',  # LEGACY
                    ],
                },
                {
                    "path": "audio",
                    "title": "Audio data loading",
                    "generate": [
                        "tf_keras.utils.audio_dataset_from_directory",
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
                    "generate": ["tf_keras.datasets.mnist.load_data"],
                },
                {
                    "path": "cifar10",
                    "title": "CIFAR10 small images classification dataset",
                    "generate": ["tf_keras.datasets.cifar10.load_data"],
                },
                {
                    "path": "cifar100",
                    "title": "CIFAR100 small images classification dataset",
                    "generate": ["tf_keras.datasets.cifar100.load_data"],
                },
                {
                    "path": "imdb",
                    "title": "IMDB movie review sentiment classification dataset",
                    "generate": [
                        "tf_keras.datasets.imdb.load_data",
                        "tf_keras.datasets.imdb.get_word_index",
                    ],
                },
                {
                    "path": "reuters",
                    "title": "Reuters newswire classification dataset",
                    "generate": [
                        "tf_keras.datasets.reuters.load_data",
                        "tf_keras.datasets.reuters.get_word_index",
                    ],
                },
                {
                    "path": "fashion_mnist",
                    "title": "Fashion MNIST dataset, an alternative to MNIST",
                    "generate": ["tf_keras.datasets.fashion_mnist.load_data"],
                },
                {
                    "path": "boston_housing",
                    "title": "Boston Housing price regression dataset",
                    "generate": ["tf_keras.datasets.boston_housing.load_data"],
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
                    "generate": ["tf_keras.applications.Xception"],
                },
                {
                    "path": "efficientnet",
                    "title": "EfficientNet B0 to B7",
                    "generate": [
                        "tf_keras.applications.EfficientNetB0",
                        "tf_keras.applications.EfficientNetB1",
                        "tf_keras.applications.EfficientNetB2",
                        "tf_keras.applications.EfficientNetB3",
                        "tf_keras.applications.EfficientNetB4",
                        "tf_keras.applications.EfficientNetB5",
                        "tf_keras.applications.EfficientNetB6",
                        "tf_keras.applications.EfficientNetB7",
                    ],
                },
                {
                    "path": "efficientnet_v2",
                    "title": "EfficientNetV2 B0 to B3 and S, M, L",
                    "generate": [
                        "tf_keras.applications.EfficientNetV2B0",
                        "tf_keras.applications.EfficientNetV2B1",
                        "tf_keras.applications.EfficientNetV2B2",
                        "tf_keras.applications.EfficientNetV2B3",
                        "tf_keras.applications.EfficientNetV2S",
                        "tf_keras.applications.EfficientNetV2M",
                        "tf_keras.applications.EfficientNetV2L",
                    ],
                },
                {
                    "path": "convnext",
                    "title": "ConvNeXt Tiny, Small, Base, Large, XLarge",
                    "generate": [
                        "tf_keras.applications.ConvNeXtTiny",
                        "tf_keras.applications.ConvNeXtSmall",
                        "tf_keras.applications.ConvNeXtBase",
                        "tf_keras.applications.ConvNeXtLarge",
                        "tf_keras.applications.ConvNeXtXLarge",
                    ],
                },
                {
                    "path": "vgg",
                    "title": "VGG16 and VGG19",
                    "generate": [
                        "tf_keras.applications.VGG16",
                        "tf_keras.applications.VGG19",
                    ],
                },
                {
                    "path": "resnet",
                    "title": "ResNet and ResNetV2",
                    "generate": [
                        "tf_keras.applications.ResNet50",
                        "tf_keras.applications.ResNet101",
                        "tf_keras.applications.ResNet152",
                        "tf_keras.applications.ResNet50V2",
                        "tf_keras.applications.ResNet101V2",
                        "tf_keras.applications.ResNet152V2",
                    ],
                },
                {
                    "path": "mobilenet",
                    "title": "MobileNet, MobileNetV2, and MobileNetV3",
                    "generate": [
                        "tf_keras.applications.MobileNet",
                        "tf_keras.applications.MobileNetV2",
                        "tf_keras.applications.MobileNetV3Small",
                        "tf_keras.applications.MobileNetV3Large",
                    ],
                },
                {
                    "path": "densenet",
                    "title": "DenseNet",
                    "generate": [
                        "tf_keras.applications.DenseNet121",
                        "tf_keras.applications.DenseNet169",
                        "tf_keras.applications.DenseNet201",
                    ],
                },
                {
                    "path": "nasnet",
                    "title": "NasNetLarge and NasNetMobile",
                    "generate": [
                        "tf_keras.applications.NASNetLarge",
                        "tf_keras.applications.NASNetMobile",
                    ],
                },
                {
                    "path": "inceptionv3",
                    "title": "InceptionV3",
                    "generate": [
                        "tf_keras.applications.InceptionV3",
                    ],
                },
                {
                    "path": "inceptionresnetv2",
                    "title": "InceptionResNetV2",
                    "generate": [
                        "tf_keras.applications.InceptionResNetV2",
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
                        "tf_keras.mixed_precision.Policy",
                        "tf_keras.mixed_precision.global_policy",
                        "tf_keras.mixed_precision.set_global_policy",
                    ],
                },
                {
                    "path": "loss_scale_optimizer",
                    "title": "LossScaleOptimizer",
                    "generate": [
                        "tf_keras.mixed_precision.LossScaleOptimizer",
                    ],
                },
            ],
        },
        {
            "path": "utils/",
            "title": "Utilities",
            "toc": True,
            "children": [
                {
                    "path": "model_plotting_utils",
                    "title": "Model plotting utilities",
                    "generate": [
                        "tf_keras.utils.plot_model",
                        "tf_keras.utils.model_to_dot",
                    ],
                },
                {
                    "path": "feature_space",
                    "title": "Structured data preprocessing utilities",
                    "generate": [
                        "tf_keras.utils.FeatureSpace",
                    ],
                },
                {
                    "path": "python_utils",
                    "title": "Python & NumPy utilities",
                    "generate": [
                        "tf_keras.utils.set_random_seed",
                        "tf_keras.utils.split_dataset",
                        "tf_keras.utils.get_file",
                        "tf_keras.utils.Progbar",
                        "tf_keras.utils.Sequence",
                        "tf_keras.utils.to_categorical",
                        "tf_keras.utils.to_ordinal",
                        "tf_keras.utils.normalize",
                    ],
                },
                {
                    "path": "backend_utils",
                    "title": "Backend utilities",
                    "generate": [
                        "tf_keras.backend.clear_session",
                        "tf_keras.backend.floatx",
                        "tf_keras.backend.set_floatx",
                        "tf_keras.backend.image_data_format",
                        "tf_keras.backend.set_image_data_format",
                        "tf_keras.backend.epsilon",
                        "tf_keras.backend.set_epsilon",
                        "tf_keras.backend.is_keras_tensor",
                        "tf_keras.backend.get_uid",
                        "tf_keras.backend.rnn",
                    ],
                },
            ],
        },
    ],
}
