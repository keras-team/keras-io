EXAMPLES_MASTER = {
    "path": "examples/",
    "title": "Code examples",
    "toc": False,
    "children": [
        {
            "path": "vision/",
            "title": "Computer Vision",
            "toc": True,
            "children": [
                # Image classification
                {
                    "path": "image_classification_from_scratch",
                    "title": "Image classification from scratch",
                    "subcategory": "Image classification",
                    "highlight": True,
                },
                {
                    "path": "mnist_convnet",
                    "title": "Simple MNIST convnet",
                    "subcategory": "Image classification",
                    "highlight": True,
                },
                {
                    "path": "image_classification_efficientnet_fine_tuning",
                    "title": "Image classification via fine-tuning with EfficientNet",
                    "subcategory": "Image classification",
                    "highlight": True,
                },
                {
                    "path": "image_classification_with_vision_transformer",
                    "title": "Image classification with Vision Transformer",
                    "subcategory": "Image classification",
                },
                {
                    "path": "bit",
                    "title": "Image Classification using BigTransfer (BiT)",
                    "subcategory": "Image classification",
                },
                {
                    "path": "attention_mil_classification",
                    "title": "Classification using Attention-based Deep Multiple Instance Learning",
                    "subcategory": "Image classification",
                },
                {
                    "path": "mlp_image_classification",
                    "title": "Image classification with modern MLP models",
                    "subcategory": "Image classification",
                },
                {
                    "path": "mobilevit",
                    "title": "A mobile-friendly Transformer-based model for image classification",
                    "subcategory": "Image classification",
                },
                {
                    "path": "xray_classification_with_tpus",
                    "title": "Pneumonia Classification on TPU",
                    "subcategory": "Image classification",
                },
                {
                    "path": "cct",
                    "title": "Compact Convolutional Transformers",
                    "subcategory": "Image classification",
                },
                {
                    "path": "convmixer",
                    "title": "Image classification with ConvMixer",
                    "subcategory": "Image classification",
                },
                {
                    "path": "eanet",
                    "title": "Image classification with EANet (External Attention Transformer)",
                    "subcategory": "Image classification",
                },
                {
                    "path": "involution",
                    "title": "Involutional neural networks",
                    "subcategory": "Image classification",
                },
                {
                    "path": "perceiver_image_classification",
                    "title": "Image classification with Perceiver",
                    "subcategory": "Image classification",
                },
                {
                    "path": "reptile",
                    "title": "Few-Shot learning with Reptile",
                    "subcategory": "Image classification",
                },
                {
                    "path": "semisupervised_simclr",
                    "title": "Semi-supervised image classification using contrastive pretraining with SimCLR",
                    "subcategory": "Image classification",
                },
                {
                    "path": "swin_transformers",
                    "title": "Image classification with Swin Transformers",
                    "subcategory": "Image classification",
                },
                {
                    "path": "vit_small_ds",
                    "title": "Train a Vision Transformer on small datasets",
                    "subcategory": "Image classification",
                },
                {
                    "path": "shiftvit",
                    "title": "A Vision Transformer without Attention",
                    "subcategory": "Image classification",
                },
                # Image segmentation
                {
                    "path": "oxford_pets_image_segmentation",
                    "title": "Image segmentation with a U-Net-like architecture",
                    "subcategory": "Image segmentation",
                    "highlight": True,
                },
                {
                    "path": "deeplabv3_plus",
                    "title": "Multiclass semantic segmentation using DeepLabV3+",
                    "subcategory": "Image segmentation",
                },
                # Object Detection
                {
                    "path": "retinanet",
                    "title": "Object Detection with RetinaNet",
                    "subcategory": "Object detection",
                },
                {
                    "path": "keypoint_detection",
                    "title": "Keypoint Detection with Transfer Learning",
                    "subcategory": "Object detection",
                },
                {
                    "path": "object_detection_using_vision_transformer",
                    "title": "Object detection with Vision Transformers",
                    "subcategory": "Object detection",
                },
                # 3D
                {
                    "path": "3D_image_classification",
                    "title": "3D image classification from CT scans",
                    "subcategory": "3D",
                },
                {
                    "path": "depth_estimation",
                    "title": "Monocular depth estimation",
                    "subcategory": "3D",
                },
                {
                    "path": "nerf",
                    "title": "3D volumetric rendering with NeRF",
                    "subcategory": "3D",
                },
                {
                    "path": "pointnet",
                    "title": "Point cloud classification",
                    "subcategory": "3D",
                },
                # OCR
                {
                    "path": "captcha_ocr",
                    "title": "OCR model for reading Captchas",
                    "subcategory": "OCR",
                    "highlight": True,
                },
                {
                    "path": "handwriting_recognition",
                    "title": "Handwriting recognition",
                    "subcategory": "OCR",
                },
                # Image enhancement
                {
                    "path": "autoencoder",
                    "title": "Convolutional autoencoder for image denoising",
                    "subcategory": "Image enhancement",
                },
                {
                    "path": "mirnet",
                    "title": "Low-light image enhancement using MIRNet",
                    "subcategory": "Image enhancement",
                },
                {
                    "path": "super_resolution_sub_pixel",
                    "title": "Image Super-Resolution using an Efficient Sub-Pixel CNN",
                    "subcategory": "Image enhancement",
                },
                {
                    "path": "edsr",
                    "title": "Enhanced Deep Residual Networks for single-image super-resolution",
                    "subcategory": "Image enhancement",
                },
                {
                    "path": "zero_dce",
                    "title": "Zero-DCE for low-light image enhancement",
                    "subcategory": "Image enhancement",
                },
                # Data augmentation
                {
                    "path": "cutmix",
                    "title": "CutMix data augmentation for image classification",
                    "subcategory": "Data augmentation",
                },
                {
                    "path": "mixup",
                    "title": "MixUp augmentation for image classification",
                    "subcategory": "Data augmentation",
                },
                {
                    "path": "randaugment",
                    "title": "RandAugment for Image Classification for Improved Robustness",
                    "subcategory": "Data augmentation",
                },
                # Image & Text
                {
                    "path": "image_captioning",
                    "title": "Image captioning",
                    "subcategory": "Image & Text",
                },
                {
                    "path": "nl_image_search",
                    "title": "Natural language image search with a Dual Encoder",
                    "subcategory": "Image & Text",
                },
                # Vision models interpretability
                {
                    "path": "visualizing_what_convnets_learn",
                    "title": "Visualizing what convnets learn",
                    "subcategory": "Vision models interpretability",
                },
                {
                    "path": "integrated_gradients",
                    "title": "Model interpretability with Integrated Gradients",
                    "subcategory": "Vision models interpretability",
                },
                {
                    "path": "probing_vits",
                    "title": "Investigating Vision Transformer representations",
                    "subcategory": "Vision models interpretability",
                },
                {
                    "path": "grad_cam",
                    "title": "Grad-CAM class activation visualization",
                    "subcategory": "Vision models interpretability",
                },
                # Image similarity search
                {
                    "path": "near_dup_search",
                    "title": "Near-duplicate image search",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "semantic_image_clustering",
                    "title": "Semantic Image Clustering",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "siamese_contrastive",
                    "title": "Image similarity estimation using a Siamese Network with a contrastive loss",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "siamese_network",
                    "title": "Image similarity estimation using a Siamese Network with a triplet loss",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "metric_learning",
                    "title": "Metric learning for image similarity search",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "metric_learning_tf_similarity",
                    "title": "Metric learning for image similarity search using TensorFlow Similarity",
                    "subcategory": "Image similarity search",
                },
                # Video
                {
                    "path": "video_classification",
                    "title": "Video Classification with a CNN-RNN Architecture",
                    "subcategory": "Video",
                },
                {
                    "path": "conv_lstm",
                    "title": "Next-Frame Video Prediction with Convolutional LSTMs",
                    "subcategory": "Video",
                },
                {
                    "path": "video_transformers",
                    "title": "Video Classification with Transformers",
                    "subcategory": "Video",
                },
                {
                    "path": "vivit",
                    "title": "Video Vision Transformer",
                    "subcategory": "Video",
                },
            ],
        },
        {
            "path": "nlp/",
            "title": "Natural Language Processing",
            "toc": True,
            "children": [
                # Text classification
                {
                    "path": "text_classification_from_scratch",
                    "title": "Text classification from scratch",
                    "subcategory": "Text classification",
                    "highlight": True,
                },
                {
                    "path": "active_learning_review_classification",
                    "title": "Review Classification using Active Learning",
                    "subcategory": "Text classification",
                },
                {
                    "path": "fnet_classification_with_keras_nlp",
                    "title": "Text Classification using FNet",
                    "subcategory": "Text classification",
                },
                {
                    "path": "multi_label_classification",
                    "title": "Large-scale multi-label text classification",
                    "subcategory": "Text classification",
                },
                {
                    "path": "text_classification_with_transformer",
                    "title": "Text classification with Transformer",
                    "subcategory": "Text classification",
                },
                {
                    "path": "text_classification_with_switch_transformer",
                    "title": "Text classification with Switch Transformer",
                    "subcategory": "Text classification",
                },
                {
                    "path": "tweet-classification-using-tfdf",
                    "title": "Text classification using Decision Forests and pretrained embeddings",
                    "subcategory": "Text classification",
                },
                {
                    "path": "pretrained_word_embeddings",
                    "title": "Using pre-trained word embeddings",
                    "subcategory": "Text classification",
                },
                {
                    "path": "bidirectional_lstm_imdb",
                    "title": "Bidirectional LSTM on IMDB",
                    "subcategory": "Text classification",
                },
                # Machine translation
                {
                    "path": "neural_machine_translation_with_keras_nlp",
                    "title": "English-to-Spanish translation with KerasNLP",
                    "subcategory": "Machine translation",
                    "highlight": True,
                },
                {
                    "path": "neural_machine_translation_with_transformer",
                    "title": "English-to-Spanish translation with a sequence-to-sequence Transformer",
                    "subcategory": "Machine translation",
                },
                {
                    "path": "lstm_seq2seq",
                    "title": "Character-level recurrent sequence-to-sequence model",
                    "subcategory": "Machine translation",
                },
                # Entailement prediction
                {
                    "path": "multimodal_entailment",
                    "title": "Multimodal entailment",
                    "subcategory": "Entailment prediction",
                },
                # Named entity recognition
                {
                    "path": "ner_transformers",
                    "title": "Named Entity Recognition using Transformers",
                    "subcategory": "Named entity recognition",
                },
                # Sequence-to-sequence
                {
                    "path": "text_extraction_with_bert",
                    "title": "Text Extraction with BERT",
                    "subcategory": "Sequence-to-sequence",
                },
                {
                    "path": "addition_rnn",
                    "title": "Sequence to sequence learning for performing number addition",
                    "subcategory": "Sequence-to-sequence",
                },
                # Text similarity search
                {
                    "path": "semantic_similarity_with_bert",
                    "title": "Semantic Similarity with BERT",
                    "subcategory": "Text similarity search",
                },
                # Language modeling
                {
                    "path": "masked_language_modeling",
                    "title": "End-to-end Masked Language Modeling with BERT",
                    "subcategory": "Language modeling",
                },
                {
                    "path": "pretraining_BERT",
                    "title": "Pretraining BERT with Hugging Face Transformers",
                    "subcategory": "Language modeling",
                },
                # Remainder is autogenerated
            ],
        },
        {
            "path": "structured_data/",
            "title": "Structured Data",
            "toc": True,
            "children": [
                {
                    "path": "structured_data_classification_with_feature_space",
                    "title": "Structured data classification with FeatureSpace",
                    "subcategory": "Structured data classification",
                    "highlight": True,
                },
                {
                    "path": "imbalanced_classification",
                    "title": "Imbalanced classification: credit card fraud detection",
                    "subcategory": "Structured data classification",
                    "highlight": True,
                },
                {
                    "path": "structured_data_classification_from_scratch",
                    "title": "Structured data classification from scratch",
                    "subcategory": "Structured data classification",
                },
                {
                    "path": "wide_deep_cross_networks",
                    "title": "Structured data learning with Wide, Deep, and Cross networks",
                    "subcategory": "Structured data classification",
                },
                {
                    "path": "classification_with_grn_and_vsn",
                    "title": "Classification with Gated Residual and Variable Selection Networks",
                    "subcategory": "Structured data classification",
                },
                {
                    "path": "classification_with_tfdf",
                    "title": "Classification with TensorFlow Decision Forests",
                    "subcategory": "Structured data classification",
                },
                {
                    "path": "deep_neural_decision_forests",
                    "title": "Classification with Neural Decision Forests",
                    "subcategory": "Structured data classification",
                },
                {
                    "path": "tabtransformer",
                    "title": "Structured data learning with TabTransformer",
                    "subcategory": "Structured data classification",
                },
                # Recommendation
                {
                    "path": "collaborative_filtering_movielens",
                    "title": "Collaborative Filtering for Movie Recommendations",
                    "subcategory": "Recommendation",
                },
                {
                    "path": "movielens_recommendations_transformers",
                    "title": "A Transformer-based recommendation system",
                    "subcategory": "Recommendation",
                },
            ],
        },
        {
            "path": "timeseries/",
            "title": "Timeseries",
            "toc": True,
            "children": [
                # Timeseries classification
                {
                    "path": "timeseries_classification_from_scratch",
                    "title": "Timeseries classification from scratch",
                    "subcategory": "Timeseries classification",
                    "highlight": True,
                },
                {
                    "path": "timeseries_classification_transformer",
                    "title": "Timeseries classification with a Transformer model",
                    "subcategory": "Timeseries classification",
                },
                {
                    "path": "eeg_signal_classification",
                    "title": "Electroencephalogram Signal Classification for action identification",
                    "subcategory": "Timeseries classification",
                },
                # Anomaly detection
                {
                    "path": "timeseries_anomaly_detection",
                    "title": "Timeseries anomaly detection using an Autoencoder",
                    "subcategory": "Anomaly detection",
                },
                # Timeseries forecasting
                {
                    "path": "timeseries_traffic_forecasting",
                    "title": "Traffic forecasting using graph neural networks and LSTM",
                    "subcategory": "Timeseries forecasting",
                },
                {
                    "path": "timeseries_weather_forecasting",
                    "title": "Timeseries forecasting for weather prediction",
                    "subcategory": "Timeseries forecasting",
                },
            ],
        },
        {
            "path": "generative/",
            "title": "Generative Deep Learning",
            "toc": True,
            "children": [
                # Image generation
                {
                    "path": "ddim",
                    "title": "Denoising Diffusion Implicit Models",
                    "subcategory": "Image generation",
                    "highlight": True,
                },
                {
                    "path": "random_walks_with_stable_diffusion",
                    "title": "A walk through latent space with Stable Diffusion",
                    "subcategory": "Image generation",
                    "highlight": True,
                },
                {
                    "path": "dreambooth",
                    "title": "DreamBooth",
                    "subcategory": "Image generation",
                },
                {
                    "path": "ddpm",
                    "title": "Denoising Diffusion Probabilistic Models",
                    "subcategory": "Image generation",
                },
                {
                    "path": "fine_tune_via_textual_inversion",
                    "title": "Teach StableDiffusion new concepts via Textual Inversion",
                    "subcategory": "Image generation",
                },
                {
                    "path": "finetune_stable_diffusion",
                    "title": "Fine-tuning Stable Diffusion",
                    "subcategory": "Image generation",
                },
                {
                    "path": "vae",
                    "title": "Variational AutoEncoder",
                    "subcategory": "Image generation",
                },
                {
                    "path": "dcgan_overriding_train_step",
                    "title": "GAN overriding Model.train_step",
                    "subcategory": "Image generation",
                },
                {
                    "path": "wgan_gp",
                    "title": "WGAN-GP overriding Model.train_step",
                    "subcategory": "Image generation",
                },
                {
                    "path": "conditional_gan",
                    "title": "Conditional GAN",
                    "subcategory": "Image generation",
                },
                {
                    "path": "cyclegan",
                    "title": "CycleGAN",
                    "subcategory": "Image generation",
                },
                {
                    "path": "gan_ada",
                    "title": "Data-efficient GANs with Adaptive Discriminator Augmentation",
                    "subcategory": "Image generation",
                },
                {
                    "path": "deep_dream",
                    "title": "Deep Dream",
                    "subcategory": "Image generation",
                },
                {
                    "path": "gaugan",
                    "title": "GauGAN for conditional image generation",
                    "subcategory": "Image generation",
                },
                {
                    "path": "pixelcnn",
                    "title": "PixelCNN",
                    "subcategory": "Image generation",
                },
                {
                    "path": "stylegan",
                    "title": "Face image generation with StyleGAN",
                    "subcategory": "Image generation",
                },
                {
                    "path": "vq_vae",
                    "title": "Vector-Quantized Variational Autoencoders",
                    "subcategory": "Image generation",
                },
                # Style transfer
                {
                    "path": "neural_style_transfer",
                    "title": "Neural style transfer",
                    "subcategory": "Style transfer",
                },
                {
                    "path": "adain",
                    "title": "Neural Style Transfer with AdaIN",
                    "subcategory": "Style transfer",
                },
                # Text generation
                {
                    "path": "gpt2_text_generation_with_kerasnlp",
                    "title": "GPT2 Text Generation with KerasNLP",
                    "subcategory": "Text generation",
                    "highlight": True,
                },
                {
                    "path": "text_generation_gpt",
                    "title": "GPT text generation from scratch with KerasNLP",
                    "subcategory": "Text generation",
                },
                {
                    "path": "text_generation_with_miniature_gpt",
                    "title": "Text generation with a miniature GPT",
                    "subcategory": "Text generation",
                },
                {
                    "path": "lstm_character_level_text_generation",
                    "title": "Character-level text generation with LSTM",
                    "subcategory": "Text generation",
                },
                {
                    "path": "text_generation_fnet",
                    "title": "Text Generation using FNet",
                    "subcategory": "Text generation",
                },
                # Graph generation
                {
                    "path": "molecule_generation",
                    "title": "Drug Molecule Generation with VAE",
                    "subcategory": "Graph generation",
                },
                {
                    "path": "wgan-graphs",
                    "title": "WGAN-GP with R-GCN for the generation of small molecular graphs",
                    "subcategory": "Graph generation",
                },
            ],
        },
        {
            "path": "audio/",
            "title": "Audio Data",
            "toc": True,
            "children": [
                # Will be autogenerated
            ],
        },
        {
            "path": "rl/",
            "title": "Reinforcement Learning",
            "toc": True,
            "children": [
                # Will be autogenerated
            ],
        },
        {
            "path": "graph/",
            "title": "Graph Data",
            "toc": True,
            "children": [
                # Will be autogenerated
            ],
        },
        {
            "path": "keras_recipes/",
            "title": "Quick Keras Recipes",
            "toc": True,
            "children": [
                # Will be autogenerated
            ],
        },
    ],
}
