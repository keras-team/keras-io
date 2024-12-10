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
                    "keras_3": True,
                },
                {
                    "path": "mnist_convnet",
                    "title": "Simple MNIST convnet",
                    "subcategory": "Image classification",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "image_classification_efficientnet_fine_tuning",
                    "title": "Image classification via fine-tuning with EfficientNet",
                    "subcategory": "Image classification",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "image_classification_with_vision_transformer",
                    "title": "Image classification with Vision Transformer",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "attention_mil_classification",
                    "title": "Classification using Attention-based Deep Multiple Instance Learning",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "mlp_image_classification",
                    "title": "Image classification with modern MLP models",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "mobilevit",
                    "title": "A mobile-friendly Transformer-based model for image classification",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "xray_classification_with_tpus",
                    "title": "Pneumonia Classification on TPU",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "cct",
                    "title": "Compact Convolutional Transformers",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "convmixer",
                    "title": "Image classification with ConvMixer",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "eanet",
                    "title": "Image classification with EANet (External Attention Transformer)",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "involution",
                    "title": "Involutional neural networks",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "perceiver_image_classification",
                    "title": "Image classification with Perceiver",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "reptile",
                    "title": "Few-Shot learning with Reptile",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "semisupervised_simclr",
                    "title": "Semi-supervised image classification using contrastive pretraining with SimCLR",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "swin_transformers",
                    "title": "Image classification with Swin Transformers",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "vit_small_ds",
                    "title": "Train a Vision Transformer on small datasets",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "shiftvit",
                    "title": "A Vision Transformer without Attention",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "image_classification_using_global_context_vision_transformer",
                    "title": "Image Classification using Global Context Vision Transformer",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                {
                    "path": "temporal_latent_bottleneck",
                    "title": "When Recurrence meets Transformers",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                # Image segmentation
                {
                    "path": "oxford_pets_image_segmentation",
                    "title": "Image segmentation with a U-Net-like architecture",
                    "subcategory": "Image segmentation",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "deeplabv3_plus",
                    "title": "Multiclass semantic segmentation using DeepLabV3+",
                    "subcategory": "Image segmentation",
                    "keras_3": True,
                },
                {
                    "path": "basnet_segmentation",
                    "title": "Highly accurate boundaries segmentation using BASNet",
                    "subcategory": "Image segmentation",
                },
                {
                    "path": "fully_convolutional_network",
                    "title": "Image Segmentation using Composable Fully-Convolutional Networks",
                    "subcategory": "Image segmentation",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "object_detection_using_vision_transformer",
                    "title": "Object detection with Vision Transformers",
                    "subcategory": "Object detection",
                    "keras_3": True,
                },
                # 3D
                {
                    "path": "3D_image_classification",
                    "title": "3D image classification from CT scans",
                    "subcategory": "3D",
                    "keras_3": True,
                },
                {
                    "path": "depth_estimation",
                    "title": "Monocular depth estimation",
                    "subcategory": "3D",
                    "keras_3": True,
                },
                {
                    "path": "nerf",
                    "title": "3D volumetric rendering with NeRF",
                    "subcategory": "3D",
                    "keras_3": True,
                    "highlight": True,
                },
                {
                    "path": "pointnet_segmentation",
                    "title": "Point cloud segmentation with PointNet",
                    "subcategory": "3D",
                    "keras_3": True,
                },
                {
                    "path": "pointnet",
                    "title": "Point cloud classification",
                    "subcategory": "3D",
                    "keras_3": True,
                },
                # OCR
                {
                    "path": "captcha_ocr",
                    "title": "OCR model for reading Captchas",
                    "subcategory": "OCR",
                    "keras_3": True,
                },
                {
                    "path": "handwriting_recognition",
                    "title": "Handwriting recognition",
                    "subcategory": "OCR",
                    "keras_3": True,
                },
                # Image enhancement
                {
                    "path": "autoencoder",
                    "title": "Convolutional autoencoder for image denoising",
                    "subcategory": "Image enhancement",
                    "keras_3": True,
                },
                {
                    "path": "mirnet",
                    "title": "Low-light image enhancement using MIRNet",
                    "subcategory": "Image enhancement",
                    "keras_3": True,
                },
                {
                    "path": "super_resolution_sub_pixel",
                    "title": "Image Super-Resolution using an Efficient Sub-Pixel CNN",
                    "subcategory": "Image enhancement",
                    "keras_3": True,
                },
                {
                    "path": "edsr",
                    "title": "Enhanced Deep Residual Networks for single-image super-resolution",
                    "subcategory": "Image enhancement",
                    "keras_3": True,
                },
                {
                    "path": "zero_dce",
                    "title": "Zero-DCE for low-light image enhancement",
                    "subcategory": "Image enhancement",
                    "keras_3": True,
                },
                # Data augmentation
                {
                    "path": "cutmix",
                    "title": "CutMix data augmentation for image classification",
                    "subcategory": "Data augmentation",
                    "keras_3": True,
                },
                {
                    "path": "mixup",
                    "title": "MixUp augmentation for image classification",
                    "subcategory": "Data augmentation",
                    "keras_3": True,
                },
                {
                    "path": "randaugment",
                    "title": "RandAugment for Image Classification for Improved Robustness",
                    "subcategory": "Data augmentation",
                    "keras_3": True,
                },
                # Image & Text
                {
                    "path": "image_captioning",
                    "title": "Image captioning",
                    "subcategory": "Image & Text",
                    "highlight": True,
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "integrated_gradients",
                    "title": "Model interpretability with Integrated Gradients",
                    "subcategory": "Vision models interpretability",
                    "keras_3": True,
                },
                {
                    "path": "probing_vits",
                    "title": "Investigating Vision Transformer representations",
                    "subcategory": "Vision models interpretability",
                    "keras_3": True,
                },
                {
                    "path": "grad_cam",
                    "title": "Grad-CAM class activation visualization",
                    "subcategory": "Vision models interpretability",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "siamese_contrastive",
                    "title": "Image similarity estimation using a Siamese Network with a contrastive loss",
                    "subcategory": "Image similarity search",
                    "keras_3": True,
                },
                {
                    "path": "siamese_network",
                    "title": "Image similarity estimation using a Siamese Network with a triplet loss",
                    "subcategory": "Image similarity search",
                    "keras_3": True,
                },
                {
                    "path": "metric_learning",
                    "title": "Metric learning for image similarity search",
                    "subcategory": "Image similarity search",
                    "keras_3": True,
                },
                {
                    "path": "metric_learning_tf_similarity",
                    "title": "Metric learning for image similarity search using TensorFlow Similarity",
                    "subcategory": "Image similarity search",
                },
                {
                    "path": "nnclr",
                    "title": "Self-supervised contrastive learning with NNCLR",
                    "subcategory": "Image similarity search",
                    "keras_3": True,
                },
                # Video
                {
                    "path": "video_classification",
                    "title": "Video Classification with a CNN-RNN Architecture",
                    "subcategory": "Video",
                    "keras_3": True,
                },
                {
                    "path": "conv_lstm",
                    "title": "Next-Frame Video Prediction with Convolutional LSTMs",
                    "subcategory": "Video",
                    "keras_3": True,
                },
                {
                    "path": "video_transformers",
                    "title": "Video Classification with Transformers",
                    "subcategory": "Video",
                    "keras_3": True,
                },
                {
                    "path": "vivit",
                    "title": "Video Vision Transformer",
                    "subcategory": "Video",
                    "keras_3": True,
                },
                {
                    "path": "bit",
                    "title": "Image Classification using BigTransfer (BiT)",
                    "subcategory": "Image classification",
                    "keras_3": True,
                },
                # Performance recipes
                {
                    "path": "gradient_centralization",
                    "title": "Gradient Centralization for Better Training Performance",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "token_learner",
                    "title": "Learning to tokenize in Vision Transformers",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "knowledge_distillation",
                    "title": "Knowledge Distillation",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "fixres",
                    "title": "FixRes: Fixing train-test resolution discrepancy",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "cait",
                    "title": "Class Attention Image Transformers with LayerScale",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "patch_convnet",
                    "title": "Augmenting convnets with aggregated attention",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
                },
                {
                    "path": "learnable_resizer",
                    "title": "Learning to Resize",
                    "subcategory": "Performance recipes",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "active_learning_review_classification",
                    "title": "Review Classification using Active Learning",
                    "subcategory": "Text classification",
                    "keras_3": True,
                },
                {
                    "path": "fnet_classification_with_keras_hub",
                    "title": "Text Classification using FNet",
                    "subcategory": "Text classification",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "text_classification_with_switch_transformer",
                    "title": "Text classification with Switch Transformer",
                    "subcategory": "Text classification",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "bidirectional_lstm_imdb",
                    "title": "Bidirectional LSTM on IMDB",
                    "subcategory": "Text classification",
                    "keras_3": True,
                },
                {
                    "path": "data_parallel_training_with_keras_hub",
                    "title": "Data Parallel Training with KerasHub and tf.distribute",
                    "subcategory": "Text classification",
                    "keras_3": True,
                },
                # Machine translation
                {
                    "path": "neural_machine_translation_with_keras_hub",
                    "title": "English-to-Spanish translation with KerasHub",
                    "subcategory": "Machine translation",
                    "keras_3": True,
                },
                {
                    "path": "neural_machine_translation_with_transformer",
                    "title": "English-to-Spanish translation with a sequence-to-sequence Transformer",
                    "subcategory": "Machine translation",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "lstm_seq2seq",
                    "title": "Character-level recurrent sequence-to-sequence model",
                    "subcategory": "Machine translation",
                    "keras_3": True,
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
                    "keras_3": True,
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
                    "keras_3": True,
                },
                # Text similarity search
                {
                    "path": "semantic_similarity_with_keras_hub",
                    "title": "Semantic Similarity with KerasHub",
                    "subcategory": "Text similarity search",
                    "keras_3": True,
                },
                {
                    "path": "semantic_similarity_with_bert",
                    "title": "Semantic Similarity with BERT",
                    "subcategory": "Text similarity search",
                    "keras_3": True,
                },
                {
                    "path": "sentence_embeddings_with_sbert",
                    "title": "Sentence embeddings using Siamese RoBERTa-networks",
                    "subcategory": "Text similarity search",
                    "keras_3": True,
                },
                # Language modeling
                {
                    "path": "masked_language_modeling",
                    "title": "End-to-end Masked Language Modeling with BERT",
                    "subcategory": "Language modeling",
                    "keras_3": True,
                },
                {
                    "path": "abstractive_summarization_with_bart",
                    "title": "Abstractive Text Summarization with BART",
                    "subcategory": "Language modeling",
                    "keras_3": True,
                },
                {
                    "path": "pretraining_BERT",
                    "title": "Pretraining BERT with Hugging Face Transformers",
                    "subcategory": "Language modeling",
                },
                # Parameter efficient fine-tuning.
                {
                    "path": "parameter_efficient_finetuning_of_gpt2_with_lora",
                    "title": "Parameter-efficient fine-tuning of GPT-2 with LoRA",
                    "subcategory": "Parameter efficient fine-tuning",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "feature_space_advanced",
                    "title": "FeatureSpace advanced use cases",
                    "subcategory": "Structured data classification",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "imbalanced_classification",
                    "title": "Imbalanced classification: credit card fraud detection",
                    "subcategory": "Structured data classification",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "structured_data_classification_from_scratch",
                    "title": "Structured data classification from scratch",
                    "subcategory": "Structured data classification",
                    "keras_3": True,
                },
                {
                    "path": "wide_deep_cross_networks",
                    "title": "Structured data learning with Wide, Deep, and Cross networks",
                    "subcategory": "Structured data classification",
                    "keras_3": True,
                },
                {
                    "path": "customer_lifetime_value",
                    "title": "Deep Learning for Customer Lifetime Value",
                    "subcategory": "Structured data regression",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "tabtransformer",
                    "title": "Structured data learning with TabTransformer",
                    "subcategory": "Structured data classification",
                    "keras_3": True,
                },
                # Recommendation
                {
                    "path": "collaborative_filtering_movielens",
                    "title": "Collaborative Filtering for Movie Recommendations",
                    "subcategory": "Recommendation",
                    "keras_3": True,
                },
                {
                    "path": "movielens_recommendations_transformers",
                    "title": "A Transformer-based recommendation system",
                    "subcategory": "Recommendation",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "timeseries_classification_transformer",
                    "title": "Timeseries classification with a Transformer model",
                    "subcategory": "Timeseries classification",
                    "keras_3": True,
                },
                {
                    "path": "eeg_signal_classification",
                    "title": "Electroencephalogram Signal Classification for action identification",
                    "subcategory": "Timeseries classification",
                    "keras_3": True,
                },
                {
                    "path": "event_classification_for_payment_card_fraud_detection",
                    "title": "Event classification for payment card fraud detection",
                    "subcategory": "Timeseries classification",
                    "keras_3": True,
                },
                # Anomaly detection
                {
                    "path": "timeseries_anomaly_detection",
                    "title": "Timeseries anomaly detection using an Autoencoder",
                    "subcategory": "Anomaly detection",
                    "keras_3": True,
                },
                # Timeseries forecasting
                {
                    "path": "timeseries_traffic_forecasting",
                    "title": "Traffic forecasting using graph neural networks and LSTM",
                    "subcategory": "Timeseries forecasting",
                    "keras_3": True,
                },
                {
                    "path": "timeseries_weather_forecasting",
                    "title": "Timeseries forecasting for weather prediction",
                    "subcategory": "Timeseries forecasting",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "random_walks_with_stable_diffusion",
                    "title": "A walk through latent space with Stable Diffusion",
                    "subcategory": "Image generation",
                    "highlight": True,
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "dcgan_overriding_train_step",
                    "title": "GAN overriding Model.train_step",
                    "subcategory": "Image generation",
                    "keras_3": True,
                },
                {
                    "path": "wgan_gp",
                    "title": "WGAN-GP overriding Model.train_step",
                    "subcategory": "Image generation",
                    "keras_3": True,
                },
                {
                    "path": "conditional_gan",
                    "title": "Conditional GAN",
                    "subcategory": "Image generation",
                    "keras_3": True,
                },
                {
                    "path": "cyclegan",
                    "title": "CycleGAN",
                    "subcategory": "Image generation",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "gaugan",
                    "title": "GauGAN for conditional image generation",
                    "subcategory": "Image generation",
                    "keras_3": True,
                },
                {
                    "path": "pixelcnn",
                    "title": "PixelCNN",
                    "subcategory": "Image generation",
                    "keras_3": True,
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
                    "keras_3": True,
                },
                {
                    "path": "adain",
                    "title": "Neural Style Transfer with AdaIN",
                    "subcategory": "Style transfer",
                },
                # Text generation
                {
                    "path": "gpt2_text_generation_with_keras_hub",
                    "title": "GPT2 Text Generation with KerasHub",
                    "subcategory": "Text generation",
                    "highlight": True,
                    "keras_3": True,
                },
                {
                    "path": "text_generation_gpt",
                    "title": "GPT text generation from scratch with KerasHub",
                    "subcategory": "Text generation",
                    "keras_3": True,
                },
                {
                    "path": "text_generation_with_miniature_gpt",
                    "title": "Text generation with a miniature GPT",
                    "subcategory": "Text generation",
                    "keras_3": True,
                },
                {
                    "path": "lstm_character_level_text_generation",
                    "title": "Character-level text generation with LSTM",
                    "subcategory": "Text generation",
                    "keras_3": True,
                },
                {
                    "path": "text_generation_fnet",
                    "title": "Text Generation using FNet",
                    "subcategory": "Text generation",
                },
                # Audio / midi generation
                {
                    "path": "midi_generation_with_transformer",
                    "title": "Music Generation with Transformer Models",
                    "subcategory": "Audio generation",
                    "keras_3": True,
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
                {
                    "path": "transformer_asr",
                    "title": "Automatic Speech Recognition with Transformer",
                    "subcategory": "Speech recognition",
                    "keras_3": True,
                },
                # Rest will be autogenerated
            ],
        },
        {
            "path": "rl/",
            "title": "Reinforcement Learning",
            "toc": True,
            "children": [
                {
                    "path": "actor_critic_cartpole",
                    "title": "Actor Critic Method",
                    "subcategory": "RL algorithms",
                    "keras_3": True,
                },
                {
                    "path": "ppo_cartpole",
                    "title": "Proximal Policy Optimization",
                    "subcategory": "RL algorithms",
                    "keras_3": True,
                },
                {
                    "path": "deep_q_network_breakout",
                    "title": "Deep Q-Learning for Atari Breakout",
                    "subcategory": "RL algorithms",
                    "keras_3": True,
                },
                {
                    "path": "ddpg_pendulum",
                    "title": "Deep Deterministic Policy Gradient (DDPG)",
                    "subcategory": "RL algorithms",
                    "keras_3": True,
                },
                # Rest will be autogenerated
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
                {
                    "path": "parameter_efficient_finetuning_of_gemma_with_lora_and_qlora",
                    "title": "Parameter-efficient fine-tuning of Gemma with LoRA and QLoRA",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "float8_training_and_inference_with_transformer",
                    "title": "Float8 training and inference with a simple Transformer model",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "tf_serving",
                    "title": "Serving TensorFlow models with TFServing",
                    "subcategory": "Serving",
                    "keras_3": True,
                },
                {
                    "path": "debugging_tips",
                    "title": "Keras debugging tips",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "subclassing_conv_layers",
                    "title": "Customizing the convolution operation of a Conv2D layer",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "trainer_pattern",
                    "title": "Trainer pattern",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "endpoint_layer_pattern",
                    "title": "Endpoint layer pattern",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "reproducibility_recipes",
                    "title": "Reproducibility in Keras Models",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "tensorflow_numpy_models",
                    "title": "Writing Keras Models With TensorFlow NumPy",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "antirectifier",
                    "title": "Simple custom layer example: Antirectifier",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                {
                    "path": "sample_size_estimate",
                    "title": "Estimating required sample size for model training",
                    "subcategory": "ML best practices",
                    "keras_3": True,
                },
                {
                    "path": "memory_efficient_embeddings",
                    "title": "Memory-efficient embeddings for recommendation systems",
                    "subcategory": "ML best practices",
                    "keras_3": True,
                },
                {
                    "path": "creating_tfrecords",
                    "title": "Creating TFRecords",
                    "subcategory": "ML best practices",
                    "keras_3": True,
                },
                {
                    "path": "packaging_keras_models_for_wide_distribution",
                    "title": "Packaging Keras models for wide distribution using Functional Subclassing",
                    "subcategory": "Keras usage tips",
                    "keras_3": True,
                },
                # Rest will be autogenerated
            ],
        },
    ],
}
