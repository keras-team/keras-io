BASE_CLASSES = {
    "path": "base_classes/",
    "title": "Modeling API",
    "toc": True,
    "children": [
        {
            "path": "backbone",
            "title": "Backbone",
            "generate": [
                "keras_hub.models.Backbone",
                "keras_hub.models.Backbone.from_preset",
                "keras_hub.models.Backbone.token_embedding",
                "keras_hub.models.Backbone.enable_lora",
                "keras_hub.models.Backbone.save_lora_weights",
                "keras_hub.models.Backbone.load_lora_weights",
                "keras_hub.models.Backbone.save_to_preset",
            ],
        },
        {
            "path": "causal_lm",
            "title": "CausalLM",
            "generate": [
                "keras_hub.models.CausalLM",
                "keras_hub.models.CausalLM.from_preset",
                "keras_hub.models.CausalLM.compile",
                "keras_hub.models.CausalLM.generate",
                "keras_hub.models.CausalLM.save_to_preset",
                "keras_hub.models.CausalLM.preprocessor",
                "keras_hub.models.CausalLM.backbone",
            ],
        },
        {
            "path": "causal_lm_preprocessor",
            "title": "CausalLMPreprocessor",
            "generate": [
                "keras_hub.models.CausalLMPreprocessor",
                "keras_hub.models.CausalLMPreprocessor.from_preset",
                "keras_hub.models.CausalLMPreprocessor.save_to_preset",
                "keras_hub.models.CausalLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "image_classifier",
            "title": "ImageClassifier",
            "generate": [
                "keras_hub.models.ImageClassifier",
                "keras_hub.models.ImageClassifier.from_preset",
                "keras_hub.models.ImageClassifier.compile",
                "keras_hub.models.ImageClassifier.save_to_preset",
                "keras_hub.models.ImageClassifier.preprocessor",
                "keras_hub.models.ImageClassifier.backbone",
            ],
        },
        {
            "path": "image_classifier_preprocessor",
            "title": "ImageClassifierPreprocessor",
            "generate": [
                "keras_hub.models.ImageClassifierPreprocessor",
                "keras_hub.models.ImageClassifierPreprocessor.from_preset",
                "keras_hub.models.ImageClassifier.save_to_preset",
            ],
        },
        {
            "path": "image_to_image",
            "title": "ImageToImage",
            "generate": [
                "keras_hub.models.ImageToImage",
                "keras_hub.models.ImageToImage.from_preset",
                "keras_hub.models.ImageToImage.compile",
                "keras_hub.models.ImageToImage.save_to_preset",
                "keras_hub.models.ImageToImage.preprocessor",
                "keras_hub.models.ImageToImage.backbone",
                "keras_hub.models.ImageToImage.generate",
            ],
        },
        {
            "path": "image_segmenter",
            "title": "ImageSegmenter",
            "generate": [
                "keras_hub.models.ImageSegmenter",
                "keras_hub.models.ImageSegmenter.from_preset",
                "keras_hub.models.ImageSegmenter.compile",
                "keras_hub.models.ImageSegmenter.save_to_preset",
                "keras_hub.models.ImageSegmenter.preprocessor",
                "keras_hub.models.ImageSegmenter.backbone",
            ],
        },
        {
            "path": "inpaint",
            "title": "Inpaint",
            "generate": [
                "keras_hub.models.Inpaint",
                "keras_hub.models.Inpaint.from_preset",
                "keras_hub.models.Inpaint.compile",
                "keras_hub.models.Inpaint.save_to_preset",
                "keras_hub.models.Inpaint.preprocessor",
                "keras_hub.models.Inpaint.backbone",
                "keras_hub.models.Inpaint.generate",
            ],
        },
        {
            "path": "masked_lm",
            "title": "MaskedLM",
            "generate": [
                "keras_hub.models.MaskedLM",
                "keras_hub.models.MaskedLM.from_preset",
                "keras_hub.models.MaskedLM.compile",
                "keras_hub.models.MaskedLM.save_to_preset",
                "keras_hub.models.MaskedLM.preprocessor",
                "keras_hub.models.MaskedLM.backbone",
            ],
        },
        {
            "path": "masked_lm_preprocessor",
            "title": "MaskedLMPreprocessor",
            "generate": [
                "keras_hub.models.MaskedLMPreprocessor",
                "keras_hub.models.MaskedLMPreprocessor.from_preset",
                "keras_hub.models.MaskedLMPreprocessor.save_to_preset",
                "keras_hub.models.MaskedLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "preprocessor",
            "title": "Preprocessor",
            "generate": [
                "keras_hub.models.Preprocessor",
                "keras_hub.models.Preprocessor.from_preset",
                "keras_hub.models.Preprocessor.save_to_preset",
                "keras_hub.models.Preprocessor.tokenizer",
            ],
        },
        {
            "path": "seq_2_seq_lm",
            "title": "Seq2SeqLM",
            "generate": [
                "keras_hub.models.Seq2SeqLM",
                "keras_hub.models.Seq2SeqLM.from_preset",
                "keras_hub.models.Seq2SeqLM.compile",
                "keras_hub.models.Seq2SeqLM.generate",
                "keras_hub.models.Seq2SeqLM.save_to_preset",
                "keras_hub.models.Seq2SeqLM.preprocessor",
                "keras_hub.models.Seq2SeqLM.backbone",
            ],
        },
        {
            "path": "seq_2_seq_lm_preprocessor",
            "title": "Seq2SeqLMPreprocessor",
            "generate": [
                "keras_hub.models.Seq2SeqLMPreprocessor",
                "keras_hub.models.Seq2SeqLMPreprocessor.from_preset",
                "keras_hub.models.Seq2SeqLMPreprocessor.save_to_preset",
                "keras_hub.models.Seq2SeqLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "task",
            "title": "Task",
            "generate": [
                "keras_hub.models.Task",
                "keras_hub.models.Task.from_preset",
                "keras_hub.models.Task.save_to_preset",
                "keras_hub.models.Task.preprocessor",
                "keras_hub.models.Task.backbone",
            ],
        },
        {
            "path": "text_classifier",
            "title": "TextClassifier",
            "generate": [
                "keras_hub.models.TextClassifier",
                "keras_hub.models.TextClassifier.from_preset",
                "keras_hub.models.TextClassifier.compile",
                "keras_hub.models.TextClassifier.save_to_preset",
                "keras_hub.models.TextClassifier.preprocessor",
                "keras_hub.models.TextClassifier.backbone",
            ],
        },
        {
            "path": "text_classifier_preprocessor",
            "title": "TextClassifierPreprocessor",
            "generate": [
                "keras_hub.models.TextClassifierPreprocessor",
                "keras_hub.models.TextClassifierPreprocessor.from_preset",
                "keras_hub.models.TextClassifierPreprocessor.save_to_preset",
                "keras_hub.models.TextClassifierPreprocessor.tokenizer",
            ],
        },
        {
            "path": "text_to_image",
            "title": "TextToImage",
            "generate": [
                "keras_hub.models.TextToImage",
                "keras_hub.models.TextToImage.from_preset",
                "keras_hub.models.TextToImage.compile",
                "keras_hub.models.TextToImage.save_to_preset",
                "keras_hub.models.TextToImage.preprocessor",
                "keras_hub.models.TextToImage.backbone",
                "keras_hub.models.TextToImage.generate",
            ],
        },
        {
            "path": "upload_preset",
            "title": "upload_preset",
            "generate": ["keras_hub.upload_preset"],
        },
    ],
}

MODELS_MASTER = {
    "path": "models/",
    "title": "Model Architectures",
    "toc": True,
    "children": [
        {
            "path": "albert/",
            "title": "Albert",
            "toc": True,
            "children": [
                {
                    "path": "albert_tokenizer",
                    "title": "AlbertTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.AlbertTokenizer",
                        "keras_hub.tokenizers.AlbertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "albert_backbone",
                    "title": "AlbertBackbone model",
                    "generate": [
                        "keras_hub.models.AlbertBackbone",
                        "keras_hub.models.AlbertBackbone.from_preset",
                        "keras_hub.models.AlbertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "albert_text_classifier",
                    "title": "AlbertTextClassifier model",
                    "generate": [
                        "keras_hub.models.AlbertTextClassifier",
                        "keras_hub.models.AlbertTextClassifier.from_preset",
                        "keras_hub.models.AlbertTextClassifier.backbone",
                        "keras_hub.models.AlbertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "albert_text_classifier_preprocessor",
                    "title": "AlbertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.AlbertTextClassifierPreprocessor",
                        "keras_hub.models.AlbertTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.AlbertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "albert_masked_lm",
                    "title": "AlbertMaskedLM model",
                    "generate": [
                        "keras_hub.models.AlbertMaskedLM",
                        "keras_hub.models.AlbertMaskedLM.from_preset",
                        "keras_hub.models.AlbertMaskedLM.backbone",
                        "keras_hub.models.AlbertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "albert_masked_lm_preprocessor",
                    "title": "AlbertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.AlbertMaskedLMPreprocessor",
                        "keras_hub.models.AlbertMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.AlbertMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "bart/",
            "title": "Bart",
            "toc": True,
            "children": [
                {
                    "path": "bart_tokenizer",
                    "title": "BartTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.BartTokenizer",
                        "keras_hub.tokenizers.BartTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bart_backbone",
                    "title": "BartBackbone model",
                    "generate": [
                        "keras_hub.models.BartBackbone",
                        "keras_hub.models.BartBackbone.from_preset",
                        "keras_hub.models.BartBackbone.token_embedding",
                    ],
                },
                {
                    "path": "bart_seq_2_seq_lm",
                    "title": "BartSeq2SeqLM model",
                    "generate": [
                        "keras_hub.models.BartSeq2SeqLM",
                        "keras_hub.models.BartSeq2SeqLM.from_preset",
                        "keras_hub.models.BartSeq2SeqLM.generate",
                        "keras_hub.models.BartSeq2SeqLM.backbone",
                        "keras_hub.models.BartSeq2SeqLM.preprocessor",
                    ],
                },
                {
                    "path": "bart_seq_2_seq_lm_preprocessor",
                    "title": "BartSeq2SeqLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.BartSeq2SeqLMPreprocessor",
                        "keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset",
                        "keras_hub.models.BartSeq2SeqLMPreprocessor.generate_preprocess",
                        "keras_hub.models.BartSeq2SeqLMPreprocessor.generate_postprocess",
                        "keras_hub.models.BartSeq2SeqLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "basnet/",
            "title": "BASNet",
            "toc": True,
            "children": [
                {
                    "path": "basnet_image_converter",
                    "title": "BASNetImageConverter",
                    "generate": [
                        "keras_hub.layers.BASNetImageConverter",
                        "keras_hub.layers.BASNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "basnet_backbone",
                    "title": "BASNetBackbone model",
                    "generate": [
                        "keras_hub.models.BASNetBackbone",
                        "keras_hub.models.BASNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "basnet_image_segmenter",
                    "title": "BASNetImageSegmenter model",
                    "generate": [
                        "keras_hub.models.BASNetImageSegmenter",
                        "keras_hub.models.BASNetImageSegmenter.from_preset",
                        "keras_hub.models.BASNetImageSegmenter.backbone",
                        "keras_hub.models.BASNetImageSegmenter.preprocessor",
                    ],
                },
                {
                    "path": "basnet_preprocessor",
                    "title": "BASNetPreprocessor layer",
                    "generate": [
                        "keras_hub.models.BASNetPreprocessor",
                        "keras_hub.models.BASNetPreprocessor.from_preset",
                        "keras_hub.models.BASNetPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "bert/",
            "title": "Bert",
            "toc": True,
            "children": [
                {
                    "path": "bert_tokenizer",
                    "title": "BertTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.BertTokenizer",
                        "keras_hub.tokenizers.BertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bert_backbone",
                    "title": "BertBackbone model",
                    "generate": [
                        "keras_hub.models.BertBackbone",
                        "keras_hub.models.BertBackbone.from_preset",
                        "keras_hub.models.BertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "bert_text_classifier",
                    "title": "BertTextClassifier model",
                    "generate": [
                        "keras_hub.models.BertTextClassifier",
                        "keras_hub.models.BertTextClassifier.from_preset",
                        "keras_hub.models.BertTextClassifier.backbone",
                        "keras_hub.models.BertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "bert_text_classifier_preprocessor",
                    "title": "BertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.BertTextClassifierPreprocessor",
                        "keras_hub.models.BertTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.BertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "bert_masked_lm",
                    "title": "BertMaskedLM model",
                    "generate": [
                        "keras_hub.models.BertMaskedLM",
                        "keras_hub.models.BertMaskedLM.from_preset",
                        "keras_hub.models.BertMaskedLM.backbone",
                        "keras_hub.models.BertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "bert_masked_lm_preprocessor",
                    "title": "BertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.BertMaskedLMPreprocessor",
                        "keras_hub.models.BertMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.BertMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "bloom/",
            "title": "Bloom",
            "toc": True,
            "children": [
                {
                    "path": "bloom_tokenizer",
                    "title": "BloomTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.BloomTokenizer",
                        "keras_hub.tokenizers.BloomTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bloom_backbone",
                    "title": "BloomBackbone model",
                    "generate": [
                        "keras_hub.models.BloomBackbone",
                        "keras_hub.models.BloomBackbone.from_preset",
                        "keras_hub.models.BloomBackbone.token_embedding",
                        "keras_hub.models.BloomBackbone.enable_lora",
                    ],
                },
                {
                    "path": "bloom_causal_lm",
                    "title": "BloomCausalLM model",
                    "generate": [
                        "keras_hub.models.BloomCausalLM",
                        "keras_hub.models.BloomCausalLM.from_preset",
                        "keras_hub.models.BloomCausalLM.generate",
                        "keras_hub.models.BloomCausalLM.backbone",
                        "keras_hub.models.BloomCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "bloom_causal_lm_preprocessor",
                    "title": "BloomCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.BloomCausalLMPreprocessor",
                        "keras_hub.models.BloomCausalLMPreprocessor.from_preset",
                        "keras_hub.models.BloomCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "clip/",
            "title": "CLIP",
            "toc": True,
            "children": [
                {
                    "path": "clip_tokenizer",
                    "title": "CLIPTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.CLIPTokenizer",
                        "keras_hub.tokenizers.CLIPTokenizer.from_preset",
                    ],
                },
                {
                    "path": "clip_image_converter",
                    "title": "CLIPImageConverter",
                    "generate": [
                        "keras_hub.layers.CLIPImageConverter",
                        "keras_hub.layers.CLIPImageConverter.from_preset",
                    ],
                },
                {
                    "path": "clip_backbone",
                    "title": "CLIPBackbone model",
                    "generate": [
                        "keras_hub.models.CLIPBackbone",
                        "keras_hub.models.CLIPBackbone.from_preset",
                    ],
                },
                {
                    "path": "clip_preprocessor",
                    "title": "CLIPPreprocessor",
                    "generate": [
                        "keras_hub.models.CLIPPreprocessor",
                        "keras_hub.models.CLIPPreprocessor.from_preset",
                        "keras_hub.models.CLIPPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "cspnet/",
            "title": "CSPNet",
            "toc": True,
            "children": [
                {
                    "path": "cspnet_image_converter",
                    "title": "CSPNetImageConverter",
                    "generate": [
                        "keras_hub.layers.CSPNetImageConverter",
                        "keras_hub.layers.CSPNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "cspnet_backbone",
                    "title": "CSPNetBackbone model",
                    "generate": [
                        "keras_hub.models.CSPNetBackbone",
                        "keras_hub.models.CSPNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "cspnet_image_classifier",
                    "title": "CSPNetImageClassifier model",
                    "generate": [
                        "keras_hub.models.CSPNetImageClassifier",
                        "keras_hub.models.CSPNetImageClassifier.from_preset",
                        "keras_hub.models.CSPNetImageClassifier.backbone",
                        "keras_hub.models.CSPNetImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "cspnet_image_classifier_preprocessor",
                    "title": "CSPNetImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.CSPNetImageClassifierPreprocessor",
                        "keras_hub.models.CSPNetImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.CSPNetImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "d_fine/",
            "title": "D-FINE",
            "toc": True,
            "children": [
                {
                    "path": "d_fine_image_converter",
                    "title": "DFineImageConverter",
                    "generate": [
                        "keras_hub.layers.DFineImageConverter",
                        "keras_hub.layers.DFineImageConverter.from_preset",
                    ],
                },
                {
                    "path": "d_fine_backbone",
                    "title": "DFineBackbone model",
                    "generate": [
                        "keras_hub.models.DFineBackbone",
                        "keras_hub.models.DFineBackbone.from_preset",
                    ],
                },
                {
                    "path": "d_fine_object_detector",
                    "title": "DFineObjectDetector model",
                    "generate": [
                        "keras_hub.models.DFineObjectDetector",
                        "keras_hub.models.DFineObjectDetector.from_preset",
                        "keras_hub.models.DFineObjectDetector.backbone",
                        "keras_hub.models.DFineObjectDetector.preprocessor",
                    ],
                },
                {
                    "path": "d_fine_object_detector_preprocessor",
                    "title": "DFineObjectDetectorPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DFineObjectDetectorPreprocessor",
                        "keras_hub.models.DFineObjectDetectorPreprocessor.from_preset",
                        "keras_hub.models.DFineObjectDetectorPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "deberta_v3/",
            "title": "DebertaV3",
            "toc": True,
            "children": [
                {
                    "path": "deberta_v3_tokenizer",
                    "title": "DebertaV3Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.DebertaV3Tokenizer",
                        "keras_hub.tokenizers.DebertaV3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "deberta_v3_backbone",
                    "title": "DebertaV3Backbone model",
                    "generate": [
                        "keras_hub.models.DebertaV3Backbone",
                        "keras_hub.models.DebertaV3Backbone.from_preset",
                        "keras_hub.models.DebertaV3Backbone.token_embedding",
                    ],
                },
                {
                    "path": "deberta_v3_text_classifier",
                    "title": "DebertaV3TextClassifier model",
                    "generate": [
                        "keras_hub.models.DebertaV3TextClassifier",
                        "keras_hub.models.DebertaV3TextClassifier.from_preset",
                        "keras_hub.models.DebertaV3TextClassifier.backbone",
                        "keras_hub.models.DebertaV3TextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "deberta_v3_text_classifier_preprocessor",
                    "title": "DebertaV3TextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DebertaV3TextClassifierPreprocessor",
                        "keras_hub.models.DebertaV3TextClassifierPreprocessor.from_preset",
                        "keras_hub.models.DebertaV3TextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "deberta_v3_masked_lm",
                    "title": "DebertaV3MaskedLM model",
                    "generate": [
                        "keras_hub.models.DebertaV3MaskedLM",
                        "keras_hub.models.DebertaV3MaskedLM.from_preset",
                        "keras_hub.models.DebertaV3MaskedLM.backbone",
                        "keras_hub.models.DebertaV3MaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "deberta_v3_masked_lm_preprocessor",
                    "title": "DebertaV3MaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DebertaV3MaskedLMPreprocessor",
                        "keras_hub.models.DebertaV3MaskedLMPreprocessor.from_preset",
                        "keras_hub.models.DebertaV3MaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "deeplab_v3/",
            "title": "DeepLabV3",
            "toc": True,
            "children": [
                {
                    "path": "deeplab_v3_image_converter",
                    "title": "DeepLabV3ImageConverter",
                    "generate": [
                        "keras_hub.layers.DeepLabV3ImageConverter",
                        "keras_hub.layers.DeepLabV3ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "deeplab_v3_backbone",
                    "title": "DeepLabV3Backbone model",
                    "generate": [
                        "keras_hub.models.DeepLabV3Backbone",
                        "keras_hub.models.DeepLabV3Backbone.from_preset",
                    ],
                },
                {
                    "path": "deeplab_v3_image_segmenter",
                    "title": "DeepLabV3ImageSegmenter model",
                    "generate": [
                        "keras_hub.models.DeepLabV3ImageSegmenter",
                        "keras_hub.models.DeepLabV3ImageSegmenter.from_preset",
                        "keras_hub.models.DeepLabV3ImageSegmenter.backbone",
                        "keras_hub.models.DeepLabV3ImageSegmenter.preprocessor",
                    ],
                },
                {
                    "path": "deeplab_v3_image_segmenter_preprocessor",
                    "title": "DeepLabV3ImageSegmenterPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DeepLabV3ImageSegmenterPreprocessor",
                        "keras_hub.models.DeepLabV3ImageSegmenterPreprocessor.from_preset",
                        "keras_hub.models.DeepLabV3ImageSegmenterPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "deit/",
            "title": "DeiT",
            "toc": True,
            "children": [
                {
                    "path": "deit_image_converter",
                    "title": "DeiTImageConverter",
                    "generate": [
                        "keras_hub.layers.DeiTImageConverter",
                        "keras_hub.layers.DeiTImageConverter.from_preset",
                    ],
                },
                {
                    "path": "deit_backbone",
                    "title": "DeiTBackbone model",
                    "generate": [
                        "keras_hub.models.DeiTBackbone",
                        "keras_hub.models.DeiTBackbone.from_preset",
                    ],
                },
                {
                    "path": "deit_image_classifier",
                    "title": "DeiTImageClassifier model",
                    "generate": [
                        "keras_hub.models.DeiTImageClassifier",
                        "keras_hub.models.DeiTImageClassifier.from_preset",
                        "keras_hub.models.DeiTImageClassifier.backbone",
                        "keras_hub.models.DeiTImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "deit_image_classifier_preprocessor",
                    "title": "DeiTImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DeiTImageClassifierPreprocessor",
                        "keras_hub.models.DeiTImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.DeiTImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "densenet/",
            "title": "DenseNet",
            "toc": True,
            "children": [
                {
                    "path": "densenet_image_converter",
                    "title": "DenseNetImageConverter",
                    "generate": [
                        "keras_hub.layers.DenseNetImageConverter",
                        "keras_hub.layers.DenseNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "densenet_backbone",
                    "title": "DensNetBackbone model",
                    "generate": [
                        "keras_hub.models.DenseNetBackbone",
                        "keras_hub.models.DenseNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "densenet_image_classifier",
                    "title": "DenseNetImageClassifier model",
                    "generate": [
                        "keras_hub.models.DenseNetImageClassifier",
                        "keras_hub.models.DenseNetImageClassifier.from_preset",
                        "keras_hub.models.DenseNetImageClassifier.backbone",
                        "keras_hub.models.DenseNetImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "densenet_image_classifier_preprocessor",
                    "title": "DenseNetImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DenseNetImageClassifierPreprocessor",
                        "keras_hub.models.DenseNetImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.DenseNetImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "depth_anything/",
            "title": "DepthAnything",
            "toc": True,
            "children": [
                {
                    "path": "depth_anything_image_converter",
                    "title": "DepthAnythingImageConverter",
                    "generate": [
                        "keras_hub.layers.DepthAnythingImageConverter",
                        "keras_hub.layers.DepthAnythingImageConverter.from_preset",
                    ],
                },
                {
                    "path": "depth_anything_backbone",
                    "title": "DepthAnythingBackbone model",
                    "generate": [
                        "keras_hub.models.DepthAnythingBackbone",
                        "keras_hub.models.DepthAnythingBackbone.from_preset",
                    ],
                },
                {
                    "path": "depth_anything_depth_estimator",
                    "title": "DepthAnythingDepthEstimator model",
                    "generate": [
                        "keras_hub.models.DepthAnythingDepthEstimator",
                        "keras_hub.models.DepthAnythingDepthEstimator.from_preset",
                        "keras_hub.models.DepthAnythingDepthEstimator.backbone",
                    ],
                },
            ],
        },
        {
            "path": "dinov2/",
            "title": "DINOV2",
            "toc": True,
            "children": [
                {
                    "path": "dinov2_image_converter",
                    "title": "DINOV2ImageConverter",
                    "generate": [
                        "keras_hub.layers.DINOV2ImageConverter",
                        "keras_hub.layers.DINOV2ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "dinov2_backbone",
                    "title": "DINOV2Backbone model",
                    "generate": [
                        "keras_hub.models.DINOV2Backbone",
                        "keras_hub.models.DINOV2Backbone.from_preset",
                    ],
                },
            ],
        },
        {
            "path": "dinov3/",
            "title": "DINOV3",
            "toc": True,
            "children": [
                {
                    "path": "dinov3_image_converter",
                    "title": "DINOV3ImageConverter",
                    "generate": [
                        "keras_hub.layers.DINOV3ImageConverter",
                        "keras_hub.layers.DINOV3ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "dinov3_backbone",
                    "title": "DINOV3Backbone model",
                    "generate": [
                        "keras_hub.models.DINOV3Backbone",
                        "keras_hub.models.DINOV3Backbone.from_preset",
                    ],
                },
            ],
        },
        {
            "path": "distil_bert/",
            "title": "DistilBert",
            "toc": True,
            "children": [
                {
                    "path": "distil_bert_tokenizer",
                    "title": "DistilBertTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.DistilBertTokenizer",
                        "keras_hub.tokenizers.DistilBertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "distil_bert_backbone",
                    "title": "DistilBertBackbone model",
                    "generate": [
                        "keras_hub.models.DistilBertBackbone",
                        "keras_hub.models.DistilBertBackbone.from_preset",
                        "keras_hub.models.DistilBertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "distil_bert_text_classifier",
                    "title": "DistilBertTextClassifier model",
                    "generate": [
                        "keras_hub.models.DistilBertTextClassifier",
                        "keras_hub.models.DistilBertTextClassifier.from_preset",
                        "keras_hub.models.DistilBertTextClassifier.backbone",
                        "keras_hub.models.DistilBertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "distil_bert_text_classifier_preprocessor",
                    "title": "DistilBertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DistilBertTextClassifierPreprocessor",
                        "keras_hub.models.DistilBertTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.DistilBertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "distil_bert_masked_lm",
                    "title": "DistilBertMaskedLM model",
                    "generate": [
                        "keras_hub.models.DistilBertMaskedLM",
                        "keras_hub.models.DistilBertMaskedLM.from_preset",
                        "keras_hub.models.DistilBertMaskedLM.backbone",
                        "keras_hub.models.DistilBertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "distil_bert_masked_lm_preprocessor",
                    "title": "DistilBertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.DistilBertMaskedLMPreprocessor",
                        "keras_hub.models.DistilBertMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.DistilBertMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "efficientnet/",
            "title": "EfficientNet",
            "toc": True,
            "children": [
                {
                    "path": "efficientnet_image_converter",
                    "title": "EfficientNetImageConverter",
                    "generate": [
                        "keras_hub.layers.EfficientNetImageConverter",
                        "keras_hub.layers.EfficientNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "efficientnet_backbone",
                    "title": "EfficientNetBackbone model",
                    "generate": [
                        "keras_hub.models.EfficientNetBackbone",
                        "keras_hub.models.EfficientNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "efficientnet_image_classifier",
                    "title": "EfficientNetImageClassifier model",
                    "generate": [
                        "keras_hub.models.EfficientNetImageClassifier",
                        "keras_hub.models.EfficientNetImageClassifier.from_preset",
                        "keras_hub.models.EfficientNetImageClassifier.backbone",
                        "keras_hub.models.EfficientNetImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "efficientnet_image_classifier_preprocessor",
                    "title": "EfficientNetImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.EfficientNetImageClassifierPreprocessor",
                        "keras_hub.models.EfficientNetImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.EfficientNetImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "electra/",
            "title": "Electra",
            "toc": True,
            "children": [
                {
                    "path": "electra_tokenizer",
                    "title": "ElectraTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.ElectraTokenizer",
                        "keras_hub.tokenizers.ElectraTokenizer.from_preset",
                    ],
                },
                {
                    "path": "electra_backbone",
                    "title": "ElectraBackbone model",
                    "generate": [
                        "keras_hub.models.ElectraBackbone",
                        "keras_hub.models.ElectraBackbone.from_preset",
                        "keras_hub.models.ElectraBackbone.token_embedding",
                    ],
                },
            ],
        },
        {
            "path": "esm/",
            "title": "ESM",
            "toc": True,
            "children": [
                {
                    "path": "esm_tokenizer",
                    "title": "ESMTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.ESMTokenizer",
                        "keras_hub.tokenizers.ESMTokenizer.from_preset",
                    ],
                },
                {
                    "path": "esm_backbone",
                    "title": "ESMBackbone model",
                    "generate": [
                        "keras_hub.models.ESMBackbone",
                        "keras_hub.models.ESMBackbone.from_preset",
                        "keras_hub.models.ESMBackbone.token_embedding",
                    ],
                },
                {
                    "path": "esm_classifier",
                    "title": "ESMProteinClassifier model",
                    "generate": [
                        "keras_hub.models.ESMProteinClassifier",
                        "keras_hub.models.ESMProteinClassifier.from_preset",
                        "keras_hub.models.ESMProteinClassifier.backbone",
                        "keras_hub.models.ESMProteinClassifier.preprocessor",
                    ],
                },
                {
                    "path": "esm_classifier_preprocessor",
                    "title": "ESMProteinClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.ESMProteinClassifierPreprocessor",
                        "keras_hub.models.ESMProteinClassifierPreprocessor.from_preset",
                        "keras_hub.models.ESMProteinClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "esm_masked_plm",
                    "title": "ESMMaskedPLM model",
                    "generate": [
                        "keras_hub.models.ESMMaskedPLM",
                        "keras_hub.models.ESMMaskedPLM.from_preset",
                        "keras_hub.models.ESMMaskedPLM.backbone",
                        "keras_hub.models.ESMMaskedPLM.preprocessor",
                    ],
                },
                {
                    "path": "esm_masked_plm_preprocessor",
                    "title": "ESMMaskedPLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.ESMMaskedPLMPreprocessor",
                        "keras_hub.models.ESMMaskedPLMPreprocessor.from_preset",
                        "keras_hub.models.ESMMaskedPLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "falcon/",
            "title": "Falcon",
            "toc": True,
            "children": [
                {
                    "path": "falcon_tokenizer",
                    "title": "FalconTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.FalconTokenizer",
                        "keras_hub.tokenizers.FalconTokenizer.from_preset",
                    ],
                },
                {
                    "path": "falcon_backbone",
                    "title": "FalconBackbone model",
                    "generate": [
                        "keras_hub.models.FalconBackbone",
                        "keras_hub.models.FalconBackbone.from_preset",
                        "keras_hub.models.FalconBackbone.token_embedding",
                    ],
                },
                {
                    "path": "falcon_causal_lm",
                    "title": "FalconCausalLM model",
                    "generate": [
                        "keras_hub.models.FalconCausalLM",
                        "keras_hub.models.FalconCausalLM.from_preset",
                        "keras_hub.models.FalconCausalLM.generate",
                        "keras_hub.models.FalconCausalLM.backbone",
                        "keras_hub.models.FalconCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "falcon_causal_lm_preprocessor",
                    "title": "FalconCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.FalconCausalLMPreprocessor",
                        "keras_hub.models.FalconCausalLMPreprocessor.from_preset",
                        "keras_hub.models.FalconCausalLMPreprocessor.generate_preprocess",
                        "keras_hub.models.FalconCausalLMPreprocessor.generate_postprocess",
                        "keras_hub.models.FalconCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "flux/",
            "title": "Flux",
            "toc": True,
            "children": [
                {
                    "path": "flux_model",
                    "title": "FluxBackbone model",
                    "generate": [
                        "keras_hub.models.FluxBackbone",
                        "keras_hub.models.FluxBackbone.from_preset",
                    ],
                },
                {
                    "path": "flux_text_to_image",
                    "title": "FluxTextToImage model",
                    "generate": [
                        "keras_hub.models.FluxTextToImage",
                        "keras_hub.models.FluxTextToImage.from_preset",
                        "keras_hub.models.FluxTextToImage.backbone",
                        "keras_hub.models.FluxTextToImage.generate",
                        "keras_hub.models.FluxTextToImage.preprocessor",
                    ],
                },
                {
                    "path": "flux_text_to_image_preprocessor",
                    "title": "FluxTextToImagePreprocessor layer",
                    "generate": [
                        "keras_hub.models.FluxTextToImagePreprocessor",
                        "keras_hub.models.FluxTextToImagePreprocessor.from_preset",
                    ],
                },
            ],
        },
        {
            "path": "f_net/",
            "title": "FNet",
            "toc": True,
            "children": [
                {
                    "path": "f_net_tokenizer",
                    "title": "FNetTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.FNetTokenizer",
                        "keras_hub.tokenizers.FNetTokenizer.from_preset",
                    ],
                },
                {
                    "path": "f_net_backbone",
                    "title": "FNetBackbone model",
                    "generate": [
                        "keras_hub.models.FNetBackbone",
                        "keras_hub.models.FNetBackbone.from_preset",
                        "keras_hub.models.FNetBackbone.token_embedding",
                    ],
                },
                {
                    "path": "f_net_text_classifier",
                    "title": "FNetTextClassifier model",
                    "generate": [
                        "keras_hub.models.FNetTextClassifier",
                        "keras_hub.models.FNetTextClassifier.from_preset",
                        "keras_hub.models.FNetTextClassifier.backbone",
                        "keras_hub.models.FNetTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "f_net_text_classifier_preprocessor",
                    "title": "FNetTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.FNetTextClassifierPreprocessor",
                        "keras_hub.models.FNetTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.FNetTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "f_net_masked_lm",
                    "title": "FNetMaskedLM model",
                    "generate": [
                        "keras_hub.models.FNetMaskedLM",
                        "keras_hub.models.FNetMaskedLM.from_preset",
                        "keras_hub.models.FNetMaskedLM.backbone",
                        "keras_hub.models.FNetMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "f_net_masked_lm_preprocessor",
                    "title": "FNetMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.FNetMaskedLMPreprocessor",
                        "keras_hub.models.FNetMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.FNetMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "gemma/",
            "title": "Gemma",
            "toc": True,
            "children": [
                {
                    "path": "gemma_tokenizer",
                    "title": "GemmaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.GemmaTokenizer",
                        "keras_hub.tokenizers.GemmaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "gemma_backbone",
                    "title": "GemmaBackbone model",
                    "generate": [
                        "keras_hub.models.GemmaBackbone",
                        "keras_hub.models.GemmaBackbone.from_preset",
                        "keras_hub.models.GemmaBackbone.token_embedding",
                        "keras_hub.models.GemmaBackbone.enable_lora",
                        "keras_hub.models.GemmaBackbone.get_layout_map",
                    ],
                },
                {
                    "path": "gemma_causal_lm",
                    "title": "GemmaCausalLM model",
                    "generate": [
                        "keras_hub.models.GemmaCausalLM",
                        "keras_hub.models.GemmaCausalLM.from_preset",
                        "keras_hub.models.GemmaCausalLM.generate",
                        "keras_hub.models.GemmaCausalLM.backbone",
                        "keras_hub.models.GemmaCausalLM.preprocessor",
                        "keras_hub.models.GemmaCausalLM.score",
                    ],
                },
                {
                    "path": "gemma_causal_lm_preprocessor",
                    "title": "GemmaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.GemmaCausalLMPreprocessor",
                        "keras_hub.models.GemmaCausalLMPreprocessor.from_preset",
                        "keras_hub.models.GemmaCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "gemma3/",
            "title": "Gemma3",
            "toc": True,
            "children": [
                {
                    "path": "gemma3_tokenizer",
                    "title": "Gemma3Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.Gemma3Tokenizer",
                        "keras_hub.tokenizers.Gemma3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "gemma3_image_converter",
                    "title": "Gemma3ImageConverter",
                    "generate": [
                        "keras_hub.layers.Gemma3ImageConverter",
                        "keras_hub.layers.Gemma3ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "gemma3_backbone",
                    "title": "Gemma3Backbone model",
                    "generate": [
                        "keras_hub.models.Gemma3Backbone",
                        "keras_hub.models.Gemma3Backbone.from_preset",
                        "keras_hub.models.Gemma3Backbone.token_embedding",
                        "keras_hub.models.Gemma3Backbone.enable_lora",
                    ],
                },
                {
                    "path": "gemma3_causal_lm",
                    "title": "Gemma3CausalLM model",
                    "generate": [
                        "keras_hub.models.Gemma3CausalLM",
                        "keras_hub.models.Gemma3CausalLM.from_preset",
                        "keras_hub.models.Gemma3CausalLM.generate",
                        "keras_hub.models.Gemma3CausalLM.backbone",
                        "keras_hub.models.Gemma3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "gemma3_causal_lm_preprocessor",
                    "title": "Gemma3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.Gemma3CausalLMPreprocessor",
                        "keras_hub.models.Gemma3CausalLMPreprocessor.from_preset",
                        "keras_hub.models.Gemma3CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "gpt2/",
            "title": "GPT2",
            "toc": True,
            "children": [
                {
                    "path": "gpt2_tokenizer",
                    "title": "GPT2Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.GPT2Tokenizer",
                        "keras_hub.tokenizers.GPT2Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "gpt2_backbone",
                    "title": "GPT2Backbone model",
                    "generate": [
                        "keras_hub.models.GPT2Backbone",
                        "keras_hub.models.GPT2Backbone.from_preset",
                        "keras_hub.models.GPT2Backbone.token_embedding",
                    ],
                },
                {
                    "path": "gpt2_causal_lm",
                    "title": "GPT2CausalLM model",
                    "generate": [
                        "keras_hub.models.GPT2CausalLM",
                        "keras_hub.models.GPT2CausalLM.from_preset",
                        "keras_hub.models.GPT2CausalLM.generate",
                        "keras_hub.models.GPT2CausalLM.backbone",
                        "keras_hub.models.GPT2CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "gpt2_causal_lm_preprocessor",
                    "title": "GPT2CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.GPT2CausalLMPreprocessor",
                        "keras_hub.models.GPT2CausalLMPreprocessor.from_preset",
                        "keras_hub.models.GPT2CausalLMPreprocessor.generate_preprocess",
                        "keras_hub.models.GPT2CausalLMPreprocessor.generate_postprocess",
                        "keras_hub.models.GPT2CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "gpt_neo_x/",
            "title": "GPT-NeoX",
            "toc": True,
            "children": [
                {
                    "path": "gpt_neo_x_tokenizer",
                    "title": "GPTNeoXTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.GPTNeoXTokenizer",
                        "keras_hub.tokenizers.GPTNeoXTokenizer.from_preset",
                    ],
                },
                {
                    "path": "gpt_neo_x_backbone",
                    "title": "GPTNeoXBackbone model",
                    "generate": [
                        "keras_hub.models.GPTNeoXBackbone",
                        "keras_hub.models.GPTNeoXBackbone.from_preset",
                        "keras_hub.models.GPTNeoXBackbone.token_embedding",
                        "keras_hub.models.GPTNeoXBackbone.enable_lora",
                    ],
                },
                {
                    "path": "gpt_neo_x_causal_lm",
                    "title": "GPTNeoXCausalLM model",
                    "generate": [
                        "keras_hub.models.GPTNeoXCausalLM",
                        "keras_hub.models.GPTNeoXCausalLM.from_preset",
                        "keras_hub.models.GPTNeoXCausalLM.generate",
                        "keras_hub.models.GPTNeoXCausalLM.backbone",
                        "keras_hub.models.GPTNeoXCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "gpt_neo_x_causal_lm_preprocessor",
                    "title": "GPTNeoXCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.GPTNeoXCausalLMPreprocessor",
                        "keras_hub.models.GPTNeoXCausalLMPreprocessor.from_preset",
                        "keras_hub.models.GPTNeoXCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "gpt_oss/",
            "title": "GPT-OSS",
            "toc": True,
            "children": [
                {
                    "path": "gpt_oss_tokenizer",
                    "title": "GptOssTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.GptOssTokenizer",
                        "keras_hub.tokenizers.GptOssTokenizer.from_preset",
                    ],
                },
                {
                    "path": "gpt_oss_backbone",
                    "title": "GptOssBackbone model",
                    "generate": [
                        "keras_hub.models.GptOssBackbone",
                        "keras_hub.models.GptOssBackbone.from_preset",
                        "keras_hub.models.GptOssBackbone.token_embedding",
                    ],
                },
                {
                    "path": "gpt_oss_causal_lm",
                    "title": "GptOssCausalLM model",
                    "generate": [
                        "keras_hub.models.GptOssCausalLM",
                        "keras_hub.models.GptOssCausalLM.from_preset",
                        "keras_hub.models.GptOssCausalLM.generate",
                        "keras_hub.models.GptOssCausalLM.backbone",
                        "keras_hub.models.GptOssCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "gpt_oss_causal_lm_preprocessor",
                    "title": "GptOssCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.GptOssCausalLMPreprocessor",
                        "keras_hub.models.GptOssCausalLMPreprocessor.from_preset",
                        "keras_hub.models.GptOssCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "hgnetv2/",
            "title": "HGNetV2",
            "toc": True,
            "children": [
                {
                    "path": "hgnetv2_image_converter",
                    "title": "HGNetV2ImageConverter",
                    "generate": [
                        "keras_hub.layers.HGNetV2ImageConverter",
                        "keras_hub.layers.HGNetV2ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "hgnetv2_backbone",
                    "title": "HGNetV2Backbone model",
                    "generate": [
                        "keras_hub.models.HGNetV2Backbone",
                        "keras_hub.models.HGNetV2Backbone.from_preset",
                    ],
                },
                {
                    "path": "hgnetv2_image_classifier",
                    "title": "HGNetV2ImageClassifier model",
                    "generate": [
                        "keras_hub.models.HGNetV2ImageClassifier",
                        "keras_hub.models.HGNetV2ImageClassifier.from_preset",
                        "keras_hub.models.HGNetV2ImageClassifier.backbone",
                        "keras_hub.models.HGNetV2ImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "hgnetv2_image_classifier_preprocessor",
                    "title": "HGNetV2ImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.HGNetV2ImageClassifierPreprocessor",
                        "keras_hub.models.HGNetV2ImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.HGNetV2ImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "llama/",
            "title": "Llama",
            "toc": True,
            "children": [
                {
                    "path": "llama_tokenizer",
                    "title": "LlamaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.LlamaTokenizer",
                        "keras_hub.tokenizers.LlamaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "llama_backbone",
                    "title": "LlamaBackbone model",
                    "generate": [
                        "keras_hub.models.LlamaBackbone",
                        "keras_hub.models.LlamaBackbone.from_preset",
                        "keras_hub.models.LlamaBackbone.token_embedding",
                        "keras_hub.models.LlamaBackbone.enable_lora",
                    ],
                },
                {
                    "path": "llama_causal_lm",
                    "title": "LlamaCausalLM model",
                    "generate": [
                        "keras_hub.models.LlamaCausalLM",
                        "keras_hub.models.LlamaCausalLM.from_preset",
                        "keras_hub.models.LlamaCausalLM.generate",
                        "keras_hub.models.LlamaCausalLM.backbone",
                        "keras_hub.models.LlamaCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "llama_causal_lm_preprocessor",
                    "title": "LlamaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.LlamaCausalLMPreprocessor",
                        "keras_hub.models.LlamaCausalLMPreprocessor.from_preset",
                        "keras_hub.models.LlamaCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "llama3/",
            "title": "Llama3",
            "toc": True,
            "children": [
                {
                    "path": "llama3_tokenizer",
                    "title": "Llama3Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.Llama3Tokenizer",
                        "keras_hub.tokenizers.Llama3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "llama3_backbone",
                    "title": "Llama3Backbone model",
                    "generate": [
                        "keras_hub.models.Llama3Backbone",
                        "keras_hub.models.Llama3Backbone.from_preset",
                        "keras_hub.models.Llama3Backbone.token_embedding",
                        "keras_hub.models.Llama3Backbone.enable_lora",
                    ],
                },
                {
                    "path": "llama3_causal_lm",
                    "title": "Llama3CausalLM model",
                    "generate": [
                        "keras_hub.models.Llama3CausalLM",
                        "keras_hub.models.Llama3CausalLM.from_preset",
                        "keras_hub.models.Llama3CausalLM.generate",
                        "keras_hub.models.Llama3CausalLM.backbone",
                        "keras_hub.models.Llama3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "llama3_causal_lm_preprocessor",
                    "title": "Llama3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.Llama3CausalLMPreprocessor",
                        "keras_hub.models.Llama3CausalLMPreprocessor.from_preset",
                        "keras_hub.models.Llama3CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "mistral/",
            "title": "Mistral",
            "toc": True,
            "children": [
                {
                    "path": "mistral_tokenizer",
                    "title": "MistralTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.MistralTokenizer",
                        "keras_hub.tokenizers.MistralTokenizer.from_preset",
                    ],
                },
                {
                    "path": "mistral_backbone",
                    "title": "MistralBackbone model",
                    "generate": [
                        "keras_hub.models.MistralBackbone",
                        "keras_hub.models.MistralBackbone.from_preset",
                        "keras_hub.models.MistralBackbone.token_embedding",
                        "keras_hub.models.MistralBackbone.enable_lora",
                    ],
                },
                {
                    "path": "mistral_causal_lm",
                    "title": "MistralCausalLM model",
                    "generate": [
                        "keras_hub.models.MistralCausalLM",
                        "keras_hub.models.MistralCausalLM.from_preset",
                        "keras_hub.models.MistralCausalLM.generate",
                        "keras_hub.models.MistralCausalLM.backbone",
                        "keras_hub.models.MistralCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "mistral_causal_lm_preprocessor",
                    "title": "MistralCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MistralCausalLMPreprocessor",
                        "keras_hub.models.MistralCausalLMPreprocessor.from_preset",
                        "keras_hub.models.MistralCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "mit/",
            "title": "MiT",
            "toc": True,
            "children": [
                {
                    "path": "mit_image_converter",
                    "title": "MiTImageConverter",
                    "generate": [
                        "keras_hub.layers.MiTImageConverter",
                        "keras_hub.layers.MiTImageConverter.from_preset",
                    ],
                },
                {
                    "path": "mit_backbone",
                    "title": "MiTBackbone model",
                    "generate": [
                        "keras_hub.models.MiTBackbone",
                        "keras_hub.models.MiTBackbone.from_preset",
                    ],
                },
                {
                    "path": "mit_image_classifier",
                    "title": "MiTImageClassifier model",
                    "generate": [
                        "keras_hub.models.MiTImageClassifier",
                        "keras_hub.models.MiTImageClassifier.from_preset",
                        "keras_hub.models.MiTImageClassifier.backbone",
                        "keras_hub.models.MiTImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "mit_image_classifier_preprocessor",
                    "title": "MiTImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MiTImageClassifierPreprocessor",
                        "keras_hub.models.MiTImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.MiTImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "mixtral/",
            "title": "Mixtral",
            "toc": True,
            "children": [
                {
                    "path": "mixtral_tokenizer",
                    "title": "MixtralTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.MixtralTokenizer",
                        "keras_hub.tokenizers.MixtralTokenizer.from_preset",
                    ],
                },
                {
                    "path": "mixtral_backbone",
                    "title": "MixtralBackbone model",
                    "generate": [
                        "keras_hub.models.MixtralBackbone",
                        "keras_hub.models.MixtralBackbone.from_preset",
                        "keras_hub.models.MixtralBackbone.token_embedding",
                        "keras_hub.models.MixtralBackbone.enable_lora",
                    ],
                },
                {
                    "path": "mixtral_causal_lm",
                    "title": "MixtralCausalLM model",
                    "generate": [
                        "keras_hub.models.MixtralCausalLM",
                        "keras_hub.models.MixtralCausalLM.from_preset",
                        "keras_hub.models.MixtralCausalLM.generate",
                        "keras_hub.models.MixtralCausalLM.backbone",
                        "keras_hub.models.MixtralCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "mixtral_causal_lm_preprocessor",
                    "title": "MixtralCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MixtralCausalLMPreprocessor",
                        "keras_hub.models.MixtralCausalLMPreprocessor.from_preset",
                        "keras_hub.models.MixtralCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "mobilenet/",
            "title": "MobileNet",
            "toc": True,
            "children": [
                {
                    "path": "mobilenet_image_converter",
                    "title": "MobileNetImageConverter",
                    "generate": [
                        "keras_hub.layers.MobileNetImageConverter",
                        "keras_hub.layers.MobileNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "mobilenet_backbone",
                    "title": "MobileNetBackbone model",
                    "generate": [
                        "keras_hub.models.MobileNetBackbone",
                        "keras_hub.models.MobileNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "mobilenet_image_classifier",
                    "title": "MobileNetImageClassifier model",
                    "generate": [
                        "keras_hub.models.MobileNetImageClassifier",
                        "keras_hub.models.MobileNetImageClassifier.from_preset",
                        "keras_hub.models.MobileNetImageClassifier.backbone",
                        "keras_hub.models.MobileNetImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "mobilenet_image_classifier_preprocessor",
                    "title": "MobileNetImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MobileNetImageClassifierPreprocessor",
                        "keras_hub.models.MobileNetImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.MobileNetImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "mobilenetv5/",
            "title": "MobileNetV5",
            "toc": True,
            "children": [
                {
                    "path": "mobilenetv5_image_converter",
                    "title": "MobileNetV5ImageConverter",
                    "generate": [
                        "keras_hub.layers.MobileNetV5ImageConverter",
                        "keras_hub.layers.MobileNetV5ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "mobilenetv5_backbone",
                    "title": "MobileNetV5Backbone model",
                    "generate": [
                        "keras_hub.models.MobileNetV5Backbone",
                        "keras_hub.models.MobileNetV5Backbone.from_preset",
                    ],
                },
                {
                    "path": "mobilenetv5_image_classifier",
                    "title": "MobileNetV5ImageClassifier model",
                    "generate": [
                        "keras_hub.models.MobileNetV5ImageClassifier",
                        "keras_hub.models.MobileNetV5ImageClassifier.backbone",
                        "keras_hub.models.MobileNetV5ImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "mobilenetv5_image_classifier_preprocessor",
                    "title": "MobileNetV5ImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MobileNetV5ImageClassifierPreprocessor",
                        "keras_hub.models.MobileNetV5ImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "moonshine/",
            "title": "Moonshine",
            "toc": True,
            "children": [
                {
                    "path": "moonshine_tokenizer",
                    "title": "MoonshineTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.MoonshineTokenizer",
                        "keras_hub.tokenizers.MoonshineTokenizer.from_preset",
                    ],
                },
                {
                    "path": "moonshine_audio_converter",
                    "title": "MoonshineAudioConverter",
                    "generate": [
                        "keras_hub.layers.MoonshineAudioConverter",
                        "keras_hub.layers.MoonshineAudioConverter.from_preset",
                    ],
                },
                {
                    "path": "moonshine_backbone",
                    "title": "MoonshineBackbone model",
                    "generate": [
                        "keras_hub.models.MoonshineBackbone",
                        "keras_hub.models.MoonshineBackbone.from_preset",
                        "keras_hub.models.MoonshineBackbone.token_embedding",
                    ],
                },
                {
                    "path": "moonshine_audio_to_text",
                    "title": "MoonshineAudioToText model",
                    "generate": [
                        "keras_hub.models.MoonshineAudioToText",
                        "keras_hub.models.MoonshineAudioToText.from_preset",
                        "keras_hub.models.MoonshineAudioToText.generate",
                        "keras_hub.models.MoonshineAudioToText.backbone",
                        "keras_hub.models.MoonshineAudioToText.preprocessor",
                    ],
                },
                {
                    "path": "moonshine_audio_to_text_preprocessor",
                    "title": "MoonshineAudioToTextPreprocessor layer",
                    "generate": [
                        "keras_hub.models.MoonshineAudioToTextPreprocessor",
                        "keras_hub.models.MoonshineAudioToTextPreprocessor.from_preset",
                        "keras_hub.models.MoonshineAudioToTextPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "opt/",
            "title": "OPT",
            "toc": True,
            "children": [
                {
                    "path": "opt_tokenizer",
                    "title": "OPTTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.OPTTokenizer",
                        "keras_hub.tokenizers.OPTTokenizer.from_preset",
                    ],
                },
                {
                    "path": "opt_backbone",
                    "title": "OPTBackbone model",
                    "generate": [
                        "keras_hub.models.OPTBackbone",
                        "keras_hub.models.OPTBackbone.from_preset",
                        "keras_hub.models.OPTBackbone.token_embedding",
                    ],
                },
                {
                    "path": "opt_causal_lm",
                    "title": "OPTCausalLM model",
                    "generate": [
                        "keras_hub.models.OPTCausalLM",
                        "keras_hub.models.OPTCausalLM.from_preset",
                        "keras_hub.models.OPTCausalLM.generate",
                        "keras_hub.models.OPTCausalLM.backbone",
                        "keras_hub.models.OPTCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "opt_causal_lm_preprocessor",
                    "title": "OPTCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.OPTCausalLMPreprocessor",
                        "keras_hub.models.OPTCausalLMPreprocessor.from_preset",
                        "keras_hub.models.OPTCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "pali_gemma/",
            "title": "PaliGemma",
            "toc": True,
            "children": [
                {
                    "path": "pali_gemma_tokenizer",
                    "title": "PaliGemmaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.PaliGemmaTokenizer",
                        "keras_hub.tokenizers.PaliGemmaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "pali_gemma_backbone",
                    "title": "PaliGemmaBackbone model",
                    "generate": [
                        "keras_hub.models.PaliGemmaBackbone",
                        "keras_hub.models.PaliGemmaBackbone.from_preset",
                        "keras_hub.models.PaliGemmaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "pali_gemma_causal_lm",
                    "title": "PaliGemmaCausalLM model",
                    "generate": [
                        "keras_hub.models.PaliGemmaCausalLM",
                        "keras_hub.models.PaliGemmaCausalLM.from_preset",
                        "keras_hub.models.PaliGemmaCausalLM.generate",
                        "keras_hub.models.PaliGemmaCausalLM.backbone",
                        "keras_hub.models.PaliGemmaCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "pali_gemma_causal_lm_preprocessor",
                    "title": "PaliGemmaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.PaliGemmaCausalLMPreprocessor",
                        "keras_hub.models.PaliGemmaCausalLMPreprocessor.from_preset",
                        "keras_hub.models.PaliGemmaCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "parseq/",
            "title": "PARSeq",
            "toc": True,
            "children": [
                {
                    "path": "parseq_tokenizer",
                    "title": "PARSeqTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.PARSeqTokenizer",
                        "keras_hub.tokenizers.PARSeqTokenizer.from_preset",
                    ],
                },
                {
                    "path": "parseq_backbone",
                    "title": "PARSeqBackbone model",
                    "generate": [
                        "keras_hub.models.PARSeqBackbone",
                        "keras_hub.models.PARSeqBackbone.from_preset",
                    ],
                },
                {
                    "path": "parseq_causal_lm",
                    "title": "PARSeqCausalLM model",
                    "generate": [
                        "keras_hub.models.PARSeqCausalLM",
                        "keras_hub.models.PARSeqCausalLM.from_preset",
                        "keras_hub.models.PARSeqCausalLM.generate",
                        "keras_hub.models.PARSeqCausalLM.backbone",
                        "keras_hub.models.PARSeqCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "parseq_causal_lmpreprocessor",
                    "title": "PARSeqCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.PARSeqCausalLMPreprocessor",
                        "keras_hub.models.PARSeqCausalLMPreprocessor.from_preset",
                        "keras_hub.models.PARSeqCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "phi3/",
            "title": "Phi3",
            "toc": True,
            "children": [
                {
                    "path": "phi3_tokenizer",
                    "title": "Phi3Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.Phi3Tokenizer",
                        "keras_hub.tokenizers.Phi3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "phi3_backbone",
                    "title": "Phi3Backbone model",
                    "generate": [
                        "keras_hub.models.Phi3Backbone",
                        "keras_hub.models.Phi3Backbone.from_preset",
                        "keras_hub.models.Phi3Backbone.token_embedding",
                    ],
                },
                {
                    "path": "phi3_causal_lm",
                    "title": "Phi3CausalLM model",
                    "generate": [
                        "keras_hub.models.Phi3CausalLM",
                        "keras_hub.models.Phi3CausalLM.from_preset",
                        "keras_hub.models.Phi3CausalLM.generate",
                        "keras_hub.models.Phi3CausalLM.backbone",
                        "keras_hub.models.Phi3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "phi3_causal_lm_preprocessor",
                    "title": "Phi3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.Phi3CausalLMPreprocessor",
                        "keras_hub.models.Phi3CausalLMPreprocessor.from_preset",
                        "keras_hub.models.Phi3CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "qwen/",
            "title": "Qwen",
            "toc": True,
            "children": [
                {
                    "path": "qwen_tokenizer",
                    "title": "QwenTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.QwenTokenizer",
                        "keras_hub.tokenizers.QwenTokenizer.from_preset",
                    ],
                },
                {
                    "path": "qwen_backbone",
                    "title": "QwenBackbone model",
                    "generate": [
                        "keras_hub.models.QwenBackbone",
                        "keras_hub.models.QwenBackbone.from_preset",
                        "keras_hub.models.QwenBackbone.token_embedding",
                        "keras_hub.models.QwenBackbone.enable_lora",
                    ],
                },
                {
                    "path": "qwen_causal_lm",
                    "title": "QwenCausalLM model",
                    "generate": [
                        "keras_hub.models.QwenCausalLM",
                        "keras_hub.models.QwenCausalLM.from_preset",
                        "keras_hub.models.QwenCausalLM.generate",
                        "keras_hub.models.QwenCausalLM.backbone",
                        "keras_hub.models.QwenCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "qwen_causal_lm_preprocessor",
                    "title": "QwenCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.QwenCausalLMPreprocessor",
                        "keras_hub.models.QwenCausalLMPreprocessor.from_preset",
                        "keras_hub.models.QwenCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "qwen3/",
            "title": "Qwen3",
            "toc": True,
            "children": [
                {
                    "path": "qwen3_tokenizer",
                    "title": "Qwen3Tokenizer",
                    "generate": [
                        "keras_hub.models.Qwen3Tokenizer",
                        "keras_hub.models.Qwen3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "qwen3_backbone",
                    "title": "Qwen3Backbone model",
                    "generate": [
                        "keras_hub.models.Qwen3Backbone",
                        "keras_hub.models.Qwen3Backbone.from_preset",
                        "keras_hub.models.Qwen3Backbone.token_embedding",
                        "keras_hub.models.Qwen3Backbone.enable_lora",
                    ],
                },
                {
                    "path": "qwen3_causal_lm",
                    "title": "Qwen3CausalLM model",
                    "generate": [
                        "keras_hub.models.Qwen3CausalLM",
                        "keras_hub.models.Qwen3CausalLM.from_preset",
                        "keras_hub.models.Qwen3CausalLM.generate",
                        "keras_hub.models.Qwen3CausalLM.backbone",
                        "keras_hub.models.Qwen3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "qwen3_causal_lm_preprocessor",
                    "title": "Qwen3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.Qwen3CausalLMPreprocessor",
                        "keras_hub.models.Qwen3CausalLMPreprocessor.from_preset",
                        "keras_hub.models.Qwen3CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "qwen3_moe/",
            "title": "Qwen3Moe",
            "toc": True,
            "children": [
                {
                    "path": "qwen3_moe_tokenizer",
                    "title": "Qwen3MoeTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.Qwen3MoeTokenizer",
                        "keras_hub.tokenizers.Qwen3MoeTokenizer.from_preset",
                    ],
                },
                {
                    "path": "qwen3_moe_backbone",
                    "title": "Qwen3MoeBackbone model",
                    "generate": [
                        "keras_hub.models.Qwen3MoeBackbone",
                        "keras_hub.models.Qwen3MoeBackbone.from_preset",
                        "keras_hub.models.Qwen3MoeBackbone.token_embedding",
                        "keras_hub.models.Qwen3MoeBackbone.enable_lora",
                    ],
                },
                {
                    "path": "qwen3_moe_causal_lm",
                    "title": "Qwen3MoeCausalLM model",
                    "generate": [
                        "keras_hub.models.Qwen3MoeCausalLM",
                        "keras_hub.models.Qwen3MoeCausalLM.from_preset",
                        "keras_hub.models.Qwen3MoeCausalLM.generate",
                        "keras_hub.models.Qwen3MoeCausalLM.backbone",
                        "keras_hub.models.Qwen3MoeCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "qwen3_moe_causal_lm_preprocessor",
                    "title": "Qwen3MoeCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.Qwen3MoeCausalLMPreprocessor",
                        "keras_hub.models.Qwen3MoeCausalLMPreprocessor.from_preset",
                        "keras_hub.models.Qwen3MoeCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "qwen_moe/",
            "title": "QwenMoe",
            "toc": True,
            "children": [
                {
                    "path": "qwen_moe_tokenizer",
                    "title": "QwenMoeTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.QwenMoeTokenizer",
                        "keras_hub.tokenizers.QwenMoeTokenizer.from_preset",
                    ],
                },
                {
                    "path": "qwen_moe_backbone",
                    "title": "QwenMoeBackbone model",
                    "generate": [
                        "keras_hub.models.QwenMoeBackbone",
                        "keras_hub.models.QwenMoeBackbone.from_preset",
                        "keras_hub.models.QwenMoeBackbone.token_embedding",
                        "keras_hub.models.QwenMoeBackbone.enable_lora",
                    ],
                },
                {
                    "path": "qwen_moe_causal_lm",
                    "title": "QwenMoeCausalLM model",
                    "generate": [
                        "keras_hub.models.QwenMoeCausalLM",
                        "keras_hub.models.QwenMoeCausalLM.from_preset",
                        "keras_hub.models.QwenMoeCausalLM.generate",
                        "keras_hub.models.QwenMoeCausalLM.backbone",
                        "keras_hub.models.QwenMoeCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "qwen_moe_causal_lm_preprocessor",
                    "title": "QwenMoeCausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.QwenMoeCausalLMPreprocessor",
                        "keras_hub.models.QwenMoeCausalLMPreprocessor.from_preset",
                        "keras_hub.models.QwenMoeCausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "resnet/",
            "title": "ResNet",
            "toc": True,
            "children": [
                {
                    "path": "resnet_image_converter",
                    "title": "ResNetImageConverter",
                    "generate": [
                        "keras_hub.layers.ResNetImageConverter",
                        "keras_hub.layers.ResNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "resnet_backbone",
                    "title": "ResNetBackbone model",
                    "generate": [
                        "keras_hub.models.ResNetBackbone",
                        "keras_hub.models.ResNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "resnet_image_classifier",
                    "title": "ResNetImageClassifier model",
                    "generate": [
                        "keras_hub.models.ResNetImageClassifier",
                        "keras_hub.models.ResNetImageClassifier.from_preset",
                        "keras_hub.models.ResNetImageClassifier.backbone",
                        "keras_hub.models.ResNetImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "resnet_image_classifier_preprocessor",
                    "title": "ResNetImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.ResNetImageClassifierPreprocessor",
                        "keras_hub.models.ResNetImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.ResNetImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "retinanet/",
            "title": "RetinaNet",
            "toc": True,
            "children": [
                {
                    "path": "retinanet_image_converter",
                    "title": "RetinaNetImageConverter",
                    "generate": [
                        "keras_hub.layers.RetinaNetImageConverter",
                        "keras_hub.layers.RetinaNetImageConverter.from_preset",
                    ],
                },
                {
                    "path": "retinanet_backbone",
                    "title": "RetinaNetBackbone model",
                    "generate": [
                        "keras_hub.models.RetinaNetBackbone",
                        "keras_hub.models.RetinaNetBackbone.from_preset",
                    ],
                },
                {
                    "path": "retinanet_object_detector",
                    "title": "RetinaNetObjectDetector model",
                    "generate": [
                        "keras_hub.models.RetinaNetObjectDetector",
                        "keras_hub.models.RetinaNetObjectDetector.from_preset",
                        "keras_hub.models.RetinaNetObjectDetector.backbone",
                        "keras_hub.models.RetinaNetObjectDetector.preprocessor",
                    ],
                },
                {
                    "path": "retinanet_object_detector_preprocessor",
                    "title": "RetinaNetObjectDetectorPreprocessor layer",
                    "generate": [
                        "keras_hub.models.RetinaNetObjectDetectorPreprocessor",
                        "keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset",
                        "keras_hub.models.RetinaNetObjectDetectorPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "roberta/",
            "title": "Roberta",
            "toc": True,
            "children": [
                {
                    "path": "roberta_tokenizer",
                    "title": "RobertaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.RobertaTokenizer",
                        "keras_hub.tokenizers.RobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "roberta_backbone",
                    "title": "RobertaBackbone model",
                    "generate": [
                        "keras_hub.models.RobertaBackbone",
                        "keras_hub.models.RobertaBackbone.from_preset",
                        "keras_hub.models.RobertaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "roberta_text_classifier",
                    "title": "RobertaTextClassifier model",
                    "generate": [
                        "keras_hub.models.RobertaTextClassifier",
                        "keras_hub.models.RobertaTextClassifier.from_preset",
                        "keras_hub.models.RobertaTextClassifier.backbone",
                        "keras_hub.models.RobertaTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "roberta_text_classifier_preprocessor",
                    "title": "RobertaTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.RobertaTextClassifierPreprocessor",
                        "keras_hub.models.RobertaTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.RobertaTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "roberta_masked_lm",
                    "title": "RobertaMaskedLM model",
                    "generate": [
                        "keras_hub.models.RobertaMaskedLM",
                        "keras_hub.models.RobertaMaskedLM.from_preset",
                        "keras_hub.models.RobertaMaskedLM.backbone",
                        "keras_hub.models.RobertaMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "roberta_masked_lm_preprocessor",
                    "title": "RobertaMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.RobertaMaskedLMPreprocessor",
                        "keras_hub.models.RobertaMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.RobertaMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "rwkv7/",
            "title": "RWKV7",
            "toc": True,
            "children": [
                {
                    "path": "rwkv7_tokenizer",
                    "title": "RWKV7Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.RWKVTokenizer",
                        "keras_hub.tokenizers.RWKVTokenizer.from_preset",
                    ],
                },
                {
                    "path": "rwkv7_backbone",
                    "title": "RWKV7Backbone model",
                    "generate": [
                        "keras_hub.models.RWKV7Backbone",
                        "keras_hub.models.RWKV7Backbone.from_preset",
                        "keras_hub.models.RWKV7Backbone.token_embedding",
                    ],
                },
                {
                    "path": "rwkv7_causal_lm",
                    "title": "RWKV7CausalLM model",
                    "generate": [
                        "keras_hub.models.RWKV7CausalLM",
                        "keras_hub.models.RWKV7CausalLM.from_preset",
                        "keras_hub.models.RWKV7CausalLM.generate",
                        "keras_hub.models.RWKV7CausalLM.backbone",
                        "keras_hub.models.RWKV7CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "rwkv7_causal_lm_preprocessor",
                    "title": "RWKV7CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.RWKV7CausalLMPreprocessor",
                        "keras_hub.models.RWKV7CausalLMPreprocessor.from_preset",
                        "keras_hub.models.RWKV7CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "segformer/",
            "title": "SegFormer",
            "toc": True,
            "children": [
                {
                    "path": "segformer_image_converter",
                    "title": "SegFormerImageConverter",
                    "generate": [
                        "keras_hub.layers.SegFormerImageConverter",
                        "keras_hub.layers.SegFormerImageConverter.from_preset",
                    ],
                },
                {
                    "path": "segformer_backbone",
                    "title": "SegFormerBackbone model",
                    "generate": [
                        "keras_hub.models.SegFormerBackbone",
                        "keras_hub.models.SegFormerBackbone.from_preset",
                    ],
                },
                {
                    "path": "segformer_image_segmenter",
                    "title": "SegFormerImageSegmenter model",
                    "generate": [
                        "keras_hub.models.SegFormerImageSegmenter",
                        "keras_hub.models.SegFormerImageSegmenter.from_preset",
                        "keras_hub.models.SegFormerImageSegmenter.backbone",
                        "keras_hub.models.SegFormerImageSegmenter.preprocessor",
                    ],
                },
                {
                    "path": "segformer_image_segmenter_preprocessor",
                    "title": "SegFormerImageSegmenterPreprocessor layer",
                    "generate": [
                        "keras_hub.models.SegFormerImageSegmenterPreprocessor",
                        "keras_hub.models.SegFormerImageSegmenterPreprocessor.from_preset",
                        "keras_hub.models.SegFormerImageSegmenterPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "sam/",
            "title": "Segment Anything Model",
            "toc": True,
            "children": [
                {
                    "path": "sam_image_converter",
                    "title": "SAMImageConverter",
                    "generate": [
                        "keras_hub.layers.SAMImageConverter",
                        "keras_hub.layers.SAMImageConverter.from_preset",
                    ],
                },
                {
                    "path": "sam_backbone",
                    "title": "SAMBackbone model",
                    "generate": [
                        "keras_hub.models.SAMBackbone",
                        "keras_hub.models.SAMBackbone.from_preset",
                    ],
                },
                {
                    "path": "sam_image_segmenter",
                    "title": "SAMImageSegmenter model",
                    "generate": [
                        "keras_hub.models.SAMImageSegmenter",
                        "keras_hub.models.SAMImageSegmenter.from_preset",
                        "keras_hub.models.SAMImageSegmenter.backbone",
                        "keras_hub.models.SAMImageSegmenter.preprocessor",
                    ],
                },
                {
                    "path": "sam_image_segmenter_preprocessor",
                    "title": "SAMImageSegmenterPreprocessor layer",
                    "generate": [
                        "keras_hub.models.SAMImageSegmenterPreprocessor",
                        "keras_hub.models.SAMImageSegmenterPreprocessor.from_preset",
                        "keras_hub.models.SAMImageSegmenterPreprocessor.image_converter",
                    ],
                },
                {
                    "path": "sam_prompt_encoder",
                    "title": "SAMPromptEncoder layer",
                    "generate": [
                        "keras_hub.layers.SAMPromptEncoder",
                    ],
                },
                {
                    "path": "sam_mask_decoder",
                    "title": "SAMMaskDecoder layer",
                    "generate": [
                        "keras_hub.layers.SAMMaskDecoder",
                    ],
                },
            ],
        },
        {
            "path": "sam3/",
            "title": "Segment Anything Model 3",
            "toc": True,
            "children": [
                {
                    "path": "sam3_tokenizer",
                    "title": "SAM3Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.SAM3Tokenizer",
                        "keras_hub.tokenizers.SAM3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "sam3_image_converter",
                    "title": "SAM3ImageConverter",
                    "generate": [
                        "keras_hub.layers.SAM3ImageConverter",
                        "keras_hub.layers.SAM3ImageConverter.from_preset",
                    ],
                },
                {
                    "path": "sam3_pc_backbone",
                    "title": "SAM3PromptableConceptBackbone model",
                    "generate": [
                        "keras_hub.models.SAM3PromptableConceptBackbone",
                        "keras_hub.models.SAM3PromptableConceptBackbone.from_preset",
                    ],
                },
                {
                    "path": "sam3_pc_image_segmenter",
                    "title": "SAM3PromptableConceptImageSegmenter model",
                    "generate": [
                        "keras_hub.models.SAM3PromptableConceptImageSegmenter",
                        "keras_hub.models.SAM3PromptableConceptImageSegmenter.from_preset",
                        "keras_hub.models.SAM3PromptableConceptImageSegmenter.backbone",
                        "keras_hub.models.SAM3PromptableConceptImageSegmenter.preprocessor",
                    ],
                },
                {
                    "path": "sam3_pc_image_segmenter_preprocessor",
                    "title": "SAM3PromptableConceptImageSegmenterPreprocessor layer",
                    "generate": [
                        "keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor",
                        "keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor.from_preset",
                        "keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "siglip/",
            "title": "SigLIP",
            "toc": True,
            "children": [
                {
                    "path": "siglip_tokenizer",
                    "title": "SigLIPTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.SigLIPTokenizer",
                        "keras_hub.tokenizers.SigLIPTokenizer.from_preset",
                    ],
                },
                {
                    "path": "siglip_image_converter",
                    "title": "SigLIPImageConverter",
                    "generate": [
                        "keras_hub.layers.SigLIPImageConverter",
                        "keras_hub.layers.SigLIPImageConverter.from_preset",
                    ],
                },
                {
                    "path": "siglip_backbone",
                    "title": "SigLIPBackbone model",
                    "generate": [
                        "keras_hub.models.SigLIPBackbone",
                        "keras_hub.models.SigLIPBackbone.from_preset",
                    ],
                },
                {
                    "path": "siglip_preprocessor",
                    "title": "SigLIPPreprocessor",
                    "generate": [
                        "keras_hub.models.SigLIPPreprocessor",
                        "keras_hub.models.SigLIPPreprocessor.from_preset",
                        "keras_hub.models.SigLIPPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "smollm3/",
            "title": "SmolLM3",
            "toc": True,
            "children": [
                {
                    "path": "smollm3_tokenizer",
                    "title": "SmolLM3Tokenizer",
                    "generate": [
                        "keras_hub.models.SmolLM3Tokenizer",
                        "keras_hub.models.SmolLM3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "smollm3_backbone",
                    "title": "SmolLM3Backbone model",
                    "generate": [
                        "keras_hub.models.SmolLM3Backbone",
                        "keras_hub.models.SmolLM3Backbone.from_preset",
                        "keras_hub.models.SmolLM3Backbone.token_embedding",
                    ],
                },
                {
                    "path": "smollm3_causal_lm",
                    "title": "SmolLM3CausalLM model",
                    "generate": [
                        "keras_hub.models.SmolLM3CausalLM",
                        "keras_hub.models.SmolLM3CausalLM.from_preset",
                        "keras_hub.models.SmolLM3CausalLM.generate",
                        "keras_hub.models.SmolLM3CausalLM.backbone",
                        "keras_hub.models.SmolLM3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "smollm3_causal_lm_preprocessor",
                    "title": "SmolLM3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.SmolLM3CausalLMPreprocessor",
                        "keras_hub.models.SmolLM3CausalLMPreprocessor.from_preset",
                        "keras_hub.models.SmolLM3CausalLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "stable_diffusion_3/",
            "title": "Stable Diffusion 3",
            "toc": True,
            "children": [
                {
                    "path": "sam_image_converter",
                    "title": "SAMImageConverter",
                    "generate": [
                        "keras_hub.layers.SAMImageConverter",
                        "keras_hub.layers.SAMImageConverter.from_preset",
                    ],
                },
                {
                    "path": "stable_diffusion_3_backbone",
                    "title": "StableDiffusion3Backbone model",
                    "generate": [
                        "keras_hub.models.StableDiffusion3Backbone",
                        "keras_hub.models.StableDiffusion3Backbone.from_preset",
                    ],
                },
                {
                    "path": "stable_diffusion_3_text_to_image",
                    "title": "StableDiffusion3TextToImage model",
                    "generate": [
                        "keras_hub.models.StableDiffusion3TextToImage",
                        "keras_hub.models.StableDiffusion3TextToImage.from_preset",
                        "keras_hub.models.StableDiffusion3TextToImage.backbone",
                        "keras_hub.models.StableDiffusion3TextToImage.generate",
                        "keras_hub.models.StableDiffusion3TextToImage.preprocessor",
                    ],
                },
                {
                    "path": "stable_diffusion_3_text_to_image_preprocessor",
                    "title": "StableDiffusion3TextToImagePreprocessor layer",
                    "generate": [
                        "keras_hub.models.StableDiffusion3TextToImagePreprocessor",
                        "keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset",
                    ],
                },
                {
                    "path": "stable_diffusion_3_image_to_image",
                    "title": "StableDiffusion3ImageToImage model",
                    "generate": [
                        "keras_hub.models.StableDiffusion3ImageToImage",
                        "keras_hub.models.StableDiffusion3ImageToImage.from_preset",
                        "keras_hub.models.StableDiffusion3ImageToImage.backbone",
                        "keras_hub.models.StableDiffusion3ImageToImage.generate",
                        "keras_hub.models.StableDiffusion3ImageToImage.preprocessor",
                    ],
                },
                {
                    "path": "stable_diffusion_3_inpaint",
                    "title": "StableDiffusion3Inpaint model",
                    "generate": [
                        "keras_hub.models.StableDiffusion3Inpaint",
                        "keras_hub.models.StableDiffusion3Inpaint.from_preset",
                        "keras_hub.models.StableDiffusion3Inpaint.backbone",
                        "keras_hub.models.StableDiffusion3Inpaint.generate",
                        "keras_hub.models.StableDiffusion3Inpaint.preprocessor",
                    ],
                },
            ],
        },
        {
            "path": "t5/",
            "title": "T5",
            "toc": True,
            "children": [
                {
                    "path": "t5_tokenizer",
                    "title": "T5Tokenizer",
                    "generate": [
                        "keras_hub.tokenizers.T5Tokenizer",
                        "keras_hub.tokenizers.T5Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "t5_backbone",
                    "title": "T5Backbone model",
                    "generate": [
                        "keras_hub.models.T5Backbone",
                        "keras_hub.models.T5Backbone.from_preset",
                        "keras_hub.models.T5Backbone.token_embedding",
                    ],
                },
                {
                    "path": "t5_preprocessor",
                    "title": "T5Preprocessor layer",
                    "generate": [
                        "keras_hub.models.T5Preprocessor",
                        "keras_hub.models.T5Preprocessor.from_preset",
                        "keras_hub.models.T5Preprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "t5gemma/",
            "title": "T5Gemma",
            "toc": True,
            "children": [
                {
                    "path": "t5gemma_tokenizer",
                    "title": "T5GemmaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.T5GemmaTokenizer",
                        "keras_hub.tokenizers.T5GemmaTokenizer.from_preset",
                        "keras_hub.models.T5GemmaTokenizer",
                        "keras_hub.models.T5GemmaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "t5gemma_backbone",
                    "title": "T5GemmaBackbone model",
                    "generate": [
                        "keras_hub.models.T5GemmaBackbone",
                        "keras_hub.models.T5GemmaBackbone.from_preset",
                        "keras_hub.models.T5GemmaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "t5gemma_seq_2_seq_lm",
                    "title": "T5GemmaSeq2SeqLM model",
                    "generate": [
                        "keras_hub.models.T5GemmaSeq2SeqLM",
                        "keras_hub.models.T5GemmaSeq2SeqLM.from_preset",
                        "keras_hub.models.T5GemmaSeq2SeqLM.generate",
                        "keras_hub.models.T5GemmaSeq2SeqLM.backbone",
                        "keras_hub.models.T5GemmaSeq2SeqLM.preprocessor",
                    ],
                },
                {
                    "path": "t5gemma_seq_2_seq_lm_preprocessor",
                    "title": "T5GemmaSeq2SeqLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.T5GemmaSeq2SeqLMPreprocessor",
                        "keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.from_preset",
                        "keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.generate_preprocess",
                        "keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.generate_postprocess",
                        "keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
        {
            "path": "vgg/",
            "title": "VGG",
            "toc": True,
            "children": [
                {
                    "path": "vgg_image_converter",
                    "title": "VGGImageConverter",
                    "generate": [
                        "keras_hub.layers.VGGImageConverter",
                        "keras_hub.layers.VGGImageConverter.from_preset",
                    ],
                },
                {
                    "path": "vgg_backbone",
                    "title": "VGGBackbone model",
                    "generate": [
                        "keras_hub.models.VGGBackbone",
                        "keras_hub.models.VGGBackbone.from_preset",
                    ],
                },
                {
                    "path": "vgg_image_classifier",
                    "title": "VGGImageClassifier model",
                    "generate": [
                        "keras_hub.models.VGGImageClassifier",
                        "keras_hub.models.VGGImageClassifier.from_preset",
                        "keras_hub.models.VGGImageClassifier.backbone",
                        "keras_hub.models.VGGImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "vgg_image_classifier_preprocessor",
                    "title": "VGGImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.VGGImageClassifierPreprocessor",
                        "keras_hub.models.VGGImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.VGGImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "vit/",
            "title": "ViT",
            "toc": True,
            "children": [
                {
                    "path": "vit_image_converter",
                    "title": "ViTImageConverter",
                    "generate": [
                        "keras_hub.layers.ViTImageConverter",
                        "keras_hub.layers.ViTImageConverter.from_preset",
                    ],
                },
                {
                    "path": "vit_backbone",
                    "title": "ViTBackbone model",
                    "generate": [
                        "keras_hub.models.ViTBackbone",
                        "keras_hub.models.ViTBackbone.from_preset",
                    ],
                },
                {
                    "path": "vit_image_classifier",
                    "title": "ViTImageClassifier model",
                    "generate": [
                        "keras_hub.models.ViTImageClassifier",
                        "keras_hub.models.ViTImageClassifier.from_preset",
                        "keras_hub.models.ViTImageClassifier.backbone",
                        "keras_hub.models.ViTImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "vit_image_classifier_preprocessor",
                    "title": "ViTImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.ViTImageClassifierPreprocessor",
                        "keras_hub.models.ViTImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.ViTImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "vit_det/",
            "title": "ViTDet",
            "toc": True,
            "children": [
                {
                    "path": "ViTDetBackbone",
                    "title": "VitDet model",
                    "generate": [
                        "keras_hub.models.ViTDetBackbone",
                        "keras_hub.models.ViTDetBackbone.from_preset",
                    ],
                },
            ],
        },
        {
            "path": "whisper/",
            "title": "Whisper",
            "toc": True,
            "children": [
                {
                    "path": "whisper_tokenizer",
                    "title": "WhisperTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.WhisperTokenizer",
                        "keras_hub.tokenizers.WhisperTokenizer.from_preset",
                    ],
                },
                {
                    "path": "whisper_audio_converter",
                    "title": "WhisperAudioConverter",
                    "generate": [
                        "keras_hub.layers.WhisperAudioConverter",
                        "keras_hub.layers.WhisperAudioConverter.from_preset",
                    ],
                },
                {
                    "path": "whisper_backbone",
                    "title": "WhisperBackbone model",
                    "generate": [
                        "keras_hub.models.WhisperBackbone",
                        "keras_hub.models.WhisperBackbone.from_preset",
                    ],
                },
            ],
        },
        {
            "path": "xception/",
            "title": "Xception",
            "toc": True,
            "children": [
                {
                    "path": "xception_image_converter",
                    "title": "XceptionImageConverter",
                    "generate": [
                        "keras_hub.layers.XceptionImageConverter",
                        "keras_hub.layers.XceptionImageConverter.from_preset",
                    ],
                },
                {
                    "path": "xception_backbone",
                    "title": "XceptionBackbone model",
                    "generate": [
                        "keras_hub.models.XceptionBackbone",
                        "keras_hub.models.XceptionBackbone.from_preset",
                    ],
                },
                {
                    "path": "xception_image_classifier",
                    "title": "XceptionImageClassifier model",
                    "generate": [
                        "keras_hub.models.XceptionImageClassifier",
                        "keras_hub.models.XceptionImageClassifier.from_preset",
                        "keras_hub.models.XceptionImageClassifier.backbone",
                        "keras_hub.models.XceptionImageClassifier.preprocessor",
                    ],
                },
                {
                    "path": "xception_image_classifier_preprocessor",
                    "title": "XceptionImageClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.XceptionImageClassifierPreprocessor",
                        "keras_hub.models.XceptionImageClassifierPreprocessor.from_preset",
                        "keras_hub.models.XceptionImageClassifierPreprocessor.image_converter",
                    ],
                },
            ],
        },
        {
            "path": "xlnet/",
            "title": "XLNet",
            "toc": True,
            "children": [
                {
                    "path": "xlnet_backbone",
                    "title": "XLNetBackbone model",
                    "generate": [
                        "keras_hub.models.XLNetBackbone",
                        "keras_hub.models.XLNetBackbone.from_preset",
                        "keras_hub.models.XLNetBackbone.token_embedding",
                    ],
                },
            ],
        },
        {
            "path": "xlm_roberta/",
            "title": "XLMRoberta",
            "toc": True,
            "children": [
                {
                    "path": "xlm_roberta_tokenizer",
                    "title": "XLMRobertaTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.XLMRobertaTokenizer",
                        "keras_hub.tokenizers.XLMRobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "xlm_roberta_backbone",
                    "title": "XLMRobertaBackbone model",
                    "generate": [
                        "keras_hub.models.XLMRobertaBackbone",
                        "keras_hub.models.XLMRobertaBackbone.from_preset",
                        "keras_hub.models.XLMRobertaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "xlm_roberta_text_classifier",
                    "title": "XLMRobertaTextClassifier model",
                    "generate": [
                        "keras_hub.models.XLMRobertaTextClassifier",
                        "keras_hub.models.XLMRobertaTextClassifier.from_preset",
                        "keras_hub.models.XLMRobertaTextClassifier.backbone",
                        "keras_hub.models.XLMRobertaTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "xlm_roberta_text_classifier_preprocessor",
                    "title": "XLMRobertaTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_hub.models.XLMRobertaTextClassifierPreprocessor",
                        "keras_hub.models.XLMRobertaTextClassifierPreprocessor.from_preset",
                        "keras_hub.models.XLMRobertaTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "xlm_roberta_masked_lm",
                    "title": "XLMRobertaMaskedLM model",
                    "generate": [
                        "keras_hub.models.XLMRobertaMaskedLM",
                        "keras_hub.models.XLMRobertaMaskedLM.from_preset",
                        "keras_hub.models.XLMRobertaMaskedLM.backbone",
                        "keras_hub.models.XLMRobertaMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "xlm_roberta_masked_lm_preprocessor",
                    "title": "XLMRobertaMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_hub.models.XLMRobertaMaskedLMPreprocessor",
                        "keras_hub.models.XLMRobertaMaskedLMPreprocessor.from_preset",
                        "keras_hub.models.XLMRobertaMaskedLMPreprocessor.tokenizer",
                    ],
                },
            ],
        },
    ],
}

SAMPLERS_MASTER = {
    "path": "samplers/",
    "title": "Samplers",
    "toc": True,
    "children": [
        {
            "path": "samplers",
            "title": "Sampler base class",
            "generate": [
                "keras_hub.samplers.Sampler",
                "keras_hub.samplers.Sampler.get_next_token",
            ],
        },
        {
            "path": "beam_sampler",
            "title": "BeamSampler",
            "generate": ["keras_hub.samplers.BeamSampler"],
        },
        {
            "path": "contrastive_sampler",
            "title": "ContrastiveSampler",
            "generate": ["keras_hub.samplers.ContrastiveSampler"],
        },
        {
            "path": "greedy_sampler",
            "title": "GreedySampler",
            "generate": ["keras_hub.samplers.GreedySampler"],
        },
        {
            "path": "random_sampler",
            "title": "RandomSampler",
            "generate": ["keras_hub.samplers.RandomSampler"],
        },
        {
            "path": "top_k_sampler",
            "title": "TopKSampler",
            "generate": ["keras_hub.samplers.TopKSampler"],
        },
        {
            "path": "top_p_sampler",
            "title": "TopPSampler",
            "generate": ["keras_hub.samplers.TopPSampler"],
        },
    ],
}

TOKENIZERS_MASTER = {
    "path": "tokenizers/",
    "title": "Tokenizers",
    "toc": True,
    "children": [
        {
            "path": "tokenizer",
            "title": "Tokenizer",
            "generate": [
                "keras_hub.tokenizers.Tokenizer",
                "keras_hub.tokenizers.Tokenizer.from_preset",
                "keras_hub.tokenizers.Tokenizer.save_to_preset",
            ],
        },
        {
            "path": "word_piece_tokenizer",
            "title": "WordPieceTokenizer",
            "generate": [
                "keras_hub.tokenizers.WordPieceTokenizer",
                "keras_hub.tokenizers.WordPieceTokenizer.tokenize",
                "keras_hub.tokenizers.WordPieceTokenizer.detokenize",
                "keras_hub.tokenizers.WordPieceTokenizer.get_vocabulary",
                "keras_hub.tokenizers.WordPieceTokenizer.vocabulary_size",
                "keras_hub.tokenizers.WordPieceTokenizer.token_to_id",
                "keras_hub.tokenizers.WordPieceTokenizer.id_to_token",
            ],
        },
        {
            "path": "sentence_piece_tokenizer",
            "title": "SentencePieceTokenizer",
            "generate": [
                "keras_hub.tokenizers.SentencePieceTokenizer",
                "keras_hub.tokenizers.SentencePieceTokenizer.tokenize",
                "keras_hub.tokenizers.SentencePieceTokenizer.detokenize",
                "keras_hub.tokenizers.SentencePieceTokenizer.get_vocabulary",
                "keras_hub.tokenizers.SentencePieceTokenizer.vocabulary_size",
                "keras_hub.tokenizers.SentencePieceTokenizer.token_to_id",
                "keras_hub.tokenizers.SentencePieceTokenizer.id_to_token",
            ],
        },
        {
            "path": "byte_pair_tokenizer",
            "title": "BytePairTokenizer",
            "generate": [
                "keras_hub.tokenizers.BytePairTokenizer",
                "keras_hub.tokenizers.BytePairTokenizer.tokenize",
                "keras_hub.tokenizers.BytePairTokenizer.detokenize",
                "keras_hub.tokenizers.BytePairTokenizer.get_vocabulary",
                "keras_hub.tokenizers.BytePairTokenizer.vocabulary_size",
                "keras_hub.tokenizers.BytePairTokenizer.token_to_id",
                "keras_hub.tokenizers.BytePairTokenizer.id_to_token",
            ],
        },
        {
            "path": "byte_tokenizer",
            "title": "ByteTokenizer",
            "generate": [
                "keras_hub.tokenizers.ByteTokenizer",
                "keras_hub.tokenizers.ByteTokenizer.tokenize",
                "keras_hub.tokenizers.ByteTokenizer.detokenize",
                "keras_hub.tokenizers.ByteTokenizer.get_vocabulary",
                "keras_hub.tokenizers.ByteTokenizer.vocabulary_size",
                "keras_hub.tokenizers.ByteTokenizer.token_to_id",
                "keras_hub.tokenizers.ByteTokenizer.id_to_token",
            ],
        },
        {
            "path": "unicode_codepoint_tokenizer",
            "title": "UnicodeCodepointTokenizer",
            "generate": [
                "keras_hub.tokenizers.UnicodeCodepointTokenizer",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.tokenize",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.detokenize",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.get_vocabulary",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.vocabulary_size",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.token_to_id",
                "keras_hub.tokenizers.UnicodeCodepointTokenizer.id_to_token",
            ],
        },
        {
            "path": "compute_word_piece_vocabulary",
            "title": "compute_word_piece_vocabulary function",
            "generate": ["keras_hub.tokenizers.compute_word_piece_vocabulary"],
        },
        {
            "path": "compute_sentence_piece_proto",
            "title": "compute_sentence_piece_proto function",
            "generate": ["keras_hub.tokenizers.compute_sentence_piece_proto"],
        },
    ],
}

PREPROCESSING_LAYERS_MASTER = {
    "path": "preprocessing_layers/",
    "title": "Preprocessing Layers",
    "toc": True,
    "children": [
        {
            "path": "audio_converter",
            "title": "AudioConverter layer",
            "generate": [
                "keras_hub.layers.AudioConverter",
                "keras_hub.layers.AudioConverter.from_preset",
            ],
        },
        {
            "path": "image_converter",
            "title": "ImageConverter layer",
            "generate": [
                "keras_hub.layers.ImageConverter",
                "keras_hub.layers.ImageConverter.from_preset",
            ],
        },
        {
            "path": "start_end_packer",
            "title": "StartEndPacker layer",
            "generate": ["keras_hub.layers.StartEndPacker"],
        },
        {
            "path": "multi_segment_packer",
            "title": "MultiSegmentPacker layer",
            "generate": ["keras_hub.layers.MultiSegmentPacker"],
        },
        {
            "path": "random_swap",
            "title": "RandomSwap layer",
            "generate": ["keras_hub.layers.RandomSwap"],
        },
        {
            "path": "random_deletion",
            "title": "RandomDeletion layer",
            "generate": ["keras_hub.layers.RandomDeletion"],
        },
        {
            "path": "masked_lm_mask_generator",
            "title": "MaskedLMMaskGenerator layer",
            "generate": ["keras_hub.layers.MaskedLMMaskGenerator"],
        },
    ],
}

MODELING_LAYERS_MASTER = {
    "path": "modeling_layers/",
    "title": "Modeling Layers",
    "toc": True,
    "children": [
        {
            "path": "transformer_encoder",
            "title": "TransformerEncoder layer",
            "generate": [
                "keras_hub.layers.TransformerEncoder",
                "keras_hub.layers.TransformerEncoder.call",
            ],
        },
        {
            "path": "transformer_decoder",
            "title": "TransformerDecoder layer",
            "generate": [
                "keras_hub.layers.TransformerDecoder",
                "keras_hub.layers.TransformerDecoder.call",
            ],
        },
        {
            "path": "fnet_encoder",
            "title": "FNetEncoder layer",
            "generate": ["keras_hub.layers.FNetEncoder"],
        },
        {
            "path": "position_embedding",
            "title": "PositionEmbedding layer",
            "generate": ["keras_hub.layers.PositionEmbedding"],
        },
        {
            "path": "rotary_embedding",
            "title": "RotaryEmbedding layer",
            "generate": ["keras_hub.layers.RotaryEmbedding"],
        },
        {
            "path": "sine_position_encoding",
            "title": "SinePositionEncoding layer",
            "generate": ["keras_hub.layers.SinePositionEncoding"],
        },
        {
            "path": "reversible_embedding",
            "title": "ReversibleEmbedding layer",
            "generate": ["keras_hub.layers.ReversibleEmbedding"],
        },
        {
            "path": "token_and_position_embedding",
            "title": "TokenAndPositionEmbedding layer",
            "generate": ["keras_hub.layers.TokenAndPositionEmbedding"],
        },
        {
            "path": "alibi_bias",
            "title": "AlibiBias layer",
            "generate": ["keras_hub.layers.AlibiBias"],
        },
        {
            "path": "masked_lm_head",
            "title": "MaskedLMHead layer",
            "generate": ["keras_hub.layers.MaskedLMHead"],
        },
        {
            "path": "cached_multi_head_attention",
            "title": "CachedMultiHeadAttention layer",
            "generate": ["keras_hub.layers.CachedMultiHeadAttention"],
        },
    ],
}


METRICS_MASTER = {
    "path": "metrics/",
    "title": "Metrics",
    "toc": True,
    "children": [
        {
            "path": "perplexity",
            "title": "Perplexity metric",
            "generate": ["keras_hub.metrics.Perplexity"],
        },
    ],
}

HUB_API_MASTER = {
    "path": "api/",
    "title": "API documentation",
    "toc": True,
    "children": [
        BASE_CLASSES,
        MODELS_MASTER,
        TOKENIZERS_MASTER,
        PREPROCESSING_LAYERS_MASTER,
        MODELING_LAYERS_MASTER,
        SAMPLERS_MASTER,
        METRICS_MASTER,
    ],
}

HUB_GUIDES_MASTER = {
    "path": "guides/",
    "title": "Developer guides",
    "toc": True,
    "children": [
        {
            "path": "upload",
            "title": "Uploading Models",
        },
        {
            "path": "stable_diffusion_3_in_keras_hub",
            "title": "Stable Diffusion 3",
        },
        {
            "path": "segment_anything_in_keras_hub",
            "title": "Segment Anything",
        },
        {
            "path": "classification_with_keras_hub",
            "title": "Image Classification",
        },
        {
            "path": "semantic_segmentation_deeplab_v3",
            "title": "Semantic Segmentation",
        },
        {
            "path": "transformer_pretraining",
            "title": "Pretraining a Transformer from scratch",
        },
        {
            "path": "hugging_face_keras_integration",
            "title": "Loading Hugging Face Transformers Checkpoints",
        },
        {
            "path": "function_calling_with_keras_hub",
            "title": "Function Calling with KerasHub models",
        },
        {
            "path": "rag_pipeline_with_keras_hub",
            "title": "RAG Pipeline with KerasHub",
        },
    ],
}

HUB_MASTER = {
    "path": "keras_hub/",
    "title": "KerasHub: Pretrained Models",
    "children": [
        {
            "path": "getting_started",
            "title": "Getting started",
        },
        HUB_GUIDES_MASTER,
        HUB_API_MASTER,
        {
            "path": "presets/",
            "title": "Pretrained models list",
        },
    ],
}
