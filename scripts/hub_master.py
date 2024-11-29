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
                    "title": "BertTokenizer",
                    "generate": [
                        "keras_hub.tokenizers.BertTokenizer",
                        "keras_hub.tokenizers.BertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bart_backbone",
                    "title": "BertBackbone model",
                    "generate": [
                        "keras_hub.models.BertBackbone",
                        "keras_hub.models.BertBackbone.from_preset",
                        "keras_hub.models.BertBackbone.token_embedding",
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
