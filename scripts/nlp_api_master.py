BASE_CLASSES = {
    "path": "base_classes/",
    "title": "Models API",
    "toc": True,
    "children": [
        {
            "path": "backbone",
            "title": "Backbone",
            "generate": [
                "keras_nlp.models.Backbone",
                "keras_nlp.models.Backbone.from_preset",
                "keras_nlp.models.Backbone.token_embedding",
                "keras_nlp.models.Backbone.enable_lora",
                "keras_nlp.models.Backbone.save_lora_weights",
                "keras_nlp.models.Backbone.load_lora_weights",
                "keras_nlp.models.Backbone.save_to_preset",
            ],
        },
        {
            "path": "task",
            "title": "Task",
            "generate": [
                "keras_nlp.models.Task",
                "keras_nlp.models.Task.from_preset",
                "keras_nlp.models.Task.save_to_preset",
                "keras_nlp.models.Task.preprocessor",
                "keras_nlp.models.Task.backbone",
            ],
        },
        {
            "path": "preprocessor",
            "title": "Preprocessor",
            "generate": [
                "keras_nlp.models.Preprocessor",
                "keras_nlp.models.Preprocessor.from_preset",
                "keras_nlp.models.Preprocessor.save_to_preset",
                "keras_nlp.models.Preprocessor.tokenizer",
            ],
        },
        {
            "path": "causal_lm",
            "title": "CausalLM",
            "generate": [
                "keras_nlp.models.CausalLM",
                "keras_nlp.models.CausalLM.from_preset",
                "keras_nlp.models.CausalLM.compile",
                "keras_nlp.models.CausalLM.generate",
                "keras_nlp.models.CausalLM.save_to_preset",
                "keras_nlp.models.CausalLM.preprocessor",
                "keras_nlp.models.CausalLM.backbone",
            ],
        },
        {
            "path": "causal_lm_preprocessor",
            "title": "CausalLMPreprocessor",
            "generate": [
                "keras_nlp.models.CausalLMPreprocessor",
                "keras_nlp.models.CausalLMPreprocessor.from_preset",
                "keras_nlp.models.CausalLMPreprocessor.save_to_preset",
                "keras_nlp.models.CausalLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "seq_2_seq_lm",
            "title": "Seq2SeqLM",
            "generate": [
                "keras_nlp.models.Seq2SeqLM",
                "keras_nlp.models.Seq2SeqLM.from_preset",
                "keras_nlp.models.Seq2SeqLM.compile",
                "keras_nlp.models.Seq2SeqLM.generate",
                "keras_nlp.models.Seq2SeqLM.save_to_preset",
                "keras_nlp.models.Seq2SeqLM.preprocessor",
                "keras_nlp.models.Seq2SeqLM.backbone",
            ],
        },
        {
            "path": "seq_2_seq_lm_preprocessor",
            "title": "Seq2SeqLMPreprocessor",
            "generate": [
                "keras_nlp.models.Seq2SeqLMPreprocessor",
                "keras_nlp.models.Seq2SeqLMPreprocessor.from_preset",
                "keras_nlp.models.Seq2SeqLMPreprocessor.save_to_preset",
                "keras_nlp.models.Seq2SeqLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "text_classifier",
            "title": "TextClassifier",
            "generate": [
                "keras_nlp.models.TextClassifier",
                "keras_nlp.models.TextClassifier.from_preset",
                "keras_nlp.models.TextClassifier.compile",
                "keras_nlp.models.TextClassifier.save_to_preset",
                "keras_nlp.models.TextClassifier.preprocessor",
                "keras_nlp.models.TextClassifier.backbone",
            ],
        },
        {
            "path": "text_classifier_preprocessor",
            "title": "TextClassifierPreprocessor",
            "generate": [
                "keras_nlp.models.TextClassifierPreprocessor",
                "keras_nlp.models.TextClassifierPreprocessor.from_preset",
                "keras_nlp.models.TextClassifierPreprocessor.save_to_preset",
                "keras_nlp.models.TextClassifierPreprocessor.tokenizer",
            ],
        },
        {
            "path": "masked_lm",
            "title": "MaskedLM",
            "generate": [
                "keras_nlp.models.MaskedLM",
                "keras_nlp.models.MaskedLM.from_preset",
                "keras_nlp.models.MaskedLM.compile",
                "keras_nlp.models.MaskedLM.save_to_preset",
                "keras_nlp.models.MaskedLM.preprocessor",
                "keras_nlp.models.MaskedLM.backbone",
            ],
        },
        {
            "path": "masked_lm_preprocessor",
            "title": "MaskedLMPreprocessor",
            "generate": [
                "keras_nlp.models.MaskedLMPreprocessor",
                "keras_nlp.models.MaskedLMPreprocessor.from_preset",
                "keras_nlp.models.MaskedLMPreprocessor.save_to_preset",
                "keras_nlp.models.MaskedLMPreprocessor.tokenizer",
            ],
        },
        {
            "path": "upload_preset",
            "title": "upload_preset",
            "generate": ["keras_nlp.upload_preset"],
        },
    ],
}

MODELS_MASTER = {
    "path": "models/",
    "title": "Pretrained Models",
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
                        "keras_nlp.tokenizers.AlbertTokenizer",
                        "keras_nlp.tokenizers.AlbertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "albert_backbone",
                    "title": "AlbertBackbone model",
                    "generate": [
                        "keras_nlp.models.AlbertBackbone",
                        "keras_nlp.models.AlbertBackbone.from_preset",
                        "keras_nlp.models.AlbertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "albert_text_classifier",
                    "title": "AlbertTextClassifier model",
                    "generate": [
                        "keras_nlp.models.AlbertTextClassifier",
                        "keras_nlp.models.AlbertTextClassifier.from_preset",
                        "keras_nlp.models.AlbertTextClassifier.backbone",
                        "keras_nlp.models.AlbertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "albert_text_classifier_preprocessor",
                    "title": "AlbertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.AlbertTextClassifierPreprocessor",
                        "keras_nlp.models.AlbertTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.AlbertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "albert_masked_lm",
                    "title": "AlbertMaskedLM model",
                    "generate": [
                        "keras_nlp.models.AlbertMaskedLM",
                        "keras_nlp.models.AlbertMaskedLM.from_preset",
                        "keras_nlp.models.AlbertMaskedLM.backbone",
                        "keras_nlp.models.AlbertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "albert_masked_lm_preprocessor",
                    "title": "AlbertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.AlbertMaskedLMPreprocessor",
                        "keras_nlp.models.AlbertMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.AlbertMaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.BertTokenizer",
                        "keras_nlp.tokenizers.BertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bart_backbone",
                    "title": "BertBackbone model",
                    "generate": [
                        "keras_nlp.models.BertBackbone",
                        "keras_nlp.models.BertBackbone.from_preset",
                        "keras_nlp.models.BertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "bart_seq_2_seq_lm",
                    "title": "BartSeq2SeqLM model",
                    "generate": [
                        "keras_nlp.models.BartSeq2SeqLM",
                        "keras_nlp.models.BartSeq2SeqLM.from_preset",
                        "keras_nlp.models.BartSeq2SeqLM.generate",
                        "keras_nlp.models.BartSeq2SeqLM.backbone",
                        "keras_nlp.models.BartSeq2SeqLM.preprocessor",
                    ],
                },
                {
                    "path": "bart_seq_2_seq_lm_preprocessor",
                    "title": "BartSeq2SeqLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.BartSeq2SeqLMPreprocessor",
                        "keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset",
                        "keras_nlp.models.BartSeq2SeqLMPreprocessor.generate_preprocess",
                        "keras_nlp.models.BartSeq2SeqLMPreprocessor.generate_postprocess",
                        "keras_nlp.models.BartSeq2SeqLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.BertTokenizer",
                        "keras_nlp.tokenizers.BertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bert_backbone",
                    "title": "BertBackbone model",
                    "generate": [
                        "keras_nlp.models.BertBackbone",
                        "keras_nlp.models.BertBackbone.from_preset",
                        "keras_nlp.models.BertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "bert_text_classifier",
                    "title": "BertTextClassifier model",
                    "generate": [
                        "keras_nlp.models.BertTextClassifier",
                        "keras_nlp.models.BertTextClassifier.from_preset",
                        "keras_nlp.models.BertTextClassifier.backbone",
                        "keras_nlp.models.BertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "bert_text_classifier_preprocessor",
                    "title": "BertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.BertTextClassifierPreprocessor",
                        "keras_nlp.models.BertTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.BertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "bert_masked_lm",
                    "title": "BertMaskedLM model",
                    "generate": [
                        "keras_nlp.models.BertMaskedLM",
                        "keras_nlp.models.BertMaskedLM.from_preset",
                        "keras_nlp.models.BertMaskedLM.backbone",
                        "keras_nlp.models.BertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "bert_masked_lm_preprocessor",
                    "title": "BertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.BertMaskedLMPreprocessor",
                        "keras_nlp.models.BertMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.BertMaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.BloomTokenizer",
                        "keras_nlp.tokenizers.BloomTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bloom_backbone",
                    "title": "BloomBackbone model",
                    "generate": [
                        "keras_nlp.models.BloomBackbone",
                        "keras_nlp.models.BloomBackbone.from_preset",
                        "keras_nlp.models.BloomBackbone.token_embedding",
                        "keras_nlp.models.BloomBackbone.enable_lora",
                    ],
                },
                {
                    "path": "bloom_causal_lm",
                    "title": "BloomCausalLM model",
                    "generate": [
                        "keras_nlp.models.BloomCausalLM",
                        "keras_nlp.models.BloomCausalLM.from_preset",
                        "keras_nlp.models.BloomCausalLM.generate",
                        "keras_nlp.models.BloomCausalLM.backbone",
                        "keras_nlp.models.BloomCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "bloom_causal_lm_preprocessor",
                    "title": "BloomCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.BloomCausalLMPreprocessor",
                        "keras_nlp.models.BloomCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.BloomCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.DebertaV3Tokenizer",
                        "keras_nlp.tokenizers.DebertaV3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "deberta_v3_backbone",
                    "title": "DebertaV3Backbone model",
                    "generate": [
                        "keras_nlp.models.DebertaV3Backbone",
                        "keras_nlp.models.DebertaV3Backbone.from_preset",
                        "keras_nlp.models.DebertaV3Backbone.token_embedding",
                    ],
                },
                {
                    "path": "deberta_v3_text_classifier",
                    "title": "DebertaV3TextClassifier model",
                    "generate": [
                        "keras_nlp.models.DebertaV3TextClassifier",
                        "keras_nlp.models.DebertaV3TextClassifier.from_preset",
                        "keras_nlp.models.DebertaV3TextClassifier.backbone",
                        "keras_nlp.models.DebertaV3TextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "deberta_v3_text_classifier_preprocessor",
                    "title": "DebertaV3TextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.DebertaV3TextClassifierPreprocessor",
                        "keras_nlp.models.DebertaV3TextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.DebertaV3TextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "deberta_v3_masked_lm",
                    "title": "DebertaV3MaskedLM model",
                    "generate": [
                        "keras_nlp.models.DebertaV3MaskedLM",
                        "keras_nlp.models.DebertaV3MaskedLM.from_preset",
                        "keras_nlp.models.DebertaV3MaskedLM.backbone",
                        "keras_nlp.models.DebertaV3MaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "deberta_v3_masked_lm_preprocessor",
                    "title": "DebertaV3MaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.DebertaV3MaskedLMPreprocessor",
                        "keras_nlp.models.DebertaV3MaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.DebertaV3MaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.DistilBertTokenizer",
                        "keras_nlp.tokenizers.DistilBertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "distil_bert_backbone",
                    "title": "DistilBertBackbone model",
                    "generate": [
                        "keras_nlp.models.DistilBertBackbone",
                        "keras_nlp.models.DistilBertBackbone.from_preset",
                        "keras_nlp.models.DistilBertBackbone.token_embedding",
                    ],
                },
                {
                    "path": "distil_bert_text_classifier",
                    "title": "DistilBertTextClassifier model",
                    "generate": [
                        "keras_nlp.models.DistilBertTextClassifier",
                        "keras_nlp.models.DistilBertTextClassifier.from_preset",
                        "keras_nlp.models.DistilBertTextClassifier.backbone",
                        "keras_nlp.models.DistilBertTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "distil_bert_text_classifier_preprocessor",
                    "title": "DistilBertTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.DistilBertTextClassifierPreprocessor",
                        "keras_nlp.models.DistilBertTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.DistilBertTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "distil_bert_masked_lm",
                    "title": "DistilBertMaskedLM model",
                    "generate": [
                        "keras_nlp.models.DistilBertMaskedLM",
                        "keras_nlp.models.DistilBertMaskedLM.from_preset",
                        "keras_nlp.models.DistilBertMaskedLM.backbone",
                        "keras_nlp.models.DistilBertMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "distil_bert_masked_lm_preprocessor",
                    "title": "DistilBertMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.DistilBertMaskedLMPreprocessor",
                        "keras_nlp.models.DistilBertMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.DistilBertMaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.GemmaTokenizer",
                        "keras_nlp.tokenizers.GemmaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "gemma_backbone",
                    "title": "GemmaBackbone model",
                    "generate": [
                        "keras_nlp.models.GemmaBackbone",
                        "keras_nlp.models.GemmaBackbone.from_preset",
                        "keras_nlp.models.GemmaBackbone.token_embedding",
                        "keras_nlp.models.GemmaBackbone.enable_lora",
                        "keras_nlp.models.GemmaBackbone.get_layout_map",
                    ],
                },
                {
                    "path": "gemma_causal_lm",
                    "title": "GemmaCausalLM model",
                    "generate": [
                        "keras_nlp.models.GemmaCausalLM",
                        "keras_nlp.models.GemmaCausalLM.from_preset",
                        "keras_nlp.models.GemmaCausalLM.generate",
                        "keras_nlp.models.GemmaCausalLM.backbone",
                        "keras_nlp.models.GemmaCausalLM.preprocessor",
                        "keras_nlp.models.GemmaCausalLM.score",
                    ],
                },
                {
                    "path": "gemma_causal_lm_preprocessor",
                    "title": "GemmaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.GemmaCausalLMPreprocessor",
                        "keras_nlp.models.GemmaCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.GemmaCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.ElectraTokenizer",
                        "keras_nlp.tokenizers.ElectraTokenizer.from_preset",
                    ],
                },
                {
                    "path": "electra_backbone",
                    "title": "ElectraBackbone model",
                    "generate": [
                        "keras_nlp.models.ElectraBackbone",
                        "keras_nlp.models.ElectraBackbone.from_preset",
                        "keras_nlp.models.ElectraBackbone.token_embedding",
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
                        "keras_nlp.tokenizers.FalconTokenizer",
                        "keras_nlp.tokenizers.FalconTokenizer.from_preset",
                    ],
                },
                {
                    "path": "falcon_backbone",
                    "title": "FalconBackbone model",
                    "generate": [
                        "keras_nlp.models.FalconBackbone",
                        "keras_nlp.models.FalconBackbone.from_preset",
                        "keras_nlp.models.FalconBackbone.token_embedding",
                    ],
                },
                {
                    "path": "falcon_causal_lm",
                    "title": "FalconCausalLM model",
                    "generate": [
                        "keras_nlp.models.FalconCausalLM",
                        "keras_nlp.models.FalconCausalLM.from_preset",
                        "keras_nlp.models.FalconCausalLM.generate",
                        "keras_nlp.models.FalconCausalLM.backbone",
                        "keras_nlp.models.FalconCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "falcon_causal_lm_preprocessor",
                    "title": "FalconCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.FalconCausalLMPreprocessor",
                        "keras_nlp.models.FalconCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.FalconCausalLMPreprocessor.generate_preprocess",
                        "keras_nlp.models.FalconCausalLMPreprocessor.generate_postprocess",
                        "keras_nlp.models.FalconCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.FNetTokenizer",
                        "keras_nlp.tokenizers.FNetTokenizer.from_preset",
                    ],
                },
                {
                    "path": "f_net_backbone",
                    "title": "FNetBackbone model",
                    "generate": [
                        "keras_nlp.models.FNetBackbone",
                        "keras_nlp.models.FNetBackbone.from_preset",
                        "keras_nlp.models.FNetBackbone.token_embedding",
                    ],
                },
                {
                    "path": "f_net_text_classifier",
                    "title": "FNetTextClassifier model",
                    "generate": [
                        "keras_nlp.models.FNetTextClassifier",
                        "keras_nlp.models.FNetTextClassifier.from_preset",
                        "keras_nlp.models.FNetTextClassifier.backbone",
                        "keras_nlp.models.FNetTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "f_net_text_classifier_preprocessor",
                    "title": "FNetTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.FNetTextClassifierPreprocessor",
                        "keras_nlp.models.FNetTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.FNetTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "f_net_masked_lm",
                    "title": "FNetMaskedLM model",
                    "generate": [
                        "keras_nlp.models.FNetMaskedLM",
                        "keras_nlp.models.FNetMaskedLM.from_preset",
                        "keras_nlp.models.FNetMaskedLM.backbone",
                        "keras_nlp.models.FNetMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "f_net_masked_lm_preprocessor",
                    "title": "FNetMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.FNetMaskedLMPreprocessor",
                        "keras_nlp.models.FNetMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.FNetMaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.GPT2Tokenizer",
                        "keras_nlp.tokenizers.GPT2Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "gpt2_backbone",
                    "title": "GPT2Backbone model",
                    "generate": [
                        "keras_nlp.models.GPT2Backbone",
                        "keras_nlp.models.GPT2Backbone.from_preset",
                        "keras_nlp.models.GPT2Backbone.token_embedding",
                    ],
                },
                {
                    "path": "gpt2_causal_lm",
                    "title": "GPT2CausalLM model",
                    "generate": [
                        "keras_nlp.models.GPT2CausalLM",
                        "keras_nlp.models.GPT2CausalLM.from_preset",
                        "keras_nlp.models.GPT2CausalLM.generate",
                        "keras_nlp.models.GPT2CausalLM.backbone",
                        "keras_nlp.models.GPT2CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "gpt2_causal_lm_preprocessor",
                    "title": "GPT2CausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.GPT2CausalLMPreprocessor",
                        "keras_nlp.models.GPT2CausalLMPreprocessor.from_preset",
                        "keras_nlp.models.GPT2CausalLMPreprocessor.generate_preprocess",
                        "keras_nlp.models.GPT2CausalLMPreprocessor.generate_postprocess",
                        "keras_nlp.models.GPT2CausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.LlamaTokenizer",
                        "keras_nlp.tokenizers.LlamaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "llama_backbone",
                    "title": "LlamaBackbone model",
                    "generate": [
                        "keras_nlp.models.LlamaBackbone",
                        "keras_nlp.models.LlamaBackbone.from_preset",
                        "keras_nlp.models.LlamaBackbone.token_embedding",
                        "keras_nlp.models.LlamaBackbone.enable_lora",
                    ],
                },
                {
                    "path": "llama_causal_lm",
                    "title": "LlamaCausalLM model",
                    "generate": [
                        "keras_nlp.models.LlamaCausalLM",
                        "keras_nlp.models.LlamaCausalLM.from_preset",
                        "keras_nlp.models.LlamaCausalLM.generate",
                        "keras_nlp.models.LlamaCausalLM.backbone",
                        "keras_nlp.models.LlamaCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "llama_causal_lm_preprocessor",
                    "title": "LlamaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.LlamaCausalLMPreprocessor",
                        "keras_nlp.models.LlamaCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.LlamaCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.Llama3Tokenizer",
                        "keras_nlp.tokenizers.Llama3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "llama3_backbone",
                    "title": "Llama3Backbone model",
                    "generate": [
                        "keras_nlp.models.Llama3Backbone",
                        "keras_nlp.models.Llama3Backbone.from_preset",
                        "keras_nlp.models.Llama3Backbone.token_embedding",
                        "keras_nlp.models.Llama3Backbone.enable_lora",
                    ],
                },
                {
                    "path": "llama3_causal_lm",
                    "title": "Llama3CausalLM model",
                    "generate": [
                        "keras_nlp.models.Llama3CausalLM",
                        "keras_nlp.models.Llama3CausalLM.from_preset",
                        "keras_nlp.models.Llama3CausalLM.generate",
                        "keras_nlp.models.Llama3CausalLM.backbone",
                        "keras_nlp.models.Llama3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "llama3_causal_lm_preprocessor",
                    "title": "Llama3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.Llama3CausalLMPreprocessor",
                        "keras_nlp.models.Llama3CausalLMPreprocessor.from_preset",
                        "keras_nlp.models.Llama3CausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.MistralTokenizer",
                        "keras_nlp.tokenizers.MistralTokenizer.from_preset",
                    ],
                },
                {
                    "path": "mistral_backbone",
                    "title": "MistralBackbone model",
                    "generate": [
                        "keras_nlp.models.MistralBackbone",
                        "keras_nlp.models.MistralBackbone.from_preset",
                        "keras_nlp.models.MistralBackbone.token_embedding",
                        "keras_nlp.models.MistralBackbone.enable_lora",
                    ],
                },
                {
                    "path": "mistral_causal_lm",
                    "title": "MistralCausalLM model",
                    "generate": [
                        "keras_nlp.models.MistralCausalLM",
                        "keras_nlp.models.MistralCausalLM.from_preset",
                        "keras_nlp.models.MistralCausalLM.generate",
                        "keras_nlp.models.MistralCausalLM.backbone",
                        "keras_nlp.models.MistralCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "mistral_causal_lm_preprocessor",
                    "title": "MistralCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.MistralCausalLMPreprocessor",
                        "keras_nlp.models.MistralCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.MistralCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.OPTTokenizer",
                        "keras_nlp.tokenizers.OPTTokenizer.from_preset",
                    ],
                },
                {
                    "path": "opt_backbone",
                    "title": "OPTBackbone model",
                    "generate": [
                        "keras_nlp.models.OPTBackbone",
                        "keras_nlp.models.OPTBackbone.from_preset",
                        "keras_nlp.models.OPTBackbone.token_embedding",
                    ],
                },
                {
                    "path": "opt_causal_lm",
                    "title": "OPTCausalLM model",
                    "generate": [
                        "keras_nlp.models.OPTCausalLM",
                        "keras_nlp.models.OPTCausalLM.from_preset",
                        "keras_nlp.models.OPTCausalLM.generate",
                        "keras_nlp.models.OPTCausalLM.backbone",
                        "keras_nlp.models.OPTCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "opt_causal_lm_preprocessor",
                    "title": "OPTCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.OPTCausalLMPreprocessor",
                        "keras_nlp.models.OPTCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.OPTCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.PaliGemmaTokenizer",
                        "keras_nlp.tokenizers.PaliGemmaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "pali_gemma_backbone",
                    "title": "PaliGemmaBackbone model",
                    "generate": [
                        "keras_nlp.models.PaliGemmaBackbone",
                        "keras_nlp.models.PaliGemmaBackbone.from_preset",
                        "keras_nlp.models.PaliGemmaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "pali_gemma_causal_lm",
                    "title": "PaliGemmaCausalLM model",
                    "generate": [
                        "keras_nlp.models.PaliGemmaCausalLM",
                        "keras_nlp.models.PaliGemmaCausalLM.from_preset",
                        "keras_nlp.models.PaliGemmaCausalLM.generate",
                        "keras_nlp.models.PaliGemmaCausalLM.backbone",
                        "keras_nlp.models.PaliGemmaCausalLM.preprocessor",
                    ],
                },
                {
                    "path": "pali_gemma_causal_lm_preprocessor",
                    "title": "PaliGemmaCausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.PaliGemmaCausalLMPreprocessor",
                        "keras_nlp.models.PaliGemmaCausalLMPreprocessor.from_preset",
                        "keras_nlp.models.PaliGemmaCausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.Phi3Tokenizer",
                        "keras_nlp.tokenizers.Phi3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "phi3_backbone",
                    "title": "Phi3Backbone model",
                    "generate": [
                        "keras_nlp.models.Phi3Backbone",
                        "keras_nlp.models.Phi3Backbone.from_preset",
                        "keras_nlp.models.Phi3Backbone.token_embedding",
                    ],
                },
                {
                    "path": "phi3_causal_lm",
                    "title": "Phi3CausalLM model",
                    "generate": [
                        "keras_nlp.models.Phi3CausalLM",
                        "keras_nlp.models.Phi3CausalLM.from_preset",
                        "keras_nlp.models.Phi3CausalLM.generate",
                        "keras_nlp.models.Phi3CausalLM.backbone",
                        "keras_nlp.models.Phi3CausalLM.preprocessor",
                    ],
                },
                {
                    "path": "phi3_causal_lm_preprocessor",
                    "title": "Phi3CausalLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.Phi3CausalLMPreprocessor",
                        "keras_nlp.models.Phi3CausalLMPreprocessor.from_preset",
                        "keras_nlp.models.Phi3CausalLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.RobertaTokenizer",
                        "keras_nlp.tokenizers.RobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "roberta_backbone",
                    "title": "RobertaBackbone model",
                    "generate": [
                        "keras_nlp.models.RobertaBackbone",
                        "keras_nlp.models.RobertaBackbone.from_preset",
                        "keras_nlp.models.RobertaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "roberta_text_classifier",
                    "title": "RobertaTextClassifier model",
                    "generate": [
                        "keras_nlp.models.RobertaTextClassifier",
                        "keras_nlp.models.RobertaTextClassifier.from_preset",
                        "keras_nlp.models.RobertaTextClassifier.backbone",
                        "keras_nlp.models.RobertaTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "roberta_text_classifier_preprocessor",
                    "title": "RobertaTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.RobertaTextClassifierPreprocessor",
                        "keras_nlp.models.RobertaTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.RobertaTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "roberta_masked_lm",
                    "title": "RobertaMaskedLM model",
                    "generate": [
                        "keras_nlp.models.RobertaMaskedLM",
                        "keras_nlp.models.RobertaMaskedLM.from_preset",
                        "keras_nlp.models.RobertaMaskedLM.backbone",
                        "keras_nlp.models.RobertaMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "roberta_masked_lm_preprocessor",
                    "title": "RobertaMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.RobertaMaskedLMPreprocessor",
                        "keras_nlp.models.RobertaMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.RobertaMaskedLMPreprocessor.tokenizer",
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
                        "keras_nlp.tokenizers.XLMRobertaTokenizer",
                        "keras_nlp.tokenizers.XLMRobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "xlm_roberta_backbone",
                    "title": "XLMRobertaBackbone model",
                    "generate": [
                        "keras_nlp.models.XLMRobertaBackbone",
                        "keras_nlp.models.XLMRobertaBackbone.from_preset",
                        "keras_nlp.models.XLMRobertaBackbone.token_embedding",
                    ],
                },
                {
                    "path": "xlm_roberta_text_classifier",
                    "title": "XLMRobertaTextClassifier model",
                    "generate": [
                        "keras_nlp.models.XLMRobertaTextClassifier",
                        "keras_nlp.models.XLMRobertaTextClassifier.from_preset",
                        "keras_nlp.models.XLMRobertaTextClassifier.backbone",
                        "keras_nlp.models.XLMRobertaTextClassifier.preprocessor",
                    ],
                },
                {
                    "path": "xlm_roberta_text_classifier_preprocessor",
                    "title": "XLMRobertaTextClassifierPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.XLMRobertaTextClassifierPreprocessor",
                        "keras_nlp.models.XLMRobertaTextClassifierPreprocessor.from_preset",
                        "keras_nlp.models.XLMRobertaTextClassifierPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "xlm_roberta_masked_lm",
                    "title": "XLMRobertaMaskedLM model",
                    "generate": [
                        "keras_nlp.models.XLMRobertaMaskedLM",
                        "keras_nlp.models.XLMRobertaMaskedLM.from_preset",
                        "keras_nlp.models.XLMRobertaMaskedLM.backbone",
                        "keras_nlp.models.XLMRobertaMaskedLM.preprocessor",
                    ],
                },
                {
                    "path": "xlm_roberta_masked_lm_preprocessor",
                    "title": "XLMRobertaMaskedLMPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.XLMRobertaMaskedLMPreprocessor",
                        "keras_nlp.models.XLMRobertaMaskedLMPreprocessor.from_preset",
                        "keras_nlp.models.XLMRobertaMaskedLMPreprocessor.tokenizer",
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
                "keras_nlp.samplers.Sampler",
                "keras_nlp.samplers.Sampler.get_next_token",
            ],
        },
        {
            "path": "beam_sampler",
            "title": "BeamSampler",
            "generate": ["keras_nlp.samplers.BeamSampler"],
        },
        {
            "path": "contrastive_sampler",
            "title": "ContrastiveSampler",
            "generate": ["keras_nlp.samplers.ContrastiveSampler"],
        },
        {
            "path": "greedy_sampler",
            "title": "GreedySampler",
            "generate": ["keras_nlp.samplers.GreedySampler"],
        },
        {
            "path": "random_sampler",
            "title": "RandomSampler",
            "generate": ["keras_nlp.samplers.RandomSampler"],
        },
        {
            "path": "top_k_sampler",
            "title": "TopKSampler",
            "generate": ["keras_nlp.samplers.TopKSampler"],
        },
        {
            "path": "top_p_sampler",
            "title": "TopPSampler",
            "generate": ["keras_nlp.samplers.TopPSampler"],
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
                "keras_nlp.tokenizers.Tokenizer",
                "keras_nlp.tokenizers.Tokenizer.from_preset",
                "keras_nlp.tokenizers.Tokenizer.save_to_preset",
            ],
        },
        {
            "path": "word_piece_tokenizer",
            "title": "WordPieceTokenizer",
            "generate": [
                "keras_nlp.tokenizers.WordPieceTokenizer",
                "keras_nlp.tokenizers.WordPieceTokenizer.tokenize",
                "keras_nlp.tokenizers.WordPieceTokenizer.detokenize",
                "keras_nlp.tokenizers.WordPieceTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.WordPieceTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.WordPieceTokenizer.token_to_id",
                "keras_nlp.tokenizers.WordPieceTokenizer.id_to_token",
            ],
        },
        {
            "path": "sentence_piece_tokenizer",
            "title": "SentencePieceTokenizer",
            "generate": [
                "keras_nlp.tokenizers.SentencePieceTokenizer",
                "keras_nlp.tokenizers.SentencePieceTokenizer.tokenize",
                "keras_nlp.tokenizers.SentencePieceTokenizer.detokenize",
                "keras_nlp.tokenizers.SentencePieceTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.SentencePieceTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.SentencePieceTokenizer.token_to_id",
                "keras_nlp.tokenizers.SentencePieceTokenizer.id_to_token",
            ],
        },
        {
            "path": "byte_pair_tokenizer",
            "title": "BytePairTokenizer",
            "generate": [
                "keras_nlp.tokenizers.BytePairTokenizer",
                "keras_nlp.tokenizers.BytePairTokenizer.tokenize",
                "keras_nlp.tokenizers.BytePairTokenizer.detokenize",
                "keras_nlp.tokenizers.BytePairTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.BytePairTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.BytePairTokenizer.token_to_id",
                "keras_nlp.tokenizers.BytePairTokenizer.id_to_token",
            ],
        },
        {
            "path": "byte_tokenizer",
            "title": "ByteTokenizer",
            "generate": [
                "keras_nlp.tokenizers.ByteTokenizer",
                "keras_nlp.tokenizers.ByteTokenizer.tokenize",
                "keras_nlp.tokenizers.ByteTokenizer.detokenize",
                "keras_nlp.tokenizers.ByteTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.ByteTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.ByteTokenizer.token_to_id",
                "keras_nlp.tokenizers.ByteTokenizer.id_to_token",
            ],
        },
        {
            "path": "unicode_codepoint_tokenizer",
            "title": "UnicodeCodepointTokenizer",
            "generate": [
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.tokenize",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.detokenize",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.token_to_id",
                "keras_nlp.tokenizers.UnicodeCodepointTokenizer.id_to_token",
            ],
        },
        {
            "path": "compute_word_piece_vocabulary",
            "title": "compute_word_piece_vocabulary function",
            "generate": ["keras_nlp.tokenizers.compute_word_piece_vocabulary"],
        },
        {
            "path": "compute_sentence_piece_proto",
            "title": "compute_sentence_piece_proto function",
            "generate": ["keras_nlp.tokenizers.compute_sentence_piece_proto"],
        },
    ],
}

PREPROCESSING_LAYERS_MASTER = {
    "path": "preprocessing_layers/",
    "title": "Preprocessing Layers",
    "toc": True,
    "children": [
        {
            "path": "start_end_packer",
            "title": "StartEndPacker layer",
            "generate": ["keras_nlp.layers.StartEndPacker"],
        },
        {
            "path": "multi_segment_packer",
            "title": "MultiSegmentPacker layer",
            "generate": ["keras_nlp.layers.MultiSegmentPacker"],
        },
        {
            "path": "random_swap",
            "title": "RandomSwap layer",
            "generate": ["keras_nlp.layers.RandomSwap"],
        },
        {
            "path": "random_deletion",
            "title": "RandomDeletion layer",
            "generate": ["keras_nlp.layers.RandomDeletion"],
        },
        {
            "path": "masked_lm_mask_generator",
            "title": "MaskedLMMaskGenerator layer",
            "generate": ["keras_nlp.layers.MaskedLMMaskGenerator"],
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
                "keras_nlp.layers.TransformerEncoder",
                "keras_nlp.layers.TransformerEncoder.call",
            ],
        },
        {
            "path": "transformer_decoder",
            "title": "TransformerDecoder layer",
            "generate": [
                "keras_nlp.layers.TransformerDecoder",
                "keras_nlp.layers.TransformerDecoder.call",
            ],
        },
        {
            "path": "fnet_encoder",
            "title": "FNetEncoder layer",
            "generate": ["keras_nlp.layers.FNetEncoder"],
        },
        {
            "path": "position_embedding",
            "title": "PositionEmbedding layer",
            "generate": ["keras_nlp.layers.PositionEmbedding"],
        },
        {
            "path": "rotary_embedding",
            "title": "RotaryEmbedding layer",
            "generate": ["keras_nlp.layers.RotaryEmbedding"],
        },
        {
            "path": "sine_position_encoding",
            "title": "SinePositionEncoding layer",
            "generate": ["keras_nlp.layers.SinePositionEncoding"],
        },
        {
            "path": "reversible_embedding",
            "title": "ReversibleEmbedding layer",
            "generate": ["keras_nlp.layers.ReversibleEmbedding"],
        },
        {
            "path": "token_and_position_embedding",
            "title": "TokenAndPositionEmbedding layer",
            "generate": ["keras_nlp.layers.TokenAndPositionEmbedding"],
        },
        {
            "path": "alibi_bias",
            "title": "AlibiBias layer",
            "generate": ["keras_nlp.layers.AlibiBias"],
        },
        {
            "path": "masked_lm_head",
            "title": "MaskedLMHead layer",
            "generate": ["keras_nlp.layers.MaskedLMHead"],
        },
        {
            "path": "cached_multi_head_attention",
            "title": "CachedMultiHeadAttention layer",
            "generate": ["keras_nlp.layers.CachedMultiHeadAttention"],
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
            "generate": ["keras_nlp.metrics.Perplexity"],
        },
    ],
}

NLP_API_MASTER = {
    "path": "keras_nlp/",
    "title": "KerasNLP",
    "toc": True,
    "children": [
        MODELS_MASTER,
        BASE_CLASSES,
        TOKENIZERS_MASTER,
        PREPROCESSING_LAYERS_MASTER,
        MODELING_LAYERS_MASTER,
        SAMPLERS_MASTER,
        METRICS_MASTER,
    ],
}
