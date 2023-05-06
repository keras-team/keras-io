MODELS_MASTER = {
    "path": "models/",
    "title": "Models",
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
                        "keras_nlp.models.AlbertTokenizer",
                        "keras_nlp.models.AlbertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "albert_preprocessor",
                    "title": "AlbertPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.AlbertPreprocessor",
                        "keras_nlp.models.AlbertPreprocessor.from_preset",
                        "keras_nlp.models.AlbertPreprocessor.tokenizer",
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
                    "path": "albert_classifier",
                    "title": "AlbertClassifier model",
                    "generate": [
                        "keras_nlp.models.AlbertClassifier",
                        "keras_nlp.models.AlbertClassifier.from_preset",
                        "keras_nlp.models.AlbertClassifier.backbone",
                        "keras_nlp.models.AlbertClassifier.preprocessor",
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
            "path": "bert/",
            "title": "Bert",
            "toc": True,
            "children": [
                {
                    "path": "bert_tokenizer",
                    "title": "BertTokenizer",
                    "generate": [
                        "keras_nlp.models.BertTokenizer",
                        "keras_nlp.models.BertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "bert_preprocessor",
                    "title": "BertPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.BertPreprocessor",
                        "keras_nlp.models.BertPreprocessor.from_preset",
                        "keras_nlp.models.BertPreprocessor.tokenizer",
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
                    "path": "bert_classifier",
                    "title": "BertClassifier model",
                    "generate": [
                        "keras_nlp.models.BertClassifier",
                        "keras_nlp.models.BertClassifier.from_preset",
                        "keras_nlp.models.BertClassifier.backbone",
                        "keras_nlp.models.BertClassifier.preprocessor",
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
            "path": "deberta_v3/",
            "title": "DebertaV3",
            "toc": True,
            "children": [
                {
                    "path": "deberta_v3_tokenizer",
                    "title": "DebertaV3Tokenizer",
                    "generate": [
                        "keras_nlp.models.DebertaV3Tokenizer",
                        "keras_nlp.models.DebertaV3Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "deberta_v3_preprocessor",
                    "title": "DebertaV3Preprocessor layer",
                    "generate": [
                        "keras_nlp.models.DebertaV3Preprocessor",
                        "keras_nlp.models.DebertaV3Preprocessor.from_preset",
                        "keras_nlp.models.DebertaV3Preprocessor.tokenizer",
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
                    "path": "deberta_v3_classifier",
                    "title": "DebertaV3Classifier model",
                    "generate": [
                        "keras_nlp.models.DebertaV3Classifier",
                        "keras_nlp.models.DebertaV3Classifier.from_preset",
                        "keras_nlp.models.DebertaV3Classifier.backbone",
                        "keras_nlp.models.DebertaV3Classifier.preprocessor",
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
                        "keras_nlp.models.DistilBertTokenizer",
                        "keras_nlp.models.DistilBertTokenizer.from_preset",
                    ],
                },
                {
                    "path": "distil_bert_preprocessor",
                    "title": "DistilBertPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.DistilBertPreprocessor",
                        "keras_nlp.models.DistilBertPreprocessor.from_preset",
                        "keras_nlp.models.DistilBertPreprocessor.tokenizer",
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
                    "path": "distil_bert_classifier",
                    "title": "DistilBertClassifier model",
                    "generate": [
                        "keras_nlp.models.DistilBertClassifier",
                        "keras_nlp.models.DistilBertClassifier.from_preset",
                        "keras_nlp.models.DistilBertClassifier.backbone",
                        "keras_nlp.models.DistilBertClassifier.preprocessor",
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
            "path": "gpt2/",
            "title": "GPT2",
            "toc": True,
            "children": [
                {
                    "path": "gpt2_tokenizer",
                    "title": "GPT2Tokenizer",
                    "generate": [
                        "keras_nlp.models.GPT2Tokenizer",
                        "keras_nlp.models.GPT2Tokenizer.from_preset",
                    ],
                },
                {
                    "path": "gpt2_preprocessor",
                    "title": "GPT2Preprocessor layer",
                    "generate": [
                        "keras_nlp.models.GPT2Preprocessor",
                        "keras_nlp.models.GPT2Preprocessor.from_preset",
                        "keras_nlp.models.GPT2Preprocessor.tokenizer",
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
            "path": "f_net/",
            "title": "FNet",
            "toc": True,
            "children": [
                {
                    "path": "f_net_tokenizer",
                    "title": "FNetTokenizer",
                    "generate": [
                        "keras_nlp.models.FNetTokenizer",
                        "keras_nlp.models.FNetTokenizer.from_preset",
                    ],
                },
                {
                    "path": "f_net_preprocessor",
                    "title": "FNetPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.FNetPreprocessor",
                        "keras_nlp.models.FNetPreprocessor.from_preset",
                        "keras_nlp.models.FNetPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "f_net3_backbone",
                    "title": "FNetBackbone model",
                    "generate": [
                        "keras_nlp.models.FNetBackbone",
                        "keras_nlp.models.FNetBackbone.from_preset",
                        "keras_nlp.models.FNetBackbone.token_embedding",
                    ],
                },
                {
                    "path": "f_net_classifier",
                    "title": "FNetClassifier model",
                    "generate": [
                        "keras_nlp.models.FNetClassifier",
                        "keras_nlp.models.FNetClassifier.from_preset",
                        "keras_nlp.models.FNetClassifier.backbone",
                        "keras_nlp.models.FNetClassifier.preprocessor",
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
            "path": "opt/",
            "title": "OPT",
            "toc": True,
            "children": [
                {
                    "path": "opt_tokenizer",
                    "title": "OPTTokenizer",
                    "generate": [
                        "keras_nlp.models.OPTTokenizer",
                        "keras_nlp.models.OPTTokenizer.from_preset",
                    ],
                },
                {
                    "path": "opt_preprocessor",
                    "title": "OPTPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.OPTPreprocessor",
                        "keras_nlp.models.OPTPreprocessor.from_preset",
                        "keras_nlp.models.OPTPreprocessor.tokenizer",
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
            "path": "roberta/",
            "title": "Roberta",
            "toc": True,
            "children": [
                {
                    "path": "roberta_tokenizer",
                    "title": "RobertaTokenizer",
                    "generate": [
                        "keras_nlp.models.RobertaTokenizer",
                        "keras_nlp.models.RobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "roberta_preprocessor",
                    "title": "RobertaPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.RobertaPreprocessor",
                        "keras_nlp.models.RobertaPreprocessor.from_preset",
                        "keras_nlp.models.RobertaPreprocessor.tokenizer",
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
                    "path": "roberta_classifier",
                    "title": "RobertaClassifier model",
                    "generate": [
                        "keras_nlp.models.RobertaClassifier",
                        "keras_nlp.models.RobertaClassifier.from_preset",
                        "keras_nlp.models.RobertaClassifier.backbone",
                        "keras_nlp.models.RobertaClassifier.preprocessor",
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
                        "keras_nlp.models.XLMRobertaTokenizer",
                        "keras_nlp.models.XLMRobertaTokenizer.from_preset",
                    ],
                },
                {
                    "path": "xlm_roberta_preprocessor",
                    "title": "XLMRobertaPreprocessor layer",
                    "generate": [
                        "keras_nlp.models.XLMRobertaPreprocessor",
                        "keras_nlp.models.XLMRobertaPreprocessor.from_preset",
                        "keras_nlp.models.XLMRobertaPreprocessor.tokenizer",
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
                    "path": "xlm_roberta_classifier",
                    "title": "XLMRobertaClassifier model",
                    "generate": [
                        "keras_nlp.models.XLMRobertaClassifier",
                        "keras_nlp.models.XLMRobertaClassifier.from_preset",
                        "keras_nlp.models.XLMRobertaClassifier.backbone",
                        "keras_nlp.models.XLMRobertaClassifier.preprocessor",
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
            "title": "Tokenizer base class",
            "generate": [
                "keras_nlp.tokenizers.Tokenizer",
                "keras_nlp.tokenizers.Tokenizer.tokenize",
                "keras_nlp.tokenizers.Tokenizer.detokenize",
                "keras_nlp.tokenizers.Tokenizer.get_vocabulary",
                "keras_nlp.tokenizers.Tokenizer.vocabulary_size",
                "keras_nlp.tokenizers.Tokenizer.token_to_id",
                "keras_nlp.tokenizers.Tokenizer.id_to_token",
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
            "path": "sine_position_encoding",
            "title": "SinePositionEncoding layer",
            "generate": ["keras_nlp.layers.SinePositionEncoding"],
        },
        {
            "path": "token_and_position_embedding",
            "title": "TokenAndPositionEmbedding layer",
            "generate": ["keras_nlp.layers.TokenAndPositionEmbedding"],
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
        {
            "path": "rouge_l",
            "title": "RougeL metric",
            "generate": ["keras_nlp.metrics.RougeL"],
        },
        {
            "path": "rouge_n",
            "title": "RougeN metric",
            "generate": ["keras_nlp.metrics.RougeN"],
        },
        {
            "path": "bleu",
            "title": "Bleu metric",
            "generate": ["keras_nlp.metrics.Bleu"],
        },
        {
            "path": "edit_distance",
            "title": "EditDistance metric",
            "generate": ["keras_nlp.metrics.EditDistance"],
        },
    ],
}

NLP_API_MASTER = {
    "path": "keras_nlp/",
    "title": "KerasNLP",
    "toc": True,
    "children": [
        MODELS_MASTER,
        TOKENIZERS_MASTER,
        PREPROCESSING_LAYERS_MASTER,
        MODELING_LAYERS_MASTER,
        SAMPLERS_MASTER,
        METRICS_MASTER,
    ],
}
