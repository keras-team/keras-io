MODELS_MASTER = {
    "path": "models/",
    "title": "Models",
    "toc": True,
    "children": [
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
                    "title": "BertPreprocessor",
                    "generate": [
                        "keras_nlp.models.BertPreprocessor",
                        "keras_nlp.models.BertPreprocessor.from_preset",
                        "keras_nlp.models.BertPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "bert_backbone",
                    "title": "BertBackbone",
                    "generate": [
                        "keras_nlp.models.BertBackbone",
                        "keras_nlp.models.BertBackbone.from_preset",
                    ],
                },
                {
                    "path": "bert_classifier",
                    "title": "BertClassifier",
                    "generate": [
                        "keras_nlp.models.BertClassifier",
                        "keras_nlp.models.BertClassifier.from_preset",
                        "keras_nlp.models.BertClassifier.backbone",
                        "keras_nlp.models.BertClassifier.preprocessor",
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
                    "title": "DistilBertPreprocessor",
                    "generate": [
                        "keras_nlp.models.DistilBertPreprocessor",
                        "keras_nlp.models.DistilBertPreprocessor.from_preset",
                        "keras_nlp.models.DistilBertPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "distil_bert_backbone",
                    "title": "DistilBertBackbone",
                    "generate": [
                        "keras_nlp.models.DistilBertBackbone",
                        "keras_nlp.models.DistilBertBackbone.from_preset",
                    ],
                },
                {
                    "path": "distil_bert_classifier",
                    "title": "DistilBertClassifier",
                    "generate": [
                        "keras_nlp.models.DistilBertClassifier",
                        "keras_nlp.models.DistilBertClassifier.from_preset",
                        "keras_nlp.models.DistilBertClassifier.backbone",
                        "keras_nlp.models.DistilBertClassifier.preprocessor",
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
                    "title": "RobertaPreprocessor",
                    "generate": [
                        "keras_nlp.models.RobertaPreprocessor",
                        "keras_nlp.models.RobertaPreprocessor.from_preset",
                        "keras_nlp.models.RobertaPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "roberta_backbone",
                    "title": "RobertaBackbone",
                    "generate": [
                        "keras_nlp.models.RobertaBackbone",
                        "keras_nlp.models.RobertaBackbone.from_preset",
                    ],
                },
                {
                    "path": "roberta_classifier",
                    "title": "RobertaClassifier",
                    "generate": [
                        "keras_nlp.models.RobertaClassifier",
                        "keras_nlp.models.RobertaClassifier.from_preset",
                        "keras_nlp.models.RobertaClassifier.backbone",
                        "keras_nlp.models.RobertaClassifier.preprocessor",
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
                    "title": "XLMRobertaPreprocessor",
                    "generate": [
                        "keras_nlp.models.XLMRobertaPreprocessor",
                        "keras_nlp.models.XLMRobertaPreprocessor.from_preset",
                        "keras_nlp.models.XLMRobertaPreprocessor.tokenizer",
                    ],
                },
                {
                    "path": "xlm_roberta_backbone",
                    "title": "XLMRobertaBackbone",
                    "generate": [
                        "keras_nlp.models.XLMRobertaBackbone",
                        "keras_nlp.models.XLMRobertaBackbone.from_preset",
                    ],
                },
                {
                    "path": "xlm_roberta_classifier",
                    "title": "XLMRobertaClassifier",
                    "generate": [
                        "keras_nlp.models.XLMRobertaClassifier",
                        "keras_nlp.models.XLMRobertaClassifier.from_preset",
                        "keras_nlp.models.XLMRobertaClassifier.backbone",
                        "keras_nlp.models.XLMRobertaClassifier.preprocessor",
                    ],
                },
            ],
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
            "title": "compute_word_piece_vocabulary",
            "generate": ["keras_nlp.tokenizers.compute_word_piece_vocabulary"],
        },
        {
            "path": "compute_sentence_piece_proto",
            "title": "compute_sentence_piece_proto",
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
            "path": "mlm_mask_generator",
            "title": "MLMMaskGenerator",
            "generate": ["keras_nlp.layers.MLMMaskGenerator"],
        },
        {
            "path": "mlm_head",
            "title": "MLMHead",
            "generate": ["keras_nlp.layers.MLMHead"],
        },
        {
            "path": "start_end_packer",
            "title": "StartEndPacker",
            "generate": ["keras_nlp.layers.StartEndPacker"],
        },
        {
            "path": "multi_segment_packer",
            "title": "MultiSegmentPacker",
            "generate": ["keras_nlp.layers.MultiSegmentPacker"],
        },
        {
            "path": "random_swap",
            "title": "RandomSwap",
            "generate": ["keras_nlp.layers.RandomSwap"],
        },
        {
            "path": "random_deletion",
            "title": "RandomDeletion",
            "generate": ["keras_nlp.layers.RandomDeletion"],
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
            "title": "TransformerEncoder",
            "generate": [
                "keras_nlp.layers.TransformerEncoder",
                "keras_nlp.layers.TransformerEncoder.call",
            ],
        },
        {
            "path": "transformer_decoder",
            "title": "TransformerDecoder",
            "generate": [
                "keras_nlp.layers.TransformerDecoder",
                "keras_nlp.layers.TransformerDecoder.call",
            ],
        },
        {
            "path": "fnet_encoder",
            "title": "FNetEncoder",
            "generate": ["keras_nlp.layers.FNetEncoder"],
        },
        {
            "path": "position_embedding",
            "title": "PositionEmbedding",
            "generate": ["keras_nlp.layers.PositionEmbedding"],
        },
        {
            "path": "sine_position_encoding",
            "title": "SinePositionEncoding",
            "generate": ["keras_nlp.layers.SinePositionEncoding"],
        },
        {
            "path": "token_and_position_embedding",
            "title": "TokenAndPositionEmbedding",
            "generate": ["keras_nlp.layers.TokenAndPositionEmbedding"],
        },
        {
            "path": "mlm_head",
            "title": "MLMHead",
            "generate": ["keras_nlp.layers.MLMHead"],
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
            "title": "Perplexity",
            "generate": ["keras_nlp.metrics.Perplexity"],
        },
        {
            "path": "rouge_l",
            "title": "RougeL",
            "generate": ["keras_nlp.metrics.RougeL"],
        },
        {
            "path": "rouge_n",
            "title": "RougeN",
            "generate": ["keras_nlp.metrics.RougeN"],
        },
        {
            "path": "bleu",
            "title": "Bleu",
            "generate": ["keras_nlp.metrics.Bleu"],
        },
        {
            "path": "edit_distance",
            "title": "EditDistance",
            "generate": ["keras_nlp.metrics.EditDistance"],
        },

    ],
}

UTILS_MASTER = {
    "path": "utils/",
    "title": "Utils",
    "toc": True,
    "children": [
        {
            "path": "greedy_search",
            "title": "greedy_search function",
            "generate": ["keras_nlp.utils.greedy_search"],
        },
        {
            "path": "top_k_search",
            "title": "top_k_search function",
            "generate": ["keras_nlp.utils.top_k_search"],
        },
        {
            "path": "top_p_search",
            "title": "top_p_search function",
            "generate": ["keras_nlp.utils.top_p_search"],
        },
        {
            "path": "random_search",
            "title": "random_search function",
            "generate": ["keras_nlp.utils.random_search"],
        },
        {
            "path": "beam_search",
            "title": "beam_search function",
            "generate": ["keras_nlp.utils.beam_search"],
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
        METRICS_MASTER,
        UTILS_MASTER,
    ],
}
