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
            "path": "unicode_character_tokenizer",
            "title": "UnicodeCharacterTokenizer",
            "generate": [
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.tokenize",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.detokenize",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.get_vocabulary",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.vocabulary_size",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.token_to_id",
                "keras_nlp.tokenizers.UnicodeCharacterTokenizer.id_to_token",
            ],
        },
    ],
}

LAYERS_MASTER = {
    "path": "layers/",
    "title": "Layers",
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
            "path": "mlm_mask_generator",
            "title": "MLMMaskGenerator layer",
            "generate": ["keras_nlp.layers.MLMMaskGenerator"],
        },
        {
            "path": "mlm_head",
            "title": "MLMHead layer",
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
            "title": "Perplexity metric",
            "generate": ["keras_nlp.metrics.Perplexity"],
        },
    ],
}

NLP_API_MASTER = {
    "path": "keras_nlp/",
    "title": "KerasNLP",
    "toc": True,
    "children": [TOKENIZERS_MASTER, LAYERS_MASTER, METRICS_MASTER],
}
