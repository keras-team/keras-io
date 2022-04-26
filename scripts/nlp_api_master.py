TOKENIZERS_MASTER = {
    'path': 'tokenizers/',
    'title': 'Tokenizers',
    'toc': True,
    'children': [
        {
            'path': 'tokenizer',
            'title': 'Tokenizer base class',
            'generate': ['keras_nlp.tokenizers.Tokenizer']
        },
        {
            'path': 'word_piece_tokenizer',
            'title': 'WordPieceTokenizer',
            'generate': ['keras_nlp.tokenizers.WordPieceTokenizer']
        },
        {
            'path': 'byte_tokenizer',
            'title': 'WordPieceTokenizer',
            'generate': ['keras_nlp.tokenizers.ByteTokenizer']
        },
        {
            'path': 'unicode_character_tokenizer',
            'title': 'UnicodeCharacterTokenizer',
            'generate': ['keras_nlp.tokenizers.UnicodeCharacterTokenizer']
        },
    ]
}

PREPROCESSING_MASTER = {
    'path': 'preprocessing/',
    'title': 'Preprocessing layers',
    'toc': True,
    'children': [
        {
            'path': 'mlm_mask_generator',
            'title': 'MLMMaskGenerator layer',
            'generate': ['keras_nlp.layers.MLMMaskGenerator']
        },
    ]
}

LAYERS_MASTER = {
    'path': 'layers/',
    'title': 'Layers',
    'toc': True,
    'children': [
        {
            'path': 'transformer_encoder',
            'title': 'TransformerEncoder layer',
            'generate': [
                'keras_nlp.layers.TransformerEncoder',
            ]
        },
        {
            'path': 'transformer_decoder',
            'title': 'TransformerDecoder layer',
            'generate': [
                'keras_nlp.layers.TransformerDecoder',
            ]
        },
        {
            'path': 'fnet_encoder',
            'title': 'FNetEncoder layer',
            'generate': [
                'keras_nlp.layers.FNetEncoder',
            ]
        },
        {
            'path': 'position_embedding',
            'title': 'PositionEmbedding layer',
            'generate': [
                'keras_nlp.layers.PositionEmbedding',
            ]
        },
        {
            'path': 'sine_position_encoding',
            'title': 'SinePositionEncoding layer',
            'generate': [
                'keras_nlp.layers.SinePositionEncoding',
            ]
        },
        {
            'path': 'token_and_position_embedding',
            'title': 'TokenAndPositionEmbedding layer',
            'generate': [
                'keras_nlp.layers.TokenAndPositionEmbedding',
            ]
        },
        {
            'path': 'mlm_classification_head',
            'title': 'MLMHead layer',
            'generate': [
                'keras_nlp.layers.MLMHead',
            ]
        },
        PREPROCESSING_MASTER,
    ]
}


METRICS_MASTER = {
    'path': 'metrics/',
    'title': 'Metrics',
    'toc': True,
    'children': [
        {
            'path': 'perplexity',
            'title': 'Perplexity metric',
            'generate': [
                'keras_nlp.metrics.Perplexity',
            ]
        },
    ]
}

NLP_API_MASTER = {
    'path': 'keras_nlp/',
    'title': 'KerasNLP',
    'toc': True,
    'children': [
        TOKENIZERS_MASTER,
        LAYERS_MASTER,
        METRICS_MASTER
    ]
}
