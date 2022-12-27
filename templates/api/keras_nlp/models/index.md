# KerasNLP Models

KerasNLP contains end-to-end implementations of popular model
architectures. These models can be created in two ways:

- Through the `from_preset()` constructor, which instantiates an object with
  a pre-trained configurations, vocabularies, and (optionally) weights.
  Available preset IDs are listed on this page.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased")
```

- Through custom configuration controlled by the user. To do this, simply
  pass the desired configuration parameters to the default constructors of the
  symbols documented below.

```python
tokenizer = keras_nlp.models.BertTokenizer(
    vocabulary="./vocab.txt",
)
preprocessor = keras_nlp.models.BertPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_nlp.models.BertBackbone(
    vocabulary_size=30552,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=3072,
    max_sequence_length=128,
)
classifier = keras_nlp.models.BertClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=4,
)
```

For a more in depth introduction to how our API fits together, see the
[getting started guide](guides/keras_nlp/getting_started/).

## Backbone presets

The following preset IDs correspond to a configuration, weights and vocabulary
for a model **backbone**. These presets are not inference-ready, and must be
fine-tuned for a given task!

IDs below can be used with any `from_preset()` constructor for a given model.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased")
backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
```

| Preset ID                    | Model                      | Parameters  | Description |
| ------------------------     | ------------               | ----------- | ----------- |
| bert_tiny_en_uncased         | [BERT](bert)               | 4M          | 2-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_small_en_uncased        | [BERT](bert)               | 28M         | 4-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_medium_en_uncased       | [BERT](bert)               | 41M         | 8-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_base_en_uncased         | [BERT](bert)               | 109M        | 12-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_base_en                 | [BERT](bert)               | 108M        | 12-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| bert_base_zh                 | [BERT](bert)               | 102M        | 12-layer BERT model. Trained on Chinese Wikipedia. |
| bert_base_multi              | [BERT](bert)               | 177M        | 12-layer BERT model where case is maintained. Trained on Wikipedias of 104 languages. |
| bert_large_en_uncased        | [BERT](bert)               | 335M        | 24-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_large_en                | [BERT](bert)               | 333M        | 24-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| distil_bert_base_en_uncased  | [DistilBert](distil_bert)  | 66M         | 6-layer DistilBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| distil_bert_base_en          | [DistilBert](distil_bert)  | 65M         | 6-layer DistilBERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| distil_bert_base_multi       | [DistilBert](distil_bert)  | 134M        | 6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages. |
| roberta_base_en              | [RoBERTa](roberta)         | 124M        | 12-layer RoBERTa model where case is maintained. Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| roberta_large_en             | [RoBERTa](roberta)         | 354M        | 24-layer RoBERTa model where case is maintained. Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| xlm_roberta_base_multi       | [XLM-RoBERTa](xlm_roberta) | 277M        | 12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |
| xlm_roberta_large_multi      | [XLM-RoBERTa](xlm_roberta) | 558M        | 24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |

## Classification presets

The following preset ID correspond to a configuration, weights and vocabulary
for a model **classifier**. These models are inference ready, but can be further
fine-tuned if desired.

IDs below can be used with the `from_preset()` constructor for classifier models
and preprocessing layers.

```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased_sst2")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased_sst2")
```

| Preset ID                    | Model                      | Parameters  | Description |
| ------------------------     | ------------               | ----------- | ----------- |
| bert_tiny_en_uncased_sst2    | [BERT](bert)               | 4M          | The `bert_tiny_en_uncased` backbone model fine-tuned on the SST-2 sentiment analysis dataset. |

## API Documentation

{{toc}}
