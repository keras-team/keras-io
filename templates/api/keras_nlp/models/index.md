# KerasNLP Models

KerasNLP contains end-to-end implementations of popular model
architectures. These models can be created in two ways:

1) Through the `from_preset()` constructor, which instantiates an object with
a pre-trained configurations, vocabularies, and (optionally) weights. Available
preset IDs are listed on this page.

2) Through custom configuration controlled by the user. To do this, simply
pass the desired configuration parameters to the default constructors of the
symbols documented below.

## Backbone presets

The following preset IDs correspond to a configuration, weights and vocabulary
for a model **backbone**. These presets are not inference ready, and must be
fine-tuned for a given task!

IDs below can be used with any `from_preset()` constructor for a given model.
For example, a BERT backbone preset can be used with
`keras_nlp.models.BertTokenizer`,
`keras_nlp.models.BertPreprocessor`,
`keras_nlp.models.BertBackbone`, or
`keras_nlp.models.BertClassifier`.

| Preset ID                    | Model                      | Parameters  | Description |
| ------------------------     | ------------               | ----------- | ----------- |
| bert_tiny_en_uncased         | [BERT](bert)               | 4M          | 2-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_small_en_uncased        | [BERT](bert)               | 28M         | 4-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_medium_en_uncased       | [BERT](bert)               | 41M         | 8-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_base_en_uncased         | [BERT](bert)               | 109M        | 12-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_base_en_cased           | [BERT](bert)               | 108M        | 12-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| bert_base_zh                 | [BERT](bert)               | 102M        | 12-layer BERT model. Trained on Chinese Wikipedia. |
| bert_base_multi_cased        | [BERT](bert)               | 177M        | 12-layer BERT model where case is maintained. Trained on Wikipedias of 104 languages. |
| bert_large_en_uncased        | [BERT](bert)               | 335M        | 24-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| bert_large_en_cased          | [BERT](bert)               | 333M        | 24-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| roberta_base                 | [RoBERTa](roberta)         | 124M        | 12-layer RoBERTa model where case is maintained. Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| roberta_large                | [RoBERTa](roberta)         | 354M        | 24-layer RoBERTa model where case is maintained. Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| xlm_roberta_base             | [XLM-RoBERTa](xlm_roberta) | 277M        | 12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |
| xlm_roberta_large            | [XLM-RoBERTa](xlm_roberta) | 558M        | 24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |
| distil_bert_base_en_uncased  | [DistilBert](distil_bert)  | 66M         | 6-layer DistilBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| distil_bert_base_en_cased    | [DistilBert](distil_bert)  | 65M         | 6-layer DistilBERT model where case is maintained. Trained on English Wikipedia + BooksCorpus. |
| distil_bert_base_multi_cased | [DistilBert](distil_bert)  | 134M        | 6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages. |

## Classification presets

The following preset ID correspond to a configuration, weights and vocabulary
for a model **classifier**. These models are inference ready, but can be further
fine-tuned if desired.

IDs below can be used with the `from_preset()` constructor for classifier models
and preprocessing layers.
For example, a BERT classifier preset can be used with
`keras_nlp.models.BertTokenizer`,
`keras_nlp.models.BertPreprocessor`, or
`keras_nlp.models.BertClassifier`.

| Preset ID                    | Model                      | Parameters  | Description |
| ------------------------     | ------------               | ----------- | ----------- |
| bert_tiny_en_uncased_sst2    | [BERT](bert)               | 4M          | The `bert_tiny_en_uncased` backbone model fine-tuned on the SST-2 sentiment analysis dataset. |

## API Documentation

{{toc}}
