from datasets import load_dataset
import tensorflow as tf
from tensorflow import keras
import keras_nlp

conll = load_dataset("conll2003")

label_list = conll["train"].features[f"ner_tags"].feature.names
label_list

preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_base_uncased", sequence_length=128)
tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_base_uncased")

example = conll['train'][4]
sentence_toks = example["tokens"]
sentence = " ".join(sentence_toks)
labels = example["ner_tags"]

print(sentence_toks)
print(len(sentence_toks))

tokenized_sentence = tokenizer(sentence)
tokenized_sentence_to_words = []
for tok in tokenized_sentence:
  tokenized_sentence_to_words.append(tokenizer.id_to_token(tok))
print(tokenized_sentence_to_words)
print(len(tokenized_sentence_to_words))

def align_labels_to_tokens(sample):
  sentence, labels = sample[2], sample[1]
  rowIndex = sample.name
  sentence_toks = sentence.split(" ")#examples[0]

  #sentence = " ".join(sentence_toks)
  #labels = examples[1]
  
  preprocessed_sentence = {}
  preprocessed_sentence["token_ids"] = preprocessed_mini["token_ids"][rowIndex]
  preprocessed_sentence["padding_mask"] = preprocessed_mini["padding_mask"][rowIndex]
  #preprocessor(sentence)
  tokenized_sentence = preprocessed_sentence["token_ids"]
  tokenized_sentence_to_words = []
  for tok in tokenized_sentence:
    tokenized_sentence_to_words.append(preprocessor.tokenizer.id_to_token(tok))

  parent_words = []
  """
  loop over tokenized_sentence_to_words, keep on taking words until current word 
  in sentence_toks is formed
  """

  i = 0 # pointer to sentence_toks
  j = 0 # pointer to tokenized_sentence_to_words

  current_word = ""
  buffer_cnt = 0

  while j<len(tokenized_sentence_to_words):

    if tokenized_sentence[j] <= 3:
      parent_words.append(-100)
    else:    
      current_token = tokenized_sentence_to_words[j]
      buffer_cnt += 1

      if current_token[0] == 'Ġ':
        current_token = current_token[1:]
      
      current_word += current_token

      if current_word == sentence_toks[i]:
        buffer_parent = [i] * buffer_cnt
        parent_words = [*parent_words, *buffer_parent]

        current_word = ""
        buffer_cnt = 0
        i += 1
    j+=1

  aligned_labels = []
  # print(parent_words)
  # print(labels)
  for i, parent_id in enumerate(parent_words):
    #print(i,end=" ")
    if parent_id>=0:
      # check if previous word had same parent
      if i > 0 and parent_id == parent_words[i-1]:
        aligned_labels.append(-100)
      else:
        aligned_labels.append(labels[parent_id])
    else:
      aligned_labels.append(-100) # special token <s>, </s>, <pad>
  
  #print(sample.name, type(sample.name))
  return aligned_labels

train_df_mini = train_df.loc[:10, :]
preprocessed_mini = preprocessor(train_df_mini["sentence"])
train_df_mini.head()

train_df_mini.loc["new_labels"] = train_df_mini.apply(lambda x: align_labels_to_tokens(x), axis=1)

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_base_uncased", num_classes=128, preprocessor=None) # sequence_length is 128

classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #optimizer=tf.keras.optimizers.Adam(1e-5),
              metrics=["accuracy"])

classifier.summary()

x_train = np.array(train_df_mini["sentence"])
x_train = preprocessor(x_train)
#y_train = tf.constant(train_df_mini["new_labels"])

import numpy as np
y_train = train_df_mini["new_labels"].to_list()
y_train = [tf.convert_to_tensor(item) for item in y_train]
y_train = tf.convert_to_tensor(y_train)
y_train.shape

y_train = tf.convert_to_tensor(y_train)
y_train

classifier.summary()

classifier.fit(x=x_train, y=y_train, epochs=1)
classifier.predict(x_train)