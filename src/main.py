import os
import pandas as pd
from numpy.lib.function_base import median
from numpy import mean
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from official.nlp import optimization
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from models import build_classifier_model
from preprocess import parse_spam_dataset, remove_punctuation, detect_and_shorten_url, remove_empty_rows


def train_model(train_df, test_df):
  # Print class distribution for verification
  print("Training class distribution:\n", train_df['Label'].value_counts(normalize=True))
  print("Test class distribution:\n", test_df['Label'].value_counts(normalize=True))

  print("\nDATASET CLASSES: ", pd.get_dummies(train_df['Label'].values).columns.tolist())
  print("\nActual test classes: ", test_df['Label'])

  roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  X_train_encoded = roberta_tokenizer(text=train_df["Text"].tolist(), padding='max_length', truncation=True, max_length=256,return_token_type_ids=True,return_tensors='tf')
  X_test_encoded = roberta_tokenizer(text=test_df["Text"].tolist(), padding='max_length', truncation=True, max_length=256,return_token_type_ids=True,return_tensors='tf')

  train_dataset = tf.data.Dataset.from_tensor_slices(({'input_word_ids': X_train_encoded['input_ids'],'input_mask': X_train_encoded['attention_mask']}, pd.get_dummies(train_df['Label'].values))).shuffle(buffer_size=len(X_train_encoded)).batch(BATCH_SIZE).prefetch(1)
  test_dataset = tf.data.Dataset.from_tensor_slices(({'input_word_ids': X_test_encoded['input_ids'],'input_mask': X_test_encoded['attention_mask']}, pd.get_dummies(test_df['Label'].values))).shuffle(buffer_size=len(X_test_encoded)).batch(BATCH_SIZE).prefetch(1)

  classifier_model = build_classifier_model()
  loss = tf.keras.losses.CategoricalCrossentropy()
  metrics = tf.metrics.CategoricalAccuracy()
  epochs = 10
  steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  init_lr = 2e-5
  optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

  classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  hist = classifier_model.fit(x=train_dataset, validation_data=test_dataset, epochs=epochs)

  print("History: \n", hist)
  loss, accuracy = classifier_model.evaluate(test_dataset)

  print(f'Loss: {loss}')
  print(f'Accuracy: {accuracy}')
  classifier_model.save("../spam_roberta_classifier.pth", include_optimizer=False)

# def test_model(test_df):

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  BATCH_SIZE = 32
  SEED = 42

  texts, labels = parse_spam_dataset(filename="../data/SMSSpamCollection", zero_class="ham", one_class="spam")
  # Minimal text preprocessing
  for i in range(len(texts)):
    texts[i] = remove_punctuation(detect_and_shorten_url(texts[i]))

  df = pd.DataFrame({'Text': texts, 'Label': labels})
  df = df.sample(frac=1, random_state=SEED)
  reduced_df = remove_empty_rows(df)

  train_df, test_df = train_test_split(reduced_df, test_size=0.3, stratify=reduced_df['Label'], random_state=SEED)

  train_model(train_df, test_df)
  # test_model(test_df)
