import pandas as pd
import tensorflow as tf
from official.nlp import optimization
from transformers import RobertaTokenizer
from config import *
from models import build_classifier_model

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
  classifier_model.save("../spam_roberta_classifier", include_optimizer=False)