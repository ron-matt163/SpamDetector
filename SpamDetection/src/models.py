import tensorflow as tf
from transformers import TFRobertaModel


def build_classifier_model():
  input_word_ids = tf.keras.Input(shape=(256,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.Input(shape=(256,), dtype=tf.int32, name='input_mask')
  roberta_tf = TFRobertaModel.from_pretrained("roberta-base")
  embeddings = roberta_tf(input_word_ids, attention_mask = input_mask)[0]
  net = tf.keras.layers.GlobalMaxPool1D()(embeddings)
  net = tf.keras.layers.Dropout(0.45)(net)
  op = tf.keras.layers.Dense(26, activation='softmax')(net)
  return tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=op)  

