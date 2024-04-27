import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from transformers import RobertaTokenizer
from config import *

def test_model(test_df):
    loaded_model = tf.saved_model.load('../spam_roberta_classifier')
    pred_classes = []
    last_index = 0

    # for i in range(int(len(test_df)/1500)):
    #     pred = loaded_model(tf.constant((test_df['combined_text'].values.tolist())[i*1500:(i+1)*1500]))
    #     pred_classes_per_batch = [dataset_classes[i] for i in np.argmax(pred, axis=1)]
    #     pred_classes = pred_classes + pred_classes_per_batch
    #     last_index = i
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    X_test_encoded = roberta_tokenizer(text=test_df["Text"].tolist(), padding='max_length', truncation=True, max_length=256,return_token_type_ids=True,return_tensors='tf')
    test_dataset = tf.data.Dataset.from_tensor_slices(({'input_word_ids': X_test_encoded['input_ids'],'input_mask': X_test_encoded['attention_mask']}, pd.get_dummies(test_df['Label'].values))).shuffle(buffer_size=len(X_test_encoded)).batch(BATCH_SIZE).prefetch(1)

    TEST_BATCH_SIZE = 150
    for i in range(int(len(test_df)/TEST_BATCH_SIZE)):
        pred = loaded_model([X_test_encoded['input_ids'][i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE], X_test_encoded['attention_mask'][i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE]], training=False)
        pred_classes_per_batch = [i for i in np.argmax(pred, axis=1)]
        pred_classes = pred_classes + pred_classes_per_batch
        last_index = i

    pred = loaded_model([X_test_encoded['input_ids'][(last_index+1)*TEST_BATCH_SIZE:], X_test_encoded['attention_mask'][(last_index+1)*TEST_BATCH_SIZE:]], training=False)
    pred_classes_per_batch = [i for i in np.argmax(pred, axis=1)]
    pred_classes = pred_classes + pred_classes_per_batch
    print("\n\nPredicted classes: \n", pred_classes)

    print("Model performance based on the test dataset")

    test_actual_classes = test_df["Label"]
    print("\n\nActual classes: \n", test_actual_classes)

    test_df['prediction'] = pred_classes
    test_df.to_csv('test_dataset_withpred.csv', index=False)

    # Calculate metrics
    accuracy = accuracy_score(test_actual_classes, pred_classes)
    f1 = f1_score(test_actual_classes, pred_classes, average='weighted')
    precision = precision_score(test_actual_classes, pred_classes, average='weighted')
    recall = recall_score(test_actual_classes, pred_classes, average='weighted')

    # Print metrics
    print("Accuracy: {:.5f}".format(accuracy))
    print("F1-score: {:.5f}".format(f1))
    print("Precision: {:.5f}".format(precision))
    print("Recall: {:.5f}".format(recall))

    # Confusion matrix
    cm = confusion_matrix(test_actual_classes, pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=pd.get_dummies(["Ham", "Spam"]).columns.tolist(), yticklabels=pd.get_dummies(["Ham", "Spam"]).columns.tolist())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig('../figs/confmatrix.png')
