import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import *
from train import train_model
from test import test_model
from config import *

# def test_model(test_df):
if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("The command to run the program is as follows:\n\"python main.py train\" for training")
    print("\"python main.py test\" for testing")
    exit(1)

  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

  texts, labels = parse_spam_dataset(filename="../data/SMSSpamCollection", zero_class="ham", one_class="spam")
  # Dataset statistics
  spam_dataset_stats(texts, labels)

  # Minimal text preprocessing
  for i in range(len(texts)):
    texts[i] = remove_punctuation(detect_and_shorten_url(texts[i]))

  df = pd.DataFrame({'Text': texts, 'Label': labels})
  df = df.sample(frac=1, random_state=SEED)
  reduced_df = remove_empty_rows(df)

  train_df, test_df = train_test_split(reduced_df, test_size=0.3, stratify=reduced_df['Label'], random_state=SEED)

  if sys.argv[1] == "train":
    train_model(train_df, test_df)
  elif sys.argv[1] == "test":
    test_model(test_df)
