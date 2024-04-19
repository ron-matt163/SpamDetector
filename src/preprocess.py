import numpy as np
import re
from nltk.tokenize import RegexpTokenizer

def parse_spam_dataset(filename, zero_class, one_class):
    dataset_dict = {0:[], 1:[]}
    with open(filename, "r") as file:
        line_idx = 0
        for line in file:
            line_idx += 1
            line = line.strip()
            i = 0
            while i < len(line) and not line[i].isspace():
                i += 1
            if i==len(line):
                print(f"The class is not provided in the {line_idx}'th record")
                exit(1)
            else:
                _class = line[:i]
                text = line[i+1]
                if _class == zero_class:
                    dataset_dict[0].append(text)
                elif _class == one_class:
                    dataset_dict[1].append(text)
                else:
                    print(f"Invalid class at record {line_idx} of the dataset")
                    exit(1)
    print(f"Number of hams: {len(dataset_dict[0])}, Number of spams: {len(dataset_dict[1])}")
    texts = dataset_dict[0] + dataset_dict[1]
    labels = [0]*len(dataset_dict[0]) + [1]*len(dataset_dict[1])
    return texts, labels

def detect_and_shorten_url(text):
    # Define a regex pattern to match URLs
    url_pattern = r'\b(?:https?://)\S+\b'
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    if urls:
        for url in urls:
            shortened_url = '/'.join(url.split('/', 3)[:3])
            text = text.replace(url, shortened_url)
    return text

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

def remove_empty_rows(df):
    df_cleaned = df.dropna(subset=['Text', 'Label'])
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned
