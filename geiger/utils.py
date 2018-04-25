"""General utilities"""
from textblob import TextBlob
import pandas as pd

def load_file(path, encoding="utf-8"):
    """
    Load file line generator
    Args:
        path: str: Lines
        encoding: str: encoding to open

    Returns: Generator

    """
    with open(path, encoding=encoding) as in_file:
        for l in in_file:
            yield l.strip()

def get_word_blob(word):
    return TextBlob(word)


def toxicity_label_map(labels):
    labels['Tag'] = ['NAG' if t == 0 else 'OAG' for t in labels.iloc[:, 0:5].max(axis=1)]
    return labels['Tag']


def load_toxicity_data_set(train_path=None):
    train_set_path = '~/Documents/toxicity_train.csv' if train_path is None else train_path
    train_data = pd.read_csv(train_set_path)

    X_train = train_data['comment_text']
    Y_train = toxicity_label_map(train_data.iloc[:, 2:7])
    return X_train, Y_train
