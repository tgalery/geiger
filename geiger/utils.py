"""General utilities"""
import os

import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm


def get_file_lines(fpath):
    output = os.popen('wc -l {}'.format(fpath)).read()
    split_out = output.strip().split(" ")
    if split_out:
        return int(split_out[0])
    return 0


def load_file(fpath, encoding="utf-8"):
    """
    Load file line generator
    Args:
        fpath: str: Lines
        encoding: str: encoding to open

    Returns: Generator
    """
    max_lines = get_file_lines(fpath)
    with open(fpath, encoding=encoding) as in_file:
        with tqdm(in_file, total=max_lines) as line_gen:
            for l in line_gen:
                yield l.rstrip()


def get_word_blob(word):
    return TextBlob(word)


def toxicity_label_map(labels):
    labels['Tag'] = ['NAG' if t == 0 else 'OAG' for t in labels.iloc[:, 0:5].max(axis=1)]
    return labels['Tag']


def load_toxicity_data_set(fpath, column_name="comment_text"):
    """
    Load a toxic dataset
    Args:
        fpath: str: file where data is located
        column_name: str or list: name of column to load:
            X = comment_text
            Y = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    Returns: np.array like object
    """
    dframe = pd.read_csv(fpath)
    return dframe[column_name].fillna("fillna").values


def to_np_array(word, *arr):
    """
    Transform a sequence of numbers into a numpy array.
    Args:
        word: string: representing word vector.
        *arr: list: [0.2, 0.3 ..]

    Returns: tuple
    """
    return word, np.asarray(arr, dtype='float32')


def load_word_vectors(fpath):

    return dict(to_np_array(*vec_line.rsplit(' ')) for vec_line in load_file(fpath))
