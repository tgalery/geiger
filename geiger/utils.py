"""General utilities"""
import os

import numpy as np
import pandas as pd
import pycld2 as cld2
from textblob import TextBlob
from tqdm import tqdm
from geiger.libs.fastText_multilingual import fasttext
import unicodedata


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


def load_word_vectors(fpath, transform_fpath=None):
    model = fasttext.FastVector(vector_file=fpath)
    if isinstance(transform_fpath, str):
        model.apply_transform(transform_fpath)
    return model


def load_coling_data(dir_name):
    """
    Load Coling data
    Args:
        dir_name:

    Returns:
    """
    col_names = ['id', 'text', 'class']
    train = pd.read_csv(os.path.join(dir_name, "agr_en_train.csv"), names=col_names)
    dev = pd.read_csv(os.path.join(dir_name, "agr_en_dev.csv"), names=col_names)

    x_train = train["text"].fillna("fillna").values
    y_train = train["class"].fillna("fillna").values

    x_dev = dev["text"].values
    y_dev = dev["class"].values
    return x_train, x_dev, y_train, y_dev


def is_devanagari(word):
    try:
        return any((unicodedata.name(c).startswith("DEVANAGARI") for c in word))
    except Exception as ex:
        print("Got {}".format(ex))
        return False


def find_ngrams(input_list, n):
    return ["".join(t) for t in zip(*[input_list[i:] for i in range(n)])]


def generate_n_grams(pseudo_word):
    n_gram_len = min(len(pseudo_word), 3)
    return find_ngrams(pseudo_word, n_gram_len)


def softmax_array_to_categorical(np_array):
    argmax = max(np_array)
    return np.array([1 if n == argmax else 0 for n in np_array])

def softmax_to_categorical(np_matrix):
    return np.apply_along_axis(softmax_array_to_categorical, 1, np_matrix)
