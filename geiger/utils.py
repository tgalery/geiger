"""General utilities"""
import os

import numpy as np
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

