from os import path

from geiger.utils import load_file, get_word_blob
import numpy as np

ROOT_DIR = path.abspath(__file__).rsplit("/geiger", 1)[0]

BAD_WORDS = set(load_file(path.join(ROOT_DIR, "resources/bad_word_list.txt"), encoding="latin-1")).union(
    set((l.lower() for l in load_file(path.join(ROOT_DIR, "resources/ethnic_slurs.txt"))))
)


def is_bad_word(word):
    """
    Determine if a word is blacklisted.
    Args:
        word: str: word to be checked

    Returns: bool
    """
    return word.lower() in BAD_WORDS


def get_sentiment_vector(word):
    """
    Retrieve sentiment vector of word.
    Args:
        word: str: word to be evaluated

    Returns: [polarity, subjectivity]
    """
    wb = get_word_blob(word)
    if wb:
        return [wb.sentiment.polarity, wb.sentiment.subjectivity]
    else:
        return [0., 0.]


class FeatureAugmenter:
    n_features = 3

    @staticmethod
    def get_features(word):

        sent_vec = get_sentiment_vector(word)
        sent_vec.append(float(is_bad_word(word)))
        return np.asarray(sent_vec)
