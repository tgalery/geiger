"""General utilities"""
import functools
from textblob import TextBlob
import logging
import spacy
nlp = spacy.load('en_core_web_lg')

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

def memoize(func):
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func

@memoize
def get_text_blob(text):
    return TextBlob(text)

@memoize
def get_text_nlp(text):
    return nlp(text)

@memoize
def nlp_get_word_vector(word):
    return nlp.vocab.get_vector(word)
