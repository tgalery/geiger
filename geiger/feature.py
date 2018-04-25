from os import path

from geiger.utils import load_file, get_text_blob, nlp_get_word_vector, get_text_nlp

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

    Returns: list
    """
    wb = get_text_blob(word)
    if wb:
        return [wb.sentiment.polarity, wb.sentiment.subjectivity]
    else:
        return [0, 0]

def get_pos_tag(word):
    """
    Retrieve pos tag of word.
    Args:
        word: str: word to be evaluated

    Returns: string
    """
    wb = get_text_blob(word)
    if wb:
        return wb.tags[0][1]
    else:
        return 'NN'

def get_word_vector(word):
    """
    Retrieve embedding vector of word.
    Args:
        word: str: word to be evaluated

    Returns: array
    """
    return nlp_get_word_vector(word)

def get_word_is_stop(word):
    """
    Retrieve wether a word is a stopword.
    Args:
        word: str: word to be evaluated

    Returns: bool
    """
    wb = get_text_nlp(word)
    if wb:
        return list(wb)[0].is_stop
    else:
        return False
