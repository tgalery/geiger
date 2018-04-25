from os import path

from geiger.utils import load_file

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
