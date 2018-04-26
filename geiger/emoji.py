import unicodedata
import re

EMOJI_EXP = re.compile('[\U0001F602-\U0001F64F]')  # Commpile regex for re use


def is_emoji(word):
    """
    Determine if token is emoji or not
    Args:
        word: str: word to be determined as emoji

    Returns:

    """
    m = re.match(EMOJI_EXP, word)
    return True if m else False


def get_emoji_description(emoji):
    """
    Retrieve emoji description
    Args:
        emoji: str: emoji's description

    Returns: str

    """
    return unicodedata.name(emoji).lower()
