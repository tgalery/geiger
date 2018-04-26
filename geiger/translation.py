# Imports the Google Cloud client library
from googletrans import Translator

def detect_lang(text):
    # detects language of text
    pass


def translate_to_target(translator, text, src, dest):
    return translator.translate(text,src=src,dest=dest)


def translate_text(text):
    """ Translates string from src to each language in dest
    Translate back to english.
    returns list of strings
    """
    src = 'en'
    langs = ['nl','de','it','fr']
    translator = Translator()
    trans = [translate_to_target(translator, text, src, l) for l in langs]
    trans = [t.text for t in trans]
    dest = 'en'
    trans = [translate_to_target(translator, text, l, dest) for l in langs]
    trans = [t.text for t in trans]
    return set([t.text for t in trans if t])