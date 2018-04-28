from geiger import utils
import numpy as np


def test_is_devanagari():
    lang = utils.is_devanagari("बिक्रम मेरो नाम हो")
    assert lang is True


def test_find_nrams():
    n_grams = utils.find_ngrams("ABC", 2)
    assert set(n_grams) == {'AB', 'BC'}


def test_generate_ngrams():
    pseudo_word = "porkistan"
    ngrams = utils.generate_n_grams(pseudo_word)
    assert set(ngrams) == {'por', 'ork', 'rki', 'kis', 'ist', 'sta', 'tan'}


