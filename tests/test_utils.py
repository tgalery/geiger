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


def test_categorical_array():
    soft_max = np.asarray([0.5, 0.2, 0.3])
    assert np.array_equal(utils.softmax_array_to_categorical(soft_max), np.asarray([1, 0, 0]))


def test_categorical_matrix():
    soft_max = np.array([[0.5, 0.2, 0.3], [0.4, 0.1, 0.5]])
    assert np.array_equal(utils.softmax_to_categorical(soft_max), np.asmatrix([[1, 0, 0], [0, 0, 1]]))
