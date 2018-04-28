# COLING_CLASSES = ['NAG', 'CAG', 'OAG']
import numpy as np
from geiger.coling import one_hot_encode, one_hot_decode, softmax_array_to_categorical, softmax_to_categorical


def test_encode():
    assert one_hot_encode("NAG") == [1, 0, 0]
    assert one_hot_encode("CAG") == [0, 1, 0]
    assert one_hot_encode("OAG") == [0, 0, 1]


def test_decode():
    assert one_hot_decode([1, 0, 0]) == "NAG"
    assert one_hot_decode([0, 1, 0]) == "CAG"
    assert one_hot_decode([0, 0, 1]) == "OAG"


def test_categorical_array():
    soft_max = np.asarray([0.5, 0.2, 0.3])
    assert np.array_equal(softmax_array_to_categorical(soft_max), np.asarray([1, 0, 0]))


def test_categorical_matrix():
    soft_max = np.array([[0.5, 0.2, 0.3], [0.4, 0.1, 0.5]])
    assert np.array_equal(softmax_to_categorical(soft_max), np.asmatrix([[1, 0, 0], [0, 0, 1]]))

