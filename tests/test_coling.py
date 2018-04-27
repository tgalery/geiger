# COLING_CLASSES = ['NAG', 'CAG', 'OAG']
from geiger.coling import one_hot_encode, one_hot_decode


def test_encode():
    assert one_hot_encode("NAG") == [1, 0, 0]
    assert one_hot_encode("CAG") == [0, 1, 0]
    assert one_hot_encode("OAG") == [0, 0, 1]


def test_decode():
    assert one_hot_decode([1, 0, 0]) == "NAG"
    assert one_hot_decode([0, 1, 0]) == "CAG"
    assert one_hot_decode([0, 0, 1]) == "OAG"
