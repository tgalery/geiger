import numpy as np
from geiger.toxic import to_coling_categorical, toxic_to_coling, OAG, NAG

toxic_y = np.asarray([0, 1, 1, 0, 1, 1])
non_toxic_y = np.asarray([0, 0, 0, 0, 0, 0])


def test_toxic_to_coling():
    assert np.array_equal(toxic_to_coling(toxic_y), OAG)
    assert np.array_equal(toxic_to_coling(non_toxic_y), NAG)


def test_to_coling_categorical():
    Y = np.asarray([toxic_y, non_toxic_y])
    assert np.array_equal(to_coling_categorical(Y), np.asarray([OAG, NAG]))
