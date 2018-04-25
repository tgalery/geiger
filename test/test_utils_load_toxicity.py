from geiger.utils import load_toxicity_data_set
import pandas as pd

def test_load_toxicity_data_set():
    X_train, Y_train = load_toxicity_data_set()

    assert 'Explanation\nWhy the edits made under my' in X_train[0]
    assert set(Y_train[0:10]) == set(['NAG', 'NAG', 'NAG', 'NAG', 'NAG', 'NAG', 'OAG', 'NAG', 'NAG', 'NAG',
                                              'NAG'])