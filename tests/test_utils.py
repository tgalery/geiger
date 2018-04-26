from os import path
from geiger.utils import load_word_vectors


ROOT_FOLDER = path.abspath(__file__).rsplit("/tests")[0]
# from geiger.utils import load_toxicity_data_set
#
#
# def test_load_toxicity_data_set():
#
#     x_train, y_train = load_toxicity_data_set()
#
#     assert 'Explanation\nWhy the edits made under my' in x_train[0]
#     assert set(y_train[0:10]) == {'NAG', 'OAG'}


def test_load_word_vectors():
    fpath = path.join(ROOT_FOLDER, "resources/wiki-news-300d-1M-subword.vec")
    if path.isfile(fpath):
        print("\nLoading vectors for test.")
        vecs = load_word_vectors(fpath)
        assert len(vecs["hello"]) == 300
