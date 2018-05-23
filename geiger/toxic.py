import numpy as np
import pandas as pd

OAG = np.asarray([0, 0, 1])
NAG = np.asarray([1, 0, 0])


def load_toxic_data(fpath):
    """
    Load a toxic dataset
    Args:
        fpath: str: file where data is located
            X = comment_text
            Y = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    Returns: np.array like object
    """
    dframe = pd.read_csv(fpath)
    x = dframe["comment_text"].fillna("fillna").values
    y = dframe[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    return x, y


def toxic_to_coling(arr):
    return OAG if np.any(arr) else NAG


def to_coling_categorical(Y):
    return np.apply_along_axis(toxic_to_coling, 1, Y)


def load_toxic_as_coling(fpath):
    x, y = load_toxic_data(fpath)
    return x, to_coling_categorical(y)
