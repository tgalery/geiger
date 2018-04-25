import pandas as pd


def load_jigsaw_dataset(fpath):
    """
    Loads a jigsaw dataset using pandas csv reader
    :param path: string: path to dataset
    :return: DataFrame
    """
    return pd.read_csv(fpath)