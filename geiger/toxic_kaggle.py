import pandas as pd

def toxicity_label_map(labels):
    labels['Tag'] = ['NAG' if t == 0 else 'OAG' for t in labels.iloc[:, 0:5].max(axis=1)]
    return labels['Tag']


def load_toxicity_data_set(fpath, column_name="comment_text"):
    """
    Load a toxic dataset
    Args:
        fpath: str: file where data is located
        column_name: str or list: name of column to load:
            X = comment_text
            Y = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    Returns: np.array like object
    """
    dframe = pd.read_csv(fpath)
    return dframe[column_name].fillna("fillna").values
