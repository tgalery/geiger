import numpy as np
import pandas as pd
import os
# possible values are {'OAG', 'CAG', 'NAG'}
COLING_CLASSES = ['NAG', 'CAG', 'OAG']


def one_hot_encode(tag):
    """
    Transform a class tag
    Args:
        tag: str: corresponding to the class tag

    Returns:
        list
    """
    return [1 if tag == class_tag else 0 for class_tag in COLING_CLASSES]


def one_hot_decode(class_array):
    """
    Decode an array of ints representing the tweet class
    Args:
        class_array:

    Returns:

    """
    class_idx = class_array.index(1)
    return COLING_CLASSES[int(class_idx)]


def softmax_array_to_categorical(np_array):
    argmax = max(np_array)
    return np.array([1 if n == argmax else 0 for n in np_array])


def softmax_to_categorical(np_matrix):
    return np.apply_along_axis(softmax_array_to_categorical, 1, np_matrix)

def pred_to_label(np_array):
    return one_hot_decode(list(softmax_array_to_categorical(np_array)))


def dump_coling_predictions(doc_ids, pred_classes, fpath):
    class_labels = list(np.apply_along_axis(pred_to_label, 1, pred_classes))
    dataf = pd.DataFrame(list(zip(doc_ids, class_labels)))
    dataf.to_csv(fpath, header=False, index=False)


def load_coling_file(fname):
    col_names = ['id', 'text', 'class']
    data = pd.read_csv(os.path.join(fname), names=col_names)
    doc_ids = data["id"].values
    x = data["text"].values
    y = data["class"].values
    return doc_ids, x, y


def load_coling_data(dir_name):
    """
    Load Coling data
    Args:
        dir_name:

    Returns:
    """
    _, x_train, y_train = load_coling_file(os.path.join(dir_name, "agr_en_train.csv"))
    _, x_dev, y_dev = load_coling_file(os.path.join(dir_name, "agr_en_dev.csv"))
    y_train = np.asarray([one_hot_encode(class_tag) for class_tag in y_train])
    y_dev = np.asarray([one_hot_encode(class_tag) for class_tag in y_dev])
    return x_train, x_dev, y_train, y_dev
