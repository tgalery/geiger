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
