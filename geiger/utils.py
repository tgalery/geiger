"""General utilities"""


def load_file(path, encoding="utf-8"):
    """
    Load file line generator
    Args:
        path: str: Lines
        encoding: str: encoding to open

    Returns: Generator

    """
    with open(path, encoding=encoding) as in_file:
        for l in in_file:
            yield l.strip()
