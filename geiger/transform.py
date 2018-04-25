"""
Module for transforming sequences of text into Network Inputs
"""
from keras.preprocessing import text, sequence


class KerasTransformer:
    """
    A class for transforming text sequences
    """

    def __init__(self, x_texts, max_features, max_seq_len=-1):
        """
        Build a transformer
        Args:
            x_texts: x matrix of texts
            max_features: int: maximum number of features to be used for the model
            max_seq_len: int: maximum individual sequence size.
        """

        self.max_seq_len = max_seq_len
        self.tokenizer = text.Tokenizer(num_words=max_features)
        self.build_vocabulary(x_texts)

    def build_vocabulary(self, x_texts):
        """
        Fit text to tokenizer. Note both train and test sets should be used here.
        Args:
            x_texts: list: of texts to be transformed

        Returns: None

        """
        self.tokenizer.fit_on_texts(x_texts)

    def texts_to_seq(self, texts, pad=True):
        """
        Transform a text sequence to the proper sequential representation.
        Args:
            texts: list: of strings representing textual seq
            pad: bool: if sequences should be padded

        Returns: list of sequences with the right padding
        """
        text_seqs = self.tokenizer.texts_to_sequences(texts)
        if pad and self.max_seq_len > 0:
            return sequence.pad_sequences(text_seqs, maxlen=self.max_seq_len)
        return text_seqs