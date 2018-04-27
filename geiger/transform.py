"""
Module for transforming sequences of text into Network Inputs
"""
import numpy as np
from keras.preprocessing import text, sequence
import tweetokenize as tt
from geiger.emoji import is_emoji, get_emoji_description
from textblob import Word
from tqdm import tqdm

tokenizer = tt.Tokenizer()
def fix_repeated_token (word):
    if len(set(word)) == 1:
        return word[0]
    else:
        return word

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

        self.max_features = max_features
        self.max_seq_len = max_seq_len
        self.tokenizer = text.Tokenizer(num_words=max_features)
        self.build_vocabulary([self.preprocess_text(t) for t in x_texts])
        self.rel_features = min(self.max_features, len(self.tokenizer.word_index) + 1)

    def build_vocabulary(self, x_texts):
        """
        Fit text to tokenizer. Note both train and test sets should be used here.
        Args:
            x_texts: list: of texts to be transformed

        Returns: None

        """
        self.tokenizer.fit_on_texts(x_texts)

    @staticmethod
    def preprocess_text(text):
        tokens = tokenizer.tokenize(text)
        tokens = [ fix_repeated_token(token) for token in tokens ]
        return ' '.join(tokens)

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

    def handle_emoji(self, word, embedding_lookup):
        description = get_emoji_description(word)
        embedding_vectors = [
            embedding_lookup.get(token)
            for token in tokenizer.tokenize(description)]
        embedding_vectors = [ x for x in embedding_vectors if x is not None ]
        return np.mean(embedding_vectors, axis = 0)

    def generate_embedding_matrix(self, embedding_lookup, embedding_size):
        """
        Generate an embedding matrix [vocab_size, n_dimensions]

        Args:
            embedding_lookup: dict: like interface that implements .get_vector and .get_vectors methods
            embedding_size: int: dimensionality of embedding

        Returns: np.array
        """
        unhandled = []
        embedding_matrix = np.zeros((self.rel_features, embedding_size))
        for word, i in tqdm(self.tokenizer.word_index.items()):
            if i >= self.rel_features:
                return embedding_matrix
            else:
                lang = "en"
                embedding_vector = embedding_lookup.get_vector(word, lang)

                if embedding_vector is None and is_emoji(word):
                    embedding_vector = self.handle_emoji(word, embedding_lookup)

                # Try lemmatization
                if embedding_vector is None:
                    word = Word(word).lemmatize()
                    embedding_vector = embedding_lookup.get(word)

                # Try spellcheck
                # if embedding_vector is None:
                #     word = Word(word).correct()
                #     embedding_vector = embedding_lookup.get(word)

                if embedding_vector is not None:
                    # TODO, we need to feature augumentation here if we are feeding this to the neuronet
                    embedding_matrix[i] = embedding_vector
                else:
                    # print("Word {} is out of vocabulary.".format(word))
                    unhandled.append(word)
                    # Todo we are setting the vectors of <UNK> as zeros, maybe there's a better way
                    continue
        print("{} words were out of vocabulary.".format(len(unhandled)))
        return embedding_matrix
