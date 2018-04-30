"""
Module for transforming sequences of text into Network Inputs
"""
import numpy as np
from keras.preprocessing import text, sequence
import tweetokenize as tt
from geiger.emoji import is_emoji, get_emoji_description
from geiger.utils import is_devanagari, generate_n_grams
from textblob import Word
from tqdm import tqdm
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string

TOKENIZER = tt.Tokenizer()


def fix_repeated_token(word):
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
        tokens = TOKENIZER.tokenize(text.replace("'", " ' "))
        tokens = [Word(fix_repeated_token(token)).lemmatize() for token in tokens]
        return ' '.join([tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and
                         len(tok) > 1 and not all([char.isdigit() or char in [",.:-[]'`"] for char in tok]) and
                         not is_devanagari(tok)
                         ])

    def texts_to_seq(self, texts, pad=True):
        """
        Transform a text sequence to the proper sequential representation.
        Args:
            texts: list: of strings representing textual seq
            pad: bool: if sequences should be padded

        Returns: list of sequences with the right padding
        """
        text_seqs = self.tokenizer.texts_to_sequences([self.preprocess_text(t) for t in texts])
        if pad and self.max_seq_len > 0:
            return sequence.pad_sequences(text_seqs, maxlen=self.max_seq_len)
        return text_seqs


    @staticmethod
    def handle_oov_tokens(words, embedding_lookup):
        embedding_vectors = embedding_lookup.get_vectors(words, "en")
        if embedding_vectors:
            return np.mean(embedding_vectors, axis=0)

    def handle_emoji(self, word, embedding_lookup):
        description = get_emoji_description(word)
        if description:
            normal_description = self.preprocess_text(description)
            word_tokens = [w.strip() for w in normal_description if w.strip()]
            return self.handle_oov_tokens(word_tokens, embedding_lookup)
        return None

    def generate_embedding_matrix(self, embedding_lookup, embedding_size, feat_augmenter=None):
        """
        Generate an embedding matrix [vocab_size, n_dimensions]

        Args:
            embedding_lookup: dict: like interface that implements .get_vector and .get_vectors methods
            embedding_size: int: dimensionality of embedding

        Returns: np.array
        """
        unhandled = []
        if feat_augmenter:
            embedding_size += feat_augmenter.n_features
        embedding_matrix = np.zeros((self.rel_features, embedding_size))
        for word, i in tqdm(self.tokenizer.word_index.items()):
            if i >= self.rel_features:
                continue
            else:

                embedding_vector = embedding_lookup.get_vector(word, "en")
                word_emoji = is_emoji(word)
                word_devanagari = is_devanagari(word)

                if embedding_vector is None and word_emoji:
                    embedding_vector = self.handle_emoji(word, embedding_lookup)

                # if embedding_vector is None and not word_emoji:
                #     if word_devanagari:
                #         embedding_vector = embedding_lookup.get_vector(word, "hi")

                # Try generating char_ngrams
                if embedding_vector is None and not word_emoji and not word_devanagari:
                    char_ngrams = generate_n_grams(word)
                    embedding_vector = self.handle_oov_tokens(char_ngrams, embedding_lookup)

                if embedding_vector is not None:
                    # TODO, we need to feature augumentation here if we are feeding this to the neuronet
                    if feat_augmenter:
                        extra_feats = feat_augmenter.get_features(word)
                        embedding_vector = np.concatenate((embedding_vector, extra_feats))
                    embedding_matrix[i] = embedding_vector
                else:
                    print("Could not find vector for word {}.".format(word))
                    unhandled.append(word)
                    # Todo we are setting the vectors of <UNK> as zeros, maybe there's a better way
                    continue
        print("{} words were out of vocabulary.".format(len(unhandled)))
        return embedding_matrix, embedding_size
