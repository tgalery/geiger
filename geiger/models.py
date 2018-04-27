from keras.models import Model
from keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D
from geiger import evaluate


def build_pooled_gru(num_classes, vocab_size, max_seq_len, embedding_matrix,
                     embedding_dims=300):
    """
    Build a token encoder model for a classification task.
    Args:
        num_classes: int: number of classes for classification
        vocab_size: int: vocab_size
        max_seq_len: int: max number of features or token types considered
        embedding_matrix: np.array: matrix of [word, word_vector]
        embedding_dims: int: word vector dimensions.

    Returns: keras.models.Model
    """
    sequence_input = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size,
                  embedding_dims,
                  weights=[embedding_matrix])(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(num_classes, activation="softmax")(conc)

    model = Model(inputs=sequence_input, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



