import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import argparse

parser = argparse.ArgumentParser(description='The geiger programm')
parser.add_argument('train', type=str, help='The csv with the input data')
parser.add_argument('test', type=str, help='The csv with the test data')
parser.add_argument('embeddings', type=str, help='The embeddings file (e.g. fasttext)')
parser.add_argument('--features', type=int, default=30000, help='The numbers of features of the embedding file. (in the first line of the file)')
parser.add_argument('--maxlen', type=int, default=100, help='The length of the max sequence')
parser.add_argument('--embed_size', type=int, default=300, help='The size of the embedding layer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size ')
parser.add_argument('--epochs', type=int, default=2, help='Total number of epochs to run for')

args = parser.parse_args()


# EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

def load_files():
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X_train = train["comment_text"].fillna("fillna").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_test = test["comment_text"].fillna("fillna").values

    tokenizer = text.Tokenizer(num_words=args.features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=args.maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=args.maxlen)
    return x_train, x_test, y_train, tokenizer

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(tokenizer):
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(args.embeddings))

    word_index = tokenizer.word_index
    nb_words = min(args.features, len(word_index))
    embedding_matrix = np.zeros((nb_words, args.embed_size))
    for word, i in word_index.items():
        if i >= args.features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model(embedding_matrix):
    inp = Input(shape=(args.maxlen, ))
    x = Embedding(args.features, args.embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def validation_analysis(X_tra, y_val):
    #produce comparison file for validation dataset
    text = [map_idx_to_words(tokenizer, words) for words in X_val]
    #
    y_val_pred = model.predict(X_val)
    y_val_1 = y_val_pred.copy()
    #threshold the probabilities at 0.5
    y_val_1[y_val_pred < 0.5] = 0
    y_val_1[y_val_pred > 0.5] = 1
    d = pd.DataFrame(0, index=np.arange(X_val.shape[0]), columns=['text','p_toxic','p_severe_toxic','p_obscene','p_threat','p_insult','p_identity_hate','toxic','severe_toxic','obscene','threat','insult','identity_hate', 'disagree'])
    d[['text']] = text
    d[["p_toxic", "p_severe_toxic", "p_obscene", "p_threat", "p_insult", "p_identity_hate"]] = y_val_1
    d[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_val
    
    truth_matrix = y_val_1 == y_val
    equal_all = []
    for i in range(truth_matrix.shape[0]):
        equal_all.append(not(all(truth_matrix[i])))
    d[['disagree']] = equal_all
    d.to_csv('validation_false_preds.csv')


def map_idx_to_words(tokenizer, words):
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return " ".join([reverse_word_map[i] for i in words if i != 0])

if __name__ == '__main__':
    x_train, x_test, y_train, tokenizer = load_files()
    embedding_matrix = load_embeddings(tokenizer)

    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

    model = get_model(embedding_matrix)

    hist = model.fit(X_tra, y_tra, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val),
                    callbacks=[RocAuc], verbose=2)

    y_pred = model.predict(x_test, batch_size=1024)
    submission = pd.read_csv('~/Documents/kaggle/toxic_comments/sample_submission.csv')

    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv('submission.csv', index=False)

