import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


class DataPhaser(object):
    def __init__(self, train_path, dev_path, col_names=None):
        self.corp_dir = 'datasets/'
        self.train_path = train_path
        self.dev_path = dev_path
        self.col_names = ['Code', 'Body', 'Tag'] if col_names is None else col_names
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopWords = set(stopwords.words('english'))

    def load_files(self):
        train = pd.read_csv(self.corp_dir + self.train_path, names=self.col_names)
        dev = pd.read_csv(self.corp_dir + self.train_path, names=self.col_names)
        return train, dev

    def split_sets(self, train_set, dev_set, tokenize=False):
        X_train = train_set[self.col_names[1]]
        Y_train = train_set[self.col_names[2]]

        X_val = dev_set[self.col_names[1]]
        Y_val = dev_set[self.col_names[2]]

        if tokenize is True:
            X_train = X_train.apply(self.tokenize)
            X_val = X_val.apply(self.tokenize)
        return X_train, X_val, Y_train, Y_val

    def tokenize(self, text):
        return pd.DataFrame([self.lemmatizer.lemmatize(token) for token in self.tokenizer.tokenize(text)
                            if token not in self.stopWords])


###############
#Example of how to use
################
#     getdata = DataPhaser(train_path='agr_en_train.csv', dev_path='agr_en_dev.csv')
#     train_set, dev_set = getdata.load_files()
#     X_train, X_val, Y_train, Y_val = getdata.split_sets(train_set, dev_set, tokenize=True)
#     print(X_train[0])
