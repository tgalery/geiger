import pandas as pd


class DataPhaser(object):
    def __init__(self, train_path, dev_path, col_names=None):
        self.corp_dir = 'datasets/'
        self.train_path = train_path
        self.dev_path = dev_path
        self.col_names = ['Code', 'Body', 'Tag'] if col_names is None else col_names

    def load_files(self):
        train = pd.read_csv(self.corp_dir + self.train_path, names=self.col_names)
        dev = pd.read_csv(self.corp_dir + self.dev_path, names=self.col_names)

        X_train = train[self.col_names[1]]
        Y_train = train[self.col_names[2]]

        X_val = dev[self.col_names[1]]
        Y_val = dev[self.col_names[2]]
        return X_train, X_val, Y_train, Y_val
