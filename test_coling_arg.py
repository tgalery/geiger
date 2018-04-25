import pandas as pd
import coling_arg


corp_dir = 'datasets/'
train_path = 'agr_en_train.csv'
dev_path = 'arg_en_dev.csv'
col_names = ['Code', 'Body', 'Tag']
test_func = coling_arg.DataPhaser(train_path=train_path, dev_path=dev_path, col_names=col_names)


def load_files():
    train = pd.read_csv(corp_dir + train_path, names=col_names)
    dev = pd.read_csv(corp_dir + train_path, names=col_names)
    return train[col_names[1]], dev[col_names[1]], train[col_names[2]], dev[col_names[2]]


X_train, X_val, Y_train, Y_val = test_func.load_files()
test_X_train, test_X_val, test_Y_train, test_Y_val = load_files()

assert X_train == test_X_train
assert X_val == test_X_val
assert Y_val == test_Y_val
assert Y_train == test_Y_train
