from coling_arg import DataPhaser

test_func = DataPhaser(train_path='agr_en_train.csv', dev_path='agr_en_train.csv', col_names=['Code', 'Body', 'Tag'])

def test():
    X_train, X_val, Y_train, Y_val = test_func.load_files()

    assert 'The quality of re made now makes me think it is something to be bought from fish market' in X_val[0]
    assert 'Well said sonu..you have courage to stand against dadagiri of Muslims' in X_train[0]

    assert Y_train == 'OAG'
    assert Y_val == 'CAG'

    assert len(X_val) == 3001
    assert len(Y_val) == 3001
    assert len(X_train) == 11999
    assert len(Y_train) == 11999
