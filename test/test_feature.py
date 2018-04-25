from geiger.feature import is_bad_word, get_sentiment_vector, get_pos_tag

def test_bad_words():
    # Expectations
    bad_words = ["anus"]
    for word in bad_words:
        assert is_bad_word(word)
        # Case shouldn't matter
        assert is_bad_word(word.title())

def test_word_sentiment():
    word = "hate"
    sentiment = get_sentiment_vector(word)
    assert sentiment[0] == -0.8
    assert sentiment[1] == 0.9

def test_word_pos_tag():
    word = "hate"
    pos_tag = get_pos_tag(word)
    assert pos_tag == 'NN'
