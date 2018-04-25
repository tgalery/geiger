from geiger.feature import is_bad_word


def test_bad_words():
    # Expectations
    bad_words = ["anus"]
    for word in bad_words:
        assert is_bad_word(word)
        # Case shouldn't matter
        assert is_bad_word(word.title())
