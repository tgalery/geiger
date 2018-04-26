from geiger.emoji import get_emoji_description
from geiger.emoji import is_emoji


def test_check_if_emoji():
    expected = {"😜": True, "a": False, "©": False}
    for char, exp in expected.items():
        assert is_emoji(char) == exp


def test_emoji_to_description():
    expected = {"😜": "face with stuck-out tongue and winking eye"}
    for emoji, exp in expected.items():
        assert get_emoji_description(emoji)
