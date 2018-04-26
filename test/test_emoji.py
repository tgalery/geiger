from geiger.emoji import emoji_to_description
from geiger.emoji import tokenise_description
from geiger.emoji import check_if_emoji

def test_check_if_emoji():
	expected = {"😜" : True,"a" : False,"©" : False}
	for char, exp in expected.items():
		assert check_if_emoji(char) == exp

def test_emoji_to_description():
	expected = {"😜": "face with stuck-out tongue and winking eye"}
	for emoji, exp in expected.items():
		assert emoji_to_description(emoji)

def test_emoji_to_tokenised():
	expected = {"😜": ["face","with","stuck-out","tongue","and","winking","eye"]}
	for emoji, exp in expected.items():
		description = emoji_to_description(emoji)
		assert tokenise_description(description)