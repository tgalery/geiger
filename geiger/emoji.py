import unicodedata
import re

def check_if_emoji(text):
	"""args: string
	returns: True if string is an emoji"""
	emoji_exp = re.compile('[\U0001F602-\U0001F64F]')
	m = re.match(emoji_exp,text)
	if re.match(emoji_exp,text):
		is_emoji = True
	else:
		is_emoji = False
	return is_emoji

def emoji_to_description(emoji):
	"""args: emoji
	returns: description in lower case
	"""
	return unicodedata.name(emoji).lower()

def tokenise_description(string):
	"""args: string, description of emoji
	returns: list, tokenised string
	"""
	return string.split(" ")