from geiger.translation import translate_to_target
from googletrans import Translator

def test_translate_to_target():
	expected = {"fr": "Salut Monde","it": "Ciao Mondo", "nl": "Hallo Wereld", "de": "Hallo Welt"}
	text = "Hello World"
	src = "en"
	translator = Translator()
	for lang, exp in expected.items():
		assert translate_to_target(translator, text, src=src, dest=lang)