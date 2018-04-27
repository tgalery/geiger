from os import path

from geiger.utils import load_word_vectors

REPO_ROOT = path.abspath(__file__).rsplit("/geiger/", 1)[0]


class MultiLangVectorStore:

    vector_locs = {
        "en": path.join(REPO_ROOT, "resources/wiki-news-300d-1M-subword.vec"),
        "hi": path.join(REPO_ROOT, "resources/wiki.hi.vec"),
    }

    transform_locs = {
        "en": path.join(REPO_ROOT, "geiger/libs/fastText_multilingual/alignment_matrices/en.txt"),
        "hi": path.join(REPO_ROOT, "geiger/libs/fastText_multilingual/alignment_matrices/hi.txt"),
    }

    supported_langs = ["en", "hi"]

    vector_stores = dict()

    def __init__(self):

        for lang in self.supported_langs:
            vec_path = self.vector_locs[lang]
            transform_path = self.transform_locs[lang]
            self.vector_stores[lang] = load_word_vectors(vec_path, transform_fpath=transform_path)

    def get_vector(self, word, lang):
        if lang not in self.supported_langs:
            raise ValueError("No support for language {}".format(lang))
        try:
            return self.vector_stores[lang][word]
        except KeyError:
            return None

    def get_vectors(self, words, lang):
        vecs = []
        for word in words:
            try:
                word_vec = self.get_vector(word, lang)
                if word_vec:
                    vecs.append(word_vec)
            except ValueError:
                pass
        return vecs
