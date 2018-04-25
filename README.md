#### Instructions
For WordBlob based features to work we need to run

    python3
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

For spacy to work we need to load the 'en' model by doing

    python3 -m spacy download en (In this one vectors are of shape (384,))

    We can also download this one for vectors of (300,)

    python3 -m spacy download en_core_web_lg

    See [here](https://github.com/explosion/spaCy/issues/1657)
