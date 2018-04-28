from geiger import transform


class MockVectorStore:
    embedding_lookup = {
        "ramu": [0.2, 0.4],
        "raj": [0.3, 0.5],
        "play": [0.4, 0.6],
        "basketball": [0.7, 0.8]
    }

    def get_vector(self, word, lang="en"):
        return self.embedding_lookup.get(word)

    def get_vectors(self, words, lang="en"):
        vectors = [self.get_vector(w) for w in words]
        vectors = [vec for vec in vectors if vec]
        if vectors:
            return vectors
        else:
            return None


def test_text_transformation():
    # Expectations
    max_seq_len = 100
    max_features = 10000

    embedding_size = 2

    embedding_lookup = MockVectorStore()

    texts = [
        "We catch the car every day.",
        "It rains every afternoon.",
        "They drive to New York every summer.",
        "Her mother is New York.",
        "Ramu, who will be 55 this year, jogs every day.",
        "Ramu never jogs.",
        "Does Ramu jog on Sundays?",
        "Raj works hard.",
        "Raj doesn't work hard at all!",
        "Jim builds houses for a living.",
        "What does Jim do for a living?",
        "They play basketball every Sunday.",
        "At what time, do you usually eat dinner?"
    ]
    transformer = transform.KerasTransformer(texts, max_features, max_seq_len)
    text_seq = transformer.texts_to_seq(["Raj drives her mother to New York", "Ramu plays basketball every summer"])
    embed_matrix = transformer.generate_embedding_matrix(embedding_lookup, embedding_size)
    mask = embed_matrix > 0
    assert len(text_seq) == 2, "We assume only 2 texts are processed."
    assert [t for t in text_seq[0] if t > 0] == [8, 17, 19, 4, 5], "Vocab items for tokens in first text"
    assert [t for t in text_seq[1] if t > 0] == [1, 25, 26, 18], "Vocab items for tokens in second text"
    assert set(embed_matrix[mask]) == {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
