from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return embeddings

    def encode_query(self, query):

        embedding = self.model.encode([query])[0]

        return embedding