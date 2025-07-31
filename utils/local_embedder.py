from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def name(self):
        return "local-sbert"
