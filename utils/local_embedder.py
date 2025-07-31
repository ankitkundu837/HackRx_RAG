from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def name(self) -> str:
        return "all-MiniLM-L6-v2"
