# rag/embeddings.py

from sentence_transformers import SentenceTransformer

# 选用轻量高效模型
MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False)
