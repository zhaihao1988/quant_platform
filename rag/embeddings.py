# rag/embeddings.py
from sentence_transformers import SentenceTransformer
from config.settings import settings # Import settings

class Embedder:
    def __init__(self):
        # Use the model name from settings
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        # You might want to log which model is being loaded
        print(f"Embedder initialized with model: {settings.EMBEDDING_MODEL_NAME}")


    def embed(self, texts: list[str]) -> list[list[float]]:
        # The encode method might also benefit from settings.EMBEDDING_MAX_LENGTH
        # if the model supports a max_seq_length parameter during encoding,
        # but SentenceTransformer usually handles this via its tokenizer's max_length.
        return self.model.encode(texts, show_progress_bar=False)

# MODEL_NAME constant can be removed from here if always using settings
# MODEL_NAME = "all-MiniLM-L6-v2" # This would be superseded by settings.EMBEDDING_MODEL_NAME
