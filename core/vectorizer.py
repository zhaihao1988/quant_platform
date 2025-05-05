# core/vectorizer.py
import logging
from sentence_transformers import SentenceTransformer
from config.settings import settings # Import settings

logger = logging.getLogger(__name__)

# --- Configuration from settings ---
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
EMBEDDING_DIM = settings.EMBEDDING_DIM
MAX_MODEL_INPUT_LENGTH = settings.EMBEDDING_MAX_LENGTH # Max sequence length for the embedding model

# --- Global Model Loading ---
embedding_model = None
try:
    # Load the model specified in settings
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    loaded_dim = embedding_model.get_sentence_embedding_dimension()

    # Validate dimension
    if loaded_dim != EMBEDDING_DIM:
        logger.error(f"CRITICAL: Loaded model '{EMBEDDING_MODEL_NAME}' dimension ({loaded_dim}) "
                     f"does not match configured EMBEDDING_DIM ({EMBEDDING_DIM})! Check your model and settings.")
        # Decide how to handle mismatch, e.g., raise an error or disable embedding
        embedding_model = None # Disable embedding if dimensions mismatch critically
        # raise ValueError("Embedding model dimension mismatch!")
    else:
        logger.info(f"Successfully loaded embedding model '{EMBEDDING_MODEL_NAME}' (dimension: {loaded_dim}).")

except Exception as e:
    logger.error(f"Failed to load sentence transformer model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    embedding_model = None # Ensure model is None if loading fails

def get_embedding(text: str, is_query: bool = False) -> list[float] | None:
    """
    Generates an embedding vector for the given text.

    Args:
        text: The input text to embed.
        is_query: Flag indicating if the text is a short query (True) or a longer document (False).
                  Currently affects only logging, but could influence future chunking/processing.

    Returns:
        A list of floats representing the embedding vector, or None if embedding fails.
    """
    if not embedding_model:
        logger.error("Embedding model is not loaded. Cannot generate embedding.")
        return None
    if not text:
        logger.warning("Input text for embedding is empty.")
        return None

    try:
        # Simple truncation based on estimated character limit derived from max token length
        # NOTE: This is a basic approach. Proper token-based truncation or chunking is recommended for production.
        # Using a rough multiplier (e.g., 2-3 chars per token)
        char_limit_multiplier = 2.5
        estimated_char_limit = int(MAX_MODEL_INPUT_LENGTH * char_limit_multiplier)

        if not is_query and len(text) > estimated_char_limit:
            processed_text = text[:estimated_char_limit]
            logger.warning(f"Document text truncated for embedding "
                         f"(original: {len(text)} chars, limit: ~{estimated_char_limit} chars, truncated: {len(processed_text)} chars). "
                         f"Consider implementing text chunking.")
        else:
            processed_text = text # No truncation for queries or short docs

        # Generate embedding using the loaded SentenceTransformer model
        # convert_to_tensor=False returns a numpy array, which is converted to list
        embedding_vector = embedding_model.encode(processed_text, convert_to_tensor=False)
        return embedding_vector.tolist()

    except Exception as e:
        logger.error(f"Error generating embedding for text snippet '{text[:100]}...': {e}", exc_info=True)
        return None

# Future Enhancement: Add text chunking logic here if needed
# def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
#    ... implementation using libraries like LangChain's text_splitter ...