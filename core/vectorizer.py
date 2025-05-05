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

def split_text_into_chunks(text: str, chunk_size: int = 450, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks based on character count and sentence boundaries.

    Args:
        text: The input text.
        chunk_size: The target size for each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # Basic sentence splitting using common delimiters
    sentences = re.split(r'(?<=[。！？\n\.\?!])\s*', text)
    sentences = [s for s in sentences if s] # Remove empty strings

    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if not current_chunk: # Start a new chunk
             current_chunk = sentence
             current_length = sentence_length
        elif current_length + sentence_length <= chunk_size: # Add to current chunk
            current_chunk += " " + sentence
            current_length += sentence_length + 1 # Add 1 for space
        else: # Current chunk is full, start a new one
            chunks.append(current_chunk)
            # Create overlap
            overlap_start_index = max(0, len(current_chunk) - chunk_overlap)
            # Try to find a sentence boundary within the overlap window for cleaner cuts
            overlap_text = current_chunk[overlap_start_index:]
            sentence_end_in_overlap = re.search(r'[。！？\n\.\?!]', overlap_text)
            if sentence_end_in_overlap:
                 overlap_point = overlap_start_index + sentence_end_in_overlap.end()
                 next_chunk_start_text = current_chunk[overlap_point:].lstrip()
            else:
                 # If no sentence end found, just use the character overlap
                 next_chunk_start_text = current_chunk[len(current_chunk) - chunk_overlap:].lstrip()

            # Start new chunk with overlap and the new sentence
            current_chunk = next_chunk_start_text + " " + sentence if next_chunk_start_text else sentence
            current_length = len(current_chunk)


    if current_chunk: # Add the last chunk
        chunks.append(current_chunk)

    # Further refinement: ensure no chunk significantly exceeds chunk_size (optional)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.2: # If a single sentence chunk is too large
             # Simple split if a chunk is still too large
             start = 0
             while start < len(chunk):
                  end = min(start + chunk_size, len(chunk))
                  final_chunks.append(chunk[start:end])
                  start += chunk_size - chunk_overlap # Move forward with overlap
        else:
             final_chunks.append(chunk)


    logger.info(f"Split text (length: {len(text)}) into {len(final_chunks)} chunks.")
    return final_chunks
