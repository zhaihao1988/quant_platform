# rag/embeddings.py
import logging
import torch
# --- 关键修复：导入 models 模块以手动构建 pipeline ---
from sentence_transformers import SentenceTransformer, models
from config.settings import settings, CORRECT_DIMENSION_1024

logger = logging.getLogger(__name__)

class Embedder:
    """
    A professional-grade wrapper for SentenceTransformer models.

    Handles model loading, device placement, batching, normalization,
    and provides warnings for input truncation.
    """
    def __init__(self):
        # 1. Device Selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            logger.info("CUDA device selected. Forcibly clearing CUDA cache before model loading...")
            try:
                torch.cuda.empty_cache()
                logger.info("torch.cuda.empty_cache() called successfully.")
            except Exception as e:
                logger.warning(f"Could not empty CUDA cache: {e}")
        
        # --- WORKAROUND: Hardcode model path to bypass faulty settings file ---
        model_path = "D:/project/quant_platform/models/Qwen3-Embedding-0.6B"
        logger.warning(f"WORKAROUND: Bypassing settings.EMBEDDING_MODEL_NAME. Using hardcoded path: {model_path}")
        # --- End of WORKAROUND ---

        # --- 增强日志：明确打印出将要使用的配置 ---
        logger.info("--- Embedding Model Configuration ---")
        logger.info(f"Model Name from settings (IGNORED): '{settings.EMBEDDING_MODEL_NAME}'")
        logger.info(f"Model Name being used (HARDCODED): '{model_path}'")
        logger.info(f"Expected Dimension (from CORRECT_DIMENSION_1024): {CORRECT_DIMENSION_1024}")
        logger.info(f"Device: '{device}'")
        logger.info("------------------------------------")

        # --- 关键修复：手动构建模型 pipeline 以支持半精度加载 ---
        try:
            # Step 1: Create the Transformer layer with half-precision arguments
            model_args = {'trust_remote_code': True}
            if device == 'cuda':
                logger.info("Attempting to load Transformer layer in half-precision (float16)...")
                model_args['torch_dtype'] = torch.float16
            
            word_embedding_model = models.Transformer(
                model_path, # <-- Use the hardcoded path
                model_args=model_args
            )

            # Step 2: Create a pooling layer (mean pooling is a standard choice)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

            # Step 3: Create the final SentenceTransformer pipeline
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_path}'. Error: {e}", exc_info=True)
            raise

        # 3. Store model properties for later use (e.g., truncation checks)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.get_max_seq_length()
        self.actual_dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedder initialized successfully.")
        logger.info(f"Model's max sequence length: {self.max_seq_length} tokens.")
        logger.info(f"Model's actual output dimension: {self.actual_dimension}")
        
        # --- 验证维度配置 ---
        if self.actual_dimension != CORRECT_DIMENSION_1024:
            logger.warning(
                f"CRITICAL MISMATCH: Actual model dimension ({self.actual_dimension}) "
                f"does NOT match CORRECT_DIMENSION_1024 ({CORRECT_DIMENSION_1024}). "
                "Please update CORRECT_DIMENSION_1024 in your settings.py file if model has changed."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encodes a list of texts into normalized embeddings, ready for vector search.
        """
        # --- Truncation Warning (Data Quality Assurance) ---
        for i, text in enumerate(texts):
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if token_count > self.max_seq_length:
                logger.warning(
                    f"Input at index {i} (len: {len(text)} chars, ~{token_count} tokens) "
                    f"exceeds model's max length ({self.max_seq_length}). "
                    "It will be truncated, potentially losing information."
                )

        # --- Batch Encoding with Normalization (Performance and Accuracy) ---
        batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 32)
        logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}...")

        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=True,  # CRITICAL for accurate cosine similarity
            show_progress_bar=False     
        )

        return embeddings.tolist()

# MODEL_NAME constant can be removed from here if always using settings
# MODEL_NAME = "all-MiniLM-L6-v2" # This would be superseded by settings.EMBEDDING_MODEL_NAME
