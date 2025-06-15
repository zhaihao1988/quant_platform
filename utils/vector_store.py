# utils/vector_store.py
import logging
from rag.embeddings import Embedder

logger = logging.getLogger(__name__)

# --- 全局的、从中央配置加载的 Embedder 实例 ---
try:
    embedder_instance = Embedder()
    logger.info("Successfully initialized shared embedder instance in vector_store.py.")
except Exception as e:
    logger.error("Failed to initialize shared embedder instance in vector_store.py: %s", e, exc_info=True)
    embedder_instance = None


def embed_text(text: str) -> list[float] | None:
    """
    【V2 已升级】将输入文本编码为向量。
    该函数现在使用从 rag.embeddings 导入的、由中央配置驱动的 Embedder。
    """
    if not embedder_instance:
        logger.error("Embedder is not available. Cannot embed text.")
        return None
        
    if not text or not isinstance(text, str):
        logger.warning("embed_text received empty or invalid input.")
        return None

    # 3. 使用我们的 Embedder 实例进行编码
    # 我们的 embed 方法接收一个列表，所以我们将单个文本放入列表中
    embeddings = embedder_instance.embed([text])
    
    # 返回第一个（也是唯一一个）结果
    return embeddings[0] if embeddings else None
