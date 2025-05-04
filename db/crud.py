# db/crud.py
import logging
from sqlalchemy.orm import Session
# 使用您正确的模型路径
from .models import StockDisclosure, StockList # 导入 StockList
# 假设向量嵌入模型在此处加载或从 vectorizer 模块导入
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- 嵌入模型加载 (应与 scraper 中一致，或从公共模块导入) ---
embedding_model_name_crud = 'shibing624/text2vec-base-chinese'
embedding_dim_crud = 768 # 确保维度一致
try:
    embedding_model_crud = SentenceTransformer(embedding_model_name_crud)
    loaded_dim_crud = embedding_model_crud.get_sentence_embedding_dimension()
    if loaded_dim_crud != embedding_dim_crud:
        logger.warning(f"[CRUD] Model '{embedding_model_name_crud}' dimension ({loaded_dim_crud}) does not match expected dimension ({embedding_dim_crud})!")
    logger.info(f"Loaded embedding model '{embedding_model_name_crud}' in crud.")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model in crud: {e}")
    embedding_model_crud = None

def get_text_embedding_crud(text: str) -> list[float] | None:
    """为查询文本生成嵌入"""
    if not embedding_model_crud or not text:
        return None
    try:
        # 考虑查询文本的长度限制（虽然通常较短）
        embedding = embedding_model_crud.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None
# --- 嵌入逻辑结束 ---

def get_stock_list_info(db: Session, symbol: str) -> StockList | None: # 重命名函数并使用 StockList
    """获取股票列表中的基本信息 (如名称)"""
    try:
        # 使用 code 字段查询
        stock_info = db.query(StockList).filter(StockList.code == symbol).first()
        if not stock_info:
            logger.warning(f"No entry found in stock_list for code: {symbol}")
        return stock_info
    except Exception as e:
        logger.error(f"Error getting stock list info for {symbol}: {e}")
        return None

def retrieve_relevant_disclosures(db: Session, symbol: str, query_text: str, top_k: int = 5) -> list[StockDisclosure]:
    """
    从本地知识库检索与查询文本最相关的公告内容。
    使用 content_vector 进行搜索，返回包含 raw_content 的对象。
    """
    if not embedding_model_crud:
        logger.error("Embedding model not loaded, cannot perform vector search.")
        return []

    query_embedding = get_text_embedding_crud(query_text)
    if not query_embedding:
        logger.error("Failed to generate embedding for the query text.")
        return []

    try:
        logger.info(f"Performing vector search for symbol '{symbol}' with query '{query_text}'")
        # 使用 content_vector 和余弦距离进行搜索
        similar_disclosures = db.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            StockDisclosure.content_vector != None # 确保向量存在
        ).order_by(
            StockDisclosure.content_vector.cosine_distance(query_embedding) # type: ignore
        ).limit(top_k).all()

        logger.info(f"Found {len(similar_disclosures)} relevant disclosures in KB.")
        # 返回完整的 StockDisclosure 对象，包含 raw_content 和 ann_date
        return similar_disclosures

    except Exception as e:
        logger.error(f"Error during vector search for {symbol} with query '{query_text}': {e}")
        return []

# --- 其他可能的 CRUD 函数 ---