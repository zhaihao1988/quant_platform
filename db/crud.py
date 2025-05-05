# database/crud.py
import logging
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional, Dict, Any

# Use correct path for models
from .models import StockDisclosure, StockList, StockFinancial, StockDaily, StockDisclosureChunk
# Import the centralized embedding function
from core.vectorizer import get_embedding

logger = logging.getLogger(__name__)

# Remove redundant embedding model loading from here

def get_stock_list_info(db: Session, symbol: str) -> Optional[StockList]:
    """Gets basic stock information (name, industry) from the StockList table."""
    logger.debug(f"Querying StockList for symbol: {symbol}")
    try:
        stock_info = db.query(StockList).filter(StockList.code == symbol).first()
        if not stock_info:
            logger.warning(f"No entry found in stock_list for code: {symbol}")
        return stock_info
    except Exception as e:
        logger.error(f"Error getting stock list info for {symbol}: {e}", exc_info=True)
        return None
def save_disclosure_chunk(db: Session, disclosure_id: int, chunk_order: int, chunk_text: str, vector: List[float]) -> bool:
    """Saves a single disclosure chunk and its vector."""
    try:
        db_chunk = StockDisclosureChunk(
            disclosure_id=disclosure_id,
            chunk_order=chunk_order,
            chunk_text=chunk_text,
            chunk_vector=vector
        )
        db.add(db_chunk)
        # db.commit() # 通常在调用者那里统一 commit
        logger.debug(f"Added chunk {chunk_order} for disclosure {disclosure_id} to session.")
        return True
    except Exception as e:
        # db.rollback() # 通常在调用者那里统一 rollback
        logger.error(f"Error saving disclosure chunk (Disclosure ID: {disclosure_id}, Order: {chunk_order}): {e}", exc_info=True)
        return False
def retrieve_relevant_disclosures(db: Session, symbol: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves relevant disclosure CHUNKS from the knowledge base using vector similarity search.
    Returns a list of dictionaries containing chunk info and original disclosure metadata.
    """
    logger.info(f"Retrieving relevant disclosure chunks for symbol '{symbol}' with query: '{query_text[:50]}...'")
    query_embedding = get_embedding(query_text, is_query=True)
    if not query_embedding:
        logger.error("Failed to generate query embedding.")
        return []

    try:
        # 查询 chunk 表，并通过 relationship 加载关联的 disclosure 信息
        similar_chunks = db.query(StockDisclosureChunk).options(
            joinedload(StockDisclosureChunk.disclosure) # 预加载关联的公告信息
        ).join(StockDisclosure).filter( # 确保只查询指定 symbol 的公告块
             StockDisclosure.symbol == symbol,
             StockDisclosureChunk.chunk_vector != None
        ).order_by(
            StockDisclosureChunk.chunk_vector.cosine_distance(query_embedding)
        ).limit(top_k).all()

        results = []
        if similar_chunks:
             logger.info(f"Found {len(similar_chunks)} relevant disclosure chunks in KB for query.")
             for chunk in similar_chunks:
                  results.append({
                       "chunk_text": chunk.chunk_text,
                       "chunk_order": chunk.chunk_order,
                       "disclosure_id": chunk.disclosure_id,
                       "title": chunk.disclosure.title, # 从关联对象获取
                       "ann_date": chunk.disclosure.ann_date # 从关联对象获取
                  })
        else:
             logger.info("No relevant disclosure chunks found.")

        return results

    except Exception as e:
        logger.error(f"Error during chunk vector search for {symbol}: {e}", exc_info=True)
        db.rollback() # <--- 发生查询错误时回滚
        return []

# Add other necessary CRUD operations here, e.g., for saving StockDaily, StockFinancial data if needed by other parts of the system.
# Example:
# def save_stock_daily_data(db: Session, data: List[Dict]): ...
# def save_stock_financial_data(db: Session, symbol: str, report_type: str, report_date: Date, data: Dict): ...