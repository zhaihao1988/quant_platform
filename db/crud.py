# database/crud.py
import logging
from sqlalchemy.orm import Session
from typing import List, Optional

# Use correct path for models
from .models import StockDisclosure, StockList, StockFinancial, StockDaily
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

def retrieve_relevant_disclosures(db: Session, symbol: str, query_text: str, top_k: int = 5) -> List[StockDisclosure]:
    """
    Retrieves relevant disclosures from the knowledge base using vector similarity search.
    """
    logger.info(f"Retrieving relevant disclosures for symbol '{symbol}' with query: '{query_text[:50]}...'")

    # Generate embedding for the query text using the centralized vectorizer
    query_embedding = get_embedding(query_text, is_query=True)

    if not query_embedding:
        logger.error("Failed to generate embedding for the query text. Cannot perform vector search.")
        return []

    try:
        # Perform vector search using cosine distance (or other distance metric)
        # Ensure the vector column and query embedding dimension match
        similar_disclosures = db.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            StockDisclosure.content_vector != None # Only search items with vectors
        ).order_by(
            StockDisclosure.content_vector.cosine_distance(query_embedding) # Assumes cosine distance is preferred
        ).limit(top_k).all()

        logger.info(f"Found {len(similar_disclosures)} relevant disclosures in KB for query.")
        return similar_disclosures

    except Exception as e:
        # Catch potential errors from the vector operation or database query
        logger.error(f"Error during vector search for {symbol} with query '{query_text[:50]}...': {e}", exc_info=True)
        return []

def update_disclosure_content_vector(db: Session, disclosure_id: int, text_content: Optional[str], vector: Optional[List[float]]):
    """
    Updates the raw_content and content_vector for a specific disclosure.
    Handles potential errors during the update.
    Note: This commits changes immediately for the single disclosure.
          For batch processing, consider committing outside this function.
    """
    logger.debug(f"Updating disclosure ID: {disclosure_id}")
    try:
        disclosure = db.query(StockDisclosure).filter(StockDisclosure.id == disclosure_id).first()
        if not disclosure:
            logger.error(f"Disclosure with ID {disclosure_id} not found for update.")
            return False

        update_fields = {}
        if text_content is not None:
            disclosure.raw_content = text_content
            update_fields['raw_content'] = f"{len(text_content)} chars" if text_content else "empty"
        if vector is not None:
            disclosure.content_vector = vector
            update_fields['content_vector'] = f"vector[{len(vector)}]" if vector else "empty"

        if update_fields:
            db.add(disclosure) # Add to session for update
            # db.commit() # Commit immediately - remove if batching
            logger.info(f"Updated fields {list(update_fields.keys())} for disclosure ID {disclosure_id}.")
            return True
        else:
            logger.warning(f"No content or vector provided to update disclosure ID {disclosure_id}.")
            return False

    except Exception as e:
        # db.rollback() # Rollback if commit was inside try block
        logger.error(f"Error updating disclosure ID {disclosure_id}: {e}", exc_info=True)
        return False

# Add other necessary CRUD operations here, e.g., for saving StockDaily, StockFinancial data if needed by other parts of the system.
# Example:
# def save_stock_daily_data(db: Session, data: List[Dict]): ...
# def save_stock_financial_data(db: Session, symbol: str, report_type: str, report_date: Date, data: Dict): ...