# data_processing/loader.py
import logging
import pandas as pd
from sqlalchemy import text, or_, and_, not_
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

# Use correct path for models and database session
from db.models import StockDisclosure, StockDaily, StockFinancial
from db.database import get_engine_instance  # Use engine instance for pandas

logger = logging.getLogger(__name__)
engine = get_engine_instance()

# Keywords for filtering disclosures
DISCLOSURE_KEYWORDS = ['年度报告', '半年度报告', '调研', '股权激励', '回购']

def load_price_data(symbol: str, window: int = 90) -> Optional[pd.DataFrame]:
    """Loads recent daily price data for a given stock symbol."""
    if engine is None:
        logger.error("Database engine not available for loading price data.")
        return None
    logger.info(f"Loading price data for {symbol}, window={window} days.")
    sql = text("""
      SELECT date, open, close, high, low, volume, pct_change, amount, turnover
      FROM stock_daily
      WHERE symbol = :symbol
      ORDER BY date DESC
      LIMIT :limit
    """)
    try:
        # Ensure correct parameter binding
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection, params={"symbol": symbol, "limit": window})
        if df.empty:
            logger.warning(f"No price data found for symbol {symbol} within the last {window} days.")
            return None
        # Reverse to have chronological order for analysis
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}", exc_info=True)
        return None

def load_financial_data(symbol: str, report_type: str) -> Optional[Dict[str, Any]]:
    """Loads the latest financial report data (JSONB) for a given type."""
    # Valid report types might be 'benefit', 'balance', 'cashflow' etc.
    if engine is None:
        logger.error("Database engine not available for loading financial data.")
        return None
    logger.info(f"Loading latest '{report_type}' financial data for {symbol}.")
    sql = text("""
      SELECT data
      FROM stock_financial
      WHERE symbol = :symbol AND report_type = :rtype
      ORDER BY report_date DESC
      LIMIT 1
    """)
    try:
        # Ensure correct parameter binding
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection, params={"symbol": symbol, "rtype": report_type})
        if df.empty or 'data' not in df.columns or df['data'].iloc[0] is None:
            logger.warning(f"No financial data found for symbol {symbol}, report_type '{report_type}'.")
            return None
        # Assuming the 'data' column stores JSON
        return df["data"].iloc[0]
    except Exception as e:
        logger.error(f"Error loading financial data for {symbol} (type: {report_type}): {e}", exc_info=True)
        return None

def get_disclosures_needing_content(db: Session, symbol: str) -> List[StockDisclosure]:
    """
    Finds disclosures for a symbol that match keywords and haven't been processed yet
    (raw_content is NULL).
    """
    logger.info(f"Querying database for disclosures needing content for symbol: {symbol}")
    try:
        # Split keywords into time-sensitive and non-time-sensitive
        non_time_keywords = ['年度报告', '半年度报告']
        time_keywords = ['调研', '股权激励', '回购']
        one_year_ago = datetime.now() - timedelta(days=365)

        # Build filters for non-time-sensitive keywords
        non_time_filters = [StockDisclosure.title.ilike(f'%{kw}%') for kw in non_time_keywords]

        # Build filters for time-sensitive keywords with date constraint
        time_filters = [
            and_(
                StockDisclosure.title.ilike(f'%{kw}%'),
                StockDisclosure.ann_date >= one_year_ago
            )
            for kw in time_keywords
        ]

        # Combine all filters with OR
        combined_filters = or_(*(non_time_filters + time_filters))

        # Query disclosures with all conditions
        disclosures = db.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            StockDisclosure.raw_content == None,
            not_(StockDisclosure.title.ilike('%摘要%')),
            combined_filters
        ).order_by(StockDisclosure.ann_date.desc()).all()

        logger.info(f"Found {len(disclosures)} disclosures needing content for {symbol}.")
        return disclosures
    except Exception as e:
        logger.error(f"Error querying disclosures needing content for {symbol}: {e}", exc_info=True)
        return []

# Removed the faulty load_announcements function which duplicated orchestration logic.
# Orchestration will now happen in main.py or a dedicated processing script.