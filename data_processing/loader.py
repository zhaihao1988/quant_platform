# rag/loader.py
from db.models import  StockDisclosure
import pandas as pd
from sqlalchemy import text
from db.database import get_engine
from data_processing.scraper import fetch_announcement_text
from utils.vector_store import embed_text

engine = get_engine()
KEYWORDS = ['年度报告', '半年度报告', '调研', '股权激励', '回购']
def load_price_data(symbol: str, window: int = 90) -> pd.DataFrame:
    """加载最近 window 天的日线数据"""
    sql = text("""
      SELECT date, open, close, high, low, volume
      FROM stock_daily
      WHERE symbol = :symbol
      ORDER BY date DESC
      LIMIT :limit
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol, "limit": window})
    return df

def load_financial_data(symbol: str, report_type: str = "benefit") -> dict:
    """加载最新一期指定类型财务报表的 JSONB 字段"""
    sql = text("""
      SELECT data
      FROM stock_financial
      WHERE symbol = :symbol AND report_type = :rtype
      ORDER BY report_date DESC
      LIMIT 1
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol, "rtype": report_type})
    return df["data"].iloc[0] if not df.empty else {}

def load_announcements(db_session, symbol: str):
    items = (
        db_session.query(StockDisclosure)
        .filter(StockDisclosure.symbol == symbol)
        .filter(StockDisclosure.raw_content.is_(None))
        .all()
    )
    for ann in items:
        text = fetch_announcement_text(ann.url, ann.title)
        if not text:
            continue
        ann.raw_content = text
        vec = embed_text(text)  # 返回 length-768 的 list[float]
        ann.content_vector = vec
        db_session.add(ann)
    db_session.commit()
