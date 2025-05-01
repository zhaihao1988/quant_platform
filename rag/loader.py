# rag/loader.py

import pandas as pd
from sqlalchemy import text
from db.database import get_engine, get_session
from utils.scraper import fetch_announcement_text

engine = get_engine()

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

def load_announcements(symbol: str, top_n: int = 3) -> list[str]:
    """加载最近 top_n 条公告链接并抓取正文"""
    sql = text("""
      SELECT url
      FROM stock_disclosure
      WHERE symbol = :symbol
      ORDER BY ann_date DESC
      LIMIT :limit
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol, "limit": top_n})
    texts = []
    for url in df["url"]:
        text = fetch_announcement_text(url)
        if text:
            texts.append(text[:1000])  # 截取前1000字符
    return texts
