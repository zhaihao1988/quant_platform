# database/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Text, JSON # Added JSON for financial data
from pgvector.sqlalchemy import Vector
from config.settings import settings # Import settings to get dimension

Base = declarative_base()

class StockList(Base):
    __tablename__ = "stock_list"
    code = Column(String, primary_key=True)     # 股票代码
    name = Column(String)                       # 股票名称
    area = Column(String, nullable=True)        # 所属地域（可选）
    industry = Column(String, nullable=True)    # 所属行业（可选）
    list_date = Column(Date, nullable=True)     # 上市日期（可选）

class News(Base):
    """公司公告/新闻表 (Not actively used in core report flow yet)"""
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    symbol = Column(String, index=True)
    title = Column(String)
    content = Column(String)
    sentiment = Column(Float)

class StockDaily(Base):
    """日线行情数据表"""
    __tablename__ = "stock_daily"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)              # 股票代码
    date = Column(Date, index=True)                  # 交易日期
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Integer)                         # 单位：手
    amount = Column(Float)                           # 成交额（元）
    amplitude = Column(Float)                        # 振幅（%）
    pct_change = Column(Float)                       # 涨跌幅（%）
    price_change = Column(Float)                     # 涨跌额（元）
    turnover = Column(Float)                         # 换手率（%）

class StockFinancial(Base):
    """财务数据表 (Assuming structure based on loader.py)"""
    __tablename__ = "stock_financial"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    report_type = Column(String, index=True) # e.g., "benefit", "balance", "cashflow"
    report_date = Column(Date, index=True)
    data = Column(JSON) # Assuming financial data stored as JSON

class StockDisclosure(Base):
    """A股上市公司公告表"""
    __tablename__ = "stock_disclosure"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    short_name = Column(String(50), nullable=True)
    title = Column(String(500), nullable=False)
    ann_date = Column(Date, nullable=False, index=True)
    url = Column(String(500), nullable=False)
    raw_content = Column(Text, nullable=True)
    # Use dimension from settings
    content_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True)