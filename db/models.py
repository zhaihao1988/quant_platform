# database/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Text, JSON, ForeignKey, Index   # Added JSON for financial data
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

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
class StockDisclosureChunk(Base):
    __tablename__ = "stock_disclosure_chunk"
    id = Column(Integer, primary_key=True, autoincrement=True)
    disclosure_id = Column(Integer, ForeignKey('stock_disclosure.id'), nullable=False, index=True) # 关联回原公告
    chunk_order = Column(Integer, nullable=False) # 块的顺序
    chunk_text = Column(Text, nullable=False) # 块文本
    chunk_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True) # 块向量

    disclosure = relationship("StockDisclosure") # 可选，方便查询

    # 创建向量索引 (需要手动或通过 alembic 在数据库中执行)
    # __table_args__ = (
    #     Index('idx_chunk_vector', chunk_vector, postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}),
    # )
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
    #content_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True)


class StockWeekly(Base):
    __tablename__ = "stock_weekly"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), ForeignKey('stock_list.code'), nullable=False, index=True)  # 股票代码
    date = Column(Date, nullable=False, index=True)  # 周的结束日期 (例如，周五)

    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)  # 成交量，根据您的数据源可能是Integer或Float
    turnover = Column(Float, nullable=True)  # 成交额，字段名与 StockDaily 一致
    # 如果需要，可以从 StockDaily 的 pct_change 计算周涨跌幅
    # pct_chg = Column(Float, nullable=True) # 周涨跌幅

    # 联合唯一约束，确保每个股票每周只有一条记录
    __table_args__ = (
        Index('idx_stock_weekly_stock_date', 'stock_code', 'date', unique=True),
    )
    # 可选: 如果需要从 StockWeekly 反向查询 StockList
    # stock = relationship("StockList", back_populates="weekly_data") # 假设 StockList 中有 weekly_data 关系


class StockMonthly(Base):
    __tablename__ = "stock_monthly"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), ForeignKey('stock_list.code'), nullable=False, index=True)  # 股票代码
    date = Column(Date, nullable=False, index=True)  # 月的结束日期 (例如，月末最后一个交易日)

    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)  # 成交量
    turnover = Column(Float, nullable=True)  # 成交额
    # pct_chg = Column(Float, nullable=True) # 月涨跌幅

    # 联合唯一约束
    __table_args__ = (
        Index('idx_stock_monthly_stock_date', 'stock_code', 'date', unique=True),
    )
    # 可选: 如果需要从 StockMonthly 反向查询 StockList
    # stock = relationship("StockList", back_populates="monthly_data") # 假设 StockList 中有 monthly_data 关系