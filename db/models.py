# db/models.py (您的文件内容)
from datetime import date

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Text, JSON, ForeignKey, Index, \
    BigInteger, Numeric, DateTime, func  # Added JSON for financial data
from pgvector.sqlalchemy import Vector # pgvector 导入
from sqlalchemy.orm import relationship

from config.settings import settings # Import settings to get dimension

Base = declarative_base()

class StockList(Base):
    __tablename__ = "stock_list"
    code = Column(String, primary_key=True)
    name = Column(String)
    area = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    list_date = Column(Date, nullable=True)

class News(Base): # 注意：您提到 StockDisclosureChunk 可能暂时为空，但 News 模型也存在
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

    __table_args__ = (
        Index('idx_stock_daily_stock_date', 'symbol', 'date', unique=True),
    )

class StockWeekly(Base):
    __tablename__ = "stock_weekly"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), ForeignKey('stock_list.code'), nullable=False, index=True)  # 股票代码
    date = Column(Date, nullable=False, index=True)  # 周的结束日期 (例如，周五)

    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)  # 成交量，根据您的数据源可能是Integer或Float
    amount = Column(Float, nullable=True)
    amplitude = Column(Float, nullable=True)
    pct_change = Column(Float, nullable=True)
    price_change = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)  # 成交额，字段名与 StockDaily 一致
    __table_args__ = (
        Index('idx_stock_weekly_stock_date', 'symbol', 'date', unique=True),
    )

class StockMonthly(Base):
    __tablename__ = "stock_monthly"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), ForeignKey('stock_list.code'), nullable=False, index=True)  # 股票代码
    date = Column(Date, nullable=False, index=True)  # 月的结束日期 (例如，月末最后一个交易日)

    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)  # 成交量
    amount = Column(Float, nullable=True)
    amplitude = Column(Float, nullable=True)
    pct_change = Column(Float, nullable=True)
    price_change = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)  # 成交额
    # pct_chg = Column(Float, nullable=True) # 月涨跌幅
    __table_args__ = (
        Index('idx_stock_monthly_stock_date', 'symbol', 'date', unique=True),
    )

class StockFinancial(Base):
    """财务数据表 (Assuming structure based on loader.py)"""
    __tablename__ = "stock_financial"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    report_type = Column(String, index=True) # e.g., "benefit", "balance", "cashflow"
    report_date = Column(Date, index=True)
    data = Column(JSON) # Assuming financial data stored as JSON
    __table_args__ = (
        Index('idx_financial_stock_report_type_date', 'symbol', 'report_type', 'report_date', unique=True),
        # 如果财报发布日期也需要唯一性，则可以调整或增加索引
    )

class StockDisclosure(Base): # 上市公司公告元数据
    __tablename__ = "stock_disclosure"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    short_name = Column(String(50), nullable=True)
    title = Column(String(500), nullable=False) # 确保长度足够
    ann_date = Column(Date, nullable=False, index=True)
    url = Column(String(500), nullable=False, unique=True) # 公告URL通常是唯一的
    raw_content = Column(Text, nullable=True) # 全文，用于后续chunk处理
    # content_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True) # 已移至 StockDisclosureChunk
    tag = Column(String(50), nullable=True, index=True)
class StockDisclosureChunk(Base): # 公告文本分块及向量化
    __tablename__ = "stock_disclosure_chunk"
    id = Column(Integer, primary_key=True, autoincrement=True)
    disclosure_id = Column(Integer, ForeignKey('stock_disclosure.id'), nullable=False, index=True) # 自动创建索引
    chunk_order = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
    disclosure = relationship("StockDisclosure")
    disclosure_ann_date = Column(Date, index=True, nullable=False )
    # __table_args__ 用于定义复合索引、唯一约束等
    # 尝试在这里定义 HNSW 索引 (需要 pgvector 和 SQLAlchemy 支持)
    __table_args__ = (
        Index(
            'idx_chunk_vector_hnsw_cosine', # 索引名称
            chunk_vector,                   # 索引的列
            postgresql_using='hnsw',        # 使用 HNSW 方法
            postgresql_with={               # HNSW 特定参数
                'm': 16,
                'ef_construction': 64
            },
            postgresql_ops={'chunk_vector': 'vector_cosine_ops'} # 指定操作符类以支持余弦距离
        ),
        Index('idx_chunk_disclosure_order', 'disclosure_id', 'chunk_order', unique=True) # 确保同一公告内块顺序唯一
    )

class StockShareDetail(Base): # 新增的股本详情表
    __tablename__ = "stock_share_details"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False, unique=True)
    stock_name = Column(String(50))
    total_shares = Column(BigInteger)
    float_shares = Column(BigInteger)
    total_market_cap = Column(Numeric(precision=20, scale=4))
    float_market_cap = Column(Numeric(precision=20, scale=4))
    industry = Column(String(50))
    listing_date = Column(Date)
    data_source_date = Column(Date, default=date.today)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())