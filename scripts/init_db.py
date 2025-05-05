# scripts/init_db.py
from sqlalchemy import Column, Integer, String, Float, Date, JSON,Text,text,DDL
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm  import declarative_base
from db.database import get_engine_instance
from config.settings import settings
from sqlalchemy import inspect
Base = declarative_base()

class StockDaily(Base):
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

class StockDisclosure(Base):
    __tablename__ = "stock_disclosure"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)       # 股票代码
    short_name = Column(String(50), nullable=True)                # 股票简称
    title = Column(String(500), nullable=False)                   # 公告标题
    ann_date = Column(Date, nullable=False, index=True)           # 公告日期
    url = Column(String(500), nullable=False)                     # 公告链接
    raw_content = Column(Text, nullable=True)  # 公告内容
    content_vector = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
class StockList(Base):
    __tablename__ = "stock_list"
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False, index=True)     # 股票代码
    name = Column(String(100), nullable=True)                 # 股票简称
    area = Column(String(50), nullable=True)                  # 地区（可选）
    industry = Column(String(50), nullable=True)              # 行业（可选）
    list_date = Column(Date, nullable=True)                   # 上市日期（可选）
class StockFinancial(Base):
    """
    存储三大报表（资产负债、利润、现金流）数据，
    data 字段以 JSONB 存储原始字段与数值映射
    """
    __tablename__ = "stock_financial"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    report_date = Column(Date, nullable=False, index=True)   # 报告期末日期，如 2023-12-31
    report_type = Column(String(20), nullable=False, index=True)  # debt/benefit/cash
    data = Column(JSON, nullable=False)  # JSONB 存放所有字段的键值对


def add_missing_columns(engine):
    """检查并添加缺失的列"""
    inspector = inspect(engine)

    if 'stock_disclosure' in inspector.get_table_names():
        existing_columns = [col['name'] for col in inspector.get_columns('stock_disclosure')]

        # 使用 SQLAlchemy 的 text() 函数包装 SQL 语句
        if 'raw_content' not in existing_columns:
            with engine.begin() as conn:  # 使用 begin() 自动提交事务
                conn.execute(text("ALTER TABLE stock_disclosure ADD COLUMN raw_content TEXT"))
                print("✅ 已添加 raw_content 列到 stock_disclosure 表")

        if 'content_vector' not in existing_columns:
            with engine.begin() as conn:
                # 对于 pgvector 类型，需要使用正确的语法
                conn.execute(
                    text(f"ALTER TABLE stock_disclosure ADD COLUMN content_vector vector({settings.EMBEDDING_DIM})"))
                print(f"✅ 已添加 content_vector 列到 stock_disclosure 表 (维度: {settings.EMBEDDING_DIM})")


def init_db():
    engine = get_engine_instance()
    add_missing_columns(engine)
    Base.metadata.create_all(engine)
    print("✅ 数据库初始化成功，已建表 stock_daily, stock_disclosure, stock_list, stock_financial, news")

if __name__ == "__main__":
    init_db()
