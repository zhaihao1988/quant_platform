# db/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date

Base = declarative_base()

class StockList(Base):
    __tablename__ = "stock_list"
    code = Column(String, primary_key=True)     # 股票代码
    name = Column(String)                       # 股票名称
    area = Column(String, nullable=True)        # 所属地域（可选）
    industry = Column(String, nullable=True)    # 所属行业（可选）
    list_date = Column(Date, nullable=True)     # 上市日期（可选）


class News(Base):
    """公司公告/新闻表"""
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    symbol = Column(String, index=True)
    title = Column(String)
    content = Column(String)
    # 例如，我们可以存储简单情感值
    sentiment = Column(Float)
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