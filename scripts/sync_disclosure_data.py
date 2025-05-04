# sync_disclosure_data.py

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import Index
from sqlalchemy.orm import sessionmaker

from db.database import get_engine
from db.models import StockList, StockDisclosure

# 创建数据库引擎与 Session
engine = get_engine()
Session = sessionmaker(bind=engine)

def get_stock_pool() -> pd.DataFrame:
    """
    从 stock_list 表中获取所有股票代码及上市日期
    """
    session = Session()
    df = pd.read_sql("SELECT code AS symbol, list_date FROM stock_list", con=engine)
    session.close()
    return df

def get_last_disclosure_date(symbol: str) -> datetime.date:
    """
    查询 stock_disclosure 表中指定股票的最新公告日期
    """
    session = Session()
    last = (
        session.query(StockDisclosure)
        .filter_by(symbol=symbol)
        .order_by(StockDisclosure.ann_date.desc())
        .first()
    )
    session.close()
    return last.ann_date if last else None

def fetch_disclosures(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    调用 AkShare 获取指定股票、指定日期区间的公告数据
    参数:
      symbol     股票代码，如 "000001"
      market     市场类型，这里固定 "沪深京"
      category   公告类别，空字符串表示所有类别
      start_date 起始日期，格式 "YYYYMMDD"
      end_date   结束日期，格式 "YYYYMMDD"
    返回:
      DataFrame，包含字段 ['代码','简称','公告标题','公告时间','公告链接']
    """
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol,
            market="沪深京",
            category="",
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return None
        # 重命名列为模型属性名
        df.rename(columns={
            "代码": "symbol",
            "简称": "short_name",
            "公告标题": "title",
            "公告时间": "ann_date",
            "公告链接": "url",
        }, inplace=True)
        df["symbol"] = df["symbol"].astype(str)
        df["ann_date"] = pd.to_datetime(df["ann_date"]).dt.date
        return df[["symbol", "short_name", "title", "ann_date", "url"]]
    except Exception as e:
        print(f"❌ 拉取 {symbol} 公告失败: {e}")
        return None

def sync_all_disclosures():
    stock_df = get_stock_pool()
    today_str = datetime.now().strftime("%Y%m%d")
    print(f"🔄 开始同步公告，共 {len(stock_df)} 只股票，截止日期：{today_str}")

    for idx, row in stock_df.iterrows():
        symbol = row["symbol"]
        list_date = row["list_date"]
        last_date = get_last_disclosure_date(symbol)

        # 确定起始日期
        if last_date:
            start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        else:
            # 若无历史，则从上市日起或 2010-01-01 开始
            start = (
                pd.to_datetime(list_date).strftime("%Y%m%d")
                if list_date
                else "20100101"
            )

        if start > today_str:
            print(f"⏭️ {symbol} 无需更新")
            continue

        print(f"⬇️ [{idx+1}/{len(stock_df)}] {symbol} 从 {start} 更新至 {today_str}")
        df = fetch_disclosures(symbol, start, today_str)
        if df is not None and not df.empty:
            # 写入数据库
            df.to_sql("stock_disclosure", con=engine, index=False, if_exists="append")
            print(f"✅ 写入 {len(df)} 条公告")
        time.sleep(1)  # 限速以防封 IP

    # 同步完成后添加索引（如果尚未建立）
    print("🏭 确保索引已创建")
    Index("idx_disclosure_date", StockDisclosure.ann_date).create(bind=engine, checkfirst=True)
    Index("idx_disclosure_symbol", StockDisclosure.symbol).create(bind=engine, checkfirst=True)
    print("🎉 所有公告同步完成")

if __name__ == "__main__":
    sync_all_disclosures()
