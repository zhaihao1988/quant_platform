# scripts/sync_daily_data.py
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from db.database import get_engine
from db.models import StockDaily, StockList
import time

engine = get_engine()
Session = sessionmaker(bind=engine)

def get_stock_pool():
    """从 stock_list 表中获取所有股票及上市日期"""
    session = Session()
    stock_df = pd.read_sql("SELECT code, list_date FROM stock_list", con=engine)
    session.close()
    return stock_df

def get_last_trade_date(symbol):
    """从 stock_daily 表中查找某股票最后的交易日"""
    session = Session()
    result = session.query(StockDaily).filter_by(symbol=symbol).order_by(StockDaily.date.desc()).first()
    session.close()
    return result.date if result else None

def fetch_data(symbol, start_date, end_date):
    """调用 AkShare 拉取数据（封装）"""
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, period="daily", adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "price_change",
            "换手率": "turnover",
        }, inplace=True)
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        df = df[[
            "symbol", "date", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "pct_change", "price_change", "turnover"
        ]]
        return df
    except Exception as e:
        print(f"❌ 拉取 {symbol} 失败: {e}")
        return None

def sync_all_data():
    stock_df = get_stock_pool()
    today = datetime.now().strftime("%Y%m%d")
    print(f"📊 股票数量：{len(stock_df)}，当前日期：{today}")

    for idx, row in stock_df.iterrows():
        symbol = row["code"]
        list_date = row["list_date"]
        last_date = get_last_trade_date(symbol)

        if not list_date:
            list_date = "20100101"  # 若无上市时间，默认10年前
        else:
            list_date = pd.to_datetime(list_date).strftime("%Y%m%d")

        if last_date:
            # 增量更新：从上次+1天开始
            start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        else:
            # 新股：从上市时间开始全量下载
            start_date = list_date

        if start_date > today:
            print(f"⏩ {symbol} 无需更新")
            continue

        print(f"⬇️  [{idx+1}/{len(stock_df)}] {symbol} 从 {start_date} 更新到 {today}")
        df = fetch_data(symbol, start_date, today)
        if df is not None:
            df.to_sql("stock_daily", con=engine, index=False, if_exists="append")
            print(f"✅ 成功写入 {len(df)} 行")
        time.sleep(1.2)  # 限速，防封IP

    print("🎉 所有股票同步完成")

if __name__ == "__main__":
    sync_all_data()
