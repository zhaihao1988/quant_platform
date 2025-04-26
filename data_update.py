# data_update.py
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from db.database import get_engine
from db.models import StockDaily

def update_daily_data(symbol: str):
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    # 获取已有的最后一个交易日
    latest = session.query(StockDaily).filter_by(symbol=symbol).order_by(StockDaily.date.desc()).first()
    if latest:
        start_date = (latest.date + timedelta(days=1)).strftime("%Y%m%d")
    else:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")

    end_date = datetime.now().strftime("%Y%m%d")
    print(f"⏳ 正在拉取 {symbol} 从 {start_date} 到 {end_date} 的日线数据")
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, period='daily', adjust="")
    except Exception as e:
        print(f"❌ 获取数据失败：{e}")
        return

    if df.empty:
        print("⚠️ 无新增数据。")
        return

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
        "volume", "amount", "amplitude", "pct_change",
        "price_change", "turnover"
    ]]

    # 写入数据库
    df.to_sql("stock_daily", con=engine, if_exists="append", index=False)
    print(f"✅ 成功插入 {len(df)} 条新数据：{symbol}")
    session.close()

if __name__ == "__main__":
    update_daily_data("000001")  # 示例：平安银行
