# sync_financial_data.py

import re
import time
from datetime import datetime

import akshare as ak
import pandas as pd
from sqlalchemy.orm import sessionmaker

from db.database import get_engine
from db.init_db import Base, StockFinancial
from db.models import StockList

# ———— 初始化 ————
engine = get_engine()
Session = sessionmaker(bind=engine)

def parse_amount(x: str) -> float:
    """把 '1.23亿'、'456.7万' 等字符串转换为元"""
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").strip()
    m = re.match(r"([\d\.]+)([万亿]?)", s)
    if not m:
        try: return float(s)
        except: return None
    num, unit = m.groups()
    v = float(num)
    if unit == "亿": v *= 1e8
    elif unit == "万": v *= 1e4
    return v

def get_stock_pool() -> pd.DataFrame:
    """读取所有 A 股代码及上市日"""
    session = Session()
    df = pd.read_sql("SELECT code AS symbol, list_date FROM stock_list", con=engine)
    session.close()
    return df

def get_last_report_date(symbol: str, report_type: str):
    """查询数据库中该(symbol, report_type)的最新report_date"""
    session = Session()
    last = (
        session.query(StockFinancial)
        .filter_by(symbol=symbol, report_type=report_type)
        .order_by(StockFinancial.report_date.desc())
        .first()
    )
    session.close()
    return last.report_date if last else None

def fetch_latest_and_store(symbol: str, report_type: str):
    """
    1. 从 AKShare 拉取全量报表；
    2. 在 DataFrame 中找最新 report_date，若已存则跳过；
    3. 否则只处理最新那一期并写入数据库。
    """
    # 接口映射
    api_map = {
        "debt": ak.stock_financial_debt_ths,
        "benefit": ak.stock_financial_benefit_ths,
        "cash": ak.stock_financial_cash_ths,
    }
    func = api_map[report_type]

    # 1) 拿到本地最新期
    last_date = get_last_report_date(symbol, report_type)

    # 2) 拉全量数据
    try:
        df = func(symbol, indicator="按报告期")
    except Exception as e:
        print(f"❌ 接口调用失败：{symbol} {report_type} — {e}")
        return

    if df is None or df.empty:
        return

    # 3) 规范列名并转日期
    df.rename(columns={df.columns[0]: "symbol", df.columns[1]: "report_date"}, inplace=True)
    df["symbol"] = df["symbol"].astype(str)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

    # 4) 找到远端最新期
    max_date = df["report_date"].max()
    if last_date and max_date <= last_date:
        return

    # 5) 筛出最新一期并校验
    latest_rows = df[df["report_date"] == max_date]
    if latest_rows.empty:
        # 如果没有匹配行，跳过写入
        return
    new_row = latest_rows.iloc[0]  # 安全调用

    # 6) 单位换算
    data_fields = {}
    for col in df.columns:
        if col in ("symbol", "report_date"):
            continue
        data_fields[col] = parse_amount(new_row[col])

    # 7) 写入数据库
    session = Session()
    record = {
        "symbol": symbol,
        "report_date": max_date,
        "report_type": report_type,
        "data": data_fields,
    }
    session.bulk_insert_mappings(StockFinancial, [record])
    session.commit()
    session.close()
    print(f"✅ {symbol} {report_type} 新增报告期 {max_date}")

def sync_all_financial_daily():
    stock_df = get_stock_pool()
    types = ["debt", "benefit", "cash"]

    for report_type in types:
        print(f"\n🔄 同步（仅最新一期）报表：{report_type}")
        for idx, row in stock_df.iterrows():
            symbol = row["symbol"]
            print(f"  → [{idx+1}/{len(stock_df)}] 开始处理：{symbol} …", end="", flush=True)
            fetch_latest_and_store(symbol, report_type)
            print(" 完成")    # 每只股票处理完毕都能看到
            time.sleep(0.5)


if __name__ == "__main__":
    sync_all_financial_daily()
