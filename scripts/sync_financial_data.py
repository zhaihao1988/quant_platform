# sync_financial_data.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import akshare as ak
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db.database import get_engine
from scripts.init_db import Base, StockFinancial

# 数据库引擎和会话
engine = get_engine()
Session = sessionmaker(bind=engine)

# 三大财报接口映射
API_MAP = {
    "debt": ak.stock_financial_debt_ths,
    "benefit": ak.stock_financial_benefit_ths,
    "cash": ak.stock_financial_cash_ths,
}

def get_stock_pool() -> pd.DataFrame:
    """读取所有 A 股代码及上市日"""
    df = pd.read_sql(
        "SELECT code AS symbol, list_date FROM stock_list",
        con=engine
    )
    return df


def get_last_report_date(session, symbol: str, rpt_type: str):
    """查询数据库中某股票某报表类型的最新报告期"""
    result = (
        session.query(StockFinancial.report_date)
        .filter_by(symbol=symbol, report_type=rpt_type)
        .order_by(StockFinancial.report_date.desc())
        .first()
    )
    return result[0] if result else None


def sync_report_for_symbol(session, symbol: str, rpt_type: str) -> int:
    """
    拉取单只股票单个报表类型的最新更新，并存库
    返回新增记录数
    """
    func = API_MAP[rpt_type]
    df = func(symbol=symbol, indicator="按报告期")
    if df is None or df.empty:
        print(f"{symbol} [{rpt_type}]: AKShare 无数据返回")
        return 0

    # 重命名第一列为 report_date 并转 date
    df.rename(columns={df.columns[0]: "report_date"}, inplace=True)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

    # 打印可用报告期及最新存储报告期以便调试
    available = sorted(df["report_date"].unique())
    print(f"{symbol} [{rpt_type}] AKShare 可用报告期: {available}")
    last_date = get_last_report_date(session, symbol, rpt_type)
    print(f"{symbol} [{rpt_type}] 数据库最后报告期: {last_date}")

    # 仅保留更新后的记录
    if last_date:
        df = df[df["report_date"] > last_date]
        print(f"{symbol} [{rpt_type}] 筛选后剩余报告期: {sorted(df['report_date'].unique())}")
        if df.empty:
            return 0

    # 构造记录并插入
    inserted = 0
    for _, row in df.iterrows():
        rd = row["report_date"]
        data = row.drop(labels=["report_date"]).to_dict()
        rec = StockFinancial(
            symbol=symbol,
            report_type=rpt_type,
            report_date=rd,
            data=data
        )
        session.add(rec)
        inserted += 1
        print(f"准备插入: {symbol} {rpt_type} {rd}")

    # 提交本报表类型新增
    session.commit()
    return inserted


def sync_financial_data():
    """
    主流程：增量同步所有股票所有报表类型
    """
    # 确保表结构
    Base.metadata.create_all(engine)
    session = Session()

    # 获取股票池
    pool_df = get_stock_pool()
    symbols = pool_df['symbol'].tolist()

    total_counts = {k: 0 for k in API_MAP}

    # 遍历同步
    for symbol in symbols:
        for rpt_type in API_MAP:
            try:
                cnt = sync_report_for_symbol(session, symbol, rpt_type)
                total_counts[rpt_type] += cnt
            except Exception as e:
                print(f"⚠️ {symbol} [{rpt_type}] 同步失败: {e}")

    session.close()

    # 输出汇总
    print("同步完成，汇总:")
    for rpt_type, cnt in total_counts.items():
        print(f"  - {rpt_type}: {cnt} 条新增")

if __name__ == "__main__":
    sync_financial_data()
