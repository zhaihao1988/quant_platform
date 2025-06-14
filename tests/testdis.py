# scripts/sync_disclosure_data.py (完整重构版)

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from db.database import get_engine_instance
from db.models import StockList, StockDisclosure

# --- 配置 (保持不变) ---
engine = get_engine_instance()
Session = sessionmaker(bind=engine)
logger = pd.options.mode.chained_assignment = None  # 忽略pandas的链式赋值警告


# --- 辅助函数 (保持不变) ---
def get_stock_pool() -> pd.DataFrame:
    session = Session()
    df = pd.read_sql("SELECT code AS symbol, list_date FROM stock_list", con=engine)
    session.close()
    return df


def get_last_disclosure_date(symbol: str) -> datetime.date:
    session = Session()
    last = (
        session.query(StockDisclosure)
        .filter_by(symbol=symbol)
        .order_by(StockDisclosure.ann_date.desc())
        .first()
    )
    session.close()
    return last.ann_date if last else None


# --- 数据获取函数 (模块化) ---

def fetch_periodic_reports(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    【模块一】调用 AkShare 获取定期报告、日常经营等公告
    """
    try:
        # 注意：这里的函数名是 stock_disclosure_report_cninfo
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol, market="沪深京", category="",
            start_date=start_date, end_date=end_date,
        )
        if df.empty:
            return None
        # 标准化列名和数据类型
        df.rename(columns={"代码": "symbol", "简称": "short_name", "公告标题": "title", "公告时间": "ann_date",
                           "公告链接": "url"}, inplace=True)
        df["symbol"] = df["symbol"].astype(str)
        df["ann_date"] = pd.to_datetime(df["ann_date"]).dt.date
        return df[["symbol", "short_name", "title", "ann_date", "url"]]
    except Exception as e:
        print(f"❌ 拉取 {symbol} [定期报告]失败: {e}")
        return None


def fetch_research_announcements(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    【模块二】新增的函数，专门用于获取“调研”相关公告
    """
    try:
        # 注意：这里的函数名是 stock_zh_a_disclosure_relation_cninfo
        df = ak.stock_zh_a_disclosure_relation_cninfo(
            symbol=symbol, market="沪深京",
            start_date=start_date, end_date=end_date,
        )
        if df.empty:
            return None
        # 同样进行标准化，确保两个函数的输出格式完全一致
        df.rename(columns={"代码": "symbol", "简称": "short_name", "公告标题": "title", "公告时间": "ann_date",
                           "公告链接": "url"}, inplace=True)
        df["tag"] = "调研"
        df["symbol"] = df["symbol"].astype(str)
        df["ann_date"] = pd.to_datetime(df["ann_date"]).dt.date
        return df[["symbol", "short_name", "title", "ann_date", "url"]]
    except Exception as e:
        print(f"❌ 拉取 {symbol} [调研公告]失败: {e}")
        return None


# --- 主流程 ---

def sync_all_disclosures():
    stock_df = get_stock_pool()
    today_str = datetime.now().strftime("%Y%m%d")
    print(f"🔄 开始同步所有类型公告，共 {len(stock_df)} 只股票，截止日期：{today_str}")

    for idx, row in stock_df.iterrows():
        symbol = row["symbol"]
        list_date = row["list_date"]
        last_date = get_last_disclosure_date(symbol)

        if last_date:
            start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start_date = pd.to_datetime(list_date).strftime("%Y%m%d") if list_date else "20100101"

        if start_date > today_str:
            print(f"⏭️ {symbol} 无需更新")
            continue

        print(f"⬇️ [{idx + 1}/{len(stock_df)}] {symbol} 从 {start_date} 更新至 {today_str}")

        # 1. 分别获取两种类型的公告
        df_reports = fetch_periodic_reports(symbol, start_date, today_str)
        time.sleep(0.5)  # 短暂休眠
        df_research = fetch_research_announcements(symbol, start_date, today_str)

        # 2. 合并结果
        combined_df = pd.concat([df_reports, df_research], ignore_index=True)

        if combined_df.empty:
            print(f"⚪️ {symbol} 在此期间无新公告。")
            time.sleep(1)
            continue

        # 3. 去重：防止两个接口返回相同公告，或与数据库中已有数据重复
        combined_df.drop_duplicates(subset=['url'], keep='first', inplace=True)

        # 4. 写入数据库
        try:
            combined_df.to_sql("stock_disclosure", con=engine, index=False, if_exists="append")
            print(f"✅ {symbol} 成功写入 {len(combined_df)} 条新公告。")
        except IntegrityError:
            print(f"⚠️ {symbol} 的部分公告已存在于数据库中，跳过重复部分。这是正常现象。")
        except Exception as e:
            print(f"❌ {symbol} 写入数据库时发生未知错误: {e}")

        time.sleep(1)

    print("🏭 确保索引已创建...")
    Index("idx_disclosure_date", StockDisclosure.ann_date).create(bind=engine, checkfirst=True)
    Index("idx_disclosure_symbol", StockDisclosure.symbol).create(bind=engine, checkfirst=True)
    print("🎉 所有公告同步完成！")


if __name__ == "__main__":
    sync_all_disclosures()