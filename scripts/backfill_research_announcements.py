# scripts/backfill_research_announcements.py

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from db.database import get_engine_instance
from db.models import StockList  # StockDisclosure 模型由 SQLAlchemy 在写入时自动使用

# --- 配置 ---
engine = get_engine_instance()
Session = sessionmaker(bind=engine)
# 忽略pandas在DataFrame切片上赋值时可能产生的警告
pd.options.mode.chained_assignment = None


# --- 辅助函数 ---

def get_stock_pool() -> pd.DataFrame:
    """从 stock_list 表中获取所有股票代码"""
    session = Session()
    try:
        df = pd.read_sql("SELECT code AS symbol FROM stock_list", con=engine)
        return df
    finally:
        session.close()


def fetch_research_announcements(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """专门用于获取“调研”相关公告，并打上标签"""
    try:
        df = ak.stock_zh_a_disclosure_relation_cninfo(
            symbol=symbol,
            market="沪深京",
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return None

        # 标准化列名和数据类型
        df.rename(columns={"代码": "symbol", "简称": "short_name", "公告标题": "title", "公告时间": "ann_date",
                           "公告链接": "url"}, inplace=True)

        # --- 为数据打上'调研活动'标签 ---
        df["tag"] = "调研活动"

        df["symbol"] = df["symbol"].astype(str)
        df["ann_date"] = pd.to_datetime(df["ann_date"]).dt.date

        # 返回带有标签的完整数据
        return df[["symbol", "short_name", "title", "ann_date", "url", "tag"]]
    except Exception as e:
        print(f"❌ 拉取 {symbol} [调研公告]失败: {e}")
        return None


# --- 主流程 ---

def backfill_all_research_announcements():
    """
    一次性补录所有股票近五年的调研公告。
    """
    stock_df = get_stock_pool()

    # 1. 定义固定的时间范围
    today = datetime.now()
    five_years_ago = today - timedelta(days=5 * 365)

    start_date_str = five_years_ago.strftime("%Y%m%d")
    end_date_str = today.strftime("%Y%m%d")

    print(f"🔄 开始一次性补录任务：调研公告")
    print(f"时间范围: {start_date_str} 至 {end_date_str}")
    print(f"股票总数: {len(stock_df)}")
    print("-" * 50)

    for idx, row in stock_df.iterrows():
        symbol = row["symbol"]

        print(f"⬇️ [{idx + 1}/{len(stock_df)}] 正在处理股票: {symbol}")

        # 2. 调用专用的调研公告获取函数
        df = fetch_research_announcements(symbol, start_date_str, end_date_str)

        if df is not None and not df.empty:
            # 3. 写入数据库 (利用唯一约束自动去重)
            try:
                # 写入 stock_disclosure 表，如果URL已存在，数据库会阻止插入
                df.to_sql("stock_disclosure", con=engine, index=False, if_exists="append")
                print(f"✅ {symbol} 成功写入 {len(df)} 条调研公告。")
            except IntegrityError:
                # 这是预料之中的情况，说明部分或全部数据已存在
                print(f"⚠️ {symbol} 的部分或全部调研公告已存在于数据库中，跳过重复部分。")
            except Exception as e:
                print(f"❌ {symbol} 写入数据库时发生未知错误: {e}")
        else:
            print(f"⚪️ {symbol} 在此5年期间无调研公告。")

        # 4. 限速以防封IP
        time.sleep(1)

    print("\n🎉 所有股票的5年调研公告补录任务完成！")


# --- 主执行部分 ---

if __name__ == "__main__":
    backfill_all_research_announcements()