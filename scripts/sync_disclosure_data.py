# scripts/sync_disclosure_data.py (修复版)

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import Index
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from db.database import get_engine_instance
from db.models import StockList, StockDisclosure

# --- 配置 ---
engine = get_engine_instance()
Session = sessionmaker(bind=engine)
pd.options.mode.chained_assignment = None  # 忽略pandas的链式赋值警告


# --- 辅助函数 ---
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


def fetch_all_disclosures(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    【最终修复版】独立处理每个API调用，使其能抵抗单个接口的失败。
    """
    all_dfs = []  # 创建一个列表来存放所有成功获取的DataFrame

    # --- 1. 获取常规报告，用独立的 try-except 包裹 ---
    try:
        df_reports = ak.stock_zh_a_disclosure_report_cninfo(symbol=symbol, start_date=start_date, end_date=end_date)
        if df_reports is not None and not df_reports.empty:
            all_dfs.append(df_reports)
    except Exception as e:
        print(f"⚠️  警告：拉取 {symbol} [常规报告] 时发生错误，已跳过。错误信息: {e}")

    time.sleep(0.3)  # 保持休眠

    # --- 2. 获取调研公告，用独立的 try-except 包裹 ---
    try:
        df_relations = ak.stock_zh_a_disclosure_relation_cninfo(symbol=symbol, start_date=start_date, end_date=end_date)
        if df_relations is not None and not df_relations.empty:
            df_relations['tag'] = '调研活动'
            all_dfs.append(df_relations)
    except Exception as e:
        # 这是我们之前遇到的主要问题点，现在它可以被安全地捕获而不会中断整个流程
        print(f"⚠️  警告：拉取 {symbol} [调研公告] 时发生错误，已跳过。错误信息: {e}")

    # --- 3. 如果两个接口都失败或没数据，则返回 None ---
    if not all_dfs:
        return None

    # --- 4. 合并所有成功获取的数据 ---
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if combined_df.empty:
        return None

    combined_df.drop_duplicates(subset=['公告链接'], keep='first', inplace=True)

    # --- 5. 标准化处理 (使用"公告时间"，因为诊断结果证明了它是正确的列名) ---
    combined_df.rename(columns={
        "代码": "symbol",
        "简称": "short_name",
        "公告标题": "title",
        "公告时间": "ann_date",  # 使用诊断探针确认的正确列名
        "公告链接": "url",
    }, inplace=True)

    combined_df["symbol"] = combined_df["symbol"].astype(str)
    # 使用 format='mixed' 来灵活处理 "YYYY-MM-DD" 和 "YYYY-MM-DD HH:MM:SS" 两种格式
    combined_df["ann_date"] = pd.to_datetime(combined_df["ann_date"], format='mixed').dt.date

    final_columns = ["symbol", "short_name", "title", "ann_date", "url"]
    return combined_df.reindex(columns=final_columns)

# --- 主流程 (更新以使用修复后的函数) ---
def sync_all_disclosures():
    stock_df = get_stock_pool()
    today_str = datetime.now().strftime("%Y%m%d")
    print(f"🔄 开始同步所有类型公告，共 {len(stock_df)} 只股票，截止日期：{today_str}")

    session = Session() # 在循环外创建 Session，提高效率
    try:
        for idx, row in stock_df.iterrows():
            symbol = row["symbol"]
            list_date = row["list_date"]
            
            # 使用当前会话查询
            last_record = (
                session.query(StockDisclosure)
                .filter_by(symbol=symbol)
                .order_by(StockDisclosure.ann_date.desc())
                .first()
            )
            last_date = last_record.ann_date if last_record else None

            if last_date:
                # 新逻辑：删除最后一个交易日的数据，以便重新获取当天的全部公告
                print(f"ℹ️  为确保数据完整性，正在删除 {symbol} 在 {last_date} 的旧公告...")
                session.query(StockDisclosure).filter(
                    StockDisclosure.symbol == symbol,
                    StockDisclosure.ann_date == last_date
                ).delete(synchronize_session=False)
                session.commit() # 立即提交删除操作
                
                # 从被删除的日期开始重新爬取
                start_date = last_date.strftime("%Y%m%d")
            else:
                # 对于新股票，从上市日期开始
                start_date = pd.to_datetime(list_date).strftime("%Y%m%d") if list_date else "20100101"

            if start_date > today_str:
                print(f"⏭️ {symbol} 无需更新")
                continue

            print(f"⬇️ [{idx + 1}/{len(stock_df)}] {symbol} 从 {start_date} 更新至 {today_str}")

            # 使用统一的获取函数
            df_new = fetch_all_disclosures(symbol, start_date, today_str)

            if df_new is None or df_new.empty:
                print(f"⚪️ {symbol} 在此期间无新公告。")
                time.sleep(1) # 保持适当的延迟
                continue

            # 双重保险：在写入前，先查询并排除数据库中已存在的URL
            new_urls = df_new['url'].tolist()
            existing_urls_query = session.query(StockDisclosure.url).filter(StockDisclosure.url.in_(new_urls))
            existing_urls = {url for url, in existing_urls_query}
            
            final_df = df_new[~df_new['url'].isin(existing_urls)]

            if final_df.empty:
                print(f"ℹ️ {symbol} 获取到的公告均已存在于数据库中，无需写入。")
                time.sleep(1)
                continue

            try:
                final_df.to_sql("stock_disclosure", con=engine, index=False, if_exists="append")
                print(f"✅ {symbol} 成功写入 {len(final_df)} 条新公告。")
            except Exception as e:
                # 移除了对 IntegrityError 的捕获，因为我们已经主动去重
                print(f"❌ {symbol} 写入数据库时发生未知错误: {e}")

            time.sleep(1)
    finally:
        session.close() # 确保会话被关闭

    print("🏭 确保索引已创建...")
    Index("idx_disclosure_date", StockDisclosure.ann_date).create(bind=engine, checkfirst=True)
    Index("idx_disclosure_symbol", StockDisclosure.symbol).create(bind=engine, checkfirst=True)
    print("🎉 所有公告同步完成！")


if __name__ == "__main__":
    sync_all_disclosures()