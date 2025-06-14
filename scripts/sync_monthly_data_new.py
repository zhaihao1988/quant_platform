# scripts/sync_monthly_data_new.py
import pandas as pd
from datetime import datetime, timedelta, date
from sqlalchemy.orm import sessionmaker
from db.database import get_engine_instance
from db.models import StockMonthly, StockList  # <-- 导入 StockMonthly
import time
import concurrent.futures
from typing import Optional, Tuple

# --- 全局配置 (与日线脚本相同) ---
try:
    from WindPy import w

    if not w.isconnected():
        w.start()
    print("✅ Wind API 连接成功。")
except ImportError:
    print("❌ 错误: 未找到 WindPy 库。")
    exit()

engine = get_engine_instance()
Session = sessionmaker(bind=engine)
MAX_CONCURRENT_WORKERS = 10
SLEEP_PER_TASK = 0


# --- 数据库和数据获取函数 ---

def get_stock_pool() -> pd.DataFrame:
    """从 stock_list 表中获取所有股票 (无需改动)"""
    db_session = Session()
    try:
        stock_df = pd.read_sql("SELECT code, list_date FROM stock_list", con=db_session.bind)
    finally:
        db_session.close()
    return stock_df


def get_last_monthly_trade_date(symbol: str) -> Optional[date]:
    """从 stock_monthly 表中查找某股票最后的交易月"""
    db_session = Session()
    try:
        # 查询的表从 StockDaily 改为 StockMonthly
        result_proxy = db_session.query(StockMonthly.date).filter_by(symbol=symbol).order_by(
            StockMonthly.date.desc()).first()
        return result_proxy[0] if result_proxy else None
    finally:
        db_session.close()


def fetch_monthly_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """调用 WindPy 拉取月线数据"""
    wind_fields = "open,high,low,close,volume,amt,swing,pct_chg,chg,turn"
    try:
        # --- 核心改动：加入 Period='M' 参数 ---
        wind_data = w.wsd(symbol, wind_fields, start_date, end_date, "adj=F;Period=M")

        if wind_data.ErrorCode != 0:
            # (错误处理逻辑与日线脚本相同)
            error_msg = f"ErrorCode: {wind_data.ErrorCode}"
            if wind_data.Data and wind_data.Data[0]:
                error_msg += f", Message: {wind_data.Data[0][0]}"
            print(f"❌ [{symbol}] 调用 Wind API 时发生错误: {error_msg}")
            return None

        if not wind_data.Data or not wind_data.Data[0] or wind_data.Data[0][0] is None:
            return pd.DataFrame()

        df = pd.DataFrame(wind_data.Data, index=wind_data.Fields).T
        df['date'] = wind_data.Times
        df.columns = df.columns.str.lower()

        # (数据处理和重命名逻辑与日线脚本相同)
        df.rename(columns={"amt": "amount", "swing": "amplitude", "pct_chg": "pct_change", "chg": "price_change",
                           "turn": "turnover"}, inplace=True)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce') / 100
        df["symbol"] = symbol.split('.')[0]
        df["date"] = pd.to_datetime(df["date"]).dt.date

        return df

    except Exception as e:
        print(f"❌ [{symbol}] fetch_monthly_data 函数内部发生异常: {e}")
        return None


# --- 单个股票处理的核心逻辑 ---

def process_single_stock_monthly(stock_info: Tuple[int, pd.Series], today_str_param: str) -> str:
    """处理单个股票的月线数据获取和存储"""
    idx, stock_row = stock_info
    symbol_raw = stock_row["code"]

    # (代码加后缀的逻辑相同)
    if symbol_raw.startswith('6'):
        symbol_for_api = f"{symbol_raw}.SH"
    elif symbol_raw.startswith(('0', '3')):
        symbol_for_api = f"{symbol_raw}.SZ"
    elif symbol_raw.startswith(('8', '4')):
        symbol_for_api = f"{symbol_raw}.BJ"
    else:
        return f"⏩ [{symbol_raw}] 未知代码格式，已跳过。"

    log_prefix = f"[{symbol_for_api}]"

    # 增量更新模式
    last_date_in_db = get_last_monthly_trade_date(symbol_raw)

    if last_date_in_db:
        start_date_for_api = (last_date_in_db + timedelta(days=1)).strftime("%Y%m%d")
    else:
        start_date_for_api = pd.to_datetime(stock_row["list_date"]).strftime("%Y%m%d") if pd.notna(
            stock_row["list_date"]) else "20100101"

    end_date_for_api = today_str_param

    if start_date_for_api > end_date_for_api:
        return f"⏩ {log_prefix} 月线数据已是最新，无需更新。"

    df = fetch_monthly_data(symbol_for_api, start_date_for_api, end_date_for_api)

    if df is not None and not df.empty:
        try:
            # --- 核心改动：写入 stock_monthly 表 ---
            df.to_sql("stock_monthly", con=engine, index=False, if_exists="append")
            rows_written = len(df)
            time.sleep(SLEEP_PER_TASK)
            return f"✅ {log_prefix} 成功写入 {rows_written} 行月线数据。"
        except Exception as e:
            return f"❌ {log_prefix} 月线数据写入数据库失败: {e}"
    elif df is not None and df.empty:
        return f"ℹ️ {log_prefix} 在指定日期范围未返回月线数据。"
    else:  # df is None
        return f"❌ {log_prefix} fetch_monthly_data 执行失败。"


# --- 主同步函数 (并发版本) ---

def sync_all_monthly_data_concurrent():
    """并发同步所有股票的月线交易数据"""

    # --- 核心逻辑：在同步前，先删除本月已存在的不完整数据 ---
    db_session = Session()
    try:
        today = date.today()
        # 计算本月的星期一 (weekday: Monday is 0 and Sunday is 6)
        start_of_month = today.replace(day=1)

        print(f"🧹 正在删除本月 ({start_of_month.strftime('%Y-%m-%d')} 至今) 已存在的不完整月线数据...")
        stmt = StockMonthly.__table__.delete().where(StockMonthly.date >= start_of_month)
        result = db_session.execute(stmt)
        db_session.commit()
        print(f"✅ 成功删除 {result.rowcount} 条旧的本月记录，准备写入最新数据。")
    except Exception as e:
        db_session.rollback()
        print(f"❌ 删除本月旧数据时发生错误: {e}")
        return
    finally:
        db_session.close()
    # --- 逻辑结束 ---

    stock_df = get_stock_pool()
    today_str = datetime.now().strftime("%Y%m%d")
    print(f"📊 待处理股票总数: {len(stock_df)}")
    print(f"🚀 开始执行月线增量更新，数据将尝试更新到 {today_str}")

    # (并发执行逻辑与日线脚本相同)
    tasks_args_list = [((idx, row), today_str) for idx, row in stock_df.iterrows()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        futures = [executor.submit(process_single_stock_monthly, args[0], args[1]) for args in tasks_args_list]
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            print(f"({count}/{len(stock_df)}) {future.result()}")


# --- 主执行部分 ---

if __name__ == "__main__":
    print("脚本开始执行：同步月线数据...")
    sync_all_monthly_data_concurrent()
    print("\n月线数据同步脚本执行完毕。")