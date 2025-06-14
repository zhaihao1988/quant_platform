# utils/full_history_refresh.py
import time
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import concurrent.futures

from db.database import get_engine_instance
from db.models import StockList  # Models are used implicitly by table names

# --- 配置 ---
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
MAX_CONCURRENT_WORKERS = 10  # 可以适当调高并发，因为是首次写入，没有数据库锁的竞争


# --- 数据获取与处理函数 ---
def fetch_and_process_data_for_stock(stock_info: pd.Series):
    """为单只股票获取并处理其完整的日、周、月线历史数据"""
    symbol_raw = stock_info["code"]
    list_date = stock_info["list_date"]

    # 构造带后缀的股票代码
    if symbol_raw.startswith('6'):
        symbol_for_api = f"{symbol_raw}.SH"
    elif symbol_raw.startswith(('0', '3')):
        symbol_for_api = f"{symbol_raw}.SZ"
    elif symbol_raw.startswith(('8', '4')):
        symbol_for_api = f"{symbol_raw}.BJ"
    else:
        return f"⏩ [{symbol_raw}] 未知代码格式，已跳过。"

    start_date = pd.to_datetime(list_date).strftime("%Y%m%d") if pd.notna(list_date) else "19900101"
    end_date = datetime.now().strftime("%Y%m%d")

    log_prefix = f"[{symbol_for_api}]"

    # 循环处理日、周、月三个周期
    for period, table_name in [('D', 'stock_daily'), ('W', 'stock_weekly'), ('M', 'stock_monthly')]:
        try:
            wind_fields = "open,high,low,close,volume,amt,swing,pct_chg,chg,turn"

            # ==================== 核心修改点 ====================
            # 使用 PriceAdj=F 来确保获取的是正确的前复权数据
            wind_data = w.wsd(symbol_for_api, wind_fields, start_date, end_date, f"PriceAdj=F;Period={period}")
            # ====================================================

            if wind_data.ErrorCode != 0 or not wind_data.Data or not wind_data.Data[0]:
                continue  # 如果某周期没数据，跳到下一个周期

            df = pd.DataFrame(wind_data.Data, index=wind_data.Fields).T
            df['date'] = wind_data.Times
            df.columns = df.columns.str.lower()

            df.rename(columns={"amt": "amount", "swing": "amplitude", "pct_chg": "pct_change", "chg": "price_change",
                               "turn": "turnover"}, inplace=True)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce') / 100
            df["symbol"] = symbol_raw
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # 写入数据库
            df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize=1000)
        except Exception as e:
            print(f"❌ {log_prefix} 处理 {table_name} 时失败: {e}")

    return f"✅ {log_prefix} 全周期历史数据刷新完成。"


# --- 主流程 ---
def run_full_refresh():
    """执行全量刷新"""
    print("=" * 60)
    print("              ⚠️  警告：即将开始全量数据刷新！ ⚠️")
    print("本操作会【清空】stock_daily, stock_weekly, stock_monthly 三张表！")
    print("请确保您已经备份了数据库。")
    print("=" * 60)

    confirm = input("请输入 'yes' 以确认执行操作: ")
    if confirm.lower() != 'yes':
        print("操作已取消。")
        return

    # 1. 清空数据表
    db_session = Session()
    try:
        print("🧹 正在清空历史行情数据表...")
        db_session.execute(text("TRUNCATE TABLE stock_daily, stock_weekly, stock_monthly RESTART IDENTITY;"))
        db_session.commit()
        print("✅ 数据表已清空。")
    except Exception as e:
        db_session.rollback()
        print(f"❌ 清空数据表时失败: {e}")
        return
    finally:
        db_session.close()

    # 2. 获取股票池并并发执行刷新
    stock_df = pd.read_sql("SELECT code, list_date FROM stock_list", con=engine)
    print(f"🚀 开始为 {len(stock_df)} 只股票全量刷新历史数据，请耐心等待...")

    tasks = [row for _, row in stock_df.iterrows()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_stock = {executor.submit(fetch_and_process_data_for_stock, task): task['code'] for task in tasks}
        count = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            result_message = future.result()
            print(f"({count}/{len(stock_df)}) {result_message}")

    print("\n🎉🎉🎉 所有股票全历史周期数据刷新完成！🎉🎉🎉")


if __name__ == "__main__":
    run_full_refresh()