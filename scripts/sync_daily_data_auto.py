# scripts/sync_daily_data_auto.py
import os
import sys
from datetime import datetime, timedelta, date # 导入 date
import time
import schedule # 重新导入 schedule

# --- Python Path Setup ---
# (保持不变)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- 正常导入 ---
import akshare as ak
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db.database import get_engine_instance
from db.models import StockDaily, StockList

# --- 导入要运行的策略脚本的 main 函数 ---
try:
    from strategies.multi_level_cross_strategy_new import main as run_strategy_main
    STRATEGY_IMPORT_SUCCESS = True
except ImportError as import_err:
    print(f"❌ Error importing strategy main function: {import_err}")
    STRATEGY_IMPORT_SUCCESS = False

# --- 数据库和其他设置 ---
engine = get_engine_instance()
Session = sessionmaker(bind=engine)

# --- 函数定义 (get_stock_pool, get_last_trade_date, fetch_data, save_data_to_db 保持不变) ---
def get_stock_pool():
    session = Session(); stock_df = pd.DataFrame(columns=['code', 'list_date'])
    try: stock_df = pd.read_sql("SELECT code, list_date FROM stock_list", con=engine)
    except Exception as e: print(f"❌ Error fetching stock pool: {e}")
    finally: session.close()
    return stock_df

def get_last_trade_date(symbol):
    session = Session(); last_date = None
    try:
        result = session.query(StockDaily.date).filter_by(symbol=symbol).order_by(StockDaily.date.desc()).first()
        last_date = result[0] if result else None
    except Exception as e: print(f"❌ Error fetching last trade date for {symbol}: {e}")
    finally: session.close()
    return last_date if isinstance(last_date, datetime.date) else None

def fetch_data(symbol, start_date, end_date):
    try:
        time.sleep(0.15) # Reduce delay slightly if needed
        print(f"   Fetching {symbol} from {start_date} to {end_date}...")
        df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, period="daily", adjust="qfq")
        if df is None or df.empty: print(f"   ⚠️ No data for {symbol}."); return None
        if '日期' not in df.columns: print(f"   ❌ Bad format for {symbol}."); return None
        df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount","振幅":"amplitude","涨跌幅":"pct_change","涨跌额":"price_change","换手率":"turnover"}, inplace=True)
        df["symbol"] = symbol; df["date"] = pd.to_datetime(df["date"])
        num_cols = ['open','close','high','low','volume','amount','amplitude','pct_change','price_change','turnover']
        for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df[[ "symbol","date","open","close","high","low","volume","amount","amplitude","pct_change","price_change","turnover" ]]
        print(f"   ✅ Fetched {len(df)} rows for {symbol}.")
        return df
    except Exception as e: print(f"   ❌ Fetch error for {symbol}: {e}"); return None

def save_data_to_db(df, symbol):
    if df is None or df.empty: return 0
    df['date'] = pd.to_datetime(df['date']).dt.date
    session = Session()
    try:
        df.to_sql('stock_daily', con=engine, if_exists='append', index=False, chunksize=1000)
        session.commit(); print(f"   💾 Saved {len(df)} rows for {symbol}."); return len(df)
    except Exception as e: session.rollback(); print(f"   ❌ DB Save error for {symbol}: {e}"); return -1
    finally: session.close()

# --- 数据同步主函数 (保持不变) ---
def sync_all_data():
    start_time = time.time()
    print(f"\n======= Starting Data Synchronization at {datetime.now()} =======")
    stock_df = get_stock_pool(); today_dt = datetime.now(); today_str = today_dt.strftime("%Y%m%d")
    print(f"📊 Stock Pool Size: {len(stock_df)}, Current Date: {today_str}")
    error_count = 0; success_count = 0; total_rows_saved = 0
    for idx, row in stock_df.iterrows():
        symbol = row["code"]; list_date_obj = None
        if pd.notna(row["list_date"]):
             try: list_date_obj = pd.to_datetime(row["list_date"]).date()
             except Exception: list_date_obj = datetime(2010, 1, 1).date()
        else: list_date_obj = datetime(2010, 1, 1).date()
        list_date_str = list_date_obj.strftime("%Y%m%d")
        # print(f"\nProcessing [{idx+1}/{len(stock_df)}]: {symbol} (Listed: {list_date_str})") # Can be verbose
        last_date_obj = get_last_trade_date(symbol)
        start_date_obj = (last_date_obj + timedelta(days=1)) if last_date_obj else list_date_obj
        # if last_date_obj: print(f"   Last: {last_date_obj.strftime('%Y-%m-%d')}. Fetch from {start_date_obj.strftime('%Y%m%d')}.")
        # else: print(f"   No data. Fetching from {start_date_obj.strftime('%Y%m%d')}.")
        end_date_obj = today_dt.date()
        if start_date_obj > end_date_obj:
            # print(f"   ✅ Up-to-date: {symbol}.") # Can be verbose
            success_count += 1; continue
        start_date_str = start_date_obj.strftime("%Y%m%d"); end_date_str = end_date_obj.strftime("%Y%m%d")
        df_data = fetch_data(symbol, start_date_str, end_date_str)
        if df_data is not None and not df_data.empty:
            save_result = save_data_to_db(df_data, symbol)
            if save_result > 0: success_count += 1; total_rows_saved += save_result
            elif save_result == -1: error_count += 1
        elif df_data is None: error_count += 1
        else: success_count += 1
    print("\n--- Synchronization Summary ---"); print(f"Success: {success_count}, Errors: {error_count}, Rows Saved: {total_rows_saved}")
    end_time = time.time(); print(f"Sync Duration: {end_time - start_time:.2f} seconds"); print("------------------------------")
    return error_count == 0

# --- 交易日判断函数 ---
def is_trade_date(check_date=None):
    """检查指定日期是否为A股交易日"""
    if check_date is None: check_date = date.today()
    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        trade_dates = set(pd.to_datetime(trade_cal['trade_date']).dt.date)
        is_trading = check_date in trade_dates
        print(f"Checking trade date for {check_date}: {'Yes' if is_trading else 'No'}")
        return is_trading
    except Exception as e:
        print(f"❌ Error fetching trade calendar: {e}. Assuming it IS a trade date for safety.")
        return True # 保守处理

# --- 封装要执行的任务 ---
def run_sync_and_strategy_job():
    """封装数据同步和策略运行的任务"""
    print(f"\n{'='*10} Triggering scheduled job at {datetime.now()} {'='*10}")
    job_start_time = time.time()
    # 1. 同步数据
    sync_successful = sync_all_data()

    # 2. 如果同步成功且策略导入成功，则运行策略
    if sync_successful:
        print("\n======= Attempting to Run Strategy Main Function =======")
        if STRATEGY_IMPORT_SUCCESS:
            try:
                strategy_start_time = time.time()
                run_strategy_main()
                strategy_end_time = time.time()
                print(f"\n======= Strategy Main Function Executed Successfully (Duration: {strategy_end_time - strategy_start_time:.2f} seconds) =======")
            except Exception as e:
                print(f"❌ Error running strategy's main function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Strategy main function could not be imported. Skipping execution.")
    else:
        print("\n⚠️ Data synchronization encountered errors. Skipping strategy execution.")
    job_end_time = time.time()
    print(f"\n{'='*10} Scheduled job finished at {datetime.now()} (Total Duration: {job_end_time - job_start_time:.2f} seconds) {'='*10}")

# --- 调度主逻辑 ---
def schedule_job():
    """设置调度任务"""
    # 定义目标运行时间（服务器本地时间 16:00）
    # !! 确保服务器时区正确，或调整此时间字符串以匹配目标时区 !!
    run_time_str = "16:00"
    print(f"Scheduling job to run daily at {run_time_str} (server local time).")
    print("Script needs to keep running in the background.")
    print("Press Ctrl+C to stop the scheduler.")

    # 安排任务
    schedule.every().day.at(run_time_str).do(job_wrapper)

    # 持续运行调度检查
    while True:
        try:
            schedule.run_pending()
            time.sleep(30) # 每 30 秒检查一次
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Scheduler error: {e}. Restarting check loop after delay...")
            time.sleep(60) # 发生错误后稍等再试

def job_wrapper():
    """包装器函数，检查是否为交易日再执行任务"""
    today = date.today()
    print(f"\nChecking schedule for {today}...")
    if is_trade_date(today):
        print(f"{today} is a trade date. Running the job...")
        # 在新的线程或进程中运行，避免阻塞调度器（如果任务耗时较长）？
        # 对于这个场景，顺序执行可能也可以，但长时间任务最好异步。
        # 为简单起见，暂时同步执行：
        run_sync_and_strategy_job()
    else:
        print(f"{today} is not a trade date. Skipping job.")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 直接启动调度器
    schedule_job()