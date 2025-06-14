import pandas as pd
from datetime import datetime, timedelta, date
from sqlalchemy.orm import sessionmaker
from db.database import get_engine_instance
from db.models import StockDaily, StockList
import time
import concurrent.futures
from typing import Optional, Tuple

# 导入 WindPy 并启动
try:
    from WindPy import w
    if not w.isconnected():
        w.start()
    print("✅ Wind API 连接成功。")
except ImportError:
    print("❌ 错误: 未找到 WindPy 库，请先安装 `pip install windpy`。")
    exit() # 如果没有WindPy则无法继续，直接退出

# --- 全局配置 ---
engine = get_engine_instance()
Session = sessionmaker(bind=engine)

# 并发和频率控制参数 (请谨慎调整！)
MAX_CONCURRENT_WORKERS = 10
SLEEP_PER_TASK = 0


# --- 数据库和数据获取函数 (与之前类似，确保它们是线程安全的或在线程内正确使用) ---
def get_stock_pool() -> pd.DataFrame:
    """从 stock_list 表中获取所有股票及上市日期"""
    db_session = Session()
    try:
        # 直接读取为DataFrame，list_date的类型取决于数据库和驱动
        stock_df = pd.read_sql("SELECT code, list_date FROM stock_list", con=db_session.bind)  # 使用 session.bind 获取连接
    finally:
        db_session.close()
    return stock_df


def get_last_trade_date(symbol: str) -> Optional[date]:
    """从 stock_daily 表中查找某股票最后的交易日 (返回 date 对象)"""
    db_session = Session()
    try:
        result_proxy = db_session.query(StockDaily.date).filter_by(symbol=symbol).order_by(
            StockDaily.date.desc()).first()
        # .first() 返回的是一个 Row 对象 (类似元组)，或者 None
        # 如果 result_proxy 不是 None，它包含一个元素，即日期
        return result_proxy[0] if result_proxy else None
    finally:
        db_session.close()


def fetch_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    调用 WindPy 的 w.sd API 拉取日线数据。（已修复大小写问题）
    """
    wind_fields = "open,high,low,close,volume,amt,swing,pct_chg,chg,turn"

    try:
        wind_data = w.wsd(symbol, wind_fields, start_date, end_date, "adj=F")

        if wind_data.ErrorCode != 0:
            error_msg = f"ErrorCode: {wind_data.ErrorCode}"
            if wind_data.Data and wind_data.Data[0]:
                error_msg += f", Message: {wind_data.Data[0][0]}"
            print(f"❌ [{symbol}] 调用 Wind API 时发生错误: {error_msg}")
            return None

        if not wind_data.Data or not wind_data.Data[0] or wind_data.Data[0][0] is None:
            return pd.DataFrame()

        df = pd.DataFrame(wind_data.Data, index=wind_data.Fields).T
        df['date'] = wind_data.Times

        # --- 关键修复：将所有列名统一转换为小写 ---
        df.columns = df.columns.str.lower()
        # -----------------------------------------

        # 现在所有列名都是小写的了，我们可以安全地进行后续操作
        df.rename(columns={
            "amt": "amount",
            "swing": "amplitude",
            "pct_chg": "pct_change",
            "chg": "price_change",
            "turn": "turnover"
        }, inplace=True)

        # 这一行现在可以正常工作了
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce') / 100

        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"])

        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_change', 'price_change',
                        'turnover']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        required_columns = [
            "symbol", "date", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "pct_change", "price_change", "turnover"
        ]

        df['symbol'] = symbol.split('.')[0]
        return df[required_columns]

    except KeyError as e:
        # 增加一个更具体的KeyError捕获，方便定位问题
        print(f"❌ [{symbol}] 处理数据时发生列名错误 (KeyError): {e}。请检查API返回的字段。")
        return None
    except Exception as e:
        print(f"❌ [{symbol}] fetch_data 函数内部发生未预料的异常 ({start_date} 到 {end_date}): {e}")
        return None
# --- 单个股票处理的核心逻辑 (由并发工作线程调用) ---
def process_single_stock(stock_info: Tuple[int, pd.Series], today_str_param: str,
                         target_date_manual_param: Optional[str]) -> str:
    """
    处理单个股票的数据获取和存储。
    stock_info: 一个包含 (index, row_data) 的元组，row_data 是从 stock_df.iterrows() 来的一行。
    today_str_param: '今天'的日期字符串 (YYYYMMDD)，用于增量更新模式的结束日期。
    target_date_manual_param: 手动指定的目标日期 (YYYYMMDD)，如果提供，则只获取该日期的数据。
    """
    idx, stock_row = stock_info  # stock_row 是一个 pandas Series
    symbol = stock_row["code"]
    list_date_from_db = stock_row["list_date"]
    # --- 新增逻辑：为股票代码添加交易所后缀 ---
    if symbol.startswith('6'):
        symbol_for_api = f"{symbol}.SH"
    elif symbol.startswith(('0', '3', '2')):  # 沪市股票以 0, 3, 2 开头
        symbol_for_api = f"{symbol}.SZ"
    elif symbol.startswith(('8', '4')):  # 北交所
        symbol_for_api = f"{symbol}.BJ"
    else:
        # 如果有其他情况，可以选择跳过或记录日志
        return f"⏩ [{symbol}] 未知的代码格式，已跳过。"
    # --- 新增逻辑结束 ---
    log_prefix = f"[{symbol_for_api}]"  # 用于日志输出

    start_date_for_api: str
    end_date_for_api: str

    if target_date_manual_param:
        # 手动指定日期模式
        start_date_for_api = target_date_manual_param
        end_date_for_api = target_date_manual_param
        # print(f"🎯 {log_prefix} 指定日期模式，准备获取 {target_date_manual_param} 的数据")
    else:
        # 增量更新模式
        last_date_in_db = get_last_trade_date(symbol)

        default_list_date_api = "20100101"
        if pd.isna(list_date_from_db) or str(list_date_from_db).strip() == "":
            list_date_for_api = default_list_date_api
        else:
            try:
                list_date_for_api = pd.to_datetime(list_date_from_db).strftime("%Y%m%d")
            except Exception as e:
                print(
                    f"⚠️ {log_prefix} 解析上市日期 '{list_date_from_db}' 错误: {e}. 使用默认日期 {default_list_date_api}.")
                list_date_for_api = default_list_date_api

        if last_date_in_db:
            start_date_for_api = (last_date_in_db + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start_date_for_api = list_date_for_api

        end_date_for_api = today_str_param

        if start_date_for_api > end_date_for_api:
            return f"⏩ {log_prefix} 数据已是最新或起始日期 {start_date_for_api} > 结束日期 {end_date_for_api}。无需更新。"
        # print(f"⬇️  {log_prefix} 增量模式，准备从 {start_date_for_api} 更新到 {end_date_for_api}")

    # 调用 fetch_data
    # print(f"🌀 {log_prefix} 正在获取 {start_date_for_api} 到 {end_date_for_api} 的数据...") # 更详细的日志
    df = fetch_data(symbol_for_api, start_date_for_api, end_date_for_api)

    rows_written = 0
    if df is not None and not df.empty:
        try:
            # 使用 engine 进行数据库操作，通常是线程安全的
            df.to_sql("stock_daily", con=engine, index=False, if_exists="append")
            rows_written = len(df)
        except Exception as e:
            # 休眠一下，避免因DB错误导致连续快速重试（如果适用）
            time.sleep(SLEEP_PER_TASK)
            return f"❌ {log_prefix} 数据写入数据库失败 (针对日期 {start_date_for_api}-{end_date_for_api}): {e}"
    elif df is not None and df.empty:
        # 休眠一下，即使没有数据也保持一定的访问间隔
        time.sleep(SLEEP_PER_TASK)
        return f"ℹ️ {log_prefix} AkShare 未返回日期范围 {start_date_for_api}-{end_date_for_api} 的数据 (例如停牌、非交易日)。"
    else:  # df is None, fetch_data 内部已打印错误
        # 休眠一下
        time.sleep(SLEEP_PER_TASK)
        return f"❌ {log_prefix} fetch_data 执行失败 (针对日期 {start_date_for_api}-{end_date_for_api})。"

    # !!! 关键的休眠 !!!
    # 这个休眠在每个任务完成其主要工作后执行。
    # 5个worker，每个休眠0.3秒，意味着初始突发后，如果任务本身很快，
    # API的请求频率仍然会很高。
    time.sleep(SLEEP_PER_TASK)

    if rows_written > 0:
        return f"✅ {log_prefix} 成功写入 {rows_written} 行 (日期 {start_date_for_api}-{end_date_for_api})。"
    else:
        return f"ℹ️ {log_prefix} 没有新数据行被写入 (日期 {start_date_for_api}-{end_date_for_api})。"


# --- 主同步函数 (并发版本) ---
def sync_all_data_concurrent(target_date_manual: Optional[str] = None):
    """
    并发同步所有股票的日交易数据。
    target_date_manual: 如果提供，则为所有股票获取该特定日期的数据。否则，执行增量更新。
    """
    stock_df = get_stock_pool()
    today_str = datetime.now().strftime("%Y%m%d")  # 用于增量更新模式

    print(f"📊 待处理股票总数: {len(stock_df)}")
    if target_date_manual:
        try:
            datetime.strptime(target_date_manual, "%Y%m%d")  # 验证日期格式
            print(f"模式: 指定日期单日更新 (并发)，目标日期: {target_date_manual}")
        except ValueError:
            print(f"❌ 错误: 手动指定的目标日期 '{target_date_manual}' 格式无效，应为YYYYMMDD。脚本将退出。")
            return
    else:
        print(f"模式: 增量更新 (并发)，数据将尝试更新到日期: {today_str}")

    print(f"🚀 使用最多 {MAX_CONCURRENT_WORKERS} 个并发工作线程，每个任务完成后休眠 {SLEEP_PER_TASK} 秒。")
    print(f"⚠️ 警告: 当前设置可能导致非常高的API请求频率和初始并发突发，有被服务器限制的风险！请谨慎！")

    # 准备任务参数列表
    tasks_args_list = []
    for idx, row in stock_df.iterrows():
        tasks_args_list.append(((idx, row), today_str, target_date_manual))

    successful_updates = 0
    failed_updates = 0
    no_data_updates = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        # executor.submit(fn, *args, **kwargs)
        future_to_symbol = {
            executor.submit(process_single_stock, args[0], args[1], args[2]): args[0][1]['code']
            for args in tasks_args_list
        }

        count = 0
        total_tasks = len(future_to_symbol)
        for future in concurrent.futures.as_completed(future_to_symbol):
            count += 1
            symbol_completed = future_to_symbol[future]
            try:
                result_message = future.result()
                print(f"({count}/{total_tasks}) {result_message}")  # 打印每个任务的结果
                if "✅" in result_message:
                    successful_updates += 1
                elif "❌" in result_message:
                    failed_updates += 1
                else:  # "ℹ️" or "⏩"
                    no_data_updates += 1
            except Exception as exc:
                failed_updates += 1
                # 此处的异常通常是 process_single_stock 未能捕获的更深层次问题，或者 future 本身的问题
                print(f"❌ [{symbol_completed}] ({count}/{total_tasks}) 任务执行中产生未预料的严重异常: {exc}")

    print("\n--- 同步结果汇总 ---")
    print(f"🎉 所有 {len(tasks_args_list)} 只股票的处理任务已提交并完成。")
    print(f"   ✅ 成功更新/写入数据: {successful_updates} 只股票")
    print(f"   ℹ️ 无新数据或无需更新: {no_data_updates} 只股票")
    print(f"   ❌ 处理失败: {failed_updates} 只股票")


# --- 主执行部分 ---
if __name__ == "__main__":
    print("脚本开始执行...")




    sync_all_data_concurrent()

    # --- 如果要运行常规的增量更新，可以取消注释下面的行，并注释掉上面的测试模式调用 ---
    # print("\n>>> 常规模式：执行增量更新 <<<\n")
    # sync_all_data_concurrent(target_date_manual=None)

    print("\n脚本执行完毕。")