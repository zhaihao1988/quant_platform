# multi_strategy_screener.py
import pandas as pd
from datetime import datetime
from typing import List, Dict
from db.database import get_engine
from strategies.factor_strategy import multi_factor_select
from strategies.fundamental_strategy import fundamental_filter
from strategies.technical_strategy import simple_ma_crossover
from strategies.multi_level_cross_strategy import MultiLevelCrossStrategy

from utils.db_utils import insert_signal_results
from utils.push_utils import pushplus_send_message

def get_trading_date_bounds() -> Dict[str, str]:
    """
    获取市场的最早交易日和最新交易日
    返回：{'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}
    """
    engine = get_engine()
    df = pd.read_sql("SELECT MIN(date) AS start, MAX(date) AS end FROM stock_daily", con=engine)
    start = df.at[0, 'start'].strftime('%Y-%m-%d')
    end = df.at[0, 'end'].strftime('%Y-%m-%d')
    return {"start_date": start, "end_date": end}

def get_stock_list() -> List[str]:
    """获取全市场股票列表（不剔除任何）"""
    engine = get_engine()
    df = pd.read_sql("SELECT DISTINCT symbol FROM stock_daily", con=engine)
    return df['symbol'].tolist()

def run_factor_strategy(end_date: str) -> List[str]:
    print(f"\n🔍 因子策略 全历史至 {end_date}")
    return multi_factor_select(end_date)

def run_fundamental_strategy() -> List[str]:
    print("\n🔍 基本面策略 全历史")
    return fundamental_filter()

def run_technical_strategy(symbols: List[str], start_date: str, end_date: str) -> List[str]:
    print(f"\n🔍 技术面策略(均线交叉) 从 {start_date} 到 {end_date}")
    hits = []
    for sym in symbols:
        sigs = simple_ma_crossover(sym, start_date, end_date)
        if sigs:
            hits.append(sym)
    return hits


def run_cross_strategy(symbols: List[str], start_date: str, end_date: str) -> Dict[str, List[str]]:
    print(f"\n🔍 一阳穿四线多级别策略 从 {start_date} 到 {end_date}")

    strategy = MultiLevelCrossStrategy()
    daily_hits, weekly_hits, monthly_hits = [], [], []

    for sym in symbols:
        results = strategy.find_signals(sym, start_date, end_date)
        if results.get('daily'):
            daily_hits.append(sym)
        if results.get('weekly'):
            weekly_hits.append(sym)
        if results.get('monthly'):
            monthly_hits.append(sym)

    return {
        "cross_daily": daily_hits,
        "cross_weekly": weekly_hits,
        "cross_monthly": monthly_hits
    }


def main():
    print("🚀 开始多策略选股扫描...")

    # 交易日区间
    bounds = get_trading_date_bounds()
    start_date, end_date = bounds['start_date'], bounds['end_date']
    print(f"🗓 分析区间：{start_date} 至 {end_date}")

    # 全市场股票
    stocks = get_stock_list()
    print(f"📊 待分析股票数: {len(stocks)}")

    # 策略映射
    strategy_funcs = {
        "factor":      lambda: run_factor_strategy(end_date),
        "fundamental": lambda: run_fundamental_strategy(),
        "technical":   lambda: run_technical_strategy(stocks, start_date, end_date),
        "cross_daily":   lambda: run_cross_strategy(stocks, start_date, end_date)["cross_daily"],
        "cross_weekly":  lambda: run_cross_strategy(stocks, start_date, end_date)["cross_weekly"],
        "cross_monthly": lambda: run_cross_strategy(stocks, start_date, end_date)["cross_monthly"],
    }

    # ==== 在这里指定要运行的策略 Key 列表 ====
    selected_keys = ["cross_daily", "cross_weekly", "cross_monthly"]
    # 可改为 e.g. ["factor","technical"] 或全部 list(strategy_funcs.keys())

    records = []
    for key in selected_keys:
        if key not in strategy_funcs:
            print(f"⚠️ 未知策略: {key}")
            continue
        symbols_hit = strategy_funcs[key]()
        print(f"  策略 {key} 命中 {len(symbols_hit)} 支股票")
        for sym in symbols_hit:
            records.append({
                "signal_date": end_date,
                "strategy": key,
                "symbol": sym
            })

    if not records:
        print("❗ 未命中任何信号，退出。")
        return

    # 转 DataFrame 并写库
    df_res = pd.DataFrame(records)
    insert_signal_results(df_res)
    print(f"✅ 写入 {len(df_res)} 条信号到 signal_results 表")

    # 生成 CSV 报表
    report = f"strategy_results_{end_date}.csv"
    df_res.to_csv(report, index=False)
    print(f"✅ 报表已保存：{report}")
    '''
    # 微信推送
    content = f"{end_date} 选股信号共 {len(df_res)} 条，详情请查看报告。"
    pushplus_send_message(content)
    print("📨 微信推送完成。")
    '''
if __name__ == "__main__":
    main()
