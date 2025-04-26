# multi_strategy_screener.py
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from db.database import get_engine
from strategies.factor_strategy import multi_factor_select
from strategies.fundamental_strategy import fundamental_filter
from strategies.technical_strategy import simple_ma_crossover
from strategies.multi_level_cross_strategy import multi_level_cross_strategy


def get_recent_trading_days(days: int = 2) -> List[str]:
    """获取最近N个交易日"""
    engine = get_engine()
    query = """
    SELECT DISTINCT date FROM stock_daily 
    WHERE date <= CURRENT_DATE
    ORDER BY date DESC
    LIMIT %s
    """
    df = pd.read_sql(query, con=engine, params=[days])
    return df['date'].dt.strftime('%Y-%m-%d').tolist()


def get_stock_list() -> List[str]:
    """获取所有股票列表"""
    engine = get_engine()
    query = "SELECT DISTINCT symbol FROM stock_daily"
    df = pd.read_sql(query, con=engine)
    return df['symbol'].tolist()


def run_factor_strategy(date: str) -> Dict[str, List[str]]:
    """运行因子策略"""
    print(f"\n🔍 运行因子策略(动量+规模) @ {date}")
    selected = multi_factor_select(date)
    return {"factor": selected}


def run_fundamental_strategy() -> Dict[str, List[str]]:
    """运行基本面策略"""
    print("\n🔍 运行基本面策略(低PE+高净利润)")
    selected = fundamental_filter()
    return {"fundamental": selected}


def run_technical_strategy(symbols: List[str], dates: List[str]) -> Dict[str, List[str]]:
    """运行技术面策略"""
    print("\n🔍 运行技术面策略(均线交叉)")
    results = {"technical": []}
    for symbol in symbols:
        for date in dates:
            signals = simple_ma_crossover(symbol, date, date)
            if signals:
                results["technical"].append(symbol)
                break  # 只要有一个信号就记录
    return results


def run_cross_strategy(symbols: List[str], dates: List[str]) -> Dict[str, List[str]]:
    """运行一阳穿四线策略"""
    print("\n🔍 运行多级别一阳穿四线策略")
    results = {"cross_daily": [], "cross_weekly": [], "cross_monthly": []}

    for symbol in symbols:
        # 检查最近2个交易日是否有信号
        start_date = (pd.to_datetime(dates[-1]) - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = dates[-1]

        try:
            signals = multi_level_cross_strategy(symbol, start_date, end_date)

            # 检查日线信号
            if signals["daily"] and any(sig["date"] in dates for sig in signals["daily"]):
                results["cross_daily"].append(symbol)

            # 检查周线信号
            if signals["weekly"] and any(sig["date"] in dates for sig in signals["weekly"]):
                results["cross_weekly"].append(symbol)

            # 检查月线信号
            if signals["monthly"] and any(sig["date"] in dates for sig in signals["monthly"]):
                results["cross_monthly"].append(symbol)

        except Exception as e:
            print(f"⚠️ 处理{symbol}时出错: {e}")

    return results


def combine_results(all_results: List[Dict[str, List[str]]]) -> pd.DataFrame:
    """合并所有策略结果"""
    combined = {}

    # 初始化所有股票
    all_stocks = set()
    for result in all_results:
        for stocks in result.values():
            all_stocks.update(stocks)

    # 创建结果字典
    for stock in all_stocks:
        combined[stock] = []
        for result in all_results:
            for strategy, stocks in result.items():
                if stock in stocks:
                    combined[stock].append(strategy)

    # 转换为DataFrame
    df = pd.DataFrame.from_dict(combined, orient='index', columns=['策略'])
    df.index.name = '股票代码'
    df['策略'] = df['策略'].apply(lambda x: ', '.join(x))
    return df.sort_index()


def main():
    print("🚀 开始多策略选股扫描...")

    # 获取最近2个交易日
    dates = get_recent_trading_days(2)
    print(f"📅 分析日期范围: {', '.join(dates)}")

    # 获取股票列表
    all_stocks = get_stock_list()
    print(f"📊 待分析股票数量: {len(all_stocks)}")

    # 运行各种策略
    results = []
    '''
    # 因子策略(使用最近一个交易日)
    results.append(run_factor_strategy(dates[-1]))

    # 基本面策略
    results.append(run_fundamental_strategy())

    # 技术面策略(均线交叉)
    results.append(run_technical_strategy(all_stocks, dates))
    '''
    # 一阳穿四线策略
    results.append(run_cross_strategy(all_stocks, dates))

    # 合并结果
    final_df = combine_results(results)

    # 保存结果
    output_file = f"strategy_results_{datetime.now().strftime('%Y%m%d')}.csv"
    final_df.to_csv(output_file)
    print(f"\n🎉 分析完成! 结果已保存到 {output_file}")
    print("\n📋 结果预览:")
    print(final_df.head(20))


if __name__ == "__main__":
    main()