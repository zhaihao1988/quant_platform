# strategies/multi_level_cross_strategy.py
import os

import pandas as pd
import numpy as np
from typing import List, Dict
from utils.data_loader import load_daily_data
from strategies.base_strategy import BaseStrategy

class MultiLevelCrossStrategy(BaseStrategy):
    """多级别一阳穿四线策略"""

    def __init__(self, timeframe="multi"):
        super().__init__(name="MultiLevelCross", timeframe=timeframe)

    def calculate_ma(self, df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        """计算各种均线"""
        for ma in ma_list:
            df[f'MA{ma}'] = df['close'].rolling(window=ma).mean().round(2)
        return df

    def is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> bool:
        """判断均线是否走平或向上"""
        if len(ma_series) < window:
            return False
        x = np.arange(window)
        y = ma_series[-window:].values
        slope = np.polyfit(x, y, 1)[0]
        return slope >= 0

    def detect_cross(self, df: pd.DataFrame, ma_list: List[int], check_ma30=False, check_volume=False) -> List[Dict]:
        """检测一阳穿四线信号"""
        signals = []
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i - 1]

            cross_condition = all(current['close'] > current[f'MA{ma}'] for ma in ma_list)
            below_condition = all(prev['close'] <= prev[f'MA{ma}'] for ma in ma_list)

            ma30_condition = True
            if check_ma30 and 'MA30' in df.columns:
                ma30_condition = round(current['MA30'], 2) >= round(prev['MA30'], 2)

            volume_condition = True
            if check_volume and 'volume' in df.columns:
                volume_condition = current['volume'] >= prev['volume'] * 1.5

            if cross_condition and below_condition and ma30_condition and volume_condition:
                signal_date = pd.to_datetime(current['date']).strftime('%Y-%m-%d')
                signals.append({
                    'symbol': current.get('symbol', ''),   # 确保有 symbol 字段
                    'signal_date': signal_date,
                    'strategy': self.name,
                    'timeframe': self.timeframe,
                })
        return signals

    def process_daily(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """处理日线级别"""
        df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
        if df.empty:
            return []

        df = self.calculate_ma(df, [5, 10, 20, 30])
        df.dropna(inplace=True)
        signals = self.detect_cross(df, [5, 10, 20, 30], check_ma30=True, check_volume=True)
        return signals

    def process_weekly(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """处理周线级别"""
        df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
        if df.empty:
            return []

        df_weekly = df.set_index('date').resample('W-FRI').last().reset_index()
        df_weekly = self.calculate_ma(df_weekly, [5, 10, 20, 30])
        df_weekly.dropna(inplace=True)

        if len(df_weekly) >= 4 and not self.is_ma_trending_up(df_weekly['MA30'], window=4):
            return []

        signals = self.detect_cross(df_weekly, [5, 10, 20, 30], check_volume=True)
        return signals

    def process_monthly(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """处理月线级别"""
        df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
        if df.empty:
            return []

        df_monthly = df.set_index('date').resample('ME').last().reset_index()
        df_monthly = self.calculate_ma(df_monthly, [3, 5, 10, 12])
        df_monthly.dropna(inplace=True)

        signals = self.detect_cross(df_monthly, [3, 5, 10, 12])
        return signals

    def find_signals(self, symbol: str, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """统一接口：返回日线/周线/月线信号"""
        print(f"🔍 扫描 {symbol} 从 {start_date} 到 {end_date} 的多级别一阳穿四线...")
        return {
            "daily": self.process_daily(symbol, start_date, end_date),
            "weekly": self.process_weekly(symbol, start_date, end_date),
            "monthly": self.process_monthly(symbol, start_date, end_date),
        }

# 调试测试
if __name__ == "__main__":
    from db.database import get_engine
    from sqlalchemy import text

    strategy = MultiLevelCrossStrategy()

    # 1. 获取股票列表
    engine = get_engine()
    stock_query = "SELECT DISTINCT symbol FROM stock_daily"
    df_stocks = pd.read_sql(stock_query, con=engine)
    stock_list = df_stocks['symbol'].tolist()
    print(f"📈 股票数量: {len(stock_list)}")

    # 2. 获取日期范围
    date_query = "SELECT MAX(date) AS max_date FROM stock_daily"
    df_dates = pd.read_sql(date_query, con=engine)
    max_date = df_dates.at[0, 'max_date'].strftime('%Y-%m-%d')

    start_date = "2022-01-01"
    end_date = max_date
    print(f"📅 分析区间：{start_date} 到 {end_date}")
    print(f"🛎 只筛选发生在最后交易日 {end_date} 的信号")

    # 3. 遍历每只股票寻找信号
    total_signals = 0
    final_signals = []

    for symbol in stock_list:
        results = strategy.find_signals(symbol, start_date, end_date)
        for level, signals in results.items():
            for sig in signals:
                if sig['signal_date'] == end_date:
                    print(f"✅ {symbol} 在 {level.upper()} 级别发生信号: {sig}")
                    final_signals.append({
                        "symbol": symbol,
                        "signal_date": sig['signal_date'],
                        "strategy": strategy.name,
                        "timeframe": level
                    })
                    total_signals += 1

    print(f"\n🎯 最终找到 {total_signals} 个买点信号（全部发生在 {end_date}）")

    # 4. 可选：保存成CSV文件
    if final_signals:
        # 确保上一级目录的output文件夹存在
        os.makedirs('../output', exist_ok=True)  # 使用 `../` 表示上一级目录

        df_final = pd.DataFrame(final_signals)
        filename = f"../output/final_signals_{end_date}.csv"  # 保存在上一级目录的output下
        df_final.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存信号到 {filename}")
