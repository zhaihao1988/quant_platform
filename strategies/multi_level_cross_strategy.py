# strategies/multi_level_cross_strategy.py
import pandas as pd
import numpy as np
from utils.data_loader import load_daily_data
from typing import List, Dict, Tuple


def calculate_ma(df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
    """计算各种均线并保留两位小数"""
    for ma in ma_list:
        df[f'MA{ma}'] = df['close'].rolling(window=ma).mean().round(2)
    return df


def is_ma_trending_up(ma_series: pd.Series, window: int = 5) -> bool:
    """判断均线是否走平或向上"""
    if len(ma_series) < window:
        return False
    # 计算最近window天的斜率
    x = np.arange(window)
    y = ma_series[-window:].values
    slope = np.polyfit(x, y, 1)[0]
    return slope >= 0  # 斜率为正表示向上


def detect_cross(df: pd.DataFrame, ma_list: List[int], check_ma30: bool = False, check_volume: bool = False) -> List[
    Dict]:
    """
    检测一阳穿四线信号
    :param check_ma30: 是否检查30日均线趋势
    :param check_volume: 是否检查成交量条件
    :return: 包含信号日期和详细信息的字典列表
    """
    signals = []
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # 检查是否一阳穿四线
        cross_condition = all(current['close'] > current[f'MA{ma}'] for ma in ma_list)
        below_condition = all(prev['close'] <= prev[f'MA{ma}'] for ma in ma_list)

        # 30日均线条件（保留两位小数比较）
        ma30_condition = True
        if check_ma30 and f'MA30' in df.columns:
            ma30_condition = round(current['MA30'], 2) >= round(prev['MA30'], 2)

        # 成交量条件（大于前一日50%以上）
        volume_condition = True
        if check_volume and 'volume' in df.columns:
            volume_condition = current['volume'] >= prev['volume'] * 1.5

        if cross_condition and below_condition and ma30_condition and volume_condition:
            # 处理日期列
            date_value = current.name if 'date' not in df.columns else current['date']
            signal_date = pd.to_datetime(date_value).strftime('%Y-%m-%d')

            # 收集信号详细信息
            signal_info = {
                'date': signal_date,
                'close': round(current['close'], 2),
                'volume': current.get('volume', 0),
                'volume_pct_change': round(
                    (current.get('volume', 0) - prev.get('volume', 0)) / prev.get('volume', 1) * 100,
                    2) if 'volume' in df.columns else 0,
                'cross_ma': [f'MA{ma}' for ma in ma_list]
            }

            # 添加MA30信息（如果存在）
            if f'MA30' in df.columns:
                signal_info['MA30'] = round(current['MA30'], 2)
                signal_info['MA30_change'] = round(current['MA30'] - prev['MA30'], 2)

            signals.append(signal_info)
    return signals


def daily_cross_strategy(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """日线级别一阳穿四线策略（增加30日均线和成交量条件）"""
    df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
    if df.empty:
        print(f"⚠️ 无数据：{symbol}")
        return []

    # 计算日线均线
    daily_ma = [5, 10, 20, 30]
    df = calculate_ma(df, daily_ma)
    df.dropna(inplace=True)

    # 检测日线一阳穿四线（启用30日均线和成交量检查）
    signals = detect_cross(df, daily_ma, check_ma30=True, check_volume=True)
    if signals:
        print(f"\n✅ {symbol} 日线一阳穿四线信号（30MA↑+放量50%↑）：")
        for sig in signals:
            print(f"  日期: {sig['date']} | 收盘价: {sig['close']} | 成交量: {sig['volume'] / 10000:.2f}万手")
            print(f"  30日均线: {sig['MA30']} ({'+' if sig['MA30_change'] >= 0 else ''}{sig['MA30_change']})")
            print(f"  成交量变化: {sig['volume_pct_change']}% | 上穿均线: {', '.join(sig['cross_ma'])}")
            print("  ---------------------------------")
    return signals


def weekly_cross_strategy(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """周线级别一阳穿四线策略，同时检查30周线趋势"""
    df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
    if df.empty:
        print(f"⚠️ 无数据：{symbol}")
        return []

    # 转换为周线数据
    df_weekly = df.set_index('date').resample('W-FRI').last().reset_index()
    df_weekly['close'] = df_weekly['close'].astype(float)
    df_weekly['volume'] = df_weekly['volume'].astype(float)

    # 计算周线均线
    weekly_ma = [5, 10, 20, 30]
    df_weekly = calculate_ma(df_weekly, weekly_ma)
    df_weekly.dropna(inplace=True)

    # 检查30周线是否走平或向上
    if len(df_weekly) >= 4 and not is_ma_trending_up(df_weekly['MA30'], window=4):
        return []

    # 检测周线一阳穿四线
    signals = detect_cross(df_weekly, weekly_ma, check_volume=True)
    if signals:
        print(f"\n✅ {symbol} 周线一阳穿四线信号（30MA↑+放量50%↑）：")
        for sig in signals:
            print(f"  日期: {sig['date']} | 收盘价: {sig['close']} | 成交量: {sig['volume'] / 10000:.2f}万手")
            print(f"  30周均线: {sig['MA30']} | 成交量变化: {sig['volume_pct_change']}%")
            print(f"  上穿均线: {', '.join(sig['cross_ma'])}")
            print("  ---------------------------------")
    return signals


def monthly_cross_strategy(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """月线级别一阳穿四线策略"""
    df = load_daily_data(symbol, start_date, end_date, fields=["date", "close", "volume"])
    if df.empty:
        print(f"⚠️ 无数据：{symbol}")
        return []

    # 转换为月线数据（使用ME替代已弃用的M）
    df_monthly = df.set_index('date').resample('ME').last().reset_index()
    df_monthly['close'] = df_monthly['close'].astype(float)
    df_monthly['volume'] = df_monthly['volume'].astype(float)

    # 计算月线均线
    monthly_ma = [3, 5, 10, 12]  # 月线使用不同的均线参数
    df_monthly = calculate_ma(df_monthly, monthly_ma)
    df_monthly.dropna(inplace=True)

    # 检测月线一阳穿四线
    signals = detect_cross(df_monthly, monthly_ma)
    if signals:
        print(f"\n✅ {symbol} 月线一阳穿四线信号：")
        for sig in signals:
            print(f"  日期: {sig['date']} | 收盘价: {sig['close']}")
            print(f"  上穿均线: {', '.join(sig['cross_ma'])}")
            print("  ---------------------------------")
    return signals


def multi_level_cross_strategy(symbol: str, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
    """
    多级别一阳穿四线策略
    返回: {
        "daily": 日线信号列表,
        "weekly": 周线信号列表,
        "monthly": 月线信号列表
    }
    """
    print(f"\n🔍 正在分析 {symbol} 的多级别一阳穿四线信号（{start_date} 至 {end_date}）...")
    results = {
        "daily": daily_cross_strategy(symbol, start_date, end_date),
        "weekly": weekly_cross_strategy(symbol, start_date, end_date),
        "monthly": monthly_cross_strategy(symbol, start_date, end_date)
    }
    return results


if __name__ == "__main__":
    symbol = "000001"
    start_date = "2010-01-01"
    end_date = "2025-04-26"

    signals = multi_level_cross_strategy(symbol, start_date, end_date)

    print("\n📊 最终信号汇总:")
    for level, sig_list in signals.items():
        if sig_list:
            print(f"\n{level.upper()}级别信号:")
            for sig in sig_list:
                print(f"  {sig['date']} - 收盘价: {sig['close']}", end="")
                if 'MA30' in sig:
                    print(f" | 30MA: {sig['MA30']}", end="")
                if 'volume_pct_change' in sig:
                    print(f" | 成交量: +{sig['volume_pct_change']}%", end="")
                print()