# strategies/technical_strategy.py
import pandas as pd
from utils.data_loader import load_daily_data

def simple_ma_crossover(symbol: str, start_date: str, end_date: str):
    df = load_daily_data(symbol, start_date, end_date, fields=["date", "close"])
    if df.empty:
        print(f"⚠️ 无数据：{symbol}")
        return []

    df["SMA5"] = df["close"].rolling(window=5).mean()
    df["SMA20"] = df["close"].rolling(window=20).mean()
    df.dropna(inplace=True)

    df["signal"] = (df["SMA5"] > df["SMA20"]) & (df["SMA5"].shift(1) <= df["SMA20"].shift(1))
    signals = df[df["signal"]]["date"].tolist()
    print(f"✅ 买入信号：{symbol} 在以下日期：{signals}")
    return signals

if __name__ == "__main__":
    simple_ma_crossover("000001", "2023-01-01", "2023-12-31")
