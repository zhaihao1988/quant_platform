# strategies/ma_strategies.py
import pandas as pd
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """简单均线交叉：短期均线上穿长期均线时买入。"""
    def __init__(self, short_window=5, long_window=20):
        super().__init__(name="MA_Crossover")
        self.short_window = short_window
        self.long_window = long_window

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        df[f"SMA{self.short_window}"] = df["close"].rolling(window=self.short_window).mean()
        df[f"SMA{self.long_window}"] = df["close"].rolling(window=self.long_window).mean()
        df.dropna(inplace=True)
        # 短期均线上穿长期均线时为买点
        df['signal'] = (df[f"SMA{self.short_window}"] > df[f"SMA{self.long_window}"]) & \
                       (df[f"SMA{self.short_window}"].shift(1) <= df[f"SMA{self.long_window}"].shift(1))
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals

class MovingAverageBullish(BaseStrategy):
    """均线多头排列：短中长期均线呈多头排列时买入信号。"""
    def __init__(self, windows=(5,10,20)):
        super().__init__(name="MA_Bullish")
        self.windows = windows

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        for w in self.windows:
            df[f"SMA{w}"] = df["close"].rolling(window=w).mean()
        df.dropna(inplace=True)
        # 条件：短期均线 > 中期均线 > 长期均线
        cond = True
        sorted_windows = sorted(self.windows)
        for i in range(len(sorted_windows)-1):
            cond &= (df[f"SMA{sorted_windows[i]}"] > df[f"SMA{sorted_windows[i+1]}"])
        df['signal'] = cond
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals
