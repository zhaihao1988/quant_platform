# strategies/trend_strategies.py
import pandas as pd
from .base_strategy import BaseStrategy

class TurtleBreakout(BaseStrategy):
    """海龟交易突破：突破N日最高价时买入。"""
    def __init__(self, window=20):
        super().__init__(name="Turtle_Breakout")
        self.window = window

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        df['highest'] = df['high'].rolling(window=self.window).max()
        # 当日收盘价大于前N日最高价时买入
        df['signal'] = df['close'] > df['highest'].shift(1)
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals

class BollingerBreakout(BaseStrategy):
    """布林带突破：收盘价突破布林带上轨时买入。"""
    def __init__(self, period=20, num_std=2):
        super().__init__(name="Bollinger_Breakout")
        self.period = period; self.num_std = num_std

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        df['MB'] = df['close'].rolling(window=self.period).mean()
        df['STD'] = df['close'].rolling(window=self.period).std()
        df['UB'] = df['MB'] + self.num_std * df['STD']
        df['signal'] = (df['close'] > df['UB'])
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals

class ADXTrendStrength(BaseStrategy):
    """ADX趋势强度：当ADX指标超过阈值时发出信号。"""
    def __init__(self, period=14, threshold=25):
        super().__init__(name="ADX_StrongTrend")
        self.period = period; self.threshold = threshold

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        # 计算ADX指标 (简化版本)
        high = df['high']; low = df['low']; close = df['close']
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/self.period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/self.period).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.rolling(window=self.period).mean()
        df['ADX'] = adx
        df['signal'] = df['ADX'] > self.threshold
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals
