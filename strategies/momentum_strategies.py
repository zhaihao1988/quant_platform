# strategies/momentum_strategies.py
import pandas as pd
from .base_strategy import BaseStrategy

class RSIOversoldRebound(BaseStrategy):
    """RSI超卖反弹：RSI<30后向上回升时买入。"""
    def __init__(self, period=14, threshold=30):
        super().__init__(name="RSI_Oversold")
        self.period = period
        self.threshold = threshold

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        # 计算RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(com=self.period-1, adjust=False).mean()
        ema_down = down.ewm(com=self.period-1, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - 100 / (1 + rs)
        # RSI从下向上穿过阈值时为买点
        df['signal'] = (df['RSI'] < self.threshold) & (df['RSI'].shift(1) < self.threshold) & (df['RSI'] > df['RSI'].shift(1))
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals

class MACDBullishDivergence(BaseStrategy):
    """MACD底背离：价格创新低而MACD指标不创新低。"""
    def __init__(self, short=12, long=26, signal=9):
        super().__init__(name="MACD_Divergence")
        self.short = short; self.long = long; self.signal = signal

    def find_signals(self, df):
        df = df.sort_values("date").copy()
        # 计算MACD线（DIFF）和信号线（DEA）
        ema_short = df['close'].ewm(span=self.short, adjust=False).mean()
        ema_long = df['close'].ewm(span=self.long, adjust=False).mean()
        df['DIFF'] = ema_short - ema_long
        df['DEA'] = df['DIFF'].ewm(span=self.signal, adjust=False).mean()
        df['MACD_bar'] = 2 * (df['DIFF'] - df['DEA'])
        # 简单示例：找价格新低而MACD柱线不创新低的点
        df['low_low'] = df['close'].rolling(window=5).apply(lambda x: x.argmin() == len(x)-1)
        df['macd_low'] = df['MACD_bar'].rolling(window=5).apply(lambda x: x.argmin() == len(x)-1)
        df['signal'] = df['low_low'] & (~df['macd_low'])
        buy_dates = df[df['signal']]['date']
        signals = pd.DataFrame({
            "symbol": df.loc[buy_dates.index, "symbol"],
            "signal_date": buy_dates.values,
            "strategy": self.name,
            "timeframe": self.timeframe
        })
        return signals
