# strategies/base_strategy.py
import pandas as pd

class BaseStrategy:
    """策略基类：定义接口和公共方法。"""
    name = "BaseStrategy"
    timeframe = "daily"  # 默认周期

    def __init__(self, name=None, timeframe="daily"):
        if name: self.name = name
        self.timeframe = timeframe

    def find_signals(self, df):
        """
        计算买点信号。输入交易数据 df（包含日期、开盘、收盘、最高、最低等列），
        输出包含信号日期的 DataFrame，列至少包含 ['symbol','signal_date','strategy','timeframe']。
        """
        raise NotImplementedError("请在子类中实现具体策略逻辑")
