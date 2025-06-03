# strategies/base_strategy.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import date
from sqlalchemy.orm import Session # 用于 StrategyContext
from dataclasses import dataclass, field

@dataclass
class StrategyContext:
    """
    策略执行的上下文环境。
    """
    db_session: Session                             # 数据库会话
    current_date: date                              # 当前分析日期 (策略应基于此日期的数据产生信号)
    strategy_params: Dict[str, Any] = field(default_factory=dict) # 包含所有策略的参数字典
    # stock_list: Optional[List[str]] = None        # 扫描的股票列表 (可选)
    # data_loader_params: Dict[str, Any] = field(default_factory=dict) # 数据加载器参数 (可选)
    # metadata: Dict[str, Any] = field(default_factory=dict) # 其他元数据 (可选)

@dataclass
class StrategyResult:
    """
    策略信号的标准化输出格式。
    """
    symbol: str                                 # 股票代码
    signal_date: date                               # 信号产生的日期
    strategy_name: str                              # 产生信号的策略名称
    signal_type: str = "BUY"                        # 信号类型 (e.g., "BUY", "SELL", "HOLD")
    signal_score: Optional[float] = None            # 信号评分 (可选)
    details: Dict[str, Any] = field(default_factory=dict) # 包含策略特定信息的字典 (例如，信号级别、关键指标值等)
    notes: Optional[str] = None
class BaseStrategy(ABC):
    """
    多策略系统中策略的抽象基类。
    """
    def __init__(self, context: StrategyContext):
        self.context = context
        # 从 context.strategy_params 中获取特定于此策略的参数
        # self.params = self.context.strategy_params.get(self.strategy_name(), {})

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """策略的唯一名称。"""
        pass

    def preload_data(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        可选的预加载数据方法。
        如果策略执行器 (如 multi_strategy_screener.py) 负责数据加载，
        此方法可以为空或用于加载一些策略特定的、不常用的数据。
        返回一个字典，键是股票代码，值是包含不同时间周期DataFrame的字典。
        例如: {'000001': {'daily': df_daily, 'weekly': df_weekly}}
        """
        # 默认不执行任何操作，假设数据由调用者通过 run_for_stock 传入
        return {}

    @abstractmethod
    def run_for_stock(self, symbol: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[StrategyResult]:
        """
        为单个股票在指定的当前日期执行策略逻辑。

        参数:
        - symbol (str): 股票代码。
        - current_date (date): 当前分析的日期。策略应基于此日期的数据判断是否产生信号。
        - data (Dict[str, pd.DataFrame]): 一个字典，包含该股票所需时间周期的数据。
          键是时间周期 (例如 "daily", "weekly", "monthly")，值是对应的Pandas DataFrame。
          DataFrame 应包含 'date', 'open', 'high', 'low', 'close', 'volume' 等列，
          并且按日期升序排列。'date' 列应为 datetime 类型。

        返回:
        - List[StrategyResult]: 在 current_date 为该股票产生的所有信号的列表。
                                如果没有信号，则返回空列表。
        """
        pass