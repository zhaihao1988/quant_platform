# strategies/multi_level_cross_refactored_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import date

# 假设这是您项目中更完善的 BaseStrategy 定义，与 AdaptedMAPullbackStrategy 使用的类似
# 如果实际的 base_strategy.py 与此不同，您可能需要调整或提供正确的基类定义
# 为保持一致性，我们这里假设的 BaseStrategy 包含 StrategyContext 和 StrategyResult
from .base_strategy import BaseStrategy, StrategyResult, StrategyContext

# 注意：原脚本中的 sys.path 修改和全局的 logging.basicConfig 已移除，
# 这些应由项目的入口点或主执行器统一管理。

logger = logging.getLogger(__name__)


class RefactoredMultiLevelCrossStrategy(BaseStrategy):
    """
    重构后的多级别一阳穿四线策略。
    该策略仅生成技术信号，不包含基本面分析。
    数据由外部提供，不再自行加载。
    """

    def __init__(self, context: StrategyContext):
        super().__init__(context)
        self.strategy_params = self.context.strategy_params.get(self.strategy_name, {})

        # 均线配置，可以从 context.strategy_params 中获取以增加灵活性
        self.ma_list_map = self.strategy_params.get("ma_list_map", {
            "daily": [5, 10, 20, 30],
            "weekly": [5, 10, 20, 30],
            "monthly": [3, 5, 10, 12]  # 原始脚本中的月线均线组合
        })
        # MA30 和成交量检查的配置也可以参数化
        self.check_ma30_daily = self.strategy_params.get("check_ma30_daily", True)
        self.check_volume_daily = self.strategy_params.get("check_volume_daily", True)
        self.check_volume_weekly = self.strategy_params.get("check_volume_weekly", True)
        # 周线级别MA30趋势检查（原脚本中隐含有此逻辑）
        self.check_ma30_trend_weekly = self.strategy_params.get("check_ma30_trend_weekly", True)

    @property
    def strategy_name(self) -> str:
        return "RefactoredMultiLevelCrossStrategy"

    def _calculate_ma(self, df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        """计算各种均线"""
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("无法计算MA：DataFrame为空或缺少'close'列。")
            # 如果df不为None，则尝试返回带有预期MA列的空DataFrame
            ma_cols = [f'MA{ma}' for ma in ma_list]
            existing_cols = df.columns.tolist() if df is not None else []
            return pd.DataFrame(columns=existing_cols + ma_cols)

        df_copy = df.copy()
        for ma in ma_list:
            # 确保有足够的非NaN值来计算窗口，否则结果为NaN
            df_copy[f'MA{ma}'] = df_copy['close'].rolling(window=ma, min_periods=ma).mean().round(2)
        return df_copy

    def _is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> bool:
        """判断均线是否走平或向上（基于线性回归斜率）"""
        if ma_series is None:
            logger.debug("用于趋势检查的MA序列为None。")
            return False

        valid_series = ma_series.dropna()  # 移除NaN值
        if len(valid_series) < 2:  # 至少需要两个点来确定趋势
            logger.debug(f"MA序列中有效数据点不足 ({len(valid_series)}) (最少需要2个) 来判断趋势。")
            return False

        effective_window = min(window, len(valid_series))
        if effective_window < 2:  # 调整后窗口大小仍需至少2个点
            logger.debug(f"有效窗口大小 ({effective_window}) 过小，无法计算趋势。")
            return False

        x = np.arange(effective_window)
        y = valid_series[-effective_window:].values  # 取最近的 effective_window 个有效数据点
        try:
            coeffs = np.polyfit(x, y, 1)  # 线性拟合 y = slope * x + intercept
            slope = coeffs[0]
            is_trending_up = slope >= -1e-6  # 允许非常小的负斜率，视为走平
            return is_trending_up
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"无法为MA序列拟合趋势线: {e}")
            return False

    def _detect_cross_signals_on_df(self, df: pd.DataFrame, ma_list: List[int], level: str, symbol: str) -> List[
        StrategyResult]:
        """在处理好（已重采样并计算完MA）的DataFrame上检测金叉信号"""
        signals = []
        # 根据级别确定是否检查MA30趋势和成交量
        check_ma30_trend_cond = (level == "daily" and self.check_ma30_daily) or \
                                (level == "weekly" and self.check_ma30_trend_weekly)
        check_volume_cond = (level == "daily" and self.check_volume_daily) or \
                            (level == "weekly" and self.check_volume_weekly)

        required_ma_cols = [f'MA{ma}' for ma in ma_list]
        if check_ma30_trend_cond and 'MA30' not in required_ma_cols:  # 确保MA30被计算
            # 这通常意味着MA30应该在ma_list中，或者额外计算
            # 为简化，我们假设如果check_ma30_trend_cond为True，则MA30已被包含在df的列中
            if 'MA30' not in df.columns:
                logger.warning(f"[{symbol}-{level}] 需要检查MA30趋势，但MA30列未计算。")
                return signals

        if df is None or df.empty or len(df) < 2:
            logger.debug(f"[{symbol}-{level}] DataFrame为空或过短，无法检测金叉。")
            return signals
        if not all(col in df.columns for col in required_ma_cols):
            missing_cols = [col for col in required_ma_cols if col not in df.columns]
            logger.warning(f"[{symbol}-{level}] 缺少必要的MA列: {missing_cols}。跳过金叉检测。")
            return signals

        # 确保 'close', 'date' 存在，以及 'volume' (如果需要检查)
        data_cols_to_check = ['close', 'date']
        if check_volume_cond: data_cols_to_check.append('volume')
        if not all(col in df.columns for col in data_cols_to_check):
            missing_cols = [col for col in data_cols_to_check if col not in df.columns]
            logger.warning(f"[{symbol}-{level}] 缺少必要的数据列: {missing_cols}。跳过金叉检测。")
            return signals

        # 从有足够数据可以比较前一天开始迭代 (索引1)
        # MA的有效性由min_periods=ma保证，这里只需确保能取到prev
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i - 1]

            # 核心数据有效性检查
            if pd.isna(current['close']) or pd.isna(prev['close']): continue
            # 检查当前行和前一行的所有相关MA值是否有效
            current_mas_valid = not any(pd.isna(current.get(f'MA{ma}')) for ma in ma_list)
            prev_mas_valid = not any(pd.isna(prev.get(f'MA{ma}')) for ma in ma_list)
            if not current_mas_valid or not prev_mas_valid: continue

            # 核心金叉条件
            all_mas_crossed = all(current['close'] > current.get(f'MA{ma}', np.inf) for ma in ma_list)
            all_prev_below = all(prev['close'] <= prev.get(f'MA{ma}', -np.inf) for ma in ma_list)

            if not (all_mas_crossed and all_prev_below): continue

            # MA30趋势条件 (日线级别MA30向上，周线级别MA30走平或向上)
            ma30_trend_ok = True
            if check_ma30_trend_cond:
                current_ma30 = current.get('MA30')
                prev_ma30 = prev.get('MA30')  # 注意：对于周线，这指的是上一周的MA30
                if pd.notna(current_ma30) and pd.notna(prev_ma30):
                    if level == "daily":  # 日线要求MA30严格上升
                        ma30_trend_ok = round(current_ma30, 2) > round(prev_ma30, 2)  # 原版可能是 >=
                    elif level == "weekly":  # 周线MA30走平或向上 (使用 is_ma_trending_up)
                        # is_ma_trending_up 需要一个Series。这里我们只有两个点。
                        # 或者，如果周线MA30趋势检查是用 is_ma_trending_up 做的，它应该在 process_level 中完成
                        # 这里简化为 current_ma30 >= prev_ma30
                        ma30_trend_ok = round(current_ma30, 2) >= round(prev_ma30, 2)  # 保持与原脚本一致，用前后两天比较
                else:
                    ma30_trend_ok = False

            if not ma30_trend_ok: continue

            # 成交量条件 (日线和周线级别)
            volume_ok = True
            if check_volume_cond:
                current_volume = current.get('volume')
                prev_volume = prev.get('volume')
                if pd.notna(current_volume) and pd.notna(prev_volume) and prev_volume > 1e-6:  # 避免除以零或无效比较
                    volume_ok = current_volume >= prev_volume * 1.5  # 原脚本的成交量放大条件
                else:
                    volume_ok = False

            if not volume_ok: continue

            # 如果所有条件都满足
            signal_date_obj = pd.to_datetime(current['date']).date()  # 确保是 date 对象
            signal_details = {
                "level": level,
                "close": f"{current['close']:.2f}",
                "mas_crossed": ", ".join([f"MA{m}={current.get(f'MA{m}', np.nan):.2f}" for m in ma_list])
            }
            if check_ma30_trend_cond: signal_details["MA30"] = f"{current.get('MA30', np.nan):.2f}"
            if check_volume_cond: signal_details["volume"] = f"{current.get('volume', np.nan):.0f}"

            signals.append(StrategyResult(
                symbol=symbol,
                signal_date=signal_date_obj,
                strategy_name=self.strategy_name,
                signal_type="BUY",
                details=signal_details
            ))
        return signals

    def run_for_stock(self, symbol: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[StrategyResult]:
        """为单只股票在指定当前日期执行策略逻辑。"""
        all_level_signals: List[StrategyResult] = []

        daily_df_full_history = data.get("daily")  # 完整历史日线数据
        if daily_df_full_history is None or daily_df_full_history.empty:
            logger.warning(f"[{symbol}@{current_date.isoformat()}] {self.strategy_name}: 无日线数据。")
            return all_level_signals

        # 确保 'date' 列存在，并转换为 datetime 对象以便正确处理
        if 'date' not in daily_df_full_history.columns:
            logger.error(f"[{symbol}@{current_date.isoformat()}] {self.strategy_name}: 日线数据缺少 'date' 列。")
            return all_level_signals

        # 创建副本并处理日期
        df_daily_processed = daily_df_full_history.copy()
        df_daily_processed['date'] = pd.to_datetime(df_daily_processed['date'])

        # 筛选出 current_date 当天及之前的历史数据用于分析和信号生成
        # 信号只在 current_date 当天产生
        df_for_analysis = df_daily_processed[df_daily_processed['date'].dt.date <= current_date].copy()
        if df_for_analysis.empty:
            logger.info(
                f"[{symbol}@{current_date.isoformat()}] {self.strategy_name}: 无截至当前日期的历史日线数据。")
            return all_level_signals

        for level in ["daily", "weekly", "monthly"]:
            logger.debug(f"[{symbol}@{current_date.isoformat()}] {self.strategy_name}: 处理 {level} 级别...")

            df_level_specific = df_for_analysis.copy()  # 从截至 current_date 的日线数据开始处理每个级别

            # 数据重采样以获取周线和月线数据
            if level == "weekly":
                if len(df_level_specific) < 5:  # 需要至少5天数据才能形成一个周线点（粗略估计）
                    logger.debug(f"[{symbol}-{level}] 日线数据不足 ({len(df_level_specific)}) 无法生成周线。")
                    continue
                aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                try:
                    df_level_specific = df_level_specific.set_index('date').resample('W-FRI').agg(aggregation).dropna(
                        how='all').reset_index()
                except Exception as e:
                    logger.error(f"[{symbol}-{level}] 周线重采样失败: {e}", exc_info=True)
                    continue
            elif level == "monthly":
                if len(df_level_specific) < 20:  # 需要至少约20天数据才能形成一个月线点
                    logger.debug(f"[{symbol}-{level}] 日线数据不足 ({len(df_level_specific)}) 无法生成月线。")
                    continue
                aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                try:
                    df_level_specific = df_level_specific.set_index('date').resample('ME').agg(aggregation).dropna(
                        how='all').reset_index()  # 'M' for month end
                except Exception as e:
                    logger.error(f"[{symbol}-{level}] 月线重采样失败: {e}", exc_info=True)
                    continue

            if df_level_specific.empty:
                logger.warning(f"[{symbol}-{level}] 重采样后的 {level} 级别DataFrame为空。")
                continue

            # 计算MA
            ma_config_for_level = self.ma_list_map[level]
            # 确保计算所有可能需要的MA，例如日线和周线都可能需要MA30趋势
            ma_to_calculate = list(set(ma_config_for_level + ([30] if level in ["daily", "weekly"] else [])))
            df_with_ma = self._calculate_ma(df_level_specific, ma_to_calculate)

            if df_with_ma.empty:
                logger.warning(f"[{symbol}-{level}] 计算MA后DataFrame为空。")
                continue

            # 特殊的周线MA30趋势判断 (如果配置了)
            if level == 'weekly' and self.check_ma30_trend_weekly:
                if 'MA30' in df_with_ma.columns and not df_with_ma['MA30'].dropna().empty:
                    if len(df_with_ma['MA30'].dropna()) >= 4:  # 需要至少4个点（原脚本用4）来判断周线MA30趋势
                        if not self._is_ma_trending_up(df_with_ma['MA30'], window=4):
                            logger.debug(f"[{symbol}-{level}] 周线MA30未形成上升趋势，跳过此级别。")
                            continue  # 周线MA30趋势不满足，则不在此周线级别产生信号
                    else:
                        logger.debug(f"[{symbol}-{level}] 周线MA30有效数据点不足 (<4)，无法判断趋势。")
                        continue  # 数据不足以判断趋势
                else:
                    logger.debug(f"[{symbol}-{level}] 周线MA30未计算或无有效值。")
                    continue  # MA30不存在或无效

            # 检测信号 (只关心在 current_date 产生的信号)
            level_signals = self._detect_cross_signals_on_df(df_with_ma, ma_config_for_level, level, symbol)

            for sig_result in level_signals:
                # _detect_cross_signals_on_df 返回的 signal_date 已经是 date 对象
                if sig_result.signal_date == current_date:
                    all_level_signals.append(sig_result)
                    logger.info(
                        f"策略 {self.strategy_name} 为股票 {symbol} 在日期 {sig_result.signal_date.isoformat()} ({level}级别) 生成买入信号。")

        return all_level_signals