# strategies/ma_pullback_strategy_enhanced.py
import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date

# --- Python Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger_strat = logging.getLogger(__name__)  # Use a distinct logger name
    logger_strat.info(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
else:
    logger_strat = logging.getLogger(__name__)
# --- End Path Setup ---

try:
    from utils.data_loader import load_daily_data
    # from data_processing.loader import load_multiple_financial_reports # 可能不需要在策略类中直接加载财报做过滤
    from strategies.base_strategy import BaseStrategy
    # from db.database import get_engine_instance # 通常回测框架会提供数据
    import talib  # 用于MACD和ATR

    TALIB_AVAILABLE = True
    logger_strat.info("TALIB imported successfully.")
except ImportError as e:
    logger_strat.warning(f"Failed to import TALIB: {e}. MACD and ATR functionalities will be limited.")
    TALIB_AVAILABLE = False

# --- Constants Definition ---
PEAK_WINDOW = 5
MA_PEAK_THRESHOLD = 1.20
MA_LONG_PERIOD_FOR_PEAK = 30
PULLBACK_THRESHOLD = 0.05
MA_SHORT_PERIOD = 5
MA_LONG_PERIOD = 30  # 主要的回踩参考均线和趋势判断均线
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SELL_STOP_LOSS_MA30_PCT = 0.03
SELL_VOLUME_RATIO = 1.5

# MACD Parameters
MACD_FASTPERIOD = 12
MACD_SLOWPERIOD = 26
MACD_SIGNALPERIOD = 9


class MAPullbackEnhancedStrategy(BaseStrategy):
    def __init__(self, ma_short=MA_SHORT_PERIOD, ma_long=MA_LONG_PERIOD,
                 pullback_pct=PULLBACK_THRESHOLD, trend_window=5,
                 peak_window=PEAK_WINDOW, ma_peak_threshold=MA_PEAK_THRESHOLD,
                 ma_long_for_peak=MA_LONG_PERIOD_FOR_PEAK,
                 atr_period=ATR_PERIOD, atr_multiplier=ATR_MULTIPLIER,
                 sell_ma30_pct=SELL_STOP_LOSS_MA30_PCT, sell_volume_ratio=SELL_VOLUME_RATIO,
                 timeframe="daily"):  # timeframe 现在主要用于区分信号来源

        super().__init__(name=f"MAPullbackPeakCondEnhanced", timeframe=timeframe)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.pullback_pct = pullback_pct
        self.trend_window = trend_window
        self.peak_window = peak_window
        self.ma_peak_threshold = ma_peak_threshold
        self.ma_long_for_peak = ma_long_for_peak  # 通常与 self.ma_long 一致

        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.sell_ma30_pct = sell_ma30_pct
        self.sell_volume_ratio = sell_volume_ratio
        # Note: Fundamental data fetching is removed from here, assumed to be pre-filtered or handled by RQAlpha context if needed

    def calculate_ma(self, df_in: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        df = df_in.copy()
        if df.empty or 'close' not in df.columns:
            logger_strat.warning("MA Calc: DataFrame empty or no 'close' column.")
            return df
        for ma in ma_list:
            if len(df) >= ma:
                df[f'MA{ma}'] = talib.SMA(df['close'], timeperiod=ma).round(2) if TALIB_AVAILABLE else df[
                    'close'].rolling(window=ma, min_periods=ma).mean().round(2)
            else:
                df[f'MA{ma}'] = np.nan
        return df

    def calculate_atr(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        if not TALIB_AVAILABLE:
            logger_strat.warning("TALIB not available, ATR calculation skipped.")
            df['atr'] = np.nan
            return df
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger_strat.warning("ATR Calc: DataFrame empty or missing HLC columns.")
            df['atr'] = np.nan
            return df
        if len(df) > self.atr_period:  # talib.ATR needs length > timeperiod
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        else:
            df['atr'] = np.nan
        return df

    def calculate_macd(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        if not TALIB_AVAILABLE:
            logger_strat.warning("TALIB not available, MACD calculation skipped.")
            df['macd'], df['macdsignal'], df['macdhist'] = np.nan, np.nan, np.nan
            return df
        if df.empty or 'close' not in df.columns:
            logger_strat.warning("MACD Calc: DataFrame empty or no 'close' column.")
            df['macd'], df['macdsignal'], df['macdhist'] = np.nan, np.nan, np.nan
            return df

        # talib.MACD needs length > slowperiod + signalperiod -1, roughly
        required_len_macd = MACD_SLOWPERIOD + MACD_SIGNALPERIOD
        if len(df) > required_len_macd:
            macd, macdsignal, macdhist = talib.MACD(df['close'],
                                                    fastperiod=MACD_FASTPERIOD,
                                                    slowperiod=MACD_SLOWPERIOD,
                                                    signalperiod=MACD_SIGNALPERIOD)
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist  # MACD柱
        else:
            df['macd'], df['macdsignal'], df['macdhist'] = np.nan, np.nan, np.nan
        return df

    def is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> Optional[bool]:
        if ma_series is None: return None
        valid_series = ma_series.dropna()
        if len(valid_series) < 2: return None
        effective_window = min(window, len(valid_series))
        if effective_window < 2: return None
        x = np.arange(effective_window);
        y = valid_series[-effective_window:].values
        try:
            coeffs = np.polyfit(x, y, 1);
            slope = coeffs[0]
            return slope >= -1e-6
        except (np.linalg.LinAlgError, ValueError):
            return None

    def find_signal_relevant_peak_and_low(self, df_history: pd.DataFrame, ma_long_col: str) -> Dict[
        str, Optional[float]]:
        """
        Identifies the 'prior_peak_price' and 'recent_low_price' based on historical data up to the signal day.
        This function is called ONCE when a buy signal is being confirmed.
        'df_history' should be the data available up to and including the signal day (T-1).
        """
        results = {"prior_peak_price": None, "recent_low_price": None}
        if len(df_history) < max(self.peak_window, self.ma_long_for_peak, 50):  # Need enough data
            logger_strat.debug("Not enough historical data to find peak and low.")
            return results

        # 1. Identify the "Prior Peak"
        # Calculate rolling max for 'high' prices to identify peaks
        df_history['rolling_peak_val'] = df_history['high'].rolling(window=self.peak_window, center=True,
                                                                    min_periods=1).max()

        # Iterate backwards from one day before the last day of df_history (signal day)
        # to find the most recent peak that meets the criteria.
        peak_found_idx = -1
        for i in range(len(df_history) - 2, self.peak_window - 2, -1):
            # Ensure MA_long_for_peak is available at index i
            if f'MA{self.ma_long_for_peak}' not in df_history.columns or pd.isna(
                    df_history[f'MA{self.ma_long_for_peak}'].iloc[i]):
                continue

            potential_peak_price = df_history['high'].iloc[i]
            # Check if this point is a rolling peak
            if pd.notna(df_history['rolling_peak_val'].iloc[i]) and potential_peak_price == \
                    df_history['rolling_peak_val'].iloc[i]:
                ma_at_peak_time = df_history[f'MA{self.ma_long_for_peak}'].iloc[i]
                if pd.notna(ma_at_peak_time) and ma_at_peak_time > 0:
                    if potential_peak_price >= ma_at_peak_time * self.ma_peak_threshold:
                        results["prior_peak_price"] = potential_peak_price
                        peak_found_idx = i
                        # logger_strat.debug(f"Prior Peak for signal found at {df_history['date'].iloc[i]}: Price {potential_peak_price:.2f}")
                        break

        if results["prior_peak_price"] is None or peak_found_idx == -1:
            logger_strat.debug("No valid prior peak found for signal.")
            return results  # Cannot proceed without a prior peak for this strategy

        # 2. Identify the "Recent Low" (after the prior_peak_price, up to the signal day)
        # The search window for the low is between the found peak and the signal day.
        # If peak is too close to signal day, window might be small.
        # If no peak found, this part is skipped.

        # Search for low from peak_found_idx + 1 up to end of df_history (signal day)
        low_search_df = df_history.iloc[peak_found_idx + 1:]
        if not low_search_df.empty and len(
                low_search_df) >= self.peak_window:  # Need at least peak_window bars for rolling min
            low_search_df['rolling_low_val'] = low_search_df['low'].rolling(window=self.peak_window, center=True,
                                                                            min_periods=1).min()

            # Find the minimum of these 'rolling_low_val' in the window
            # Or, more simply, find the lowest 'low' in this period that's confirmed by a rolling window.
            # For simplicity here, let's find the absolute low in the window that is also a rolling low.
            # We need a low that is *before* the buy signal.

            # Find the most recent rolling low *before* the last day (signal day).
            # Iterate up to second to last element of low_search_df, as last element is the signal day.
            for i in range(len(low_search_df) - 2, self.peak_window - 2, -1):
                potential_low_price = low_search_df['low'].iloc[i]
                if pd.notna(low_search_df['rolling_low_val'].iloc[i]) and potential_low_price == \
                        low_search_df['rolling_low_val'].iloc[i]:
                    results["recent_low_price"] = potential_low_price
                    # logger_strat.debug(f"Recent Low for signal found at {low_search_df['date'].iloc[i]}: Price {potential_low_price:.2f}")
                    break  # Found the most recent one
            if results[
                "recent_low_price"] is None and not low_search_df.empty:  # Fallback if no rolling low found, use min low of the period
                results["recent_low_price"] = low_search_df['low'].min()

        if results["recent_low_price"] is None:  # If still none, e.g. peak was very recent
            # Fallback: use the low of the signal day candle itself if no other low is found,
            # or the low of the pullback formation period. This needs careful definition.
            # For now, if no distinct low is found after peak, this might invalidate some profit targets.
            # A simple fallback: if no clear low after peak, use the low of the bar just before signal bar.
            if len(df_history) >= 2:
                results["recent_low_price"] = df_history['low'].iloc[-2]  # Low of T-2 day if signal is T-1
            logger_strat.debug(f"Using fallback for recent_low_price: {results['recent_low_price']}")

        return results

    def get_buy_signals(self, df_orig: pd.DataFrame, level: str) -> List[Dict]:
        """
        Generates buy signals based on the last day of the provided DataFrame.
        The DataFrame should be resampled to the desired level (daily, weekly, monthly) if necessary.
        """
        signals = []
        # Ensure enough data for MA calculations and peak/low finding
        # Minimum length should accommodate the longest MA period and window for peak/low finding
        min_len_required = max(self.ma_long, self.ma_long_for_peak, self.trend_window, self.peak_window,
                               50)  # 50 as a general buffer
        if len(df_orig) < min_len_required:
            # logger_strat.debug(f"[{level}] DataFrame length {len(df_orig)} too short for signal generation (min {min_len_required} required).")
            return signals

        df_with_ma = df_orig.copy()
        # Calculate all necessary MAs
        all_ma_periods = list(set([self.ma_short, self.ma_long, self.ma_long_for_peak]))
        df_with_ma = self.calculate_ma(df_with_ma, all_ma_periods)

        # Drop rows with NaN in essential MAs before proceeding
        essential_ma_cols = [f'MA{p}' for p in all_ma_periods]
        df_with_ma.dropna(subset=essential_ma_cols, inplace=True)

        if len(df_with_ma) < self.trend_window:  # Check length again after dropna
            # logger_strat.debug(f"[{level}] DataFrame too short after MA calculation and dropna.")
            return signals

        # Signal is based on the last row of df_with_ma (which corresponds to T-1)
        last_idx = len(df_with_ma) - 1
        last_data = df_with_ma.iloc[last_idx]

        # --- Buy Conditions ---
        # 0. Find relevant "Prior Peak" and "Recent Low" based on data up to T-1
        # Pass df_with_ma itself as it contains data up to T-1 (the last row)
        peak_low_info = self.find_signal_relevant_peak_and_low(df_with_ma, f'MA{self.ma_long_for_peak}')
        prior_peak_price = peak_low_info["prior_peak_price"]
        recent_low_price = peak_low_info["recent_low_price"]

        if prior_peak_price is None or recent_low_price is None:
            # logger_strat.debug(f"[{level}] Condition 0 Fail for {last_data['date']}: Valid prior peak or recent low not found.")
            return signals

        # 1. MA_long (e.g., MA30 for pullback) trend判断
        ma_long_col = f'MA{self.ma_long}'
        is_trend_ok = self.is_ma_trending_up(df_with_ma[ma_long_col], window=self.trend_window)
        if is_trend_ok is None or not is_trend_ok:
            # logger_strat.debug(f"[{level}] Condition 1 Fail for {last_data['date']}: MA{self.ma_long} trend not up.")
            return signals

        # 2. 股价 > MA_long (pullback MA)
        if not (last_data['close'] > last_data[ma_long_col]):
            # logger_strat.debug(f"[{level}] Condition 2 Fail for {last_data['date']}: Close <= MA{self.ma_long}.")
            return signals

        # 3. MA_short > MA_long (pullback MA)
        ma_short_col = f'MA{self.ma_short}'
        if not (last_data[ma_short_col] > last_data[ma_long_col]):
            # logger_strat.debug(f"[{level}] Condition 3 Fail for {last_data['date']}: MA{self.ma_short} <= MA{self.ma_long}.")
            return signals

        # 4. 股价回踩 MA_long (pullback MA)
        if last_data[ma_long_col] <= 1e-6:  # Avoid division by zero or near-zero
            # logger_strat.debug(f"[{level}] Condition 4 Fail for {last_data['date']}: MA{self.ma_long} is zero or negative.")
            return signals

        is_pullback = (last_data['close'] >= last_data[ma_long_col]) and \
                      ((last_data['close'] - last_data[ma_long_col]) / last_data[ma_long_col] <= self.pullback_pct)
        if not is_pullback:
            # logger_strat.debug(f"[{level}] Condition 4 Fail for {last_data['date']}: Pullback condition not met.")
            return signals

        # All buy conditions met
        signal_date_str = pd.to_datetime(last_data['date']).strftime('%Y-%m-%d')
        logger_strat.info(f"✅ [{last_data.get('symbol', 'N/A')} @ {level.upper()}] BUY Signal on {signal_date_str}")
        signals.append({
            "symbol": last_data.get('symbol', 'N/A'),  # Symbol should be in df_orig
            "signal_date": signal_date_str,  # T-1 date
            "strategy_name": self.name,
            "timeframe": level,
            "entry_price_signal_day_close": last_data['close'],  # Close of T-1
            "prior_peak_price": prior_peak_price,
            "recent_low_price": recent_low_price,
            f"ma{self.ma_long}": last_data[ma_long_col],
            f"ma{self.ma_short}": last_data[ma_short_col]
        })
        return signals

    # Note: find_signals is the main interface for external callers like the main script.
    # For RQAlpha, we'll typically use methods like get_buy_signals directly in before_trading/handle_bar
    # after preparing the data for the specific stock and timeframe.

    def get_sell_signals(self, df_current_bar_data: pd.DataFrame, position_info: Dict) -> Optional[str]:
        """
        Checks sell conditions for a currently held position.
        df_current_bar_data: DataFrame containing HLCV data up to the CURRENT bar (T).
                             It should have at least MA30, MA5, ATR, MACD calculated.
        position_info: Dictionary containing info about the held position, e.g.,
                       'entry_price', 'initial_shares', 'shares_held',
                       'prior_peak_price', 'recent_low_price' (fixed at buy time),
                       'atr_stop_loss_price', 'take_profit_targets_hit' (list of bools [T1, T2, T3])
        Returns: A string reason for selling, or None if no sell signal.
        """
        if df_current_bar_data.empty or len(df_current_bar_data) < 2:  # Need at least 2 bars for prev_vol
            return None

        last_bar = df_current_bar_data.iloc[-1]
        prev_bar = df_current_bar_data.iloc[-2] if len(df_current_bar_data) >= 2 else last_bar

        # --- Retrieve necessary info from position_info ---
        entry_price = position_info['entry_price']
        prior_peak_price = position_info['prior_peak_price']
        recent_low_price = position_info['recent_low_price']  # This is fixed at buy time
        current_atr_stop_loss = position_info.get('atr_stop_loss_price')  # Updated by handle_bar
        take_profit_targets_hit = position_info.get('take_profit_targets_hit', [False, False, False])

        # --- Calculate Sell Targets (these are fixed based on buy-time info) ---
        # Ensure these are calculated once at buy time and stored in position_info if not already
        if 'tp1' not in position_info:
            position_info['tp1'] = (prior_peak_price + recent_low_price) / 2
            position_info['tp2'] = prior_peak_price
            position_info['tp3'] = prior_peak_price + (entry_price - recent_low_price)

        tp1 = position_info['tp1']
        tp2 = position_info['tp2']
        tp3 = position_info['tp3']

        # --- Check Stop Loss Conditions ---
        # Condition A:跌破MA30的3%止损，且跌破那天的的成交量上升，是前一天1.5倍以上
        ma30_val = last_bar.get(f'MA{self.ma_long}')
        if pd.notna(ma30_val):
            stop_loss_price_ma30 = ma30_val * (1 - self.sell_ma30_pct)
            if last_bar['close'] < stop_loss_price_ma30:
                if pd.notna(last_bar.get('volume')) and pd.notna(prev_bar.get('volume')) and prev_bar['volume'] > 0:
                    if last_bar['volume'] >= prev_bar['volume'] * self.sell_volume_ratio:
                        logger_strat.info(
                            f"SELL REASON (A): {last_bar.get('symbol', '')} - MA30 DRP 3% VOL UP. Price {last_bar['close']:.2f} < MA30_SL {stop_loss_price_ma30:.2f}, VOL_Ratio {last_bar['volume'] / prev_bar['volume']:.2f}")
                        return "stop_loss_ma30_volume"

        # Condition B: MA5与MA30发生死叉
        ma5_val = last_bar.get(f'MA{self.ma_short}')
        # Check for crossover on the current bar based on previous bar's values
        prev_ma5_val = prev_bar.get(f'MA{self.ma_short}')
        prev_ma30_val = prev_bar.get(f'MA{self.ma_long}')
        if pd.notna(ma5_val) and pd.notna(ma30_val) and pd.notna(prev_ma5_val) and pd.notna(prev_ma30_val):
            if prev_ma5_val >= prev_ma30_val and ma5_val < ma30_val:  # Dead cross occurred
                logger_strat.info(
                    f"SELL REASON (B): {last_bar.get('symbol', '')} - MA5/MA30 Dead Cross. MA5={ma5_val:.2f}, MA30={ma30_val:.2f}")
                return "stop_loss_ma_dead_cross"

        # Condition C: ATR追踪止损 (ATR stop loss price is updated and passed in position_info by handle_bar)
        if current_atr_stop_loss is not None and pd.notna(current_atr_stop_loss):
            if last_bar['low'] < current_atr_stop_loss:  # Using low for ATR stop
                logger_strat.info(
                    f"SELL REASON (C): {last_bar.get('symbol', '')} - ATR Stop Loss. Low {last_bar['low']:.2f} < ATR_SL {current_atr_stop_loss:.2f}")
                return "stop_loss_atr_trailing"

        # --- Check Take Profit Conditions (Partial Sells) ---
        # Sell 1/3 if TP1 hit and not yet sold
        if not take_profit_targets_hit[0] and last_bar['high'] >= tp1:
            logger_strat.info(
                f"SELL REASON (TP1): {last_bar.get('symbol', '')} - Target 1 Hit. High {last_bar['high']:.2f} >= TP1 {tp1:.2f}")
            return "take_profit_1"

        # Sell next 1/3 if TP2 hit and TP1 was hit, and TP2 not yet sold
        if take_profit_targets_hit[0] and not take_profit_targets_hit[1] and last_bar['high'] >= tp2:
            logger_strat.info(
                f"SELL REASON (TP2): {last_bar.get('symbol', '')} - Target 2 Hit. High {last_bar['high']:.2f} >= TP2 {tp2:.2f}")
            return "take_profit_2"

        # Sell remaining 1/3 if TP3 hit and TP1, TP2 were hit, and TP3 not yet sold
        if take_profit_targets_hit[0] and take_profit_targets_hit[1] and not take_profit_targets_hit[2] and last_bar[
            'high'] >= tp3:
            logger_strat.info(
                f"SELL REASON (TP3): {last_bar.get('symbol', '')} - Target 3 Hit. High {last_bar['high']:.2f} >= TP3 {tp3:.2f}")
            return "take_profit_3"

        # --- Check MACD Top Divergence (as a warning, actual sell might be combined) ---
        # This is more complex to implement robustly here without full history tracking in position_info
        # For RQAlpha, this would typically be checked in handle_bar and might modify ATR multiplier or set a flag.
        # Here, we'll just log if a bearish divergence pattern is forming on the current data.
        # A full divergence check needs to compare two peaks in price and two peaks in MACD hist.
        # Simple check: If price makes a new high compared to N bars ago, but macdhist doesn't.
        if len(df_current_bar_data) > 20 and pd.notna(last_bar.get('macdhist')):  # Check for enough data
            # Simplified: if close is a new high in 20 bars, but macdhist is lower than its peak in 20 bars
            recent_high_price = df_current_bar_data['high'].iloc[-20:-1].max()
            recent_high_macdhist = df_current_bar_data['macdhist'].iloc[-20:-1].max()
            if last_bar['high'] > recent_high_price and pd.notna(recent_high_macdhist) and last_bar[
                'macdhist'] < recent_high_macdhist * 0.8:  # 0.8 to allow some leeway
                logger_strat.warning(
                    f"WARNING: {last_bar.get('symbol', '')} - Potential MACD Top Divergence forming. Price new high, MACD hist not confirming.")
                # This doesn't trigger a sell directly but is a warning.
                # In RQAlpha, you could set context.security_alerts[symbol]['macd_divergence'] = True

        return None  # No sell signal