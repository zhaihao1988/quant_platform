# strategies/weekly_breakout_strategy.py

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from collections import namedtuple
# Ensure correct import of base classes and shared data structures
try:
    from .base_strategy import BaseStrategy, StrategyResult, StrategyContext
    from .ma_pullback_strategy import KLineRaw, MergedKLine, Fractal, Stroke
except ImportError:
    # --- MOCK DEFINITIONS FOR STANDALONE TESTING (Remove if using your actual base_strategy) ---
    from dataclasses import dataclass, field
    from abc import ABC, abstractmethod

    logger_base_mock = logging.getLogger(__name__ + "_base_mock_weekly_cb")


    @dataclass
    class StrategyContext:
        db_session: Optional[Any] = None
        current_date: Optional[date] = None
        strategy_params: Dict[str, Any] = field(default_factory=dict)


    @dataclass
    class StrategyResult:
        symbol: str
        signal_date: date
        strategy_name: str
        signal_type: str = "BUY"
        signal_score: Optional[float] = None
        details: Dict[str, Any] = field(default_factory=dict)

        def __str__(self):
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            return (f"StrategyResult(symbol='{self.symbol}', signal_date='{self.signal_date}', "
                    f"strategy_name='{self.strategy_name}', signal_type='{self.signal_type}', details={{{details_str}}})")


    class BaseStrategy(ABC):
        def __init__(self, context: StrategyContext):
            self.context = context
            if not hasattr(self.context, 'strategy_params') or self.context.strategy_params is None:
                self.context.strategy_params = {}

        @property
        @abstractmethod
        def strategy_name(self) -> str: pass

        @abstractmethod
        def run_for_stock(self, symbol: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[
            StrategyResult]: pass


    from collections import namedtuple

    KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx', 'original_idx'])
    MergedKLine = namedtuple('MergedKLine',
                             ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx', 'raw_kline_indices'])
    Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
    Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction', 'start_m_idx', 'end_m_idx'])
    # --- END MOCK DEFINITIONS ---

logger = logging.getLogger(__name__)

# Helper data structure for H points
H_Point = namedtuple('H_Point', ['price', 'date', 'level_name'])


class WeeklyBreakoutStrategy(BaseStrategy):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        default_params = {
            'fractal_definition_lookback': 1,
            'min_bars_between_fractals_bt': 1,
            'pattern_identification_lookback_days': 104,  # 周线级别：约2年 (500天 -> 104周)
            'pullback_depth_pct': 0.20,
            'days_for_3year_low': 156,                    # 周线级别：约3年 (750天 -> 156周)
            'max_price_vs_3year_low_ratio': 2.0,
            'volume_lookback_period': 12,                 # 周线级别：约3个月 (60天 -> 12周)
            'volume_ratio_threshold': 2.0,
        }
        strategy_specific_params = self.context.strategy_params.get(self.strategy_name, {})
        self.params = {**default_params, **strategy_specific_params}

        self._merged_klines_state: List[MergedKLine] = []
        self._current_segment_trend_state: int = 0
        self._fractals: List[Fractal] = []
        self._strokes: List[Stroke] = []

    @property
    def strategy_name(self) -> str:
        return "WeeklyChanBreakoutStrategyV1"

    def _initialize_state_for_stock(self):
        self._merged_klines_state = []
        self._current_segment_trend_state = 0
        self._fractals = []
        self._strokes = []
        logger.debug(f"[{self.strategy_name}] State initialized.")

    def _df_to_raw_klines(self, df: pd.DataFrame) -> List[KLineRaw]:
        raw_klines = []
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        if hasattr(df_copy['date'].dt, 'date'):
            df_copy['date'] = df_copy['date'].dt.date
        else:
            df_copy['date'] = df_copy['date'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else (
                pd.to_datetime(x).date() if not isinstance(x, date) else x))

        for i in range(len(df_copy)):
            row = df_copy.iloc[i]
            original_df_index = df_copy.index[i] if df_copy.index is not None else i
            current_date_val = row['date']
            if isinstance(current_date_val, pd.Timestamp):
                current_date_val = current_date_val.date()
            elif not isinstance(current_date_val, date):
                try:
                    current_date_val = pd.to_datetime(current_date_val).date()
                except Exception:
                    logger.error(f"Could not convert date {current_date_val} in _df_to_raw_klines for {row}"); continue
            raw_klines.append(
                KLineRaw(dt=current_date_val, o=row['open'], h=row['high'], l=row['low'], c=row['close'], idx=i,
                         original_idx=original_df_index))
        return raw_klines

    def _process_raw_kline_for_merging(self, k2_raw: KLineRaw):
        if not self._merged_klines_state:
            mk = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c, idx=k2_raw.original_idx,
                             direction=0, high_idx=k2_raw.original_idx, low_idx=k2_raw.original_idx,
                             raw_kline_indices=[k2_raw.original_idx])
            self._merged_klines_state.append(mk);
            self._current_segment_trend_state = 0;
            return
        k1_merged = self._merged_klines_state[-1]
        k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l)
        k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)
        if k1_includes_k2 or k2_includes_k1:
            m_o, m_h, m_l, m_c, m_dt, m_idx_end = k1_merged.o, k1_merged.h, k1_merged.l, k2_raw.c, k2_raw.dt, k2_raw.original_idx
            m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
            m_raw_indices = list(k1_merged.raw_kline_indices);
            m_raw_indices.append(k2_raw.original_idx)
            trend_for_inclusion = self._current_segment_trend_state
            if trend_for_inclusion == 1:
                if k2_raw.h >= k1_merged.h:
                    m_h, m_high_idx = k2_raw.h, k2_raw.original_idx
                else:
                    m_h = k1_merged.h  # Keep k1's high
            elif trend_for_inclusion == -1:
                if k2_raw.l <= k1_merged.l:
                    m_l, m_low_idx = k2_raw.l, k2_raw.original_idx
                else:
                    m_l = k1_merged.l  # Keep k1's low
            else:
                if k2_includes_k1: m_h, m_l, m_high_idx, m_low_idx = k2_raw.h, k2_raw.l, k2_raw.original_idx, k2_raw.original_idx
            self._merged_klines_state[-1] = MergedKLine(dt=m_dt, o=m_o, h=m_h, l=m_l, c=m_c, idx=m_idx_end,
                                                        direction=k1_merged.direction, high_idx=m_high_idx,
                                                        low_idx=m_low_idx, raw_kline_indices=m_raw_indices)
        else:
            if k1_merged.direction == 0 and len(self._merged_klines_state) > 1:
                k_prev_prev = self._merged_klines_state[-2];
                final_k1_dir = 0
                if k1_merged.h > k_prev_prev.h and k1_merged.l > k_prev_prev.l:
                    final_k1_dir = 1
                elif k1_merged.h < k_prev_prev.h and k1_merged.l < k_prev_prev.l:
                    final_k1_dir = -1
                self._merged_klines_state[-1] = k1_merged._replace(direction=final_k1_dir)
            k1_finalized = self._merged_klines_state[-1];
            new_seg_dir = 0
            if k2_raw.h > k1_finalized.h and k2_raw.l > k1_finalized.l:
                new_seg_dir = 1
            elif k2_raw.h < k1_finalized.h and k2_raw.l < k1_finalized.l:
                new_seg_dir = -1
            self._current_segment_trend_state = new_seg_dir
            mk_new = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c, idx=k2_raw.original_idx,
                                 direction=new_seg_dir, high_idx=k2_raw.original_idx, low_idx=k2_raw.original_idx,
                                 raw_kline_indices=[k2_raw.original_idx])
            self._merged_klines_state.append(mk_new)

    def _finalize_merged_kline_directions(self):
        if not self._merged_klines_state: return
        for i in range(len(self._merged_klines_state)):
            if self._merged_klines_state[i].direction == 0:
                final_dir = 0
                if i > 0:
                    if self._merged_klines_state[i].h > self._merged_klines_state[i - 1].h and \
                            self._merged_klines_state[i].l > self._merged_klines_state[i - 1].l:
                        final_dir = 1
                    elif self._merged_klines_state[i].h < self._merged_klines_state[i - 1].h and \
                            self._merged_klines_state[i].l < self._merged_klines_state[i - 1].l:
                        final_dir = -1
                if final_dir == 0 and i < len(self._merged_klines_state) - 1:
                    if self._merged_klines_state[i].h < self._merged_klines_state[i + 1].h and \
                            self._merged_klines_state[i].l < self._merged_klines_state[i + 1].l:
                        final_dir = 1
                    elif self._merged_klines_state[i].h > self._merged_klines_state[i + 1].h and \
                            self._merged_klines_state[i].l > self._merged_klines_state[i + 1].l:
                        final_dir = -1
                if final_dir == 0: final_dir = 1 if self._merged_klines_state[i].c >= self._merged_klines_state[
                    i].o else -1
                self._merged_klines_state[i] = self._merged_klines_state[i]._replace(direction=final_dir)

    def _identify_fractals_batch(self) -> List[Fractal]:
        fractals: List[Fractal] = [];
        fb = self.params['fractal_definition_lookback']
        if len(self._merged_klines_state) < (2 * fb + 1): return fractals
        for i in range(fb, len(self._merged_klines_state) - fb):
            is_top = all(
                self._merged_klines_state[i].h > self._merged_klines_state[i - j].h and self._merged_klines_state[i].h >
                self._merged_klines_state[i + j].h for j in range(1, fb + 1))
            is_bottom = all(
                self._merged_klines_state[i].l < self._merged_klines_state[i - j].l and self._merged_klines_state[i].l <
                self._merged_klines_state[i + j].l for j in range(1, fb + 1))
            if is_top and is_bottom: continue
            if is_top:
                fractals.append(Fractal(kline=self._merged_klines_state[i], m_idx=i, type=1))
            elif is_bottom:
                fractals.append(Fractal(kline=self._merged_klines_state[i], m_idx=i, type=-1))
        fractals.sort(key=lambda f: (f.m_idx, -f.type));
        return fractals

    def _connect_fractals_to_strokes_batch(self) -> List[Stroke]:
        strokes: List[Stroke] = [];
        processed_fractals: List[Fractal] = []
        if not self._fractals: return strokes
        processed_fractals.append(self._fractals[0])
        for i in range(1, len(self._fractals)):
            current_f, last_f = self._fractals[i], processed_fractals[-1]
            if current_f.type == last_f.type:
                if (current_f.type == 1 and current_f.kline.h >= last_f.kline.h) or \
                        (current_f.type == -1 and current_f.kline.l <= last_f.kline.l):
                    processed_fractals[-1] = current_f
            else:
                processed_fractals.append(current_f)
        self._fractals = processed_fractals
        if len(self._fractals) < 2: return strokes
        last_confirmed_fractal = self._fractals[0]
        for i in range(1, len(self._fractals)):
            current_fractal = self._fractals[i]
            if current_fractal.type == last_confirmed_fractal.type: continue
            bars_between_merged = abs(current_fractal.m_idx - last_confirmed_fractal.m_idx) - 1
            if bars_between_merged < self.params['min_bars_between_fractals_bt']:
                if (
                        current_fractal.type == 1 and current_fractal.kline.h < last_confirmed_fractal.kline.h and last_confirmed_fractal.type == 1) or \
                        (
                                current_fractal.type == -1 and current_fractal.kline.l > last_confirmed_fractal.kline.l and last_confirmed_fractal.type == -1):
                    pass
                else:
                    last_confirmed_fractal = current_fractal
                continue
            stroke_direction = 0;
            start_f, end_f = last_confirmed_fractal, current_fractal
            if start_f.type == -1 and end_f.type == 1 and end_f.kline.h > start_f.kline.h:
                stroke_direction = 1
            elif start_f.type == 1 and end_f.type == -1 and end_f.kline.l < start_f.kline.l:
                stroke_direction = -1
            if stroke_direction != 0: strokes.append(
                Stroke(start_f, end_f, stroke_direction, start_f.m_idx, end_f.m_idx))
            last_confirmed_fractal = end_f
        return strokes

    def _calculate_volume_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if 'volume' in df_copy.columns:
            df_copy[f'volume_ma{self.params["volume_lookback_period"]}'] = df_copy['volume'].rolling(
                window=self.params["volume_lookback_period"], min_periods=1).mean()
        else:
            logger.warning(f"[{self.strategy_name}] 'volume' column not found for MA calculation.");
            df_copy[f'volume_ma{self.params["volume_lookback_period"]}'] = np.nan
        return df_copy

    def _get_completed_upstrokes(self, max_lookback_date: date) -> List[Stroke]:
        completed_upstrokes = []
        for i in range(len(self._strokes)):
            stroke = self._strokes[i]
            if stroke.direction == 1 and stroke.end_fractal.kline.dt >= max_lookback_date:
                if i < len(self._strokes) - 1:
                    next_stroke = self._strokes[i + 1]
                    if next_stroke.direction == -1 and next_stroke.start_fractal.m_idx == stroke.end_fractal.m_idx:
                        completed_upstrokes.append(stroke)
        return sorted(completed_upstrokes, key=lambda s: s.end_fractal.kline.dt, reverse=True)

    def run_for_stock(self, symbol: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[StrategyResult]:
        self._initialize_state_for_stock()
        results: List[StrategyResult] = []

        weekly_df_orig = data.get("weekly")
        if weekly_df_orig is None or weekly_df_orig.empty or 'date' not in weekly_df_orig.columns:
            logger.warning(f"[{self.strategy_name}@{symbol}@{current_date.isoformat()}] No weekly data.");
            return results

        weekly_df = weekly_df_orig.copy()
        if not pd.api.types.is_datetime64_any_dtype(weekly_df['date']): weekly_df['date'] = pd.to_datetime(
            weekly_df['date'])
        if hasattr(weekly_df['date'].dt, 'date'):
            weekly_df['date'] = weekly_df['date'].dt.date
        else:
            weekly_df['date'] = weekly_df['date'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else (
                pd.to_datetime(x).date() if not isinstance(x, date) else x))

        df_full_history_for_analysis = weekly_df[weekly_df['date'] <= current_date].copy()
        if len(df_full_history_for_analysis) < 2:
            logger.info(f"[{self.strategy_name}@{symbol}@{current_date.isoformat()}] Not enough data ( < 2 bars).");
            return results

        df_full_history_for_analysis = self._calculate_volume_ma(df_full_history_for_analysis)

        current_bar_data = df_full_history_for_analysis.iloc[-1]
        previous_bar_data = df_full_history_for_analysis.iloc[-2]
        current_high_price = current_bar_data['high']
        previous_high_price = previous_bar_data['high']
        current_volume = current_bar_data['volume']
        ma_volume_col = f'volume_ma{self.params["volume_lookback_period"]}'
        current_ma_volume = current_bar_data.get(ma_volume_col)

        pattern_lookback_weeks = self.params['pattern_identification_lookback_days']
        df_slice_for_patterns = df_full_history_for_analysis.iloc[
                                -(pattern_lookback_weeks + 5):].copy()

        if len(df_slice_for_patterns) < (
                self.params['fractal_definition_lookback'] * 2 + self.params['min_bars_between_fractals_bt'] + 3):
            logger.debug(f"[{self.strategy_name}@{symbol}] Not enough data in slice for pattern id.");
            return results

        raw_klines = self._df_to_raw_klines(df_slice_for_patterns)
        if raw_klines:
            self._merged_klines_state = [];
            self._current_segment_trend_state = 0
            for rk in raw_klines: self._process_raw_kline_for_merging(rk)
            self._finalize_merged_kline_directions()
            if len(self._merged_klines_state) >= (2 * self.params['fractal_definition_lookback'] + 1):
                self._fractals = self._identify_fractals_batch()
                if len(self._fractals) >= 2: self._strokes = self._connect_fractals_to_strokes_batch()

        if not self._strokes: logger.debug(f"[{self.strategy_name}@{symbol}] No strokes."); return results

        lookback_weeks_for_H = self.params['pattern_identification_lookback_days']
        max_lookback_date = current_date - timedelta(weeks=lookback_weeks_for_H)

        completed_upstrokes = self._get_completed_upstrokes(max_lookback_date=max_lookback_date)

        h1: Optional[H_Point] = None
        h2: Optional[H_Point] = None
        h3: Optional[H_Point] = None

        if completed_upstrokes:
            h1_stroke = completed_upstrokes[0]
            h1 = H_Point(price=h1_stroke.end_fractal.kline.h, date=h1_stroke.end_fractal.kline.dt, level_name="H1")

            h2_candidate_strokes = [s for s in completed_upstrokes if
                                    s.end_fractal.kline.dt < h1.date and s.end_fractal.kline.h > h1.price]
            if h2_candidate_strokes:
                h2_stroke = h2_candidate_strokes[0]
                h2 = H_Point(price=h2_stroke.end_fractal.kline.h, date=h2_stroke.end_fractal.kline.dt, level_name="H2")

                h3_candidate_strokes = [s for s in completed_upstrokes if
                                        s.end_fractal.kline.dt < h2.date and s.end_fractal.kline.h > h2.price]
                if h3_candidate_strokes:
                    h3_stroke = h3_candidate_strokes[0]
                    h3 = H_Point(price=h3_stroke.end_fractal.kline.h, date=h3_stroke.end_fractal.kline.dt,
                                 level_name="H3")

        if not (h3 and h2 and h1 and h3.price > h2.price > h1.price):
            logger.debug(
                f"[{self.strategy_name}@{symbol}] Did not find the required H3>H2>H1 structure. H1:{h1}, H2:{h2}, H3:{h3}")
            return results

        logger.debug(
            f"[{self.strategy_name}@{symbol}] Found structure H3({h3.price:.2f}@{h3.date}) > H2({h2.price:.2f}@{h2.date}) > H1({h1.price:.2f}@{h1.date})")

        potential_breakout_targets: List[H_Point] = []
        if h3: potential_breakout_targets.append(h3)
        if h2: potential_breakout_targets.append(h2)
        if h1: potential_breakout_targets.append(h1)

        for h_target in sorted(potential_breakout_targets, key=lambda p: p.price, reverse=True):
            h_price = h_target.price
            h_date = h_target.date
            level_name = h_target.level_name

            if h_date >= current_date: continue
            df_after_h = df_full_history_for_analysis[
                (df_full_history_for_analysis['date'] > h_date) &
                (df_full_history_for_analysis['date'] < current_date)
                ]
            if df_after_h.empty:
                logger.debug(f"[{self.strategy_name}@{symbol}] No data after {level_name} for pullback check."); continue

            min_low_since_h = df_after_h['low'].min()
            required_pullback_price = h_price * (1 - self.params['pullback_depth_pct'])
            if not (min_low_since_h <= required_pullback_price):
                logger.debug(
                    f"[{self.strategy_name}@{symbol}] Pullback for {level_name}({h_price:.2f}) not met. MinLow={min_low_since_h:.2f}, Req<={required_pullback_price:.2f}"); continue

            lookback_weeks_for_3y_low = self.params['days_for_3year_low']
            start_date_3year_low = current_date - timedelta(weeks=lookback_weeks_for_3y_low)
            df_for_3year_low = df_full_history_for_analysis[
                df_full_history_for_analysis['date'] >= start_date_3year_low]
            min_low_past_3_years = df_for_3year_low['low'].min() if not df_for_3year_low.empty else None
            if min_low_past_3_years is not None:
                if current_high_price > min_low_past_3_years * self.params['max_price_vs_3year_low_ratio']:
                    logger.debug(
                        f"[{self.strategy_name}@{symbol}] Price position limit exceeded. CurrHigh {current_high_price:.2f} vs Limit {min_low_past_3_years * self.params['max_price_vs_3year_low_ratio']:.2f}"); continue
            else:
                logger.warning(f"[{self.strategy_name}@{symbol}] Could not check 3-year low.")

            cond_c_prev_high_below_h = previous_high_price < h_price
            cond_c_curr_high_above_h = current_high_price > h_price
            breakthrough_cond_met = cond_c_prev_high_below_h and cond_c_curr_high_above_h
            if not breakthrough_cond_met:
                logger.debug(f"[{self.strategy_name}@{symbol}] Breakout for {level_name} not met."); continue

            if pd.isna(current_ma_volume) or current_ma_volume == 0:
                logger.debug(f"[{self.strategy_name}@{symbol}] Volume MA NaN/Zero."); continue
            if not (current_volume > current_ma_volume * self.params['volume_ratio_threshold']):
                logger.debug(f"[{self.strategy_name}@{symbol}] Volume not met."); continue

            signal_details = {
                "breakout_level": level_name,
                "h_price": f"{h_price:.2f}", "h_date": h_date.isoformat(),
                "pullback_low_since_h": f"{min_low_since_h:.2f}",
                "pullback_depth_actual_pct": f"{(1 - min_low_since_h / h_price) * 100:.2f}%" if h_price > 0 else "N/A",
                "current_high_price": f"{current_high_price:.2f}", "previous_high_price": f"{previous_high_price:.2f}",
                "current_volume": f"{current_volume:.0f}",
                f"volume_ma{self.params['volume_lookback_period']}": f"{current_ma_volume:.0f}" if pd.notna(
                    current_ma_volume) else "N/A",
                "min_low_past_3_years": f"{min_low_past_3_years:.2f}" if pd.notna(min_low_past_3_years) else "N/A",
                "h1_detail": f"{h1.price:.2f}@{h1.date.isoformat()}" if h1 else "N/A",
                "h2_detail": f"{h2.price:.2f}@{h2.date.isoformat()}" if h2 else "N/A",
                "h3_detail": f"{h3.price:.2f}@{h3.date.isoformat()}" if h3 else "N/A",
            }
            results.append(StrategyResult(symbol, current_date, self.strategy_name, "BUY", details=signal_details))
            logger.info(
                f"[{self.strategy_name}@{symbol}@{current_date.isoformat()}] BUY SIGNAL (Breakout {level_name}). Details: {signal_details}")
            return results

        return results


# --- Main execution for testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s',
                        handlers=[logging.StreamHandler()])
    test_logger = logging.getLogger(__name__ + "_test_main_weekly_cb_v1")
    test_logger.warning("This is a template test block for the WeeklyBreakoutStrategy.")
    test_logger.warning("To run a test, you need to:")
    test_logger.warning("1. Provide a CSV file with weekly stock data (e.g., 'stock_weekly.csv').")
    test_logger.warning("2. Set the 'stock_to_test' symbol and 'date_to_test_str'.")
    test_logger.warning("3. Implement the data loading and strategy execution logic similar to the original daily strategy test block.")

    # Example structure (commented out):
    #
    # csv_file_path = 'path_to_your_weekly_data.csv'
    # stock_to_test = "000001"
    # date_to_test_str = "2025-05-16"
    # target_date = datetime.strptime(date_to_test_str, "%Y-%m-%d").date()
    # try:
    #     # ... (Load and prepare weekly data)
    #     weekly_df_stock = ...
    #     data_for_strategy = {"weekly": weekly_df_stock}
    #
    #     strategy_context = StrategyContext(current_date=target_date, strategy_params={
    #         "WeeklyChanBreakoutStrategyV1": { ... }
    #     })
    #     strategy_instance = WeeklyBreakoutStrategy(context=strategy_context)
    #     signals = strategy_instance.run_for_stock(symbol=stock_to_test, current_date=target_date, data=data_for_strategy)
    #     # ... (Print signals)
    # except Exception as e:
    #     test_logger.error(f"Error in test run: {e}", exc_info=True)


