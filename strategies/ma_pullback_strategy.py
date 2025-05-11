# strategies/ma_pullback_strategy.py

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import math
from collections import namedtuple, deque
from typing import List, Dict, Optional, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Attempt to import from local .base_strategy if run as part of a package
try:
    from .base_strategy import BaseStrategy, StrategyResult, StrategyContext
except ImportError:
    # Define minimal versions if run as a standalone script for testing
    logger_base = logging.getLogger(__name__ + "_base_mock")
    logger_base.warning(
        "Could not import from .base_strategy. Using minimal mock definitions for BaseStrategy, StrategyContext, and StrategyResult for standalone testing."
    )


    @dataclass
    class StrategyContext:
        db_session: Optional[Any] = None
        current_date: Optional[date] = None
        strategy_params: Dict[str, Any] = field(default_factory=dict)


    @dataclass
    class StrategyResult:
        stock_code: str
        signal_date: date
        strategy_name: str
        signal_type: str = "BUY"
        signal_score: Optional[float] = None
        details: Dict[str, Any] = field(default_factory=dict)

        def __str__(self):
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            return (f"StrategyResult(stock_code='{self.stock_code}', signal_date='{self.signal_date}', "
                    f"strategy_name='{self.strategy_name}', signal_type='{self.signal_type}', details={{{details_str}}})")


    class BaseStrategy(ABC):
        def __init__(self, context: StrategyContext):
            self.context = context

        @property
        @abstractmethod
        def strategy_name(self) -> str:
            pass

        @abstractmethod
        def run_for_stock(self, stock_code: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[
            StrategyResult]:
            pass

logger = logging.getLogger(__name__)

# --- Data Structures ---
KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx', 'original_idx'])
MergedKLine = namedtuple('MergedKLine',
                         ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx', 'raw_kline_indices'])
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction', 'start_m_idx', 'end_m_idx'])

QualifiedRefHighInfo = namedtuple('QualifiedRefHighInfo', [
    'price', 'date', 'original_idx',
    'ma_at_high', 'is_ma_valid',
    'is_recent_gain_valid', 'is_fully_qualified'
])
ActivePeakCandidateInfo = namedtuple('ActivePeakCandidateInfo', [
    'price', 'date', 'original_idx', 'ma_at_peak', 'is_ma_valid'
])
LastDownstrokeInfo = namedtuple('LastDownstrokeInfo',
                                [  # Kept for potential other uses, though not primary in new invalidation
                                    'end_date', 'low_price', 'low_idx', 'ma_at_low', 'is_significant_break'
                                ])


class AdaptedMAPullbackStrategy(BaseStrategy):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        default_params = {
            'ma_short': 5,
            'ma_long': 30,  # This is the MA30 used for invalidation rules
            'ma_long_for_peak_qualify_period': 30,
            'ma_peak_threshold': 1.15,
            'peak_recent_gain_days': 30,
            'peak_recent_gain_ratio': 1.20,
            # 'downstroke_invalidate_threshold': 0.03, # Original parameter, may become less relevant or removed
            'high_invalidate_close_below_ma30_pct': 0.97,  # New parameter for the 97% rule
            'weekly_ma_period': 30,
            'fractal_definition_lookback': 1,
            'min_bars_between_fractals_bt': 1,
            'pattern_identification_lookback_days': 150,
        }
        self.params = default_params
        if self.context and self.context.strategy_params:
            self.params.update(self.context.strategy_params.get(self.strategy_name, {}))

        self.params['stroke_min_len_merged_klines'] = self.params['min_bars_between_fractals_bt'] + 2
        self.ma30_col_name = f'ma{self.params["ma_long"]}'  # Store for convenience

        self._merged_klines_state: List[MergedKLine] = []
        self._current_segment_trend_state: int = 0
        self._fractals: List[Fractal] = []
        self._strokes: List[Stroke] = []
        self._active_peak_candidate: Optional[ActivePeakCandidateInfo] = None
        self._qualified_ref_high: Optional[QualifiedRefHighInfo] = None
        # self._last_downstroke: Optional[LastDownstrokeInfo] = None # Potentially less used by new logic
        self._uptrend_invalidated: bool = False

    @property
    def strategy_name(self) -> str:
        return "AdaptedMAPullbackStrategy"

    def _initialize_state_for_stock(self):
        self._merged_klines_state = []
        self._current_segment_trend_state = 0
        self._fractals = []
        self._strokes = []
        self._active_peak_candidate = None
        self._qualified_ref_high = None
        # self._last_downstroke = None
        self._uptrend_invalidated = False
        logger.debug(f"[{self.strategy_name}] State initialized for new stock.")

    def _df_to_raw_klines(self, df: pd.DataFrame) -> List[KLineRaw]:
        raw_klines = []
        df_copy = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        if hasattr(df_copy['date'].dt, 'date'):
            df_copy['date'] = df_copy['date'].dt.date

        for i in range(len(df_copy)):
            row = df_copy.iloc[i]
            original_df_index = df_copy.index[i]
            current_date_val = row['date']

            if isinstance(current_date_val, pd.Timestamp):
                current_date_val = current_date_val.date()
            elif not isinstance(current_date_val, date):
                try:
                    current_date_val = pd.to_datetime(current_date_val).date()
                except Exception:
                    logger.error(f"Could not convert date {current_date_val} to date object in _df_to_raw_klines")
                    continue
            raw_klines.append(KLineRaw(
                dt=current_date_val,
                o=row['open'], h=row['high'], l=row['low'], c=row['close'],
                idx=i, original_idx=original_df_index
            ))
        return raw_klines

    def _process_raw_kline_for_merging(self, k2_raw: KLineRaw):
        if not self._merged_klines_state:
            mk = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c,
                             idx=k2_raw.original_idx, direction=0,
                             high_idx=k2_raw.original_idx, low_idx=k2_raw.original_idx,
                             raw_kline_indices=[k2_raw.original_idx])
            self._merged_klines_state.append(mk)
            self._current_segment_trend_state = 0
            return

        k1_merged = self._merged_klines_state[-1]
        k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l)
        k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)

        if k1_includes_k2 or k2_includes_k1:
            m_o, m_h, m_l, m_c, m_dt, m_idx_end = \
                k1_merged.o, k1_merged.h, k1_merged.l, k2_raw.c, k2_raw.dt, k2_raw.original_idx
            m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
            m_raw_indices = list(k1_merged.raw_kline_indices)
            m_raw_indices.append(k2_raw.original_idx)
            trend_for_inclusion = self._current_segment_trend_state
            if trend_for_inclusion == 1:
                m_h = max(k1_merged.h, k2_raw.h)
                if k2_raw.h >= k1_merged.h: m_high_idx = k2_raw.original_idx
            elif trend_for_inclusion == -1:
                m_l = min(k1_merged.l, k2_raw.l)
                if k2_raw.l <= k1_merged.l: m_low_idx = k2_raw.original_idx
            else:
                if k2_includes_k1:
                    m_h, m_l = k2_raw.h, k2_raw.l
                    m_high_idx, m_low_idx = k2_raw.original_idx, k2_raw.original_idx
            self._merged_klines_state[-1] = MergedKLine(
                dt=m_dt, o=m_o, h=m_h, l=m_l, c=m_c, idx=m_idx_end,
                direction=k1_merged.direction,
                high_idx=m_high_idx, low_idx=m_low_idx,
                raw_kline_indices=m_raw_indices
            )
        else:
            if k1_merged.direction == 0 and len(self._merged_klines_state) > 1:
                k_prev_prev_merged = self._merged_klines_state[-2]
                final_k1_dir = 0
                if k1_merged.h > k_prev_prev_merged.h and k1_merged.l > k_prev_prev_merged.l:
                    final_k1_dir = 1
                elif k1_merged.h < k_prev_prev_merged.h and k1_merged.l < k_prev_prev_merged.l:
                    final_k1_dir = -1
                self._merged_klines_state[-1] = k1_merged._replace(direction=final_k1_dir)

            k1_finalized_merged = self._merged_klines_state[-1]
            new_segment_direction = 0
            if k2_raw.h > k1_finalized_merged.h and k2_raw.l > k1_finalized_merged.l:
                new_segment_direction = 1
            elif k2_raw.h < k1_finalized_merged.h and k2_raw.l < k1_finalized_merged.l:
                new_segment_direction = -1
            self._current_segment_trend_state = new_segment_direction
            mk_new = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c,
                                 idx=k2_raw.original_idx, direction=new_segment_direction,
                                 high_idx=k2_raw.original_idx, low_idx=k2_raw.original_idx,
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
                if final_dir == 0:
                    final_dir = 1 if self._merged_klines_state[i].c >= self._merged_klines_state[i].o else -1
                self._merged_klines_state[i] = self._merged_klines_state[i]._replace(direction=final_dir)

    def _identify_fractals_batch(self) -> List[Fractal]:
        fractals: List[Fractal] = []
        fb = self.params['fractal_definition_lookback']
        if len(self._merged_klines_state) < (2 * fb + 1): return fractals
        for i in range(fb, len(self._merged_klines_state) - fb):
            k_prev = self._merged_klines_state[i - 1];
            k_curr = self._merged_klines_state[i];
            k_next = self._merged_klines_state[i + 1]
            is_top = k_curr.h > k_prev.h and k_curr.h > k_next.h
            is_bottom = k_curr.l < k_prev.l and k_curr.l < k_next.l
            if is_top and is_bottom: continue
            if is_top:
                fractals.append(Fractal(kline=k_curr, m_idx=i, type=1))
            elif is_bottom:
                fractals.append(Fractal(kline=k_curr, m_idx=i, type=-1))
        fractals.sort(key=lambda f: (f.m_idx, -f.type))
        return fractals

    def _connect_fractals_to_strokes_batch(self) -> List[Stroke]:
        strokes: List[Stroke] = []
        if len(self._fractals) < 2: return strokes
        min_bars_between_fractals = self.params['min_bars_between_fractals_bt']
        last_confirmed_fractal = self._fractals[0]
        for i in range(1, len(self._fractals)):
            current_fractal = self._fractals[i]
            if current_fractal.type == last_confirmed_fractal.type:
                if (current_fractal.type == 1 and current_fractal.kline.h > last_confirmed_fractal.kline.h) or \
                        (current_fractal.type == -1 and current_fractal.kline.l < last_confirmed_fractal.kline.l):
                    last_confirmed_fractal = current_fractal
                continue
            bars_between_merged = abs(current_fractal.m_idx - last_confirmed_fractal.m_idx) - 1
            if bars_between_merged < min_bars_between_fractals:
                last_confirmed_fractal = current_fractal  # Crucial: ensure this assignment happens to progress
                continue
            stroke_direction = 0;
            start_f, end_f = last_confirmed_fractal, current_fractal
            if start_f.type == -1 and end_f.type == 1:  # Up
                if end_f.kline.h > start_f.kline.l: stroke_direction = 1
            elif start_f.type == 1 and end_f.type == -1:  # Down
                if end_f.kline.l < start_f.kline.h: stroke_direction = -1
            if stroke_direction != 0:
                strokes.append(Stroke(start_f, end_f, stroke_direction, start_f.m_idx, end_f.m_idx))
            last_confirmed_fractal = end_f  # Progress confirmed fractal
        return strokes

    def _calculate_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[f'ma{self.params["ma_short"]}'] = df_copy['close'].rolling(window=self.params["ma_short"],
                                                                           min_periods=1).mean()
        # Ensure self.ma30_col_name is correctly used for MA30 calculation
        df_copy[self.ma30_col_name] = df_copy['close'].rolling(window=self.params["ma_long"],
                                                               min_periods=1).mean()
        df_copy[f'ma_peak_qualify{self.params["ma_long_for_peak_qualify_period"]}'] = df_copy['close'].rolling(
            window=self.params["ma_long_for_peak_qualify_period"], min_periods=1).mean()
        return df_copy

    def _calculate_weekly_mas(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        weekly_df_copy = weekly_df.copy()
        weekly_df_copy[f'weekly_ma{self.params["weekly_ma_period"]}'] = weekly_df_copy['close'].rolling(
            window=self.params["weekly_ma_period"], min_periods=1).mean()
        return weekly_df_copy

    def _update_active_peak_candidate(self, current_bar_data_with_mas: pd.Series,
                                      df_full_history_with_mas: pd.DataFrame):
        new_candidate_found_this_step = False  # Track if a new peak is found in this specific call
        # Check strokes for peak candidates
        for stroke in reversed(self._strokes):
            if stroke.direction == 1:  # Upward stroke
                original_peak_bar_idx = stroke.end_fractal.kline.high_idx
                # Ensure stroke peak is within the history considered up to current_bar_data_with_mas
                if original_peak_bar_idx > current_bar_data_with_mas.name: continue

                try:
                    peak_bar_data = df_full_history_with_mas.loc[original_peak_bar_idx]
                    peak_price = peak_bar_data['high']
                    ma_at_peak = peak_bar_data.get(f'ma_peak_qualify{self.params["ma_long_for_peak_qualify_period"]}')
                except KeyError:
                    logger.warning(
                        f"[{self.strategy_name}] Peak Idx {original_peak_bar_idx} from stroke not in full history for MA. Stock: {current_bar_data_with_mas.get('symbol', 'N/A')}, Current Bar Date: {current_bar_data_with_mas['date']}")
                    continue

                if pd.isna(ma_at_peak): continue

                if peak_price > ma_at_peak * self.params['ma_peak_threshold']:
                    # Found a valid peak from a stroke
                    if self._active_peak_candidate is None or peak_price > self._active_peak_candidate.price:
                        self._active_peak_candidate = ActivePeakCandidateInfo(
                            price=peak_price, date=peak_bar_data['date'],
                            original_idx=original_peak_bar_idx, ma_at_peak=ma_at_peak, is_ma_valid=True)
                        self._qualified_ref_high = None  # Reset qualified high if a new active peak is stronger
                        new_candidate_found_this_step = True
                        logger.debug(
                            f"[{self.strategy_name}] New active peak candidate from stroke: P={peak_price:.2f} D={peak_bar_data['date']} Idx={original_peak_bar_idx}")

        # Check current bar as a potential peak candidate (e.g. new all time high in recent period)
        current_high = current_bar_data_with_mas['high']
        ma_qualify_curr = current_bar_data_with_mas.get(
            f'ma_peak_qualify{self.params["ma_long_for_peak_qualify_period"]}')

        if pd.notna(ma_qualify_curr) and current_high > ma_qualify_curr * self.params['ma_peak_threshold']:
            if self._active_peak_candidate is None or current_high > self._active_peak_candidate.price:
                self._active_peak_candidate = ActivePeakCandidateInfo(
                    price=current_high, date=current_bar_data_with_mas['date'],
                    original_idx=current_bar_data_with_mas.name, ma_at_peak=ma_qualify_curr, is_ma_valid=True)
                self._qualified_ref_high = None  # Reset qualified high
                new_candidate_found_this_step = True
                logger.debug(
                    f"[{self.strategy_name}] New active peak candidate from current bar: P={current_high:.2f} D={current_bar_data_with_mas['date']} Idx={current_bar_data_with_mas.name}")

        if new_candidate_found_this_step:
            self._uptrend_invalidated = False  # If a new valid peak is found, assume uptrend (for this peak) is not invalidated yet

    def _check_uptrend_invalidation(self, df_full_history_with_mas: pd.DataFrame, current_stock_code: str,
                                    current_eval_date: date):
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Entered _check_uptrend_invalidation.")
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Current _qualified_ref_high: {self._qualified_ref_high}")
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Current _uptrend_invalidated flag: {self._uptrend_invalidated}")

        if not self._qualified_ref_high or not self._qualified_ref_high.is_fully_qualified:
            logger.debug(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Exiting _check_uptrend_invalidation: No qualified_ref_high or not fully qualified.")
            return False

        qrh_idx = self._qualified_ref_high.original_idx
        qrh_date = self._qualified_ref_high.date
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] QualifiedRefHigh: Idx={qrh_idx}, Date={qrh_date}")

        if qrh_idx not in df_full_history_with_mas.index:
            logger.warning(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] QualifiedRefHigh original_idx {qrh_idx} (date {qrh_date}) not found in df_full_history_with_mas. Cannot check invalidation.")
            return False

        qrh_loc = df_full_history_with_mas.index.get_loc(qrh_idx)
        df_after_qrh = df_full_history_with_mas.iloc[qrh_loc + 1:]
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] df_after_qrh has {len(df_after_qrh)} rows. Is empty: {df_after_qrh.empty}")

        if df_after_qrh.empty:
            logger.debug(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Exiting _check_uptrend_invalidation: df_after_qrh is empty.")
            return False

        ma30_threshold_pct = self.params['high_invalidate_close_below_ma30_pct']
        invalidation_reason = None
        logger.debug(
            f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Starting loop for Condition 1 (single day break).")

        # Condition 1: Single day close significantly below MA30
        for idx, row in df_after_qrh.iterrows():
            # --- 您可以在这里也加一些 print 或 logger.debug ---
            logger.debug(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Checking row: Date={row['date']}, Close={row['close']}, MA30={row.get(self.ma30_col_name)}")
            close_price = row['close']
            ma30_val = row.get(self.ma30_col_name)
            if pd.notna(ma30_val) and pd.notna(close_price):
                if close_price < ma30_val * ma30_threshold_pct:
                    invalidation_reason = (
                        f"Close {close_price:.2f} on {row['date']} < {ma30_threshold_pct * 100:.0f}% of MA30 {ma30_val:.2f}. "
                        f"QRH was P={self._qualified_ref_high.price:.2f} on {qrh_date}."
                    )
                    logger.debug(
                        f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Condition 1 met: {invalidation_reason}")
                    break
        if invalidation_reason:
            pass  # Skip to invalidation
        else:
            # Condition 2: Three consecutive closes below MA30
            consecutive_closes_below_ma30 = 0
            for idx, row in df_after_qrh.iterrows():
                close_price = row['close']
                ma30_val = row.get(self.ma30_col_name)
                if pd.notna(ma30_val) and pd.notna(close_price):
                    if close_price < ma30_val:
                        consecutive_closes_below_ma30 += 1
                        if consecutive_closes_below_ma30 >= 3:
                            invalidation_reason = (
                                f"3 consecutive closes below MA30 ending on {row['date']}. "
                                f"QRH was P={self._qualified_ref_high.price:.2f} on {qrh_date}."
                            )
                            break  # Found violation for condition 2
                    else:
                        consecutive_closes_below_ma30 = 0  # Reset counter
                else:  # If data is NaN, reset counter as continuity is broken
                    consecutive_closes_below_ma30 = 0

        if invalidation_reason:
            logger.info(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] QualifiedRefHigh invalidated. Reason: {invalidation_reason}")
            self._active_peak_candidate = None
            self._qualified_ref_high = None
            self._uptrend_invalidated = True  # General flag indicating some form of trend weakness / high invalidation
            return True

        return False

    def _validate_and_set_qualified_ref_high(self, df_full_history_with_mas: pd.DataFrame, current_stock_code: str,
                                             current_eval_date: date):
        if self._uptrend_invalidated:  # If general uptrend invalidated by other means (or previously by new rules)
            self._qualified_ref_high = None
            return False

        if not self._active_peak_candidate or not self._active_peak_candidate.is_ma_valid:
            self._qualified_ref_high = None
            return False

        candidate = self._active_peak_candidate
        if candidate.original_idx not in df_full_history_with_mas.index:
            logger.warning(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Active peak original_idx {candidate.original_idx} (date: {candidate.date}) not found in history for gain calculation.")
            self._qualified_ref_high = None
            return False

        peak_loc = df_full_history_with_mas.index.get_loc(candidate.original_idx)
        # Ensure lookback for gain does not go out of bounds
        start_loc_for_gain = max(0, peak_loc - self.params['peak_recent_gain_days'] + 1)

        # Slice up to and including the peak_loc
        period_df_for_gain = df_full_history_with_mas.iloc[start_loc_for_gain: peak_loc + 1]

        if period_df_for_gain.empty:
            logger.warning(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Empty period_df for gain calculation for peak at {candidate.date} (idx {candidate.original_idx}).")
            self._qualified_ref_high = None
            return False

        high_in_period = period_df_for_gain['high'].max()
        low_in_period = period_df_for_gain['low'].min()

        is_recent_gain_valid = False
        gain_ratio_calculated = np.nan

        if pd.notna(low_in_period) and low_in_period > 1e-9 and pd.notna(
                high_in_period):  # Avoid division by zero/small_number
            gain_ratio_calculated = high_in_period / low_in_period
            if gain_ratio_calculated > self.params['peak_recent_gain_ratio']:
                is_recent_gain_valid = True

        # A peak candidate becomes a qualified reference high if its MA was valid AND recent gain is valid
        is_fully_qualified_now = candidate.is_ma_valid and is_recent_gain_valid

        q_info = QualifiedRefHighInfo(
            price=candidate.price, date=candidate.date, original_idx=candidate.original_idx,
            ma_at_high=candidate.ma_at_peak, is_ma_valid=candidate.is_ma_valid,
            is_recent_gain_valid=is_recent_gain_valid,
            is_fully_qualified=is_fully_qualified_now
        )
        self._qualified_ref_high = q_info  # Store it regardless of full qualification for logging/debugging

        if is_fully_qualified_now:
            logger.debug(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Set qualified ref high: P={q_info.price:.2f} D={q_info.date}, GainRatio={gain_ratio_calculated:.2f}. Details: {q_info}")
        else:
            reason_for_fail = []
            if not candidate.is_ma_valid: reason_for_fail.append("MA_at_peak_invalid")
            if not is_recent_gain_valid: reason_for_fail.append(
                f"RecentGainInvalid (Ratio:{gain_ratio_calculated:.2f} vs Need:{self.params['peak_recent_gain_ratio']})")
            logger.debug(
                f"[{self.strategy_name}@{current_stock_code}@{current_eval_date}] Active peak P={candidate.price:.2f} D={candidate.date} did NOT qualify. Reasons: {', '.join(reason_for_fail)}. StoredInfo: {q_info}")
        return is_fully_qualified_now

    def run_for_stock(self, stock_code: str, current_date: date, data: Dict[str, pd.DataFrame]) -> List[StrategyResult]:
        self._initialize_state_for_stock()
        results: List[StrategyResult] = []

        daily_df_orig = data.get("daily")
        if daily_df_orig is None or daily_df_orig.empty or 'date' not in daily_df_orig.columns:
            logger.warning(
                f"[{current_date.isoformat()}] {self.strategy_name}: {stock_code} No daily data or 'date' column missing.")
            return results

        daily_df = daily_df_orig.copy()
        if not pd.api.types.is_datetime64_any_dtype(daily_df['date']):
            daily_df['date'] = pd.to_datetime(daily_df['date'])
        if hasattr(daily_df['date'].dt, 'date'):
            daily_df['date'] = daily_df['date'].dt.date
        else:
            daily_df['date'] = daily_df['date'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)

        # Filter data up to and including the current_date for analysis
        df_full_history_for_analysis = daily_df[daily_df['date'] <= current_date].copy()
        if df_full_history_for_analysis.empty:
            logger.info(
                f"[{current_date.isoformat()}] {self.strategy_name}: {stock_code} No historical daily data up to current date.")
            return results

        min_bars_needed = max(self.params['ma_long'], self.params['ma_long_for_peak_qualify_period'],
                              self.params['peak_recent_gain_days']) + self.params.get('fractal_definition_lookback',
                                                                                      1) + 5  # Ensure enough for all calcs
        if len(df_full_history_for_analysis) < min_bars_needed:
            logger.info(
                f"[{self.strategy_name}@{stock_code}@{current_date}] Data too short ({len(df_full_history_for_analysis)} < {min_bars_needed}) for full analysis.")
            return results

        df_full_history_with_mas = self._calculate_mas(df_full_history_for_analysis.copy())
        if df_full_history_with_mas.empty:  # Should not happen if previous checks pass
            logger.error(
                f"[{self.strategy_name}@{stock_code}@{current_date}] df_full_history_with_mas is empty after MA calculation.")
            return results
        current_bar_data_with_mas = df_full_history_with_mas.iloc[-1]

        # --- Stroke and Fractal Analysis ---
        # Determine date range for pattern identification (fractals/strokes)
        # Look back N days from the current evaluation date
        lookback_start_date_for_pattern = current_date - timedelta(
            days=self.params['pattern_identification_lookback_days'])

        # Slice the historical data (without MAs yet for this part) for pattern analysis period
        df_slice_for_patterns = df_full_history_for_analysis[
            df_full_history_for_analysis['date'] >= lookback_start_date_for_pattern].copy()

        if not df_slice_for_patterns.empty:
            raw_klines_for_pattern_period = self._df_to_raw_klines(df_slice_for_patterns)
            if raw_klines_for_pattern_period:  # Build merged k-lines and identify fractals/strokes
                self._merged_klines_state = []  # Re-initialize for current run
                self._current_segment_trend_state = 0
                for raw_kline in raw_klines_for_pattern_period:
                    self._process_raw_kline_for_merging(raw_kline)
                self._finalize_merged_kline_directions()

                if len(self._merged_klines_state) >= (2 * self.params['fractal_definition_lookback'] + 1):
                    self._fractals = self._identify_fractals_batch()
                    if len(self._fractals) >= 2:
                        self._strokes = self._connect_fractals_to_strokes_batch()
        # --- End Stroke and Fractal Analysis ---

        weekly_ma_val = None  # Initialize weekly MA value
        weekly_df_orig = data.get("weekly")
        if weekly_df_orig is not None and not weekly_df_orig.empty and 'date' in weekly_df_orig.columns:
            weekly_df = weekly_df_orig.copy()
            if not pd.api.types.is_datetime64_any_dtype(weekly_df['date']):
                weekly_df['date'] = pd.to_datetime(weekly_df['date'])

            # Ensure date column is python date objects
            if hasattr(weekly_df['date'].dt, 'date'):
                weekly_df['date'] = weekly_df['date'].dt.date
            else:
                weekly_df['date'] = weekly_df['date'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)

            weekly_df_history = weekly_df[weekly_df['date'] <= current_date].copy()
            if not weekly_df_history.empty:
                weekly_df_with_mas = self._calculate_weekly_mas(weekly_df_history)
                if not weekly_df_with_mas.empty:
                    # Get MA from the last row of the weekly data (up to current_date)
                    weekly_ma_val = weekly_df_with_mas.iloc[-1].get(f'weekly_ma{self.params["weekly_ma_period"]}')

        # Core Logic Flow:
        # 1. Check if the current qualified high (if any) is invalidated by NEW rules.
        #    This uses df_full_history_with_mas which has all data up to current_eval_date.
        if self._check_uptrend_invalidation(df_full_history_with_mas, stock_code, current_date):
            logger.debug(
                f"[{self.strategy_name}@{stock_code}@{current_date}] Qualified high was invalidated by new rules.")
            # _qualified_ref_high and _active_peak_candidate are reset within the method if invalidated.
            # _uptrend_invalidated is also set.

        # 2. If not invalidated (or no QRH to invalidate), try to update/find an active peak candidate.
        #    This uses strokes (from pattern window) and current bar data.
        #    The df_full_history_with_mas is used to get MA values at the time of those peaks.
        if not self._uptrend_invalidated:  # Only try to find new peaks if trend not marked as generally invalid
            self._update_active_peak_candidate(current_bar_data_with_mas, df_full_history_with_mas)

        # 3. Try to validate the current active_peak_candidate to become a qualified_ref_high.
        #    This also uses df_full_history_with_mas for gain calculations.
        #    This step will overwrite _qualified_ref_high if a new one is validated.
        #    If _uptrend_invalidated was set by _check_uptrend_invalidation, this validation will likely fail or be skipped internally.
        self._validate_and_set_qualified_ref_high(df_full_history_with_mas, stock_code, current_date)

        # --- Signal Generation ---
        if self._qualified_ref_high and self._qualified_ref_high.is_fully_qualified:
            qrh = self._qualified_ref_high
            ref_high_price = qrh.price
            current_close = current_bar_data_with_mas['close']
            current_low = current_bar_data_with_mas['low']
            ma_short_val = current_bar_data_with_mas.get(f'ma{self.params["ma_short"]}')
            ma_long_val = current_bar_data_with_mas.get(self.ma30_col_name)  # Use stored MA30 col name

            ma_long_prev_val = None
            if len(df_full_history_with_mas) > 1:  # Ensure there is a previous bar
                # Get MA30 from the second to last row (previous bar)
                ma_long_prev_val = df_full_history_with_mas.iloc[-2].get(self.ma30_col_name)

            # Prepare strings for logging, handling potential NaN values
            ma_long_str = f"{ma_long_val:.2f}" if pd.notna(ma_long_val) else "N/A"
            weekly_ma_str = f"{weekly_ma_val:.2f}" if pd.notna(weekly_ma_val) else "N/A"
            ma_short_str = f"{ma_short_val:.2f}" if pd.notna(ma_short_val) else "N/A"
            ma_long_prev_str = f"{ma_long_prev_val:.2f}" if pd.notna(ma_long_prev_val) else "N/A"

            # Buy Conditions (Unchanged from your provided snippet)
            cond_A_pullback_from_high = current_close < ref_high_price
            cond_B_near_ma30 = pd.notna(ma_long_val) and \
                               (current_low <= ma_long_val * 1.05) and \
                               (current_low >= ma_long_val * 0.97)  # Original 0.97 threshold for pullback buy
            cond_C_above_weekly_ma = pd.notna(weekly_ma_val) and current_close > weekly_ma_val
            cond_D_short_ma_above_long_ma = pd.notna(ma_short_val) and pd.notna(ma_long_val) and \
                                            ma_short_val > ma_long_val
            cond_E_long_ma_rising = pd.notna(ma_long_val) and pd.notna(ma_long_prev_val) and \
                                    round(ma_long_val, 2) >= round(ma_long_prev_val, 2)

            logger.debug(
                f"[{stock_code}@{current_date}] QRH: P={qrh.price:.2f} D={qrh.date}. "
                f"SignalEval: C={current_close:.2f} L={current_low:.2f} MA{self.params['ma_short']}={ma_short_str} MA{self.params['ma_long']}={ma_long_str} PrevMA{self.params['ma_long']}={ma_long_prev_str} WMA={weekly_ma_str}"
            )
            logger.debug(
                f"[{stock_code}@{current_date}] Conditions: A(PullbackFromHigh):{cond_A_pullback_from_high}, B(NearMA30):{cond_B_near_ma30}, "
                f"C(AboveWMA):{cond_C_above_weekly_ma}, D(MAShort>MALong):{cond_D_short_ma_above_long_ma}, E(MALongRising):{cond_E_long_ma_rising}"
            )

            if cond_A_pullback_from_high and cond_B_near_ma30 and cond_C_above_weekly_ma and \
                    cond_D_short_ma_above_long_ma and cond_E_long_ma_rising:

                signal_details = {
                    "reference_high_price": f"{ref_high_price:.2f}",
                    "reference_high_date": qrh.date.isoformat() if isinstance(qrh.date, date) else str(qrh.date),
                    "current_close": f"{current_close:.2f}",
                    "current_low": f"{current_low:.2f}",
                    f"ma{self.params['ma_short']}": ma_short_str,
                    f"ma{self.params['ma_long']} (buy_ref)": ma_long_str,
                    f"ma_peak_qualify{self.params['ma_long_for_peak_qualify_period']} (at_qrh)": f"{qrh.ma_at_high:.2f}" if pd.notna(
                        qrh.ma_at_high) else "N/A",
                    f"weekly_ma{self.params['weekly_ma_period']}": weekly_ma_str,
                    "CondA_PullbackFromHigh": cond_A_pullback_from_high, "CondB_NearMA30": cond_B_near_ma30,
                    "CondC_AboveWMA": cond_C_above_weekly_ma, "CondD_MAShort>MALong": cond_D_short_ma_above_long_ma,
                    "CondE_MALongRising": cond_E_long_ma_rising,
                    "QrhDetails": str(qrh)
                }
                results.append(
                    StrategyResult(stock_code, current_date, self.strategy_name, "BUY", details=signal_details))

                logger.info(
                    f"[{self.strategy_name}@{stock_code}@{current_date}] BUY SIGNAL. RefHigh P={ref_high_price:.2f} D={qrh.date}. "
                    f"Trig C={current_close:.2f} L={current_low:.2f} MA{self.params['ma_long']}BuyRef={ma_long_str} WMA={weekly_ma_str}")
            elif qrh and qrh.is_fully_qualified:  # Log if QRH exists but other conditions not met
                logger.info(
                    f"[{self.strategy_name}@{stock_code}@{current_date}] Qualified high P={qrh.price:.2f} D={qrh.date} found but other BUY conditions not met.")
        else:  # No qualified reference high, or it's not fully qualified
            qrh_status = "None"
            if self._qualified_ref_high:
                qrh_status = f"Exists but NotFullyQualified (P={self._qualified_ref_high.price:.2f} D={self._qualified_ref_high.date} FullyQual={self._qualified_ref_high.is_fully_qualified})"
            elif self._active_peak_candidate:
                qrh_status = f"NoQRH, ActivePeak P={self._active_peak_candidate.price:.2f} D={self._active_peak_candidate.date}"
            elif self._uptrend_invalidated:
                qrh_status = "UptrendInvalidated earlier"

            logger.info(
                f"[{self.strategy_name}@{stock_code}@{current_date}] No BUY signal. Reason: No (or not fully qualified) reference high. QRH_Status: {qrh_status}")
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG level for detailed output
        format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    test_logger = logging.getLogger(__name__ + "_test_main")

    # --- Test Configuration ---
    # Ensure this CSV file is in the same directory or provide a full path
    csv_file_path = 'stock_daily.csv'
    stock_to_test = "000425"  # Example stock code
    # Choose a date for testing that allows for sufficient historical data
    date_to_test_str = "2025-05-09"  # Example date, adjust as needed
    # --- End Test Configuration ---

    target_date = datetime.strptime(date_to_test_str, "%Y-%m-%d").date()

    try:
        test_logger.info(f"Loading data from: {csv_file_path}")
        try:
            # Assuming CSV has columns: date, open, high, low, close, volume, symbol
            daily_df_full = pd.read_csv(csv_file_path, thousands=',', dtype={'symbol': str})
            test_logger.info(f"CSV columns: {daily_df_full.columns.tolist()}")
        except FileNotFoundError:
            test_logger.error(f"CSV file not found at {csv_file_path}. Please check the path and filename.")
            exit()
        except Exception as e:
            test_logger.error(f"Error loading CSV: {e}")
            exit()

        test_logger.info(f"Loaded {len(daily_df_full)} total rows from CSV.")

        daily_df_stock = daily_df_full[daily_df_full['symbol'] == stock_to_test].copy()
        if daily_df_stock.empty:
            test_logger.error(f"No data found for stock {stock_to_test} in the CSV.")
            exit()

        test_logger.info(f"Found {len(daily_df_stock)} rows for stock {stock_to_test}.")

        # Data Preprocessing
        # Ensure 'date' is datetime and then convert to date objects
        try:
            daily_df_stock['date'] = pd.to_datetime(daily_df_stock['date'])  # Handles various date formats
        except Exception as e:
            test_logger.error(
                f"Error converting 'date' column to datetime for stock {stock_to_test}: {e}. Ensure date format is consistent (e.g., YYYY-MM-DD or YYYY/MM/DD).")
            exit()

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            daily_df_stock[col] = pd.to_numeric(daily_df_stock[col], errors='coerce')

        daily_df_stock.dropna(subset=numeric_cols,
                              inplace=True)  # Drop rows where essential price/volume data is missing
        daily_df_stock.sort_values(by='date', inplace=True)
        daily_df_stock.reset_index(drop=True, inplace=True)  # Reset index after sorting and potential drops

        test_logger.info(f"Preprocessed daily data for {stock_to_test}. Rows after cleaning: {len(daily_df_stock)}")

        if len(daily_df_stock) > 0:
            min_date_in_csv = daily_df_stock['date'].dt.date.min()
            max_date_in_csv = daily_df_stock['date'].dt.date.max()
            test_logger.debug(
                f"Date range for selected stock {stock_to_test} in CSV: {min_date_in_csv} to {max_date_in_csv}")
            if target_date < min_date_in_csv or target_date > max_date_in_csv:
                test_logger.warning(
                    f"Target date {target_date} is outside the CSV data range for {stock_to_test} ({min_date_in_csv} to {max_date_in_csv}).")
                if target_date > max_date_in_csv:
                    test_logger.error(
                        "Target date is after the last date in CSV. Cannot test accurately for this date.")
                    # exit() # Allow to run if target date is within, but just a warning for now
            if target_date < min_date_in_csv:
                test_logger.error("Target date is before the first date in CSV. Cannot test.")
                exit()

        else:
            test_logger.error(f"No daily data remaining for {stock_to_test} after preprocessing. Check data quality.")
            exit()

        # Prepare daily data for strategy (convert datetime64[ns] 'date' to python date objects)
        daily_df_stock_for_strategy = daily_df_stock.copy()
        daily_df_stock_for_strategy['date'] = daily_df_stock_for_strategy['date'].dt.date

        # Generate Weekly Data from Daily Data
        # Ensure the 'date' column is a DatetimeIndex for resampling
        daily_df_for_weekly_resample = daily_df_stock.set_index(pd.DatetimeIndex(daily_df_stock['date'])).copy()
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'first'}

        # Resample to weekly, ending on Friday. label='right' means the week is labeled by its end date.
        weekly_df_stock = daily_df_for_weekly_resample.resample('W-FRI', label='right', closed='right').agg(agg_dict)
        weekly_df_stock.dropna(subset=['close'],
                               inplace=True)  # Drop weeks with no closing price (e.g., holidays if not handled)
        weekly_df_stock.reset_index(inplace=True)  # 'date' column is now the end-of-week date

        # Convert weekly 'date' column to python date objects
        weekly_df_stock['date'] = weekly_df_stock['date'].dt.date

        test_logger.info(f"Generated {len(weekly_df_stock)} weekly data rows for {stock_to_test}.")

        # Setup and Run Strategy
        strategy_context = StrategyContext(current_date=target_date, strategy_params={
            "AdaptedMAPullbackStrategy": {
                # You can override default_params here if needed for the test
                # 'peak_recent_gain_days': 25,
                # 'high_invalidate_close_below_ma30_pct': 0.98
            }
        })
        strategy_instance = AdaptedMAPullbackStrategy(context=strategy_context)
        test_logger.info(
            f"Strategy params being used for {strategy_instance.strategy_name}: {strategy_instance.params}")

        data_for_strategy = {
            "daily": daily_df_stock_for_strategy,
            "weekly": weekly_df_stock
        }

        test_logger.info(f"Running strategy for {stock_to_test} on {target_date.isoformat()}...")
        signals = strategy_instance.run_for_stock(
            stock_code=stock_to_test,
            current_date=target_date,
            data=data_for_strategy
        )

        if signals:
            test_logger.info(f"--- Signals for {stock_to_test} on {target_date.isoformat()} ---")
            for signal_idx, signal in enumerate(signals):
                test_logger.info(f"Signal {signal_idx + 1}: {signal}")
        else:
            test_logger.info(f"No BUY signals generated for {stock_to_test} on {target_date.isoformat()}.")

    except Exception as e:
        test_logger.error(f"An error occurred during the test run: {e}", exc_info=True)