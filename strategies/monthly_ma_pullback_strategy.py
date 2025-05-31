# strategies/monthly_ma_pullback_strategy.py
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from typing import List, Dict, Optional, Any
import logging
from collections import namedtuple
from dataclasses import dataclass, field
import sys
import os

# --- 基础类定义 ---
try:
    from .base_strategy import BaseStrategy, StrategyResult, StrategyContext
except ImportError:
    logger_base_mock = logging.getLogger(__name__ + "_base_mock")
    logger_base_mock.warning(
        "Could not import from .base_strategy. Using minimal mock definitions."
    )


    @dataclass
    class StrategyContext:
        db_session: Optional[Any] = None;
        current_date: Optional[date] = None
        strategy_params: Dict[str, Any] = field(default_factory=dict)


    @dataclass
    class StrategyResult:
        stock_code: str;
        signal_date: date;
        strategy_name: str
        signal_type: str = "BUY";
        signal_score: Optional[float] = None
        details: Dict[str, Any] = field(default_factory=dict);
        timeframe: Optional[str] = None

        def __str__(self):
            details_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            tf_str = f", timeframe='{self.timeframe}'" if self.timeframe else ""
            return (f"StrategyResult(stock_code='{self.stock_code}', signal_date='{self.signal_date}', "
                    f"strategy_name='{self.strategy_name}', signal_type='{self.signal_type}'{tf_str}, details={{{details_str}}})")


    class BaseStrategy:
        def __init__(self, context: StrategyContext): self.context = context; self.params = {}

        @property
        def strategy_name(self) -> str: raise NotImplementedError

        def run_for_stock(self, stock_code: str, current_eval_date: date, data: Dict[str, pd.DataFrame]) -> List[
            StrategyResult]: raise NotImplementedError

# --- 缠论相关数据结构 ---
KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx', 'original_idx'])
MergedKLine = namedtuple('MergedKLine',
                         ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx', 'raw_kline_indices'])
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction', 'start_m_idx', 'end_m_idx'])

# --- 策略特定状态数据类 ---
QualifiedRefHighInfoM = namedtuple('QualifiedRefHighInfoM',
                                   ['price', 'date', 'original_idx', 'ma_at_high', 'is_ma_valid',
                                    'is_recent_gain_valid', 'is_fully_qualified'])
ActivePeakCandidateInfoM = namedtuple('ActivePeakCandidateInfoM',
                                      ['price', 'date', 'original_idx', 'ma_at_peak', 'is_ma_valid'])

logger = logging.getLogger(__name__)


class MonthlyMAPullbackStrategy(BaseStrategy):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        default_params = {
            'monthly_ma_short': 5,
            'monthly_ma_long': 30,
            'monthly_ma_peak_qualify_period': 20,
            'monthly_ma_peak_threshold': 1.20,
            'monthly_peak_recent_gain_periods': 18,
            'monthly_peak_recent_gain_ratio': 1.30,
            'monthly_invalidate_close_below_ma_long_pct': 0.97,
            'monthly_invalidate_consecutive_closes_below_ma': 3,
            'monthly_pullback_ma_touch_upper_pct': 1.05,
            'monthly_pullback_ma_close_lower_pct': 0.97,
            'monthly_fractal_definition_lookback': 1,
            'monthly_min_bars_between_fractals_bt': 0,
            'monthly_chanlun_lookback_periods': 150,
        }
        strategy_specific_params = self.context.strategy_params.get(self.strategy_name, {})
        self.params = {**default_params, **strategy_specific_params}
        self.params['monthly_stroke_min_len_merged_klines'] = self.params['monthly_min_bars_between_fractals_bt'] + 2

        self.ma_long_col_m = f'm_ma{self.params["monthly_ma_long"]}'
        self.ma_short_col_m = f'm_ma{self.params["monthly_ma_short"]}'
        self.ma_peak_qualify_col_m = f'm_ma_peak_q_{self.params["monthly_ma_peak_qualify_period"]}'

        self._initialize_state_for_stock()

    @property
    def strategy_name(self) -> str:
        return "MonthlyMAPullbackStrategy"

    def _initialize_state_for_stock(self):
        self._merged_klines_m: List[MergedKLine] = []
        self._current_segment_trend_m_state: int = 0
        self._fractals_m: List[Fractal] = []
        self._strokes_m: List[Stroke] = []
        # _active_peak_candidate_m 概念被融入到 _get_potential_peak_candidates_monthly
        self._qualified_ref_high_m: Optional[QualifiedRefHighInfoM] = None
        logger.debug(f"[{self.strategy_name}] 月线策略K线及QRH状态已为新评估周期初始化。")

    def _get_potential_peak_candidates_monthly(self, current_bar_monthly_with_mas: pd.Series,
                                               df_monthly_history_with_mas: pd.DataFrame) -> List[
        ActivePeakCandidateInfoM]:
        potential_candidates: List[ActivePeakCandidateInfoM] = []
        logger.debug(f"[{self.strategy_name}] 收集潜在月线峰值候选，当前评估月: {current_bar_monthly_with_mas['date']}")

        # 1. 从缠论笔的顶分型寻找
        if self._strokes_m:
            for stroke_idx_enum, stroke in enumerate(reversed(self._strokes_m)):
                if stroke.direction == 1:
                    peak_fractal = stroke.end_fractal
                    peak_merged_kline = peak_fractal.kline
                    original_peak_bar_idx = peak_merged_kline.high_idx
                    if original_peak_bar_idx > current_bar_monthly_with_mas.name: continue
                    try:
                        peak_bar_data = df_monthly_history_with_mas.loc[original_peak_bar_idx]
                        peak_price = peak_bar_data['high']
                        peak_date_obj = peak_bar_data['date']
                        ma_at_peak = peak_bar_data.get(self.ma_peak_qualify_col_m)
                        if pd.notna(ma_at_peak) and peak_price > ma_at_peak * self.params['monthly_ma_peak_threshold']:
                            potential_candidates.append(ActivePeakCandidateInfoM(
                                peak_price, peak_date_obj, original_peak_bar_idx, ma_at_peak, True))
                            logger.debug(
                                f"  月线笔 {stroke_idx_enum} 顶 ({peak_date_obj}, Px:{peak_price:.2f}) 满足MA阈值，加入候选。")
                    except KeyError:
                        continue

        # 2. 考虑当前K线自身的高点
        current_high_m = current_bar_monthly_with_mas['high']
        current_bar_original_idx_m = current_bar_monthly_with_mas.name
        current_bar_date_m = current_bar_monthly_with_mas['date']
        ma_qual_curr_m = current_bar_monthly_with_mas.get(self.ma_peak_qualify_col_m)
        if pd.notna(ma_qual_curr_m) and current_high_m > ma_qual_curr_m * self.params['monthly_ma_peak_threshold']:
            potential_candidates.append(ActivePeakCandidateInfoM(
                current_high_m, current_bar_date_m, current_bar_original_idx_m, ma_qual_curr_m, True))
            logger.debug(f"  当前月 ({current_bar_date_m}) 自身高点 H={current_high_m:.2f} 满足MA阈值，加入候选。")

        # 去重并按价格排序 (价格高的优先)
        unique_candidates = []
        seen_dates_prices = set()
        for cand in sorted(potential_candidates, key=lambda x: x.price, reverse=True):
            if (cand.date, cand.price) not in seen_dates_prices:
                unique_candidates.append(cand)
                seen_dates_prices.add((cand.date, cand.price))

        logger.debug(f"  共找到 {len(unique_candidates)} 个初步月线峰值候选。")
        return unique_candidates

    def _check_if_qrh_has_invalidated_monthly(self,
                                              qrh_candidate_date: date,
                                              qrh_candidate_original_idx: int,
                                              current_eval_date: date,
                                              df_monthly_history_with_mas: pd.DataFrame,
                                              stock_code: str) -> tuple[bool, Optional[str]]:
        """检查一个给定的QRH候选从其形成后到当前评估日之间是否已失效"""
        try:
            qrh_loc_in_history = df_monthly_history_with_mas.index.get_loc(qrh_candidate_original_idx)
        except KeyError:
            logger.error(
                f"[{self.strategy_name}@{stock_code}] 在检查失效时, QRH候选 original_idx ({qrh_candidate_original_idx}) 在历史中未找到。")
            return True, "QRH index not found in history"

        df_to_check_invalidation = df_monthly_history_with_mas[
            (df_monthly_history_with_mas['date'] > qrh_candidate_date) &
            (df_monthly_history_with_mas['date'] < current_eval_date)
            ]

        if df_to_check_invalidation.empty:
            return False, None

        reason = None
        # 条件1: 单周期最低价大幅低于长期均线
        # 参数名 'monthly_invalidate_close_below_ma_long_pct' 保持不变，但现在用于最低价
        pct_thresh_low = self.params['monthly_invalidate_close_below_ma_long_pct']
        for _, row in df_to_check_invalidation.iterrows():
            low_price, ma_val = row['low'], row.get(self.ma_long_col_m)  # 使用最低价
            if pd.notna(ma_val) and pd.notna(low_price) and (low_price < ma_val * pct_thresh_low):
                reason = f"M.Low {low_price:.2f} on {row['date']} < {pct_thresh_low * 100:.0f}% of MA30M {ma_val:.2f}"
                logger.debug(f"  失效原因 (单周期破位): {reason}")
                return True, reason

        # 条件2: 连续多周期收盘价低于长期均线 (这个条件保持不变，仍基于收盘价)
        consec_below = 0
        consec_needed = self.params['monthly_invalidate_consecutive_closes_below_ma']
        for _, row in df_to_check_invalidation.iterrows():
            close_price, ma_val = row['close'], row.get(self.ma_long_col_m)  # 这里仍然用收盘价
            if pd.notna(ma_val) and pd.notna(close_price):
                if close_price < ma_val:
                    consec_below += 1
                    if consec_below >= consec_needed:
                        reason = f"{consec_needed} consec. M.Closes < MA30M ending {row['date']}"
                        logger.debug(f"  失效原因 (连续收盘破位): {reason}")
                        return True, reason
                else:
                    consec_below = 0
            else:  # 如果数据无效，重置连续计数
                consec_below = 0

        return False, None

    def _get_highest_valid_qrh_monthly(self,
                                       potential_peak_candidates: List[ActivePeakCandidateInfoM],
                                       df_monthly_history_with_mas: pd.DataFrame,
                                       current_eval_date: date,
                                       stock_code: str) -> Optional[QualifiedRefHighInfoM]:
        valid_qrhs: List[QualifiedRefHighInfoM] = []
        logger.debug(
            f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 开始从 {len(potential_peak_candidates)} 个初步候选者中筛选有效QRH-M。")

        for candidate in potential_peak_candidates:
            # 1. 验证近期涨幅
            try:
                peak_iloc = df_monthly_history_with_mas.index.get_loc(candidate.original_idx)
            except KeyError:
                logger.warning(
                    f"  候选 {candidate.date} Px:{candidate.price:.2f} 的索引 {candidate.original_idx} 未在历史中找到，跳过。")
                continue

            start_iloc_for_gain = max(0, peak_iloc - self.params['monthly_peak_recent_gain_periods'] + 1)
            period_df_for_gain = df_monthly_history_with_mas.iloc[start_iloc_for_gain: peak_iloc + 1]
            if period_df_for_gain.empty: continue

            high_in_period = period_df_for_gain['high'].max()
            low_in_period = period_df_for_gain['low'].min()
            is_recent_gain_valid = False
            gain_ratio_calculated = np.nan
            if pd.notna(low_in_period) and low_in_period > 1e-9 and pd.notna(high_in_period):
                gain_ratio_calculated = high_in_period / low_in_period
                if gain_ratio_calculated >= self.params['monthly_peak_recent_gain_ratio']:
                    is_recent_gain_valid = True

            if not (candidate.is_ma_valid and is_recent_gain_valid):
                logger.debug(
                    f"  候选 {candidate.date} Px:{candidate.price:.2f} 未通过初步验证 (MA_ok:{candidate.is_ma_valid}, Gain_ok:{is_recent_gain_valid})。")
                continue

            # 2. 检查从该候选形成后到当前评估日之间是否已失效
            # 注意：current_eval_date 是本轮评估的日期，失效检查应该检查到这个日期之前
            has_invalidated, invalidation_reason = self._check_if_qrh_has_invalidated_monthly(
                qrh_candidate_date=candidate.date,
                qrh_candidate_original_idx=candidate.original_idx,
                current_eval_date=current_eval_date,  # 传递当前评估日
                df_monthly_history_with_mas=df_monthly_history_with_mas,
                stock_code=stock_code
            )

            if has_invalidated:
                logger.debug(f"  候选QRH-M {candidate.date} Px:{candidate.price:.2f} 已于 {invalidation_reason} 失效。")
                continue

            # 如果通过所有检查，则是一个有效的QRH
            valid_qrhs.append(QualifiedRefHighInfoM(
                price=candidate.price, date=candidate.date, original_idx=candidate.original_idx,
                ma_at_high=candidate.ma_at_peak, is_ma_valid=True,  # 已在传入前确认
                is_recent_gain_valid=True,  # 已确认
                is_fully_qualified=True
            ))
            logger.debug(
                f"  候选QRH-M {candidate.date} Px:{candidate.price:.2f} (GainRatio:{gain_ratio_calculated:.2f}) 验证通过且至今未失效。")

        if not valid_qrhs:
            logger.info(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 未找到任何有效的、未失效的QRH-M。")
            return None

        # 从所有有效的QRH中选择价格最高的
        highest_valid_qrh = max(valid_qrhs, key=lambda q: q.price)
        logger.info(
            f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 选择的最终有效QRH-M: {highest_valid_qrh.price:.2f} @ {highest_valid_qrh.date}")
        return highest_valid_qrh

    def _calculate_monthly_mas(self, df_monthly: pd.DataFrame) -> pd.DataFrame:
        df = df_monthly.copy()
        if 'close' not in df.columns:
            logger.error(f"[{self.strategy_name}] _calculate_monthly_mas: 'close'列不存在于月线数据中。")
            return df
        df[self.ma_short_col_m] = df['close'].rolling(window=self.params['monthly_ma_short'], min_periods=1).mean()
        df[self.ma_long_col_m] = df['close'].rolling(window=self.params['monthly_ma_long'], min_periods=1).mean()
        df[self.ma_peak_qualify_col_m] = df['close'].rolling(window=self.params['monthly_ma_peak_qualify_period'],
                                                             min_periods=1).mean()
        return df

    def _df_to_raw_klines_monthly(self, df_segment_with_original_indices: pd.DataFrame) -> List[KLineRaw]:
        raw_klines = []
        for i in range(len(df_segment_with_original_indices)):
            row_data = df_segment_with_original_indices.iloc[i]
            original_idx_val = df_segment_with_original_indices.index[i]
            current_date_val = row_data['date']
            if isinstance(current_date_val, pd.Timestamp): current_date_val = current_date_val.date()
            raw_klines.append(KLineRaw(
                dt=current_date_val, o=row_data['open'], h=row_data['high'], l=row_data['low'], c=row_data['close'],
                idx=i, original_idx=original_idx_val
            ))
        return raw_klines

    def _process_raw_kline_for_merging_monthly(self, k2_raw: KLineRaw):
        if not self._merged_klines_m:
            mk = MergedKLine(k2_raw.dt, k2_raw.o, k2_raw.h, k2_raw.l, k2_raw.c, k2_raw.original_idx, 0,
                             k2_raw.original_idx, k2_raw.original_idx, [k2_raw.original_idx])
            self._merged_klines_m.append(mk);
            self._current_segment_trend_m_state = 0;
            return
        k1_merged = self._merged_klines_m[-1]
        k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l)
        k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)
        if k1_includes_k2 or k2_includes_k1:
            m_o, m_h, m_l, m_c, m_dt, m_idx_end = k1_merged.o, k1_merged.h, k1_merged.l, k2_raw.c, k2_raw.dt, k2_raw.original_idx
            m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
            m_raw_indices = list(k1_merged.raw_kline_indices) + [k2_raw.original_idx]
            trend_for_inclusion = self._current_segment_trend_m_state
            if k1_includes_k2:
                if trend_for_inclusion == 1:
                    m_h = k1_merged.h
                elif trend_for_inclusion == -1:
                    m_l = k1_merged.l
            elif k2_includes_k1:
                if trend_for_inclusion == 1:
                    if k2_raw.h >= k1_merged.h: m_h, m_high_idx = k2_raw.h, k2_raw.original_idx
                elif trend_for_inclusion == -1:
                    if k2_raw.l <= k1_merged.l: m_l, m_low_idx = k2_raw.l, k2_raw.original_idx
                else:
                    m_h, m_l, m_high_idx, m_low_idx = k2_raw.h, k2_raw.l, k2_raw.original_idx, k2_raw.original_idx
            self._merged_klines_m[-1] = MergedKLine(m_dt, m_o, m_h, m_l, m_c, m_idx_end, k1_merged.direction,
                                                    m_high_idx, m_low_idx, m_raw_indices)
        else:
            if k1_merged.direction == 0 and len(self._merged_klines_m) > 1:
                k_prev_prev = self._merged_klines_m[-2];
                final_k1_dir = 0
                if k1_merged.h > k_prev_prev.h and k1_merged.l > k_prev_prev.l:
                    final_k1_dir = 1
                elif k1_merged.h < k_prev_prev.h and k1_merged.l < k_prev_prev.l:
                    final_k1_dir = -1
                self._merged_klines_m[-1] = k1_merged._replace(direction=final_k1_dir)
            k1_final = self._merged_klines_m[-1];
            new_dir = 0
            if k2_raw.h > k1_final.h and k2_raw.l > k1_final.l:
                new_dir = 1
            elif k2_raw.h < k1_final.h and k2_raw.l < k1_final.l:
                new_dir = -1
            self._current_segment_trend_m_state = new_dir
            mk_new = MergedKLine(k2_raw.dt, k2_raw.o, k2_raw.h, k2_raw.l, k2_raw.c, k2_raw.original_idx, new_dir,
                                 k2_raw.original_idx, k2_raw.original_idx, [k2_raw.original_idx])
            self._merged_klines_m.append(mk_new)

    def _finalize_merged_kline_directions_monthly(self):
        if not self._merged_klines_m: return
        for i in range(len(self._merged_klines_m)):
            if self._merged_klines_m[i].direction == 0:
                final_dir = 0
                if i > 0:
                    if self._merged_klines_m[i].h > self._merged_klines_m[i - 1].h and self._merged_klines_m[i].l > \
                            self._merged_klines_m[i - 1].l:
                        final_dir = 1
                    elif self._merged_klines_m[i].h < self._merged_klines_m[i - 1].h and self._merged_klines_m[i].l < \
                            self._merged_klines_m[i - 1].l:
                        final_dir = -1
                if final_dir == 0 and i < len(self._merged_klines_m) - 1:
                    if self._merged_klines_m[i].h < self._merged_klines_m[i + 1].h and self._merged_klines_m[i].l < \
                            self._merged_klines_m[i + 1].l:
                        final_dir = 1
                    elif self._merged_klines_m[i].h > self._merged_klines_m[i + 1].h and self._merged_klines_m[i].l > \
                            self._merged_klines_m[i + 1].l:
                        final_dir = -1
                if final_dir == 0: final_dir = 1 if self._merged_klines_m[i].c >= self._merged_klines_m[i].o else -1
                self._merged_klines_m[i] = self._merged_klines_m[i]._replace(direction=final_dir)

    def _identify_fractals_batch_monthly(self) -> List[Fractal]:
        fractals: List[Fractal] = [];
        fb = self.params['monthly_fractal_definition_lookback']
        if len(self._merged_klines_m) < (2 * fb + 1): return fractals
        for i in range(fb, len(self._merged_klines_m) - fb):
            k_curr = self._merged_klines_m[i]
            is_top = all(k_curr.h > self._merged_klines_m[i - j].h for j in range(1, fb + 1)) and \
                     all(k_curr.h > self._merged_klines_m[i + j].h for j in range(1, fb + 1))
            is_bottom = all(k_curr.l < self._merged_klines_m[i - j].l for j in range(1, fb + 1)) and \
                        all(k_curr.l < self._merged_klines_m[i + j].l for j in range(1, fb + 1))
            if is_top and is_bottom: continue
            if is_top:
                fractals.append(Fractal(k_curr, i, 1))
            elif is_bottom:
                fractals.append(Fractal(k_curr, i, -1))
        fractals.sort(key=lambda f: (f.m_idx, -f.type));
        return fractals

    def _connect_fractals_to_strokes_batch_monthly(self) -> List[Stroke]:
        strokes: List[Stroke] = [];
        if len(self._fractals_m) < 2: return strokes
        min_bars_between = self.params['monthly_min_bars_between_fractals_bt']
        processed_fractals: List[Fractal] = []
        if not self._fractals_m: return strokes
        current_f_type = 0;
        candidate_f = None
        for f_item in self._fractals_m:
            if f_item.type != current_f_type:
                if candidate_f: processed_fractals.append(candidate_f)
                candidate_f = f_item;
                current_f_type = f_item.type
            else:
                if f_item.type == 1:
                    if f_item.kline.h > candidate_f.kline.h: candidate_f = f_item
                else:
                    if f_item.kline.l < candidate_f.kline.l: candidate_f = f_item
        if candidate_f: processed_fractals.append(candidate_f)
        if len(processed_fractals) < 2: return strokes
        self._fractals_m = processed_fractals
        last_confirmed_f = self._fractals_m[0]
        for i in range(1, len(self._fractals_m)):
            current_f = self._fractals_m[i]
            if current_f.type == last_confirmed_f.type:
                # This should ideally not happen after processing_fractals if it works perfectly
                # If it does, we take the more extreme one as per Chanlun general rules
                if current_f.type == 1 and current_f.kline.h > last_confirmed_f.kline.h:
                    last_confirmed_f = current_f
                elif current_f.type == -1 and current_f.kline.l < last_confirmed_f.kline.l:
                    last_confirmed_f = current_f
                continue

            bars_between = abs(current_f.m_idx - last_confirmed_f.m_idx) - 1
            if bars_between < min_bars_between:
                # Logic if K-lines between fractals are too few
                # For monthly, if min_bars_between is 0, this implies they must be strictly alternating.
                # If they are alternating but still 'too close' by some other metric (not just bar count),
                # the Chanlun purist way is complex. Here, we adopt the simpler 'replace with more extreme'
                # or 'replace with current and continue search' if a valid stroke cannot be formed.
                # With min_bars_between = 0, a "stroke" always has at least 2 merged K-lines (the fractals themselves).
                # The key is that the fractals must alternate.
                # If min_bars_between was > 0, then this block would be more relevant.
                # For now, with min_bars_between = 0, this 'if' block might not be hit often
                # unless there's a logic error in processed_fractals that allows non-alternating types.
                logger.debug(f"  月线分型间合并K线数 {bars_between} < {min_bars_between}。尝试更新 last_confirmed_f。")
                if current_f.type == 1 and current_f.kline.h > last_confirmed_f.kline.h:
                    last_confirmed_f = current_f
                elif current_f.type == -1 and current_f.kline.l < last_confirmed_f.kline.l:
                    last_confirmed_f = current_f
                else:
                    last_confirmed_f = current_f
                continue

            stroke_dir = 0;
            start_f, end_f = last_confirmed_f, current_f
            if start_f.type == -1 and end_f.type == 1:
                if end_f.kline.h > start_f.kline.l: stroke_dir = 1
            elif start_f.type == 1 and end_f.type == -1:
                if end_f.kline.l < start_f.kline.h: stroke_dir = -1
            if stroke_dir != 0:
                strokes.append(Stroke(start_f, end_f, stroke_dir, start_f.m_idx, end_f.m_idx))
            last_confirmed_f = end_f
        self._strokes_m = strokes
        return strokes

    def _run_chanlun_analysis_monthly(self, df_monthly_segment_with_orig_index: pd.DataFrame):
        logger.debug(
            f"[{self.strategy_name}] 为月线数据片段 (长度 {len(df_monthly_segment_with_orig_index)}) 运行缠论分析...")
        min_len_for_fractal = self.params['monthly_fractal_definition_lookback'] * 2 + 1
        if df_monthly_segment_with_orig_index.empty or len(df_monthly_segment_with_orig_index) < min_len_for_fractal:
            self._merged_klines_m, self._fractals_m, self._strokes_m = [], [], []
            logger.debug(f"[{self.strategy_name}] 月线数据片段过短 (需要至少 {min_len_for_fractal} 条)，跳过缠论分析。")
            return

        raw_klines_m = self._df_to_raw_klines_monthly(df_monthly_segment_with_orig_index)
        if not raw_klines_m: logger.warning(f"[{self.strategy_name}] 月线 df 转换到 raw_klines_m 为空。"); return

        self._merged_klines_m = [];
        self._current_segment_trend_m_state = 0
        for rk_m in raw_klines_m: self._process_raw_kline_for_merging_monthly(rk_m)
        self._finalize_merged_kline_directions_monthly()
        logger.debug(f"[{self.strategy_name}] 生成了 {len(self._merged_klines_m)} 条合并后月K线。")

        self._fractals_m = self._identify_fractals_batch_monthly()
        logger.debug(f"[{self.strategy_name}] 识别出 {len(self._fractals_m)} 个月线分型。")

        self._strokes_m = self._connect_fractals_to_strokes_batch_monthly()
        logger.debug(f"[{self.strategy_name}] 连接成 {len(self._strokes_m)} 条月线笔。")
        if self._strokes_m:
            for i, stroke in enumerate(self._strokes_m[-5:]):
                start_k, end_k = stroke.start_fractal.kline, stroke.end_fractal.kline
                log_stroke_idx = len(self._strokes_m) - 5 + i;
                if log_stroke_idx < 0: log_stroke_idx = i
                logger.debug(
                    f"  StrokeM {log_stroke_idx}: Dir={stroke.direction}, Start={start_k.dt}(H:{start_k.h:.2f},L:{start_k.l:.2f}), End={end_k.dt}(H:{end_k.h:.2f},L:{end_k.l:.2f})")


    def run_for_stock(self, stock_code: str, current_eval_date: date, data: Dict[str, pd.DataFrame]) -> List[
        StrategyResult]:
        self._initialize_state_for_stock()  # 只重置K线和当前QRH
        results: List[StrategyResult] = []

        df_monthly_orig = data.get("monthly")
        if df_monthly_orig is None or df_monthly_orig.empty: return results
        df_monthly_history = df_monthly_orig[df_monthly_orig['date'] <= current_eval_date].copy()

        min_len_req = max(self.params['monthly_ma_long'], self.params['monthly_ma_peak_qualify_period'],
                          self.params['monthly_peak_recent_gain_periods'],
                          self.params['monthly_fractal_definition_lookback'] * 2 + 1)
        if len(df_monthly_history) < min_len_req: return results

        df_monthly_history = self._calculate_monthly_mas(df_monthly_history)
        current_bar_m = df_monthly_history[df_monthly_history['date'] == current_eval_date].iloc[0]
        current_bar_m_iloc = df_monthly_history.index.get_loc(current_bar_m.name)

        chanlun_lookback = self.params['monthly_chanlun_lookback_periods']
        start_iloc_for_chanlun = max(0, current_bar_m_iloc - chanlun_lookback + 1)
        df_monthly_for_chanlun = df_monthly_history.iloc[start_iloc_for_chanlun: current_bar_m_iloc + 1].copy()

        if not df_monthly_for_chanlun.empty:
            self._run_chanlun_analysis_monthly(df_monthly_for_chanlun)
        else:
            self._strokes_m = []

        # 1. 获取所有初步合格的峰值候选
        potential_peaks = self._get_potential_peak_candidates_monthly(current_bar_m, df_monthly_history)

        # 2. 从这些候选者中筛选出最高的、有效的、且至今未失效的QRH-M
        self._qualified_ref_high_m = self._get_highest_valid_qrh_monthly(
            potential_peaks, df_monthly_history, current_eval_date, stock_code
        )

        # 3. 如果存在有效的QRH-M，则检查买入信号
        if self._qualified_ref_high_m:
            qrh_m = self._qualified_ref_high_m
            cm_close = current_bar_m['close'];
            cm_low = current_bar_m['low']
            ma30m_val = current_bar_m.get(self.ma_long_col_m)
            ma5m_val = current_bar_m.get(self.ma_short_col_m)
            ma30m_prev_val = None
            if current_bar_m_iloc > 0:
                ma30m_prev_val = df_monthly_history.iloc[current_bar_m_iloc - 1].get(self.ma_long_col_m)

            cond_m1_pullback = cm_close < qrh_m.price
            cond_m2_near_ma30m = pd.notna(ma30m_val) and \
                                 (cm_low <= ma30m_val * self.params['monthly_pullback_ma_touch_upper_pct']) and \
                                 (cm_low >= ma30m_val * self.params['monthly_pullback_ma_close_lower_pct'])
            cond_m3_ma_alignment = pd.notna(ma5m_val) and pd.notna(ma30m_val) and (ma5m_val > ma30m_val)
            cond_m4_ma30m_rising_or_flat = (pd.notna(ma30m_val) and pd.notna(ma30m_prev_val) and \
                                            (round(ma30m_val, 2) >= round(ma30m_prev_val, 2))) or \
                                           (pd.notna(ma30m_val) and ma30m_prev_val is None and (
                                                       current_bar_m_iloc + 1) == 1)

            ma30m_str = f"{ma30m_val:.2f}" if pd.notna(ma30m_val) else "N/A"
            logger.debug(
                f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 月线信号条件评估 (QRH: {qrh_m.price:.2f}@{qrh_m.date}): "
                f"M1(回撤):{cond_m1_pullback}, M2(近MA30M):{cond_m2_near_ma30m} (Low:{cm_low:.2f}, MA30M:{ma30m_str}), "
                f"M3(MA排列):{cond_m3_ma_alignment}, M4(MA30M升/平):{cond_m4_ma30m_rising_or_flat}")

            if cond_m1_pullback and cond_m2_near_ma30m and cond_m3_ma_alignment and cond_m4_ma30m_rising_or_flat:
                # ... (details 和 results.append 不变) ...
                details = {
                    "qrh_m_price": f"{qrh_m.price:.2f}", "qrh_m_date": str(qrh_m.date),
                    "monthly_close": f"{cm_close:.2f}", "monthly_low": f"{cm_low:.2f}",
                    self.ma_long_col_m: ma30m_str,
                    self.ma_short_col_m: f"{ma5m_val:.2f}" if pd.notna(ma5m_val) else "N/A",
                    "conditions_met": {"M1": cond_m1_pullback, "M2": cond_m2_near_ma30m, "M3": cond_m3_ma_alignment,
                                       "M4": cond_m4_ma30m_rising_or_flat}
                }
                results.append(StrategyResult(stock_code, current_eval_date, self.strategy_name, details=details,
                                              timeframe="monthly"))
                logger.info(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 月线级别买入信号产生！")
        else:
            logger.debug(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 无有效QRH-M，不产生信号。")

        return results


# --- Main Test Function (与上一轮基本一致，确保日志级别和测试日期) ---
if __name__ == '__main__':
    # (sys.path modification as before)
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_package_root = os.path.dirname(current_file_dir)
        project_base_dir = os.path.dirname(project_package_root)
        if project_base_dir not in sys.path: sys.path.insert(0, project_base_dir)
    except NameError:
        pass

    # 修正导入路径为相对项目根的绝对路径
    from quant_platform.db.database import SessionLocal
    from quant_platform.utils import data_loader as dl_db
    from quant_platform.db.models import StockDaily
    from sqlalchemy import func
    from sqlalchemy.orm import Session  # 确保 Session 被导入

    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG可以看到更多缠论细节, INFO只看信号和关键QRH信息
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_logger = logging.getLogger(__name__ + "_main_test")

    # --- 要分析的目标日期 ---
    target_stock_code = "000887"
    target_eval_date_str = "2021-04-30"  # 您关注的日期
    # target_eval_date_str = "2015-10-30" # 另一个您可以测试的日期
    target_eval_date = datetime.strptime(target_eval_date_str, "%Y-%m-%d").date()
    temp_context_for_name = StrategyContext()
    strategy_name_for_log = MonthlyMAPullbackStrategy(temp_context_for_name).strategy_name
    main_logger.info(
        f"开始 {strategy_name_for_log} 特定日期 [{target_eval_date_str}] 信号测试 for {target_stock_code}...")

    db_session: Optional[Session] = None
    try:
        db_session = SessionLocal()

        main_logger.info(f"\n\n========== 开始测试股票: {target_stock_code} on {target_eval_date_str} ==========")
        first_daily_date_obj = db_session.query(func.min(StockDaily.date)).filter(
            StockDaily.symbol == target_stock_code).scalar()
        last_daily_date_obj = db_session.query(func.max(StockDaily.date)).filter(
            StockDaily.symbol == target_stock_code).scalar()

        if not first_daily_date_obj or not last_daily_date_obj:
            main_logger.error(f"数据库中未找到股票 {target_stock_code} 的日线数据范围。");
            exit()

        start_date_load = first_daily_date_obj.strftime('%Y-%m-%d')
        end_date_load = last_daily_date_obj.strftime('%Y-%m-%d')

        monthly_df_stock = dl_db.load_monthly_data(symbol=target_stock_code, start_date=start_date_load,
                                                   end_date=end_date_load, db_session=db_session)
        if monthly_df_stock is None or monthly_df_stock.empty:
            main_logger.error(f"未能从数据库加载股票 {target_stock_code} 的月线数据。");
            exit()

        monthly_df_stock['date'] = pd.to_datetime(monthly_df_stock['date']).dt.date
        monthly_df_stock.sort_values(by='date', inplace=True)
        monthly_df_stock.reset_index(drop=True, inplace=True)
        main_logger.info(f"从数据库为 {target_stock_code} 加载并预处理了 {len(monthly_df_stock)} 条月线数据。")

        if target_eval_date not in monthly_df_stock['date'].values:
            main_logger.error(
                f"目标评估日期 {target_eval_date_str} 不在股票 {target_stock_code} 的月线数据中。请检查日期或数据源。")
            exit()

        context = StrategyContext(db_session=db_session, strategy_params={})
        strategy = MonthlyMAPullbackStrategy(context=context)
        main_logger.info(f"[{target_stock_code}] 使用策略参数: {strategy.params}")

        data_slice_for_run = {"monthly": monthly_df_stock[monthly_df_stock['date'] <= target_eval_date].copy()}
        if data_slice_for_run["monthly"].empty:
            main_logger.error(f"评估日期 {target_eval_date.isoformat()}: 月线数据切片为空。");
            exit()

        main_logger.info(f"\n--- [{target_stock_code}] 运行策略: EvalDate={target_eval_date.isoformat()} ---")
        signals = strategy.run_for_stock(
            stock_code=target_stock_code,
            current_eval_date=target_eval_date,
            data=data_slice_for_run
        )
        if signals:
            for sig in signals:
                main_logger.info(
                    f"[{target_stock_code}] 买入信号 @ {sig.signal_date.isoformat()} (策略名: {sig.strategy_name}, 时间级别: {sig.timeframe})\n详细信息: {sig.details}")
        else:
            main_logger.info(
                f"评估日期 {target_eval_date.isoformat()}: 股票 {target_stock_code} 未产生任何月线回踩买入信号。")

    except ImportError as e:
        main_logger.error(f"导入模块失败: {e}.")
    except Exception as e:
        main_logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    finally:
        if db_session: db_session.close(); main_logger.info("数据库会话已关闭。")
    main_logger.info(f"{MonthlyMAPullbackStrategy(StrategyContext()).strategy_name} 特定日期信号测试结束。")