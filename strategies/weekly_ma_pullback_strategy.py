# strategies/weekly_ma_pullback_strategy.py
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from typing import List, Dict, Optional, Any
import logging
from collections import namedtuple
from dataclasses import dataclass, field
import sys
import os

# --- 基础类定义 (保持不变) ---
try:
    from .base_strategy import BaseStrategy, StrategyResult, StrategyContext
except ImportError:
    logger_base_mock = logging.getLogger(__name__ + "_base_mock")
    logger_base_mock.warning(
        "Could not import from .base_strategy. Using minimal mock definitions."
    )


    @dataclass
    class StrategyContext:
        db_session: Optional[Any] = None
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

# --- 缠论相关数据结构 (保持不变) ---
KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx', 'original_idx'])
MergedKLine = namedtuple('MergedKLine',
                         ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx', 'raw_kline_indices'])
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction', 'start_m_idx', 'end_m_idx'])

# --- 策略特定状态数据类 (保持不变) ---
QualifiedRefHighInfoW = namedtuple('QualifiedRefHighInfoW',
                                   ['price', 'date', 'original_idx', 'ma_at_high', 'is_ma_valid',
                                    'is_recent_gain_valid', 'is_fully_qualified'])
ActivePeakCandidateInfoW = namedtuple('ActivePeakCandidateInfoW',  # 用于 _get_potential_peak_candidates_weekly 返回
                                      ['price', 'date', 'original_idx', 'ma_at_peak', 'is_ma_valid'])

logger = logging.getLogger(__name__)


class WeeklyMAPullbackStrategy(BaseStrategy):
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        default_params = {
            'weekly_ma_short': 5,
            'weekly_ma_long': 30,
            'weekly_ma_peak_qualify_period': 30,
            'weekly_ma_peak_threshold': 1.20,
            'weekly_peak_recent_gain_periods': 20,  # 周
            'weekly_peak_recent_gain_ratio': 1.25,  # 25%涨幅
            'weekly_invalidate_close_below_ma_long_pct': 0.97,
            'weekly_invalidate_consecutive_closes_below_ma': 3,
            'weekly_pullback_ma_touch_upper_pct': 1.03,
            'weekly_pullback_ma_close_lower_pct': 0.98,
            'monthly_ma_filter_period': 30,  # 用于月线过滤的MA周期
            'weekly_fractal_definition_lookback': 1,  # 周线3K分型
            'weekly_min_bars_between_fractals_bt': 1,  # 周线分型间至少隔1根合并K线
            'weekly_chanlun_lookback_periods': 200,  # 周线缠论回看期数
        }
        strategy_specific_params = self.context.strategy_params.get(self.strategy_name, {})
        self.params = {**default_params, **strategy_specific_params}
        # 根据分型间K线数计算笔的最小合并K线长度
        self.params['weekly_stroke_min_len_merged_klines'] = self.params['weekly_min_bars_between_fractals_bt'] + 2

        self.ma_long_col_w = f'w_ma{self.params["weekly_ma_long"]}'
        self.ma_short_col_w = f'w_ma{self.params["weekly_ma_short"]}'
        self.ma_peak_qualify_col_w = f'w_ma_peak_q_{self.params["weekly_ma_peak_qualify_period"]}'
        self.ma_monthly_filter_col_m = f'm_ma{self.params["monthly_ma_filter_period"]}'  # 月线MA的列名

        self._initialize_internal_states()

    @property
    def strategy_name(self) -> str:
        return "WeeklyMAPullbackStrategy"

    def _initialize_internal_states(self):
        self._merged_klines_w: List[MergedKLine] = []
        self._current_segment_trend_w_state: int = 0
        self._fractals_w: List[Fractal] = []
        self._strokes_w: List[Stroke] = []
        self._qualified_ref_high_w: Optional[QualifiedRefHighInfoW] = None
        # 注意：不再需要 _active_peak_candidate_w 和 _uptrend_invalidated_w 作为类成员状态
        logger.debug(f"[{self.strategy_name}] 周线策略内部K线及QRH状态已为新评估周期初始化。")

    def _calculate_weekly_mas(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        df = df_weekly.copy()
        if 'close' not in df.columns:
            logger.error(f"[{self.strategy_name}] _calculate_weekly_mas: 'close'列不存在于周线数据中。")
            return df
        df[self.ma_short_col_w] = df['close'].rolling(window=self.params['weekly_ma_short'], min_periods=1).mean()
        df[self.ma_long_col_w] = df['close'].rolling(window=self.params['weekly_ma_long'], min_periods=1).mean()
        df[self.ma_peak_qualify_col_w] = df['close'].rolling(window=self.params['weekly_ma_peak_qualify_period'],
                                                             min_periods=1).mean()
        return df

    def _calculate_monthly_mas(self, df_monthly: pd.DataFrame) -> pd.DataFrame:
        df = df_monthly.copy()
        if 'close' not in df.columns:
            logger.error(f"[{self.strategy_name}] _calculate_monthly_mas: 'close'列不存在于月线数据中。")
            df[self.ma_monthly_filter_col_m] = np.nan  # 确保列存在，即使数据为空
            return df
        if df.empty:  # 如果传入空DF，也确保列存在
            df[self.ma_monthly_filter_col_m] = np.nan
            return df
        df[self.ma_monthly_filter_col_m] = df['close'].rolling(window=self.params['monthly_ma_filter_period'],
                                                               min_periods=1).mean()
        return df

    def _df_to_raw_klines_weekly(self, df_segment_with_original_indices: pd.DataFrame) -> List[KLineRaw]:
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

    def _process_raw_kline_for_merging_weekly(self, k2_raw: KLineRaw):
        if not self._merged_klines_w:
            mk = MergedKLine(k2_raw.dt, k2_raw.o, k2_raw.h, k2_raw.l, k2_raw.c, k2_raw.original_idx, 0,
                             k2_raw.original_idx, k2_raw.original_idx, [k2_raw.original_idx])
            self._merged_klines_w.append(mk);
            self._current_segment_trend_w_state = 0;
            return
        k1_merged = self._merged_klines_w[-1]
        k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l)
        k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)
        if k1_includes_k2 or k2_includes_k1:
            m_o, m_h, m_l, m_c, m_dt, m_idx_end = k1_merged.o, k1_merged.h, k1_merged.l, k2_raw.c, k2_raw.dt, k2_raw.original_idx
            m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
            m_raw_indices = list(k1_merged.raw_kline_indices) + [k2_raw.original_idx]
            trend_for_inclusion = self._current_segment_trend_w_state
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
            self._merged_klines_w[-1] = MergedKLine(m_dt, m_o, m_h, m_l, m_c, m_idx_end, k1_merged.direction,
                                                    m_high_idx, m_low_idx, m_raw_indices)
        else:
            if k1_merged.direction == 0 and len(self._merged_klines_w) > 1:
                k_prev_prev = self._merged_klines_w[-2];
                final_k1_dir = 0
                if k1_merged.h > k_prev_prev.h and k1_merged.l > k_prev_prev.l:
                    final_k1_dir = 1
                elif k1_merged.h < k_prev_prev.h and k1_merged.l < k_prev_prev.l:
                    final_k1_dir = -1
                self._merged_klines_w[-1] = k1_merged._replace(direction=final_k1_dir)
            k1_final = self._merged_klines_w[-1];
            new_dir = 0
            if k2_raw.h > k1_final.h and k2_raw.l > k1_final.l:
                new_dir = 1
            elif k2_raw.h < k1_final.h and k2_raw.l < k1_final.l:
                new_dir = -1
            self._current_segment_trend_w_state = new_dir
            mk_new = MergedKLine(k2_raw.dt, k2_raw.o, k2_raw.h, k2_raw.l, k2_raw.c, k2_raw.original_idx, new_dir,
                                 k2_raw.original_idx, k2_raw.original_idx, [k2_raw.original_idx])
            self._merged_klines_w.append(mk_new)

    def _finalize_merged_kline_directions_weekly(self):
        if not self._merged_klines_w: return
        for i in range(len(self._merged_klines_w)):
            if self._merged_klines_w[i].direction == 0:
                final_dir = 0
                if i > 0:
                    if self._merged_klines_w[i].h > self._merged_klines_w[i - 1].h and self._merged_klines_w[i].l > \
                            self._merged_klines_w[i - 1].l:
                        final_dir = 1
                    elif self._merged_klines_w[i].h < self._merged_klines_w[i - 1].h and self._merged_klines_w[i].l < \
                            self._merged_klines_w[i - 1].l:
                        final_dir = -1
                if final_dir == 0 and i < len(self._merged_klines_w) - 1:
                    if self._merged_klines_w[i].h < self._merged_klines_w[i + 1].h and self._merged_klines_w[i].l < \
                            self._merged_klines_w[i + 1].l:
                        final_dir = 1
                    elif self._merged_klines_w[i].h > self._merged_klines_w[i + 1].h and self._merged_klines_w[i].l > \
                            self._merged_klines_w[i + 1].l:
                        final_dir = -1
                if final_dir == 0: final_dir = 1 if self._merged_klines_w[i].c >= self._merged_klines_w[i].o else -1
                self._merged_klines_w[i] = self._merged_klines_w[i]._replace(direction=final_dir)

    def _identify_fractals_batch_weekly(self) -> List[Fractal]:
        fractals: List[Fractal] = [];
        fb = self.params['weekly_fractal_definition_lookback']
        if len(self._merged_klines_w) < (2 * fb + 1): return fractals
        for i in range(fb, len(self._merged_klines_w) - fb):
            k_curr = self._merged_klines_w[i]
            is_top = all(k_curr.h > self._merged_klines_w[i - j].h for j in range(1, fb + 1)) and \
                     all(k_curr.h > self._merged_klines_w[i + j].h for j in range(1, fb + 1))
            is_bottom = all(k_curr.l < self._merged_klines_w[i - j].l for j in range(1, fb + 1)) and \
                        all(k_curr.l < self._merged_klines_w[i + j].l for j in range(1, fb + 1))
            if is_top and is_bottom: continue
            if is_top:
                fractals.append(Fractal(k_curr, i, 1))
            elif is_bottom:
                fractals.append(Fractal(k_curr, i, -1))
        fractals.sort(key=lambda f: (f.m_idx, -f.type));
        return fractals

    def _connect_fractals_to_strokes_batch_weekly(self) -> List[Stroke]:
        strokes: List[Stroke] = [];
        if len(self._fractals_w) < 2: return strokes
        min_bars_between = self.params['weekly_min_bars_between_fractals_bt']
        processed_fractals: List[Fractal] = []
        if not self._fractals_w: return strokes
        current_f_type = 0;
        candidate_f = None
        for f_item in self._fractals_w:
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
        self._fractals_w = processed_fractals
        last_confirmed_f = self._fractals_w[0]
        for i in range(1, len(self._fractals_w)):
            current_f = self._fractals_w[i]
            if current_f.type == last_confirmed_f.type:
                if current_f.type == 1 and current_f.kline.h > last_confirmed_f.kline.h:
                    last_confirmed_f = current_f
                elif current_f.type == -1 and current_f.kline.l < last_confirmed_f.kline.l:
                    last_confirmed_f = current_f
                continue
            bars_between = abs(current_f.m_idx - last_confirmed_f.m_idx) - 1
            if bars_between < min_bars_between:
                logger.debug(f"  周线分型间合并K线数 {bars_between} < {min_bars_between}。尝试更新 last_confirmed_f。")
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
        self._strokes_w = strokes
        return strokes

    def _run_chanlun_analysis_weekly(self, df_weekly_segment_with_orig_index: pd.DataFrame):
        logger.debug(
            f"[{self.strategy_name}] 为周线数据片段 (长度 {len(df_weekly_segment_with_orig_index)}) 运行缠论分析...")
        self._merged_klines_w, self._fractals_w, self._strokes_w = [], [], []
        self._current_segment_trend_w_state = 0
        min_len_for_fractal = self.params['weekly_fractal_definition_lookback'] * 2 + 1
        if df_weekly_segment_with_orig_index.empty or len(df_weekly_segment_with_orig_index) < min_len_for_fractal:
            logger.debug(f"[{self.strategy_name}] 周线数据片段过短 (需要至少 {min_len_for_fractal} 条)，跳过缠论分析。")
            return
        raw_klines_w = self._df_to_raw_klines_weekly(df_weekly_segment_with_orig_index)
        if not raw_klines_w: logger.warning(f"[{self.strategy_name}] 周线 df 转换到 raw_klines_w 为空。"); return
        for rk_w in raw_klines_w: self._process_raw_kline_for_merging_weekly(rk_w)
        self._finalize_merged_kline_directions_weekly()
        logger.debug(f"[{self.strategy_name}] 生成了 {len(self._merged_klines_w)} 条合并后周K线。")
        self._fractals_w = self._identify_fractals_batch_weekly()
        logger.debug(f"[{self.strategy_name}] 识别出 {len(self._fractals_w)} 个周线分型。")
        self._strokes_w = self._connect_fractals_to_strokes_batch_weekly()
        logger.debug(f"[{self.strategy_name}] 连接成 {len(self._strokes_w)} 条周线笔。")
        if self._strokes_w:
            for i, stroke in enumerate(self._strokes_w[-5:]):
                start_k, end_k = stroke.start_fractal.kline, stroke.end_fractal.kline
                log_stroke_idx = len(self._strokes_w) - 5 + i;
                if log_stroke_idx < 0: log_stroke_idx = i
                logger.debug(
                    f"  StrokeW {log_stroke_idx}: Dir={stroke.direction}, Start={start_k.dt}(H:{start_k.h:.2f},L:{start_k.l:.2f}), End={end_k.dt}(H:{end_k.h:.2f},L:{end_k.l:.2f})")

    def _get_potential_peak_candidates_weekly(self, current_bar_weekly_with_mas: pd.Series,
                                              df_weekly_history_with_mas: pd.DataFrame) -> List[
        ActivePeakCandidateInfoW]:
        potential_candidates: List[ActivePeakCandidateInfoW] = []
        logger.debug(f"[{self.strategy_name}] 收集潜在周线峰值候选，当前评估周: {current_bar_weekly_with_mas['date']}")

        if self._strokes_w:
            for stroke_idx_enum, stroke in enumerate(reversed(self._strokes_w)):
                if stroke.direction == 1:
                    peak_fractal = stroke.end_fractal
                    peak_merged_kline = peak_fractal.kline
                    original_peak_bar_idx = peak_merged_kline.high_idx
                    if original_peak_bar_idx > current_bar_weekly_with_mas.name: continue
                    try:
                        peak_bar_data = df_weekly_history_with_mas.loc[original_peak_bar_idx]
                        peak_price = peak_bar_data['high']
                        peak_date_obj = peak_bar_data['date']
                        ma_at_peak = peak_bar_data.get(self.ma_peak_qualify_col_w)
                        if pd.notna(ma_at_peak) and peak_price > ma_at_peak * self.params['weekly_ma_peak_threshold']:
                            potential_candidates.append(ActivePeakCandidateInfoW(
                                peak_price, peak_date_obj, original_peak_bar_idx, ma_at_peak, True))
                            logger.debug(
                                f"  周线笔 {stroke_idx_enum} 顶 ({peak_date_obj}, Px:{peak_price:.2f}) 满足MA阈值，加入候选。")
                    except KeyError:
                        continue

        current_high_w = current_bar_weekly_with_mas['high']
        current_bar_original_idx_w = current_bar_weekly_with_mas.name
        current_bar_date_w = current_bar_weekly_with_mas['date']
        ma_qual_curr_w = current_bar_weekly_with_mas.get(self.ma_peak_qualify_col_w)
        if pd.notna(ma_qual_curr_w) and current_high_w > ma_qual_curr_w * self.params['weekly_ma_peak_threshold']:
            potential_candidates.append(ActivePeakCandidateInfoW(
                current_high_w, current_bar_date_w, current_bar_original_idx_w, ma_qual_curr_w, True))
            logger.debug(f"  当前周 ({current_bar_date_w}) 自身高点 H={current_high_w:.2f} 满足MA阈值，加入候选。")

        unique_candidates = []
        seen_dates_prices = set()
        for cand in sorted(potential_candidates, key=lambda x: x.price, reverse=True):
            if (cand.date, cand.price) not in seen_dates_prices:
                unique_candidates.append(cand)
                seen_dates_prices.add((cand.date, cand.price))

        logger.debug(f"  共找到 {len(unique_candidates)} 个初步周线峰值候选。")
        return unique_candidates

    def _check_if_qrh_has_invalidated_weekly(self,
                                             qrh_candidate_date: date,
                                             qrh_candidate_original_idx: int,
                                             current_eval_date: date,
                                             df_weekly_history_with_mas: pd.DataFrame,
                                             stock_code: str) -> tuple[bool, Optional[str]]:
        """检查一个给定的QRH候选从其形成后到当前评估日之间是否已失效"""
        try:
            qrh_loc_in_history = df_weekly_history_with_mas.index.get_loc(qrh_candidate_original_idx)
        except KeyError:
            logger.error(
                f"[{self.strategy_name}@{stock_code}] 在检查失效时, QRH候选 original_idx ({qrh_candidate_original_idx}) 在周线历史中未找到。")
            return True, "QRH index not found in history"

        df_to_check_invalidation = df_weekly_history_with_mas[
            (df_weekly_history_with_mas['date'] > qrh_candidate_date) &
            (df_weekly_history_with_mas['date'] < current_eval_date)
            ]

        if df_to_check_invalidation.empty:
            return False, None

        reason = None
        # 条件1: 单周期最低价大幅低于长期均线
        # 参数名 'weekly_invalidate_close_below_ma_long_pct' 保持不变，但现在用于最低价
        pct_thresh_low = self.params['weekly_invalidate_close_below_ma_long_pct']
        for _, row in df_to_check_invalidation.iterrows():
            low_price, ma_val = row['low'], row.get(self.ma_long_col_w)  # 使用最低价
            if pd.notna(ma_val) and pd.notna(low_price) and (low_price < ma_val * pct_thresh_low):
                reason = f"W.Low {low_price:.2f} on {row['date']} < {pct_thresh_low * 100:.0f}% of MA30W {ma_val:.2f}"
                logger.debug(f"  失效原因 (单周期破位): {reason}")
                return True, reason

        # 条件2: 连续多周期收盘价低于长期均线 (这个条件保持不变，仍基于收盘价)
        consec_below = 0
        consec_needed = self.params['weekly_invalidate_consecutive_closes_below_ma']
        for _, row in df_to_check_invalidation.iterrows():
            close_price, ma_val = row['close'], row.get(self.ma_long_col_w)  # 这里仍然用收盘价
            if pd.notna(ma_val) and pd.notna(close_price):
                if close_price < ma_val:
                    consec_below += 1
                    if consec_below >= consec_needed:
                        reason = f"{consec_needed} consec. W.Closes < MA30W ending {row['date']}"
                        logger.debug(f"  失效原因 (连续收盘破位): {reason}")
                        return True, reason
                else:
                    consec_below = 0
            else:  # 如果数据无效，重置连续计数
                consec_below = 0

        return False, None

    def _get_highest_valid_qrh_weekly(self,
                                      potential_peak_candidates: List[ActivePeakCandidateInfoW],
                                      df_weekly_history_with_mas: pd.DataFrame,
                                      current_eval_date: date,
                                      stock_code: str) -> Optional[QualifiedRefHighInfoW]:
        valid_qrhs: List[QualifiedRefHighInfoW] = []
        logger.debug(
            f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 开始从 {len(potential_peak_candidates)} 个初步候选者中筛选有效QRH-W。")

        for candidate in potential_peak_candidates:
            try:
                peak_iloc = df_weekly_history_with_mas.index.get_loc(candidate.original_idx)
            except KeyError:
                logger.warning(
                    f"  候选 {candidate.date} Px:{candidate.price:.2f} 的索引 {candidate.original_idx} 未在历史中找到，跳过。"); continue

            start_iloc_for_gain = max(0, peak_iloc - self.params['weekly_peak_recent_gain_periods'] + 1)
            period_df_for_gain = df_weekly_history_with_mas.iloc[start_iloc_for_gain: peak_iloc + 1]
            if period_df_for_gain.empty: continue

            high_in_period = period_df_for_gain['high'].max();
            low_in_period = period_df_for_gain['low'].min()
            is_recent_gain_valid = False;
            gain_ratio_calculated = np.nan
            if pd.notna(low_in_period) and low_in_period > 1e-9 and pd.notna(high_in_period):
                gain_ratio_calculated = high_in_period / low_in_period
                if gain_ratio_calculated >= self.params['weekly_peak_recent_gain_ratio']: is_recent_gain_valid = True

            if not (candidate.is_ma_valid and is_recent_gain_valid):
                logger.debug(
                    f"  候选 {candidate.date} Px:{candidate.price:.2f} 未通过初步验证 (MA_ok:{candidate.is_ma_valid}, Gain_ok:{is_recent_gain_valid})。")
                continue

            has_invalidated, invalidation_reason = self._check_if_qrh_has_invalidated_weekly(
                qrh_candidate_date=candidate.date, qrh_candidate_original_idx=candidate.original_idx,
                current_eval_date=current_eval_date, df_weekly_history_with_mas=df_weekly_history_with_mas,
                stock_code=stock_code)
            if has_invalidated:
                logger.debug(f"  候选QRH-W {candidate.date} Px:{candidate.price:.2f} 已于 {invalidation_reason} 失效。")
                continue

            valid_qrhs.append(QualifiedRefHighInfoW(
                price=candidate.price, date=candidate.date, original_idx=candidate.original_idx,
                ma_at_high=candidate.ma_at_peak, is_ma_valid=True,
                is_recent_gain_valid=True, is_fully_qualified=True))
            gain_ratio_str = f"{gain_ratio_calculated:.2f}" if pd.notna(gain_ratio_calculated) and np.isfinite(
                gain_ratio_calculated) else "N/A"
            logger.debug(
                f"  候选QRH-W {candidate.date} Px:{candidate.price:.2f} (GainRatio:{gain_ratio_str}) 验证通过且至今未失效。")
        if not valid_qrhs:
            logger.info(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 未找到任何有效的、未失效的QRH-W。")
            return None

        highest_valid_qrh = max(valid_qrhs, key=lambda q: q.price)
        logger.info(
            f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 选择的最终有效QRH-W: {highest_valid_qrh.price:.2f} @ {highest_valid_qrh.date}")
        return highest_valid_qrh

    def run_for_stock(self, stock_code: str, current_eval_date: date, data: Dict[str, pd.DataFrame]) -> List[
        StrategyResult]:
        self._initialize_internal_states()
        results: List[StrategyResult] = []

        df_weekly_orig = data.get("weekly")
        df_monthly_orig = data.get("monthly")

        if df_weekly_orig is None or df_weekly_orig.empty: return results

        df_weekly_history = df_weekly_orig[df_weekly_orig['date'] <= current_eval_date].copy()

        min_len_req = max(self.params['weekly_ma_long'], self.params['weekly_ma_peak_qualify_period'],
                          self.params['weekly_peak_recent_gain_periods'],
                          self.params['weekly_fractal_definition_lookback'] * 2 + 1)
        if len(df_weekly_history) < min_len_req: return results

        df_weekly_history = self._calculate_weekly_mas(df_weekly_history)

        current_bar_w_rows = df_weekly_history[df_weekly_history['date'] == current_eval_date]
        if current_bar_w_rows.empty: return results
        current_bar_w = current_bar_w_rows.iloc[0]
        current_bar_w_iloc = df_weekly_history.index.get_loc(current_bar_w.name)

        # 月线数据处理
        df_monthly_history = pd.DataFrame()  # 初始化为空
        if df_monthly_orig is not None and not df_monthly_orig.empty:
            df_monthly_history = df_monthly_orig[df_monthly_orig['date'] <= current_eval_date].copy()
            if not df_monthly_history.empty and len(df_monthly_history) >= self.params['monthly_ma_filter_period']:
                df_monthly_history = self._calculate_monthly_mas(df_monthly_history)
            elif not df_monthly_history.empty:  # 月线数据有，但不足以计算MA
                df_monthly_history[self.ma_monthly_filter_col_m] = np.nan
                logger.info(
                    f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 月线历史数据不足以计算{self.params['monthly_ma_filter_period']}MA，月线过滤可能不准确。")
            # else: df_monthly_history 保持空
        if df_monthly_history.empty:  # 确保即使月线数据为空，列也存在以避免后续KeyError
            df_monthly_history = pd.DataFrame(columns=['date', self.ma_monthly_filter_col_m])

        current_bar_m = pd.Series(dtype='object')
        if not df_monthly_history.empty:
            relevant_monthly_bar_series = None
            for idx_m_hist in range(len(df_monthly_history) - 1, -1, -1):
                monthly_bar_iter = df_monthly_history.iloc[idx_m_hist]
                if monthly_bar_iter['date'].year == current_bar_w['date'].year and monthly_bar_iter['date'].month == \
                        current_bar_w['date'].month:
                    relevant_monthly_bar_series = monthly_bar_iter;
                    break
                if monthly_bar_iter['date'] < current_bar_w['date'].replace(day=1):
                    relevant_monthly_bar_series = monthly_bar_iter;
                    break
            if relevant_monthly_bar_series is None and not df_monthly_history.empty:
                relevant_monthly_bar_series = df_monthly_history.iloc[-1]
            if relevant_monthly_bar_series is not None: current_bar_m = relevant_monthly_bar_series

        chanlun_lookback_periods = self.params['weekly_chanlun_lookback_periods']
        start_iloc_for_chanlun = max(0, current_bar_w_iloc - chanlun_lookback_periods + 1)
        df_weekly_for_chanlun = df_weekly_history.iloc[start_iloc_for_chanlun: current_bar_w_iloc + 1].copy()
        if not df_weekly_for_chanlun.empty:
            self._run_chanlun_analysis_weekly(df_weekly_for_chanlun)
        else:
            self._strokes_w = []

        potential_peaks_w = self._get_potential_peak_candidates_weekly(current_bar_w, df_weekly_history)
        self._qualified_ref_high_w = self._get_highest_valid_qrh_weekly(
            potential_peaks_w, df_weekly_history, current_eval_date, stock_code)

        if self._qualified_ref_high_w:
            qrh_w = self._qualified_ref_high_w
            cw_close = current_bar_w['close'];
            cw_low = current_bar_w['low']
            ma30w_val = current_bar_w.get(self.ma_long_col_w);
            ma5w_val = current_bar_w.get(self.ma_short_col_w)
            ma30w_prev_val = df_weekly_history.iloc[current_bar_w_iloc - 1].get(
                self.ma_long_col_w) if current_bar_w_iloc > 0 else None
            ma30m_val = current_bar_m.get(self.ma_monthly_filter_col_m)
            ma30m_prev_val = None
            if not current_bar_m.empty and current_bar_m.name in df_monthly_history.index:
                current_bar_m_loc_monthly = df_monthly_history.index.get_loc(current_bar_m.name)
                if current_bar_m_loc_monthly > 0:
                    ma30m_prev_val = df_monthly_history.iloc[current_bar_m_loc_monthly - 1].get(
                        self.ma_monthly_filter_col_m)
            elif not df_monthly_history.empty and len(
                    df_monthly_history) > 1 and not current_bar_m.empty and current_bar_m.name == \
                    df_monthly_history.iloc[-1].name:
                ma30m_prev_val = df_monthly_history.iloc[-2].get(self.ma_monthly_filter_col_m)

            cond_w1 = cw_close < qrh_w.price
            cond_w2 = pd.notna(ma30w_val) and (
                        cw_low <= ma30w_val * self.params['weekly_pullback_ma_touch_upper_pct']) and \
                      (cw_low >= ma30w_val * self.params['weekly_pullback_ma_close_lower_pct'])
            cond_w3 = pd.notna(ma5w_val) and pd.notna(ma30w_val) and (ma5w_val > ma30w_val)
            cond_w4 = (pd.notna(ma30w_val) and pd.notna(ma30w_prev_val) and (
                        round(ma30w_val, 2) >= round(ma30w_prev_val, 2))) or \
                      (pd.notna(ma30w_val) and ma30w_prev_val is None and (current_bar_w_iloc + 1) == 1)

            cond_w5 = False  # Default to False
            if not current_bar_m.empty and pd.notna(ma30m_val) and pd.notna(cw_close):
                price_above_ma30m = cw_close > ma30m_val
                ma30m_rising_flat = True  # Assume true if prev is not available
                if pd.notna(ma30m_prev_val): ma30m_rising_flat = (round(ma30m_val, 2) >= round(ma30m_prev_val, 2))
                cond_w5 = price_above_ma30m and ma30m_rising_flat
            elif current_bar_m.empty:
                logger.debug(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 月线数据为空，W5条件不满足。")
            elif not pd.notna(ma30m_val):
                logger.debug(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 月线MA30为空，W5条件不满足。")

            ma30w_str = f"{ma30w_val:.2f}" if pd.notna(ma30w_val) else "N/A"
            ma30m_str = f"{ma30m_val:.2f}" if pd.notna(ma30m_val) else "N/A"
            logger.debug(
                f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 周线信号条件评估 (QRH: {qrh_w.price:.2f}@{qrh_w.date}): "
                f"W1:{cond_w1}, W2:{cond_w2} (Low:{cw_low:.2f}, MA30W:{ma30w_str}), "
                f"W3:{cond_w3}, W4:{cond_w4}, W5:{cond_w5} (W.Close:{cw_close:.2f}, MA30M:{ma30m_str})")

            if cond_w1 and cond_w2 and cond_w3 and cond_w4 and cond_w5:
                details = {
                    "qrh_w_price": f"{qrh_w.price:.2f}", "qrh_w_date": str(qrh_w.date),
                    "weekly_close": f"{cw_close:.2f}", "weekly_low": f"{cw_low:.2f}",
                    self.ma_long_col_w: ma30w_str,
                    self.ma_short_col_w: f"{ma5w_val:.2f}" if pd.notna(ma5w_val) else "N/A",
                    self.ma_monthly_filter_col_m: ma30m_str,
                    "conditions_met": {"W1": cond_w1, "W2": cond_w2, "W3": cond_w3, "W4": cond_w4, "W5": cond_w5}
                }
                results.append(StrategyResult(stock_code, current_eval_date, self.strategy_name, details=details,
                                              timeframe="weekly"))
                logger.info(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 周线级别买入信号产生！")
        else:
            logger.debug(f"[{self.strategy_name}@{stock_code}@{current_eval_date}] 无有效QRH-W，不产生信号。")
        return results


# --- Main Test Function (与上一版相似，确保导入和参数) ---
if __name__ == '__main__':
    import sys;
    import os;
    import pandas as pd;
    from datetime import datetime, date
    from sqlalchemy.orm import Session;
    from sqlalchemy import func

    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_package_root = os.path.dirname(current_file_dir)
        project_base_dir = os.path.dirname(project_package_root)
        if project_base_dir not in sys.path: sys.path.insert(0, project_base_dir)
    except NameError:
        pass

    from quant_platform.db.database import SessionLocal
    from quant_platform.utils import data_loader as dl_db
    from quant_platform.db.models import StockDaily

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_logger = logging.getLogger(__name__ + "_main_test")

    temp_context_for_name = StrategyContext()
    strategy_name_for_log = WeeklyMAPullbackStrategy(temp_context_for_name).strategy_name
    main_logger.info(f"开始 {strategy_name_for_log} 全周期买入信号测试...")

    stocks_to_test = ["603920", "000887"]  # 可以测试多只股票
    db_session: Optional[Session] = None
    try:
        db_session = SessionLocal()
        all_signals_generated_overall: List[StrategyResult] = []

        for stock_to_test in stocks_to_test:
            main_logger.info(f"\n\n========== 开始测试股票: {stock_to_test} ==========")
            first_daily_date_obj = db_session.query(func.min(StockDaily.date)).filter(
                StockDaily.symbol == stock_to_test).scalar()
            last_daily_date_obj = db_session.query(func.max(StockDaily.date)).filter(
                StockDaily.symbol == stock_to_test).scalar()

            if not first_daily_date_obj or not last_daily_date_obj:
                main_logger.error(f"数据库中未找到股票 {stock_to_test} 的日线数据范围。");
                continue

            start_date_load = first_daily_date_obj.strftime('%Y-%m-%d')
            end_date_load = last_daily_date_obj.strftime('%Y-%m-%d')
            main_logger.info(
                f"从数据库为 {stock_to_test} 加载周线和月线数据 (范围: {start_date_load} to {end_date_load})")

            weekly_df_stock = dl_db.load_weekly_data(symbol=stock_to_test, start_date=start_date_load,
                                                     end_date=end_date_load, db_session=db_session)
            monthly_df_stock = dl_db.load_monthly_data(symbol=stock_to_test, start_date=start_date_load,
                                                       end_date=end_date_load, db_session=db_session)

            if weekly_df_stock is None or weekly_df_stock.empty: main_logger.error(
                f"未能加载 {stock_to_test} 周线数据。"); continue

            weekly_df_stock['date'] = pd.to_datetime(weekly_df_stock['date']).dt.date
            weekly_df_stock.sort_values(by='date', inplace=True);
            weekly_df_stock.reset_index(drop=True, inplace=True)
            main_logger.info(f"加载并预处理了 {len(weekly_df_stock)} 条周线数据。")

            if monthly_df_stock is None: monthly_df_stock = pd.DataFrame(columns=['date'])  # 空DF
            monthly_df_stock['date'] = pd.to_datetime(monthly_df_stock['date']).dt.date
            monthly_df_stock.sort_values(by='date', inplace=True);
            monthly_df_stock.reset_index(drop=True, inplace=True)
            main_logger.info(f"加载并预处理了 {len(monthly_df_stock)} 条月线数据。")

            test_eval_dates = weekly_df_stock['date'].tolist()
            if not test_eval_dates: main_logger.error(f"{stock_to_test} 无周线评估日期。"); continue

            main_logger.info(f"将对 {len(test_eval_dates)} 个周线结束日进行策略评估...")
            context = StrategyContext(db_session=db_session, strategy_params={})
            strategy = WeeklyMAPullbackStrategy(context=context)
            main_logger.info(f"[{stock_to_test}] 使用策略参数: {strategy.params}")
            stock_signals: List[StrategyResult] = []
            for eval_date in test_eval_dates:
                data_slice_for_run = {
                    "weekly": weekly_df_stock[weekly_df_stock['date'] <= eval_date].copy(),
                    "monthly": monthly_df_stock[monthly_df_stock['date'] <= eval_date].copy()
                }
                if data_slice_for_run["weekly"].empty: continue
                signals = strategy.run_for_stock(stock_to_test, eval_date, data_slice_for_run)
                if signals:
                    for sig in signals:
                        main_logger.info(
                            f"[{stock_to_test}] 买入信号 @ {sig.signal_date.isoformat()} (策略名: {sig.strategy_name}, 时间级别: {sig.timeframe})\n详细信息: {sig.details}")
                        stock_signals.append(sig)
            if not stock_signals:
                main_logger.info(f"测试周期内，股票 {stock_to_test} 未产生任何周线信号。")
            else:
                main_logger.info(f"\n--- 为股票 {stock_to_test} 生成的全部周线信号 ({len(stock_signals)}条) ---")
                for signal_idx, signal in enumerate(stock_signals): print(
                    f"股票 {stock_to_test} - 信号 {signal_idx + 1}: {signal}")
            all_signals_generated_overall.extend(stock_signals)

        if not all_signals_generated_overall:
            main_logger.info("所有测试股票均未产生任何周线信号。")
        else:
            main_logger.info(f"\n========= 所有股票的周线信号汇总 ({len(all_signals_generated_overall)}条) =========")
            for signal in all_signals_generated_overall: print(signal)
    except ImportError as e:
        main_logger.error(f"导入模块失败: {e}.")
    except Exception as e:
        main_logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    finally:
        if db_session: db_session.close(); main_logger.info("数据库会话已关闭。")
    main_logger.info(f"{strategy_name_for_log} 全周期买入信号测试结束。")