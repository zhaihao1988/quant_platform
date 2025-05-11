# strategies/backtrader_ma_pullback_strategy.py
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from collections import deque, namedtuple

try:
    import talib

    TALIB_AVAILABLE = True  # Still useful for ATR if preferred, though strategy has fallback
except ImportError:
    TALIB_AVAILABLE = False
    print("WARN: TA-Lib not found. ATR will use Backtrader's approximation if TA-Lib version was intended.")

# --- Data Structures ---
# KLineRaw for raw data processing
KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx'])
# MergedKLine from simplified Chanlun-like processing
MergedKLine = namedtuple('MergedKLine', ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx'])
# Fractal based on MergedKLine
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])  # 类型: 1 for top, -1 for bottom
# Stroke connecting two Fractals
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction'])  # direction: 1 for up, -1 for down


class MAPullbackPeakCondBtStrategy(bt.Strategy):
    params = (
        ('ma_short', 5),  # 短期均线周期
        ('ma_long', 30),  # 长期均线周期 (日线级别买入参考)
        ('ma_long_for_peak', 30),  # 用于验证候选高点有效性的MA周期
        ('ma_peak_threshold', 1.30),  # 候选高点必须高于其MA值的倍数 (MA * 1.30)
        ('peak_recent_gain_days', 30),  # 候选高点近期涨幅的回溯天数
        ('peak_recent_gain_ratio', 1.2),  # 近期涨幅的最低比例 (最高价/最低价 > 1.2)
        ('downstroke_invalidate_threshold', 0.05),  # 向下笔低点显著跌破MA30的幅度 (MA30 * (1-0.05))
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('sell_ma_pct', 0.05),  # General MA stop-loss percentage
        ('printlog', True),
        ('debug_stock', None),  # 指定要调试的股票代码
        ('debug_date', None),  # 指定要开始详细调试的日期 'YYYY-MM-DD'
        ('debug_date_is_start_date', False),  # debug_date是起始还是精确匹配
        ('max_position_ratio', 0.20),  # 最大仓位比例
        ('weekly_ma_period', 30),  # 周线MA周期
        ('monthly_ma_period', 30),  # 月线MA周期 (未使用，但保留结构)
        # Parameters for simplified stroke/fractal identification
        ('fractal_lookback', 120),  # _find_prior_peak_stroke_info的回溯K线数
        ('min_bars_between_fractals', 2),  # 形成笔的两个分型之间至少需要的合并后K线条数
        ('use_dynamic_stop_loss_level', True)  # 是否使用动态止损级别
    )

    def log(self, txt, dt=None, doprint=False, data_feed=None):
        is_debug_target = False
        log_dt_obj = dt or self.datas[0].datetime.date(0)
        current_bar_date_str = log_dt_obj.strftime('%Y-%m-%d')

        if data_feed and self.p.debug_stock and data_feed._name == self.p.debug_stock:
            if self.p.debug_date:
                if self.p.debug_date_is_start_date:
                    if current_bar_date_str >= self.p.debug_date:
                        is_debug_target = True
                else:
                    if current_bar_date_str == self.p.debug_date:
                        is_debug_target = True
            # else: # If no debug_date is set, but debug_stock is, log all for that stock
            #     is_debug_target = True # Commented out: only log if date also matches or if no date set

        if self.params.printlog or doprint or is_debug_target:
            data_name = data_feed._name if data_feed else 'Strategy'
            print(f'{log_dt_obj.isoformat()} [{data_name}] {txt}')

    def __init__(self):
        self.inds = dict()
        self.positions_info = dict()  # For managing active trade details (SL, TP etc.)
        self.atr_stop_loss_prices = dict()  # Specific for ATR trailing stop
        self.highest_highs_since_entry = dict()  # For ATR trailing stop
        self._pending_buy_info = dict()  # Temporary store for buy order details before execution

        self.strategy_states = dict()

        self.daily_bars_agg = {d._name: deque(maxlen=self.p.monthly_ma_period * 5 + 10) for d in
                               self.datas}
        self.synthesized_weekly_data_agg = {d._name: deque(maxlen=self.p.weekly_ma_period + 10) for d in
                                            self.datas}
        self.current_week_data_agg = {d._name: [] for d in self.datas}
        self.last_week_num = {d._name: -1 for d in self.datas}
        self.weekly_mas = {d._name: None for d in self.datas}
        self.last_completed_weekly_ma = {d._name: None for d in self.datas}

        self.synthesized_monthly_data_agg = {d._name: deque(maxlen=self.p.monthly_ma_period + 10) for d in self.datas}
        self.current_month_data_agg = {d._name: [] for d in self.datas}
        self.last_month_num = {d._name: -1 for d in self.datas}
        self.monthly_mas = {d._name: None for d in self.datas}
        self.last_completed_monthly_ma = {d._name: None for d in self.datas}

        for d in self.datas:
            stock_name = d._name
            self.inds[stock_name] = {}
            self.inds[stock_name]['ma_short'] = btind.SimpleMovingAverage(d.close, period=self.params.ma_short)
            self.inds[stock_name]['ma_long'] = btind.SimpleMovingAverage(d.close,
                                                                         period=self.params.ma_long)
            self.inds[stock_name]['ma_long_for_peak'] = btind.SimpleMovingAverage(d.close,
                                                                                  period=self.params.ma_long_for_peak)

            if TALIB_AVAILABLE:
                self.inds[stock_name]['atr'] = btind.ATR(d, period=self.params.atr_period, plot=False)
            else:
                self.inds[stock_name]['atr'] = btind.AverageTrueRange(d, period=self.params.atr_period, plot=False)

            self.positions_info[stock_name] = None
            self.atr_stop_loss_prices[stock_name] = float('nan')
            self.highest_highs_since_entry[stock_name] = float('-inf')

            self.strategy_states[stock_name] = {
                'active_uptrend_peak_candidate': {
                    'price': float('-inf'), 'date': None, 'original_idx': -1,
                    'ma30_at_peak': float('nan'), 'is_ma_valid': False
                },
                'qualified_ref_high_info': {
                    'price': float('-inf'), 'date': None, 'original_idx': -1,
                    'ma30_at_high': float('nan'), 'is_ma_valid': False,
                    'is_recent_gain_valid': False, 'is_fully_qualified': False
                },
                'last_downstroke_info': {
                    'end_date': None, 'low_price': float('inf'), 'low_idx': -1,
                    'ma30_at_low': float('nan'), 'is_significant_break': False
                }
            }
        self.order = None

    def _merge_klines_chanlun(self, bars_data: list[KLineRaw]) -> list[MergedKLine]:
        if not bars_data or len(bars_data) < 1:
            if bars_data:
                k_raw = bars_data[0]
                return [MergedKLine(dt=k_raw.dt, o=k_raw.o, h=k_raw.h, l=k_raw.l, c=k_raw.c,
                                    idx=k_raw.idx, direction=0,
                                    high_idx=k_raw.idx, low_idx=k_raw.idx)]
            return []

        merged_lines: list[MergedKLine] = []
        k1_raw = bars_data[0]
        k1 = MergedKLine(dt=k1_raw.dt, o=k1_raw.o, h=k1_raw.h, l=k1_raw.l, c=k1_raw.c,
                         idx=k1_raw.idx, direction=0,
                         high_idx=k1_raw.idx, low_idx=k1_raw.idx)
        merged_lines.append(k1)
        current_segment_trend = 0

        for i in range(1, len(bars_data)):
            k1_merged = merged_lines[-1]
            k2_raw = bars_data[i]

            k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l)
            k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)

            if k1_includes_k2 or k2_includes_k1:
                trend_for_inclusion = current_segment_trend
                m_h, m_l = k1_merged.h, k1_merged.l
                m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx

                if trend_for_inclusion == 1:
                    m_h = max(k1_merged.h, k2_raw.h)
                    m_l = k1_merged.l
                    m_high_idx = k2_raw.idx if k2_raw.h >= k1_merged.h else k1_merged.high_idx
                    m_low_idx = k1_merged.low_idx
                elif trend_for_inclusion == -1:
                    m_h = k1_merged.h
                    m_l = min(k1_merged.l, k2_raw.l)
                    m_high_idx = k1_merged.high_idx
                    m_low_idx = k2_raw.idx if k2_raw.l <= k1_merged.l else k1_merged.low_idx
                else:
                    if k1_includes_k2:
                        m_h, m_l = k1_merged.h, k1_merged.l
                        m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
                    elif k2_includes_k1:
                        m_h, m_l = k2_raw.h, k2_raw.l
                        m_high_idx, m_low_idx = k2_raw.idx, k2_raw.idx
                    if k1_merged.h == k2_raw.h and k1_merged.l == k2_raw.l:
                        m_h, m_l = k1_merged.h, k1_merged.l
                        m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx

                merged_lines[-1] = MergedKLine(dt=k2_raw.dt, o=k1_merged.o, h=m_h, l=m_l, c=k2_raw.c,
                                               idx=k2_raw.idx,
                                               direction=k1_merged.direction,
                                               high_idx=m_high_idx, low_idx=m_low_idx)
            else:
                new_segment_direction = 0
                if k2_raw.h > k1_merged.h and k2_raw.l > k1_merged.l:
                    new_segment_direction = 1
                elif k2_raw.h < k1_merged.h and k2_raw.l < k1_merged.l:
                    new_segment_direction = -1

                if merged_lines[-1].direction == 0 and len(merged_lines) > 1:
                    k_prev_prev = merged_lines[-2]
                    k_prev = merged_lines[-1]
                    prev_dir = 0
                    if k_prev.h > k_prev_prev.h and k_prev.l > k_prev_prev.l:
                        prev_dir = 1
                    elif k_prev.h < k_prev_prev.h and k_prev.l < k_prev_prev.l:
                        prev_dir = -1
                    merged_lines[-1] = merged_lines[-1]._replace(direction=prev_dir)

                current_segment_trend = new_segment_direction

                k2_new_merged = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c,
                                            idx=k2_raw.idx, direction=new_segment_direction,
                                            high_idx=k2_raw.idx, low_idx=k2_raw.idx)
                merged_lines.append(k2_new_merged)

        for i in range(len(merged_lines)):
            if merged_lines[i].direction == 0:
                if i > 0:
                    if merged_lines[i].h > merged_lines[i - 1].h and merged_lines[i].l > merged_lines[i - 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=1)
                    elif merged_lines[i].h < merged_lines[i - 1].h and merged_lines[i].l < merged_lines[i - 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=-1)
                elif i < len(merged_lines) - 1:
                    if merged_lines[i].h < merged_lines[i + 1].h and merged_lines[i].l < merged_lines[
                        i + 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=1)
                    elif merged_lines[i].h > merged_lines[i + 1].h and merged_lines[i].l > merged_lines[
                        i + 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=-1)
        return merged_lines

    def _find_merged_fractal(self, merged_klines: list[MergedKLine], index: int) -> Fractal | None:
        if index < 1 or index >= len(merged_klines) - 1:
            return None

        k_prev = merged_klines[index - 1]
        k_curr = merged_klines[index]
        k_next = merged_klines[index + 1]

        is_top = k_curr.h > k_prev.h and k_curr.h > k_next.h
        is_bottom = k_curr.l < k_prev.l and k_curr.l < k_next.l

        if is_top and is_bottom:
            return None
        if is_top:
            return Fractal(kline=k_curr, m_idx=index, type=1)
        if is_bottom:
            return Fractal(kline=k_curr, m_idx=index, type=-1)
        return None

    def _identify_all_merged_fractals(self, merged_klines: list[MergedKLine]) -> list[Fractal]:
        fractals = []
        if len(merged_klines) < 3:
            return fractals
        for i in range(1, len(merged_klines) - 1):
            fractal = self._find_merged_fractal(merged_klines, i)
            if fractal:
                fractals.append(fractal)
        return fractals

    def _identify_strokes(self, fractals: list[Fractal], merged_klines: list[MergedKLine]) -> list[Stroke]:
        strokes = []
        if len(fractals) < 2:
            return strokes

        last_confirmed_fractal = fractals[0]

        for i in range(1, len(fractals)):
            current_fractal = fractals[i]

            if current_fractal.type == last_confirmed_fractal.type:
                if current_fractal.type == 1 and current_fractal.kline.h > last_confirmed_fractal.kline.h:
                    last_confirmed_fractal = current_fractal
                elif current_fractal.type == -1 and current_fractal.kline.l < last_confirmed_fractal.kline.l:
                    last_confirmed_fractal = current_fractal
                continue

            bars_between_merged = abs(current_fractal.m_idx - last_confirmed_fractal.m_idx) - 1
            if bars_between_merged < self.params.min_bars_between_fractals:
                continue

            stroke_direction = 0
            if last_confirmed_fractal.type == -1 and current_fractal.type == 1:
                if current_fractal.kline.h > last_confirmed_fractal.kline.l:
                    stroke_direction = 1
            elif last_confirmed_fractal.type == 1 and current_fractal.type == -1:
                if current_fractal.kline.l < last_confirmed_fractal.kline.h:
                    stroke_direction = -1

            if stroke_direction != 0:
                strokes.append(Stroke(start_fractal=last_confirmed_fractal,
                                      end_fractal=current_fractal,
                                      direction=stroke_direction))
                last_confirmed_fractal = current_fractal
        return strokes

    def _get_raw_klines_for_period(self, data_feed, lookback_bars) -> list[KLineRaw]:
        raw_klines = []
        num_available_bars = len(data_feed.close)
        actual_lookback = min(num_available_bars - 1, lookback_bars)

        if actual_lookback < 0: return []

        for k_ago in range(actual_lookback, -1, -1):
            original_idx = (num_available_bars - 1) - k_ago
            if original_idx < 0: continue

            dt = bt.num2date(data_feed.datetime[-k_ago]).date()
            o = data_feed.open[-k_ago]
            h = data_feed.high[-k_ago]
            l = data_feed.low[-k_ago]
            c = data_feed.close[-k_ago]
            if any(pd.isna(x) for x in [o, h, l, c]):
                continue
            raw_klines.append(KLineRaw(dt=dt, o=o, h=h, l=l, c=c, idx=original_idx))
        return raw_klines

    def _update_active_peak_candidate(self, d):
        stock_name = d._name
        state = self.strategy_states[stock_name]

        raw_klines = self._get_raw_klines_for_period(d, self.p.fractal_lookback)
        if not raw_klines: return

        merged_klines = self._merge_klines_chanlun(raw_klines)
        if not merged_klines or len(merged_klines) < self.p.min_bars_between_fractals + 2:
            return

        all_fractals = self._identify_all_merged_fractals(merged_klines)
        strokes = self._identify_strokes(all_fractals, merged_klines)

        current_candidate_price = state['active_uptrend_peak_candidate']['price']
        found_new_candidate = False  # Not used actively after this line

        for stroke in reversed(strokes):
            if stroke.direction == 1:
                peak_fractal = stroke.end_fractal
                merged_k_peak = peak_fractal.kline
                original_peak_bar_idx = merged_k_peak.high_idx
                peak_price = merged_k_peak.h

                current_original_idx = len(d.close) - 1
                ago = current_original_idx - original_peak_bar_idx

                if ago < 0 or ago >= len(self.inds[stock_name]['ma_long_for_peak']):
                    continue

                ma_at_peak_time = self.inds[stock_name]['ma_long_for_peak'][-ago]
                if pd.isna(ma_at_peak_time): continue

                is_ma_valid_for_this_peak = peak_price > ma_at_peak_time * self.p.ma_peak_threshold

                if is_ma_valid_for_this_peak:
                    if peak_price > current_candidate_price:
                        state['active_uptrend_peak_candidate'] = {
                            'price': peak_price,
                            'date': merged_k_peak.dt,
                            'original_idx': original_peak_bar_idx,
                            'ma30_at_peak': ma_at_peak_time,
                            'is_ma_valid': True
                        }
                        current_candidate_price = peak_price
                        # found_new_candidate = True # This variable is not used for control flow after this
                        state['qualified_ref_high_info'] = {
                            'price': float('-inf'), 'date': None, 'original_idx': -1,
                            'ma30_at_high': float('nan'), 'is_ma_valid': False,
                            'is_recent_gain_valid': False, 'is_fully_qualified': False
                        }

        current_bar_high = d.high[0]
        ma30_current_bar = self.inds[stock_name]['ma_long_for_peak'][0]
        if not pd.isna(ma30_current_bar) and current_bar_high > ma30_current_bar * self.p.ma_peak_threshold:
            if current_bar_high > current_candidate_price:
                state['active_uptrend_peak_candidate'] = {
                    'price': current_bar_high,
                    'date': d.datetime.date(0),
                    'original_idx': len(d.close) - 1,
                    'ma30_at_peak': ma30_current_bar,
                    'is_ma_valid': True
                }
                state['qualified_ref_high_info'] = {
                    'price': float('-inf'), 'date': None, 'original_idx': -1,
                    'ma30_at_high': float('nan'), 'is_ma_valid': False,
                    'is_recent_gain_valid': False, 'is_fully_qualified': False
                }

    def _check_uptrend_invalidation(self, d):
        stock_name = d._name
        state = self.strategy_states[stock_name]

        raw_klines = self._get_raw_klines_for_period(d, self.p.fractal_lookback)
        if not raw_klines: return False

        merged_klines = self._merge_klines_chanlun(raw_klines)
        if not merged_klines or len(merged_klines) < 3: return False

        all_fractals = self._identify_all_merged_fractals(merged_klines)
        strokes = self._identify_strokes(all_fractals, merged_klines)

        last_down_stroke = None
        for stroke in reversed(strokes):
            if stroke.direction == -1:
                last_down_stroke = stroke
                break

        if last_down_stroke:
            bottom_fractal = last_down_stroke.end_fractal
            merged_k_low = bottom_fractal.kline
            original_low_bar_idx = merged_k_low.low_idx
            low_price = merged_k_low.l

            current_original_idx = len(d.close) - 1
            ago = current_original_idx - original_low_bar_idx

            if ago >= 0 and ago < len(self.inds[stock_name]['ma_long_for_peak']):
                ma30_at_low_time = self.inds[stock_name]['ma_long_for_peak'][-ago]
                if not pd.isna(ma30_at_low_time):
                    state['last_downstroke_info'] = {
                        'end_date': merged_k_low.dt,
                        'low_price': low_price,
                        'low_idx': original_low_bar_idx,
                        'ma30_at_low': ma30_at_low_time,
                        'is_significant_break': False
                    }
                    if low_price < ma30_at_low_time * (1 - self.p.downstroke_invalidate_threshold):
                        state['last_downstroke_info']['is_significant_break'] = True
                        state['active_uptrend_peak_candidate'] = {
                            'price': float('-inf'), 'date': None, 'original_idx': -1,
                            'ma30_at_peak': float('nan'), 'is_ma_valid': False
                        }
                        state['qualified_ref_high_info'] = {
                            'price': float('-inf'), 'date': None, 'original_idx': -1,
                            'ma30_at_high': float('nan'), 'is_ma_valid': False,
                            'is_recent_gain_valid': False, 'is_fully_qualified': False
                        }
                        return True
        return False

    def _validate_and_set_qualified_ref_high(self, d):
        stock_name = d._name
        state = self.strategy_states[stock_name]
        candidate = state['active_uptrend_peak_candidate']

        if not candidate or candidate['price'] <= float('-inf') or candidate['original_idx'] < 0:
            state['qualified_ref_high_info']['is_fully_qualified'] = False
            return False

        peak_original_idx = candidate['original_idx']
        peak_price = candidate['price']
        lookback_days = self.p.peak_recent_gain_days
        start_idx_for_gain_calc = max(0, peak_original_idx - lookback_days + 1)
        end_idx_for_gain_calc = peak_original_idx

        if start_idx_for_gain_calc > end_idx_for_gain_calc:
            state['qualified_ref_high_info']['is_recent_gain_valid'] = False
            state['qualified_ref_high_info']['is_fully_qualified'] = False
            return False

        highest_in_window = float('-inf')
        lowest_in_window = float('inf')
        current_len_data = len(d.close)

        for idx in range(start_idx_for_gain_calc, end_idx_for_gain_calc + 1):
            ago = (current_len_data - 1) - idx
            if ago < 0 or ago >= current_len_data:
                continue
            current_h = d.high[-ago]
            current_l = d.low[-ago]
            if pd.isna(current_h) or pd.isna(current_l): continue
            highest_in_window = max(highest_in_window, current_h)
            lowest_in_window = min(lowest_in_window, current_l)

        is_recent_gain_valid = False
        if lowest_in_window > 0 and highest_in_window > float('-inf'):  # Avoid division by zero or with uninit values
            gain_ratio = highest_in_window / lowest_in_window
            if gain_ratio > self.p.peak_recent_gain_ratio:
                is_recent_gain_valid = True

        if is_recent_gain_valid:
            state['qualified_ref_high_info'] = {
                'price': candidate['price'],
                'date': candidate['date'],
                'original_idx': candidate['original_idx'],
                'ma30_at_high': candidate['ma30_at_peak'],
                'is_ma_valid': True,
                'is_recent_gain_valid': True,
                'is_fully_qualified': True
            }
            return True
        else:
            state['qualified_ref_high_info']['is_fully_qualified'] = False
            return False

    def _synthesize_higher_tf_data(self, data_feed):
        d_name = data_feed._name
        current_date = data_feed.datetime.date(0)
        current_bar_data = {'dt': current_date, 'o': data_feed.open[0], 'h': data_feed.high[0],
                            'l': data_feed.low[0], 'c': data_feed.close[0], 'vol': data_feed.volume[0]}

        self.daily_bars_agg[d_name].append(current_bar_data)

        current_week_num = current_date.isocalendar()[1]
        if self.last_week_num[d_name] == -1:
            self.last_week_num[d_name] = current_week_num

        if current_week_num != self.last_week_num[d_name] and self.current_week_data_agg[d_name]:
            prev_week_bars = self.current_week_data_agg[d_name]
            if prev_week_bars:
                self.synthesized_weekly_data_agg[d_name].append(
                    {'dt': prev_week_bars[-1]['dt'], 'c': prev_week_bars[-1]['c']})
            self.current_week_data_agg[d_name] = []
            self.last_week_num[d_name] = current_week_num
            if len(self.synthesized_weekly_data_agg[d_name]) >= self.p.weekly_ma_period:
                completed_week_closes = [b['c'] for b in
                                         list(self.synthesized_weekly_data_agg[d_name])]
                if len(completed_week_closes) >= self.p.weekly_ma_period:
                    self.last_completed_weekly_ma[d_name] = sum(
                        completed_week_closes[-self.p.weekly_ma_period:]) / self.p.weekly_ma_period

        self.current_week_data_agg[d_name].append(current_bar_data)
        temp_weekly_closes_for_ma = [b['c'] for b in self.synthesized_weekly_data_agg[d_name]]
        if self.current_week_data_agg[d_name]:
            temp_weekly_closes_for_ma.append(
                self.current_week_data_agg[d_name][-1]['c'])

        if len(temp_weekly_closes_for_ma) >= self.p.weekly_ma_period:
            self.weekly_mas[d_name] = sum(
                temp_weekly_closes_for_ma[-self.p.weekly_ma_period:]) / self.p.weekly_ma_period
        else:
            self.weekly_mas[d_name] = None

        current_month_num = current_date.month
        if self.last_month_num[d_name] == -1: self.last_month_num[d_name] = current_month_num
        if current_month_num != self.last_month_num[d_name] and self.current_month_data_agg[d_name]:
            prev_month_bars = self.current_month_data_agg[d_name]
            if prev_month_bars: self.synthesized_monthly_data_agg[d_name].append(
                {'dt': prev_month_bars[-1]['dt'], 'c': prev_month_bars[-1]['c']})
            self.current_month_data_agg[d_name] = []
            self.last_month_num[d_name] = current_month_num
            if len(self.synthesized_monthly_data_agg[d_name]) >= self.p.monthly_ma_period:
                completed_month_closes = [b['c'] for b in list(self.synthesized_monthly_data_agg[d_name])]
                if len(completed_month_closes) >= self.p.monthly_ma_period:
                    self.last_completed_monthly_ma[d_name] = sum(
                        completed_month_closes[-self.p.monthly_ma_period:]) / self.p.monthly_ma_period
        self.current_month_data_agg[d_name].append(current_bar_data)
        temp_monthly_closes_for_ma = [b['c'] for b in self.synthesized_monthly_data_agg[d_name]]
        if self.current_month_data_agg[d_name]: temp_monthly_closes_for_ma.append(
            self.current_month_data_agg[d_name][-1]['c'])
        if len(temp_monthly_closes_for_ma) >= self.p.monthly_ma_period:
            self.monthly_mas[d_name] = sum(
                temp_monthly_closes_for_ma[-self.p.monthly_ma_period:]) / self.p.monthly_ma_period
        else:
            self.monthly_mas[d_name] = None

    def notify_order(self, order):
        stock_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}',
                    data_feed=order.data, doprint=True)
                pending_info = self._pending_buy_info.get(stock_name, {})
                prior_peak_high_price = pending_info.get("price", float('nan'))
                prior_peak_date = pending_info.get("date", "N/A")

                current_pos_size = self.getposition(order.data).size
                current_pos_avg_price = self.getposition(order.data).price
                entry_trend_level = pending_info.get("entry_trend_level", "daily_new_logic")

                self.positions_info[stock_name] = {
                    'entry_price': current_pos_avg_price,
                    'initial_shares': current_pos_size,
                    'shares_left': current_pos_size,
                    'prior_peak_high_for_tp': prior_peak_high_price,
                    'prior_peak_date_for_tp': prior_peak_date,
                    'entry_bar_idx': len(order.data.close) - 1,
                    'entry_date': bt.num2date(order.data.datetime[0]).date(),
                    'entry_trend_level': entry_trend_level,
                    'tp1_price': float('nan'), 'tp1_hit': False, 'tp1_sold_shares': 0,
                    'tp2_price': prior_peak_high_price, 'tp2_hit': False, 'tp2_sold_shares': 0,
                    'tp3_price': float('nan'), 'tp3_hit': False, 'tp3_sold_shares': 0,
                    'post_entry_bottom_fractal': None,
                    'post_entry_bottom_fractal_found': False,
                    'sell_reason': None
                }
                self.highest_highs_since_entry[stock_name] = order.data.high[0]

                atr_val = self.inds[stock_name]['atr'][0] if self.inds[stock_name]['atr'] is not None and not pd.isna(
                    self.inds[stock_name]['atr'][0]) else 0
                if atr_val > 0:
                    self.atr_stop_loss_prices[stock_name] = order.executed.price - self.params.atr_multiplier * atr_val
                else:
                    self.atr_stop_loss_prices[stock_name] = order.executed.price * (1 - 0.05)

                if stock_name in self._pending_buy_info:
                    del self._pending_buy_info[stock_name]

            elif order.issell():
                p_info = self.positions_info.get(stock_name)
                reason = "N/A"
                if p_info and p_info.get('sell_reason'):
                    reason = p_info.get('sell_reason')
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}. Reason: {reason}',
                    data_feed=order.data, doprint=True)
                if p_info:
                    p_info['shares_left'] -= abs(order.executed.size)

            if self.order == order:
                self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order Canceled/Margin/Rejected/Expired - Status: {order.getstatusname()}', data_feed=order.data,
                     doprint=True)
            if order.isbuy() and stock_name in self._pending_buy_info:
                del self._pending_buy_info[stock_name]
            if self.order == order:
                self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        stock_name = trade.data._name
        self.log(f'TRADE PROFIT ({stock_name}), GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}', data_feed=trade.data,
                 doprint=True)

        self.positions_info[stock_name] = None
        self.atr_stop_loss_prices[stock_name] = float('nan')
        self.highest_highs_since_entry[stock_name] = float('-inf')
        if stock_name in self._pending_buy_info:
            del self._pending_buy_info[stock_name]

    def next(self):
        current_date_obj = self.datas[0].datetime.date(0)
        current_date_str = current_date_obj.strftime('%Y-%m-%d')

        for i, d in enumerate(self.datas):
            stock_name = d._name
            pos = self.getposition(d)
            state = self.strategy_states[stock_name]

            is_debug_target_stock_date = False
            if self.p.debug_stock == stock_name:
                if self.p.debug_date:
                    if self.p.debug_date_is_start_date:
                        if current_date_str >= self.p.debug_date: is_debug_target_stock_date = True
                    else:
                        if current_date_str == self.p.debug_date: is_debug_target_stock_date = True

            if is_debug_target_stock_date:
                self.log(
                    f"DEBUG StartBar: {stock_name} on {current_date_str}. PosSize: {pos.size}. ActivePeak: {state['active_uptrend_peak_candidate']['price']:.2f} QualifiedPeak: {state['qualified_ref_high_info']['price']:.2f}",
                    data_feed=d, doprint=True)

            min_bars_required = max(self.p.fractal_lookback,
                                    self.p.peak_recent_gain_days,
                                    self.p.ma_long) + 20
            if len(d.close) < min_bars_required:
                if is_debug_target_stock_date: self.log(
                    f"DEBUG DataLen: Not enough data for {stock_name}. Has {len(d.close)}, needs {min_bars_required}",
                    data_feed=d, doprint=True)
                continue

            self._synthesize_higher_tf_data(d)

            if self.order and self.order.data == d:
                if is_debug_target_stock_date: self.log(f"DEBUG OrderPend: Order pending for {stock_name}, skipping.",
                                                        data_feed=d, doprint=True)
                continue

            invalidation_occurred = self._check_uptrend_invalidation(d)
            if invalidation_occurred and is_debug_target_stock_date:
                self.log(f"DEBUG Invalidate: Uptrend context invalidated for {stock_name}.", data_feed=d, doprint=True)

            if not invalidation_occurred:
                self._update_active_peak_candidate(d)

            if is_debug_target_stock_date:
                self.log(
                    f"DEBUG PeaksAfterUpdate: ActivePeak: {state['active_uptrend_peak_candidate']['price']:.2f} on {state['active_uptrend_peak_candidate']['date']}. QualifiedPeak: {state['qualified_ref_high_info']['price']:.2f} ({state['qualified_ref_high_info']['is_fully_qualified']})",
                    data_feed=d, doprint=True)

            if pos.size > 0 and self.positions_info.get(stock_name):
                p_info = self.positions_info[stock_name]
                sell_signal_triggered = False;
                sell_reason = None

                if not p_info['post_entry_bottom_fractal_found']:
                    entry_idx = p_info['entry_bar_idx'];
                    current_idx = len(d.close) - 1
                    if current_idx > entry_idx + self.params.min_bars_between_fractals + 1:
                        temp_raw_klines = self._get_raw_klines_for_period(d,
                                                                          current_idx - entry_idx + 5)
                        temp_merged = self._merge_klines_chanlun(temp_raw_klines)
                        temp_fractals = self._identify_all_merged_fractals(temp_merged)

                        for frac in temp_fractals:
                            if frac.type == -1 and frac.kline.idx > entry_idx:
                                bf_original_idx = frac.kline.low_idx
                                ago_bf_low = (len(d.close) - 1) - bf_original_idx
                                if ago_bf_low < 0: continue

                                bf_info = {"date": frac.kline.dt, "price_low": d.low[-ago_bf_low],
                                           "original_idx": bf_original_idx}
                                p_info['post_entry_bottom_fractal'] = bf_info
                                p_info['post_entry_bottom_fractal_found'] = True
                                if not pd.isna(p_info['prior_peak_high_for_tp']) and not pd.isna(bf_info['price_low']):
                                    p_info['tp1_price'] = (p_info['prior_peak_high_for_tp'] + bf_info[
                                        'price_low']) / 2.0
                                    self.log(
                                        f"TP1 CALC ({stock_name}): TP1 Price set to {p_info['tp1_price']:.2f} (PeakH: {p_info['prior_peak_high_for_tp']:.2f}, BFLow: {bf_info['price_low']:.2f})",
                                        data_feed=d, doprint=True)
                                break

                stop_loss_ma_value = self.inds[stock_name]['ma_long'][0]
                stop_loss_ma_level_name = f"DailyMA({self.p.ma_long})"
                if self.p.use_dynamic_stop_loss_level and p_info.get(
                        'entry_trend_level') == 'weekly_trend_new_logic':
                    weekly_ma_val = self.weekly_mas.get(stock_name)
                    if weekly_ma_val is not None:
                        stop_loss_ma_value = weekly_ma_val
                        stop_loss_ma_level_name = f"WeeklyMA({self.p.weekly_ma_period})"

                if not pd.isna(stop_loss_ma_value) and d.close[0] < stop_loss_ma_value * (1 - self.params.sell_ma_pct):
                    sell_signal_triggered = True
                    sell_reason = f"SL_MA_Hard (Close:{d.close[0]:.2f} < Limit:{stop_loss_ma_value * (1 - self.params.sell_ma_pct):.2f} from {stop_loss_ma_level_name}:{stop_loss_ma_value:.2f})"

                if not sell_signal_triggered:
                    self.highest_highs_since_entry[stock_name] = max(
                        self.highest_highs_since_entry.get(stock_name, float('-inf')), d.high[0])
                    atr_val = self.inds[stock_name]['atr'][0] if self.inds[stock_name][
                                                                     'atr'] is not None and not pd.isna(
                        self.inds[stock_name]['atr'][0]) else 0
                    current_atr_stop = self.atr_stop_loss_prices.get(stock_name, float('nan'))
                    if atr_val > 0:
                        new_stop_price = self.highest_highs_since_entry[
                                             stock_name] - self.params.atr_multiplier * atr_val
                        if p_info['post_entry_bottom_fractal_found'] and p_info.get('post_entry_bottom_fractal'):
                            bf_low_price = p_info['post_entry_bottom_fractal']['price_low']
                            new_stop_price = max(new_stop_price,
                                                 bf_low_price - atr_val * 0.1)

                        if pd.isna(current_atr_stop) or new_stop_price > current_atr_stop:
                            self.atr_stop_loss_prices[stock_name] = new_stop_price
                            current_atr_stop = new_stop_price

                        if not pd.isna(current_atr_stop) and d.low[0] < current_atr_stop:
                            sell_signal_triggered = True
                            sell_reason = f"SL_ATR (Low:{d.low[0]:.2f} < StopAt:{current_atr_stop:.2f}, HighestSinceEntry:{self.highest_highs_since_entry[stock_name]:.2f}, ATR:{atr_val:.3f})"

                if not sell_signal_triggered:
                    if not pd.isna(self.inds[stock_name]['ma_short'][0]) and not pd.isna(
                            self.inds[stock_name]['ma_long'][0]) and \
                            not pd.isna(self.inds[stock_name]['ma_short'][-1]) and not pd.isna(
                        self.inds[stock_name]['ma_long'][-1]) and \
                            self.inds[stock_name]['ma_short'][0] < self.inds[stock_name]['ma_long'][0] and \
                            self.inds[stock_name]['ma_short'][-1] >= self.inds[stock_name]['ma_long'][-1]:
                        sell_signal_triggered = True
                        sell_reason = f"SL_MA_DeadCross (MA{self.p.ma_short}:{self.inds[stock_name]['ma_short'][0]:.2f} < MA{self.p.ma_long}:{self.inds[stock_name]['ma_long'][0]:.2f})"

                if sell_signal_triggered:
                    p_info['sell_reason'] = sell_reason
                    self.log(f'SELL ORDER TRIGGER ({stock_name}, Reason: {sell_reason}), Shares to close: {pos.size}',
                             data_feed=d, doprint=True)
                    self.order = self.close(data=d)
                    continue

                if not p_info['tp1_hit'] and not pd.isna(p_info['tp1_price']) and p_info['tp1_price'] > 0 and d.high[
                    0] >= p_info['tp1_price']:
                    shares_to_sell = math.floor(p_info['initial_shares'] / 3 / 100) * 100
                    if shares_to_sell > 0 and p_info['shares_left'] >= shares_to_sell:
                        p_info['sell_reason'] = f"TP1 at {p_info['tp1_price']:.2f} (High:{d.high[0]:.2f})"
                        self.log(
                            f"SELL ORDER TRIGGER ({stock_name}, Reason: {p_info['sell_reason']}), Shares: {shares_to_sell}",
                            data_feed=d, doprint=True)
                        self.order = self.sell(data=d, size=shares_to_sell, exectype=bt.Order.Limit,
                                               price=p_info['tp1_price'])
                        p_info['tp1_hit'] = True;
                        p_info['tp1_sold_shares'] = shares_to_sell
                        continue

                if not p_info['tp2_hit'] and not pd.isna(p_info['tp2_price']) and p_info['tp2_price'] > 0 and d.high[
                    0] >= p_info['tp2_price']:
                    target_sell_for_tp2 = math.floor(p_info['initial_shares'] / 2 / 100) * 100
                    shares_already_sold_tp1 = p_info.get('tp1_sold_shares', 0)
                    shares_to_sell = max(0,
                                         target_sell_for_tp2 - shares_already_sold_tp1)
                    shares_to_sell = min(shares_to_sell, p_info['shares_left'])
                    shares_to_sell = math.floor(shares_to_sell / 100) * 100

                    if shares_to_sell > 0:
                        p_info['sell_reason'] = f"TP2 at {p_info['tp2_price']:.2f} (PriorPeakRef, High:{d.high[0]:.2f})"
                        self.log(
                            f"SELL ORDER TRIGGER ({stock_name}, Reason: {p_info['sell_reason']}), Shares: {shares_to_sell}",
                            data_feed=d, doprint=True)
                        self.order = self.sell(data=d, size=shares_to_sell, exectype=bt.Order.Limit,
                                               price=p_info['tp2_price'])
                        p_info['tp2_hit'] = True;
                        p_info['tp2_sold_shares'] = shares_to_sell
                        continue
            else:  # Not in position
                if is_debug_target_stock_date: self.log(f"DEBUG NoPos: Eval entry for {stock_name}.", data_feed=d,
                                                        doprint=True)

                self._validate_and_set_qualified_ref_high(d)
                q_ref_high_info = state['qualified_ref_high_info']

                if q_ref_high_info['is_fully_qualified']:
                    ref_high_price = q_ref_high_info['price']
                    ref_high_date = q_ref_high_info['date']

                    if is_debug_target_stock_date: self.log(
                        f"DEBUG BuyEval: Using QualifiedRefHigh P={ref_high_price:.2f} D={ref_high_date}", data_feed=d,
                        doprint=True)

                    if d.close[0] < ref_high_price:
                        if is_debug_target_stock_date: self.log(
                            f"DEBUG BuyEval: Pullback condition met (Close {d.close[0]:.2f} < RefHigh {ref_high_price:.2f})",
                            data_feed=d, doprint=True)

                        ma30_buy_ref = self.inds[stock_name]['ma_long'][0]
                        ma30_buy_ref_prev = self.inds[stock_name]['ma_long'][-1]
                        ma5_val = self.inds[stock_name]['ma_short'][0]

                        cond_pullback_to_ma30 = False
                        if not pd.isna(ma30_buy_ref):
                            cond_pullback_to_ma30 = (d.low[0] <= ma30_buy_ref * 1.05) and \
                                                    (d.low[0] >= ma30_buy_ref * 0.97)

                        weekly_ma_val = self.weekly_mas.get(stock_name)
                        cond_close_above_weekly_ma = False
                        if weekly_ma_val is not None and not pd.isna(weekly_ma_val):
                            cond_close_above_weekly_ma = d.close[0] > weekly_ma_val

                        cond_daily_mas_bullish = False
                        if not pd.isna(ma5_val) and not pd.isna(ma30_buy_ref):
                            cond_daily_mas_bullish = ma5_val > ma30_buy_ref

                        cond_daily_ma30_rising = False
                        if not pd.isna(ma30_buy_ref) and not pd.isna(ma30_buy_ref_prev):
                            # ##### MODIFIED ##### Compare MA30 values rounded to two decimal places
                            cond_daily_ma30_rising = round(ma30_buy_ref, 2) >= round(ma30_buy_ref_prev, 2)

                        if is_debug_target_stock_date:
                            self.log(
                                f"DEBUG BuyConds: PullbackToMA30({cond_pullback_to_ma30} Low:{d.low[0]:.2f} vs MA30Buy:{ma30_buy_ref:.2f}), "
                                f"Close>WkMA({cond_close_above_weekly_ma} Close:{d.close[0]:.2f} vs WkMA:{weekly_ma_val if weekly_ma_val is not None else 'N/A'}), "
                                f"DailyMAsBullish({cond_daily_mas_bullish} MA5:{ma5_val:.2f} vs MA30Buy:{ma30_buy_ref:.2f}), "
                                f"DailyMA30Rising({cond_daily_ma30_rising} MA30:{round(ma30_buy_ref, 2):.2f} vs PrevMA30:{round(ma30_buy_ref_prev, 2):.2f})",
                                # Log rounded values
                                data_feed=d, doprint=True)

                        if cond_pullback_to_ma30 and cond_close_above_weekly_ma and cond_daily_mas_bullish and cond_daily_ma30_rising:
                            self._pending_buy_info[stock_name] = {
                                'price': ref_high_price,
                                'date': ref_high_date,
                                'entry_trend_level': 'weekly_trend_new_logic'
                            }
                            buy_reason_details = (
                                f"Type:NewLogicPullback; RefHigh:{ref_high_price:.2f}@{ref_high_date}; "
                                f"PullbackL({d.low[0]:.2f})toMA30Buy({ma30_buy_ref:.2f}); "
                                f"C({d.close[0]:.2f})>WMA({weekly_ma_val if weekly_ma_val is not None else 'N/A'}); "
                                f"MA5({ma5_val:.2f})>MA30({ma30_buy_ref:.2f}); "
                                f"MA30Rise({round(ma30_buy_ref, 2):.2f}>={round(ma30_buy_ref_prev, 2):.2f})"
                            # Log rounded values in reason
                            )

                            cash = self.broker.get_cash()
                            value = self.broker.getvalue()
                            target_value_per_stock = value * self.params.max_position_ratio
                            size = 0
                            current_close_price = d.close[0]
                            if pd.isna(current_close_price) or current_close_price <= 0: continue

                            if target_value_per_stock > 0:
                                shares_by_target_value = math.floor(
                                    (target_value_per_stock / current_close_price) / 100.0) * 100
                                shares_by_cash = math.floor((
                                                                    cash / current_close_price) / 100.0) * 100 if cash > current_close_price * 100 else 0
                                size = min(shares_by_target_value, shares_by_cash)
                                size = max(size, 0)

                            if size > 0:
                                self.log(f'BUY ORDER TRIGGER ({stock_name}), Size:{size}. Reason: {buy_reason_details}',
                                         data_feed=d, doprint=True)
                                self.order = self.buy(data=d, size=size)
                                continue
                            elif is_debug_target_stock_date:
                                self.log(
                                    f'Buy signal for {stock_name}, but size=0. Reason: {buy_reason_details} (Cash:{cash:.0f}, TargetVal:{target_value_per_stock:.0f}, Px:{current_close_price:.2f}).',
                                    data_feed=d, doprint=True)
                                if stock_name in self._pending_buy_info: del self._pending_buy_info[
                                    stock_name]
                        elif is_debug_target_stock_date:
                            self.log(f"DEBUG BuyEval: MA/Trend conditions not fully met for {stock_name}.", data_feed=d,
                                     doprint=True)
                    elif is_debug_target_stock_date:
                        self.log(
                            f"DEBUG BuyEval: Price not in pullback from RefHigh for {stock_name} (Close {d.close[0]:.2f} not < RefHigh {ref_high_price:.2f}).",
                            data_feed=d, doprint=True)
                elif is_debug_target_stock_date:
                    self.log(
                        f"DEBUG BuyEval: No fully qualified reference high for {stock_name}. QualifiedInfo: {q_ref_high_info}",
                        data_feed=d, doprint=True)

    def stop(self):
        final_value = self.broker.getvalue()
        self.log(f'Final Portfolio Value: {final_value:.2f}', doprint=True)
        if self.p.debug_stock:
            self.log(f"Final strategy_states for {self.p.debug_stock}: {self.strategy_states.get(self.p.debug_stock)}",
                     doprint=True)