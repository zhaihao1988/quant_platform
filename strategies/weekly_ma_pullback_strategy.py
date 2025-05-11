# weekly_ma_pullback_strategy.py
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import math
from collections import deque, namedtuple

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARN: TA-Lib not found. ATR will use Backtrader's approximation if TA-Lib version was intended.")

KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'v', 'idx'])  # K线原始数据，v是volume
MergedKLine = namedtuple('MergedKLine', ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx'])
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction'])


class WeeklyMAPullbackStrategy(bt.Strategy):
    params = (
        # 周线级别参数
        ('ma_short_weekly', 5),  # 周线短周期MA
        ('ma_long_weekly', 30),  # 周线长周期MA (用于趋势和回踩参考)
        ('ma_peak_threshold_weekly', 1.30),  # 周线波峰高于其MA的阈值
        ('peak_recent_gain_periods_weekly', 5),  # 周线近期涨幅的回溯期数 (周数)
        ('peak_recent_gain_ratio_weekly', 1.2),  # 周线近期涨幅比例
        ('downstroke_invalidate_threshold_weekly', 0.05),  # 周线下破MA的无效化阈值
        ('fractal_lookback_weekly', 60),  # 周线Chanlun分析回看K线数
        ('atr_period_weekly', 10),  # 周线ATR周期
        ('atr_multiplier_weekly', 2.5),  # 周线ATR止损乘数
        ('sell_ma_pct_weekly', 0.08),  # 基于周线MA的固定百分比止损

        # 月线级别参数 (用于周线策略的更高周期过滤)
        ('ma_short_monthly', 5),  # 用于计算月线MA的参数 (如果月线MA也用金叉死叉)
        ('ma_long_monthly', 12),  # 用于计算月线MA的参数 (例如12月线)

        # 通用参数
        ('min_bars_between_fractals', 2),
        ('max_position_ratio', 0.20),
        ('printlog', True),
        ('debug_stock', None),
        ('debug_date', None),
        ('debug_date_is_start_date', False),
        ('use_dynamic_stop_loss_level', True)  # 旧策略遗留，周线策略可以简化或按周线MA类型决定
    )

    def _format_val_or_na(self, value, precision=".2f"):
        if pd.isna(value) or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
            return "N/A"
        return f"{value:{precision}}"

    def log(self, txt, dt=None, doprint=False, data_feed=None):
        # 日志函数与 DailyStrategy 保持一致，但前缀可以指明是 Weekly
        is_debug_target = False
        # 使用 self.datas[0] (即原始日线数据) 的时间来记录
        log_dt_obj = dt or self.datas[0].datetime.date(0)
        current_bar_date_str = log_dt_obj.strftime('%Y-%m-%d')

        target_data_feed = data_feed if data_feed else self.datas[0]  # 日志默认关联到主数据流

        if target_data_feed and self.p.debug_stock and target_data_feed._name == self.p.debug_stock:
            if self.p.debug_date:
                if self.p.debug_date_is_start_date:
                    if current_bar_date_str >= self.p.debug_date: is_debug_target = True
                else:
                    if current_bar_date_str == self.p.debug_date: is_debug_target = True
            else:
                is_debug_target = True

        if doprint or (self.params.printlog and is_debug_target) or \
                (self.params.printlog and not self.p.debug_stock):  # 确保 doprint=True 时总是打印
            data_name = target_data_feed._name if target_data_feed else 'StrategyWeekly'
            print(f'{log_dt_obj.isoformat()} [{data_name}] [WeeklyStrat] {txt}')

    def __init__(self):
        self.inds = dict()  # 存储日线上的指标，如ATR
        self.positions_info = dict()
        self.atr_stop_loss_prices = dict()
        self.highest_highs_since_entry = dict()
        self._pending_buy_info = dict()
        self.order = None

        # 状态管理，现在只关心周线级别，但数据源是日线，内部合成周线
        self.strategy_states = dict()

        # 从 DailyStrategy 复制并调整周线和月线数据合成所需的结构
        self.weekly_kline_data = {d._name: {
            'completed_klines': deque(),  # 存储已完成的周线KLineRaw对象
            'current_week_daily_bars': [],  # 存储当前周的日线KLineRaw对象
            'last_week_num': -1,
            'dynamic_ma_short': float('nan'),  # 当前（可能未完成的）周的短MA
            'dynamic_ma_long': float('nan'),  # 当前（可能未完成的）周的长MA
            'ma_short_history': deque(),  # 已完成周的短MA历史
            'ma_long_history': deque()  # 已完成周的长MA历史
        } for d in self.datas}

        self.monthly_kline_data = {d._name: {  # 用于周线策略的月线MA过滤
            'completed_klines': deque(),
            'current_month_daily_bars': [],
            'last_month_num': -1,
            'dynamic_ma_short': float('nan'),  # 月线短MA
            'dynamic_ma_long': float('nan'),  # 月线长MA
            'ma_short_history': deque(),
            'ma_long_history': deque()
        } for d in self.datas}

        for d in self.datas:
            stock_name = d._name
            self.inds[stock_name] = {}  # 日线指标仍在这里初始化
            # 周线策略仍然需要日线的ATR来计算止损 (DailyStrategy也是这么做的)
            if TALIB_AVAILABLE:
                self.inds[stock_name]['atr_daily'] = btind.ATR(d, period=self.p.atr_period_weekly,
                                                               plot=False)  # ATR周期参数使用周线的
            else:
                self.inds[stock_name]['atr_daily'] = btind.AverageTrueRange(d, period=self.p.atr_period_weekly,
                                                                            plot=False)

            # 注意：周线MA和月线MA将从合成数据中动态计算，不在inds中直接创建

            self.strategy_states[stock_name] = {  # 只需一套状态，代表周线级别
                'active_uptrend_peak_candidate': {'price': float('-inf'), 'date': None, 'kline_idx': -1,
                                                  'ma_at_peak': float('nan'), 'is_ma_valid': False},
                'qualified_ref_high_info': {'price': float('-inf'), 'date': None, 'kline_idx': -1,
                                            'ma_at_high': float('nan'), 'is_ma_valid': False,
                                            'is_recent_gain_valid': False, 'is_fully_qualified': False},
                'last_downstroke_info': {'end_date': None, 'low_price': float('inf'), 'kline_idx': -1,
                                         'ma_at_low': float('nan'), 'is_significant_break': False}
            }
            self.positions_info[stock_name] = None
            self.atr_stop_loss_prices[stock_name] = float('nan')
            self.highest_highs_since_entry[stock_name] = float('-inf')

    # --- 数据合成逻辑 (从DailyStrategy移植并适配) ---
    def _calculate_ma_from_klines(self, klines_closes: list, period: int) -> float:
        if not klines_closes or len(klines_closes) < period: return float('nan')
        return sum(klines_closes[-period:]) / period

    def _synthesize_higher_tf_data(self, d_feed):  # d_feed 是日线数据
        d_name = d_feed._name
        current_date = d_feed.datetime.date(0)
        # 使用 d_feed.datetime.datetime(0) 而不是 date(0) 以便获取时间信息，虽然我们这里主要用日期
        current_daily_kline = KLineRaw(dt=current_date,
                                       o=d_feed.open[0], h=d_feed.high[0], l=d_feed.low[0], c=d_feed.close[0],
                                       v=d_feed.volume[0] if hasattr(d_feed, 'volume') else 0,  # 确保有成交量
                                       idx=len(d_feed) - 1)  # 日线K线在其数据流中的索引

        # --- 周线合成 ---
        wk_data = self.weekly_kline_data[d_name]
        current_week_iso = current_date.isocalendar()  # (year, week_num, weekday)
        current_week_num_key = current_week_iso[0] * 100 + current_week_iso[1]

        if wk_data['last_week_num'] == -1:
            wk_data['last_week_num'] = current_week_num_key

        if current_week_num_key != wk_data['last_week_num'] and wk_data['current_week_daily_bars']:
            # 上一周结束
            bars = wk_data['current_week_daily_bars']
            wk_open = bars[0].o
            wk_high = max(b.h for b in bars)
            wk_low = min(b.l for b in bars)
            wk_close = bars[-1].c
            wk_volume = sum(b.v for b in bars)
            wk_dt = bars[-1].dt  # 使用周的最后一天日期
            # idx for completed_klines is its own sequence index
            completed_wk_kline = KLineRaw(dt=wk_dt, o=wk_open, h=wk_high, l=wk_low, c=wk_close, v=wk_volume,
                                          idx=len(wk_data['completed_klines']))
            wk_data['completed_klines'].append(completed_wk_kline)

            completed_week_closes = [k.c for k in wk_data['completed_klines']]
            ma_s = self._calculate_ma_from_klines(completed_week_closes, self.p.ma_short_weekly)
            ma_l = self._calculate_ma_from_klines(completed_week_closes, self.p.ma_long_weekly)
            wk_data['ma_short_history'].append(ma_s)
            wk_data['ma_long_history'].append(ma_l)

            wk_data['current_week_daily_bars'] = []  # 为新的一周重置

        wk_data['last_week_num'] = current_week_num_key
        wk_data['current_week_daily_bars'].append(current_daily_kline)

        # 计算当前（可能未完成的）周的动态MA
        all_w_closes_for_dyn_ma = [k.c for k in wk_data['completed_klines']]
        if wk_data['current_week_daily_bars']:  # 如果当前周已经有日线数据
            all_w_closes_for_dyn_ma.append(wk_data['current_week_daily_bars'][-1].c)  # 添加当前周最后一根日线的收盘价

        wk_data['dynamic_ma_short'] = self._calculate_ma_from_klines(all_w_closes_for_dyn_ma, self.p.ma_short_weekly)
        wk_data['dynamic_ma_long'] = self._calculate_ma_from_klines(all_w_closes_for_dyn_ma, self.p.ma_long_weekly)

        # --- 月线合成 (逻辑与周线类似) ---
        mo_data = self.monthly_kline_data[d_name]
        current_month_num_key = current_date.year * 100 + current_date.month

        if mo_data['last_month_num'] == -1:
            mo_data['last_month_num'] = current_month_num_key

        if current_month_num_key != mo_data['last_month_num'] and mo_data['current_month_daily_bars']:
            bars = mo_data['current_month_daily_bars']
            mo_open = bars[0].o;
            mo_high = max(b.h for b in bars);
            mo_low = min(b.l for b in bars)
            mo_close = bars[-1].c;
            mo_volume = sum(b.v for b in bars);
            mo_dt = bars[-1].dt
            completed_mo_kline = KLineRaw(dt=mo_dt, o=mo_open, h=mo_high, l=mo_low, c=mo_close, v=mo_volume,
                                          idx=len(mo_data['completed_klines']))
            mo_data['completed_klines'].append(completed_mo_kline)

            completed_month_closes = [k.c for k in mo_data['completed_klines']]
            ma_s_mo = self._calculate_ma_from_klines(completed_month_closes, self.p.ma_short_monthly)
            ma_l_mo = self._calculate_ma_from_klines(completed_month_closes, self.p.ma_long_monthly)
            mo_data['ma_short_history'].append(ma_s_mo)
            mo_data['ma_long_history'].append(ma_l_mo)

            mo_data['current_month_daily_bars'] = []

        mo_data['last_month_num'] = current_month_num_key
        mo_data['current_month_daily_bars'].append(current_daily_kline)

        all_m_closes_for_dyn_ma = [k.c for k in mo_data['completed_klines']]
        if mo_data['current_month_daily_bars']:
            all_m_closes_for_dyn_ma.append(mo_data['current_month_daily_bars'][-1].c)

        mo_data['dynamic_ma_short'] = self._calculate_ma_from_klines(all_m_closes_for_dyn_ma, self.p.ma_short_monthly)
        mo_data['dynamic_ma_long'] = self._calculate_ma_from_klines(all_m_closes_for_dyn_ma, self.p.ma_long_monthly)

    # --- Chanlun 辅助函数 (与旧策略或DailyStrategy相同, 这里省略以节约篇幅, 假设已复制) ---
    # _merge_klines_chanlun, _find_merged_fractal,
    # _identify_all_merged_fractals, _identify_strokes
    # 需要确保 KLineRaw 的定义包含 'v' 字段，如果这些函数依赖它的话
    # KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'v', 'idx'])
    # (此处省略，请从 DailyStrategy 复制这些函数)
    # ...

    # Chanlun K线合并
    def _merge_klines_chanlun(self, bars_data: list[KLineRaw]) -> list[MergedKLine]:
        if not bars_data or len(bars_data) < 1:
            if bars_data: k_raw = bars_data[0]; return [
                MergedKLine(dt=k_raw.dt, o=k_raw.o, h=k_raw.h, l=k_raw.l, c=k_raw.c, idx=k_raw.idx, direction=0,
                            high_idx=k_raw.idx, low_idx=k_raw.idx)]
            return []
        merged_lines: list[MergedKLine] = []
        k1_raw = bars_data[0];
        k1 = MergedKLine(dt=k1_raw.dt, o=k1_raw.o, h=k1_raw.h, l=k1_raw.l, c=k1_raw.c, idx=k1_raw.idx, direction=0,
                         high_idx=k1_raw.idx, low_idx=k1_raw.idx)
        merged_lines.append(k1);
        current_segment_trend = 0
        for i in range(1, len(bars_data)):
            k1_merged = merged_lines[-1];
            k2_raw = bars_data[i]
            k1_includes_k2 = (k1_merged.h >= k2_raw.h and k1_merged.l <= k2_raw.l);
            k2_includes_k1 = (k2_raw.h >= k1_merged.h and k2_raw.l <= k1_merged.l)
            if k1_includes_k2 or k2_includes_k1:
                trend_for_inclusion = current_segment_trend;
                m_h, m_l = k1_merged.h, k1_merged.l;
                m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
                if trend_for_inclusion == 1:
                    m_h = max(k1_merged.h,
                              k2_raw.h);
                    m_l = k1_merged.l;
                    m_high_idx = k2_raw.idx if k2_raw.h >= k1_merged.h else k1_merged.high_idx;
                    m_low_idx = k1_merged.low_idx
                elif trend_for_inclusion == -1:
                    m_h = k1_merged.h;
                    m_l = min(k1_merged.l,
                              k2_raw.l);
                    m_high_idx = k1_merged.high_idx;
                    m_low_idx = k2_raw.idx if k2_raw.l <= k1_merged.l else k1_merged.low_idx
                else:
                    if k1_includes_k2:
                        m_h, m_l = k1_merged.h, k1_merged.l;
                        m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
                    elif k2_includes_k1:
                        m_h, m_l = k2_raw.h, k2_raw.l;
                        m_high_idx, m_low_idx = k2_raw.idx, k2_raw.idx
                    if k1_merged.h == k2_raw.h and k1_merged.l == k2_raw.l: m_h, m_l = k1_merged.h, k1_merged.l; m_high_idx, m_low_idx = k1_merged.high_idx, k1_merged.low_idx
                merged_lines[-1] = MergedKLine(dt=k2_raw.dt, o=k1_merged.o, h=m_h, l=m_l, c=k2_raw.c, idx=k2_raw.idx,
                                               direction=k1_merged.direction, high_idx=m_high_idx, low_idx=m_low_idx)
            else:
                new_segment_direction = 0
                if k2_raw.h > k1_merged.h and k2_raw.l > k1_merged.l:
                    new_segment_direction = 1
                elif k2_raw.h < k1_merged.h and k2_raw.l < k1_merged.l:
                    new_segment_direction = -1
                if merged_lines[-1].direction == 0 and len(merged_lines) > 1:
                    k_prev_prev = merged_lines[-2];
                    k_prev = merged_lines[-1];
                    prev_dir = 0
                    if k_prev.h > k_prev_prev.h and k_prev.l > k_prev_prev.l:
                        prev_dir = 1
                    elif k_prev.h < k_prev_prev.h and k_prev.l < k_prev_prev.l:
                        prev_dir = -1
                    merged_lines[-1] = merged_lines[-1]._replace(direction=prev_dir)
                current_segment_trend = new_segment_direction
                k2_new_merged = MergedKLine(dt=k2_raw.dt, o=k2_raw.o, h=k2_raw.h, l=k2_raw.l, c=k2_raw.c,
                                            idx=k2_raw.idx, direction=new_segment_direction, high_idx=k2_raw.idx,
                                            low_idx=k2_raw.idx)
                merged_lines.append(k2_new_merged)
        for i in range(len(merged_lines)):
            if merged_lines[i].direction == 0:
                if i > 0:
                    if merged_lines[i].h > merged_lines[i - 1].h and merged_lines[i].l > merged_lines[i - 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=1)
                    elif merged_lines[i].h < merged_lines[i - 1].h and merged_lines[i].l < merged_lines[i - 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=-1)
                elif i < len(merged_lines) - 1:
                    if merged_lines[i].h < merged_lines[i + 1].h and merged_lines[i].l < merged_lines[i + 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=1)
                    elif merged_lines[i].h > merged_lines[i + 1].h and merged_lines[i].l > merged_lines[i + 1].l:
                        merged_lines[i] = merged_lines[i]._replace(direction=-1)
        return merged_lines

    def _find_merged_fractal(self, merged_klines: list[MergedKLine], index: int) -> Fractal | None:
        if index < 1 or index >= len(merged_klines) - 1: return None
        k_prev, k_curr, k_next = merged_klines[index - 1], merged_klines[index], merged_klines[index + 1]
        is_top = k_curr.h > k_prev.h and k_curr.h > k_next.h;
        is_bottom = k_curr.l < k_prev.l and k_curr.l < k_next.l
        if is_top and is_bottom: return None
        if is_top: return Fractal(kline=k_curr, m_idx=index, type=1)
        if is_bottom: return Fractal(kline=k_curr, m_idx=index, type=-1)
        return None

    def _identify_all_merged_fractals(self, merged_klines: list[MergedKLine]) -> list[Fractal]:
        fractals = [];
        if len(merged_klines) < 3: return fractals
        for i in range(1, len(merged_klines) - 1):
            fractal = self._find_merged_fractal(merged_klines, i)
            if fractal: fractals.append(fractal)
        return fractals

    def _identify_strokes(self, fractals: list[Fractal], merged_klines: list[MergedKLine]) -> list[Stroke]:
        strokes = [];
        if len(fractals) < 2: return strokes
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
            if bars_between_merged < self.params.min_bars_between_fractals: continue
            stroke_direction = 0
            if last_confirmed_fractal.type == -1 and current_fractal.type == 1:
                if current_fractal.kline.h > last_confirmed_fractal.kline.l: stroke_direction = 1
            elif last_confirmed_fractal.type == 1 and current_fractal.type == -1:
                if current_fractal.kline.l < last_confirmed_fractal.kline.h: stroke_direction = -1
            if stroke_direction != 0:
                strokes.append(Stroke(start_fractal=last_confirmed_fractal, end_fractal=current_fractal,
                                      direction=stroke_direction))
                last_confirmed_fractal = current_fractal
        return strokes

    # ... (其他 Chanlun 辅助函数需要从 DailyStrategy 完整复制过来)

    def _get_klines_for_chanlun_weekly(self, d_feed, lookback_count: int) -> list[KLineRaw]:
        # 此函数专门为周线策略从内部合成的周线数据中获取K线
        d_name = d_feed._name  # d_feed 是日线数据
        wk_data = self.weekly_kline_data[d_name]

        temp_klines = list(wk_data['completed_klines'])  # 已完成的周线

        # 添加当前（可能未完成的）周线
        if wk_data['current_week_daily_bars']:
            bars = wk_data['current_week_daily_bars']
            dyn_wk_open = bars[0].o
            dyn_wk_high = max(b.h for b in bars) if bars else float('nan')
            dyn_wk_low = min(b.l for b in bars) if bars else float('nan')
            dyn_wk_close = bars[-1].c
            dyn_wk_volume = sum(b.v for b in bars) if bars else 0
            dyn_wk_dt = bars[-1].dt  # 使用当前周的最后一天日期
            dyn_idx = len(wk_data['completed_klines'])  # 索引是它在完整周线序列中的位置

            if not pd.isna(dyn_wk_high) and not pd.isna(dyn_wk_low):
                temp_klines.append(KLineRaw(dt=dyn_wk_dt, o=dyn_wk_open, h=dyn_wk_high, l=dyn_wk_low,
                                            c=dyn_wk_close, v=dyn_wk_volume, idx=dyn_idx))

        return temp_klines[-lookback_count:] if lookback_count < len(temp_klines) else temp_klines

    def _get_ma_value_at_kline_idx_weekly(self, d_feed, ma_type: str, weekly_kline_idx: int) -> float:
        # 获取指定周线K线索引处的周线MA值
        d_name = d_feed._name
        history_deque = None

        if ma_type == 'short':
            history_deque = self.weekly_kline_data[d_name]['ma_short_history']
        elif ma_type == 'long' or ma_type == 'long_for_peak':  # 周线策略中，这两个MA周期可能相同
            history_deque = self.weekly_kline_data[d_name]['ma_long_history']
        else:
            return float('nan')

        if history_deque and 0 <= weekly_kline_idx < len(history_deque):
            return history_deque[weekly_kline_idx]

        # 如果请求的是当前动态周的MA
        num_completed_weeks = len(self.weekly_kline_data[d_name]['completed_klines'])
        if weekly_kline_idx == num_completed_weeks:  # 即当前（未完成）周
            if ma_type == 'short': return self.weekly_kline_data[d_name]['dynamic_ma_short']
            if ma_type == 'long' or ma_type == 'long_for_peak': return self.weekly_kline_data[d_name]['dynamic_ma_long']

        return float('nan')

    # --- 核心逻辑函数 (适配为周线级别) ---
    def _update_weekly_active_peak_candidate(self, d_feed):  # d_feed 是日线数据
        stock_name = d_feed._name
        state = self.strategy_states[stock_name]  # 直接用周线状态
        LOG_PREFIX = f"[{stock_name}][{d_feed.datetime.date(0)}][WEEKLY_CODE][ActPeakCand]"

        # 获取内部合成的周线K线数据
        weekly_klines = self._get_klines_for_chanlun_weekly(d_feed, self.p.fractal_lookback_weekly)
        if not weekly_klines: return

        merged_klines = self._merge_klines_chanlun(weekly_klines)  # Chanlun函数作用于周线KLineRaw
        if not merged_klines or len(merged_klines) < self.p.min_bars_between_fractals + 2: return

        all_fractals = self._identify_all_merged_fractals(merged_klines)
        strokes = self._identify_strokes(all_fractals, merged_klines)

        current_candidate_price = state['active_uptrend_peak_candidate']['price']
        if pd.isna(current_candidate_price): current_candidate_price = float('-inf')
        initial_candidate_price_for_log = current_candidate_price
        new_candidate_found_this_call = False

        for stroke in reversed(strokes):
            if stroke.direction == 1:  # 上升笔
                peak_fractal = stroke.end_fractal
                merged_k_peak = peak_fractal.kline  # 这是合并后的周线KLineRaw

                # merged_k_peak.high_idx 是指在 _get_klines_for_chanlun_weekly 返回的列表中的索引
                # 我们需要找到这个周线KLineRaw对象本身
                if not (0 <= merged_k_peak.high_idx < len(weekly_klines)): continue
                peak_kline_actual = weekly_klines[merged_k_peak.high_idx]  # 真实的周线KLineRaw对象

                peak_price = peak_kline_actual.h
                peak_kline_tf_idx = peak_kline_actual.idx  # 这是它在完整周线序列中的索引 (0, 1, 2...)

                ma_at_peak_time = self._get_ma_value_at_kline_idx_weekly(d_feed, 'long_for_peak', peak_kline_tf_idx)

                if pd.isna(ma_at_peak_time) or pd.isna(peak_price): continue
                is_ma_valid = peak_price > ma_at_peak_time * self.p.ma_peak_threshold_weekly

                if is_ma_valid:
                    if peak_price > current_candidate_price:
                        state['active_uptrend_peak_candidate'] = {
                            'price': peak_price, 'date': peak_kline_actual.dt,
                            'kline_idx': peak_kline_tf_idx,
                            'ma_at_peak': ma_at_peak_time, 'is_ma_valid': True
                        }
                        current_candidate_price = peak_price
                        new_candidate_found_this_call = True
                        state['qualified_ref_high_info']['is_fully_qualified'] = False  # Reset

        # 检查当前（可能未完成的）周线是否形成更高候选
        if weekly_klines:  # 确保有周线数据
            current_tf_bar = weekly_klines[-1]  # 当前最新的周线KLineRaw (可能是动态的)
            current_tf_bar_high = current_tf_bar.h
            ma_current_tf_bar = self.weekly_kline_data[stock_name]['dynamic_ma_long']  # 用动态长MA

            if not pd.isna(ma_current_tf_bar) and not pd.isna(current_tf_bar_high) and \
                    current_tf_bar_high > ma_current_tf_bar * self.p.ma_peak_threshold_weekly:
                if current_tf_bar_high > current_candidate_price:
                    state['active_uptrend_peak_candidate'] = {
                        'price': current_tf_bar_high, 'date': current_tf_bar.dt,
                        'kline_idx': current_tf_bar.idx,  # 这是它在周线序列中的索引
                        'ma_at_peak': ma_current_tf_bar, 'is_ma_valid': True
                    }
                    new_candidate_found_this_call = True
                    state['qualified_ref_high_info']['is_fully_qualified'] = False  # Reset

        final_active_peak = state['active_uptrend_peak_candidate']
        if new_candidate_found_this_call or \
                (self.p.debug_stock == stock_name and not math.isclose(initial_candidate_price_for_log,
                                                                       final_active_peak['price'] if not pd.isna(
                                                                           final_active_peak['price']) else float(
                                                                           '-inf'))):
            self.log(
                f"{LOG_PREFIX} Final ActivePeak: Px={self._format_val_or_na(final_active_peak['price'])}@{final_active_peak['date']}. Prev: {self._format_val_or_na(initial_candidate_price_for_log)}",
                data_feed=d_feed, doprint=True)

    def _check_weekly_uptrend_invalidation(self, d_feed):  # d_feed 是日线数据
        stock_name = d_feed._name
        state = self.strategy_states[stock_name]  # 周线状态
        LOG_PREFIX = f"[{stock_name}][{d_feed.datetime.date(0)}][WEEKLY_CODE][UptrendInvalid]"

        weekly_klines = self._get_klines_for_chanlun_weekly(d_feed, self.p.fractal_lookback_weekly)
        if not weekly_klines: return False
        # ... (复制并调整 DailyStrategy._check_uptrend_invalidation_tf(...,timeframe='weekly') 的逻辑)
        # 确保使用周线MA和周线KLineRaw对象进行判断
        merged_klines = self._merge_klines_chanlun(weekly_klines)
        if not merged_klines or len(merged_klines) < 3: return False
        all_fractals = self._identify_all_merged_fractals(merged_klines);
        strokes = self._identify_strokes(all_fractals, merged_klines)
        last_down_stroke = None
        for stroke in reversed(strokes):
            if stroke.direction == -1: last_down_stroke = stroke; break
        if last_down_stroke:
            bottom_fractal = last_down_stroke.end_fractal;
            merged_k_low = bottom_fractal.kline
            if not (0 <= merged_k_low.low_idx < len(
                weekly_klines)): return False  # merged_k_low.low_idx 是在 weekly_klines 的索引
            low_kline_actual = weekly_klines[merged_k_low.low_idx]

            low_price = low_kline_actual.l;
            low_kline_tf_idx = low_kline_actual.idx
            ma_at_low_time = self._get_ma_value_at_kline_idx_weekly(d_feed, 'long_for_peak', low_kline_tf_idx)

            if not pd.isna(ma_at_low_time) and not pd.isna(low_price):
                state['last_downstroke_info'] = {'end_date': low_kline_actual.dt, 'low_price': low_price,
                                                 'kline_idx': low_kline_tf_idx, 'ma_at_low': ma_at_low_time,
                                                 'is_significant_break': False}
                if low_price < ma_at_low_time * (1 - self.p.downstroke_invalidate_threshold_weekly):
                    state['last_downstroke_info']['is_significant_break'] = True
                    self.log(
                        f"{LOG_PREFIX} SIGNIFICANT BREAK DOWN: Low {low_price:.2f} < MA@{self._format_val_or_na(ma_at_low_time)} * Thr. Invalidating.",
                        data_feed=d_feed, doprint=True)
                    state['active_uptrend_peak_candidate']['price'] = float('-inf');
                    state['active_uptrend_peak_candidate']['is_ma_valid'] = False
                    state['qualified_ref_high_info']['is_fully_qualified'] = False
                    return True
        return False

    def _validate_and_set_weekly_qualified_ref_high(self, d_feed):  # d_feed 是日线数据
        stock_name = d_feed._name
        state = self.strategy_states[stock_name]  # 周线状态
        candidate = state['active_uptrend_peak_candidate']
        LOG_PREFIX = f"[{stock_name}][{d_feed.datetime.date(0)}][WEEKLY_CODE][ValidateQualRefHigh]"

        if not candidate or pd.isna(candidate['price']) or candidate['price'] <= float('-inf') or candidate[
            'kline_idx'] < 0:
            if state['qualified_ref_high_info']['is_fully_qualified']: self.log(
                f"{LOG_PREFIX} No active candidate, clearing.", data_feed=d_feed, doprint=True)
            state['qualified_ref_high_info']['is_fully_qualified'] = False;
            return False

        peak_kline_tf_idx = candidate['kline_idx']  # 这是周线K线在其完整序列中的索引
        peak_price = candidate['price']

        # 获取完整的历史周线K线序列，用于计算涨幅
        # lookback_count 应该足够大以包含 peak_kline_tf_idx 和 peak_recent_gain_periods_weekly 之前的K线
        full_weekly_history_klines = self._get_klines_for_chanlun_weekly(d_feed,
                                                                         peak_kline_tf_idx + self.p.peak_recent_gain_periods_weekly + 5)

        actual_list_idx_of_peak = -1  # peak_kline_tf_idx 在 full_weekly_history_klines 中的索引
        for i_k, kline_raw_k in enumerate(full_weekly_history_klines):
            if kline_raw_k.idx == peak_kline_tf_idx: actual_list_idx_of_peak = i_k; break

        if actual_list_idx_of_peak == -1:
            state['qualified_ref_high_info']['is_fully_qualified'] = False;
            return False

        lookback_periods = self.p.peak_recent_gain_periods_weekly
        start_idx_in_list = max(0, actual_list_idx_of_peak - lookback_periods + 1)
        # gain_window_klines = full_weekly_history_klines[start_idx_in_list : actual_list_idx_of_peak + 1]
        # 确保 end_idx_for_gain_calc_in_list 包含当前 peak_kline_idx 对应的K线
        end_idx_for_gain_calc_in_list = actual_list_idx_of_peak + 1
        window_klines = full_weekly_history_klines[start_idx_in_list: end_idx_for_gain_calc_in_list]

        if not window_klines:
            state['qualified_ref_high_info']['is_fully_qualified'] = False;
            return False

        highest_in_window = peak_price  # 候选峰值即为窗口内最高
        lowest_in_window_values = [k.l for k in window_klines if not pd.isna(k.l)]
        lowest_in_window = min(lowest_in_window_values) if lowest_in_window_values else float('inf')

        is_recent_gain_valid = False;
        gain_ratio = 0.0
        if lowest_in_window > 0 and lowest_in_window != float('inf') and highest_in_window > float('-inf'):
            gain_ratio = highest_in_window / lowest_in_window
            if gain_ratio > self.p.peak_recent_gain_ratio_weekly: is_recent_gain_valid = True

        if is_recent_gain_valid and candidate['is_ma_valid']:
            state['qualified_ref_high_info'] = {
                'price': candidate['price'], 'date': candidate['date'], 'kline_idx': candidate['kline_idx'],
                'ma_at_high': candidate['ma_at_peak'], 'is_ma_valid': True,
                'is_recent_gain_valid': True, 'is_fully_qualified': True
            }
            self.log(
                f"{LOG_PREFIX} QUALIFIED RefHigh SET: Px={self._format_val_or_na(candidate['price'])}@{candidate['date']}",
                data_feed=d_feed, doprint=True)
            return True
        else:
            if state['qualified_ref_high_info']['is_fully_qualified']:
                self.log(
                    f"{LOG_PREFIX} Candidate Px={self._format_val_or_na(candidate['price'])} FAILED validation (Gain:{is_recent_gain_valid},MAVal:{candidate['is_ma_valid']}). Clearing.",
                    data_feed=d_feed, doprint=True)
            state['qualified_ref_high_info']['is_fully_qualified'] = False
            return False

    def notify_order(self, order):  # 基本与DailyStrategy一致
        stock_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}',
                    data_feed=order.data, doprint=True)
                pending_info = self._pending_buy_info.get(stock_name, {})
                entry_trend_level = pending_info.get("entry_trend_level", "weekly_pullback");  # 默认为周线
                self.positions_info[stock_name] = {'entry_price': order.executed.price,
                                                   'initial_shares': order.executed.size,
                                                   'shares_left': order.executed.size,
                                                   'prior_peak_high_for_tp': pending_info.get("price", float('nan')),
                                                   'prior_peak_date_for_tp': pending_info.get("date", "N/A"),
                                                   'original_daily_entry_idx': len(order.data.close) - 1,  # 记录日线索引
                                                   'entry_date': bt.num2date(order.data.datetime[0]).date(),
                                                   'entry_trend_level': entry_trend_level,
                                                   'tp1_price': float('nan'), 'tp1_hit': False, 'tp1_sold_shares': 0,
                                                   'tp2_price': pending_info.get("price", float('nan')),
                                                   'tp2_hit': False, 'tp2_sold_shares': 0,
                                                   'post_entry_bottom_fractal': None,
                                                   'post_entry_bottom_fractal_found': False, 'sell_reason': None}
                self.highest_highs_since_entry[stock_name] = order.data.high[0]  # 用日线高点初始化
                atr_val = self.inds[stock_name]['atr_daily'][0]  # 使用日线ATR
                if not pd.isna(atr_val) and atr_val > 0:
                    self.atr_stop_loss_prices[
                        stock_name] = order.executed.price - self.p.atr_multiplier_weekly * atr_val
                else:
                    self.atr_stop_loss_prices[stock_name] = order.executed.price * (1 - 0.05)
                if stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]
            elif order.issell():
                p_info = self.positions_info.get(stock_name);
                reason = p_info.get('sell_reason', "N/A") if p_info else "N/A"
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}. Reason: {reason}',
                    data_feed=order.data, doprint=True)
                if p_info: p_info['shares_left'] -= abs(order.executed.size)
            if self.order == order: self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order Canceled/Margin/Rejected/Expired - Status: {order.getstatusname()}', data_feed=order.data,
                     doprint=True)
            if order.isbuy() and stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]
            if self.order == order: self.order = None

    def notify_trade(self, trade):  # 与DailyStrategy一致
        if not trade.isclosed: return
        stock_name = trade.data._name
        self.log(f'TRADE PROFIT ({stock_name}), GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}', data_feed=trade.data,
                 doprint=True)
        self.positions_info[stock_name] = None;
        self.atr_stop_loss_prices[stock_name] = float('nan');
        self.highest_highs_since_entry[stock_name] = float('-inf')
        if stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]

    def next(self):
        # 接收的是日线数据 d_feed
        d_feed = self.datas[0]
        stock_name = d_feed._name
        pos = self.getposition(d_feed)
        state = self.strategy_states[stock_name]  # 周线状态

        current_date_obj = d_feed.datetime.date(0)
        current_date_str = current_date_obj.strftime('%Y-%m-%d')
        LOG_PREFIX_BAR = f"[{stock_name}][{current_date_str}][WeeklyStratNext]"

        is_debug_target_stock_date = self.p.debug_stock == stock_name and (not self.p.debug_date or (
                    self.p.debug_date_is_start_date and current_date_str >= self.p.debug_date) or (
                                                                                       not self.p.debug_date_is_start_date and current_date_str == self.p.debug_date))

        # 1. 确保有足够的日线数据来合成至少一些周线/月线并计算MA
        # 这个最小长度需要基于周线和月线参数来定，例如最长的 fractal_lookback_weekly + ma_long_weekly
        min_daily_bars_for_synthesis = max(self.p.fractal_lookback_weekly, self.p.ma_long_weekly,
                                           self.p.ma_long_monthly * 4) + 20  # 粗略估计
        if len(d_feed.close) < min_daily_bars_for_synthesis:
            if is_debug_target_stock_date: self.log(
                f"{LOG_PREFIX_BAR} Daily data too short for synthesis ({len(d_feed.close)} < {min_daily_bars_for_synthesis})",
                doprint=True, data_feed=d_feed)
            return

        # 2. 调用数据合成逻辑 (每个日线bar都调用，以更新动态周线/月线MA)
        self._synthesize_higher_tf_data(d_feed)

        # 3. 周线级别的核心逻辑判断 (理想情况下，只在每周的最后一个交易日执行，或者当新的一周形成时执行)
        # 为了模拟 DailyStrategy 的行为（它在每个日线上评估所有 timeframe），我们也在每个日线上评估周线逻辑
        # 基于最新的（可能是动态合成的）周线数据。

        # 3a. 检查周线上升趋势是否被破坏 (基于合成的周线数据)
        invalidation_occurred_weekly = self._check_weekly_uptrend_invalidation(d_feed)

        # 3b. 如果趋势有效，更新周线级别的活跃波峰候选
        if not invalidation_occurred_weekly:
            self._update_weekly_active_peak_candidate(d_feed)

        # 3c. 验证并设置周线级别的合格参考高点
        self._validate_and_set_weekly_qualified_ref_high(d_feed)

        # 4. 周线买入逻辑
        if not pos.size > 0:  # 如果没有持仓
            q_ref_weekly = state['qualified_ref_high_info']  # 这是周线级别的合格高点

            if q_ref_weekly['is_fully_qualified']:
                ref_high_price_weekly = q_ref_weekly['price']

                # 条件：当前日线收盘价 < 周线级别参考高点
                if d_feed.close[0] < ref_high_price_weekly:
                    wk_ma_short_val = self.weekly_kline_data[stock_name]['dynamic_ma_short']
                    wk_ma_long_buy_val = self.weekly_kline_data[stock_name]['dynamic_ma_long']
                    mo_ma_long_trend_val = self.monthly_kline_data[stock_name]['dynamic_ma_long']

                    # 条件1: 当前日线低点回踩到动态周线MA长周期均线的某个范围内
                    cond_pullback_wk_ma = (not pd.isna(wk_ma_long_buy_val)) and \
                                          (d_feed.low[0] <= wk_ma_long_buy_val * 1.05) and \
                                          (d_feed.low[0] >= wk_ma_long_buy_val * 0.97)

                    # 条件2: 动态周线MA呈多头排列 (短周期 > 长周期)
                    cond_wk_mas_bullish = not pd.isna(wk_ma_short_val) and \
                                          not pd.isna(wk_ma_long_buy_val) and \
                                          wk_ma_short_val > wk_ma_long_buy_val

                    # 条件3: 当前日线收盘价 > 动态月线MA (更高周期趋势向上)
                    cond_mo_trend_up = not pd.isna(mo_ma_long_trend_val) and \
                                       d_feed.close[0] > mo_ma_long_trend_val

                    if cond_pullback_wk_ma and cond_wk_mas_bullish and cond_mo_trend_up:
                        self._pending_buy_info[stock_name] = {'price': ref_high_price_weekly,
                                                              'date': q_ref_weekly['date'],
                                                              'entry_trend_level': 'weekly_pullback'}  # 标记为周线买入
                        buy_reason = f"WeeklyPB; RefH:{ref_high_price_weekly:.2f}; PullL({d_feed.low[0]:.2f})toWMA{self.p.ma_long_weekly}({self._format_val_or_na(wk_ma_long_buy_val)}); C({d_feed.close[0]:.2f})>MMA{self.p.ma_long_monthly}({self._format_val_or_na(mo_ma_long_trend_val)})"

                        cash = self.broker.get_cash();
                        value = self.broker.getvalue()
                        target_value = value * self.p.max_position_ratio;
                        size = 0
                        if d_feed.close[0] > 0:
                            shares_target = math.floor((target_value / d_feed.close[0]) / 100.0) * 100
                            shares_cash = math.floor((cash / d_feed.close[0]) / 100.0) * 100 if cash > d_feed.close[
                                0] * 100 else 0
                            size = min(shares_target, shares_cash);
                            size = max(size, 0)
                        if size > 0:
                            self.log(f'BUY ORDER TRIGGER ({stock_name}), Size:{size}. Reason: {buy_reason}',
                                     data_feed=d_feed, doprint=True)
                            self.order = self.buy(data=d_feed, size=size)
                        else:
                            if stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]

        # 5. 周线级别的止盈止损逻辑 (基于日线ATR 和 周线MA)
        elif pos.size > 0 and self.positions_info.get(stock_name):
            p_info = self.positions_info[stock_name]
            sell_signal_triggered = False;
            sell_reason = None

            # 止损MA: 使用周线动态长MA
            stop_loss_ma_value = self.weekly_kline_data[stock_name]['dynamic_ma_long']
            stop_loss_ma_level_name = f"DynWeeklyMA({self.p.ma_long_weekly})"

            if not pd.isna(stop_loss_ma_value) and d_feed.close[0] < stop_loss_ma_value * (
                    1 - self.p.sell_ma_pct_weekly):
                sell_signal_triggered = True
                sell_reason = f"SL_MA_Hard_Weekly (C:{d_feed.close[0]:.2f} < Lim:{stop_loss_ma_value * (1 - self.p.sell_ma_pct_weekly):.2f} from {stop_loss_ma_level_name}:{self._format_val_or_na(stop_loss_ma_value)})"

            if not sell_signal_triggered:
                self.highest_highs_since_entry[stock_name] = max(
                    self.highest_highs_since_entry.get(stock_name, float('-inf')), d_feed.high[0])
                atr_val_daily = self.inds[stock_name]['atr_daily'][0]  # 使用日线ATR
                current_atr_stop = self.atr_stop_loss_prices.get(stock_name, float('nan'))
                if not pd.isna(atr_val_daily) and atr_val_daily > 0:
                    new_stop_price = self.highest_highs_since_entry[
                                         stock_name] - self.p.atr_multiplier_weekly * atr_val_daily
                    # (可选：结合分型低点调整ATR止损)
                    if pd.isna(current_atr_stop) or new_stop_price > current_atr_stop:
                        self.atr_stop_loss_prices[stock_name] = new_stop_price
                        current_atr_stop = new_stop_price
                    if not pd.isna(current_atr_stop) and d_feed.low[0] < current_atr_stop:
                        sell_signal_triggered = True
                        sell_reason = f"SL_ATR_Weekly (L:{d_feed.low[0]:.2f} < StopAt:{current_atr_stop:.2f}, HighestDailyH:{self.highest_highs_since_entry[stock_name]:.2f}, DailyATR:{atr_val_daily:.3f})"

            # 周线MA死叉 (动态周线MA)
            if not sell_signal_triggered:
                wk_ma_short = self.weekly_kline_data[stock_name]['dynamic_ma_short']
                wk_ma_long = self.weekly_kline_data[stock_name]['dynamic_ma_long']
                # 需要前一周期的MA值来判断死叉
                prev_wk_ma_short = self.weekly_kline_data[stock_name]['ma_short_history'][-1] if \
                self.weekly_kline_data[stock_name]['ma_short_history'] else float('nan')
                prev_wk_ma_long = self.weekly_kline_data[stock_name]['ma_long_history'][-1] if \
                self.weekly_kline_data[stock_name]['ma_long_history'] else float('nan')

                if not pd.isna(wk_ma_short) and not pd.isna(wk_ma_long) and \
                        not pd.isna(prev_wk_ma_short) and not pd.isna(prev_wk_ma_long) and \
                        wk_ma_short < wk_ma_long and prev_wk_ma_short >= prev_wk_ma_long:
                    sell_signal_triggered = True
                    sell_reason = f"SL_MA_DeadCross_Weekly (WkMA{self.p.ma_short_weekly}:{wk_ma_short:.2f} < WkMA{self.p.ma_long_weekly}:{wk_ma_long:.2f})"

            if sell_signal_triggered:
                p_info['sell_reason'] = sell_reason
                self.log(f'SELL ORDER TRIGGER ({stock_name}, Reason: {sell_reason}), Shares to close: {pos.size}',
                         data_feed=d_feed, doprint=True)
                self.order = self.close(data=d_feed)
                return  # 一旦触发卖出，处理完毕

            # 止盈逻辑 (可以简化或与DailyStrategy的TP逻辑对齐，但基于周线参考高点)
            # 例如，TP2 在入场时的周线参考高点
            if not p_info['tp2_hit'] and not pd.isna(p_info['tp2_price']) and p_info['tp2_price'] > 0 and d_feed.high[
                0] >= p_info['tp2_price']:
                shares_to_sell_tp2 = math.floor(p_info['initial_shares'] / 2 / 100) * 100  # 假设卖出一半初始仓位
                shares_to_sell_tp2 = min(shares_to_sell_tp2, p_info['shares_left'])
                if shares_to_sell_tp2 > 0:
                    p_info['sell_reason'] = f"TP2_Weekly at {p_info['tp2_price']:.2f} (H:{d_feed.high[0]:.2f})"
                    self.log(
                        f"SELL ORDER TRIGGER ({stock_name}, Reason: {p_info['sell_reason']}), Shares: {shares_to_sell_tp2}",
                        data_feed=d_feed, doprint=True)
                    self.order = self.sell(data=d_feed, size=shares_to_sell_tp2, exectype=bt.Order.Limit,
                                           price=p_info['tp2_price'])
                    p_info['tp2_hit'] = True;
                    p_info['tp2_sold_shares'] = shares_to_sell_tp2
                    return

    def stop(self):  # 与周线策略类似
        final_value = self.broker.getvalue()
        self.log(f'[WeeklyStrategy] Final Portfolio Value: {final_value:.2f}', doprint=True)

    # Chanlun K线合并等辅助函数需要从 DailyStrategy 或 WeeklyStrategy 复制过来
    # _merge_klines_chanlun, _find_merged_fractal,
    # _identify_all_merged_fractals, _identify_strokes
    # ... (此处省略，请确保这些函数已复制并适配KLineRaw定义)