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
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARN: TA-Lib not found, MACD and ATR related logic will be disabled or use Backtrader's approximation.")

# --- Data Structures ---
KLineRaw = namedtuple('KLineRaw', ['dt', 'o', 'h', 'l', 'c', 'idx'])
MergedKLine = namedtuple('MergedKLine', ['dt', 'o', 'h', 'l', 'c', 'idx', 'direction', 'high_idx', 'low_idx'])
Fractal = namedtuple('Fractal', ['kline', 'm_idx', 'type'])
Stroke = namedtuple('Stroke', ['start_fractal', 'end_fractal', 'direction'])

class MAPullbackPeakCondBtStrategy(bt.Strategy):
    params = (
        ('ma_short', 5),
        ('ma_long', 30), # Daily MA period
        ('ma_peak_threshold', 1.30), # Peak must be >= ma_long_for_peak * threshold
        ('ma_long_for_peak', 30), # MA period used for peak threshold check
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('sell_ma30_pct', 0.03), # Hard stop loss % below daily MA30
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('printlog', True),
        ('max_position_ratio', 0.20),
        ('weekly_ma_period', 30), # Weekly MA period
        ('monthly_ma_period', 30), # Monthly MA period
        ('fractal_lookback', 150), # Bars for K-line merging and stroke analysis
        ('min_bars_between_fractals', 1) # Min merged K-lines between fractals for a stroke
    )

    def log(self, txt, dt=None, doprint=False, data_feed=None):
        """ Logging function w/ specific data feed context """
        if self.params.printlog or doprint:
            data = data_feed or self.datas[0]
            try:
                 dt = dt or data.datetime.date(0)
                 log_prefix = f'{dt.isoformat()} [{data._name}]'
            except (IndexError, AttributeError):
                 log_prefix = f'????-??-?? [{data._name if hasattr(data,"_name") else "Unknown"}]'
            print(f'{log_prefix}, {txt}')

    def __init__(self):
        # --- Indicators and State (per data feed) ---
        self.ma_shorts = {}; self.ma_longs = {}; self.ma_longs_for_peak = {}
        self.atrs = {}; self.macd_objs = {}; self.macd_hists = {}
        self.positions_info = {}; self.atr_stop_loss_prices = {}; self.highest_highs_since_entry = {}
        self.daily_bars_agg = {}; self.synthesized_weekly_data_agg = {}; self.synthesized_monthly_data_agg = {}
        self.weekly_mas = {}; self.last_completed_weekly_ma = {}; self.monthly_mas = {}
        self.current_week_open_dates = {}; self.current_week_data_agg = {}
        self.current_month_open_dates = {}; self.current_month_data_agg = {}

        for d in self.datas:
            dname = d._name
            self.ma_shorts[dname] = btind.SimpleMovingAverage(d.close, period=self.params.ma_short)
            self.ma_longs[dname] = btind.SimpleMovingAverage(d.close, period=self.params.ma_long)
            self.ma_longs_for_peak[dname] = self.ma_longs[dname] if self.params.ma_long_for_peak == self.params.ma_long else btind.SimpleMovingAverage(d.close, period=self.params.ma_long_for_peak)
            self.atrs[dname] = btind.AverageTrueRange(d, period=self.params.atr_period)
            self.positions_info[dname] = {}
            self.atr_stop_loss_prices[dname] = None; self.highest_highs_since_entry[dname] = None
            self.daily_bars_agg[dname] = deque(maxlen=max(self.params.weekly_ma_period * 7, self.params.monthly_ma_period * 31) + 60)
            self.synthesized_weekly_data_agg[dname] = deque(maxlen=self.params.weekly_ma_period + 5); self.synthesized_monthly_data_agg[dname] = deque(maxlen=self.params.monthly_ma_period + 5)
            self.weekly_mas[dname] = None; self.last_completed_weekly_ma[dname] = None; self.monthly_mas[dname] = None
            self.current_week_open_dates[dname] = None; self.current_week_data_agg[dname] = None
            self.current_month_open_dates[dname] = None; self.current_month_data_agg[dname] = None
            if TALIB_AVAILABLE:
                self.macd_objs[dname] = btind.MACD(d.close, period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
                self.macd_hists[dname] = self.macd_objs[dname].macd - self.macd_objs[dname].signal
            else: self.macd_objs[dname] = None; self.macd_hists[dname] = None
        self.order = None; self._pending_buy_info = {}

    # --- Chanlun Helper Methods ---
    def _merge_klines_chanlun(self, bars_data):
        # (Keep as is)
        if len(bars_data) < 2: return bars_data;
        merged = []
        if not bars_data: return merged
        k1_raw = bars_data[0]; direction1 = 1 if k1_raw.c > k1_raw.o else (-1 if k1_raw.c < k1_raw.o else 0)
        if direction1 == 0 and k1_raw.h != k1_raw.l: direction1 = 1 if k1_raw.c >= (k1_raw.h + k1_raw.l) / 2 else -1
        k1 = MergedKLine(k1_raw.dt, k1_raw.o, k1_raw.h, k1_raw.l, k1_raw.c, k1_raw.idx, direction1, k1_raw.idx, k1_raw.idx)
        merged.append(k1); last_unmerged_direction = direction1; i = 1
        while i < len(bars_data):
            k2_raw = bars_data[i]; k1 = merged[-1]
            direction2 = 1 if k2_raw.c > k2_raw.o else (-1 if k2_raw.c < k2_raw.o else 0)
            if direction2 == 0 and k2_raw.h != k2_raw.l: direction2 = 1 if k2_raw.c >= (k2_raw.h + k2_raw.l) / 2 else -1
            k2 = MergedKLine(k2_raw.dt, k2_raw.o, k2_raw.h, k2_raw.l, k2_raw.c, k2_raw.idx, direction2, k2_raw.idx, k2_raw.idx)
            k1_includes_k2 = k1.h >= k2.h and k1.l <= k2.l; k2_includes_k1 = k2.h >= k1.h and k2.l <= k1.l
            if k1_includes_k2 or k2_includes_k1:
                trend_direction = last_unmerged_direction
                merged_h, merged_l = (max(k1.h, k2.h), max(k1.l, k2.l)) if trend_direction >= 0 else (min(k1.h, k2.h), min(k1.l, k2.l))
                high_idx = k1.high_idx if merged_h == k1.h else k2.high_idx; low_idx = k1.low_idx if merged_l == k1.l else k2.low_idx
                merged[-1] = MergedKLine(k2.dt, k1.o, merged_h, merged_l, k2.c, k2.idx, k1.direction, high_idx, low_idx)
            else: merged.append(k2); last_unmerged_direction = k2.direction
            i += 1
        final_merged = []
        for idx, k in enumerate(merged):
            direction = k.direction
            if direction == 0:
                if idx > 0: prev_k = final_merged[-1]; direction = 1 if k.h > prev_k.h and k.l > prev_k.l else (-1 if k.h < prev_k.h and k.l < prev_k.l else prev_k.direction)
                if direction == 0: direction = 1 if k.c >= k.o else -1
            final_merged.append(MergedKLine(*k[:6], direction=direction, high_idx=k.high_idx, low_idx=k.low_idx))
        return final_merged

    def _find_merged_fractal(self, merged_klines, index, fractal_type='top'):
        # (Keep as is)
        if not (1 <= index < len(merged_klines) - 1): return False
        k_prev, k_curr, k_next = merged_klines[index-1], merged_klines[index], merged_klines[index+1]
        if fractal_type == 'top': return k_curr.h > k_prev.h and k_curr.h > k_next.h
        elif fractal_type == 'bottom': return k_curr.l < k_prev.l and k_curr.l < k_next.l
        return False

    def _identify_all_merged_fractals(self, merged_klines):
        # (Keep as is)
        top_fractals, bottom_fractals = [], []
        if len(merged_klines) < 3: return top_fractals, bottom_fractals
        for i in range(1, len(merged_klines) - 1):
            if self._find_merged_fractal(merged_klines, i, fractal_type='top'): top_fractals.append(Fractal(merged_klines[i], i, 'top'))
            elif self._find_merged_fractal(merged_klines, i, fractal_type='bottom'): bottom_fractals.append(Fractal(merged_klines[i], i, 'bottom'))
        return top_fractals, bottom_fractals

    def _identify_strokes(self, merged_klines, top_fractals, bottom_fractals):
        # (Keep as is)
        strokes = []; all_fractals = sorted(top_fractals + bottom_fractals, key=lambda f: f.m_idx)
        last_stroke_end_fractal = None; potential_start_fractal = None
        for i in range(len(all_fractals)):
            current_fractal = all_fractals[i]
            if potential_start_fractal is None:
                 if last_stroke_end_fractal is None or current_fractal.m_idx > last_stroke_end_fractal.m_idx: potential_start_fractal = current_fractal
                 continue
            if current_fractal.type != potential_start_fractal.type:
                if current_fractal.m_idx - potential_start_fractal.m_idx - 1 >= self.params.min_bars_between_fractals:
                    direction = 1 if potential_start_fractal.type == 'bottom' else -1
                    strokes.append(Stroke(potential_start_fractal, current_fractal, direction))
                    last_stroke_end_fractal = current_fractal; potential_start_fractal = current_fractal
                else:
                    if current_fractal.type=='top' and current_fractal.kline.h>=potential_start_fractal.kline.h: potential_start_fractal=current_fractal
                    elif current_fractal.type=='bottom' and current_fractal.kline.l<=potential_start_fractal.kline.l: potential_start_fractal=current_fractal
            else:
                if current_fractal.type=='top' and current_fractal.kline.h>=potential_start_fractal.kline.h: potential_start_fractal=current_fractal
                elif current_fractal.type=='bottom' and current_fractal.kline.l<=potential_start_fractal.kline.l: potential_start_fractal=current_fractal
        return strokes

    # --- MODIFIED: _find_prior_peak_stroke_info to return ORIGINAL peak date ---
    def _find_prior_peak_stroke_info(self, data_feed):
        """ Finds the latest upward stroke end (peak) meeting MA condition.
            Returns info including original O/C/Date of the bar with the highest price. """
        current_idx = len(data_feed) - 1; data_name = data_feed._name
        peak_info = {"price": None, "bar_idx": None, "date": None, # This will be the ORIGINAL date
                     "merged_kline": None, # Keep merged kline info if needed elsewhere
                     "original_peak_open": None, "original_peak_close": None}
        lookback = self.params.fractal_lookback
        if current_idx < lookback + 2: return peak_info
        raw_bars = [];
        try:
            dates=data_feed.datetime.get(ago=-1,size=lookback+1); opens=data_feed.open.get(ago=-1,size=lookback+1); highs=data_feed.high.get(ago=-1,size=lookback+1); lows=data_feed.low.get(ago=-1,size=lookback+1); closes=data_feed.close.get(ago=-1,size=lookback+1)
            for i in range(len(dates)):
                if any(math.isnan(val) for val in [opens[i], highs[i], lows[i], closes[i]]): continue
                raw_bars.append(KLineRaw(bt.num2date(dates[i]).date(), opens[i],highs[i],lows[i],closes[i], current_idx-(len(dates)-1-i)))
        except IndexError: return peak_info
        if len(raw_bars) < 5: return peak_info
        merged_klines = self._merge_klines_chanlun(raw_bars)
        if len(merged_klines) < 5: return peak_info
        top_fractals, bottom_fractals = self._identify_all_merged_fractals(merged_klines)
        strokes = self._identify_strokes(merged_klines, top_fractals, bottom_fractals)
        if not strokes: return peak_info
        ma_line = self.ma_longs_for_peak[data_name].line
        for stroke in reversed(strokes):
            if stroke.direction == 1:
                peak_fractal = stroke.end_fractal;
                original_high_idx = peak_fractal.kline.high_idx # Get index of original high bar
                ago = current_idx - original_high_idx
                if ago >= 0 and ago < len(ma_line):
                     ma_val = ma_line[-ago]
                     if pd.notna(ma_val) and ma_val > 0 and peak_fractal.kline.h >= ma_val * self.params.ma_peak_threshold:
                         # Fetch original O/C and Date using 'ago' derived from original_high_idx
                         original_open = data_feed.open[-ago]
                         original_close = data_feed.close[-ago]
                         original_date = data_feed.datetime.date(-ago) # Fetch original date

                         peak_info["price"] = peak_fractal.kline.h;
                         peak_info["bar_idx"] = original_high_idx;
                         peak_info["date"] = original_date # *** Use original date ***
                         peak_info["merged_kline"] = peak_fractal.kline
                         peak_info["original_peak_open"] = original_open
                         peak_info["original_peak_close"] = original_close
                         self.log(f"Found valid prior peak stroke: Peak Date={peak_info['date']}, H={peak_info['price']:.2f}", data_feed=data_feed, doprint=True) # Log using original date
                         return peak_info
        self.log("No valid prior peak stroke found meeting MA criteria.", data_feed=data_feed, doprint=True)
        return peak_info

    def _find_first_bottom_fractal_after_entry(self, data_feed, entry_bar_idx, current_bar_idx):
        # (Keep previous version with enhanced logging)
        data_name = data_feed._name
        do_specific_log_local = (data_name == '603920' and data_feed.datetime.date(0).isoformat() >= '2024-10-17')
        lookback_start = max(0, entry_bar_idx - 5)
        if current_bar_idx < entry_bar_idx + 2 : return None
        raw_bars = [];
        try:
            num_bars_needed = current_bar_idx - lookback_start + 1
            dates=data_feed.datetime.get(ago=0, size=num_bars_needed); opens=data_feed.open.get(ago=0, size=num_bars_needed); highs=data_feed.high.get(ago=0, size=num_bars_needed); lows=data_feed.low.get(ago=0, size=num_bars_needed); closes=data_feed.close.get(ago=0, size=num_bars_needed)
            for i in range(num_bars_needed):
                if any(math.isnan(v) for v in [opens[i],highs[i],lows[i],closes[i]]): continue
                raw_bars.append(KLineRaw(bt.num2date(dates[i]).date(), opens[i],highs[i],lows[i],closes[i], lookback_start + i ))
        except IndexError: return None
        if len(raw_bars) < 3 : return None
        merged_klines = self._merge_klines_chanlun(raw_bars)
        if len(merged_klines) < 3 : return None
        if do_specific_log_local:
            self.log(f"--- Finding bottom fractal after entry {entry_bar_idx} on {data_feed.datetime.date(0)} ---", data_feed=data_feed, doprint=True)
            self.log(f"Analyzing {len(merged_klines)} merged klines:", data_feed=data_feed, doprint=True)
            for mk in merged_klines[-10:]: self.log(f"  Merged: {mk.dt} O={mk.o:.2f} H={mk.h:.2f} L={mk.l:.2f} C={mk.c:.2f} OrigIdx={mk.idx} LowIdx={mk.low_idx}", data_feed=data_feed, doprint=True)
        for j in range(1, len(merged_klines) - 1):
            fractal_kline = merged_klines[j]
            if fractal_kline.idx > entry_bar_idx:
                is_bottom = self._find_merged_fractal(merged_klines, j, fractal_type='bottom')
                if do_specific_log_local: self.log(f"  Checking merged kline index {j} (Date: {fractal_kline.dt}, OrigIdx: {fractal_kline.idx}) for bottom fractal: {is_bottom}", data_feed=data_feed, doprint=True)
                if is_bottom:
                    body_low = min(fractal_kline.o, fractal_kline.c) if pd.notna(fractal_kline.o) and pd.notna(fractal_kline.c) else np.nan
                    fractal_info = {'date': fractal_kline.dt, 'price': fractal_kline.l, 'body_low': body_low, 'bar_idx': fractal_kline.low_idx, 'merged_kline_o': fractal_kline.o, 'merged_kline_c': fractal_kline.c}
                    self.log(f"Found FIRST bottom fractal after entry: Date={fractal_info['date']}, L={fractal_info['price']:.2f}, BodyL={fractal_info['body_low']:.2f}, OrigBarIdx={fractal_info['bar_idx']}", data_feed=data_feed, doprint=True)
                    return fractal_info
        if do_specific_log_local: self.log(f"No confirmed bottom fractal found AFTER entry bar {entry_bar_idx} yet.", data_feed=data_feed, doprint=True)
        return None

    def _synthesize_higher_tf_data(self, data_feed):
        # (Keep corrected version)
        dname = data_feed._name
        try:
            current_date = data_feed.datetime.date(0); current_dt_obj = data_feed.datetime.datetime(0)
            if len(data_feed.open)==0 or any(math.isnan(x) for x in [data_feed.open[0], data_feed.high[0], data_feed.low[0], data_feed.close[0]]): return
            bar_data = {'date': current_date, 'datetime': current_dt_obj, 'open': data_feed.open[0], 'high': data_feed.high[0], 'low': data_feed.low[0], 'close': data_feed.close[0], 'volume': data_feed.volume[0] if pd.notna(data_feed.volume[0]) else 0.0}
            self.daily_bars_agg[dname].append(bar_data)
            current_iso_week = current_dt_obj.isocalendar(); current_week_num, current_year = current_iso_week[1], current_iso_week[0]
            current_week_data = self.current_week_data_agg.get(dname); current_week_open_date = self.current_week_open_dates.get(dname)
            is_new_week = (current_week_open_date is None or current_week_open_date.isocalendar()[1] != current_week_num or current_week_open_date.year != current_year)
            if is_new_week:
                if current_week_data:
                    self.synthesized_weekly_data_agg[dname].append(current_week_data)
                    synth_completed_weeks = list(self.synthesized_weekly_data_agg[dname])
                    if len(synth_completed_weeks) >= self.params.weekly_ma_period:
                        closes_completed = [b['close'] for b in synth_completed_weeks[-self.params.weekly_ma_period:] if pd.notna(b['close'])]
                        if len(closes_completed) == self.params.weekly_ma_period:
                             self.last_completed_weekly_ma[dname] = sum(closes_completed) / self.params.weekly_ma_period
                self.current_week_open_dates[dname] = current_dt_obj; self.current_week_data_agg[dname] = {**bar_data, 'datetime_start': current_dt_obj, 'datetime_end': current_dt_obj}
            elif current_week_data:
                current_week_data['high']=max(current_week_data['high'], bar_data['high']); current_week_data['low']=min(current_week_data['low'], bar_data['low']); current_week_data['close']=bar_data['close']; current_week_data['volume']+=bar_data['volume']; current_week_data['datetime_end']=current_dt_obj
            synth_weekly_data_list = list(self.synthesized_weekly_data_agg[dname]); current_w_data_dict = self.current_week_data_agg.get(dname)
            if current_w_data_dict: synth_weekly_data_list.append(current_w_data_dict)
            if len(synth_weekly_data_list) >= self.params.weekly_ma_period:
                closes_for_ma = [b['close'] for b in synth_weekly_data_list[-self.params.weekly_ma_period:] if pd.notna(b['close'])]
                if len(closes_for_ma) == self.params.weekly_ma_period: self.weekly_mas[dname]=sum(closes_for_ma)/self.params.weekly_ma_period
                else: self.weekly_mas[dname]=np.nan
            else: self.weekly_mas[dname]=np.nan
            current_month_data = self.current_month_data_agg.get(dname); current_month_open_date = self.current_month_open_dates.get(dname)
            is_new_month = (current_month_open_date is None or current_dt_obj.month != current_month_open_date.month or current_dt_obj.year != current_month_open_date.year)
            if is_new_month:
                 if current_month_data: self.synthesized_monthly_data_agg[dname].append(current_month_data)
                 self.current_month_open_dates[dname] = current_dt_obj; self.current_month_data_agg[dname] = {**bar_data, 'datetime_start': current_dt_obj, 'datetime_end': current_dt_obj}
            elif current_month_data: current_month_data['high']=max(current_month_data['high'], bar_data['high']); current_month_data['low']=min(current_month_data['low'], bar_data['low']); current_month_data['close']=bar_data['close']; current_month_data['volume']+=bar_data['volume']; current_month_data['datetime_end']=current_dt_obj
            synth_monthly_data = list(self.synthesized_monthly_data_agg[dname]); current_m_data = self.current_month_data_agg.get(dname)
            if current_m_data: synth_monthly_data.append(current_m_data)
            if len(synth_monthly_data) >= self.params.monthly_ma_period:
                closes = [b['close'] for b in synth_monthly_data[-self.params.monthly_ma_period:] if pd.notna(b['close'])]
                if len(closes)==self.params.monthly_ma_period: self.monthly_mas[dname]=sum(closes)/self.params.monthly_ma_period
                else: self.monthly_mas[dname]=np.nan
            else: self.monthly_mas[dname]=np.nan
        except Exception as e: self.log(f"Error synthesizing TFs for {dname}: {e}", data_feed=data_feed)


    # --- MODIFIED: notify_order uses original O/C for body high ---
    def notify_order(self, order):
        data_feed = order.data; stock_name = data_feed._name
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy():
                entry_price=order.executed.price; executed_size=order.executed.size; entry_bar_idx = len(data_feed)-1
                self.log(f'BUY EXECUTED, Price: {entry_price:.2f}, Size: {executed_size}', data_feed=data_feed)
                pos_info = self.positions_info[stock_name]
                if hasattr(self, '_pending_buy_info') and stock_name in self._pending_buy_info:
                    peak_data = self._pending_buy_info[stock_name]
                    prior_peak_price = peak_data.get('price')
                    # Use the original date from peak_info now
                    prior_peak_date = peak_data.get('date', 'N/A')
                    original_peak_open = peak_data.get('original_peak_open')
                    original_peak_close = peak_data.get('original_peak_close')
                    prior_peak_body_high = np.nan
                    if pd.notna(original_peak_open) and pd.notna(original_peak_close):
                         prior_peak_body_high = max(original_peak_open, original_peak_close)
                         self.log(f"   Peak Info for TP1: Orig Peak Date={prior_peak_date}, Orig Idx={peak_data.get('bar_idx')}, Orig O={original_peak_open:.2f}, Orig C={original_peak_close:.2f} => BodyH={prior_peak_body_high:.2f}", data_feed=data_feed, doprint=True)
                    else:
                         self.log(f"   Peak Info for TP1: Original O/C missing. Peak Price={prior_peak_price}", data_feed=data_feed, doprint=True)

                    pos_info['prior_peak_body_high_for_tp1'] = prior_peak_body_high

                    current_total_shares = self.getposition(data_feed).size
                    if 'initial_shares' in pos_info and pos_info['initial_shares']>0: #Add
                        old_avg=pos_info['entry_price']; old_shares=pos_info['shares_held_before_add']
                        pos_info['entry_price']=(old_avg*old_shares+entry_price*executed_size)/current_total_shares if current_total_shares>0 else entry_price
                    else: #New
                        pos_info['entry_price']=entry_price; pos_info['initial_shares']=executed_size
                    pos_info.update({
                        'shares_held': current_total_shares, 'shares_held_before_add': current_total_shares,
                        'prior_peak_price_for_tp1': prior_peak_price,
                        # 'prior_peak_body_high_for_tp1': prior_peak_body_high, # Assigned above
                        'tp1': np.nan, 'tp2': prior_peak_price if pd.notna(prior_peak_price) else np.nan, 'tp3': np.nan,
                        'entry_bar_idx': entry_bar_idx, 'entry_date': data_feed.datetime.date(0),
                        'take_profit_targets_hit': [False,False,False], 'shares_sold_tp1':0, 'shares_sold_tp2':0,
                        'post_entry_bottom_fractal': None, 'post_entry_bottom_fractal_found': False })
                    self.highest_highs_since_entry[stock_name]=data_feed.high[0] if pd.notna(data_feed.high[0]) else entry_price
                    atr=self.atrs[stock_name][-1] if TALIB_AVAILABLE and len(self.atrs[stock_name])>0 and pd.notna(self.atrs[stock_name][-1]) else np.nan
                    if pd.notna(atr) and pd.notna(pos_info['entry_price']):
                        self.atr_stop_loss_prices[stock_name]=pos_info['entry_price']-self.params.atr_multiplier*atr
                        pos_info['atr_stop_loss_price']=self.atr_stop_loss_prices[stock_name]
                        self.log(f'   Initial ATR SL: {self.atr_stop_loss_prices[stock_name]:.2f}', data_feed=data_feed)
                    if stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]
                else: self.log(f'ERROR: BUY EXECUTED for {stock_name} but no pending peak info!', data_feed=data_feed)
            elif order.issell():
                 self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {abs(order.executed.size)}', data_feed=data_feed)
                 pos_info = self.positions_info.get(stock_name, {})
                 if pos_info: pos_info['shares_held'] = self.getposition(data_feed).size
                 if hasattr(order, '_sell_reason'):
                     self.log(f'   Reason: {order._sell_reason}', data_feed=data_feed)
                     if order._sell_reason.startswith("take_profit") and pos_info.get('post_entry_bottom_fractal'):
                          bf = pos_info['post_entry_bottom_fractal']; bf_p=f"{bf['price']:.2f}"; bf_bl=f"{bf['body_low']:.2f}" if pd.notna(bf['body_low']) else "N/A"
                          self.log(f"   TP Ref Post-Entry Bottom Fractal: L={bf_p}, BodyL={bf_bl} on {bf['date']}", data_feed=data_feed)
                     if pos_info:
                         sold_size=abs(order.executed.size)
                         if order._sell_reason=="take_profit_1": pos_info['take_profit_targets_hit'][0]=True; pos_info['shares_sold_tp1']=sold_size
                         elif order._sell_reason=="take_profit_2": pos_info['take_profit_targets_hit'][1]=True; pos_info['shares_sold_tp2']=sold_size
                         elif order._sell_reason=="take_profit_3": pos_info['take_profit_targets_hit'][2]=True
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order Failed for {stock_name}', data_feed=data_feed)
            if order.isbuy() and hasattr(self, '_pending_buy_info') and stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]
        self.order = None

    def notify_trade(self, trade):
        # (Keep as previous version)
        stock_name = trade.data._name
        if not trade.isclosed: return
        self.log(f'TRADE CLOSED for {stock_name}, PNL NET {trade.pnlcomm:.2f}', data_feed=trade.data)
        self.positions_info[stock_name] = {}; self.atr_stop_loss_prices[stock_name] = None; self.highest_highs_since_entry[stock_name] = None

    def next(self):
        # (Keep as previous version, including logging)
        if not hasattr(self, '_pending_buy_info'): self._pending_buy_info = {}

        for i, d in enumerate(self.datas):
            stock_name = d._name; current_idx = len(d) - 1
            min_bars_required = self.params.fractal_lookback + 5
            if current_idx + 1 < min_bars_required: continue

            self._synthesize_higher_tf_data(d)
            current_date=d.datetime.date(0)
            do_specific_log = (stock_name == '603920' and current_date.isoformat() >= '2024-10-15')
            if do_specific_log: self.log(f"--- Debug {stock_name} on {current_date} ---", data_feed=d, doprint=True)
            if self.order and self.order.alive(): continue

            current_position = self.getposition(d)
            pos_info = self.positions_info[stock_name]

            if current_position: # --- Position Management ---
                if not pos_info or 'entry_price' not in pos_info: continue
                if not pos_info.get('post_entry_bottom_fractal_found', False):
                    entry_bar_idx = pos_info.get('entry_bar_idx')
                    if entry_bar_idx is not None and current_idx >= entry_bar_idx + 2:
                        bf_info = self._find_first_bottom_fractal_after_entry(d, entry_bar_idx, current_idx)
                        if bf_info:
                            self.positions_info[stock_name]['post_entry_bottom_fractal'] = bf_info
                            self.positions_info[stock_name]['post_entry_bottom_fractal_found'] = True
                            peak_body_h = pos_info.get('prior_peak_body_high_for_tp1')
                            bottom_body_l = bf_info.get('body_low')
                            peak_body_h_str = f"{peak_body_h:.2f}" if pd.notna(peak_body_h) else "NaN"; bottom_body_l_str = f"{bottom_body_l:.2f}" if pd.notna(bottom_body_l) else "NaN"
                            self.log(f"   Calculating TP1: PeakBodyH={peak_body_h_str}, BottomBodyL={bottom_body_l_str}", data_feed=d, doprint=True)
                            if pd.notna(peak_body_h) and pd.notna(bottom_body_l):
                                tp1_calculated = (peak_body_h + bottom_body_l) / 2
                                self.positions_info[stock_name]['tp1'] = tp1_calculated
                                self.log(f"   TP1 calculated and stored: {tp1_calculated:.2f}", data_feed=d, doprint=True)
                            else: self.log(f"   TP1 calculation failed: Inputs invalid.", data_feed=d, doprint=True)

                # Stop Loss & Take Profit Logic (paste detailed logic from previous versions here)
                sell_reason=None; current_close=d.close[0]; current_low=d.low[0]; current_high=d.high[0]
                ma30=self.ma_longs[stock_name][0]; ma5=self.ma_shorts[stock_name][0]
                current_sl = self.atr_stop_loss_prices.get(stock_name)
                if pd.notna(ma30) and pd.notna(current_close) and current_close < ma30*(1-self.params.sell_ma30_pct): sell_reason="stop_loss_ma30_hard"
                if not sell_reason: # ATR Stop
                    highest_h = self.highest_highs_since_entry[stock_name]
                    if pd.notna(current_high): highest_h = max(highest_h, current_high) if highest_h is not None else current_high; self.highest_highs_since_entry[stock_name]=highest_h
                    atr_val = self.atrs[stock_name][0] if TALIB_AVAILABLE and len(self.atrs[stock_name])>0 and pd.notna(self.atrs[stock_name][0]) else np.nan
                    if pd.notna(atr_val) and highest_h is not None:
                        new_sl = highest_h - self.params.atr_multiplier * atr_val
                        bf = pos_info.get('post_entry_bottom_fractal')
                        if bf and pd.notna(bf['price']): new_sl = max(new_sl, bf['price'] - atr_val*0.1)
                        if current_sl is None or new_sl > current_sl: self.atr_stop_loss_prices[stock_name]=new_sl; pos_info['atr_stop_loss_price']=new_sl; current_sl=new_sl
                    if current_sl is not None and pd.notna(current_low) and current_low < current_sl: sell_reason="stop_loss_atr"
                if not sell_reason: # Dead Cross
                    ma5l=self.ma_shorts[stock_name]; ma30l=self.ma_longs[stock_name]
                    if pd.notna(ma5)and pd.notna(ma30)and len(ma5l)>1 and len(ma30l)>1 and pd.notna(ma5l[-1])and pd.notna(ma30l[-1]) and ma5l[-1]>=ma30l[-1] and ma5<ma30: sell_reason="stop_loss_ma_dead_cross"
                if sell_reason: self.log(f'SELL ORDER (SL {sell_reason}) {stock_name}', data_feed=d); self.order=self.close(data=d); self.order._sell_reason=sell_reason; continue

                if not sell_reason and current_position.size > 0: # Take Profit
                    tp1=pos_info.get('tp1'); tp2=pos_info.get('tp2'); tp3=pos_info.get('tp3')
                    hit_flags=pos_info.get('take_profit_targets_hit',[False,False,False])
                    initial_shares=pos_info.get('initial_shares',0); current_size=current_position.size
                    if pd.notna(current_high):
                        tp1_str = f"{tp1:.2f}" if pd.notna(tp1) else "NaN"; tp1_check_met = pd.notna(tp1) and current_high >= tp1
                        if do_specific_log: self.log(f"TP1 Check: High={current_high:.2f}, Target={tp1_str}, Hit? {tp1_check_met}", data_feed=d, doprint=True)
                        if not hit_flags[0] and tp1_check_met:
                            size_sell=min(math.floor((initial_shares/3.0)/100.0)*100,current_size) if initial_shares>0 else 0
                            if size_sell>0: self.log(f'SELL ORDER (TP1) {stock_name} at {tp1:.2f}',data_feed=d); self.order=self.sell(data=d,size=size_sell); self.order._sell_reason="take_profit_1"; continue
                        elif not hit_flags[1] and pd.notna(tp2) and current_high >= tp2:
                            size_sell=min(math.floor((current_size/2.0)/100.0)*100,current_size)
                            if size_sell>0: self.log(f'SELL ORDER (TP2) {stock_name} at {tp2:.2f}',data_feed=d); self.order=self.sell(data=d,size=size_sell); self.order._sell_reason="take_profit_2"; continue
                        elif not hit_flags[2] and pd.notna(tp3) and current_high >= tp3:
                            if current_size>0: self.log(f'SELL ORDER (TP3) {stock_name} at {tp3:.2f}',data_feed=d); self.order=self.close(data=d); self.order._sell_reason="take_profit_3"; continue
                pass # MACD Placeholder

            else: # --- Entry Logic ---
                weekly_ma=self.weekly_mas.get(stock_name); last_completed_ma=self.last_completed_weekly_ma.get(stock_name)
                weekly_ok = (pd.notna(weekly_ma) and (pd.notna(last_completed_ma) and round(weekly_ma,2)>=round(last_completed_ma,2))) or (pd.notna(weekly_ma) and last_completed_ma is None)
                if not weekly_ok:
                    if do_specific_log: self.log("Entry Fail: Weekly Trend", data_feed=d, doprint=True); continue
                ma30=self.ma_longs[stock_name][0]; ma5=self.ma_shorts[stock_name][0]
                if pd.isna(ma30) or pd.isna(ma5):
                    if do_specific_log: self.log("Entry Fail: Daily MA NaN", data_feed=d, doprint=True); continue

                prior_peak_info = self._find_prior_peak_stroke_info(d)
                if prior_peak_info.get("price") is None:
                    if do_specific_log: self.log(f"Entry Fail: No valid prior peak stroke", data_feed=d, doprint=True);
                    continue

                buy_signal_type = None; current_low=d.low[0]; current_close=d.close[0]
                if any(pd.isna(v) for v in [current_low, current_close]): continue
                cond_daily_pullback = (current_low < ma30 * 1.05 and current_low > ma30 * 0.97)
                cond_close_above_weekly_ma30 = pd.notna(weekly_ma) and current_close > weekly_ma
                cond_mas_bullish = ma5 > ma30

                if do_specific_log:
                    wma_str = f"{weekly_ma:.4f}" if pd.notna(weekly_ma) else "NaN"
                    self.log(f"Daily Trigger Check: L={current_low:.4f}, C={current_close:.4f}, MA5={ma5:.4f}, DailyMA30={ma30:.4f}, WeeklyMA30={wma_str}", data_feed=d, doprint=True)
                    self.log(f"  Pullback? {cond_daily_pullback}", data_feed=d, doprint=True)
                    self.log(f"  C > WeeklyMA? {cond_close_above_weekly_ma30}", data_feed=d, doprint=True)
                    self.log(f"  MA5 > DailyMA30? {cond_mas_bullish}", data_feed=d, doprint=True)

                if cond_daily_pullback and cond_close_above_weekly_ma30 and cond_mas_bullish and ma30 > 1e-6:
                    buy_signal_type = "Daily Pullback & Close > Weekly MA30"
                    if do_specific_log: self.log(f"BUY SIGNAL MET: {buy_signal_type}", data_feed=d, doprint=True)

                if buy_signal_type: # Execute Buy
                    current_close_price = d.close[0]
                    if pd.isna(current_close_price) or current_close_price <= 0: continue
                    self._pending_buy_info[stock_name] = prior_peak_info
                    cash=self.broker.get_cash(); value=self.broker.getvalue()
                    target_value=value*self.params.max_position_ratio; value_to_invest=target_value; size=0
                    if value_to_invest>0 and current_close_price > 0:
                        shares_val=math.floor((value_to_invest/current_close_price)/100.0)*100
                        shares_cash=math.floor((cash/current_close_price)/100.0)*100 if cash>0 else 0
                        size=min(shares_val, shares_cash)
                    if size > 0:
                        # Use the original peak date in the log message
                        self.log(f'BUY ORDER ({buy_signal_type}) based on Peak {prior_peak_info.get("date")}. Size:{size}', data_feed=d)
                        self.order = self.buy(data=d, size=size)
                        return
                    else:
                        self.log(f'Buy signal {buy_signal_type}, but size=0.', data_feed=d)
                        if stock_name in self._pending_buy_info: del self._pending_buy_info[stock_name]
                elif do_specific_log:
                    self.log(f"No buy trigger conditions met for {stock_name}", data_feed=d, doprint=True)
    pass # End of Class