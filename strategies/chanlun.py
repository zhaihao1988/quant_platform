# strategies/chanlun.py
import pandas as pd
import numpy as np
import logging  # Added for better logging
from sqlalchemy import create_engine, text  # Added for DB interaction
import mplfinance as mpf  # Added for plotting
from datetime import datetime  # Added for date handling
import matplotlib.pyplot as plt # Added for advanced plotting (strokes, pivots)
import matplotlib.dates as mdates # Added for date conversion for plotting
from matplotlib.patches import Rectangle # Added for drawing pivots

# Existing imports from the original file
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    print("Warning: TA-Lib library not found. MACD divergence detection will be skipped.")
    TALIB_AVAILABLE = False

# Import database engine and models (adjust path if necessary based on your project structure)
# Assuming chanlun.py is in 'strategies' and database.py is in 'db' at the same root level
import sys
import os

# Add the parent directory to sys.path to allow for absolute imports
# This assumes your script is run from a context where 'db' and 'config' are importable
# For example, if your project root is 'quant_platform', and you run from 'quant_platform'
# Or if 'quant_platform' is in PYTHONPATH
# Correcting import paths relative to the project structure
# Assuming the script is run from the root of the 'quant_platform' project
# or that 'quant_platform' is in PYTHONPATH
try:
    script_dir_chanlun = os.path.dirname(os.path.abspath(__file__))
    project_root_chanlun = os.path.dirname(script_dir_chanlun)
    if project_root_chanlun not in sys.path:
        sys.path.insert(0, project_root_chanlun)
    from db.database import get_engine_instance  # Preferred way to get engine
    from db.models import StockDaily  # To know the table structure, though we'll use raw SQL for simplicity here
    from config.settings import settings  # To get DB_URL if needed, though get_engine_instance handles it
except ImportError as e:
    print(f"Error importing project modules: {e}. Make sure PYTHONPATH is set correctly or run from project root.")
    # Fallback for direct execution if imports fail, expecting modules in relative paths
    # This might be needed if the script is run directly from the 'strategies' folder
    # and the parent folder is not automatically in path.
    # However, using get_engine_instance is cleaner if project structure is well-defined.
    # For this modification, we will rely on get_engine_instance being available.
    # If direct execution of this script is intended, you might need to add:
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # from db.database import get_engine_instance
    # from db.models import StockDaily
    # from config.settings import settings
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Helper class for KLine, Fractal, Stroke, Segment, Pivot ( 그대로 유지 )
class KLine:
    def __init__(self, date, open_price, high, low, close, index_in_df):
        self.date = date
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.index_in_df = index_in_df  # Original index in the dataframe
        self.direction = 1 if close > open_price else (-1 if close < open_price else 0)


class Fractal:
    def __init__(self, kline, fractal_type, index_in_processed_k):  # "top" or "bottom"
        self.kline = kline  # The middle KLine of the fractal
        self.fractal_type = fractal_type
        self.price = kline.high if fractal_type == "top" else kline.low
        self.date = kline.date
        self.index_in_processed_k = index_in_processed_k


class Stroke:
    def __init__(self, start_fractal, end_fractal, processed_k_lines):
        self.start_fractal = start_fractal
        self.end_fractal = end_fractal
        self.start_date = start_fractal.date
        self.end_date = end_fractal.date
        self.start_price = start_fractal.price
        self.end_price = end_fractal.price
        self.type = "up" if start_fractal.fractal_type == "bottom" else "down"
        start_idx = max(0, start_fractal.index_in_processed_k)
        end_idx = min(len(processed_k_lines), end_fractal.index_in_processed_k + 1)
        if start_idx >= end_idx:
            self.high = max(start_fractal.kline.high, end_fractal.kline.high)
            self.low = min(start_fractal.kline.low, end_fractal.kline.low)
        else:
            relevant_klines = processed_k_lines[start_idx:end_idx]
            self.high = max(k.high for k in relevant_klines) if relevant_klines else max(start_fractal.kline.high,
                                                                                         end_fractal.kline.high)
            self.low = min(k.low for k in relevant_klines) if relevant_klines else min(start_fractal.kline.low,
                                                                                       end_fractal.kline.low)
        self.start_kline_original_idx = start_fractal.kline.index_in_df
        self.end_kline_original_idx = end_fractal.kline.index_in_df


class Segment:
    def __init__(self, strokes, segment_type):  # segment_type: "up" or "down"
        if not strokes:
            raise ValueError("Cannot create a segment with zero strokes.")
        self.strokes = strokes
        self.segment_type = segment_type
        self.start_date = strokes[0].start_date
        self.end_date = strokes[-1].end_date
        self.start_price = strokes[0].start_price
        self.end_price = strokes[-1].end_price
        self.high = max(s.high for s in strokes)
        self.low = min(s.low for s in strokes)
        self.start_kline_original_idx = strokes[0].start_kline_original_idx
        self.end_kline_original_idx = strokes[-1].end_kline_original_idx


class Pivot:  # Zhongshu
    def __init__(self, strokes_in_pivot, zg, zd, start_date, end_date):
        if not strokes_in_pivot:
            raise ValueError("Cannot create a pivot with zero strokes.")
        self.strokes_in_pivot = strokes_in_pivot
        self.zg = zg  # Pivot High
        self.zd = zd  # Pivot Low
        self.start_date = start_date
        self.end_date = end_date
        self.start_kline_original_idx = strokes_in_pivot[0].start_kline_original_idx
        self.end_kline_original_idx = strokes_in_pivot[-1].end_kline_original_idx


### A. K线预处理函数 (`preprocess_k_lines`) - 그대로 유지
def preprocess_k_lines(df_raw):
    processed_k_lines = []
    if df_raw is None or len(df_raw) < 1:
        print("Warning: Empty or None DataFrame passed to preprocess_k_lines.")
        return processed_k_lines
    expected_cols = {'open', 'high', 'low', 'close'}
    if not expected_cols.issubset(df_raw.columns):
        print(f"Error: Input DataFrame missing required columns. Expected: {expected_cols}, Got: {df_raw.columns}")
        return processed_k_lines
    klines_raw_obj = []
    for idx, row in enumerate(df_raw.itertuples()):
        if pd.isna(row.open) or pd.isna(row.high) or pd.isna(row.low) or pd.isna(row.close):
            print(f"Warning: Skipping row index {idx} due to NaN values.")
            continue
        klines_raw_obj.append(KLine(row.Index, row.open, row.high, row.low, row.close, idx))
    if not klines_raw_obj:
        print("Warning: No valid KLine objects created from DataFrame.")
        return processed_k_lines
    merged_klines = []
    if not klines_raw_obj: return merged_klines
    k_prev = klines_raw_obj[0]
    for i in range(1, len(klines_raw_obj)):
        k_curr = klines_raw_obj[i]
        if k_curr.high <= k_prev.high and k_curr.low >= k_prev.low:
            direction = k_prev.direction
            if direction == 0 and merged_klines: direction = merged_klines[-1].direction
            if direction == 0: direction = 1 if k_prev.close >= klines_raw_obj[i - 1].close else -1
            if direction >= 0:
                k_prev = KLine(k_prev.date, k_prev.open, max(k_prev.high, k_curr.high), max(k_prev.low, k_curr.low),
                               k_curr.close, k_prev.index_in_df)
            else:
                k_prev = KLine(k_prev.date, k_prev.open, min(k_prev.high, k_curr.high), min(k_prev.low, k_curr.low),
                               k_curr.close, k_prev.index_in_df)
        elif k_prev.high <= k_curr.high and k_prev.low >= k_curr.low:
            direction = k_curr.direction
            if direction == 0: direction = k_prev.direction
            if direction == 0 and merged_klines: direction = merged_klines[-1].direction
            if direction == 0: direction = 1 if k_curr.close >= k_prev.close else -1
            if direction >= 0:
                k_prev = KLine(k_prev.date, k_prev.open, max(k_prev.high, k_curr.high), max(k_prev.low, k_curr.low),
                               k_curr.close, k_prev.index_in_df)
            else:
                k_prev = KLine(k_prev.date, k_prev.open, min(k_prev.high, k_curr.high), min(k_prev.low, k_curr.low),
                               k_curr.close, k_prev.index_in_df)
        else:
            merged_klines.append(k_prev)
            k_prev = k_curr
    merged_klines.append(k_prev)
    # print(f"Preprocessed K-lines: {len(merged_klines)}") # Reduced verbosity
    return merged_klines


### B. 分型识别函数 (`identify_fractals`) - 그대로 유지
def identify_fractals(processed_k_lines):
    fractals = []
    if len(processed_k_lines) < 3: return fractals
    for i in range(1, len(processed_k_lines) - 1):
        k1, k2, k3 = processed_k_lines[i - 1], processed_k_lines[i], processed_k_lines[i + 1]
        is_top_fractal = k2.high >= k1.high and k2.high >= k3.high and k2.low >= k1.low and k2.low >= k3.low
        if is_top_fractal and (k2.high > k1.high or k2.high > k3.high or (k2.high == k1.high and k2.low > k1.low) or (
                k2.high == k3.high and k2.low > k3.low)):
            is_valid = True
            if fractals and fractals[-1].fractal_type == "top":
                if abs(i - fractals[-1].index_in_processed_k) < 3:
                    if k2.high > fractals[-1].price: fractals[-1] = Fractal(k2, "top", i)
                    is_valid = False
            if is_valid: fractals.append(Fractal(k2, "top", i))
        is_bottom_fractal = k2.low <= k1.low and k2.low <= k3.low and k2.high <= k1.high and k2.high <= k3.high
        if is_bottom_fractal and (k2.low < k1.low or k2.low < k3.low or (k2.low == k1.low and k2.high < k1.high) or (
                k2.low == k3.low and k2.high < k3.high)):
            is_valid = True
            if fractals and fractals[-1].fractal_type == "bottom":
                if abs(i - fractals[-1].index_in_processed_k) < 3:
                    if k2.low < fractals[-1].price: fractals[-1] = Fractal(k2, "bottom", i)
                    is_valid = False
            if is_valid: fractals.append(Fractal(k2, "bottom", i))
    # print(f"Identified fractals: {len(fractals)}") # Reduced verbosity
    return fractals


### C. 笔构建函数 (`construct_strokes`) - 그대로 유지
def construct_strokes(fractals, processed_k_lines):
    strokes = []
    if len(fractals) < 2: return strokes
    last_confirmed_stroke_end_fractal = None
    potential_stroke_start_fractal = None
    f_idx = 0
    while f_idx < len(fractals):
        current_fractal = fractals[f_idx]
        if potential_stroke_start_fractal is None:
            potential_stroke_start_fractal = current_fractal
            f_idx += 1
            continue
        if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
            if (
                    current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
                    (
                            current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                potential_stroke_start_fractal = current_fractal
            f_idx += 1
            continue
        k_line_distance = abs(
            current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k)
        if k_line_distance < 3:
            potential_stroke_start_fractal = current_fractal
            f_idx += 1
            continue
        valid_stroke_integrity = False
        if potential_stroke_start_fractal.fractal_type == "bottom" and current_fractal.fractal_type == "top":
            if current_fractal.price > potential_stroke_start_fractal.price: valid_stroke_integrity = True
        elif potential_stroke_start_fractal.fractal_type == "top" and current_fractal.fractal_type == "bottom":
            if current_fractal.price < potential_stroke_start_fractal.price: valid_stroke_integrity = True
        if valid_stroke_integrity:
            new_stroke = Stroke(potential_stroke_start_fractal, current_fractal, processed_k_lines)
            if not strokes or \
                    (new_stroke.start_fractal.index_in_processed_k == strokes[-1].end_fractal.index_in_processed_k and \
                     new_stroke.type != strokes[-1].type):
                strokes.append(new_stroke)
                last_confirmed_stroke_end_fractal = new_stroke.end_fractal
                potential_stroke_start_fractal = new_stroke.end_fractal
            else:
                last_stroke = strokes[-1]
                if current_fractal.fractal_type == last_stroke.end_fractal.fractal_type:
                    if (
                            current_fractal.fractal_type == "top" and current_fractal.price > last_stroke.end_fractal.price) or \
                            (
                                    current_fractal.fractal_type == "bottom" and current_fractal.price < last_stroke.end_fractal.price):
                        strokes.pop()
                        # More complex rule might backtrack, here we simplify: retry from the previous stroke's start fractal
                        potential_stroke_start_fractal = last_stroke.start_fractal
                        f_idx -=1 # Re-evaluate current fractal against potentially new start
                        continue # Need to restart check with the new potential start
                    else:
                        potential_stroke_start_fractal = current_fractal
                else:
                    potential_stroke_start_fractal = current_fractal
            f_idx += 1
        else:
            if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
                if (
                        current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
                        (
                                current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                    potential_stroke_start_fractal = current_fractal
            else:
                potential_stroke_start_fractal = current_fractal
            f_idx += 1
    # print(f"Constructed strokes (initial): {len(strokes)}") # Reduced verbosity
    return strokes


### D. 线段构建函数 (`construct_line_segments`) - 그대로 유지
def construct_line_segments(strokes):
    segments = []
    if len(strokes) < 3: return segments
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
        if s1.type == s3.type and s1.type != s2.type:
            overlap_exists = max(s1.low, s3.low) < min(s1.high, s3.high)
            segment_break = False
            if s1.type == "up" and s3.end_price > s1.end_price:
                segment_break = True
            elif s1.type == "down" and s3.end_price < s1.end_price:
                segment_break = True
            pullback_valid = False
            if s1.type == "up" and s2.end_price > s1.start_price:
                pullback_valid = True
            elif s1.type == "down" and s2.end_price < s1.start_price:
                pullback_valid = True
            if overlap_exists and segment_break and pullback_valid:
                current_segment_strokes = [s1, s2, s3]
                segment_type = s1.type
                j = i + 3
                while j < len(strokes) - 1:
                    s_next_opposite, s_next_same_dir = strokes[j], strokes[j + 1]
                    if s_next_opposite.type != segment_type and s_next_same_dir.type == segment_type:
                        last_seg_stroke = current_segment_strokes[-1]
                        extend_break = (
                                                   segment_type == "up" and s_next_same_dir.end_price > last_seg_stroke.end_price) or \
                                       (
                                                   segment_type == "down" and s_next_same_dir.end_price < last_seg_stroke.end_price)
                        extend_pullback_valid = (
                                                            segment_type == "up" and s_next_opposite.end_price > last_seg_stroke.start_price) or \
                                                (
                                                            segment_type == "down" and s_next_opposite.end_price < last_seg_stroke.start_price)
                        if extend_break and extend_pullback_valid:
                            current_segment_strokes.extend([s_next_opposite, s_next_same_dir])
                            j += 2
                        else:
                            break
                    else:
                        break
                segments.append(Segment(current_segment_strokes, segment_type))
                i = j
                continue
        i += 1
    # print(f"Constructed segments: {len(segments)}") # Reduced verbosity
    return segments


### E. 中枢识别函数 (`identify_pivots`) - 그대로 유지
def identify_pivots(strokes):
    pivots = []
    if len(strokes) < 3: return pivots
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
        if s1.type != s2.type and s2.type != s3.type:
            if s1.type == "up":
                zg_candidate, zd_candidate = s2.high, max(s1.low, s3.low)
            else:
                zg_candidate, zd_candidate = min(s1.high, s3.high), s2.low
            if zd_candidate < zg_candidate:
                current_pivot_strokes = [s1, s2, s3]
                current_zd, current_zg = zd_candidate, zg_candidate
                start_date, end_date = s1.start_date, s3.end_date
                j = i + 3
                while j < len(strokes):
                    next_stroke = strokes[j]
                    if next_stroke.type != current_pivot_strokes[-1].type:
                        interacts = (next_stroke.type == "up" and next_stroke.high >= current_zd) or \
                                    (next_stroke.type == "down" and next_stroke.low <= current_zg)
                        if interacts:
                            temp_strokes = current_pivot_strokes + [next_stroke]
                            up_stroke_lows = [s.low for s in temp_strokes if s.type == "up"]
                            down_stroke_highs = [s.high for s in temp_strokes if s.type == "down"]
                            if not up_stroke_lows or not down_stroke_highs: break
                            new_zd, new_zg = max(up_stroke_lows), min(down_stroke_highs)
                            if new_zd < new_zg:
                                current_pivot_strokes.append(next_stroke)
                                current_zd, current_zg = new_zd, new_zg
                                end_date = next_stroke.end_date
                                j += 1
                            else:
                                break
                        else:
                            break
                    else:
                        break
                if len(current_pivot_strokes) >= 3:
                    pivots.append(Pivot(list(current_pivot_strokes), current_zg, current_zd, start_date, end_date))
                    i += len(current_pivot_strokes)
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    # print(f"Identified pivots: {len(pivots)}") # Reduced verbosity
    return pivots


### F. 背驰检测函数 (`detect_divergence_macd`) - 그대로 유지
def detect_divergence_macd(df_with_macd, strokes_or_segments, pivots):
    divergences = []
    if not TALIB_AVAILABLE:
        print("Skipping MACD divergence detection as TA-Lib is not available.")
        return divergences
    if df_with_macd is None or df_with_macd.empty or strokes_or_segments is None or len(strokes_or_segments) < 2:
        return divergences
    if 'macdhist' not in df_with_macd.columns:
        close_prices = df_with_macd['close'].astype(float)
        if len(close_prices) < 34:
            print("Warning: Not enough data for MACD calculation in divergence check.")
            return divergences
        try:
            macd, macdsignal, macdhist = talib.MACD(close_prices.values, fastperiod=12, slowperiod=26, signalperiod=9)
            df_with_macd = df_with_macd.copy()
            df_with_macd['macd'], df_with_macd['macdsignal'], df_with_macd['macdhist'] = macd, macdsignal, macdhist
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return divergences
    for i in range(len(strokes_or_segments) - 2):
        item1, item2, item_mid = strokes_or_segments[i], strokes_or_segments[i + 2], strokes_or_segments[i + 1]
        if item1.type == item2.type and item1.type != item_mid.type:
            try:
                idx1_start, idx1_end = max(0, item1.start_kline_original_idx), min(len(df_with_macd),
                                                                                   item1.end_kline_original_idx + 1)
                idx2_start, idx2_end = max(0, item2.start_kline_original_idx), min(len(df_with_macd),
                                                                                   item2.end_kline_original_idx + 1)
                if idx1_start >= idx1_end or idx2_start >= idx2_end: continue
                macd_hist_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                macd_hist_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()
                if pd.isna(macd_hist_sum1) or pd.isna(macd_hist_sum2): continue
            except Exception as e:
                print(f"Error accessing MACD data for divergence: {e}")
                continue
            if item1.type == "up":
                if item2.end_price > item1.end_price and macd_hist_sum2 < macd_hist_sum1:
                    divergences.append({"date": item2.end_date, "type": "trend_top_divergence",
                                        "level": "stroke" if isinstance(item1, Stroke) else "segment",
                                        "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                                        "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2})
            elif item1.type == "down":
                if item2.end_price < item1.end_price and macd_hist_sum2 > macd_hist_sum1:
                    divergences.append({"date": item2.end_date, "type": "trend_bottom_divergence",
                                        "level": "stroke" if isinstance(item1, Stroke) else "segment",
                                        "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                                        "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2})
    if pivots:
        for pivot in pivots:
            departures_down, departures_up = [], []
            strokes_after_pivot = [s for s in strokes_or_segments if s.start_date >= pivot.end_date]
            for stroke in strokes_after_pivot:
                if stroke.type == "down" and stroke.end_price < pivot.zd:
                    departures_down.append(stroke)
                elif stroke.type == "up" and stroke.end_price > pivot.zg:
                    departures_up.append(stroke)
                if (departures_down and stroke.type == "up") or (departures_up and stroke.type == "down"): break
            if len(departures_down) >= 2:
                s1, s2 = departures_down[0], departures_down[1]
                if s2.end_price < s1.end_price:
                    try:
                        idx1_start, idx1_end = max(0, s1.start_kline_original_idx), min(len(df_with_macd),
                                                                                        s1.end_kline_original_idx + 1)
                        idx2_start, idx2_end = max(0, s2.start_kline_original_idx), min(len(df_with_macd),
                                                                                        s2.end_kline_original_idx + 1)
                        if idx1_start >= idx1_end or idx2_start >= idx2_end: continue
                        macd_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                        macd_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()
                        if pd.notna(macd_sum1) and pd.notna(macd_sum2) and macd_sum2 > macd_sum1:
                            divergences.append(
                                {"date": s2.end_date, "type": "consolidation_bottom_divergence", "pivot_zd": pivot.zd,
                                 "s1_price": s1.end_price, "s2_price": s2.end_price, "s1_macd_sum": macd_sum1,
                                 "s2_macd_sum": macd_sum2})
                    except Exception as e:
                        print(f"Error processing pivot divergence down: {e}")
            if len(departures_up) >= 2:
                s1, s2 = departures_up[0], departures_up[1]
                if s2.end_price > s1.end_price:
                    try:
                        idx1_start, idx1_end = max(0, s1.start_kline_original_idx), min(len(df_with_macd),
                                                                                        s1.end_kline_original_idx + 1)
                        idx2_start, idx2_end = max(0, s2.start_kline_original_idx), min(len(df_with_macd),
                                                                                        s2.end_kline_original_idx + 1)
                        if idx1_start >= idx1_end or idx2_start >= idx2_end: continue
                        macd_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                        macd_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()
                        if pd.notna(macd_sum1) and pd.notna(macd_sum2) and macd_sum2 < macd_sum1:
                            divergences.append(
                                {"date": s2.end_date, "type": "consolidation_top_divergence", "pivot_zg": pivot.zg,
                                 "s1_price": s1.end_price, "s2_price": s2.end_price, "s1_macd_sum": macd_sum1,
                                 "s2_macd_sum": macd_sum2})
                    except Exception as e:
                        print(f"Error processing pivot divergence up: {e}")
    # print(f"Detected divergences: {len(divergences)}") # Reduced verbosity
    return divergences


### G. 三类买卖点识别函数 (`find_trading_signals`) - 그대로 유지
def find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw):
    signals = []
    for div in divergences:
        signal_price = div.get("s2_price")
        if signal_price is None: continue
        if "bottom_divergence" in div["type"]:
            signals.append({"date": div["date"], "signal_type": "1B", "price": signal_price, "details": div})
        elif "top_divergence" in div["type"]:
            signals.append({"date": div["date"], "signal_type": "1S", "price": signal_price, "details": div})

    first_buy_signals = sorted([s for s in signals if s["signal_type"] == "1B"], key=lambda x: x["date"])
    first_sell_signals = sorted([s for s in signals if s["signal_type"] == "1S"], key=lambda x: x["date"])
    last_1b_date, last_1b_price = (
    first_buy_signals[0]['date'], first_buy_signals[0]['price']) if first_buy_signals else (None, -np.inf)
    last_1s_date, last_1s_price = (
    first_sell_signals[0]['date'], first_sell_signals[0]['price']) if first_sell_signals else (None, np.inf)

    for i in range(len(strokes)):
        stroke = strokes[i]
        if last_1b_date and stroke.type == "down" and stroke.start_date > last_1b_date and stroke.end_price > last_1b_price:
            is_first_pullback = all(
                not (prev_stroke.type == "down" and prev_stroke.end_price > last_1b_price) for j in range(i - 1, -1, -1)
                if strokes[j].end_date > last_1b_date for prev_stroke in [strokes[j]])
            if is_first_pullback: signals.append(
                {"date": stroke.end_date, "signal_type": "2B", "price": stroke.end_price,
                 "details": f"Follows 1B at {last_1b_date} ({last_1b_price:.2f})"})
        if last_1s_date and stroke.type == "up" and stroke.start_date > last_1s_date and stroke.end_price < last_1s_price:
            is_first_rally = all(
                not (prev_stroke.type == "up" and prev_stroke.end_price < last_1s_price) for j in range(i - 1, -1, -1)
                if strokes[j].end_date > last_1s_date for prev_stroke in [strokes[j]])
            if is_first_rally: signals.append({"date": stroke.end_date, "signal_type": "2S", "price": stroke.end_price,
                                               "details": f"Follows 1S at {last_1s_date} ({last_1s_price:.2f})"})

    processed_pivot_breaks = set()
    if pivots and strokes:
        for p_idx, pivot in enumerate(pivots):
            for s_idx in range(len(strokes)):
                stroke = strokes[s_idx]
                if stroke.type == "up" and stroke.start_date >= pivot.end_date and stroke.low > pivot.zd and stroke.end_price > pivot.zg:
                    if s_idx + 1 < len(strokes):
                        pullback_stroke = strokes[s_idx + 1]
                        if pullback_stroke.type == "down" and pullback_stroke.end_price > pivot.zg:
                            signal_key = (pivot.start_date, pivot.end_date, pullback_stroke.end_date, "3B")
                            if signal_key not in processed_pivot_breaks:
                                signals.append({"date": pullback_stroke.end_date, "signal_type": "3B",
                                                "price": pullback_stroke.end_price,
                                                "details": f"Pivot ({pivot.start_date}-{pivot.end_date}, ZG:{pivot.zg:.2f}), Breakout end: {stroke.end_price:.2f}"})
                                processed_pivot_breaks.add(signal_key)
                                break
                elif stroke.type == "down" and stroke.start_date >= pivot.end_date and stroke.high < pivot.zg and stroke.end_price < pivot.zd:
                    if s_idx + 1 < len(strokes):
                        rally_stroke = strokes[s_idx + 1]
                        if rally_stroke.type == "up" and rally_stroke.end_price < pivot.zd:
                            signal_key = (pivot.start_date, pivot.end_date, rally_stroke.end_date, "3S")
                            if signal_key not in processed_pivot_breaks:
                                signals.append({"date": rally_stroke.end_date, "signal_type": "3S",
                                                "price": rally_stroke.end_price,
                                                "details": f"Pivot ({pivot.start_date}-{pivot.end_date}, ZD:{pivot.zd:.2f}), Breakout end: {stroke.end_price:.2f}"})
                                processed_pivot_breaks.add(signal_key)
                                break
    signals.sort(key=lambda x: x["date"])
    print(f"Generated trading signals: {len(signals)}")
    return signals


### H. 主策略编排函数 (`run_chanlun_strategy`) - 修改返回值
def run_chanlun_strategy(df_raw):
    """
    Runs the full Chanlun analysis pipeline.
    MODIFIED: Returns signals, strokes, and pivots.
    """
    analysis_results = {
        "signals": [],
        "strokes": [],
        "pivots": []
    }
    if df_raw is None or len(df_raw) < 35:
        print("Error: Not enough data provided to run Chanlun strategy (need ~35 bars minimum).")
        return analysis_results # Return empty dict on failure

    print("\n--- Running Chanlun Analysis ---")
    start_time = pd.Timestamp.now()
    if not isinstance(df_raw.index, pd.DatetimeIndex):
         if 'date' in df_raw.columns:
              try:
                   df_raw['date'] = pd.to_datetime(df_raw['date'])
                   df_raw = df_raw.set_index('date')
                   print("Set 'date' column as DatetimeIndex.")
              except Exception as e:
                   print(f"Error setting 'date' column as index: {e}")
                   return analysis_results
         else:
              print("Error: DataFrame must have a DatetimeIndex or a 'date' column.")
              return analysis_results
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
         if col not in df_raw.columns:
              print(f"Error: Missing required column '{col}' in input DataFrame.")
              return analysis_results
         try: df_raw[col] = pd.to_numeric(df_raw[col])
         except ValueError:
              print(f"Error: Column '{col}' contains non-numeric data.")
              return analysis_results

    print("Step 1: Preprocessing K-lines...")
    processed_k_lines = preprocess_k_lines(df_raw.copy())
    if not processed_k_lines:
        print("Error: K-line preprocessing failed or resulted in no data.")
        return analysis_results

    print("Step 2: Identifying Fractals...")
    fractals = identify_fractals(processed_k_lines)
    if not fractals:
        print("Warning: No fractals identified. Cannot proceed further.")
        return analysis_results # Cannot build strokes/pivots without fractals

    print("Step 3: Constructing Strokes...")
    strokes = construct_strokes(fractals, processed_k_lines)
    analysis_results["strokes"] = strokes # Store strokes
    if not strokes:
        print("Warning: No strokes constructed.")
        pivots, segments = [], []
    else:
        print("Step 4: Constructing Line Segments (Simplified)...")
        segments = construct_line_segments(strokes)
        print("Step 5: Identifying Pivots (using Strokes)...")
        pivots = identify_pivots(strokes)
        analysis_results["pivots"] = pivots # Store pivots

    print("Step 6: Detecting MACD Divergence...")
    df_with_macd = df_raw.copy()
    divergences = detect_divergence_macd(df_with_macd, strokes, pivots)

    print("Step 7: Identifying Trading Signals...")
    trading_signals = find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw)
    analysis_results["signals"] = trading_signals # Store signals

    end_time = pd.Timestamp.now()
    print(f"--- Chanlun Analysis Completed in {(end_time - start_time).total_seconds():.2f} seconds ---")

    # Return the dictionary containing signals, strokes, and pivots
    return analysis_results


# ---- MODIFIED SECTION: Data Loading from PostgreSQL ----
def fetch_stock_data_from_db(symbol: str, start_date: str, end_date: str):
    """
    Fetches stock data from the PostgreSQL database (stock_daily table).
    """
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} from database.")
    engine = get_engine_instance()  # Get engine from db.database
    if engine is None:
        logger.error("Database engine is not available. Cannot fetch data.")
        return None

    try:
        # Ensure date format is YYYY-MM-DD for SQL query
        start_date_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # SQL query to fetch data from stock_daily table
        # Column names must match your StockDaily model: symbol, date, open, high, low, close, volume
        query = text(f"""
            SELECT date, open, high, low, close, volume
            FROM stock_daily
            WHERE symbol = :symbol
              AND date >= :start_date
              AND date <= :end_date
            ORDER BY date ASC;
        """)

        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params={
                "symbol": symbol,
                "start_date": start_date_dt,
                "end_date": end_date_dt
            })

        if df.empty:
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}.")
            return None

        # Prepare DataFrame as expected by chanlun strategy
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Ensure correct data types (read_sql_query usually handles this, but good to double check)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

        logger.info(f"Data fetched from DB and prepared: {len(df)} rows.")
        return df

    except Exception as e:
        logger.error(f"Error fetching data from database for {symbol}: {e}")
        return None


# ---- UPDATED SECTION: Plotting with mplfinance, Strokes, and Pivots ----
def plot_chanlun_chart_with_signals(df_ohlcv: pd.DataFrame, signals: list, strokes: list, pivots: list, symbol: str):
    """
    Plots candlestick chart with Chanlun buy signals, strokes, and pivots marked.
    df_ohlcv: DataFrame with DatetimeIndex and columns 'open', 'high', 'low', 'close', 'volume'
    signals: List of signal dictionaries, e.g., {"date": datetime_obj, "signal_type": "1B", "price": float}
    strokes: List of Stroke objects.
    pivots: List of Pivot objects.
    symbol: Stock symbol for chart title
    """
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning("No data to plot.")
        return

    # --- Prepare Buy Signals (same as before) ---
    buy_markers_data = []
    for signal in signals:
        if "B" in signal["signal_type"]:
            try:
                signal_date = pd.to_datetime(signal["date"])
                if signal_date in df_ohlcv.index:
                    marker_price = df_ohlcv.loc[signal_date, 'low'] * 0.98
                    buy_markers_data.append((signal_date, marker_price, signal["signal_type"]))
            except Exception as e:
                logger.warning(f"Could not process signal for plotting: {signal} due to {e}")

    ap_markers = [] # Additional plots for markers
    buy_signal_points = [float('nan')] * len(df_ohlcv)
    buy_signal_labels = {}
    if buy_markers_data:
        for date, price, label in buy_markers_data:
            idx_loc = df_ohlcv.index.get_loc(date)
            buy_signal_points[idx_loc] = price
            buy_signal_labels[idx_loc] = (price, label)
        ap_markers.append(mpf.make_addplot(buy_signal_points, type='scatter', marker='^', color='lime', markersize=100, panel=0)) # panel 0 for main chart

    # --- Prepare Strokes ---
    stroke_lines = [] # List to hold line segments for plotting
    for stroke in strokes:
        # Ensure dates are in the DataFrame index
        if stroke.start_date in df_ohlcv.index and stroke.end_date in df_ohlcv.index:
            stroke_data = pd.Series([np.nan] * len(df_ohlcv), index=df_ohlcv.index)
            stroke_data.loc[stroke.start_date] = stroke.start_price
            stroke_data.loc[stroke.end_date] = stroke.end_price
            # Interpolate for plotting the line segment directly
            # mplfinance's addplot with 'line' type will connect non-NaN points
            stroke_lines.append({
                'data': stroke_data.interpolate(method='linear'), # Interpolate between start/end
                'color': 'red' if stroke.type == 'up' else 'green',
                'width': 1.0,
                'panel': 0 # Plot on main panel
            })

    # --- Prepare Pivots ---
    pivot_rectangles = [] # List to hold pivot rectangle definitions
    for pivot in pivots:
         if pivot.start_date in df_ohlcv.index and pivot.end_date in df_ohlcv.index:
             pivot_rectangles.append({
                 'start_date': pivot.start_date,
                 'end_date': pivot.end_date,
                 'zd': pivot.zd,
                 'zg': pivot.zg,
             })

    # --- Plotting ---
    # Combine additional plots
    all_addplots = ap_markers # Start with signal markers

    # Add stroke lines to addplots
    for line_info in stroke_lines:
        all_addplots.append(mpf.make_addplot(line_info['data'], color=line_info['color'], width=line_info['width'], panel=line_info['panel']))

    # Plot the main chart and get figure/axes
    fig, axes = mpf.plot(df_ohlcv,
                         type='candle',
                         style='yahoo',
                         title=f'Chanlun Analysis for {symbol}',
                         ylabel='Price',
                         volume=True,
                         ylabel_lower='Volume',
                         addplot=all_addplots if all_addplots else None,
                         figsize=(18, 10), # Slightly larger figure
                         returnfig=True,
                         warn_too_much_data=10000 # Increase warning threshold
                         )

    # --- Draw Pivots and Signal Labels Manually on Axes ---
    ax_main = axes[0] # Main price axis

    # Draw Pivots
    if pivot_rectangles:
        # Need to convert dates to matplotlib's internal numbers
        df_reset = df_ohlcv.reset_index() # Get 'date' column back
        date_map = {date: i for i, date in enumerate(df_reset['date'])} # Map date to index/number

        for pivot in pivot_rectangles:
            try:
                # Get numerical index for start/end dates
                x_start = date_map.get(pivot['start_date'])
                x_end = date_map.get(pivot['end_date'])

                if x_start is not None and x_end is not None:
                     rect_width = x_end - x_start
                     rect_height = pivot['zg'] - pivot['zd']
                     # Create rectangle patch
                     rect = Rectangle((x_start, pivot['zd']), rect_width, rect_height,
                                      linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2) # Semi-transparent blue
                     ax_main.add_patch(rect)
                else:
                    logger.warning(f"Could not find map indices for pivot dates: {pivot['start_date']} or {pivot['end_date']}")
            except Exception as e_pivot:
                logger.error(f"Error drawing pivot from {pivot.get('start_date')} to {pivot.get('end_date')}: {e_pivot}")


    # Add Buy Signal Labels (adjust x-position based on matplotlib coords)
    if buy_signal_labels:
        # Use the same date_map if pivots were drawn, otherwise create it
        if not 'date_map' in locals():
             df_reset = df_ohlcv.reset_index()
             date_map = {date: i for i, date in enumerate(df_reset['date'])}

        for date_idx, (price, label) in buy_signal_labels.items():
            date_val = df_ohlcv.index[date_idx] # Get the datetime object
            x_coord = date_map.get(date_val) # Get numerical x-coordinate
            if x_coord is not None:
                ax_main.text(x_coord, # Use numerical x-coordinate
                             price * 0.99, # Adjust y-position
                             label,
                             color='lime', # Brighter green
                             fontsize=10,
                             fontweight='bold',
                             ha='center',
                             va='top',
                             bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.4, ec='none') # Add background box
                             )

    logger.info("Displaying Chanlun chart with Strokes and Pivots...")
    mpf.show() # Display the plot


# Usage Example (Modified to use DB data and plotting)
if __name__ == '__main__':
    # Ensure the project's root directory is in sys.path if running this script directly
    # ... (path setup logic remains the same) ...

    # For testing, you might need to initialize the engine if it's not done elsewhere
    # ... (engine initialization logic remains the same) ...
    try:
        from db.database import engine as db_engine, SessionLocal, create_database_tables

        if db_engine is None:
            logger.warning("Database engine not initialized by default. Attempting to create from settings.")
            from config.settings import settings as app_settings

            if app_settings.DB_URL:
                # In a standalone script context, directly use the URL if get_engine_instance fails
                try:
                   engine_direct = create_engine(app_settings.DB_URL)
                   # Assign it to a variable accessible by fetch_stock_data_from_db
                   # This assumes get_engine_instance might still fail but we can proceed
                   # Ideally, refactor get_engine_instance to be robust.
                   def get_engine_instance_fallback(): return engine_direct
                   if 'get_engine_instance' not in globals() or get_engine_instance() is None:
                        # Redefine or assign the fallback if needed
                        get_engine_instance = get_engine_instance_fallback
                        logger.info("Using directly created engine for script execution.")

                except Exception as create_e:
                    logger.error(f"Failed to create engine directly: {create_e}")
                    exit()

            else:
                logger.error("DB_URL not found in settings.")
                exit()
    except ImportError:
        logger.error("Failed to import db modules. Ensure script is run from project root or PYTHONPATH is set.")
        exit()
    except Exception as e:
        logger.error(f"Error during db setup for standalone script: {e}")
        exit()


    symbol_example = "000887"  # Example Stock Code (中鼎股份)
    # Ensure the symbol format matches what's in your 'stock_daily' table
    # If your table stores 'sz000887', use that. If '000887', use that.

    # UPDATED Date Range
    start_date_example = "2020-01-01"  # Start date YYYY-MM-DD
    end_date_example = "2025-05-06"  # End date YYYY-MM-DD (Current date or slightly future)

    # 1. Fetch data from database
    raw_data = fetch_stock_data_from_db(symbol_example, start_date_example, end_date_example)

    if raw_data is not None and not raw_data.empty:
        logger.info(f"\n--- Running Chanlun Strategy for {symbol_example} ---")
        # 2. Run Chanlun analysis - UPDATED to get dict
        analysis_output = run_chanlun_strategy(raw_data.copy())  # Pass a copy

        # Extract results from the dictionary
        final_signals = analysis_output.get("signals", [])
        strokes_list = analysis_output.get("strokes", [])
        pivots_list = analysis_output.get("pivots", [])

        if final_signals is not None: # Check if signals list exists
            logger.info(f"\n--- Generated Trading Signals for {symbol_example} ({len(final_signals)} signals) ---")
            if not final_signals:
                 logger.info("No trading signals generated.")
            else:
                 signals_df = pd.DataFrame(final_signals)
                 signals_df['price'] = signals_df['price'].apply(lambda x: f"{x:.2f}")
                 logger.info(signals_df[['date', 'signal_type', 'price']].to_string()) # Simplified log output

            # 3. Plot the chart with signals, strokes, and pivots - UPDATED call
            plot_chanlun_chart_with_signals(raw_data, final_signals, strokes_list, pivots_list, symbol_example)
        else:
            logger.error(f"Chanlun strategy execution failed or returned invalid signal data for {symbol_example}.")
            # Still attempt to plot with strokes/pivots if they exist
            if strokes_list or pivots_list:
                 logger.info("Attempting to plot chart with strokes/pivots even though signals failed.")
                 plot_chanlun_chart_with_signals(raw_data, [], strokes_list, pivots_list, symbol_example)

    else:
        logger.error(f"Could not fetch or prepare data for {symbol_example} from the database.")