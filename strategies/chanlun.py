# strategies/chanlun.py
import pandas as pd
import numpy as np
# Ensure TA-Lib is installed correctly before uncommenting/using
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("Warning: TA-Lib library not found. MACD divergence detection will be skipped.")
    TALIB_AVAILABLE = False


# Helper class for KLine, Fractal, Stroke, Segment, Pivot
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
        # Ensure indices are valid before slicing
        start_idx = max(0, start_fractal.index_in_processed_k)
        end_idx = min(len(processed_k_lines), end_fractal.index_in_processed_k + 1)
        if start_idx >= end_idx: # Handle edge case where indices are invalid
             self.high = max(start_fractal.kline.high, end_fractal.kline.high)
             self.low = min(start_fractal.kline.low, end_fractal.kline.low)
        else:
             relevant_klines = processed_k_lines[start_idx:end_idx]
             self.high = max(k.high for k in relevant_klines) if relevant_klines else max(start_fractal.kline.high, end_fractal.kline.high)
             self.low = min(k.low for k in relevant_klines) if relevant_klines else min(start_fractal.kline.low, end_fractal.kline.low)

        # Store indices for MACD calculation later
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


### A. K线预处理函数 (`preprocess_k_lines`)
def preprocess_k_lines(df_raw):
    """
    Processes raw K-lines to handle inclusion relationships.
    Outputs a list of KLine objects.
    """
    processed_k_lines = [] # Corrected initialization
    if df_raw is None or len(df_raw) < 1:
        print("Warning: Empty or None DataFrame passed to preprocess_k_lines.")
        return processed_k_lines

    # Convert DataFrame rows to KLine objects for easier manipulation
    # Store original index for later reference (e.g. MACD)
    # Ensure df_raw has expected columns: open, high, low, close
    expected_cols = {'open', 'high', 'low', 'close'}
    if not expected_cols.issubset(df_raw.columns):
        print(f"Error: Input DataFrame missing required columns. Expected: {expected_cols}, Got: {df_raw.columns}")
        return processed_k_lines

    klines_raw_obj = []
    for idx, row in enumerate(df_raw.itertuples()):
        # Check for NaN values before creating KLine object
        if pd.isna(row.open) or pd.isna(row.high) or pd.isna(row.low) or pd.isna(row.close):
            print(f"Warning: Skipping row index {idx} due to NaN values.")
            continue
        klines_raw_obj.append(KLine(row.Index, row.open, row.high, row.low, row.close, idx))


    if not klines_raw_obj:
        print("Warning: No valid KLine objects created from DataFrame.")
        return processed_k_lines # Return empty list

    # --- Inclusion Logic ---
    # This logic requires careful handling of indices and merging.
    # The original implementation had potential issues with direction determination
    # and how merged K-lines represent the underlying data.
    # A simpler, often effective approach is iterative merging:

    merged_klines = []
    if not klines_raw_obj:
        return merged_klines

    k_prev = klines_raw_obj[0]

    for i in range(1, len(klines_raw_obj)):
        k_curr = klines_raw_obj[i]

        # Check for inclusion: k_curr included in k_prev
        if k_curr.high <= k_prev.high and k_curr.low >= k_prev.low:
            # Determine direction based on k_prev vs its predecessor (if available)
            # Simplified: Use k_prev's direction or the trend leading to k_prev
            direction = k_prev.direction
            if direction == 0 and merged_klines: # Use previous merged k-line direction if k_prev is doji
                direction = merged_klines[-1].direction
            if direction == 0: # Still zero? Default to upward bias or based on price relation
                 direction = 1 if k_prev.close >= klines_raw_obj[i-1].close else -1 # Compare to actual previous

            if direction >= 0: # Upward trend or neutral, merge upwards
                k_prev = KLine(k_prev.date, k_prev.open, max(k_prev.high, k_curr.high), max(k_prev.low, k_curr.low), k_curr.close, k_prev.index_in_df)
            else: # Downward trend, merge downwards
                k_prev = KLine(k_prev.date, k_prev.open, min(k_prev.high, k_curr.high), min(k_prev.low, k_curr.low), k_curr.close, k_prev.index_in_df)
            # k_prev absorbs k_curr

        # Check for inclusion: k_prev included in k_curr
        elif k_prev.high <= k_curr.high and k_prev.low >= k_curr.low:
            # Determine direction based on k_curr vs k_prev
            direction = k_curr.direction
            if direction == 0: # Use k_prev's direction if k_curr is doji
                 direction = k_prev.direction
            if direction == 0 and merged_klines: # Still zero? Use previous merged
                 direction = merged_klines[-1].direction
            if direction == 0: # Still zero? Default
                 direction = 1 if k_curr.close >= k_prev.close else -1

            if direction >= 0: # Upward trend or neutral
                 k_prev = KLine(k_prev.date, k_prev.open, max(k_prev.high, k_curr.high), max(k_prev.low, k_curr.low), k_curr.close, k_prev.index_in_df)
            else: # Downward trend
                 k_prev = KLine(k_prev.date, k_prev.open, min(k_prev.high, k_curr.high), min(k_prev.low, k_curr.low), k_curr.close, k_prev.index_in_df)
            # k_prev is effectively replaced by a merged version dominated by k_curr's range but keeping start info

        else:
            # No inclusion, finalize k_prev and move to k_curr
            merged_klines.append(k_prev)
            k_prev = k_curr

    merged_klines.append(k_prev) # Add the last processed k-line

    print(f"Preprocessed K-lines: {len(merged_klines)}")
    return merged_klines


### B. 分型识别函数 (`identify_fractals`)
def identify_fractals(processed_k_lines):
    fractals = [] # Corrected initialization
    if len(processed_k_lines) < 3:
        return fractals

    for i in range(1, len(processed_k_lines) - 1):
        k1 = processed_k_lines[i - 1]
        k2 = processed_k_lines[i]
        k3 = processed_k_lines[i + 1]

        # Top Fractal Check (k2 is highest high AND highest low of the three)
        is_top_fractal = k2.high >= k1.high and k2.high >= k3.high and \
                         k2.low >= k1.low and k2.low >= k3.low

        # Stricter check: ensure it's a real peak if highs are equal
        if is_top_fractal and (k2.high > k1.high or k2.high > k3.high or (k2.high == k1.high and k2.low > k1.low) or (k2.high == k3.high and k2.low > k3.low)):
             # Check if this fractal is too close to the previous one of the same type
             is_valid = True
             if fractals and fractals[-1].fractal_type == "top":
                  # Avoid adding if it's essentially the same peak (e.g., index difference < 3)
                  if abs(i - fractals[-1].index_in_processed_k) < 3:
                       # Replace previous if current is higher
                       if k2.high > fractals[-1].price:
                            fractals[-1] = Fractal(k2, "top", i)
                       is_valid = False # Don't add as a new one
             if is_valid:
                  fractals.append(Fractal(k2, "top", i))


        # Bottom Fractal Check (k2 is lowest low AND lowest high of the three)
        is_bottom_fractal = k2.low <= k1.low and k2.low <= k3.low and \
                            k2.high <= k1.high and k2.high <= k3.high

        # Stricter check for real valley
        if is_bottom_fractal and (k2.low < k1.low or k2.low < k3.low or (k2.low == k1.low and k2.high < k1.high) or (k2.low == k3.low and k2.high < k3.high)):
             is_valid = True
             if fractals and fractals[-1].fractal_type == "bottom":
                  if abs(i - fractals[-1].index_in_processed_k) < 3:
                       if k2.low < fractals[-1].price:
                            fractals[-1] = Fractal(k2, "bottom", i)
                       is_valid = False
             if is_valid:
                  fractals.append(Fractal(k2, "bottom", i))

    print(f"Identified fractals: {len(fractals)}")
    return fractals


### C. 笔构建函数 (`construct_strokes`)
def construct_strokes(fractals, processed_k_lines):
    strokes = [] # Corrected initialization
    if len(fractals) < 2:
        return strokes

    last_confirmed_stroke_end_fractal = None
    potential_stroke_start_fractal = None

    f_idx = 0
    while f_idx < len(fractals):
        current_fractal = fractals[f_idx]

        if potential_stroke_start_fractal is None:
            potential_stroke_start_fractal = current_fractal
            f_idx += 1
            continue

        # Ensure alternating fractal types
        if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
            # If same type, choose the "better" one (higher top, lower bottom)
            if (current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
               (current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                potential_stroke_start_fractal = current_fractal
            f_idx += 1
            continue

        # Check K-line separation rule: >= 3 bars between fractal centers
        k_line_distance = abs(current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k)
        if k_line_distance < 3:
            # Too close, potentially invalidates the earlier fractal or requires merging.
            # Simplified: Skip this pair for now, reset potential start
            # A more complex rule might try to merge or select the dominant fractal.
            # print(f"Debug: Fractals too close at indices {potential_stroke_start_fractal.index_in_processed_k} and {current_fractal.index_in_processed_k}. Resetting potential start.")
            potential_stroke_start_fractal = current_fractal # Reset potential start
            f_idx += 1
            continue

        # Check stroke integrity: Top's high > Bottom's high, Bottom's low < Top's low
        # This was slightly off in the original logic. It should compare the *prices* of the fractals.
        valid_stroke_integrity = False
        if potential_stroke_start_fractal.fractal_type == "bottom" and current_fractal.fractal_type == "top":
            # Up-stroke: Current top fractal price must be higher than potential start bottom fractal price
            if current_fractal.price > potential_stroke_start_fractal.price:
                valid_stroke_integrity = True
        elif potential_stroke_start_fractal.fractal_type == "top" and current_fractal.fractal_type == "bottom":
            # Down-stroke: Current bottom fractal price must be lower than potential start top fractal price
            if current_fractal.price < potential_stroke_start_fractal.price:
                valid_stroke_integrity = True

        if valid_stroke_integrity:
            # Create the potential stroke
            new_stroke = Stroke(potential_stroke_start_fractal, current_fractal, processed_k_lines)

            # Stroke Confirmation Logic (Simplified - needs refinement based on Chan rules)
            # Check if it connects to the last confirmed stroke
            if not strokes or \
               (new_stroke.start_fractal.index_in_processed_k == strokes[-1].end_fractal.index_in_processed_k and \
                new_stroke.type != strokes[-1].type):
                # This stroke is potentially valid, add it for now
                strokes.append(new_stroke)
                last_confirmed_stroke_end_fractal = new_stroke.end_fractal
                potential_stroke_start_fractal = new_stroke.end_fractal # Next potential stroke starts here
            else:
                # Conflict or gap. This part needs robust Chan Lun rules for combination/destruction.
                # Simple approach: if the new fractal is "better" than the end of the last stroke,
                # potentially invalidate the last stroke and try forming a new one.
                last_stroke = strokes[-1]
                if current_fractal.fractal_type == last_stroke.end_fractal.fractal_type:
                     if (current_fractal.fractal_type == "top" and current_fractal.price > last_stroke.end_fractal.price) or \
                        (current_fractal.fractal_type == "bottom" and current_fractal.price < last_stroke.end_fractal.price):
                          # Last stroke is invalidated/updated
                          print(f"Debug: Updating last stroke ending at {last_stroke.end_date} with new fractal at {current_fractal.date}")
                          strokes.pop() # Remove last stroke
                          # Restart the process from the fractal before the invalidated stroke's start
                          # This requires better state management or backtracking.
                          # Simplified reset:
                          potential_stroke_start_fractal = current_fractal
                     else: # Not better, just reset potential start
                          potential_stroke_start_fractal = current_fractal
                else: # Doesn't connect and type is different - likely a gap or complex situation
                     potential_stroke_start_fractal = current_fractal # Reset

            f_idx += 1 # Move to next fractal

        else: # Integrity failed
            # If the current fractal is stronger than the potential start (and same type), update potential start
            if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
                if (current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
                   (current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                    potential_stroke_start_fractal = current_fractal
            else:
                 # Different type, but failed integrity. Reset potential start to current.
                 potential_stroke_start_fractal = current_fractal
            f_idx += 1

    # Post-processing refinement (optional but recommended)
    # E.g., ensure strict alternation, handle stroke destruction rules.
    # This requires implementing more advanced Chan Lun concepts.
    print(f"Constructed strokes (initial): {len(strokes)}")
    return strokes # Return the initially constructed strokes for now


### D. 线段构建函数 (`construct_line_segments`)
def construct_line_segments(strokes):
    segments = [] # Corrected initialization
    if len(strokes) < 3:
        return segments

    # Simplified logic based on 3-stroke overlap (needs refinement for feature sequence)
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]

        # Basic check: s1, s3 same direction, s2 opposite
        if s1.type == s3.type and s1.type != s2.type:
            # Check for overlap: max(s1.low, s3.low) < min(s1.high, s3.high)
            overlap_exists = max(s1.low, s3.low) < min(s1.high, s3.high)

            # Check for segment break: s3 extreme breaks s1 extreme
            segment_break = False
            if s1.type == "up" and s3.end_price > s1.end_price:
                segment_break = True
            elif s1.type == "down" and s3.end_price < s1.end_price:
                segment_break = True

            # Check for pullback validity: s2 doesn't break s1 start
            pullback_valid = False
            if s1.type == "up" and s2.end_price > s1.start_price: # s2 low > s1 low
                 pullback_valid = True
            elif s1.type == "down" and s2.end_price < s1.start_price: # s2 high < s1 high
                 pullback_valid = True

            # Combine conditions (Simplified: overlap and break implies potential segment)
            # More rigorous check needs feature sequence analysis.
            if overlap_exists and segment_break and pullback_valid:
                # Potential segment found: s1, s2, s3
                # Extend segment logic (as in original, simplified)
                current_segment_strokes = [s1, s2, s3]
                segment_type = s1.type
                j = i + 3 # Start checking from the 4th stroke relative to s1

                while j < len(strokes) - 1:
                    s_next_opposite = strokes[j]
                    s_next_same_dir = strokes[j + 1]

                    # Check if types alternate correctly for extension
                    if s_next_opposite.type != segment_type and s_next_same_dir.type == segment_type:
                        # Check extension conditions (simplified: new extreme breaks last extreme)
                        last_seg_stroke = current_segment_strokes[-1]
                        extend_break = (segment_type == "up" and s_next_same_dir.end_price > last_seg_stroke.end_price) or \
                                       (segment_type == "down" and s_next_same_dir.end_price < last_seg_stroke.end_price)

                        # Check pullback validity for extension
                        extend_pullback_valid = (segment_type == "up" and s_next_opposite.end_price > last_seg_stroke.start_price) or \
                                                (segment_type == "down" and s_next_opposite.end_price < last_seg_stroke.start_price)


                        if extend_break and extend_pullback_valid: # Simplified extension condition
                            current_segment_strokes.extend([s_next_opposite, s_next_same_dir])
                            j += 2
                        else:
                            break # Extension conditions not met
                    else:
                        break # Type pattern broken

                segments.append(Segment(current_segment_strokes, segment_type))
                i = j # Move index past the strokes consumed by the segment
                continue # Continue searching for the next segment

        i += 1 # Move to the next potential starting stroke

    print(f"Constructed segments: {len(segments)}")
    return segments


### E. 中枢识别函数 (`identify_pivots`)
def identify_pivots(strokes):
    pivots = [] # Corrected initialization
    if len(strokes) < 3:
        return pivots

    # Simplified pivot identification based on 3+ overlapping strokes
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]

        # Check for alternating types
        if s1.type != s2.type and s2.type != s3.type:
            # Calculate initial ZG/ZD based on s1, s2, s3
            if s1.type == "up": # Up-Down-Up sequence
                zg_candidate = s2.high
                zd_candidate = max(s1.low, s3.low)
            else: # Down-Up-Down sequence
                zg_candidate = min(s1.high, s3.high)
                zd_candidate = s2.low

            # Check for overlap: ZD < ZG
            if zd_candidate < zg_candidate:
                # Potential pivot found
                current_pivot_strokes = [s1, s2, s3]
                current_zd = zd_candidate
                current_zg = zg_candidate
                start_date = s1.start_date
                end_date = s3.end_date

                # Try to extend the pivot
                j = i + 3
                while j < len(strokes): # Check next stroke
                    next_stroke = strokes[j]
                    # Check if type alternates and if it interacts with ZG/ZD
                    if next_stroke.type != current_pivot_strokes[-1].type:
                        # Interaction check (simplified: does next_stroke extreme fall within ZG/ZD?)
                        # A more complex check involves how the pivot evolves (ZG/ZD update)
                        interacts = (next_stroke.type == "up" and next_stroke.high >= current_zd) or \
                                    (next_stroke.type == "down" and next_stroke.low <= current_zg)

                        if interacts:
                             # Recalculate ZG/ZD including the new stroke
                             temp_strokes = current_pivot_strokes + [next_stroke]
                             up_stroke_lows = [s.low for s in temp_strokes if s.type == "up"]
                             down_stroke_highs = [s.high for s in temp_strokes if s.type == "down"]

                             if not up_stroke_lows or not down_stroke_highs: break # Should not happen

                             new_zd = max(up_stroke_lows)
                             new_zg = min(down_stroke_highs)

                             if new_zd < new_zg: # Still a valid pivot range
                                  current_pivot_strokes.append(next_stroke)
                                  current_zd = new_zd # Update ZD/ZG
                                  current_zg = new_zg
                                  end_date = next_stroke.end_date
                                  j += 1
                             else: # Adding stroke destroyed pivot overlap
                                  break
                        else: # Stroke does not interact enough to extend pivot
                             break
                    else: # Type doesn't alternate
                        break

                # Only add pivot if it contains at least 3 strokes
                if len(current_pivot_strokes) >= 3:
                    pivots.append(Pivot(list(current_pivot_strokes), current_zg, current_zd, start_date, end_date))
                    # Advance i past the strokes in this pivot to avoid overlapping pivot detection (simplification)
                    i += len(current_pivot_strokes)
                else: # Should not happen if started with 3
                     i += 1
            else: # No overlap for s1, s2, s3
                i += 1
        else: # Types not alternating
            i += 1

    print(f"Identified pivots: {len(pivots)}")
    return pivots


### F. 背驰检测函数 (`detect_divergence_macd`)
def detect_divergence_macd(df_with_macd, strokes_or_segments, pivots):
    """
    Detects divergence using MACD.
    """
    divergences = [] # Corrected initialization
    if not TALIB_AVAILABLE:
        print("Skipping MACD divergence detection as TA-Lib is not available.")
        return divergences

    if df_with_macd is None or df_with_macd.empty or strokes_or_segments is None or len(strokes_or_segments) < 2:
        return divergences

    # Ensure MACD is calculated
    if 'macdhist' not in df_with_macd.columns:
        close_prices = df_with_macd['close'].astype(float)
        if len(close_prices) < 34: # Need enough data for MACD
            print("Warning: Not enough data for MACD calculation in divergence check.")
            return divergences
        try:
            macd, macdsignal, macdhist = talib.MACD(close_prices.values, fastperiod=12, slowperiod=26, signalperiod=9)
            # Assign back to DataFrame, ensuring index alignment
            df_with_macd = df_with_macd.copy() # Avoid SettingWithCopyWarning
            df_with_macd['macd'] = macd
            df_with_macd['macdsignal'] = macdsignal
            df_with_macd['macdhist'] = macdhist
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return divergences

    # --- Trend Divergence (using strokes/segments) ---
    for i in range(len(strokes_or_segments) - 2): # Need i, i+1, i+2
        item1 = strokes_or_segments[i]
        item2 = strokes_or_segments[i + 2]
        item_mid = strokes_or_segments[i+1] # The item between

        # Check if item1 and item2 are comparable (same type, separated by one opposite)
        if item1.type == item2.type and item1.type != item_mid.type:
            # Get MACD histogram area (sum of histogram values) for the strokes/segments
            try:
                # Ensure indices are valid and within DataFrame bounds
                idx1_start = max(0, item1.start_kline_original_idx)
                idx1_end = min(len(df_with_macd), item1.end_kline_original_idx + 1)
                idx2_start = max(0, item2.start_kline_original_idx)
                idx2_end = min(len(df_with_macd), item2.end_kline_original_idx + 1)

                if idx1_start >= idx1_end or idx2_start >= idx2_end: continue # Skip if indices are invalid

                macd_hist_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                macd_hist_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()

                # Check for NaNs in sums
                if pd.isna(macd_hist_sum1) or pd.isna(macd_hist_sum2):
                    # print(f"Debug: NaN MACD sum for items starting {item1.start_date} or {item2.start_date}")
                    continue

            except IndexError:
                print(f"Warning: Index out of bounds during MACD divergence check for items starting {item1.start_date} / {item2.start_date}.")
                continue
            except Exception as e:
                print(f"Error accessing MACD data for divergence: {e}")
                continue


            # --- Check Divergence Conditions ---
            if item1.type == "up":  # Potential top divergence (Price Higher, MACD Area Lower)
                if item2.end_price > item1.end_price and macd_hist_sum2 < macd_hist_sum1:
                    divergences.append({
                        "date": item2.end_date,
                        "type": "trend_top_divergence",
                        "level": "stroke" if isinstance(item1, Stroke) else "segment",
                        "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                        "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2
                    })
            elif item1.type == "down":  # Potential bottom divergence (Price Lower, MACD Area Higher/Less Negative)
                if item2.end_price < item1.end_price and macd_hist_sum2 > macd_hist_sum1:
                    divergences.append({
                        "date": item2.end_date,
                        "type": "trend_bottom_divergence",
                        "level": "stroke" if isinstance(item1, Stroke) else "segment",
                        "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                        "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2
                    })

    # --- Consolidation (Pivot) Divergence ---
    # This requires identifying strokes leaving a pivot and comparing them.
    if pivots:
         for pivot in pivots:
              departures_down = []
              departures_up = []
              # Find strokes starting after the pivot ends
              strokes_after_pivot = [s for s in strokes_or_segments if s.start_date >= pivot.end_date]

              for stroke in strokes_after_pivot:
                   # Check if it's a clear departure
                   if stroke.type == "down" and stroke.end_price < pivot.zd:
                        departures_down.append(stroke)
                   elif stroke.type == "up" and stroke.end_price > pivot.zg:
                        departures_up.append(stroke)
                   # Stop checking departures once a stroke of the opposite direction occurs after a departure
                   if (departures_down and stroke.type == "up") or (departures_up and stroke.type == "down"):
                        break

              # Analyze departures for divergence
              if len(departures_down) >= 2:
                   s1, s2 = departures_down[0], departures_down[1] # Compare first two downward departures
                   if s2.end_price < s1.end_price: # Second departure makes a new low
                        try:
                             idx1_start = max(0, s1.start_kline_original_idx)
                             idx1_end = min(len(df_with_macd), s1.end_kline_original_idx + 1)
                             idx2_start = max(0, s2.start_kline_original_idx)
                             idx2_end = min(len(df_with_macd), s2.end_kline_original_idx + 1)
                             if idx1_start >= idx1_end or idx2_start >= idx2_end: continue

                             macd_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                             macd_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()

                             if pd.notna(macd_sum1) and pd.notna(macd_sum2) and macd_sum2 > macd_sum1: # MACD higher (less negative)
                                  divergences.append({
                                      "date": s2.end_date, "type": "consolidation_bottom_divergence",
                                      "pivot_zd": pivot.zd, "s1_price": s1.end_price, "s2_price": s2.end_price,
                                      "s1_macd_sum": macd_sum1, "s2_macd_sum": macd_sum2
                                  })
                        except Exception as e: print(f"Error processing pivot divergence down: {e}")

              if len(departures_up) >= 2:
                   s1, s2 = departures_up[0], departures_up[1] # Compare first two upward departures
                   if s2.end_price > s1.end_price: # Second departure makes a new high
                        try:
                             idx1_start = max(0, s1.start_kline_original_idx)
                             idx1_end = min(len(df_with_macd), s1.end_kline_original_idx + 1)
                             idx2_start = max(0, s2.start_kline_original_idx)
                             idx2_end = min(len(df_with_macd), s2.end_kline_original_idx + 1)
                             if idx1_start >= idx1_end or idx2_start >= idx2_end: continue

                             macd_sum1 = df_with_macd['macdhist'].iloc[idx1_start:idx1_end].sum()
                             macd_sum2 = df_with_macd['macdhist'].iloc[idx2_start:idx2_end].sum()

                             if pd.notna(macd_sum1) and pd.notna(macd_sum2) and macd_sum2 < macd_sum1: # MACD lower
                                  divergences.append({
                                      "date": s2.end_date, "type": "consolidation_top_divergence",
                                      "pivot_zg": pivot.zg, "s1_price": s1.end_price, "s2_price": s2.end_price,
                                      "s1_macd_sum": macd_sum1, "s2_macd_sum": macd_sum2
                                  })
                        except Exception as e: print(f"Error processing pivot divergence up: {e}")


    print(f"Detected divergences: {len(divergences)}")
    return divergences


### G. 三类买卖点识别函数 (`find_trading_signals`)
def find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw):
    signals = [] # Corrected initialization

    # 1. First Class Buy/Sell Points (from Divergences)
    for div in divergences:
        signal_price = div.get("s2_price") # The price at the point of divergence
        if signal_price is None: continue

        if "bottom_divergence" in div["type"]:
            signals.append({"date": div["date"], "signal_type": "1B", "price": signal_price, "details": div})
        elif "top_divergence" in div["type"]:
            signals.append({"date": div["date"], "signal_type": "1S", "price": signal_price, "details": div})

    # 2. Second Class Buy/Sell Points (Simplified Logic)
    # 2B: After a 1B, the first pullback (down-stroke) whose low is higher than the 1B low.
    # 2S: After a 1S, the first rally (up-stroke) whose high is lower than the 1S high.
    first_buy_signals = sorted([s for s in signals if s["signal_type"] == "1B"], key=lambda x: x["date"])
    first_sell_signals = sorted([s for s in signals if s["signal_type"] == "1S"], key=lambda x: x["date"])

    last_1b_date = None
    last_1b_price = -np.inf
    if first_buy_signals:
        last_1b_date = first_buy_signals[0]['date'] # Consider the earliest 1B for subsequent 2B/3B
        last_1b_price = first_buy_signals[0]['price']

    last_1s_date = None
    last_1s_price = np.inf
    if first_sell_signals:
        last_1s_date = first_sell_signals[0]['date']
        last_1s_price = first_sell_signals[0]['price']

    for i in range(len(strokes)):
        stroke = strokes[i]
        # Check for 2B
        if last_1b_date and stroke.type == "down" and stroke.start_date > last_1b_date:
            if stroke.end_price > last_1b_price:
                 # Check if this is the *first* such pullback after the latest 1B
                 is_first_pullback = True
                 for j in range(i - 1, -1, -1):
                      prev_stroke = strokes[j]
                      if prev_stroke.end_date <= last_1b_date: break # Stop checking before 1B
                      if prev_stroke.type == "down" and prev_stroke.end_price > last_1b_price:
                           is_first_pullback = False # Found an earlier valid 2B candidate
                           break
                 if is_first_pullback:
                      signals.append({"date": stroke.end_date, "signal_type": "2B", "price": stroke.end_price, "details": f"Follows 1B at {last_1b_date} ({last_1b_price:.2f})"})
                      # Potentially update last_1b_date here if we only want one 2B per 1B sequence?

        # Check for 2S
        if last_1s_date and stroke.type == "up" and stroke.start_date > last_1s_date:
            if stroke.end_price < last_1s_price:
                 is_first_rally = True
                 for j in range(i - 1, -1, -1):
                      prev_stroke = strokes[j]
                      if prev_stroke.end_date <= last_1s_date: break
                      if prev_stroke.type == "up" and prev_stroke.end_price < last_1s_price:
                           is_first_rally = False
                           break
                 if is_first_rally:
                      signals.append({"date": stroke.end_date, "signal_type": "2S", "price": stroke.end_price, "details": f"Follows 1S at {last_1s_date} ({last_1s_price:.2f})"})
                      # Update last_1s_date?

    # 3. Third Class Buy/Sell Points (Simplified Logic)
    # 3B: Stroke breaks out above a pivot (ZG), then a subsequent pullback stroke's low stays above ZG.
    # 3S: Stroke breaks out below a pivot (ZD), then a subsequent rally stroke's high stays below ZD.
    processed_pivot_breaks = set() # Avoid multiple signals for the same breakout/pullback
    if pivots and strokes:
        for p_idx, pivot in enumerate(pivots):
            for s_idx in range(len(strokes)):
                stroke = strokes[s_idx]
                # Check for upward breakout
                if stroke.type == "up" and stroke.start_date >= pivot.end_date and stroke.low > pivot.zd and stroke.end_price > pivot.zg:
                    # Found potential breakout, look for pullback
                    if s_idx + 1 < len(strokes):
                        pullback_stroke = strokes[s_idx + 1]
                        if pullback_stroke.type == "down" and pullback_stroke.end_price > pivot.zg: # Pullback stays above ZG
                            signal_key = (pivot.start_date, pivot.end_date, pullback_stroke.end_date, "3B")
                            if signal_key not in processed_pivot_breaks:
                                signals.append({
                                    "date": pullback_stroke.end_date, "signal_type": "3B",
                                    "price": pullback_stroke.end_price,
                                    "details": f"Pivot ({pivot.start_date}-{pivot.end_date}, ZG:{pivot.zg:.2f}), Breakout end: {stroke.end_price:.2f}"
                                })
                                processed_pivot_breaks.add(signal_key)
                                break # Move to next pivot after finding a 3B for this breakout

                # Check for downward breakout
                elif stroke.type == "down" and stroke.start_date >= pivot.end_date and stroke.high < pivot.zg and stroke.end_price < pivot.zd:
                    # Found potential breakout, look for rally
                    if s_idx + 1 < len(strokes):
                        rally_stroke = strokes[s_idx + 1]
                        if rally_stroke.type == "up" and rally_stroke.end_price < pivot.zd: # Rally stays below ZD
                            signal_key = (pivot.start_date, pivot.end_date, rally_stroke.end_date, "3S")
                            if signal_key not in processed_pivot_breaks:
                                signals.append({
                                    "date": rally_stroke.end_date, "signal_type": "3S",
                                    "price": rally_stroke.end_price,
                                    "details": f"Pivot ({pivot.start_date}-{pivot.end_date}, ZD:{pivot.zd:.2f}), Breakout end: {stroke.end_price:.2f}"
                                })
                                processed_pivot_breaks.add(signal_key)
                                break # Move to next pivot

    # Sort signals by date
    signals.sort(key=lambda x: x["date"])
    print(f"Generated trading signals: {len(signals)}")
    return signals


### H. 主策略编排函数 (`run_chanlun_strategy`)
def run_chanlun_strategy(df_raw):
    """
    Runs the full Chanlun analysis pipeline.
    """
    if df_raw is None or len(df_raw) < 35:  # Need enough data for MACD and structures
        print("Error: Not enough data provided to run Chanlun strategy (need ~35 bars minimum).")
        return None # Return None or empty list on failure

    print("\n--- Running Chanlun Analysis ---")
    start_time = pd.Timestamp.now()

    # Ensure DataFrame index is datetime
    if not isinstance(df_raw.index, pd.DatetimeIndex):
         if 'date' in df_raw.columns:
              try:
                   df_raw['date'] = pd.to_datetime(df_raw['date'])
                   df_raw = df_raw.set_index('date')
                   print("Set 'date' column as DatetimeIndex.")
              except Exception as e:
                   print(f"Error setting 'date' column as index: {e}")
                   return None
         else:
              print("Error: DataFrame must have a DatetimeIndex or a 'date' column.")
              return None

    # Ensure required columns exist and are numeric
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
         if col not in df_raw.columns:
              print(f"Error: Missing required column '{col}' in input DataFrame.")
              return None
         try:
              df_raw[col] = pd.to_numeric(df_raw[col])
         except ValueError:
              print(f"Error: Column '{col}' contains non-numeric data.")
              return None


    # 1. K-Line Preprocessing
    print("Step 1: Preprocessing K-lines...")
    processed_k_lines = preprocess_k_lines(df_raw.copy())
    if not processed_k_lines:
        print("Error: K-line preprocessing failed or resulted in no data.")
        return None

    # 2. Fractal Identification
    print("Step 2: Identifying Fractals...")
    fractals = identify_fractals(processed_k_lines)
    if not fractals:
        print("Warning: No fractals identified. Cannot proceed further.")
        return [] # Return empty list if no fractals

    # 3. Stroke Construction
    print("Step 3: Constructing Strokes...")
    strokes = construct_strokes(fractals, processed_k_lines)
    if not strokes:
        print("Warning: No strokes constructed. Cannot identify pivots or segments.")
        # Can still proceed to MACD divergence on fractals if needed, but signals will be limited.
        pivots = []
        segments = []
    else:
        # 4. Line Segment Construction (Optional but useful for higher-level analysis)
        print("Step 4: Constructing Line Segments (Simplified)...")
        segments = construct_line_segments(strokes)

        # 5. Pivot Identification
        print("Step 5: Identifying Pivots (using Strokes)...")
        pivots = identify_pivots(strokes)

    # 6. Divergence Detection (using MACD)
    print("Step 6: Detecting MACD Divergence...")
    # Pass the original DataFrame for MACD calculation
    df_with_macd = df_raw.copy() # Use original df for MACD
    # Use strokes for divergence detection as they are more fundamental than segments
    divergences = detect_divergence_macd(df_with_macd, strokes, pivots)

    # 7. Identify Trading Signals
    print("Step 7: Identifying Trading Signals...")
    trading_signals = find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw)

    end_time = pd.Timestamp.now()
    print(f"--- Chanlun Analysis Completed in {(end_time - start_time).total_seconds():.2f} seconds ---")

    return trading_signals


# Usage Example (Needs a data loading function)
if __name__ == '__main__':

    # Dummy function for fetching data - REPLACE WITH YOUR ACTUAL DATA LOADER
    def fetch_stock_data(symbol, start_date, end_date):
        # Example: using akshare (make sure it's installed: pip install akshare)
        try:
            import akshare as ak
            print(f"Fetching data for {symbol} from {start_date} to {end_date} using akshare...")
            # Adjust symbol format if needed (e.g., 'sh600519' -> '600519')
            ak_symbol = symbol.replace('sh', '').replace('sz', '')
            df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq") # qfq = 前复权
            if df is None or df.empty:
                 print(f"Warning: akshare returned no data for {symbol}.")
                 return None
            # Rename columns to match 'open', 'high', 'low', 'close'
            df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                # Add other columns if needed by KLine or other parts
            }, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Ensure correct data types
            for col in ['open', 'high', 'low', 'close']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Drop rows with NaN in essential columns
            print(f"Data fetched and prepared: {len(df)} rows.")
            return df
        except ImportError:
            print("Error: akshare library not found. Please install it: pip install akshare")
            return None
        except Exception as e:
            print(f"Error fetching data using akshare for {symbol}: {e}")
            return None

    # Example Usage:
    symbol_example = "sz000887"  # 中鼎股份 (Example)
    start_date_example = "20230101" # Start date YYYYMMDD
    end_date_example = "20240430"   # End date YYYYMMDD

    raw_data = fetch_stock_data(symbol_example, start_date_example, end_date_example)

    if raw_data is not None and not raw_data.empty:
        print(f"\n--- Running Chanlun Strategy for {symbol_example} ---")
        final_signals = run_chanlun_strategy(raw_data)

        if final_signals is not None: # Check for None return on error
            print(f"\n--- Generated Trading Signals for {symbol_example} ({len(final_signals)} signals) ---")
            if not final_signals:
                 print("No trading signals generated.")
            else:
                 # Create DataFrame for better display
                 signals_df = pd.DataFrame(final_signals)
                 # Format price for display
                 signals_df['price'] = signals_df['price'].apply(lambda x: f"{x:.2f}")
                 # Optionally select/reorder columns
                 print(signals_df[['date', 'signal_type', 'price', 'details']].to_string())
        else:
            print(f"Chanlun strategy execution failed for {symbol_example}.")

    else:
        print(f"Could not fetch or prepare data for {symbol_example}.")
