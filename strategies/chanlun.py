import pandas as pd
import numpy as np
import talib


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
        self.high = max(
            k.high for k in processed_k_lines[start_fractal.index_in_processed_k: end_fractal.index_in_processed_k + 1])
        self.low = min(
            k.low for k in processed_k_lines[start_fractal.index_in_processed_k: end_fractal.index_in_processed_k + 1])
        # Store indices for MACD calculation later
        self.start_kline_original_idx = start_fractal.kline.index_in_df
        self.end_kline_original_idx = end_fractal.kline.index_in_df


class Segment:
    def __init__(self, strokes, segment_type):  # segment_type: "up" or "down"
        self.strokes = strokes
        self.segment_type = segment_type
        self.start_date = strokes.start_date
        self.end_date = strokes[-1].end_date
        self.start_price = strokes.start_price
        self.end_price = strokes[-1].end_price
        self.high = max(s.high for s in strokes)
        self.low = min(s.low for s in strokes)
        self.start_kline_original_idx = strokes.start_kline_original_idx
        self.end_kline_original_idx = strokes[-1].end_kline_original_idx


class Pivot:  # Zhongshu
    def __init__(self, strokes_in_pivot, zg, zd, start_date, end_date):
        self.strokes_in_pivot = strokes_in_pivot
        self.zg = zg  # Pivot High
        self.zd = zd  # Pivot Low
        self.start_date = start_date
        self.end_date = end_date
        self.start_kline_original_idx = strokes_in_pivot.start_kline_original_idx
        self.end_kline_original_idx = strokes_in_pivot[-1].end_kline_original_idx


### A. K线预处理函数 (`preprocess_k_lines`)
# (包含关系处理)
def preprocess_k_lines(df_raw):
    """
    Processes raw K-lines to handle inclusion relationships.
    Outputs a list of KLine objects.
    """
    processed_k_lines =
    if df_raw is None or len(df_raw) < 1:
        return processed_k_lines

    # Convert DataFrame rows to KLine objects for easier manipulation
    # Store original index for later reference (e.g. MACD)
    klines_raw_obj = [KLine(row.Index, row.open, row.high, row.low, row.close, idx)
                      for idx, row in enumerate(df_raw.itertuples())]

    if not klines_raw_obj:
        return

    current_k = klines_raw_obj

    for i in range(1, len(klines_raw_obj)):
        next_k = klines_raw_obj[i]

        # Check for inclusion: next_k is included in current_k
        if next_k.high <= current_k.high and next_k.low >= current_k.low:
            # Determine direction based on current_k vs previous actual kline (if exists)
            # For simplicity in this example, we'll use current_k's own direction
            # A more robust method would compare current_k with the last *processed_k_lines[-1]* if available
            # or its actual predecessor if current_k is the first in a merge sequence.

            # Simplified direction: if current_k is bullish, trend is up, else down.
            # This simplification might need refinement based on strict Chan Lun rules for direction.
            # [49]: "若2比1高则取向上包含；若2比1低则取向下包含."
            # This implies comparing the K-line *initiating* the inclusion processing with its predecessor.
            # For this implementation, we use the first K-line's (current_k) direction in a sequence of inclusions.

            if not processed_k_lines:  # If current_k is the first k-line being processed
                direction_for_merge = 1 if current_k.close > current_k.open else -1  # Default to current_k's direction
            else:
                # Compare current_k to the last *committed* processed K-line
                # This is a proxy for the "K_i vs K_{i-1}" comparison
                prev_committed_k = processed_k_lines[-1]
                if current_k.high > prev_committed_k.high and current_k.low > prev_committed_k.low:
                    direction_for_merge = 1  # Upward
                elif current_k.high < prev_committed_k.high and current_k.low < prev_committed_k.low:
                    direction_for_merge = -1  # Downward
                else:  # Mixed or equal, use current_k's own direction or previous direction
                    direction_for_merge = current_k.direction if current_k.direction != 0 else (
                        1 if processed_k_lines[-1].direction >= 0 else -1)

            if direction_for_merge == 1:  # Upward inclusion
                current_k.high = max(current_k.high, next_k.high)
                current_k.low = max(current_k.low, next_k.low)  # Per [49] "低点最高为低点"
            else:  # Downward inclusion or neutral (treat as downward)
                current_k.high = min(current_k.high, next_k.high)  # Per [49] "高点最低为高点"
                current_k.low = min(current_k.low, next_k.low)
            # Update date to the later K-line if merging. Chan Lun usually keeps the start date of the dominant K-line.
            # For simplicity, we keep current_k's date and original index.
            # The end K-line's information (next_k) is absorbed.
            current_k.close = next_k.close  # Take the close of the last k-line in the inclusion sequence
            current_k.open = current_k.open  # Keep the open of the first k-line in the inclusion sequence

        # Check for inclusion: current_k is included in next_k
        elif current_k.high <= next_k.high and current_k.low >= next_k.low:
            # next_k becomes the new current_k
            current_k = KLine(next_k.date, next_k.open, next_k.high, next_k.low, next_k.close, next_k.index_in_df)
            # Direction for merge would be determined by next_k vs current_k (as K_{i-1})
            # This case is less common in sequential processing if strictly following "merge K_i with K_{i+1}"
            # For this implementation, if K_i is swallowed by K_{i+1}, K_{i+1} effectively replaces K_i.
        else:
            # No inclusion, current_k is finalized
            processed_k_lines.append(current_k)
            current_k = next_k  # Move to the next k-line

    processed_k_lines.append(current_k)  # Add the last k-line
    return processed_k_lines


### B. 分型识别函数 (`identify_fractals`)
def identify_fractals(processed_k_lines):
    fractals =
    if len(processed_k_lines) < 3:
        return fractals

    for i in range(1, len(processed_k_lines) - 1):
        k1 = processed_k_lines[i - 1]
        k2 = processed_k_lines[i]
        k3 = processed_k_lines[i + 1]

        # Top Fractal
        is_top_fractal = (k2.high >= k1.high and k2.high >= k3.high and
                          k2.low >= k1.low and k2.low >= k3.low)
        # Ensure k2 is strictly higher than at least one neighbor if they have same highs
        # or ensure it's a clear peak.
        # A common stricter rule: k2.high > k1.high and k2.high > k3.high
        # For this implementation, we use the definition from [51]/[51]: highest high AND highest low.
        if is_top_fractal:
            # Check for strictness: if k2.high == k1.high, then k2.low must be > k1.low (or k1 is part of an earlier top)
            # if k2.high == k3.high, then k2.low must be > k3.low (or k3 is part of a later top)
            # This avoids flat tops being multiple fractals. For simplicity, the above definition is used.
            # Chan Lun's core definition from snippets is usually sufficient.
            fractals.append(Fractal(k2, "top", i))

        # Bottom Fractal
        is_bottom_fractal = (k2.low <= k1.low and k2.low <= k3.low and
                             k2.high <= k1.high and k2.high <= k3.high)
        # Similar strictness consideration for bottoms.
        if is_bottom_fractal:
            fractals.append(Fractal(k2, "bottom", i))

    return fractals


### C. 笔构建函数 (`construct_strokes`)
def construct_strokes(fractals, processed_k_lines):
    strokes =
    if len(fractals) < 2:
        return strokes

    last_confirmed_fractal = None
    potential_stroke_start_fractal = fractals

    for i in range(1, len(fractals)):
        current_fractal = fractals[i]

        # Ensure alternating fractal types
        if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
            # If same type, choose the "better" one (higher top, lower bottom)
            if current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price:
                potential_stroke_start_fractal = current_fractal
            elif current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price:
                potential_stroke_start_fractal = current_fractal
            continue

        # Check K-line separation rule: at least 1 K-line between fractal K-lines
        # The fractals are k2 of (k1,k2,k3). So index_in_processed_k is the index of k2.
        # The K-lines of the start fractal are from index_in_processed_k-1 to index_in_processed_k+1
        # The K-lines of the end fractal are from current_fractal.index_in_processed_k-1 to current_fractal.index_in_processed_k+1
        # Number of K-lines between the two middle K-lines of the fractals:
        # (current_fractal.index_in_processed_k - 1) - (potential_stroke_start_fractal.index_in_processed_k + 1) + 1
        # = current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k - 1
        # This count is for K-lines strictly *between* the two fractals' K-lines.
        # Chan Lun rule: "顶和底之间至少有一个K线不属于顶分型与底分型" [49]
        # This means the K-lines that form the fractals themselves are excluded.
        # The distance between the middle K-lines of the fractals must be >= 3
        # (e.g., F1_k2 at index i, F2_k2 at index j. K-lines between are i+1 to j-1. Count is (j-1)-(i+1)+1 = j-i-1. Need j-i-1 >= 1, so j-i >= 2.
        # But the fractals themselves are 3 K-lines. A more common interpretation is simply that the fractals' constituent K-lines don't overlap and there's at least one K-line between the end of the first fractal's 3-K-line pattern and the start of the second fractal's 3-K-line pattern.
        # A simpler check: distance between the indices of the middle K-lines of the fractals.
        # If F1 is at index `a` and F2 is at index `b`, then `b - a` must be at least 3 for one K-line between them.
        # (e.g., F1 is k_i, k_{i+1}, k_{i+2} with k_{i+1} as center. F2 is k_j, k_{j+1}, k_{j+2} with k_{j+1} as center)
        # We need k_{i+2} and k_j to be separated by at least one K-line.
        # So, index of k_j must be >= index of k_{i+2} + 2.
        # index_k_j = current_fractal.index_in_processed_k -1
        # index_k_i_plus_2 = potential_stroke_start_fractal.index_in_processed_k + 1
        # (current_fractal.index_in_processed_k -1) >= (potential_stroke_start_fractal.index_in_processed_k + 1) + 2
        # current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k >= 4

        # Simpler rule: number of K-bars between the two fractal points (inclusive of start, exclusive of end for count)
        # Number of K-lines between the *central* K-lines of the two fractals.
        # If start_idx is index of first fractal's central K, end_idx is for second.
        # K-lines between them are start_idx+1 to end_idx-1.
        # Count = (end_idx-1) - (start_idx+1) + 1 = end_idx - start_idx - 1.
        # This must be >= 1. So, end_idx - start_idx >= 2.
        # The definition of "at least one independent K-line" [50] is key.
        # This means the K-lines *between* the 3-bar fractal patterns.
        # If fractal 1 ends at k_idx_f1_end and fractal 2 starts at k_idx_f2_start,
        # we need k_idx_f2_start - k_idx_f1_end -1 >= 1.
        # k_idx_f1_end = potential_stroke_start_fractal.index_in_processed_k + 1
        # k_idx_f2_start = current_fractal.index_in_processed_k - 1
        # (current_fractal.index_in_processed_k - 1) - (potential_stroke_start_fractal.index_in_processed_k + 1) - 1 >= 1
        # current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k - 3 >= 1
        # current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k >= 4

        k_lines_between_fractal_centers = current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k - 1
        if k_lines_between_fractal_centers < 1:  # This means centers are too close (e.g. index 1 and 2)
            # If they are too close, we might need to update the potential_stroke_start_fractal
            # if the current_fractal is "better" (e.g. higher top, lower bottom)
            # This logic is complex. For now, we assume distinct fractals are far enough.
            # A common interpretation is that the indices of the middle K-lines of the fractals must differ by at least 2.
            # (i.e., current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k >= 2)
            # Let's use a more direct interpretation of "at least one K-line between the fractals"
            # The fractals are 3 K-lines each.
            # End of first fractal pattern: potential_stroke_start_fractal.index_in_processed_k + 1
            # Start of second fractal pattern: current_fractal.index_in_processed_k - 1
            # We need (current_fractal.index_in_processed_k - 1) > (potential_stroke_start_fractal.index_in_processed_k + 1)
            # So, current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k > 2
            # i.e. current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k >= 3
            pass  # This condition is implicitly handled by fractal identification if they are truly distinct.
            # The key is that the fractals should not share K-lines.
            # The number of K-lines from start of first fractal to end of second fractal must be >= 5 (3 for first, 1 between, 3 for second, minus overlaps)
            # A simpler rule: the gap between the end of the first fractal's K-lines and start of second fractal's K-lines.
            # If F1 is (k_i, k_{i+1}, k_{i+2}) and F2 is (k_j, k_{j+1}, k_{j+2})
            # We need j > i+2.
            # index of k_j is current_fractal.index_in_processed_k - 1
            # index of k_{i+2} is potential_stroke_start_fractal.index_in_processed_k + 1
            # So, (current_fractal.index_in_processed_k - 1) > (potential_stroke_start_fractal.index_in_processed_k + 1)
            # current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k > 2 (i.e. >=3)

        if not (current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k >= 3):
            # Fractals are too close, or overlapping.
            # This implies the earlier fractal might not be valid in context of this new one, or vice-versa.
            # Chan Lun rules for this can be complex (e.g., choosing the "stronger" fractal).
            # For now, if too close, we might update the starting fractal if the current one is "better"
            if current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price and potential_stroke_start_fractal.fractal_type == "top":
                potential_stroke_start_fractal = current_fractal
            elif current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price and potential_stroke_start_fractal.fractal_type == "bottom":
                potential_stroke_start_fractal = current_fractal
            # If types are different but too close, this configuration is usually invalid for a stroke.
            # We might need to discard `potential_stroke_start_fractal` and make `current_fractal` the new potential start.
            # Or, if current_fractal is of the same type as a previous confirmed stroke's end, it might invalidate that stroke.
            # This part of Chan Lun (笔的连接与确认) is subtle.
            # A common simplification: if a new fractal forms that would invalidate a pending stroke, the pending stroke is cancelled.
            # If they are alternating and too close, it's often not a valid stroke.
            # Let's assume for now that our fractal list is "clean" enough that alternating types imply sufficient separation.
            # The ">=" 3 rule is a good heuristic for K-line separation.
            if not (current_fractal.fractal_type != potential_stroke_start_fractal.fractal_type and \
                    abs(current_fractal.index_in_processed_k - potential_stroke_start_fractal.index_in_processed_k) >= 3):  # Min 3 K-line diff for centers
                # If same type, update potential_stroke_start_fractal if current is stronger
                if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
                    if (
                            current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
                            (
                                    current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                        potential_stroke_start_fractal = current_fractal
                # If different type but too close, this is tricky.
                # It might mean the first fractal was not a true turning point for a stroke.
                # Or the current fractal is part of a smaller oscillation.
                # For simplicity, if they are alternating but too close, we update the start fractal
                # to the current one, effectively discarding the previous potential start.
                else:  # Different types, but too close
                    potential_stroke_start_fractal = current_fractal
                continue

        # Check fractal integrity (Rule 2 from outline)
        valid_stroke = False
        if potential_stroke_start_fractal.fractal_type == "bottom" and current_fractal.fractal_type == "top":  # Up-stroke
            if current_fractal.price > potential_stroke_start_fractal.price:  # Top fractal high > Bottom fractal high (this is not the rule)
                # Rule: Top fractal's high > Bottom fractal's high (price of fractal)
                valid_stroke = True
        elif potential_stroke_start_fractal.fractal_type == "top" and current_fractal.fractal_type == "bottom":  # Down-stroke
            if current_fractal.price < potential_stroke_start_fractal.price:  # Bottom fractal low < Top fractal low
                valid_stroke = True

        if valid_stroke:
            # Check if this new stroke conflicts with the last confirmed stroke (if any)
            # A new stroke must start from the end of the last confirmed stroke.
            new_stroke = Stroke(potential_stroke_start_fractal, current_fractal, processed_k_lines)

            if not strokes:  # First stroke
                strokes.append(new_stroke)
                last_confirmed_fractal = new_stroke.end_fractal
                potential_stroke_start_fractal = new_stroke.end_fractal  # Next stroke starts from here
            else:
                last_stroke = strokes[-1]
                # Ensure new stroke starts where the last one ended and alternates type
                if new_stroke.start_fractal.date == last_stroke.end_fractal.date and \
                        new_stroke.start_fractal.price == last_stroke.end_fractal.price and \
                        new_stroke.type != last_stroke.type:
                    strokes.append(new_stroke)
                    last_confirmed_fractal = new_stroke.end_fractal
                    potential_stroke_start_fractal = new_stroke.end_fractal
                else:
                    # Conflict: this new fractal might invalidate the last stroke or start a new sequence
                    # This is where Chan Lun's stroke combination rules get complex.
                    # E.g., if a new fractal forms that is "stronger" than the end of the last stroke.
                    # For this version, we assume a simpler sequential confirmation.
                    # If it doesn't connect, we see if the current_fractal is a better endpoint for the last stroke
                    # or if potential_stroke_start_fractal should be updated.

                    # If the current_fractal is of the same type as last_stroke.end_fractal but "better"
                    if current_fractal.fractal_type == last_stroke.end_fractal.fractal_type:
                        if (
                                current_fractal.fractal_type == "top" and current_fractal.price > last_stroke.end_fractal.price) or \
                                (
                                        current_fractal.fractal_type == "bottom" and current_fractal.price < last_stroke.end_fractal.price):
                            # Try to form a new stroke from last_stroke.start_fractal to current_fractal
                            # This means the last_stroke is "updated"
                            updated_last_stroke = Stroke(last_stroke.start_fractal, current_fractal, processed_k_lines)
                            if updated_last_stroke.type == last_stroke.type:  # Must be same direction
                                strokes[-1] = updated_last_stroke
                                last_confirmed_fractal = updated_last_stroke.end_fractal
                                potential_stroke_start_fractal = updated_last_stroke.end_fractal
                            else:  # Should not happen if types are same
                                potential_stroke_start_fractal = current_fractal  # Reset
                        else:  # Not better, current fractal might start a new sequence or be ignored
                            potential_stroke_start_fractal = current_fractal
                    else:  # Different type, but doesn't connect to last_stroke.end_fractal
                        # This means the potential_stroke_start_fractal was not last_stroke.end_fractal
                        # This implies a break in sequence or a need to re-evaluate previous strokes.
                        # For simplicity, we reset potential_stroke_start_fractal
                        potential_stroke_start_fractal = current_fractal


        else:  # Not a valid stroke (e.g. integrity rule failed)
            # If current fractal is "stronger" than potential_stroke_start_fractal and same type, update start
            if current_fractal.fractal_type == potential_stroke_start_fractal.fractal_type:
                if (
                        current_fractal.fractal_type == "top" and current_fractal.price > potential_stroke_start_fractal.price) or \
                        (
                                current_fractal.fractal_type == "bottom" and current_fractal.price < potential_stroke_start_fractal.price):
                    potential_stroke_start_fractal = current_fractal
            # If different type, this current_fractal becomes the new potential_stroke_start_fractal
            # because the previous one couldn't form a valid stroke with it.
            else:
                potential_stroke_start_fractal = current_fractal

    # Refine strokes: Ensure no overlapping K-lines between strokes and strict alternation
    # This is a complex part of Chan theory, often involving iterative refinement.
    # A common issue is that a new fractal might invalidate a previously confirmed stroke.
    # The above loop is a greedy approach. A more robust method might involve backtracking or a state machine.

    # Simple post-processing for strict alternation if greedy approach failed
    if not strokes: return

    refined_strokes = [strokes]
    for k in range(1, len(strokes)):
        prev_s = refined_strokes[-1]
        curr_s = strokes[k]
        # Ensure curr_s starts where prev_s ended and types alternate
        if curr_s.start_fractal.date == prev_s.end_fractal.date and \
                curr_s.start_fractal.price == prev_s.end_fractal.price and \
                curr_s.type != prev_s.type:
            refined_strokes.append(curr_s)
        # else: current stroke is invalid in sequence, try to see if it can start a new sequence
        # from prev_s.start_fractal if it's a "better" end than prev_s.end_fractal
        # This is complex. For now, just ensure strict connection.

    return refined_strokes


### D. 线段构建函数 (`construct_line_segments`)
# (简化版，主要基于三笔重叠，未完全实现特征序列破坏)
def construct_line_segments(strokes):
    segments =
    if len(strokes) < 3:
        return segments

    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]

        # Check for basic line segment formation: 3 strokes, s1 and s3 same direction, s2 opposite
        if s1.type == s3.type and s1.type != s2.type:
            # Check overlap and progression for an "up" segment (s1 up, s2 down, s3 up)
            if s1.type == "up":
                # s2 low should not be below s1 start_price (or s1 low)
                # s3 end_price should be above s1 end_price (or s1 high)
                # Overlap: s2 range must overlap with s1 range. s3 range must overlap with s2 range.
                # A common definition of overlap for segment:
                # For up-segment (s1 up, s2 down, s3 up):
                # s2.low > s1.start_price (bottom of s1)
                # s2.high < s1.end_price (top of s1) -- this is for s2 to be contained, not general overlap
                # More generally: max(s1.low, s2.low) < min(s1.high, s2.high) for overlap

                # Segment condition: s3 must break s1's high
                # And s2's low must not break s1's low (more strictly, s1's start fractal low)
                if s3.end_price > s1.end_price and s2.end_price > s1.start_price:  # s2.end_price is bottom of s2
                    # Potential up-segment: s1, s2, s3
                    # Look for more strokes to extend this segment
                    current_segment_strokes = [s1, s2, s3]
                    j = i + 3
                    while j < len(strokes) - 1:  # Need at least two more strokes (s_next_opposite, s_next_same_dir)
                        s_next_opposite = strokes[j]
                        s_next_same_dir = strokes[j + 1]

                        if s_next_opposite.type == s2.type and s_next_same_dir.type == s1.type:
                            # Check if s_next_same_dir continues the segment
                            # (i.e., breaks the high of current_segment_strokes[-1].end_price)
                            # and s_next_opposite does not break the low of the segment
                            if s_next_same_dir.end_price > current_segment_strokes[-1].end_price and \
                                    s_next_opposite.end_price > current_segment_strokes.start_price:  # Simplified check
                                current_segment_strokes.extend([s_next_opposite, s_next_same_dir])
                                j += 2
                            else:  # Segment ends
                                break
                        else:  # Pattern broken
                            break

                    segments.append(Segment(current_segment_strokes, "up"))
                    i = j  # Move past the consumed strokes for this segment
                    continue

            # Check for "down" segment (s1 down, s2 up, s3 down)
            elif s1.type == "down":
                if s3.end_price < s1.end_price and s2.end_price < s1.start_price:  # s2.end_price is top of s2
                    current_segment_strokes = [s1, s2, s3]
                    j = i + 3
                    while j < len(strokes) - 1:
                        s_next_opposite = strokes[j]
                        s_next_same_dir = strokes[j + 1]
                        if s_next_opposite.type == s2.type and s_next_same_dir.type == s1.type:
                            if s_next_same_dir.end_price < current_segment_strokes[-1].end_price and \
                                    s_next_opposite.end_price < current_segment_strokes.start_price:  # Simplified
                                current_segment_strokes.extend([s_next_opposite, s_next_same_dir])
                                j += 2
                            else:
                                break
                        else:
                            break
                    segments.append(Segment(current_segment_strokes, "down"))
                    i = j
                    continue
        i += 1  # If no segment found starting at i, move to next stroke
    return segments


### E. 中枢识别函数 (`identify_pivots`)
def identify_pivots(strokes):
    pivots =
    if len(strokes) < 3:
        return pivots

    # This is a simplified pivot identification based on 3 overlapping strokes.
    # Chan Lun pivot definition is more nuanced (次级别走势类型, extensions, etc.)

    # Iterate through strokes to find sequences of 3 (or more) that form a pivot
    # A common way: find first 3 strokes s1, s2, s3. If they overlap, form a pivot.
    # Then check if s4, s5 extend it, etc.

    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]

        # Check for alternating types for basic pivot structure
        if s1.type != s2.type and s2.type != s3.type:  # e.g. up-down-up or down-up-down
            # Calculate overlap for these three strokes
            # For up-down-up: s1(up), s2(down), s3(up)
            # ZD = max(s1.low, s3.low) (More accurately, max of lows of up-strokes)
            # ZG = s2.high (More accurately, min of highs of down-strokes)

            # General definition:
            # GG = max of all stroke lows in the pivot candidate strokes
            # DD = min of all stroke highs in the pivot candidate strokes
            # ZD = max(low of up-strokes in pivot)
            # ZG = min(high of down-strokes in pivot)

            # For s1, s2, s3:
            if s1.type == "up":  # s1(up), s2(down), s3(up)
                zd_candidate = max(s1.low, s3.low)  # Max of the lows of the up strokes
                zg_candidate = s2.high  # Min of the highs of the down strokes (only s2 here)
            else:  # s1(down), s2(up), s3(down)
                zd_candidate = s2.low  # Max of the lows of the up strokes (only s2 here)
                zg_candidate = min(s1.high, s3.high)  # Min of the highs of the down strokes

            # Check for overlap: ZD must be less than ZG
            if zd_candidate < zg_candidate:
                # Potential pivot found with s1, s2, s3
                current_pivot_strokes = [s1, s2, s3]
                current_zd = zd_candidate
                current_zg = zg_candidate
                start_date = s1.start_date
                end_date = s3.end_date

                # Try to extend the pivot with more strokes (up to 9 typically)
                j = i + 3
                while j < len(strokes) and len(current_pivot_strokes) < 9:  # Max 9 strokes in a standard pivot
                    next_stroke = strokes[j]
                    # Next stroke must continue the zig-zag and its range must overlap/touch the ZD/ZG
                    if next_stroke.type != current_pivot_strokes[-1].type:
                        # Update ZD/ZG if next_stroke extends the pivot
                        temp_zd = current_zd
                        temp_zg = current_zg

                        current_pivot_strokes.append(next_stroke)

                        # Recalculate ZD and ZG for all strokes in current_pivot_strokes
                        up_stroke_lows = [s.low for s in current_pivot_strokes if s.type == "up"]
                        down_stroke_highs = [s.high for s in current_pivot_strokes if s.type == "down"]

                        if not up_stroke_lows or not down_stroke_highs:  # Should not happen if alternating
                            break

                        new_zd = max(up_stroke_lows) if up_stroke_lows else -np.inf
                        new_zg = min(down_stroke_highs) if down_stroke_highs else np.inf

                        if new_zd < new_zg:  # Still a valid pivot
                            current_zd = new_zd
                            current_zg = new_zg
                            end_date = next_stroke.end_date
                            j += 1
                        else:  # Adding next_stroke invalidates pivot, so pivot ended before it
                            current_pivot_strokes.pop()  # Remove next_stroke
                            break
                    else:  # Type doesn't alternate, pivot ends
                        break

                pivots.append(Pivot(list(current_pivot_strokes), current_zg, current_zd, start_date, end_date))
                # Advance i past the strokes included in this pivot
                # i should start at the beginning of the stroke that breaks the pivot, or after the last stroke of the pivot.
                # For simplicity, advance by number of strokes in pivot. This might miss overlapping pivots.
                i += len(current_pivot_strokes)
            else:  # No overlap for s1,s2,s3
                i += 1
        else:  # Types not alternating for s1,s2,s3
            i += 1

    return pivots


### F. 背驰检测函数 (`detect_divergence_macd`)
def detect_divergence_macd(df_with_macd, strokes_or_segments, pivots):
    """
    Detects divergence using MACD.
    df_with_macd should have 'macd', 'macdsignal', 'macdhist' columns.
    strokes_or_segments: list of Stroke or Segment objects.
    pivots: list of Pivot objects.
    Returns a list of divergence signals (e.g., (date, type_of_divergence, stroke_indices_involved))
    """
    divergences =
    if df_with_macd is None or strokes_or_segments is None or len(strokes_or_segments) < 2:
        return divergences

    # MACD calculation if not already present (example)
    if 'macd' not in df_with_macd.columns:
        close_prices = df_with_macd['close']
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df_with_macd['macd'] = macd
        df_with_macd['macdsignal'] = macdsignal
        df_with_macd['macdhist'] = macdhist

    # 1. Trend Divergence (using strokes/segments)
    # Needs at least two consecutive segments/strokes of the same type after a trend is established
    # (e.g. after two pivots in same direction)
    # For simplicity, let's check divergence between any two comparable strokes/segments
    for i in range(len(strokes_or_segments) - 1):
        item1 = strokes_or_segments[i]
        # Find next comparable item (same type, separated by one opposite type)
        if i + 2 < len(strokes_or_segments):
            item2 = strokes_or_segments[i + 2]
            if item1.type == item2.type and item1.type != strokes_or_segments[i + 1].type:
                # Get MACD hist sum for item1 and item2
                # Need original indices from KLine objects within strokes/segments
                macd_hist_sum1 = df_with_macd['macdhist'].iloc[
                                 item1.start_kline_original_idx: item1.end_kline_original_idx + 1].sum()
                macd_hist_sum2 = df_with_macd['macdhist'].iloc[
                                 item2.start_kline_original_idx: item2.end_kline_original_idx + 1].sum()

                if item1.type == "up":  # Potential top divergence
                    if item2.end_price > item1.end_price and macd_hist_sum2 < macd_hist_sum1:  # Price higher, MACD lower
                        divergences.append({
                            "date": item2.end_date,
                            "type": "trend_top_divergence",
                            "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                            "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2
                        })
                elif item1.type == "down":  # Potential bottom divergence
                    if item2.end_price < item1.end_price and macd_hist_sum2 > macd_hist_sum1:  # Price lower, MACD (abs value) lower or MACD value higher
                        # For macdhist (green bars are negative): sum of green bars for item2 is less negative (closer to 0)
                        # So, macd_hist_sum2 (which is negative) > macd_hist_sum1 (more negative)
                        divergences.append({
                            "date": item2.end_date,
                            "type": "trend_bottom_divergence",
                            "item1_end_price": item1.end_price, "item2_end_price": item2.end_price,
                            "item1_macd_sum": macd_hist_sum1, "item2_macd_sum": macd_hist_sum2
                        })

    # 2. Consolidation Divergence (related to pivots)
    # This requires identifying moves out of a pivot and comparing their strength.
    # Example: move1 leaves pivot downwards, move2 leaves pivot downwards again to a new low.
    # This is more complex as it requires tracking pivot exits.
    if pivots:
        for pivot in pivots:
            # Find strokes leaving this pivot
            strokes_after_pivot_start = [s for s in strokes_or_segments if
                                         s.start_date >= pivot.start_date]  # Simplified

            departures_down =
            departures_up =

            last_stroke_in_pivot_idx = -1
            for idx, s in enumerate(strokes_after_pivot_start):
                if s.end_date <= pivot.end_date:  # Stroke is part of pivot formation or ends within
                    last_stroke_in_pivot_idx = idx
                    continue

                # Stroke starts after or within pivot, and ends outside
                # This is a departure stroke
                prev_stroke_in_pivot = strokes_after_pivot_start[
                    last_stroke_in_pivot_idx] if last_stroke_in_pivot_idx != -1 else None

                if s.type == "down" and s.end_price < pivot.zd:  # Departure downwards
                    # Check if it's a new low compared to previous departures from this pivot
                    departures_down.append(s)
                elif s.type == "up" and s.end_price > pivot.zg:  # Departure upwards
                    departures_up.append(s)

            if len(departures_down) >= 2:
                s1_down, s2_down = departures_down, departures_down  # Simplistic: first two
                # Ensure s2_down is a new low after s1_down
                if s2_down.end_price < s1_down.end_price:
                    macd_hist_sum1_down = df_with_macd['macdhist'].iloc[
                                          s1_down.start_kline_original_idx:s1_down.end_kline_original_idx + 1].sum()
                    macd_hist_sum2_down = df_with_macd['macdhist'].iloc[
                                          s2_down.start_kline_original_idx:s2_down.end_kline_original_idx + 1].sum()
                    if macd_hist_sum2_down > macd_hist_sum1_down:  # MACD higher (less negative)
                        divergences.append({
                            "date": s2_down.end_date,
                            "type": "consolidation_bottom_divergence",
                            "pivot_zd": pivot.zd, "s1_price": s1_down.end_price, "s2_price": s2_down.end_price,
                            "s1_macd_sum": macd_hist_sum1_down, "s2_macd_sum": macd_hist_sum2_down
                        })
            if len(departures_up) >= 2:
                s1_up, s2_up = departures_up, departures_up
                if s2_up.end_price > s1_up.end_price:
                    macd_hist_sum1_up = df_with_macd['macdhist'].iloc[
                                        s1_up.start_kline_original_idx:s1_up.end_kline_original_idx + 1].sum()
                    macd_hist_sum2_up = df_with_macd['macdhist'].iloc[
                                        s2_up.start_kline_original_idx:s2_up.end_kline_original_idx + 1].sum()
                    if macd_hist_sum2_up < macd_hist_sum1_up:  # MACD lower
                        divergences.append({
                            "date": s2_up.end_date,
                            "type": "consolidation_top_divergence",
                            "pivot_zg": pivot.zg, "s1_price": s1_up.end_price, "s2_price": s2_up.end_price,
                            "s1_macd_sum": macd_hist_sum1_up, "s2_macd_sum": macd_hist_sum2_up
                        })
    return divergences


### G. 三类买卖点识别函数 (`find_trading_signals`)
def find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw):
    signals =  # Store as dicts: {"date": date, "signal_type": "1B/1S/2B/2S/3B/3S", "price": price, "details":...}

    # Ensure all inputs are sorted by date if they come from different processing steps
    # This implementation assumes inputs are chronologically ordered.

    # First Class Buy/Sell Points (from Divergences)
    for div in divergences:
        price_at_divergence = None
        # Find the K-line corresponding to the divergence date to get the price
        # The divergence date is div['date'], which is an end_date of a stroke/segment
        # The actual buy/sell point is the extreme of that stroke/segment.

        # Find the stroke/segment that ended on div['date'] and caused this divergence.
        # This requires linking divergence back to the specific stroke.
        # The current divergence structure is simplified.
        # For now, let's assume div['date'] is the date of the signal.
        # The price would be the high/low of the K-line on that date, or the extreme of the involved stroke.

        # A more robust way: the divergence calculation should store the stroke causing it.
        # For now, we'll use the price from the divergence dict if available.

        if div["type"] == "trend_bottom_divergence" or div["type"] == "consolidation_bottom_divergence":
            price_at_divergence = div.get("s2_price")  # This is the low point of the divergence
            if price_at_divergence is not None:
                signals.append({"date": div["date"], "signal_type": "1B", "price": price_at_divergence, "details": div})
        elif div["type"] == "trend_top_divergence" or div["type"] == "consolidation_top_divergence":
            price_at_divergence = div.get("s2_price")  # This is the high point of the divergence
            if price_at_divergence is not None:
                signals.append({"date": div["date"], "signal_type": "1S", "price": price_at_divergence, "details": div})

    # Second and Third Class Buy/Sell Points (require pivots and strokes/segments)
    # This logic is complex and requires careful state tracking.

    # Iterate through strokes/segments and pivots chronologically
    # This is a simplified conceptual outline for 2nd/3rd points. Full implementation is extensive.

    # For Second Buy (2B):
    # 1. Identify a 1B signal. Let its date be D_1B, price P_1B.
    # 2. Find the *previous relevant pivot* (P_prev) that led to the downward move ending in 1B.
    # 3. After 1B, price rallies (up-stroke/segment S_up).
    # 4. S_up's high does NOT break P_prev.ZG.
    # 5. Price then falls back (down-stroke/segment S_down).
    # 6. S_down's low (P_2B) is HIGHER than P_1B. This P_2B is the 2B.
    # (Similar logic for 2S)

    # For Third Buy (3B):
    # 1. Identify a pivot (P_current).
    # 2. Price breaks out upwards from P_current.ZG (up-stroke/segment S_breakout).
    # 3. Price then pulls back (down-stroke/segment S_pullback).
    # 4. S_pullback's low does NOT re-enter P_current (i.e., low > P_current.ZG).
    # 5. The low of S_pullback is the 3B.
    # (Similar logic for 3S)

    # Placeholder for 2nd/3rd buy/sell points - requires more intricate logic
    # This would involve iterating through strokes and pivots, checking conditions relative to prior signals.

    # Example logic sketch for 3B (highly simplified):
    if pivots and strokes:
        for p_idx, pivot in enumerate(pivots):
            # Find first stroke that clearly exits the pivot upwards
            for s_idx, stroke in enumerate(strokes):
                if stroke.start_date > pivot.end_date and stroke.type == "up" and stroke.low > pivot.zg:  # Breakout stroke
                    # Now look for a pullback stroke
                    if s_idx + 1 < len(strokes):
                        pullback_stroke = strokes[s_idx + 1]
                        if pullback_stroke.type == "down" and pullback_stroke.end_price > pivot.zg:  # Pullback stays above ZG
                            # Potential 3B at pullback_stroke.end_price (the low of the pullback)
                            signals.append({
                                "date": pullback_stroke.end_date,
                                "signal_type": "3B",
                                "price": pullback_stroke.end_price,
                                "details": f"Pivot ZG: {pivot.zg}, Breakout stroke end: {stroke.end_price}"
                            })
                            break  # Found one 3B for this pivot breakout
            # Similar for 3S

    # Sort signals by date
    signals.sort(key=lambda x: x["date"])
    return signals


### H. 主策略编排函数 (`run_chanlun_strategy`)
def run_chanlun_strategy(df_raw):
    if df_raw is None or len(df_raw) < 20:  # Need enough data for MACD and structures
        print("Not enough data to run Chanlun strategy.")
        return

    # 1. K-Line Preprocessing
    processed_k_lines = preprocess_k_lines(df_raw.copy())
    if not processed_k_lines:
        print("K-line preprocessing failed or resulted in no data.")
        return
    # print(f"Processed K-lines: {len(processed_k_lines)}")

    # 2. Fractal Identification
    fractals = identify_fractals(processed_k_lines)
    if not fractals:
        print("No fractals identified.")
        return
    # print(f"Fractals identified: {len(fractals)}")
    # for f in fractals[:5]: print(f.date, f.fractal_type, f.price)

    # 3. Stroke Construction
    # The stroke construction logic needs to be robust. The current one is a basic attempt.
    strokes = construct_strokes(fractals, processed_k_lines)
    if not strokes:
        print("No strokes constructed.")
        return
    # print(f"Strokes constructed: {len(strokes)}")
    # for s in strokes[:5]: print(s.start_date, s.end_date, s.type, s.start_price, s.end_price)

    # 4. Line Segment Construction (Simplified)
    # Segments are built from strokes. The current segment logic is very basic.
    segments = construct_line_segments(strokes)  # Using strokes as sub-level for segments
    # print(f"Segments constructed: {len(segments)}")
    # for seg in segments[:3]: print(seg.start_date, seg.end_date, seg.type, len(seg.strokes))

    # 5. Pivot Identification
    # Pivots are typically built from segments, or strokes if segments are not well-defined.
    # Using strokes directly for pivot identification is a common simplification for single-level analysis.
    pivots = identify_pivots(strokes)  # Using strokes to find pivots
    if not pivots:
        print("No pivots identified (or too few strokes).")
        # return # Strategy can proceed without pivots for 1st class signals if only strokes are used for divergence
    # print(f"Pivots identified: {len(pivots)}")
    # for p in pivots[:3]: print(p.start_date, p.end_date, p.zd, p.zg, len(p.strokes_in_pivot))

    # 6. Divergence Detection (using MACD)
    # Prepare DataFrame with MACD for divergence detection
    df_with_macd = df_raw.copy()
    close_prices = df_with_macd['close'].astype(float)  # Ensure float type for talib

    # Check for sufficient data for TALIB MACD
    if len(close_prices) < 34:  # Default slowperiod (26) + signalperiod (9) -1 approx
        print("Not enough data for MACD calculation.")
        divergences =
    else:
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df_with_macd['macd'] = macd
        df_with_macd['macdsignal'] = macdsignal
        df_with_macd['macdhist'] = macdhist
        divergences = detect_divergence_macd(df_with_macd, strokes, pivots)  # Use strokes for divergence
    # print(f"Divergences detected: {len(divergences)}")
    # for d in divergences: print(d)

    # 7. Identify Trading Signals
    # The find_trading_signals function needs to be more robust for 2nd and 3rd type.
    # Current version focuses on 1st type and a sketch of 3rd.
    trading_signals = find_trading_signals(processed_k_lines, strokes, segments, pivots, divergences, df_raw)
    # print(f"Trading signals found: {len(trading_signals)}")
    # for sig in trading_signals: print(sig)

    return trading_signals


### I. 完整代码清单与使用示例

```python
#
# ... (KLine, Fractal, Stroke, Segment, Pivot classes)...
# ... (preprocess_k_lines function)...
# ... (identify_fractals function)...
# ... (construct_strokes function)...
# ... (construct_line_segments function)...
# ... (identify_pivots function)...
# ... (detect_divergence_macd function)...
# ... (find_trading_signals function)...
# ... (run_chanlun_strategy function)...

# Usage Example:
if __name__ == '__main__':
    # Example: Fetch data for a stock
    symbol_example = "sh600519"  # Kweichow Moutai
    start_date_example = "20220101"
    end_date_example = "20231231"

    # Function to get data (defined earlier or assumed available)
    # def fetch_stock_data(symbol, start_date, end_date):...

    raw_data = fetch_stock_data(symbol_example, start_date_example, end_date_example)

    if raw_data is not None and not raw_data.empty:
        print(f"Fetched {len(raw_data)} rows for {symbol_example}")

        # Add a simple moving average to demonstrate data is present
        # raw_data = talib.SMA(raw_data['close'], timeperiod=20)
        # print(raw_data.tail())

        final_signals = run_chanlun_strategy(raw_data)

        if final_signals:
            print(f"\n--- Generated Trading Signals for {symbol_example} ---")
            for signal in final_signals:
                print(
                    f"Date: {signal['date']}, Type: {signal['signal_type']}, Price: {signal['price']:.2f}, Details: {signal.get('details', '')}")
        else:
            print(f"No trading signals generated for {symbol_example}.")

    else:
        print(f"Could not fetch data for {symbol_example}.")