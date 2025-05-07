# strategies/backtrader_ma_pullback_strategy.py
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
import numpy as np
from datetime import datetime
import math

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARN: TA-Lib not found, MACD and ATR related logic will be disabled or use Backtrader's approximation.")


class MAPullbackPeakCondBtStrategy(bt.Strategy):
    params = (
        ('ma_short', 5),
        ('ma_long', 30),
        ('pullback_pct', 0.05),
        ('trend_window', 5),
        ('peak_window', 5),
        ('ma_peak_threshold', 1.25),
        ('ma_long_for_peak', 30),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('sell_ma30_pct', 0.03),
        ('sell_volume_ratio', 1.5),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('printlog', True),
        ('max_position_ratio', 0.20)
    )

    def log(self, txt, dt=None, doprint=False):
        # ... (日志函数实现不变) ...
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} [{self.data._name}], {txt}')

    def __init__(self):
        # ... (指标和状态变量初始化不变) ...
        self.dataclose = self.data.close
        self.datahigh = self.data.high
        self.datalow = self.data.low
        self.dataopen = self.data.open
        self.datavolume = self.data.volume
        self.datetime = self.data.datetime

        self.ma_short = btind.SimpleMovingAverage(self.dataclose, period=self.params.ma_short)
        self.ma_long = btind.SimpleMovingAverage(self.dataclose, period=self.params.ma_long)
        if self.params.ma_long_for_peak == self.params.ma_long:
            self.ma_long_for_peak = self.ma_long
        else:
            self.ma_long_for_peak = btind.SimpleMovingAverage(self.dataclose, period=self.params.ma_long_for_peak)

        self.atr = btind.AverageTrueRange(period=self.params.atr_period)

        if TALIB_AVAILABLE:
            self.macd_obj = btind.MACD(self.dataclose, period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
            self.macd_hist = self.macd_obj.macd - self.macd_obj.signal
        else:
            self.macd_obj = None; self.macd_hist = None

        self.order = None
        self.position_info = {}
        self.atr_stop_loss_price = None
        self.highest_high_since_entry = None
        self.take_profit_targets_hit = [False, False, False]
        self.shares_sold_tp1 = 0
        self.shares_sold_tp2 = 0

    def notify_order(self, order):
        # ... (订单通知处理逻辑不变) ...
        dt_str = self.datetime.date(0).isoformat();
        stock_name = self.data._name
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                entry_price = order.executed.price;
                initial_shares = order.executed.size
                if hasattr(self, '_pending_buy_info'):
                    peak_info = self._pending_buy_info;
                    prior_peak = peak_info['prior_peak'];
                    recent_low = peak_info['recent_low']
                    tp1, tp2, tp3 = (prior_peak + recent_low) / 2, prior_peak, prior_peak + (entry_price - recent_low)
                    self.position_info = {'entry_price': entry_price, 'initial_shares': initial_shares,
                                          'shares_held': initial_shares, 'prior_peak': prior_peak,
                                          'recent_low': recent_low, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3}
                    self.take_profit_targets_hit = [False, False, False];
                    self.shares_sold_tp1 = 0;
                    self.shares_sold_tp2 = 0
                    self.highest_high_since_entry = self.datahigh[0]
                    if TALIB_AVAILABLE and len(self.atr) > 1 and not np.isnan(self.atr[-1]):
                        self.atr_stop_loss_price = entry_price - self.params.atr_multiplier * self.atr[-1];
                        self.position_info['atr_stop_loss_price'] = self.atr_stop_loss_price; self.log(
                            f'   Initial ATR Stop Loss set at: {self.atr_stop_loss_price:.2f} (ATR={self.atr[-1]:.2f})')
                    else:
                        self.atr_stop_loss_price = None; self.position_info['atr_stop_loss_price'] = None
                    self.log(
                        f'   Position Info Recorded: Peak={prior_peak:.2f}, Low={recent_low:.2f}, TP1={tp1:.2f}, TP2={tp2:.2f}, TP3={tp3:.2f}')
                    del self._pending_buy_info
                else:
                    self.log(f'ERROR: BUY EXECUTED for {stock_name} but _pending_buy_info not found!')
            elif order.issell():
                sold_size = abs(order.executed.size);
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {sold_size}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                if self.position_info: self.position_info['shares_held'] = self.position.size
                if hasattr(order, '_sell_reason'):
                    if order._sell_reason == "take_profit_1":
                        self.shares_sold_tp1 = sold_size
                    elif order._sell_reason == "take_profit_2":
                        self.shares_sold_tp2 = sold_size
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order Failed/Canceled/Margin/Rejected/Expired - Status: {order.getstatusname()}')
            if order.isbuy() and hasattr(self, '_pending_buy_info'): del self._pending_buy_info
        self.order = None

    def notify_trade(self, trade):
        # ... (交易关闭通知处理逻辑不变) ...
        if not trade.isclosed: return
        self.log(f'TRADE CLOSED, PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        self.position_info = {};
        self.atr_stop_loss_price = None;
        self.highest_high_since_entry = None
        self.take_profit_targets_hit = [False, False, False];
        self.shares_sold_tp1 = 0;
        self.shares_sold_tp2 = 0

    def _is_rolling_peak(self, data_line, period, index):
        # ... (滚动顶点判断逻辑不变) ...
        if index < period // 2 or index >= len(data_line) - period // 2: return False
        val_at_index = data_line[index]
        for i in range(1, period // 2 + 1):
            if data_line[index - i] > val_at_index or data_line[index + i] > val_at_index: return False
        return True

    def _is_rolling_low(self, data_line, period, index):
        # ... (滚动低点判断逻辑不变) ...
        if index < period // 2 or index >= len(data_line) - period // 2: return False
        val_at_index = data_line[index]
        for i in range(1, period // 2 + 1):
            if data_line[index - i] < val_at_index or data_line[index + i] < val_at_index: return False
        return True

    def _find_signal_peak_and_low(self):
        # ... (查找前期顶点和近期低点逻辑不变，使用Backtrader数据访问方式) ...
        current_idx = len(self) - 1;
        peak_info = {"prior_peak": None, "recent_low": None}
        required_lookback = max(self.params.peak_window * 2, self.params.ma_long_for_peak + 10)
        if current_idx < required_lookback: return peak_info
        # Find Peak
        peak_found_idx = -1
        for i in range(current_idx - 1, current_idx - required_lookback, -1):  # i 是过去某根K线的绝对索引
            if self._is_rolling_peak(self.datahigh.array, self.params.peak_window, i):
                offset = i - current_idx  # 计算相对于当前K线的偏移量 (负数)
                potential_peak_price = self.datahigh[offset];  # 使用偏移量访问
                ma_peak_val = self.ma_long_for_peak[offset]  # 使用偏移量访问
                if not np.isnan(
                        ma_peak_val) and ma_peak_val > 0 and potential_peak_price >= ma_peak_val * self.params.ma_peak_threshold:
                    peak_info["prior_peak"] = potential_peak_price;
                    peak_found_idx = i;
                    break
        if peak_info["prior_peak"] is None: return peak_info
        # Find Low
        low_found_idx = -1
        # peak_found_idx 是前期高点的绝对索引
        # 我们从当前K线的前一根 (current_idx - 1) 回溯到 peak_found_idx 之后的那一根
        for i in range(current_idx - 1, peak_found_idx, -1):  # i 是过去某根K线的绝对索引
            if self._is_rolling_low(self.datalow.array, self.params.peak_window, i):
                offset = i - current_idx  # 计算相对于当前K线的偏移量 (负数)
                peak_info["recent_low"] = self.datalow[offset];  # 使用偏移量访问
                low_found_idx = i;
                break

        # (后续 .get() 和 self.datalow[-1] 的逻辑通常是正确的，因为它们使用了正确的相对偏移)
        if peak_info["recent_low"] is None and peak_found_idx < current_idx - 1:
            # The number of bars from (peak_found_idx + 1) to (current_idx - 1) inclusive
            # If current_idx - 1 == peak_found_idx, size should be 0.
            # The range starts 1 bar ago (index current_idx - 1)
            # The number of elements is (current_idx - 1) - (peak_found_idx + 1) + 1 = current_idx - 1 - peak_found_idx
            size_to_get = current_idx - 1 - peak_found_idx
            if size_to_get > 0:
                lows_in_range = self.datalow.get(ago=1, size=size_to_get)  # ago=1 is (current_idx-1)
                if lows_in_range: peak_info["recent_low"] = min(lows_in_range)

        if peak_info["recent_low"] is None and current_idx >= 1: peak_info["recent_low"] = self.datalow[
            -1]  # 正确，-1是前一根K线
        return peak_info

    def next(self):
        # ---- PRELIMINARY DEBUG LOG (保持不变, 用于确认 next 调用) ----
        _current_bar_stock_name = self.data._name
        _current_bar_date_str = self.datetime.date(0).isoformat()


        current_date_str = _current_bar_date_str
        stock_name = _current_bar_stock_name

        # --- SPECIFIC DEBUG LOG SETUP (保持不变) ---
        target_stock_debug = '603920'      # 您想要调试的股票代码
        target_date_debug = '2019-07-17'  # 您想要调试的日期
        do_specific_log = (stock_name == target_stock_debug and current_date_str == target_date_debug)

        if do_specific_log:
            self.log(f"--- Debugging {stock_name} on {current_date_str} ---", doprint=True)
        # --- END SPECIFIC DEBUG LOG SETUP ---

        if self.order:
            # 订单处理中，保留原来的日志逻辑
            if do_specific_log: self.log("Order pending, skipping next.", doprint=True)
            return

        # Sell Logic (保持不变)
        if self.position:
            # !!! 将您原来的卖出逻辑完整地复制到这里 !!!
            # (为了简洁，这里省略了卖出逻辑代码，请确保您策略文件中的卖出逻辑部分被保留)
            pos_manage_info = self.position_info; sell_reason = None
            if not pos_manage_info: return # Error check
            current_close = self.dataclose[0]; current_high = self.datahigh[0]; current_low = self.datalow[0]
            current_volume = self.datavolume[0]; prev_volume = self.datavolume[-1] if len(self.datavolume) > 1 else 0
            ma30_val = self.ma_long[0]; ma5_val = self.ma_short[0]
            # Update highest high
            pos_manage_info['highest_price_since_entry'] = max(pos_manage_info.get('highest_price_since_entry', current_high), current_high)
            self.highest_high_since_entry = pos_manage_info['highest_price_since_entry']
            # Update ATR Stop
            current_atr_val = self.atr[0] if not np.isnan(self.atr[0]) else (self.atr.atr[0] if hasattr(self.atr, 'atr') and not np.isnan(self.atr.atr[0]) else np.nan)
            if not np.isnan(current_atr_val) and self.highest_high_since_entry is not None:
                new_atr_sl = self.highest_high_since_entry - self.params.atr_multiplier * current_atr_val
                if self.atr_stop_loss_price is None or new_atr_sl > self.atr_stop_loss_price:
                    self.atr_stop_loss_price = new_atr_sl; pos_manage_info['atr_stop_loss_price'] = self.atr_stop_loss_price
                    # if do_specific_log: self.log(f"ATR Stop Loss updated to: {self.atr_stop_loss_price:.2f}", doprint=True) # 可选的更新日志
            # Check Stop Losses
            if self.atr_stop_loss_price is not None and current_low < self.atr_stop_loss_price: sell_reason = "stop_loss_atr"; self.log(f'SELL Trigger (ATR): Low {current_low:.2f} < ATR Stop {self.atr_stop_loss_price:.2f}')
            if not sell_reason and not np.isnan(ma30_val):
                sl_ma30 = ma30_val * (1 - self.params.sell_ma30_pct)
                if current_close < sl_ma30 and prev_volume > 1e-6 and current_volume >= prev_volume * self.params.sell_volume_ratio: sell_reason = "stop_loss_ma30_volume"; self.log(f'SELL Trigger (MA30 Vol): Close {current_close:.2f} < SL {sl_ma30:.2f}, Vol Ratio {current_volume / prev_volume:.2f}')
            if not sell_reason and not np.isnan(ma5_val) and not np.isnan(ma30_val) and len(self.ma_short) > 1 and len(self.ma_long) > 1 and not np.isnan(self.ma_short[-1]) and not np.isnan(self.ma_long[-1]):
                if self.ma_short[-1] >= self.ma_long[-1] and ma5_val < ma30_val: sell_reason = "stop_loss_ma_dead_cross"; self.log(f'SELL Trigger (Dead Cross): MA5 {ma5_val:.2f} < MA30 {ma30_val:.2f}')
            # Check Take Profits (only if no stop loss yet)
            if not sell_reason:
                tp1, tp2, tp3 = pos_manage_info.get('tp1'), pos_manage_info.get('tp2'), pos_manage_info.get('tp3')
                hit = self.take_profit_targets_hit; initial_shares = pos_manage_info.get('initial_shares', 0)
                current_position_size = self.position.size; shares_per_tp_tranche = math.floor(initial_shares / 3) if initial_shares > 0 else 0
                if shares_per_tp_tranche == 0 and initial_shares > 0: shares_per_tp_tranche = initial_shares # Sell all if cannot divide by 3
                # TP1
                if not hit[0] and tp1 is not None and current_high >= tp1:
                    shares_to_sell = min(shares_per_tp_tranche, current_position_size);
                    if shares_to_sell > 0: self.log(f'SELL Trigger (TP1): High {current_high:.2f} >= TP1 {tp1:.2f}. Selling {shares_to_sell} shares.'); self.order = self.sell(size=shares_to_sell); self.order._sell_reason = "take_profit_1"; hit[0] = True
                # TP2
                elif not hit[1] and tp2 is not None and current_high >= tp2 and current_position_size > 0:
                    remaining_after_tp1_sold = current_position_size # Use current size as remaining
                    shares_to_sell = min(shares_per_tp_tranche, remaining_after_tp1_sold)
                    if shares_to_sell > 0: self.log(f'SELL Trigger (TP2): High {current_high:.2f} >= TP2 {tp2:.2f}. Selling {shares_to_sell} shares.'); self.order = self.sell(size=shares_to_sell); self.order._sell_reason = "take_profit_2"; hit[1] = True
                # TP3
                elif not hit[2] and tp3 is not None and current_high >= tp3 and current_position_size > 0:
                    shares_to_sell = current_position_size # Sell all remaining
                    if shares_to_sell > 0: self.log(f'SELL Trigger (TP3): High {current_high:.2f} >= TP3 {tp3:.2f}. Selling remaining {shares_to_sell} shares.'); self.order = self.sell(size=shares_to_sell); self.order._sell_reason = "take_profit_3"; hit[2] = True
            # Execute Stop Loss if triggered AND no TP order placed this bar
            if sell_reason and sell_reason.startswith("stop_loss") and self.position.size > 0 and not self.order:
                self.log(f'SELL Trigger (Stop Loss FINAL): Reason {sell_reason}. Closing all {self.position.size} shares.'); self.order = self.close()
            # MACD Warning (Keep unchanged)
            if TALIB_AVAILABLE and self.macd_hist and len(self.macd_hist) > 20:
                 highs_20 = self.datahigh.get(ago=0, size=20); macd_hist_20 = self.macd_hist.get(ago=0, size=20)
                 if highs_20 and macd_hist_20 and len(highs_20) == 20 and len(macd_hist_20) == 20:
                     if self.datahigh[0] == max(highs_20):
                         if len(macd_hist_20[:-1]) > 0 :
                             prev_macd_hist_peak = max(h for h in macd_hist_20[:-1] if h is not None and not np.isnan(h)) if any(h is not None and not np.isnan(h) for h in macd_hist_20[:-1]) else -np.inf
                             current_macd_hist = self.macd_hist[0]
                             if pd.notna(current_macd_hist) and pd.notna(prev_macd_hist_peak) and current_macd_hist < prev_macd_hist_peak * 0.8 :
                                 self.log(f'WARN: Potential MACD Top Divergence on {current_date_str}')
            # End of Sell Logic Block
            pass # Placeholder to indicate end of if self.position block if needed

        # Buy Logic
        else:  # No position
            if do_specific_log:
                # 这些日志保持不变
                self.log(f"Entering BUY logic block.", doprint=True)
                log_len_ma_long = len(self.ma_long) if hasattr(self.ma_long, '__len__') else 'N/A'
                log_len_ma_short = len(self.ma_short) if hasattr(self.ma_short, '__len__') else 'N/A'
                log_len_ma_long_peak = len(self.ma_long_for_peak) if hasattr(self.ma_long_for_peak, '__len__') else 'N/A'
                self.log(f"Len ma_long: {log_len_ma_long}", doprint=True)
                self.log(f"Len ma_short: {log_len_ma_short}", doprint=True)
                self.log(f"Len ma_long_for_peak: {log_len_ma_long_peak}", doprint=True)

            # Early exit if indicators don't have minimal data (保持不变)
            if np.isnan(self.ma_long[0]) or np.isnan(self.ma_long[-1]) or np.isnan(self.ma_short[0]):
                 if do_specific_log: self.log(f"Condition Fail: Essential MA values not ready (NaN). MA30[0]={self.ma_long[0]}, MA30[-1]={self.ma_long[-1]}, MA5[0]={self.ma_short[0]}", doprint=True)
                 return
            # You might need a specific check for ma_long_for_peak as well if _find_signal_peak_and_low requires it early

            peak_low_data = self._find_signal_peak_and_low() # 保持不变
            prior_peak, recent_low = peak_low_data["prior_peak"], peak_low_data["recent_low"]

            if do_specific_log: # 保持不变
                self.log(f"Peak/Low Data: prior_peak={prior_peak}, recent_low={recent_low}", doprint=True)

            if prior_peak is None or recent_low is None: # 保持不变
                if do_specific_log: self.log(f"Condition Fail: prior_peak or recent_low is None.", doprint=True)
                return
            if prior_peak <= recent_low : # 保持不变
                if do_specific_log: self.log(f"Condition Fail: prior_peak ({prior_peak}) <= recent_low ({recent_low}). Invalid signal.", doprint=True)
                return

            # ----- !!! MODIFIED Trend Check Logic START !!! -----
            is_trend_ok = False # Default state
            ma_curr_raw = self.ma_long[0]
            ma_prev_raw = self.ma_long[-1]

            # Check for NaN should have been caught above, but double check is fine
            if not np.isnan(ma_curr_raw) and not np.isnan(ma_prev_raw):
                ma_curr_rounded = round(ma_curr_raw, 2)
                ma_prev_rounded = round(ma_prev_raw, 2)

                is_trend_ok = ma_curr_rounded >= ma_prev_rounded # The new comparison logic

                if do_specific_log: # Keep the specific log for the NEW check
                    self.log(f"Trend check (MA30 Round Compare): MA30[0]={ma_curr_rounded:.2f}, MA30[-1]={ma_prev_rounded:.2f}, is_trend_ok={is_trend_ok}", doprint=True)
            else:
                 if do_specific_log:
                     self.log(f"Trend check (MA30 Round Compare): Failed due to NaN value. MA30[0]={ma_curr_raw}, MA30[-1]={ma_prev_raw}", doprint=True)

            # Check if trend condition failed (保持不变)
            if not is_trend_ok:
                if do_specific_log:
                     self.log(f"Condition Fail: Trend is not OK (is_trend_ok={is_trend_ok}).", doprint=True)
                return # Exit if trend condition not met
            # ----- !!! MODIFIED Trend Check Logic END !!! -----

            # --- Subsequent Buy Conditions (保持不变) ---
            current_close = self.dataclose[0]
            val_ma_short = self.ma_short[0]
            val_ma_long = self.ma_long[0] # Use this for consistency below

            if do_specific_log:
                self.log(f"Subsequent Checks: Close: {current_close:.2f}, MA_Short: {val_ma_short:.2f}, MA_Long: {val_ma_long:.2f}", doprint=True)

            if current_close <= val_ma_long:
                if do_specific_log: self.log(f"Condition Fail: Close ({current_close:.2f}) <= MA_Long ({val_ma_long:.2f}).", doprint=True)
                return

            if val_ma_short <= val_ma_long:
                if do_specific_log: self.log(f"Condition Fail: MA_Short ({val_ma_short:.2f}) <= MA_Long ({val_ma_long:.2f}).", doprint=True)
                return

            if val_ma_long <= 1e-6:
                if do_specific_log: self.log(f"Condition Fail: MA_Long ({val_ma_long:.2f}) is too small.", doprint=True)
                return

            pullback_value = (current_close - val_ma_long) / val_ma_long
            is_pullback = (current_close >= val_ma_long) and (pullback_value <= self.params.pullback_pct)

            if do_specific_log:
                self.log(f"Pullback check: current_close ({current_close:.2f}) >= MA_Long ({val_ma_long:.2f}) is {current_close >= val_ma_long}", doprint=True)
                self.log(f"Pullback value ( (C-MA)/MA ): {pullback_value:.4f}, pullback_pct_param: {self.params.pullback_pct:.4f}", doprint=True)
                self.log(f"Is_pullback: {is_pullback}", doprint=True)

            if not is_pullback:
                if do_specific_log: self.log(f"Condition Fail: Not a pullback (is_pullback={is_pullback}).", doprint=True)
                return

            # --- Buy conditions met (保持不变) ---
            # Log BUY SIGNAL first
            self.log(f'BUY SIGNAL on {current_date_str} for {stock_name}') # 这是您要保留的日志
            # Store peak/low info for use in notify_order
            self._pending_buy_info = {"prior_peak": prior_peak, "recent_low": recent_low}

            # Calculate size (保持不变)
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            target_position_value = value * self.params.max_position_ratio
            size = 0
            if current_close > 0:
                 size_based_on_target_value = math.floor((target_position_value / current_close) / 100.0) * 100
                 size_based_on_cash = math.floor((cash / current_close) / 100.0) * 100
                 size = min(size_based_on_target_value, size_based_on_cash)

            # Create Buy Order (保持不变)
            if size > 0:
                self.log(f'BUY CREATE, Size: {size}, Price ~ {current_close:.2f}')
                self.order = self.buy(size=size)
            else:
                self.log(f'Buy signal, but calculated size is 0. Cash={cash:.2f}, TargetPosVal={target_position_value:.2f}, Close={current_close:.2f}')
                if hasattr(self, '_pending_buy_info'):
                    del self._pending_buy_info # Clean up if no order placed