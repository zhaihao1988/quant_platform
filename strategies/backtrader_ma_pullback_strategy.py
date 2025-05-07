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
        ('ma_peak_threshold', 1.20),
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
        # ... (next 方法中的买卖逻辑不变，使用Backtrader API和数据访问) ...
        current_date_str = self.datetime.date(0).isoformat();
        stock_name = self.data._name
        if len(self.ma_long) < 1 or len(self.ma_short) < 1: return
        if self.order: return  # 有挂单，不操作

        # Sell Logic
        if self.position:
            pos_manage_info = self.position_info;
            sell_reason = None
            if not pos_manage_info: return
            current_close = self.dataclose[0];
            current_high = self.datahigh[0];
            current_low = self.datalow[0];
            current_volume = self.datavolume[0]
            prev_volume = self.datavolume[-1] if len(self.datavolume) > 1 else 0;
            ma30_val = self.ma_long[0];
            ma5_val = self.ma_short[0]
            pos_manage_info['highest_price_since_entry'] = max(
                pos_manage_info.get('highest_price_since_entry', current_high), current_high)
            current_atr = self.atr[0] if not np.isnan(self.atr[0]) else (
                self.atr.atr[0] if hasattr(self.atr, 'atr') else np.nan)  # Handle both ATR types
            if not np.isnan(current_atr):
                new_atr_sl = pos_manage_info['highest_price_since_entry'] - self.params.atr_multiplier * current_atr
                if self.atr_stop_loss_price is None or new_atr_sl > self.atr_stop_loss_price: self.atr_stop_loss_price = new_atr_sl;
                pos_manage_info['atr_stop_loss_price'] = self.atr_stop_loss_price
            # Check Sells
            if self.atr_stop_loss_price is not None and current_low < self.atr_stop_loss_price: sell_reason = "stop_loss_atr"; self.log(
                f'SELL Trigger (ATR): Low {current_low:.2f} < ATR Stop {self.atr_stop_loss_price:.2f}')
            if not sell_reason and not np.isnan(ma30_val):
                sl_ma30 = ma30_val * (1 - self.params.sell_ma30_pct)
                if current_close < sl_ma30 and prev_volume > 1e-6 and current_volume >= prev_volume * self.params.sell_volume_ratio: sell_reason = "stop_loss_ma30_volume"; self.log(
                    f'SELL Trigger (MA30 Vol): Close {current_close:.2f} < SL {sl_ma30:.2f}, Vol Ratio {current_volume / prev_volume:.2f}')
            if not sell_reason and not np.isnan(ma5_val) and not np.isnan(ma30_val) and len(self.ma_short) > 1 and len(
                    self.ma_long) > 1 and not np.isnan(self.ma_short[-1]) and not np.isnan(self.ma_long[-1]):
                if self.ma_short[-1] >= self.ma_long[
                    -1] and ma5_val < ma30_val: sell_reason = "stop_loss_ma_dead_cross"; self.log(
                    f'SELL Trigger (Dead Cross): MA5 {ma5_val:.2f} < MA30 {ma30_val:.2f}')
            # Check TPs
            tp1, tp2, tp3 = pos_manage_info.get('tp1'), pos_manage_info.get('tp2'), pos_manage_info.get('tp3')
            hit = self.take_profit_targets_hit;
            initial_shares = pos_manage_info.get('initial_shares', 0);
            current_size = self.position.size
            if not sell_reason and not hit[0] and tp1 is not None and current_high >= tp1:
                sell_reason = "take_profit_1";
                shares_to_sell = initial_shares // 3;
                shares_to_sell = min(shares_to_sell, current_size)
                if shares_to_sell > 0: self.log(
                    f'SELL Trigger (TP1): High {current_high:.2f} >= TP1 {tp1:.2f}. Selling {shares_to_sell} shares.'); self.order = self.sell(
                    size=shares_to_sell); self.order._sell_reason = sell_reason; hit[0] = True
            elif not sell_reason and hit[0] and not hit[1] and tp2 is not None and current_high >= tp2:
                sell_reason = "take_profit_2";
                shares_to_sell = initial_shares // 3;
                shares_to_sell = min(shares_to_sell, current_size - self.shares_sold_tp1)
                if shares_to_sell > 0: self.log(
                    f'SELL Trigger (TP2): High {current_high:.2f} >= TP2 {tp2:.2f}. Selling {shares_to_sell} shares.'); self.order = self.sell(
                    size=shares_to_sell); self.order._sell_reason = sell_reason; hit[1] = True
            elif not sell_reason and all(hit[:2]) and not hit[2] and tp3 is not None and current_high >= tp3:
                sell_reason = "take_profit_3";
                shares_to_sell = current_size
                if shares_to_sell > 0: self.log(
                    f'SELL Trigger (TP3): High {current_high:.2f} >= TP3 {tp3:.2f}. Selling remaining {shares_to_sell} shares.'); self.order = self.sell(
                    size=shares_to_sell); self.order._sell_reason = sell_reason; hit[2] = True
            # Execute Stop Loss if needed
            if sell_reason and sell_reason.startswith("stop_loss") and self.position.size > 0: self.log(
                f'SELL Trigger (Stop Loss): Reason {sell_reason}. Closing {self.position.size}.'); self.order = self.close()
            # MACD Warning
            if TALIB_AVAILABLE and self.macd_hist and len(self.macd_hist) > 20:
                highs_20 = self.datahigh.get(ago=0, size=20);
                macd_hist_20 = self.macd_hist.get(ago=0, size=20)
                if highs_20 and macd_hist_20:
                    current_high_idx = np.argmax(highs_20);
                    if current_high_idx == len(highs_20) - 1:
                        macd_hist_at_high = macd_hist_20[current_high_idx];
                        prev_macd_hist_peak = max(macd_hist_20[:current_high_idx]) if current_high_idx > 0 else -np.inf
                        if pd.notna(macd_hist_at_high) and pd.notna(
                            prev_macd_hist_peak) and macd_hist_at_high < prev_macd_hist_peak * 0.8: self.log(
                            f'WARN: Potential MACD Top Divergence on {current_date_str}')

        # Buy Logic
        else:  # No position
            if len(self.ma_long) < self.params.trend_window or len(self.ma_short) < 1 or len(
                self.ma_long_for_peak) < 1: return
            peak_low_data = self._find_signal_peak_and_low();
            prior_peak, recent_low = peak_low_data["prior_peak"], peak_low_data["recent_low"]
            if prior_peak is None or recent_low is None: return
            is_trend_ok = False
            if len(self.ma_long.get(size=self.params.trend_window + 1)) == self.params.trend_window + 1:
                ma_long_series = pd.Series(self.ma_long.get(ago=1, size=self.params.trend_window))
                if not ma_long_series.isnull().any():
                    try:
                        coeffs = np.polyfit(np.arange(self.params.trend_window), ma_long_series.values[::-1],
                                            1); slope = coeffs[0]; is_trend_ok = slope >= -1e-6
                    except (np.linalg.LinAlgError, ValueError, TypeError):  # 添加了 TypeError
                        pass
            if not is_trend_ok: return
            if self.dataclose[0] <= self.ma_long[0]: return
            if self.ma_short[0] <= self.ma_long[0]: return
            if self.ma_long[0] <= 1e-6: return
            is_pullback = (self.dataclose[0] >= self.ma_long[0]) and (
                        (self.dataclose[0] - self.ma_long[0]) / self.ma_long[0] <= self.params.pullback_pct)
            if not is_pullback: return
            # --- Buy conditions met ---
            self.log(f'BUY SIGNAL on {current_date_str}')
            self._pending_buy_info = {"prior_peak": prior_peak, "recent_low": recent_low}
            cash = self.broker.get_cash();
            value = self.broker.get_value();
            target_value = value * self.params.max_position_ratio
            size = math.floor((target_value / self.dataclose[0]) / 100) * 100
            max_size_by_cash = math.floor((cash / self.dataclose[0]) / 100) * 100;
            size = min(size, max_size_by_cash)
            if size > 0:
                self.log(f'BUY CREATE, Size: {size}, Price ~ {self.dataclose[0]:.2f}'); self.order = self.buy(size=size)
            else:
                self.log(
                    f'Buy signal, but size is 0. Cash={cash:.2f}, TargetVal={target_value:.2f}'); del self._pending_buy_info