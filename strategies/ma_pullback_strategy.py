# strategies/ma_pullback_strategy.py
import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date

# --- Python Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
# --- End Path Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from utils.data_loader import load_daily_data
    from data_processing.loader import load_multiple_financial_reports
    from strategies.base_strategy import BaseStrategy
    from db.database import get_engine_instance
    logger.info("Attempted imports from specific modules using project root path.")
except ImportError as e:
    logger.critical(f"Failed to import necessary modules: {e}. Check paths/files. Exiting.", exc_info=True)
    exit()

PEAK_WINDOW = 5  # Window for identifying peaks
MA_PEAK_THRESHOLD = 1.20 # Peak must be 20% above MA30
MA_LONG_PERIOD_FOR_PEAK = 30 # MA period to compare the peak against

# --- Constants Definition ---
PULLBACK_THRESHOLD = 0.05
MA_SHORT_PERIOD = 5
MA_LONG_PERIOD = 30
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0
NET_PROFIT_FIELD = '归属于母公司所有者的净利润'
REVENUE_FIELD = '营业总收入'

class MAPullbackStrategy(BaseStrategy):
    def __init__(self, ma_short=MA_SHORT_PERIOD, ma_long=MA_LONG_PERIOD,
                 pullback_pct=PULLBACK_THRESHOLD, trend_window=5, timeframe="multi",
                 peak_window=PEAK_WINDOW, ma_peak_threshold=MA_PEAK_THRESHOLD, ma_long_for_peak=MA_LONG_PERIOD_FOR_PEAK):
        super().__init__(name=f"MAPullbackPeakCondition({ma_short},{ma_long},{pullback_pct*100:.0f}%)", timeframe=timeframe)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.pullback_pct = pullback_pct
        self.trend_window = trend_window
        self.peak_window = peak_window
        self.ma_peak_threshold = ma_peak_threshold
        self.ma_long_for_peak = ma_long_for_peak


    def calculate_ma(self, df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        if df is None or df.empty or 'close' not in df.columns:
             logger.warning("Cannot calculate MA: DataFrame is empty or missing 'close' column.")
             return pd.DataFrame(columns=df.columns.tolist() + [f'MA{ma}' for ma in ma_list] if df is not None else [f'MA{ma}' for ma in ma_list])
        df_copy = df.copy()
        for ma in ma_list:
            df_copy[f'MA{ma}'] = df_copy['close'].rolling(window=ma, min_periods=ma).mean().round(2)
        return df_copy

    def is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> Optional[bool]:
        if ma_series is None: logger.debug("MA series is None."); return None
        valid_series = ma_series.dropna()
        if len(valid_series) < 2: logger.debug(f"Need >= 2 points for trend ({len(valid_series)} found)."); return None
        effective_window = min(window, len(valid_series))
        if effective_window < 2: logger.debug(f"Effective window < 2."); return None
        x = np.arange(effective_window); y = valid_series[-effective_window:].values
        try:
            coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
            return slope >= -1e-6
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit trendline: {e}"); return None

    def _safe_get_value(self, report_data: Optional[Dict], key: str) -> Optional[float]:
        if report_data is None: return None
        if key not in report_data: return None
        value = report_data[key]
        if value is None or value == '': return None
        try:
            if isinstance(value, str) and value.strip() in ('--', 'N/A', '不适用'): return None
            num_value = float(value)
            return num_value
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert financial value '{value}' for key '{key}' to float: {e}")
            return None

    def get_fundamental_data(self, symbol: str, signal_date_str: str) -> Dict[str, Any]:
        # ... (This function remains the same as in your provided code) ...
        fundamental_results = {
            'net_profit_positive_3y_latest': None, 'pe': np.nan, 'pe_lt_30': None,
            'revenue_growth_yoy': np.nan, 'profit_growth_yoy': np.nan, 'growth_positive': None,
            'peg_like_ratio': np.nan, 'peg_like_lt_1': None, 'error_reason': None
        }
        error_reasons = []
        try:
            signal_date = datetime.strptime(signal_date_str, '%Y-%m-%d').date()
            logger.info(f"[{symbol}] Getting fundamental data for signal date: {signal_date_str}")
            df_latest_daily = load_daily_data(symbol, signal_date_str, signal_date_str, fields=['amount', 'turnover'])
            market_cap = np.nan
            if df_latest_daily is not None and not df_latest_daily.empty:
                latest_trade = df_latest_daily.iloc[0]
                amount = latest_trade['amount']
                turnover = latest_trade['turnover']
                if pd.notna(amount) and pd.notna(turnover) and turnover > 1e-6:
                    market_cap = amount / (turnover / 100.0)
                else: error_reasons.append("Missing or invalid trade data for Market Cap")
            else: error_reasons.append("No trade data found for Market Cap")
            num_years_for_profit_check = 3
            try: profit_reports = load_multiple_financial_reports(symbol, report_type='benefit', num_years=num_years_for_profit_check)
            except NameError: logger.error(f"[{symbol}] load_multiple_financial_reports function not available."); profit_reports = []
            if not profit_reports:
                error_reasons.append("Missing financial reports (benefit)")
                fundamental_results['error_reason'] = "; ".join(error_reasons); return fundamental_results
            latest_report = profit_reports[0]
            latest_annual_report = next((r for r in profit_reports if r['report_date'].month == 12 and r['report_date'].day == 31), None)
            prev_year_q_report = None
            if latest_report and latest_report['report_date'].month != 12 :
                try:
                     target_prev_date = latest_report['report_date'].replace(year=latest_report['report_date'].year - 1)
                     prev_year_q_report = next((r for r in profit_reports if r['report_date'] == target_prev_date), None)
                except ValueError as ve: logger.warning(f"[{symbol}] Error calculating previous year's date for {latest_report['report_date']}: {ve}")
            target_years = range(signal_date.year - num_years_for_profit_check, signal_date.year)
            annual_reports_last_3y = [r for r in profit_reports if r['report_date'].month == 12 and r['report_date'].day == 31 and r['report_date'].year in target_years]
            all_profits_positive = True; periods_checked = []; reports_for_check = []
            if latest_report: reports_for_check.append(latest_report)
            reports_for_check.extend(annual_reports_last_3y)
            unique_reports_for_check = {r['report_date']: r for r in reports_for_check}.values()
            required_annuals = num_years_for_profit_check; found_annuals = len(annual_reports_last_3y)
            if found_annuals < required_annuals:
                 all_profits_positive = None; error_reasons.append(f"Insufficient annual reports ({found_annuals}/{required_annuals}) for 3Y profit check")
            else:
                if latest_report:
                    profit = self._safe_get_value(latest_report.get('data'), NET_PROFIT_FIELD); periods_checked.append(latest_report['report_date'])
                    if profit is None or profit <= 1e-6: all_profits_positive = False;
                if all_profits_positive:
                    sorted_annuals = sorted(annual_reports_last_3y, key=lambda x: x['report_date'], reverse=True); reports_to_check_annual = sorted_annuals[:required_annuals]
                    for report in reports_to_check_annual:
                         report_date = report['report_date'];
                         if report_date in periods_checked: continue
                         profit = self._safe_get_value(report.get('data'), NET_PROFIT_FIELD); periods_checked.append(report_date)
                         if profit is None or profit <= 1e-6: all_profits_positive = False;
                         break
            fundamental_results['net_profit_positive_3y_latest'] = all_profits_positive
            if pd.notna(market_cap) and latest_annual_report:
                latest_annual_profit = self._safe_get_value(latest_annual_report.get('data'), NET_PROFIT_FIELD)
                if latest_annual_profit is not None and latest_annual_profit > 1e-6:
                    pe = market_cap / latest_annual_profit; fundamental_results['pe'] = pe; fundamental_results['pe_lt_30'] = pe < PE_THRESHOLD
                else: error_reasons.append(f"Invalid Annual Profit ({latest_annual_profit}) for PE")
            else:
                if pd.isna(market_cap) and "Market Cap" not in " ".join(error_reasons):pass
                if not latest_annual_report and "Annual Report" not in " ".join(error_reasons): error_reasons.append("Missing Annual Report for PE")
            growth_positive = None; rev_growth = np.nan; prof_growth = np.nan
            if latest_report and prev_year_q_report:
                latest_revenue = self._safe_get_value(latest_report.get('data'), REVENUE_FIELD); prev_revenue = self._safe_get_value(prev_year_q_report.get('data'), REVENUE_FIELD)
                latest_q_profit = self._safe_get_value(latest_report.get('data'), NET_PROFIT_FIELD); prev_q_profit = self._safe_get_value(prev_year_q_report.get('data'), NET_PROFIT_FIELD)
                if latest_revenue is not None and prev_revenue is not None:
                     if abs(prev_revenue) > 1e-6 : fundamental_results['revenue_growth_yoy'] = (latest_revenue - prev_revenue) / abs(prev_revenue); rev_growth = fundamental_results['revenue_growth_yoy']
                     else: error_reasons.append(f"Prev Revenue ({prev_revenue}) near zero for growth calc")
                elif "Missing revenue data" not in " ".join(error_reasons): error_reasons.append("Missing revenue data for growth")
                if latest_q_profit is not None and prev_q_profit is not None:
                    if abs(prev_q_profit) > 1e-6 : fundamental_results['profit_growth_yoy'] = (latest_q_profit - prev_q_profit) / abs(prev_q_profit); prof_growth = fundamental_results['profit_growth_yoy']
                    elif latest_q_profit > 0 and prev_q_profit <= 1e-6: fundamental_results['profit_growth_yoy'] = np.inf; prof_growth = np.inf
                    elif "Invalid Profit Data" not in " ".join(error_reasons): error_reasons.append(f"Invalid Profit Data for Growth (L:{latest_q_profit}, P:{prev_q_profit})")
                elif "Missing profit data" not in " ".join(error_reasons): error_reasons.append("Missing profit data for growth")
                if pd.notna(rev_growth) and pd.notna(prof_growth): growth_positive = (rev_growth > 1e-6) and (prof_growth > 1e-6)
                fundamental_results['growth_positive'] = growth_positive
            elif "Missing reports for YoY growth" not in " ".join(error_reasons): error_reasons.append("Missing reports for YoY growth")
            current_pe = fundamental_results.get('pe'); current_rev_growth = fundamental_results.get('revenue_growth_yoy')
            if pd.notna(current_pe) and pd.notna(current_rev_growth) and current_rev_growth > 1e-6:
                peg_like_ratio = current_pe / current_rev_growth; fundamental_results['peg_like_ratio'] = peg_like_ratio; fundamental_results['peg_like_lt_1'] = peg_like_ratio < PEG_LIKE_THRESHOLD
            else:
                if pd.isna(current_pe) and "PE is NaN" not in " ".join(error_reasons): error_reasons.append("PE is NaN for PE/Growth calc")
                if (pd.isna(current_rev_growth) or current_rev_growth <= 1e-6) and "Rev Growth invalid" not in " ".join(error_reasons): error_reasons.append("Rev Growth invalid for PE/Growth calc")
        except Exception as e:
            logger.error(f"[{symbol}] Unhandled error during fundamental data processing for {signal_date_str}: {e}", exc_info=True)
            error_reasons.append(f"Unhandled Exception: {e}")
        if error_reasons: fundamental_results['error_reason'] = "; ".join(sorted(list(set(error_reasons))))
        return fundamental_results

    def find_recent_valid_peak(self, df: pd.DataFrame, current_index: int, ma_long_col: str) -> Optional[float]:
        """
        Finds the most recent peak before current_index that satisfies the condition:
        peak_price >= MA_long_for_peak * ma_peak_threshold (e.g., peak >= MA30 * 1.20)
        """
        if current_index < self.peak_window:
            return None

        # Consider data up to (but not including) the current_index for finding prior peaks
        historical_df = df.iloc[:current_index]
        if len(historical_df) < self.peak_window:
            return None

        # Calculate rolling max for 'high' prices to identify peaks
        historical_df['rolling_peak'] = historical_df['high'].rolling(window=self.peak_window, center=True).max()

        # Iterate backwards from the point before current_index
        for i in range(len(historical_df) - 1, self.peak_window - 2, -1): # Ensure there's enough data for rolling_peak
            potential_peak_price = historical_df['high'].iloc[i]
            # Check if this point is a rolling peak
            if pd.notna(historical_df['rolling_peak'].iloc[i]) and potential_peak_price == historical_df['rolling_peak'].iloc[i]:
                # This is a peak according to the rolling window
                ma_at_peak_time = historical_df[ma_long_col].iloc[i]
                if pd.notna(ma_at_peak_time) and ma_at_peak_time > 0:
                    if potential_peak_price >= ma_at_peak_time * self.ma_peak_threshold:
                        logger.debug(f"Valid peak found at {historical_df['date'].iloc[i]}: Price {potential_peak_price:.2f}, MA{self.ma_long_for_peak} {ma_at_peak_time:.2f} (Threshold: {self.ma_peak_threshold})")
                        return potential_peak_price # Return the price of the valid peak
        return None


    def find_pullback_signals_on_last_day(self, df_with_ma: pd.DataFrame) -> List[Dict]:
        signals = []
        if df_with_ma is None or df_with_ma.empty or len(df_with_ma) < max(self.trend_window, self.ma_long, self.ma_long_for_peak):
            return signals

        ma_short_col = f'MA{self.ma_short}'
        ma_long_col = f'MA{self.ma_long}'
        ma_long_for_peak_col = f'MA{self.ma_long_for_peak}' # MA used for peak condition

        # Ensure all necessary MAs are calculated
        # The MA for peak condition might be different from the pullback MA (e.g. MA30 for both)
        # For this request, it's specified as 30-day line (ma_long_for_peak)
        all_ma_periods = list(set([self.ma_short, self.ma_long, self.ma_long_for_peak]))
        df_with_all_mas = self.calculate_ma(df_with_ma.copy(), all_ma_periods) # Use a copy

        if df_with_all_mas.empty: return signals


        required_cols = ['close', 'high', ma_short_col, ma_long_col, ma_long_for_peak_col, 'date', 'symbol']
        if not all(col in df_with_all_mas.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_with_all_mas.columns]
            logger.warning(f"Missing columns for pullback check: {missing} in df_with_all_mas. Columns are: {df_with_all_mas.columns.tolist()}")
            return signals

        last = df_with_all_mas.iloc[-1]
        current_df_index_for_last_day = len(df_with_all_mas) -1

        if last[required_cols].isnull().any():
            return signals

        # 1. Previous Peak Condition (NEW)
        # We pass df_with_all_mas to ensure the correct MA (ma_long_for_peak_col) is used inside find_recent_valid_peak
        valid_prior_peak = self.find_recent_valid_peak(df_with_all_mas, current_df_index_for_last_day, ma_long_for_peak_col)
        if valid_prior_peak is None:
            logger.debug(f"Condition 0 Fail: No recent valid peak found meeting criteria (Peak Price >= MA{self.ma_long_for_peak} * {self.ma_peak_threshold}).")
            return signals

        # 2. MA_long (e.g., MA30 for pullback) trend判断
        ma_long_series_for_trend = df_with_all_mas[ma_long_col]
        is_trend_up = self.is_ma_trending_up(ma_long_series_for_trend, window=self.trend_window)
        if is_trend_up is None or not is_trend_up:
            return signals

        # 3. 股价 > MA_long (pullback MA)
        cond2_price_above_ma_long = last['close'] > last[ma_long_col]
        if not cond2_price_above_ma_long:
            return signals

        # 4. MA_short > MA_long (pullback MA)
        cond3_ma_short_above_ma_long = last[ma_short_col] > last[ma_long_col]
        if not cond3_ma_short_above_ma_long:
            return signals

        # 5. 股价回踩 MA_long (pullback MA)
        if last[ma_long_col] <= 1e-6:
            return signals
        pullback_check = (last['close'] >= last[ma_long_col]) and \
                         ((last['close'] - last[ma_long_col]) / last[ma_long_col] <= self.pullback_pct)
        if not pullback_check:
            return signals

        logger.info(f"✅ [{last['symbol']}] MAPullbackPeakCond Signal on {pd.to_datetime(last['date']).strftime('%Y-%m-%d')}: ValidPeakFound, TrendOK, Close>MA{self.ma_long}, MA{self.ma_short}>MA{self.ma_long}, PullbackOK.")
        signals.append({
            "symbol": last['symbol'],
            "signal_date": pd.to_datetime(last['date']).strftime('%Y-%m-%d'),
            "strategy": self.name,
        })
        return signals


    def process_level(self, symbol: str, start_date: str, end_date: str, level: str) -> List[Dict]:
        logger.debug(f"[{symbol}] Processing {level} level from {start_date} to {end_date}")

        # For this strategy, MA30 is specifically mentioned for the peak condition.
        # The pullback itself uses self.ma_long (which defaults to 30 but could be configured).
        # We need to ensure MA30 data is available for the peak check, regardless of the level's self.ma_long.
        # For weekly/monthly, MA30 will be calculated based on weekly/monthly resampled data.
        # The primary "30-day line" for the peak refers to the MA on the *current level's data*.

        trend_check_window = self.trend_window
        if level == 'weekly': trend_check_window = max(4, self.trend_window)
        if level == 'monthly': trend_check_window = max(3, self.trend_window)

        required_fields = ["date", "open", "close", "high", "low", "volume"]
        df_daily = load_daily_data(symbol, start_date, end_date, fields=required_fields)
        if df_daily is None or df_daily.empty:
             logger.warning(f"[{symbol}] No daily data loaded for level {level}.")
             return []
        df_daily['symbol'] = symbol

        df_level_data = df_daily.copy()
        if level == "weekly":
            # Ensure enough data for the longest MA period (self.ma_long or self.ma_long_for_peak) on weekly.
            # E.g., if ma_long_for_peak is 30 (weeks), we need 30 * 5 = 150 daily bars approx.
            min_periods_needed = max(self.ma_long, self.ma_long_for_peak, trend_check_window)
            if len(df_level_data) < 5 * min_periods_needed : # Rough estimate
                logger.debug(f"[{symbol}] Not enough daily data for weekly resampling and MA calculation (needs ~{5*min_periods_needed} days).")
                return []
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try:
                df_level_data = df_daily.set_index('date').resample('W-FRI', closed='right', label='right').agg(agg).dropna(how='all').reset_index()
            except Exception as e: logger.error(f"[{symbol}] Error weekly resampling: {e}", exc_info=True); return []
        elif level == "monthly":
            min_periods_needed = max(self.ma_long, self.ma_long_for_peak, trend_check_window)
            if len(df_level_data) < 20 * min_periods_needed: # Rough estimate
                 logger.debug(f"[{symbol}] Not enough daily data for monthly resampling and MA calculation (needs ~{20*min_periods_needed} days).")
                 return []
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try:
                 df_level_data = df_daily.set_index('date').resample('ME').agg(agg).dropna(how='all').reset_index()
            except Exception as e: logger.error(f"[{symbol}] Error monthly resampling: {e}", exc_info=True); return []

        if df_level_data.empty:
            logger.warning(f"[{symbol}] Resampled {level} DataFrame empty.")
            return []

        # Calculate all MAs needed: short, long (for pullback), and long_for_peak
        # The `find_pullback_signals_on_last_day` will internally call `calculate_ma`
        # with all necessary periods if they are not already present.
        # So, we can just pass df_level_data.
        signals = self.find_pullback_signals_on_last_day(df_level_data) # Pass the resampled data

        for sig in signals:
            sig['timeframe'] = level
        return signals

    def find_signals(self, symbol: str, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        logger.info(f"🔍 [{symbol}] Scanning for MA Pullback with Peak Condition from {start_date} to {end_date}")
        results = {}
        levels_to_run = ['daily', 'weekly', 'monthly'] if self.timeframe == 'multi' else [self.timeframe]

        for level in levels_to_run:
            if level not in ['daily', 'weekly', 'monthly']:
                logger.warning(f"Unsupported timeframe '{level}' requested. Skipping.")
                continue
            try:
                # Pass the ma_long_for_peak to process_level if needed,
                # or ensure it's handled within find_pullback_signals_on_last_day
                results[level] = self.process_level(symbol, start_date, end_date, level=level)
                logger.info(f"[{symbol}] Found {len(results.get(level,[]))} signals for {level} level with peak condition.")
            except Exception as e:
                 logger.error(f"[{symbol}] Error processing {level} level: {e}", exc_info=True)
                 results[level] = []
        return results


if __name__ == "__main__":
    # Note: ma_long_for_peak defaults to 30 in the constructor
    strategy = MAPullbackStrategy(ma_short=5, ma_long=30, pullback_pct=0.05, trend_window=5, ma_long_for_peak=30)
    engine = get_engine_instance()

    if engine is None: logger.critical("DB engine NG. Exit."); exit()

    try:
        stock_query = "SELECT code as symbol FROM stock_list" # Consider LIMIT for testing
        df_stocks = pd.read_sql(stock_query, con=engine)
        stock_list = df_stocks['symbol'].tolist()
        logger.info(f"📈 Stocks to process: {len(stock_list)}")
        if not stock_list: logger.warning("Stock list empty."); exit()
    except Exception as e: logger.error(f"Failed to get stock list: {e}", exc_info=True); exit()

    try:
        date_query = "SELECT MAX(date) AS max_date FROM stock_daily"
        df_dates = pd.read_sql(date_query, con=engine)
        if pd.isna(df_dates.at[0, 'max_date']):
             logger.error("Cannot get max date."); end_date_obj = datetime.now().date()
             logger.warning(f"Using current date {end_date_obj.strftime('%Y-%m-%d')} as end date.")
        else: end_date_obj = pd.to_datetime(df_dates.at[0, 'max_date']).date()
        end_date = end_date_obj.strftime('%Y-%m-%d')
        # For MA30 and peak window, might need more history
        start_date = (end_date_obj - pd.DateOffset(years=3)).strftime('%Y-%m-%d') # Approx 3 years
        logger.info(f"📅 Analysis period: {start_date} to {end_date}")
        logger.info(f"🛎 Filtering signals ONLY on: {end_date}")
    except Exception as e: logger.error(f"Failed to get date range: {e}", exc_info=True); exit()

    initial_signals = []
    processed_count = 0
    total_stocks = len(stock_list)
    scan_start_time = time.time()

    for symbol in stock_list:
        processed_count += 1
        if processed_count % 100 == 0:
            logger.info(f"--- Scanning {processed_count}/{total_stocks}: {symbol} ---")

        try:
            results = strategy.find_signals(symbol, str(start_date), str(end_date))
            for level, signals_in_level in results.items():
                for sig in signals_in_level:
                    if sig['signal_date'] == end_date:
                        logger.info(f"✅ [{symbol}] Tech Signal: {strategy.name} on {level.upper()} at {sig['signal_date']}")
                        initial_signals.append({
                            "symbol": symbol,
                            "signal_date": sig['signal_date'],
                            "strategy": strategy.name,
                            "timeframe": level
                        })
        except Exception as e:
            logger.error(f"Error processing technical signals for {symbol}: {e}", exc_info=True)

    scan_end_time = time.time()
    logger.info(f"\n📊 Found {len(initial_signals)} initial technical signals ({strategy.name}) on {end_date}. Scan took {scan_end_time - scan_start_time:.2f}s.")

    final_enhanced_signals = []
    if initial_signals:
        logger.info("\n🔬 Starting fundamental analysis...")
        fundamental_processed_count = 0
        total_initial = len(initial_signals)
        fund_start_time = time.time()

        for initial_sig in initial_signals:
            fundamental_processed_count += 1
            symbol = initial_sig['symbol']
            signal_date_str = initial_sig['signal_date']

            fundamental_data = strategy.get_fundamental_data(symbol, signal_date_str)
            enhanced_sig = {**initial_sig, **fundamental_data}
            final_enhanced_signals.append(enhanced_sig)

            error_msg = enhanced_sig.get('error_reason')
            error_str = f", Errors='{error_msg}'" if error_msg else ""
            logger.info(f"-> Fund Results {symbol} ({initial_sig['timeframe']}): PE={enhanced_sig.get('pe', np.nan):.2f}, PE<30={enhanced_sig.get('pe_lt_30')}, Growth+={enhanced_sig.get('growth_positive')}, PEG<1={enhanced_sig.get('peg_like_lt_1')}, 3YProfit+={enhanced_sig.get('net_profit_positive_3y_latest')}{error_str}")

        fund_end_time = time.time()
        logger.info(f"Fundamental analysis took {fund_end_time - fund_start_time:.2f}s.")

    if final_enhanced_signals:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'output')
        try: os.makedirs(output_dir, exist_ok=True)
        except OSError as ose: logger.error(f"Cannot create output dir {output_dir}: {ose}"); output_dir = "."

        columns_order = [
            'symbol', 'signal_date', 'strategy', 'timeframe',
            'net_profit_positive_3y_latest', 'pe', 'pe_lt_30',
            'revenue_growth_yoy', 'profit_growth_yoy', 'growth_positive',
            'peg_like_ratio', 'peg_like_lt_1', 'error_reason'
        ]
        df_final = pd.DataFrame(final_enhanced_signals)
        for col in columns_order:
            if col not in df_final.columns: df_final[col] = np.nan
        df_final = df_final[columns_order]

        float_cols = ['pe', 'revenue_growth_yoy', 'profit_growth_yoy', 'peg_like_ratio']
        for col in float_cols:
             if col in df_final.columns:
                  df_final[col] = df_final[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) and isinstance(x, (int, float)) else ('+Inf' if x == np.inf else ('-Inf' if x == -np.inf else None)))

        filename = os.path.join(output_dir, f"signals_{strategy.name}_with_fundamentals_{end_date}.csv")
        try:
            df_final.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"\n✅ Final signals saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV {filename}: {e}", exc_info=True)
    else:
        logger.info("\nℹ️ No signals found meeting criteria or after fundamental analysis.")

    total_end_time = time.time()
    logger.info(f"\n--- Script finished. Total execution time: {total_end_time - scan_start_time:.2f} seconds ---")