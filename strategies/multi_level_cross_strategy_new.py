# strategies/multi_level_cross_strategy_new.py
import os
import sys # <--- Import sys
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date
import time
# --- Python Path Setup ---
# Goal: Allow running this script directly while still finding other project modules.
# Get the absolute path of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root directory (assuming it's one level up from 'strategies')
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    logger = logging.getLogger(__name__) # Temporary logger for path message
    logger.info(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Logging Configuration (should be done early) ---
# Re-initialize logger after potential path changes if needed, or use root logger config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Logger for this module

# --- Now use absolute imports relative to the project root ---
# This might still generate warnings if the environment setup is unusual,
# but it should prevent the 'attempted relative import' error.
try:
    # Import each function from its correct location
    from utils.data_loader import load_daily_data                 # <--- 从 utils.data_loader 导入
    from data_processing.loader import load_multiple_financial_reports # <--- 从 data_processing.loader 导入
    from strategies.base_strategy import BaseStrategy
    from db.database import get_engine_instance
    logger.info("Attempted imports from specific modules using project root path.")
except ImportError as e:
    # This error is more serious now, means modules truly weren't found even with path mod
    logger.critical(f"Failed to import necessary modules: {e}. Please ensure the project structure is correct ('data_processing', 'db', 'utils' folders exist at {project_root}) and contain required files/functions. Exiting.", exc_info=True)
    exit()


# --- Constants Definition (Keep as before) ---
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0
NET_PROFIT_FIELD = '归属于母公司所有者的净利润' # Confirm this is the correct field name in your DB
REVENUE_FIELD = '营业总收入' # Confirm this is the correct field name in your DB

# --- Strategy Class Definition ---
# PASTE THE **ENTIRE CORRECTED CLASS DEFINITION** FROM THE PREVIOUS RESPONSE HERE
# (Including __init__, calculate_ma, is_ma_trending_up, detect_cross, _safe_get_value,
#  get_fundamental_data, process_level, find_signals)
class MultiLevelCrossStrategy(BaseStrategy):
    """多级别一阳穿四线策略 (包含基本面过滤逻辑)"""
    # ... (The rest of the class code remains exactly the same as the previous corrected version) ...
    # (Make sure to include the entire corrected class definition here)
    def __init__(self, timeframe="multi"):
        super().__init__(name="MultiLevelCross", timeframe=timeframe)

    def calculate_ma(self, df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        """计算各种均线"""
        if df is None or df.empty or 'close' not in df.columns:
             logger.warning("Cannot calculate MA: DataFrame is empty or missing 'close' column.")
             # Return an empty DataFrame with expected columns if possible, or just empty
             return pd.DataFrame(columns=df.columns.tolist() + [f'MA{ma}' for ma in ma_list] if df is not None else [f'MA{ma}' for ma in ma_list])

        df_copy = df.copy() # 避免 SettingWithCopyWarning
        for ma in ma_list:
            # Ensure enough non-NaN values for the window, else result is NaN
            df_copy[f'MA{ma}'] = df_copy['close'].rolling(window=ma, min_periods=ma).mean().round(2)
        return df_copy

    def is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> bool:
        """判断均线是否走平或向上"""
        if ma_series is None:
            logger.debug("MA series is None for trend check.")
            return False

        # 移除 NaN 值以进行拟合
        valid_series = ma_series.dropna()
        if len(valid_series) < 2: # 至少需要两个点来确定趋势
             logger.debug(f"Not enough valid points ({len(valid_series)}) in MA series to determine trend (min 2 required).")
             return False

        # Adjust window size if not enough data points are available
        effective_window = min(window, len(valid_series))
        if effective_window < 2: # Still need at least 2 points after adjustment
            logger.debug(f"Effective window size ({effective_window}) too small for trend calculation.")
            return False
        #if window != effective_window:
        #     logger.debug(f"Using adjusted window size {effective_window} for trend calculation due to limited valid data ({len(valid_series)} points).")


        x = np.arange(effective_window)
        y = valid_series[-effective_window:].values
        try:
            # 使用 numpy 进行线性拟合
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            is_trending_up = slope >= -1e-6 # 允许非常小的负斜率，视为走平
            # logger.debug(f"MA trend slope: {slope:.4f}. Trending up: {is_trending_up}")
            return is_trending_up
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit trendline for MA series: {e}")
            return False

    def detect_cross(self, df: pd.DataFrame, ma_list: List[int], check_ma30=False, check_volume=False) -> List[Dict]:
        """检测一阳穿四线信号"""
        signals = []
        if df is None or df.empty or len(df) < 2:
             logger.debug("DataFrame is empty or too short for cross detection.")
             return signals

        required_ma_cols = [f'MA{ma}' for ma in ma_list]
        if check_ma30:
            required_ma_cols.append('MA30') # MA30 should already be calculated if check_ma30 is True
        if not all(col in df.columns for col in required_ma_cols):
             # Log which specific columns are missing
             missing_cols = [col for col in required_ma_cols if col not in df.columns]
             logger.warning(f"Missing required MA columns ({missing_cols}) in DataFrame. Skipping cross detection.")
             return signals


        # 确保 'close', 'volume', 'date' 存在
        required_data_cols = ['close', 'date']
        if check_volume:
            required_data_cols.append('volume')
        if not all(col in df.columns for col in required_data_cols):
             missing_cols = [col for col in required_data_cols if col not in df.columns]
             logger.warning(f"Missing required data columns ({missing_cols}). Skipping cross detection.")
             return signals

        # logger.debug(f"Detecting cross for {len(df)} bars with MA list {ma_list}, check_ma30={check_ma30}, check_volume={check_volume}")
        df_length = len(df)
        # Calculate the minimum period required based on the actual MA list used
        min_required_period = max(ma_list + ([30] if check_ma30 and 30 in ma_list else [])) # Ensure 30 is checked only if needed AND in the list
        # Start loop from max MA period to ensure all MAs are valid (due to min_periods=ma in calculation)
        start_index = min_required_period # Rolling window needs this many points to be potentially valid
        if start_index < 1: start_index = 1 # Minimum start index is 1 for prev comparison
        if start_index >= df_length: # If not enough data for the longest MA period
             logger.debug(f"DataFrame length {df_length} is less than max MA period {min_required_period}. Cannot perform cross detection.")
             return signals

        for i in range(start_index, df_length):
            current = df.iloc[i]
            prev = df.iloc[i - 1]

            # Check core data validity first
            if pd.isna(current['close']) or pd.isna(prev['close']):
                # logger.debug(f"Skipping index {i}: NaN in close price.")
                continue

            # Check validity of required MAs for current and previous row
            current_mas_valid = not any(pd.isna(current.get(f'MA{ma}')) for ma in ma_list)
            prev_mas_valid = not any(pd.isna(prev.get(f'MA{ma}')) for ma in ma_list)
            if not current_mas_valid or not prev_mas_valid:
                 # logger.debug(f"Skipping index {i}: NaN in required MAs {ma_list}.")
                 continue


            # Core crossing logic
            try:
                # Use .get() for safety, fallback to values that ensure conditions fail if MA is missing
                cross_condition = current.get('close', 0) > 0 and all(current.get('close') > current.get(f'MA{ma}', np.inf) for ma in ma_list)
                below_condition = all(prev.get('close', 0) <= prev.get(f'MA{ma}', -np.inf) for ma in ma_list)
            except TypeError as te:
                 logger.warning(f"TypeError during condition check at index {i}: {te}. Current: {current.to_dict()}, Prev: {prev.to_dict()}")
                 continue


            # MA30 trend condition
            ma30_condition = True
            if check_ma30:
                current_ma30 = current.get('MA30')
                prev_ma30 = prev.get('MA30')
                if pd.notna(current_ma30) and pd.notna(prev_ma30):
                    ma30_condition = round(current_ma30, 2) >= round(prev_ma30, 2)
                else:
                    ma30_condition = False # MA30 missing fails the condition
                    # logger.debug(f"Index {i}: MA30 condition failed due to NaN.")


            # Volume condition
            volume_condition = True
            if check_volume:
                current_volume = current.get('volume')
                prev_volume = prev.get('volume')
                if pd.notna(current_volume) and pd.notna(prev_volume) and prev_volume > 1e-6:
                     volume_condition = current_volume >= prev_volume * 1.5
                else:
                     volume_condition = False # Missing volume or zero previous volume fails the condition
                     # logger.debug(f"Index {i}: Volume condition failed (Current: {current_volume}, Prev: {prev_volume}).")

            # Check final signal
            if cross_condition and below_condition and ma30_condition and volume_condition:
                signal_date = pd.to_datetime(current['date']).strftime('%Y-%m-%d')
                # logger.debug(f"Signal detected at index {i} (Date: {signal_date}). Conditions met: cross={cross_condition}, below={below_condition}, ma30={ma30_condition}, volume={volume_condition}")
                signals.append({
                    'signal_date': signal_date,
                    'strategy': self.name,
                })
        return signals

    def _safe_get_value(self, report_data: Optional[Dict], key: str) -> Optional[float]:
        """安全地从财报字典中获取数值，处理 None、空字符串和转换错误"""
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
        """获取并处理指定股票和日期的基本面数据"""
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
                    if profit is None or profit <= 1e-6: all_profits_positive = False; # logger.info(f"[{symbol}] Latest report profit ({profit}) not positive.")
                if all_profits_positive:
                    sorted_annuals = sorted(annual_reports_last_3y, key=lambda x: x['report_date'], reverse=True); reports_to_check_annual = sorted_annuals[:required_annuals]
                    for report in reports_to_check_annual:
                         report_date = report['report_date'];
                         if report_date in periods_checked: continue
                         profit = self._safe_get_value(report.get('data'), NET_PROFIT_FIELD); periods_checked.append(report_date)
                         if profit is None or profit <= 1e-6: all_profits_positive = False; # logger.info(f"[{symbol}] Annual report ({report_date}) profit ({profit}) not positive.");
                         break
            fundamental_results['net_profit_positive_3y_latest'] = all_profits_positive
            if pd.notna(market_cap) and latest_annual_report:
                latest_annual_profit = self._safe_get_value(latest_annual_report.get('data'), NET_PROFIT_FIELD)
                if latest_annual_profit is not None and latest_annual_profit > 1e-6:
                    pe = market_cap / latest_annual_profit; fundamental_results['pe'] = pe; fundamental_results['pe_lt_30'] = pe < PE_THRESHOLD
                else: error_reasons.append(f"Invalid Annual Profit ({latest_annual_profit}) for PE")
            else:
                if pd.isna(market_cap) and "Market Cap" not in " ".join(error_reasons):pass# logger.warning(f"[{symbol}] Market cap is NaN.")
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

    def process_level(self, symbol: str, start_date: str, end_date: str, level: str) -> List[Dict]:
        """处理单个级别（日、周、月）的信号检测"""
        df = load_daily_data(symbol, start_date, end_date, fields=["date", "open", "close", "high", "low", "volume"]) # Ensure 'open' is loaded
        if df is None or df.empty:
             logger.warning(f"[{symbol}] No daily data loaded for level {level}.")
             return []
        df['symbol'] = symbol
        ma_list_map = {"daily": [5, 10, 20, 30], "weekly": [5, 10, 20, 30], "monthly": [3, 5, 10, 12]}
        ma_list = ma_list_map[level]
        check_ma30 = level == "daily"; check_volume = level in ["daily", "weekly"]
        df_resampled = df.copy()
        if level == "weekly":
            if len(df_resampled) < 5: return []
            aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try: df_resampled = df.set_index('date').resample('W-FRI', closed='right', label='right').agg(aggregation).dropna(how='all').reset_index()
            except KeyError as ke: logger.error(f"[{symbol}] KeyError weekly resampling: {ke}. Cols: {df.columns.tolist()}"); return []
            except Exception as e: logger.error(f"[{symbol}] Error weekly resampling: {e}", exc_info=True); return []
        elif level == "monthly":
            if len(df_resampled) < 20: return []
            aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try: df_resampled = df.set_index('date').resample('ME').agg(aggregation).dropna(how='all').reset_index()
            except KeyError as ke: logger.error(f"[{symbol}] KeyError monthly resampling: {ke}. Cols: {df.columns.tolist()}"); return []
            except Exception as e: logger.error(f"[{symbol}] Error monthly resampling: {e}", exc_info=True); return []
        if df_resampled.empty: logger.warning(f"[{symbol}] Resampled {level} DataFrame empty."); return []
        ma_to_calculate = list(set(ma_list + ([30] if check_ma30 or level == 'weekly' else [])))
        df_ma = self.calculate_ma(df_resampled, ma_to_calculate)
        if df_ma.empty: logger.warning(f"[{symbol}] DataFrame empty after MA calculation for {level}."); return []
        missing_ma_calc = [f'MA{w}' for w in ma_to_calculate if f'MA{w}' not in df_ma.columns]
        if missing_ma_calc: logger.warning(f"[{symbol}] Could not calculate MAs: {missing_ma_calc} for {level}."); return []
        required_ma_subset = [f'MA{w}' for w in ma_list];
        if check_ma30: required_ma_subset.append('MA30')
        df_ma.dropna(subset=required_ma_subset, inplace=True) # Use subset based on required MAs for the level
        if df_ma.empty: logger.warning(f"[{symbol}] DataFrame empty after MA dropna for {level}."); return []
        if level == 'weekly':
            if 'MA30' in df_ma.columns and not df_ma.empty:
                if len(df_ma['MA30'].dropna()) >= 4:
                     if not self.is_ma_trending_up(df_ma['MA30'], window=4):
                         # logger.info(f"[{symbol}] Weekly MA30 not trending up. Skip.")
                         return [] # Trend condition not met
        signals = self.detect_cross(df_ma, ma_list, check_ma30=check_ma30, check_volume=check_volume)
        for sig in signals: sig['timeframe'] = level
        return signals

    def find_signals(self, symbol: str, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """统一接口：返回日线/周线/月线信号"""
        results = {}
        for level in ["daily", "weekly", "monthly"]:
            try:
                results[level] = self.process_level(symbol, start_date, end_date, level=level)
            except Exception as e:
                 logger.error(f"[{symbol}] Error processing {level} level: {e}", exc_info=True)
                 results[level] = []
        return results


# --- Main Execution Block (`if __name__ == "__main__":`) ---
# PASTE THE **ENTIRE CORRECTED MAIN EXECUTION BLOCK** FROM THE PREVIOUS RESPONSE HERE
if __name__ == "__main__":
    strategy = MultiLevelCrossStrategy()
    engine = get_engine_instance() # 获取数据库引擎实例

    if engine is None:
        logger.critical("Database engine not initialized. Exiting.")
        exit()

    # 1. 获取股票列表
    try:
        # stock_query = "SELECT DISTINCT symbol FROM stock_daily LIMIT 10" # DEBUG
        stock_query = "SELECT code as symbol FROM stock_list " # DEBUG from list
        # stock_query = "SELECT code as symbol FROM stock_list" # Formal
        df_stocks = pd.read_sql(stock_query, con=engine)
        stock_list = df_stocks['symbol'].tolist()
        logger.info(f"📈 Found {len(stock_list)} stocks to process from stock_list.")
        if not stock_list: logger.warning("Stock list empty."); exit()
    except Exception as e: logger.error(f"Failed to get stock list: {e}", exc_info=True); exit()

    # 2. 获取日期范围
    try:
        date_query = "SELECT MAX(date) AS max_date FROM stock_daily"
        df_dates = pd.read_sql(date_query, con=engine)
        if pd.isna(df_dates.at[0, 'max_date']):
             logger.error("Cannot get max date from stock_daily.")
             end_date_obj = datetime.now().date() # 使用当前日期
             logger.warning(f"Using current date {end_date_obj.strftime('%Y-%m-%d')} as end date.")
        else: end_date_obj = pd.to_datetime(df_dates.at[0, 'max_date']).date()
        end_date = end_date_obj.strftime('%Y-%m-%d')
        start_date = (end_date_obj - pd.DateOffset(years=5)).strftime('%Y-%m-%d') # 5 years data
        logger.info(f"📅 Analysis period: {start_date} to {end_date}")
        logger.info(f"🛎 Filtering signals ONLY on: {end_date}")
    except Exception as e: logger.error(f"Failed to get date range: {e}", exc_info=True); exit()


    # 3. 寻找技术信号
    initial_signals = []; processed_count = 0; total_stocks = len(stock_list)
    start_time_technical = time.time()
    for symbol in stock_list:
        processed_count += 1
        logger.info(f"--- Tech Scan {processed_count}/{total_stocks}: {symbol} ---")
        try:
            results = strategy.find_signals(symbol, str(start_date), str(end_date))
            signal_found = False
            for level, signals_in_level in results.items():
                for sig in signals_in_level:
                    if sig['signal_date'] == end_date:
                        logger.info(f"✅ [{symbol}] Tech signal on {level.upper()} at {sig['signal_date']}")
                        initial_signals.append({"symbol": symbol, "signal_date": sig['signal_date'], "strategy": strategy.name, "timeframe": level})
                        signal_found = True
            #if not signal_found: logger.debug(f"[{symbol}] No technical signal on {end_date}.")

        except Exception as e: logger.error(f"Error tech signals for {symbol}: {e}", exc_info=True)
    end_time_technical = time.time()
    logger.info(f"\n📊 Found {len(initial_signals)} tech signals on {end_date}. Scan took {end_time_technical - start_time_technical:.2f}s.")


    # 4. 基本面分析
    final_enhanced_signals = []
    if initial_signals:
        logger.info("\n🔬 Starting fundamental analysis...")
        fundamental_processed_count = 0; total_initial = len(initial_signals)
        start_time_fundamental = time.time()
        for initial_sig in initial_signals:
            fundamental_processed_count += 1
            symbol = initial_sig['symbol']; signal_date_str = initial_sig['signal_date']
            logger.info(f"--- Fund Analysis {fundamental_processed_count}/{total_initial}: {symbol} on {signal_date_str} ---")
            fundamental_data = strategy.get_fundamental_data(symbol, signal_date_str)
            enhanced_sig = {**initial_sig, **fundamental_data}
            final_enhanced_signals.append(enhanced_sig)
            error_msg = enhanced_sig.get('error_reason')
            error_str = f", Errors='{error_msg}'" if error_msg else ""
            logger.info(f"-> Fund Results {symbol}: PE={enhanced_sig.get('pe', np.nan):.2f}, PE<30={enhanced_sig.get('pe_lt_30')}, Growth+={enhanced_sig.get('growth_positive')}, PEG<1={enhanced_sig.get('peg_like_lt_1')}, 3YProfit+={enhanced_sig.get('net_profit_positive_3y_latest')}{error_str}")
        end_time_fundamental = time.time()
        logger.info(f"Fundamental analysis took {end_time_fundamental - start_time_fundamental:.2f}s.")


    # 5. 保存结果
    if final_enhanced_signals:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'output')
        try: os.makedirs(output_dir, exist_ok=True); # logger.info(f"Output directory: {output_dir}")
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
        filename = os.path.join(output_dir, f"signals_with_fundamentals_{end_date}.csv")
        try: df_final.to_csv(filename, index=False, encoding='utf-8-sig'); logger.info(f"\n✅ Final signals saved to: {filename}")
        except Exception as e: logger.error(f"Failed to save CSV {filename}: {e}", exc_info=True)
    else: logger.info("\nℹ️ No signals found meeting criteria or after fundamental analysis.")

    total_end_time = time.time()
    logger.info(f"\n--- Script finished. Total execution time: {total_end_time - start_time_technical:.2f} seconds ---")