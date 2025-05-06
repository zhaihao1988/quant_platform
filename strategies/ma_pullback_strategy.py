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
    # Ensure logger is configured before use if this block runs first
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Logger for this module

# --- Imports ---
try:
    from utils.data_loader import load_daily_data
    from data_processing.loader import load_multiple_financial_reports
    from strategies.base_strategy import BaseStrategy
    from db.database import get_engine_instance
    logger.info("Attempted imports from specific modules using project root path.")
except ImportError as e:
    logger.critical(f"Failed to import necessary modules: {e}. Check paths/files. Exiting.", exc_info=True)
    exit()


# --- Constants Definition ---
PULLBACK_THRESHOLD = 0.05 # 回踩幅度阈值 (5%)
MA_SHORT_PERIOD = 5       # 短期均线周期
MA_LONG_PERIOD = 30      # 长期均线周期 (基准线)
# 基本面常量 (与 multi_level_cross_strategy_new.py 保持一致)
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0
NET_PROFIT_FIELD = '归属于母公司所有者的净利润'
REVENUE_FIELD = '营业总收入'

class MAPullbackStrategy(BaseStrategy):
    """
    均线回踩策略:
    在 MA30 向上或走平的趋势中，当股价和 MA5 均在 MA30 之上，
    且股价回调至 MA30 附近 (上方5%以内) 时产生信号。
    同时进行基本面过滤。
    """
    def __init__(self, ma_short=MA_SHORT_PERIOD, ma_long=MA_LONG_PERIOD,
                 pullback_pct=PULLBACK_THRESHOLD, trend_window=5, timeframe="multi"):
        # timeframe 参数允许 Runner 指定处理哪个级别，或 'multi' 表示都处理
        super().__init__(name=f"MAPullback({ma_short},{ma_long},{pullback_pct*100:.0f}%)", timeframe=timeframe)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.pullback_pct = pullback_pct
        self.trend_window = trend_window # 用于判断 MA_long 趋势的窗口

    # --- 复用 multi_level_cross_strategy_new.py 中的辅助方法 ---
    # (确保这些方法已复制到此类中或通过继承/组合可用)

    def calculate_ma(self, df: pd.DataFrame, ma_list: List[int]) -> pd.DataFrame:
        """计算各种均线"""
        if df is None or df.empty or 'close' not in df.columns:
             logger.warning("Cannot calculate MA: DataFrame is empty or missing 'close' column.")
             return pd.DataFrame(columns=df.columns.tolist() + [f'MA{ma}' for ma in ma_list] if df is not None else [f'MA{ma}' for ma in ma_list])
        df_copy = df.copy()
        for ma in ma_list:
            df_copy[f'MA{ma}'] = df_copy['close'].rolling(window=ma, min_periods=ma).mean().round(2)
        return df_copy

    def is_ma_trending_up(self, ma_series: pd.Series, window: int = 5) -> Optional[bool]:
        """判断均线是否走平或向上, 返回 None 表示无法判断"""
        if ma_series is None: logger.debug("MA series is None."); return None
        valid_series = ma_series.dropna()
        if len(valid_series) < 2: logger.debug(f"Need >= 2 points for trend ({len(valid_series)} found)."); return None
        effective_window = min(window, len(valid_series))
        if effective_window < 2: logger.debug(f"Effective window < 2."); return None
        x = np.arange(effective_window); y = valid_series[-effective_window:].values
        try:
            coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
            return slope >= -1e-6 # 允许极小的负斜率
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit trendline: {e}"); return None

    def _safe_get_value(self, report_data: Optional[Dict], key: str) -> Optional[float]:
        # ... (代码同 multi_level_cross_strategy_new.py) ...
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
        # ... (代码完全同 multi_level_cross_strategy_new.py 的 get_fundamental_data) ...
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

    # --- 新策略的核心逻辑 ---
    def find_pullback_signals_on_last_day(self, df_with_ma: pd.DataFrame) -> List[Dict]:
        """
        在带有均线的 DataFrame 上，检查最后一天是否满足回踩买入条件。
        """
        signals = []
        if df_with_ma is None or df_with_ma.empty or len(df_with_ma) < self.trend_window:
            # logger.debug("DataFrame empty or too short for pullback check.")
            return signals

        # 获取所需的列名
        ma_short_col = f'MA{self.ma_short}'
        ma_long_col = f'MA{self.ma_long}'
        required_cols = ['close', ma_short_col, ma_long_col, 'date', 'symbol']

        if not all(col in df_with_ma.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_with_ma.columns]
            logger.warning(f"Missing columns for pullback check: {missing}")
            return signals

        # 获取最后一行数据
        last = df_with_ma.iloc[-1]

        # 检查最后一行数据是否有效
        if last[required_cols].isnull().any():
            # logger.debug(f"Last row contains NaN in required columns: {last.to_dict()}")
            return signals

        # --- 条件判断 ---
        # 1. MA_long 趋势判断 (使用指定窗口)
        ma_long_series = df_with_ma[ma_long_col]
        is_trend_up = self.is_ma_trending_up(ma_long_series, window=self.trend_window)
        if is_trend_up is None or not is_trend_up: # 如果无法判断或趋势向下/走平不严格，则不满足
            # logger.debug(f"Condition 1 Fail: MA{self.ma_long} trend not up or flat (Trend check result: {is_trend_up}).")
            return signals

        # 2. 股价 > MA_long
        cond2_price_above_ma_long = last['close'] > last[ma_long_col]
        if not cond2_price_above_ma_long:
            # logger.debug(f"Condition 2 Fail: Close {last['close']:.2f} <= MA{self.ma_long} {last[ma_long_col]:.2f}.")
            return signals

        # 3. MA_short > MA_long
        cond3_ma_short_above_ma_long = last[ma_short_col] > last[ma_long_col]
        if not cond3_ma_short_above_ma_long:
            # logger.debug(f"Condition 3 Fail: MA{self.ma_short} {last[ma_short_col]:.2f} <= MA{self.ma_long} {last[ma_long_col]:.2f}.")
            return signals

        # 4. 股价回踩 MA_long (在 MA_long 之上，且距离不超过 pullback_pct)
        # 确保 MA_long 不为 0 或负数
        if last[ma_long_col] <= 1e-6:
            # logger.debug(f"Condition 4 Fail: MA{self.ma_long} is zero or negative ({last[ma_long_col]:.2f}).")
            return signals
        pullback_check = (last['close'] >= last[ma_long_col]) and \
                         ((last['close'] - last[ma_long_col]) / last[ma_long_col] <= self.pullback_pct)
        if not pullback_check:
            # diff_pct = ((last['close'] - last[ma_long_col]) / last[ma_long_col]) * 100
            # logger.debug(f"Condition 4 Fail: Pullback condition not met (Close={last['close']:.2f}, MA={last[ma_long_col]:.2f}, Diff={diff_pct:.2f}% > {self.pullback_pct*100:.0f}% or Close < MA).")
            return signals

        # --- 所有条件满足 ---
        logger.info(f"✅ [{last['symbol']}] MAPullback Signal on {pd.to_datetime(last['date']).strftime('%Y-%m-%d')}: TrendOK={is_trend_up}, Close>MA{self.ma_long}, MA{self.ma_short}>MA{self.ma_long}, PullbackOK.")
        signals.append({
            "symbol": last['symbol'],
            "signal_date": pd.to_datetime(last['date']).strftime('%Y-%m-%d'),
            "strategy": self.name,
            # timeframe 会在 process_level 中添加
        })
        return signals


    # --- 多周期处理框架 (类似 multi_level_cross_strategy_new.py) ---
    def process_level(self, symbol: str, start_date: str, end_date: str, level: str) -> List[Dict]:
        """处理单个级别（日、周、月）的信号检测"""
        logger.debug(f"[{symbol}] Processing {level} level from {start_date} to {end_date}")

        # 确定该级别所需的均线
        ma_periods = [self.ma_short, self.ma_long]
        # 确定判断趋势所需的窗口长度 (可以根据 level 调整)
        trend_check_window = self.trend_window
        if level == 'weekly': trend_check_window = max(4, self.trend_window) # 周线至少看4周
        if level == 'monthly': trend_check_window = max(3, self.trend_window) # 月线至少看3月

        # 1. 加载日线数据 (包含所有需要的列)
        required_fields = ["date", "open", "close", "high", "low", "volume"]
        df_daily = load_daily_data(symbol, start_date, end_date, fields=required_fields)
        if df_daily is None or df_daily.empty:
             logger.warning(f"[{symbol}] No daily data loaded for level {level}.")
             return []
        df_daily['symbol'] = symbol # 确保有 symbol 列

        # 2. 根据 level 重采样
        df_level_data = df_daily.copy()
        if level == "weekly":
            min_days_for_resample = 5 * max(self.ma_long, trend_check_window) # 粗略估计需要的天数
            if len(df_level_data) < min_days_for_resample:
                logger.debug(f"[{symbol}] Not enough data ({len(df_level_data)} days) for weekly resampling and MA/trend calculation.")
                return []
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try:
                df_level_data = df_daily.set_index('date').resample('W-FRI', closed='right', label='right').agg(agg).dropna(how='all').reset_index()
            except Exception as e: logger.error(f"[{symbol}] Error weekly resampling: {e}", exc_info=True); return []
        elif level == "monthly":
            min_days_for_resample = 20 * max(self.ma_long, trend_check_window) # 粗略估计
            if len(df_level_data) < min_days_for_resample:
                 logger.debug(f"[{symbol}] Not enough data ({len(df_level_data)} days) for monthly resampling and MA/trend calculation.")
                 return []
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'symbol': 'last'}
            try:
                 df_level_data = df_daily.set_index('date').resample('ME').agg(agg).dropna(how='all').reset_index()
            except Exception as e: logger.error(f"[{symbol}] Error monthly resampling: {e}", exc_info=True); return []

        if df_level_data.empty:
            logger.warning(f"[{symbol}] Resampled {level} DataFrame empty.")
            return []

        # 3. 计算均线
        df_with_ma = self.calculate_ma(df_level_data, ma_periods)
        if df_with_ma.empty:
            logger.warning(f"[{symbol}] DataFrame empty after MA calculation for {level}.")
            return []

        # 4. 调用信号检测逻辑 (只检测最后一天)
        signals = self.find_pullback_signals_on_last_day(df_with_ma)

        # 5. 添加 timeframe
        for sig in signals:
            sig['timeframe'] = level
        return signals

    def find_signals(self, symbol: str, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """
        统一接口，扫描日线、周线、月线的回踩信号。
        """
        logger.info(f"🔍 [{symbol}] Scanning for MA Pullback from {start_date} to {end_date}")
        results = {}
        # 决定要运行哪些级别，如果 self.timeframe 是 'multi' 就全跑
        levels_to_run = ['daily', 'weekly', 'monthly'] if self.timeframe == 'multi' else [self.timeframe]

        for level in levels_to_run:
            if level not in ['daily', 'weekly', 'monthly']:
                logger.warning(f"Unsupported timeframe '{level}' requested. Skipping.")
                continue
            try:
                results[level] = self.process_level(symbol, start_date, end_date, level=level)
                logger.info(f"[{symbol}] Found {len(results.get(level,[]))} signals for {level} level.")
            except Exception as e:
                 logger.error(f"[{symbol}] Error processing {level} level: {e}", exc_info=True)
                 results[level] = [] # 出错则该级别无信号
        return results


# --- Main Execution Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    strategy = MAPullbackStrategy() # 实例化新策略
    engine = get_engine_instance()

    if engine is None: logger.critical("DB engine NG. Exit."); exit()

    # 1. 获取股票列表 (与之前类似)
    try:
        # stock_query = "SELECT code as symbol FROM stock_list WHERE code like '00%' LIMIT 20" # DEBUG: 限制范围
        stock_query = "SELECT code as symbol FROM stock_list" # 正式运行
        df_stocks = pd.read_sql(stock_query, con=engine)
        stock_list = df_stocks['symbol'].tolist()
        logger.info(f"📈 Stocks to process: {len(stock_list)}")
        if not stock_list: logger.warning("Stock list empty."); exit()
    except Exception as e: logger.error(f"Failed to get stock list: {e}", exc_info=True); exit()

    # 2. 获取日期范围 (与之前类似)
    try:
        date_query = "SELECT MAX(date) AS max_date FROM stock_daily"
        df_dates = pd.read_sql(date_query, con=engine)
        if pd.isna(df_dates.at[0, 'max_date']):
             logger.error("Cannot get max date."); end_date_obj = datetime.now().date()
             logger.warning(f"Using current date {end_date_obj.strftime('%Y-%m-%d')} as end date.")
        else: end_date_obj = pd.to_datetime(df_dates.at[0, 'max_date']).date()
        end_date = end_date_obj.strftime('%Y-%m-%d')
        # 需要足够长的历史数据来计算 MA30 和趋势
        start_date = (end_date_obj - pd.DateOffset(years=2)).strftime('%Y-%m-%d') # 至少2年数据
        logger.info(f"📅 Analysis period: {start_date} to {end_date}")
        logger.info(f"🛎 Filtering signals ONLY on: {end_date}")
    except Exception as e: logger.error(f"Failed to get date range: {e}", exc_info=True); exit()

    # --- 开始扫描 ---
    initial_signals = []
    processed_count = 0
    total_stocks = len(stock_list)
    scan_start_time = time.time()

    for symbol in stock_list:
        processed_count += 1
        if processed_count % 100 == 0: # 每处理100只股票打印一次进度
            logger.info(f"--- Scanning {processed_count}/{total_stocks}: {symbol} ---")

        try:
            # 获取该股票所有周期的技术信号
            results = strategy.find_signals(symbol, str(start_date), str(end_date))
            # 收集发生在最后一天的信号
            for level, signals_in_level in results.items():
                for sig in signals_in_level:
                    if sig['signal_date'] == end_date:
                        logger.info(f"✅ [{symbol}] Tech Signal: {strategy.name} on {level.upper()} at {sig['signal_date']}")
                        initial_signals.append({
                            "symbol": symbol,
                            "signal_date": sig['signal_date'],
                            "strategy": strategy.name, # 使用新策略的名称
                            "timeframe": level
                        })
        except Exception as e:
            logger.error(f"Error processing technical signals for {symbol}: {e}", exc_info=True)

    scan_end_time = time.time()
    logger.info(f"\n📊 Found {len(initial_signals)} initial technical signals ({strategy.name}) on {end_date}. Scan took {scan_end_time - scan_start_time:.2f}s.")

    # --- 基本面分析 ---
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
            # logger.info(f"--- Fund Analysis {fundamental_processed_count}/{total_initial}: {symbol} on {signal_date_str} ---")

            # 调用基本面数据获取与分析函数
            fundamental_data = strategy.get_fundamental_data(symbol, signal_date_str)

            # 合并技术信号和基本面结果
            enhanced_sig = {**initial_sig, **fundamental_data}
            final_enhanced_signals.append(enhanced_sig)

            # 打印简要基本面结果
            error_msg = enhanced_sig.get('error_reason')
            error_str = f", Errors='{error_msg}'" if error_msg else ""
            logger.info(f"-> Fund Results {symbol} ({initial_sig['timeframe']}): PE={enhanced_sig.get('pe', np.nan):.2f}, PE<30={enhanced_sig.get('pe_lt_30')}, Growth+={enhanced_sig.get('growth_positive')}, PEG<1={enhanced_sig.get('peg_like_lt_1')}, 3YProfit+={enhanced_sig.get('net_profit_positive_3y_latest')}{error_str}")

        fund_end_time = time.time()
        logger.info(f"Fundamental analysis took {fund_end_time - fund_start_time:.2f}s.")

    # --- 保存结果 ---
    if final_enhanced_signals:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'output')
        try: os.makedirs(output_dir, exist_ok=True)
        except OSError as ose: logger.error(f"Cannot create output dir {output_dir}: {ose}"); output_dir = "."

        columns_order = [ # 与之前保持一致
            'symbol', 'signal_date', 'strategy', 'timeframe',
            'net_profit_positive_3y_latest', 'pe', 'pe_lt_30',
            'revenue_growth_yoy', 'profit_growth_yoy', 'growth_positive',
            'peg_like_ratio', 'peg_like_lt_1', 'error_reason'
        ]
        df_final = pd.DataFrame(final_enhanced_signals)
        for col in columns_order: # 确保列存在
            if col not in df_final.columns: df_final[col] = np.nan
        df_final = df_final[columns_order] # 排序

        # 格式化输出
        float_cols = ['pe', 'revenue_growth_yoy', 'profit_growth_yoy', 'peg_like_ratio']
        for col in float_cols:
             if col in df_final.columns:
                  df_final[col] = df_final[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) and isinstance(x, (int, float)) else ('+Inf' if x == np.inf else ('-Inf' if x == -np.inf else None)))

        filename = os.path.join(output_dir, f"signals_{strategy.name}_with_fundamentals_{end_date}.csv") # 文件名包含策略名
        try:
            df_final.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"\n✅ Final signals saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV {filename}: {e}", exc_info=True)
    else:
        logger.info("\nℹ️ No signals found meeting criteria or after fundamental analysis.")

    total_end_time = time.time()
    logger.info(f"\n--- Script finished. Total execution time: {total_end_time - scan_start_time:.2f} seconds ---")