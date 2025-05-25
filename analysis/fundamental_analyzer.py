# analysis/fundamental_analyzer.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session

from db.database import get_db_session

# Assume these models are defined in your quant_platform.db.models
# from quant_platform.db.models import StockDaily, StockFinancial

# Assume these crud functions will be available in quant_platform.db.crud
# We will define their expected signatures and behavior in comments.
# from quant_platform.db import crud

logger = logging.getLogger(__name__)

# Constants from your multi_level_cross_strategy_new.py
NET_PROFIT_FIELD = '归属于母公司所有者的净利润'  # User confirmed this is the JSON key
REVENUE_FIELD = '营业总收入'  # User confirmed this is the JSON key
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0


class FundamentalAnalyzer:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def _safe_extract_json_value(self, report_data_json: Optional[Dict], key_chinese_name: str) -> Optional[float]:
        """
        安全地从类 JSONB 字典（从 StockFinancial.data 解析而来）中提取数值。
        处理 None、空字符串、特殊的非数字字符串（如 '--', 'N/A'）、
        带中文单位（'万', '亿'）的数字、数字中的逗号以及转换错误。
        """
        if report_data_json is None:
            return None

        value = report_data_json.get(key_chinese_name)

        if value is None:  # 处理键存在但值为 None 的情况
            return None

        # 处理值已经是数字类型（int, float, bool）的情况
        # bool 是 int 的子类，所以 False -> 0.0, True -> 1.0
        if isinstance(value, (int, float)):
            num_value = float(value)
            if np.isnan(num_value) or np.isinf(num_value):
                return None
            return num_value

        # 从此处开始，我们期望 value 是字符串或未处理的类型
        if not isinstance(value, str):
            logger.debug(f"键 '{key_chinese_name}' 的值 '{value}' 不是字符串或可识别的数字类型。类型为: {type(value)}.")
            return None

        # 处理字符串值
        processed_value_str = value.strip()

        if not processed_value_str:  # 处理空字符串或strip后变为空字符串的情况
            return None

        # 检查特殊的非数字字符串（不区分大小写）
        # 这些应在尝试带单位的数字转换前检查
        if processed_value_str.lower() in ('--', 'n/a', '不适用', 'nan'):
            return None

        multiplier = 1.0
        # 在移除单位/逗号之前，存储字符串的副本以进行更准确的处理尝试
        numeric_part_str = processed_value_str

        if '亿' in numeric_part_str:
            multiplier = 100_000_000.0
            numeric_part_str = numeric_part_str.replace('亿', '')
        elif '万' in numeric_part_str:  # 使用 elif 以避免在已处理“亿”后再次处理“万”
            multiplier = 10_000.0
            numeric_part_str = numeric_part_str.replace('万', '')

        # 移除单位后，再移除逗号，以处理类似 "1,234.56万" -> "1,234.56" -> "1234.56" 的情况
        numeric_part_str = numeric_part_str.replace(',', '')

        # 最后检查替换后字符串是否为空（例如，如果原始值仅为 "亿" 或 ","）
        if not numeric_part_str.strip():  # 再次 strip 以防单位/逗号周围有空格
            logger.debug(
                f"键 '{key_chinese_name}' 的值 '{value}' (原始JSON值) 在处理后得到空的数字部分 ('{numeric_part_str}')。")
            return None

        try:
            num_value = float(numeric_part_str) * multiplier
            if np.isnan(num_value) or np.isinf(num_value):
                return None
            return num_value
        except (ValueError, TypeError):
            # 记录从 JSON 中获取的原始值，以增加清晰度
            logger.debug(f"无法将键 '{key_chinese_name}' 的财务值 '{value}' (原始JSON值) 转换为浮点数。"
                         f"尝试解析的部分为: '{numeric_part_str}'，乘数为 {multiplier}.")
            return None

    def _get_total_shares(self, stock_code: str, trade_date: date) -> Optional[float]:
        """
        Calculates total shares using (成交量 / turnover) for a given date.
        Relies on a crud function to get daily market data.
        """
        # ASSUMED CRUD FUNCTION:
        # def get_daily_market_data(db_session: Session, stock_code: str, trade_date: date) -> Optional[StockDailyModel]:
        #     """Fetches StockDaily entry for a specific stock and date."""
        #     # return db_session.query(StockDaily).filter_by(symbol=stock_code, date=trade_date).first()
        # This function needs to be implemented in your crud.py
        # from quant_platform.db.crud import get_daily_market_data # Example import
        # For now, we'll use a placeholder for the call.

        # Placeholder for actual crud call:
        # daily_data_entry = crud.get_daily_market_data(self.db_session, stock_code=stock_code, trade_date=trade_date)
        # For this example, let's assume crud.py is adapted or such a function exists:
        from db import crud  # Assuming crud.py is in this path relative to project root

        daily_data_entry = None
        try:
            # This is a conceptual call. Implement `get_daily_market_data` in your `crud.py`
            # to query the 'stock_daily' table.
            daily_data_entry = crud.get_stock_daily_for_date(self.db_session, symbol=stock_code, trade_date=trade_date)
        except AttributeError:
            logger.warning(f"crud.get_stock_daily_for_date not found. Please implement it.")
        except Exception as e:
            logger.error(f"Error fetching daily market data for {stock_code} on {trade_date}: {e}", exc_info=True)
            return None

        if daily_data_entry and hasattr(daily_data_entry, 'volume') and hasattr(daily_data_entry, 'turnover'):
            volume_in_lots = daily_data_entry.volume
            turnover_rate = daily_data_entry.turnover

            if volume_in_lots is not None and turnover_rate is not None and turnover_rate > 1e-9:
                # 假设1手 = 100股
                actual_volume_in_shares = volume_in_lots * 100.0
                return actual_volume_in_shares / (turnover_rate / 100.0)
            else:
                logger.debug(
                    f"[{stock_code}@{trade_date.isoformat()}] Volume or turnover rate is invalid for total shares calculation (V:{volume}, T:{turnover_rate}).")
        else:
            logger.debug(
                f"[{stock_code}@{trade_date.isoformat()}] No daily market data found or missing volume/turnover.")
        return None

    def _fetch_financial_reports(self, stock_code: str, signal_date: date, report_type_db_key: str = 'benefit') -> Dict[
        str, Any]:
        """
        Fetches various necessary financial reports (JSON data) up to signal_date.
        'benefit' report_type is assumed for income statement items like profit and revenue.
        Relies on a flexible crud function.

        Returns a dictionary with keys like:
        - 'latest_annual_for_pe': Dict (JSON data of the latest annual report)
        - 'latest_any_for_others': Dict (JSON data of the latest report of any type)
        - 'prev_year_same_q_for_yoy': Dict (JSON data of the report from prev year, same quarter as 'latest_any_for_others')
        - 'annual_reports_for_3y_check': List[Dict] (JSON data of last 3-4 annual reports for 3-year profit check)
        """
        fetched_reports = {
            'latest_annual_for_pe': None,
            'latest_any_for_others': None,
            'prev_year_same_q_for_yoy': None,
            'annual_reports_for_3y_check': []
        }

        # ASSUMED CRUD FUNCTION:
        # def get_financial_reports(db_session: Session, stock_code: str, report_date_lte: date,
        #                           report_type_db_key: str, # e.g. 'benefit', 'balance', 'cashflow'
        #                           report_period_type: Optional[str] = None, # e.g. "年报", "一季报", "中报", "三季报"
        #                                                                    # This might map to specific month/day in report_date
        #                           limit: Optional[int] = None,
        #                           exact_report_year: Optional[int] = None,
        #                           order_by_desc: List[str] = ['report_date']) -> List[StockFinancialModel]:
        #      """Fetches StockFinancial entries. StockFinancialModel.data is the parsed JSON."""
        # This function needs to be implemented in your crud.py.
        # It should be flexible enough to get:
        # 1. Latest annual report (report_date month=12, day=31, report_date_lte=signal_date, limit=1)
        # 2. Absolute latest report of any type (report_date_lte=signal_date, limit=1, order by report_date DESC, then perhaps publish_date DESC)
        # 3. Previous year's report of a specific type and year.
        # 4. List of N recent annual reports.

        # For now, using placeholder calls. `loader.load_multiple_financial_reports` is a good reference.
        from db import crud  # Assuming crud.py
        from data_processing import loader  # For adapting load_multiple_financial_reports logic

        try:
            # 1. Latest Annual Report for PE (report_date is 12-31)
            # We need a crud function that can get the latest annual report specifically.
            # Let's assume crud.get_latest_annual_financial_report(session, stock_code, report_date_lte, report_type_db_key)
            # Based on loader.py, we can adapt load_multiple_financial_reports's behavior
            all_relevant_reports = loader.load_multiple_financial_reports(

                symbol=stock_code,
                report_type=report_type_db_key,
                num_years=4,  # Fetch enough years to cover 3-year check and recent ones

            )

            if not all_relevant_reports:
                logger.warning(
                    f"[{stock_code}@{signal_date.isoformat()}] No financial reports loaded via adapted loader logic.")
                return fetched_reports

            # Sort by report_date descending to easily get latest
            all_relevant_reports.sort(key=lambda r: r['report_date'], reverse=True)

            # Filter reports strictly up to signal_date (loader might fetch slightly beyond based on its internal logic)
            valid_reports = [r for r in all_relevant_reports if r['report_date'] <= signal_date]
            if not valid_reports:
                logger.warning(
                    f"[{stock_code}@{signal_date.isoformat()}] No financial reports found with report_date <= signal_date.")
                return fetched_reports

            # Find latest annual for PE
            for report in valid_reports:
                if report['report_date'].month == 12 and report['report_date'].day == 31:
                    fetched_reports['latest_annual_for_pe'] = report['data']
                    break

            # Find latest any type for others
            if valid_reports:
                fetched_reports['latest_any_for_others'] = valid_reports[0]['data']
                latest_any_report_date = valid_reports[0]['report_date']

                # Find previous year same quarter for YoY
                # This requires knowing the "type" of latest_any_report_date (Q1, S1, Q3, Annual)
                # For simplicity, if latest is Q1 YYYY, we look for Q1 YYYY-1.
                # This logic is complex and relies on consistent report_date patterns.
                # loader.py's load_multiple_financial_reports logic handles this by fetching a range.
                # We search within 'valid_reports' for the corresponding previous year's report.
                target_prev_year_date_month = latest_any_report_date.month
                target_prev_year_date_day = latest_any_report_date.day  # Approx for quarter end
                target_prev_year = latest_any_report_date.year - 1

                for report in valid_reports:
                    if report['report_date'].year == target_prev_year and \
                            report['report_date'].month == target_prev_year_date_month:
                        # Add day check if very specific, but month and year usually suffice for quarters/annual
                        fetched_reports['prev_year_same_q_for_yoy'] = report['data']
                        break

            # Annual reports for 3-year check
            annuals_for_check = []
            for report in valid_reports:
                if report['report_date'].month == 12 and report['report_date'].day == 31:
                    annuals_for_check.append(report['data'])
            fetched_reports['annual_reports_for_3y_check'] = annuals_for_check  # List of JSON dicts

        except AttributeError:
            logger.error(
                f"A required crud function (e.g., for financial reports) or loader function is missing or has changed interface.")
        except Exception as e:
            logger.error(f"Error fetching financial reports for {stock_code} on {signal_date}: {e}", exc_info=True)

        return fetched_reports

    def analyze_stock(self, stock_code: str, signal_date: date, current_price: float) -> Dict[str, Any]:
        """
        Performs fundamental analysis for a stock on a given signal date.
        """
        results = {
            'net_profit_positive_3y_latest': None, 'pe': np.nan, 'pe_lt_30': None,
            'revenue_growth_yoy': np.nan, 'profit_growth_yoy': np.nan, 'growth_positive': None,
            'peg_like_ratio': np.nan, 'peg_like_lt_1': None, 'error_reason': None
        }
        error_messages = []

        # 1. Calculate Market Cap
        total_shares = self._get_total_shares(stock_code, signal_date)
        market_cap = np.nan
        if total_shares is not None and current_price is not None:
            if total_shares > 0:  # Ensure total_shares is positive
                market_cap = total_shares * current_price
            else:
                error_messages.append(f"Total shares calculated as non-positive ({total_shares}).")
        else:
            if total_shares is None: error_messages.append(
                "Total shares could not be calculated (missing daily volume/turnover).")
            if current_price is None: error_messages.append("Current price not provided for market cap.")

        # 2. Fetch all necessary financial report data
        # Use 'benefit' as the report_type_db_key based on multi_level_cross_strategy_new.py
        # This key might need to be 'income' or similar depending on how StockFinancial.report_type is populated
        # loader.py seems to use 'benefit' for income-statement like items.
        financial_datas = self._fetch_financial_reports(stock_code, signal_date, report_type_db_key='benefit')

        latest_annual_report_json = financial_datas['latest_annual_for_pe']
        latest_any_report_json = financial_datas['latest_any_for_others']
        prev_year_same_q_report_json = financial_datas['prev_year_same_q_for_yoy']
        annual_reports_for_3y_list = financial_datas['annual_reports_for_3y_check']  # List of dicts

        # 3. Calculate PE
        if pd.notna(market_cap) and latest_annual_report_json:
            # '归属于母公司所有者的净利润' is confirmed JSON key for net profit
            net_profit_for_pe = self._safe_extract_json_value(latest_annual_report_json, NET_PROFIT_FIELD)
            if net_profit_for_pe is not None and net_profit_for_pe > 1e-6:  # Profit must be positive and non-negligible
                results['pe'] = round(market_cap / net_profit_for_pe, 4)
                results['pe_lt_30'] = results['pe'] < PE_THRESHOLD
            else:
                error_messages.append(
                    f"PE not calculated: Invalid or non-positive annual profit for PE ({net_profit_for_pe}).")
        else:
            if pd.isna(market_cap): error_messages.append("PE not calculated: Market cap is NaN.")
            if not latest_annual_report_json: error_messages.append(
                "PE not calculated: Latest annual report not found.")

        # 4. Calculate YoY Growths and Growth Positive
        rev_growth = np.nan
        prof_growth = np.nan
        if latest_any_report_json and prev_year_same_q_report_json:
            current_revenue = self._safe_extract_json_value(latest_any_report_json, REVENUE_FIELD)
            prev_revenue = self._safe_extract_json_value(prev_year_same_q_report_json, REVENUE_FIELD)
            current_profit = self._safe_extract_json_value(latest_any_report_json, NET_PROFIT_FIELD)
            prev_profit = self._safe_extract_json_value(prev_year_same_q_report_json, NET_PROFIT_FIELD)

            if current_revenue is not None and prev_revenue is not None:
                if abs(prev_revenue) > 1e-6:
                    rev_growth = (current_revenue - prev_revenue) / abs(prev_revenue)
                    results['revenue_growth_yoy'] = round(rev_growth, 4)
                else:
                    error_messages.append("Revenue YoY not calc: Previous revenue near zero.")
            else:
                error_messages.append("Revenue YoY not calc: Missing current or previous revenue data.")

            if current_profit is not None and prev_profit is not None:
                if abs(prev_profit) > 1e-6:
                    prof_growth = (current_profit - prev_profit) / abs(prev_profit)
                    results['profit_growth_yoy'] = round(prof_growth, 4)
                elif current_profit > 0 and prev_profit <= 1e-6:  # From loss/zero to profit
                    prof_growth = np.inf  # Represent as very high growth or handle as special case
                    results['profit_growth_yoy'] = np.inf  # Store as inf
                else:
                    error_messages.append(
                        f"Profit YoY not calc: Invalid profit data (Current:{current_profit}, Prev:{prev_profit}).")
            else:
                error_messages.append("Profit YoY not calc: Missing current or previous profit data.")

            if pd.notna(rev_growth) and pd.notna(prof_growth) and not np.isinf(
                    prof_growth):  # Ensure prof_growth is a finite number for this check
                results['growth_positive'] = (rev_growth > 1e-6) and (prof_growth > 1e-6)
            elif np.isinf(prof_growth) and prof_growth > 0 and pd.notna(
                    rev_growth) and rev_growth > 1e-6:  # Turned profitable
                results['growth_positive'] = True

        else:
            error_messages.append("YoY growths not calc: Missing current period or previous year's same period report.")

        # 5. Calculate PEG-like Ratio
        current_pe_val = results.get('pe')
        # Use finite profit_growth_yoy for PEG calculation
        finite_prof_growth_for_peg = prof_growth if pd.notna(prof_growth) and np.isfinite(prof_growth) else np.nan

        if pd.notna(current_pe_val) and pd.notna(
                finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6:  # Growth rate must be positive
            # Assuming profit_growth_yoy is a decimal (e.g., 0.25 for 25%), multiply by 100 for PEG formula
            peg_like = current_pe_val / (finite_prof_growth_for_peg * 100.0)
            results['peg_like_ratio'] = round(peg_like, 4)
            results['peg_like_lt_1'] = peg_like < PEG_LIKE_THRESHOLD
        else:
            if pd.isna(current_pe_val): error_messages.append("PEG-like not calc: PE is NaN.")
            if not (pd.notna(finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6):
                error_messages.append(
                    f"PEG-like not calc: Profit growth invalid or not positive for PEG ({finite_prof_growth_for_peg}).")

        # 6. Calculate Net Profit Positive 3 Years Latest
        # Checks latest report + last 3 full annual reports
        # Note: annual_reports_for_3y_list already contains relevant annual reports (desc sorted)
        # The logic from multi_level_cross_strategy_new.py's get_fundamental_data needs careful adaptation

        profits_to_check = []
        if latest_any_report_json:  # Include the very latest report's profit
            latest_profit_val = self._safe_extract_json_value(latest_any_report_json, NET_PROFIT_FIELD)
            if latest_profit_val is not None: profits_to_check.append(latest_profit_val)

        # Add up to 3 most recent distinct annual profits
        distinct_annual_profits_added = 0
        for annual_report_json_data in annual_reports_for_3y_list:  # Assumed sorted desc by report_date
            if distinct_annual_profits_added >= 3: break
            profit_val = self._safe_extract_json_value(annual_report_json_data, NET_PROFIT_FIELD)
            if profit_val is not None:
                # This simplistic add might double count if latest_any_report_json was an annual one.
                # A better way is to collect (date, profit) tuples then unique by date and take latest N.
                # For now, just ensuring we have some profit values.
                profits_to_check.append(profit_val)  # This might not be exactly 3 distinct years, but N recent reports
                distinct_annual_profits_added += 1

        if len(profits_to_check) > 0:  # Must have at least some profit data to check
            # Check if *all* collected profits are positive. This depends on how many reports were found.
            # Original logic: "latest + 3 years of annual reports are positive"
            # This needs to be precise: last 3 fiscal years (12-31 reports) + latest interim report.
            # The current 'profits_to_check' might not perfectly represent this.
            # Let's refine: check the latest report and the last N (e.g., 3) ANNUAL reports found.

            num_annual_reports_found = len(annual_reports_for_3y_list)
            all_checked_profits_positive = True

            # Check latest (any type) report if available
            if latest_any_report_json:
                latest_profit = self._safe_extract_json_value(latest_any_report_json, NET_PROFIT_FIELD)
                if latest_profit is None or latest_profit <= 1e-6:
                    all_checked_profits_positive = False
            else:  # If no latest report, cannot confirm this part.
                all_checked_profits_positive = False
                error_messages.append("3Y Profit: Missing latest report data.")

            # Check last 3 available annual reports (if all_checked_profits_positive is still true)
            if all_checked_profits_positive:
                if num_annual_reports_found < 3:  # Strict: require 3 full past annual reports
                    all_checked_profits_positive = False  # Or None if unsure
                    error_messages.append(
                        f"3Y Profit: Insufficient annual reports found ({num_annual_reports_found}/3).")
                else:
                    for i in range(min(3, num_annual_reports_found)):  # Check up to 3 most recent annuals from the list
                        annual_profit = self._safe_extract_json_value(annual_reports_for_3y_list[i], NET_PROFIT_FIELD)
                        if annual_profit is None or annual_profit <= 1e-6:
                            all_checked_profits_positive = False
                            break

            results[
                'net_profit_positive_3y_latest'] = all_checked_profits_positive if num_annual_reports_found >= 3 else None  # Only True if enough data
            if results['net_profit_positive_3y_latest'] is False:
                error_messages.append("3Y Profit: Not all checked recent/annual profits were positive.")

        if error_messages:
            results['error_reason'] = "; ".join(sorted(list(set(error_messages))))
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] Fundamental analysis errors: {results['error_reason']}")

        # Round relevant float results for cleaner output
        for key in ['pe', 'revenue_growth_yoy', 'profit_growth_yoy', 'peg_like_ratio']:
            if pd.notna(results[key]) and np.isfinite(results[key]):
                results[key] = round(results[key], 4)
            elif np.isinf(results[key]):  # Handle infinities explicitly if they are stored
                results[key] = str(results[key])  # e.g. "inf" or "-inf"

        return results


if __name__ == '__main__':
    import logging
    import json
    from datetime import date, datetime # 确保 datetime 也导入了
    import numpy as np
    import pandas as pd

    # --- 配置日志 ---
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # 为 main 代码块定义 logger

    # --- 从项目中导入实际模块 ---
    # 确保 Python 解释器可以找到这些模块
    # 这通常需要从项目的根目录运行，或者设置 PYTHONPATH
    from db.database import get_db_session #
    # FundamentalAnalyzer 内部会导入 db.crud 和 data_processing.loader

    logger.info("--- 开始 FundamentalAnalyzer 针对 603920 的真实数据测试 ---")

    # 1. 获取数据库会话
    # 使用 context manager 来确保会话正确关闭
    db_sess = None
    try:
        with get_db_session() as session:
            if session is None:
                logger.error("未能获取数据库会话，测试中止。请检查数据库配置和连接。")
                exit(1)
            db_sess = session

            # 2. 实例化 FundamentalAnalyzer
            analyzer = FundamentalAnalyzer(db_session=db_sess)

            stock_to_test = "603920"
            test_signal_date = date(2025, 5, 23)
            test_current_price = 26.00

            # 3. 测试 analyze_stock 方法
            logger.info(f"\n--- 测试 analyze_stock: 股票代码 {stock_to_test}, 日期 {test_signal_date}, 价格 {test_current_price} ---")
            analysis_results = analyzer.analyze_stock(
                stock_code=stock_to_test,
                signal_date=test_signal_date,
                current_price=test_current_price
            )

            # 自定义 JSON 编码器以处理 numpy 类型和日期，用于美化打印
            class NpAndDateEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        if np.isnan(obj): return None
                        if np.isinf(obj): return str(obj)
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (datetime, date)): # 处理 date/datetime 对象
                        return obj.isoformat()
                    if pd.isna(obj):
                        return None
                    return super(NpAndDateEncoder, self).default(obj)

            results_str = json.dumps(analysis_results, indent=4, ensure_ascii=False, cls=NpAndDateEncoder)
            print("\n分析结果:")
            print(results_str)

            if analysis_results.get('error_reason'):
                logger.warning(f"股票 {stock_to_test} 的错误原因: {analysis_results.get('error_reason')}")

            # 根据您的实际数据和预期，您可以在这里添加断言
            # 例如:
            # if 'pe' in analysis_results and not pd.isna(analysis_results['pe']):
            #     assert 0 < analysis_results['pe'] < 100, f"PE值异常: {analysis_results['pe']}"

    except Exception as e:
        logger.error(f"运行 FundamentalAnalyzer 测试时发生严重错误: {e}", exc_info=True)
    finally:
        logger.info("\n--- FundamentalAnalyzer 真实数据测试结束 ---")