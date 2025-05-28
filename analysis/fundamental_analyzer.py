# analysis/fundamental_analyzer.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import date, timedelta, datetime  # datetime 确保存在
from sqlalchemy.orm import Session

# 从您的项目中导入
from db.database import get_db_session
from db import crud  # 确保 crud 可被导入

# from db.models import StockFinancial # 如果需要显式类型提示crud返回内容中的对象

logger = logging.getLogger(__name__)

# Constants from your multi_level_cross_strategy_new.py
NET_PROFIT_FIELD = '归属于母公司所有者的净利润'
REVENUE_FIELD = '营业总收入'
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0


class FundamentalAnalyzer:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def _safe_extract_json_value(self, report_data_json: Optional[Dict], key_chinese_name: str) -> Optional[float]:
        """
        安全地从类 JSONB 字典中提取数值。
        优先尝试精确匹配 key_chinese_name。
        如果精确匹配失败或值无效，则尝试查找 JSON 中键名包含 key_chinese_name 的项。
        处理 None、空字符串、特殊非数字字符串、带单位（'万', '亿'）的数字、数字中的逗号及转换错误。
        """
        if report_data_json is None:
            return None

        # 内部辅助函数，用于解析具体的值
        def _parse_value(value_to_parse: Any, original_key_for_log: str) -> Optional[float]:
            if value_to_parse is None:
                return None

            # 处理值已经是数字类型（int, float, bool）的情况
            if isinstance(value_to_parse, (int, float)):
                num_val = float(value_to_parse)
                if np.isnan(num_val) or np.isinf(num_val):
                    return None
                return num_val

            # 期望 value_to_parse 是字符串
            if not isinstance(value_to_parse, str):
                logger.debug(
                    f"值 '{value_to_parse}' (来自键 '{original_key_for_log}') 不是字符串或可识别的数字类型。类型为: {type(value_to_parse)}.")
                return None

            # 处理字符串值
            processed_value_str = value_to_parse.strip()

            if not processed_value_str:  # 处理空字符串或strip后变为空字符串的情况
                return None

            if processed_value_str.lower() in ('--', 'n/a', '不适用', 'nan'):
                return None

            multiplier = 1.0
            numeric_part_str = processed_value_str

            if '亿' in numeric_part_str:
                multiplier = 100_000_000.0
                numeric_part_str = numeric_part_str.replace('亿', '')
            elif '万' in numeric_part_str:
                multiplier = 10_000.0
                numeric_part_str = numeric_part_str.replace('万', '')

            numeric_part_str = numeric_part_str.replace(',', '')

            if not numeric_part_str.strip():
                logger.debug(
                    f"值 '{value_to_parse}' (来自键 '{original_key_for_log}') 在处理后得到空的数字部分 ('{numeric_part_str}')。")
                return None

            try:
                num_val = float(numeric_part_str) * multiplier
                if np.isnan(num_val) or np.isinf(num_val):
                    return None
                return num_val
            except (ValueError, TypeError):
                logger.debug(f"无法将值 '{value_to_parse}' (来自键 '{original_key_for_log}') 转换为浮点数。"
                             f"尝试解析的部分为: '{numeric_part_str}'，乘数为 {multiplier}.")
                return None

        # --- 主要逻辑开始 ---
        # 1. 尝试精确匹配
        exact_value = report_data_json.get(key_chinese_name)
        parsed_exact_value = _parse_value(exact_value, key_chinese_name)  # 使用辅助函数解析

        if parsed_exact_value is not None:
            logger.debug(f"为 '{key_chinese_name}' 找到精确匹配，原始值: '{exact_value}', 解析后: {parsed_exact_value}")
            return parsed_exact_value
        else:
            # 如果 exact_value 存在但解析失败，日志已在 _parse_value 中记录
            # 如果 exact_value 本身是 None (键不存在)，_parse_value 会返回 None
            logger.debug(f"未能通过精确匹配 '{key_chinese_name}' 获取可用值 (原始值: '{exact_value}')。尝试模糊匹配...")

        # 2. 如果精确匹配失败或值无效，则尝试模糊匹配 (查找包含 key_chinese_name 的键)
        #    迭代顺序通常是字典的插入顺序 (Python 3.7+)
        for json_actual_key, json_actual_value in report_data_json.items():
            if isinstance(json_actual_key, str) and key_chinese_name in json_actual_key:
                # 如果这个模糊匹配到的键就是我们精确查找过的键，并且精确查找时它的值是存在的但解析失败了，
                # 那么就不需要再尝试解析同 一个无效值了。
                # (如果精确查找时键不存在即 exact_value is None，这里仍会尝试，这是期望的行为，因为我们是基于 json_actual_key 去取 json_actual_value)
                if json_actual_key == key_chinese_name and exact_value is not None:
                    continue  # 之前精确匹配时已尝试过此键且值解析失败

                parsed_like_value = _parse_value(json_actual_value, json_actual_key)  # 使用辅助函数解析
                if parsed_like_value is not None:
                    logger.debug(f"模糊匹配成功：JSON键 '{json_actual_key}' 包含目标 '{key_chinese_name}'。"
                                 f"原始值: '{json_actual_value}', 解析后: {parsed_like_value}")
                    return parsed_like_value

        logger.debug(f"未能为 '{key_chinese_name}' 通过精确或模糊匹配找到可用值。")
        return None

    def _get_total_shares(self, stock_code: str, trade_date: date) -> Optional[float]:
        """
        Calculates total shares using (成交量 / turnover) for a given date.
        Relies on a crud function to get daily market data.
        MODIFIED: Assumes volume is in lots and 1 lot = 100 shares.
        """
        daily_data_entry = None
        try:
            daily_data_entry = crud.get_stock_daily_for_date(self.db_session, symbol=stock_code, trade_date=trade_date)
        except AttributeError:  # Should not happen if crud is imported correctly
            logger.warning(f"crud.get_stock_daily_for_date not found. Please implement it or check import.")
            return None  # Explicitly return None here
        except Exception as e:
            logger.error(f"Error fetching daily market data for {stock_code} on {trade_date}: {e}", exc_info=True)
            return None  # Explicitly return None here

        if daily_data_entry and hasattr(daily_data_entry, 'volume') and hasattr(daily_data_entry, 'turnover'):
            volume_in_lots = daily_data_entry.volume
            turnover_rate = daily_data_entry.turnover

            if volume_in_lots is not None and turnover_rate is not None and turnover_rate > 1e-9:
                # 假设1手 = 100股 (根据之前的讨论，PE值过小的问题)
                actual_volume_in_shares = volume_in_lots * 100.0
                if actual_volume_in_shares > 0:  # Ensure shares are positive
                    total_shares = actual_volume_in_shares / (turnover_rate / 100.0)
                    if total_shares > 0:  # Ensure calculated total_shares is positive
                        return total_shares
                    else:
                        logger.debug(
                            f"[{stock_code}@{trade_date.isoformat()}] Calculated total shares is not positive ({total_shares}). Volume: {volume_in_lots}, Turnover: {turnover_rate}")
                else:
                    logger.debug(
                        f"[{stock_code}@{trade_date.isoformat()}] Volume in lots ({volume_in_lots}) results in non-positive actual shares.")
            else:
                logger.debug(
                    f"[{stock_code}@{trade_date.isoformat()}] Volume or turnover rate is invalid for total shares calculation (V_lots:{volume_in_lots}, T:{turnover_rate}).")
        else:
            logger.debug(
                f"[{stock_code}@{trade_date.isoformat()}] No daily market data found or missing volume/turnover fields in the fetched data.")
        return None

    def _fetch_financial_reports(self, stock_code: str, signal_date: date, report_type_db_key: str = 'benefit') -> Dict[
        str, Any]:
        """
        Fetches various necessary financial reports (JSON data) up to signal_date
        by using crud.get_financial_reports_for_analyzer.
        """
        # 这个字典将存储最终的 JSON 数据，供分析器其他部分使用
        fetched_reports_json_data = {
            'latest_annual_for_pe': None,
            'latest_any_for_others': None,
            'prev_year_same_q_for_yoy': None,
            'annual_reports_for_3y_check': []
        }

        try:
            # 1. 调用 crud 函数
            # crud.py 中的参数名为 current_signal_date
            reports_from_crud_models = crud.get_financial_reports_for_analyzer(
                db=self.db_session,
                stock_code=stock_code,
                report_type_db_key=report_type_db_key,
                current_signal_date=signal_date
            )

            # 2. 从返回的 StockFinancial 模型对象中提取 .data (JSON)
            latest_annual_sf_model = reports_from_crud_models.get('latest_annual_report')
            if latest_annual_sf_model and hasattr(latest_annual_sf_model, 'data'):
                fetched_reports_json_data['latest_annual_for_pe'] = latest_annual_sf_model.data

            latest_quarterly_sf_model = reports_from_crud_models.get(
                'latest_quarterly_report')  # 这对应之前的 'latest_any_for_others'
            if latest_quarterly_sf_model and hasattr(latest_quarterly_sf_model, 'data'):
                fetched_reports_json_data['latest_any_for_others'] = latest_quarterly_sf_model.data

            prev_year_sf_model = reports_from_crud_models.get('previous_year_same_quarter_report')
            if prev_year_sf_model and hasattr(prev_year_sf_model, 'data'):
                fetched_reports_json_data['prev_year_same_q_for_yoy'] = prev_year_sf_model.data

            # 处理年报列表
            last_3_annual_sf_models_list = reports_from_crud_models.get('last_3_annual_reports', [])  # 使用 .get 提供默认空列表
            for sf_model in last_3_annual_sf_models_list:
                if sf_model and hasattr(sf_model, 'data'):  # 确保对象存在且有 .data 属性
                    fetched_reports_json_data['annual_reports_for_3y_check'].append(sf_model.data)

            # 添加日志来确认从crud获取的数据情况
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] CRUD 函数 ({crud.get_financial_reports_for_analyzer.__name__}) 返回的报告摘要:")
            logger.debug(
                f"  最新年报 (用于PE): {'已找到' if fetched_reports_json_data['latest_annual_for_pe'] else '未找到'}")
            logger.debug(
                f"  最新季报/任何报告: {'已找到' if fetched_reports_json_data['latest_any_for_others'] else '未找到'}")
            logger.debug(
                f"  去年同期报告 (用于YoY): {'已找到' if fetched_reports_json_data['prev_year_same_q_for_yoy'] else '未找到'}")
            logger.debug(
                f"  过去3年年报数量 (用于3年盈利检查): {len(fetched_reports_json_data['annual_reports_for_3y_check'])}")

        except AttributeError as e:  # 例如，如果 .data 属性不存在
            logger.error(
                f"在处理来自 crud 的报告时发生 AttributeError ({stock_code} @ {signal_date}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"调用 crud.get_financial_reports_for_analyzer 时出错 ({stock_code} @ {signal_date}): {e}",
                         exc_info=True)
            # 在这种情况下，fetched_reports_json_data 将保持其初始的 None/空列表值

        return fetched_reports_json_data

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
        total_shares = self._get_total_shares(stock_code, signal_date)  # 使用已更新的 _get_total_shares
        market_cap = np.nan
        if total_shares is not None and current_price is not None:
            if total_shares > 1e-9:  # Ensure total_shares is positive and non-negligible
                market_cap = total_shares * current_price
            else:
                error_messages.append(f"总股本计算为非正数或过小 ({total_shares}).")
        else:
            if total_shares is None: error_messages.append(
                "总股本无法计算 (可能缺少日成交量/换手率数据，或数据无效).")
            if current_price is None: error_messages.append("当前价格未提供，无法计算市值.")

        if pd.notna(market_cap):
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] Calculated Market Cap: {market_cap:.2f} using Total Shares: {total_shares:.2f} and Price: {current_price:.2f}")
        else:
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] Market Cap could not be calculated. Total Shares: {total_shares}, Price: {current_price}")

        # 2. Fetch all necessary financial report data
        financial_datas = self._fetch_financial_reports(stock_code, signal_date,
                                                        report_type_db_key='benefit')  # 使用已更新的 _fetch_financial_reports

        latest_annual_report_json = financial_datas['latest_annual_for_pe']
        latest_any_report_json = financial_datas['latest_any_for_others']
        prev_year_same_q_report_json = financial_datas['prev_year_same_q_for_yoy']
        annual_reports_for_3y_list = financial_datas['annual_reports_for_3y_check']

        # 3. Calculate PE
        if pd.notna(market_cap) and latest_annual_report_json:
            net_profit_for_pe = self._safe_extract_json_value(latest_annual_report_json, NET_PROFIT_FIELD)
            if net_profit_for_pe is not None and net_profit_for_pe > 1e-6:
                results['pe'] = market_cap / net_profit_for_pe  # PE calculation
                results['pe_lt_30'] = results['pe'] < PE_THRESHOLD
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] PE Calculated: {results['pe']:.4f} (Market Cap: {market_cap:.2f}, Annual Profit: {net_profit_for_pe:.2f})")
            else:
                error_messages.append(
                    f"PE计算失败：年度净利润无效或非正 ({net_profit_for_pe}).")
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] PE not calculated: Invalid or non-positive annual profit for PE ({net_profit_for_pe}). Latest annual report JSON: {bool(latest_annual_report_json)}")

        else:
            if pd.isna(market_cap): error_messages.append("PE计算失败：市值是 NaN.")
            if not latest_annual_report_json: error_messages.append(
                "PE计算失败：未找到最新的年度报告.")
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] PE not calculated: Market cap is NaN ({pd.isna(market_cap)}), Latest annual report found ({bool(latest_annual_report_json)}).")

        # 4. Calculate YoY Growths and Growth Positive
        rev_growth = np.nan
        prof_growth = np.nan
        if latest_any_report_json and prev_year_same_q_report_json:
            current_revenue = self._safe_extract_json_value(latest_any_report_json, REVENUE_FIELD)
            prev_revenue = self._safe_extract_json_value(prev_year_same_q_report_json, REVENUE_FIELD)
            current_profit = self._safe_extract_json_value(latest_any_report_json, NET_PROFIT_FIELD)
            prev_profit = self._safe_extract_json_value(prev_year_same_q_report_json, NET_PROFIT_FIELD)

            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] YoY Data: CurrRev={current_revenue}, PrevRev={prev_revenue}, CurrProf={current_profit}, PrevProf={prev_profit}")

            if current_revenue is not None and prev_revenue is not None:
                if abs(prev_revenue) > 1e-9:  # Avoid division by zero or near-zero
                    rev_growth = (current_revenue - prev_revenue) / abs(prev_revenue)
                    results['revenue_growth_yoy'] = rev_growth
                else:
                    error_messages.append("营收同比增长计算失败：上一期营收接近零.")
            else:
                error_messages.append("营收同比增长计算失败：缺少当前或上一期营收数据.")

            if current_profit is not None and prev_profit is not None:
                if abs(prev_profit) > 1e-9:  # Avoid division by zero or near-zero
                    prof_growth = (current_profit - prev_profit) / abs(prev_profit)
                    results['profit_growth_yoy'] = prof_growth
                elif current_profit > 1e-9 and (prev_profit is None or abs(
                        prev_profit) < 1e-9):  # Turned profitable from zero/loss or no prev data
                    prof_growth = np.inf  # Or a very large number if np.inf is problematic for storage/JSON
                    results['profit_growth_yoy'] = prof_growth
                else:  # handles prev_profit is negative or other cases
                    error_messages.append(
                        f"利润同比增长计算失败：无效的利润数据 (当前:{current_profit}, 上一期:{prev_profit}).")
            else:
                error_messages.append("利润同比增长计算失败：缺少当前或上一期利润数据.")

            if pd.notna(rev_growth) and pd.notna(prof_growth):
                if np.isinf(prof_growth) and prof_growth > 0:  # Turned profitable
                    results['growth_positive'] = (
                                rev_growth > 1e-6)  # Check only revenue growth if profit turned inf positive
                elif not np.isinf(prof_growth):  # Both are finite numbers
                    results['growth_positive'] = (rev_growth > 1e-6) and (prof_growth > 1e-6)
                # else case: prof_growth is -inf or NaN, growth_positive remains None

            if results['growth_positive'] is not None:
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] Growth Positive Check: RevGrowth={rev_growth:.4f}, ProfGrowth={prof_growth:.4f}, Result={results['growth_positive']}")


        else:
            error_messages.append("同比增长率计算失败：缺少当期报告或去年同期报告.")
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] YoY growths not calculated: Latest any report found ({bool(latest_any_report_json)}), Prev year same Q report found ({bool(prev_year_same_q_report_json)}).")

        # 5. Calculate PEG-like Ratio
        current_pe_val = results.get('pe')
        finite_prof_growth_for_peg = prof_growth if pd.notna(prof_growth) and np.isfinite(prof_growth) else np.nan

        if pd.notna(current_pe_val) and pd.notna(finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6:
            peg_like = current_pe_val / (finite_prof_growth_for_peg * 100.0)
            results['peg_like_ratio'] = peg_like
            results['peg_like_lt_1'] = peg_like < PEG_LIKE_THRESHOLD
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] PEG-like Calculated: {peg_like:.4f} (PE: {current_pe_val:.4f}, Finite Profit Growth for PEG: {finite_prof_growth_for_peg:.4f})")

        else:
            if pd.isna(current_pe_val): error_messages.append("PEG近似值计算失败：PE 是 NaN.")
            if not (pd.notna(finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6):
                error_messages.append(
                    f"PEG近似值计算失败：利润增长率无效或非正 ({finite_prof_growth_for_peg}).")
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] PEG-like not calculated: PE is NaN ({pd.isna(current_pe_val)}), Finite profit growth for PEG ({finite_prof_growth_for_peg}).")

        # 6. Calculate Net Profit Positive 3 Years Latest
        num_annual_reports_for_check = len(annual_reports_for_3y_list)
        all_checked_profits_positive = True  # Assume true initially

        if latest_any_report_json:
            latest_profit = self._safe_extract_json_value(latest_any_report_json, NET_PROFIT_FIELD)
            if latest_profit is None or latest_profit <= 1e-6:
                all_checked_profits_positive = False
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Latest report profit non-positive ({latest_profit}).")
        else:
            all_checked_profits_positive = False  # Cannot confirm if latest report is missing
            error_messages.append("3年盈利检查失败：缺少最新的报告数据.")
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Missing latest (any type) report data.")

        if all_checked_profits_positive:  # Only proceed if latest profit was positive
            if num_annual_reports_for_check < 3:  # Strict: require 3 full past annual reports from the crud function
                # `get_financial_reports_for_analyzer` tries to get last 3, so this check is on its output
                results['net_profit_positive_3y_latest'] = None  # Not enough data for a firm True/False
                error_messages.append(
                    f"3年盈利检查存疑：找到的年度报告数量不足 ({num_annual_reports_for_check}/3).")
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Insufficient annual reports found ({num_annual_reports_for_check}/3) by crud.")
            else:
                # Check the annual reports obtained (crud function should give up to 3 most recent relevant ones)
                annual_profits_log = []
                for i in range(min(3, num_annual_reports_for_check)):  # Iterate up to 3 or actual count
                    annual_report_json_data = annual_reports_for_3y_list[i]  # crud already sorted them desc
                    annual_profit = self._safe_extract_json_value(annual_report_json_data, NET_PROFIT_FIELD)
                    annual_profits_log.append(annual_profit)
                    if annual_profit is None or annual_profit <= 1e-6:
                        all_checked_profits_positive = False
                        break
                results['net_profit_positive_3y_latest'] = all_checked_profits_positive
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Annual profits checked: {annual_profits_log}. Result: {results['net_profit_positive_3y_latest']}")
        else:  # If latest profit was already non-positive or missing
            results['net_profit_positive_3y_latest'] = False  # Cannot be true if latest is bad or missing
            if not latest_any_report_json:  # If it was false due to missing latest report
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Set to False due to missing latest report, annuals not checked further for this flag.")
            else:  # If it was false due to latest report profit being non-positive
                logger.debug(
                    f"[{stock_code}@{signal_date.isoformat()}] 3Y Profit Check: Set to False due to non-positive latest report profit, annuals not checked further for this flag.")

        if error_messages:
            results['error_reason'] = "; ".join(sorted(list(set(error_messages))))  # Unique sorted errors
            logger.debug(
                f"[{stock_code}@{signal_date.isoformat()}] Fundamental analysis errors: {results['error_reason']}")

        # Round relevant float results for cleaner output
        for key in ['pe', 'revenue_growth_yoy', 'profit_growth_yoy', 'peg_like_ratio']:
            if key in results and pd.notna(results[key]):
                if np.isinf(results[key]):
                    results[key] = str(results[key])  # Store "inf" or "-inf" as string
                elif np.isfinite(results[key]):
                    results[key] = round(results[key], 4)
                # else NaN remains NaN, which is fine for JSON (becomes null)

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