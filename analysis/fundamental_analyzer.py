# analysis/fundamental_analyzer.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session

from db.database import get_db_session # 您的 get_db_session 可能不同
from db import crud

logger = logging.getLogger(__name__)

# Constants
NET_PROFIT_FIELD = '归属于母公司所有者的净利润'
REVENUE_FIELD = '营业总收入'
EQUITY_FIELD = '归属于母公司所有者权益合计'  # <-- 新增：净资产 (股东权益) 字段名
PE_THRESHOLD = 30.0
PEG_LIKE_THRESHOLD = 1.0


class FundamentalAnalyzer:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    # _safe_extract_json_value 方法保持不变 (来自您提供的版本)
    def _safe_extract_json_value(self, report_data_json: Optional[Dict], key_chinese_name: str) -> Optional[float]:
        # ... (您提供的 _safe_extract_json_value 的完整实现) ...
        if report_data_json is None:
            return None
        def _parse_value(value_to_parse: Any, original_key_for_log: str) -> Optional[float]:
            if value_to_parse is None: return None
            if isinstance(value_to_parse, (int, float)):
                num_val = float(value_to_parse)
                return None if np.isnan(num_val) or np.isinf(num_val) else num_val
            if not isinstance(value_to_parse, str):
                logger.debug(f"Value '{value_to_parse}' (from key '{original_key_for_log}') not str or number. Type: {type(value_to_parse)}.")
                return None
            processed_value_str = value_to_parse.strip()
            if not processed_value_str or processed_value_str.lower() in ('--', 'n/a', '不适用', 'nan'): return None
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
                logger.debug(f"Value '{value_to_parse}' (from key '{original_key_for_log}') processed to empty numeric part ('{numeric_part_str}').")
                return None
            try:
                num_val = float(numeric_part_str) * multiplier
                return None if np.isnan(num_val) or np.isinf(num_val) else num_val
            except (ValueError, TypeError):
                logger.debug(f"Cannot convert value '{value_to_parse}' (from key '{original_key_for_log}') to float. Parsed part: '{numeric_part_str}', multiplier {multiplier}.")
                return None
        exact_value = report_data_json.get(key_chinese_name)
        parsed_exact_value = _parse_value(exact_value, key_chinese_name)
        if parsed_exact_value is not None:
            logger.debug(f"For '{key_chinese_name}' found exact match, original: '{exact_value}', parsed: {parsed_exact_value}")
            return parsed_exact_value
        else:
            logger.debug(f"No usable value from exact match '{key_chinese_name}' (original: '{exact_value}'). Trying fuzzy...")
        for json_actual_key, json_actual_value in report_data_json.items():
            if isinstance(json_actual_key, str) and key_chinese_name in json_actual_key:
                if json_actual_key == key_chinese_name and exact_value is not None: continue
                parsed_like_value = _parse_value(json_actual_value, json_actual_key)
                if parsed_like_value is not None:
                    logger.debug(f"Fuzzy match: JSON key '{json_actual_key}' contains target '{key_chinese_name}'. Original: '{json_actual_value}', parsed: {parsed_like_value}")
                    return parsed_like_value
        logger.debug(f"No usable value found for '{key_chinese_name}' by exact or fuzzy match.")
        return None


    def _get_total_shares(self, symbol: str, trade_date: date) -> Optional[float]:
        """
        获取指定股票代码的最新总股本。
        数据源: stock_share_details 表。
        trade_date 参数目前未使用，因为我们总是获取最新的股本信息。
        如果需要历史股本，StockShareDetail 表和此逻辑需要相应调整。
        """
        logger.debug(f"[{symbol}] 正在从 StockShareDetail 表获取总股本 (忽略日期 {trade_date})...")
        share_detail_record = crud.get_stock_share_detail(self.db_session, symbol=symbol)

        if share_detail_record and share_detail_record.total_shares is not None:
            total_shares_val = float(share_detail_record.total_shares) # 确保是 float
            if total_shares_val > 1e-9: # 股本数应为正
                logger.info(f"[{symbol}] 从数据库获取到总股本: {total_shares_val} 股 (数据源日期: {share_detail_record.data_source_date})")
                return total_shares_val
            else:
                logger.warning(f"[{symbol}] 从数据库获取的总股本非正数或过小: {total_shares_val}。")
        else:
            if share_detail_record and share_detail_record.total_shares is None:
                 logger.warning(f"[{symbol}] StockShareDetail 表中记录存在，但 total_shares 为空。")
            else:
                 logger.warning(f"[{symbol}] 未能在 StockShareDetail 表中找到总股本数据。")
        return None

    # _fetch_financial_reports 方法保持不变 (来自您提供的版本)
    def _fetch_financial_reports(self, symbol: str, signal_date: date, report_type_db_key: str = 'benefit') -> Dict[
        str, Any]:
        # ... (您提供的 _fetch_financial_reports 的完整实现) ...
        fetched_reports_json_data = {
            'latest_annual_for_pe': None, 'latest_any_for_others': None,
            'prev_year_same_q_for_yoy': None, 'annual_reports_for_3y_check': []
        }
        try:
            reports_from_crud_models = crud.get_financial_reports_for_analyzer(
                db=self.db_session, symbol=symbol,
                report_type_db_key=report_type_db_key, current_signal_date=signal_date
            )
            latest_annual_sf_model = reports_from_crud_models.get('latest_annual_report')
            if latest_annual_sf_model and hasattr(latest_annual_sf_model, 'data'):
                fetched_reports_json_data['latest_annual_for_pe'] = latest_annual_sf_model.data
            latest_quarterly_sf_model = reports_from_crud_models.get('latest_quarterly_report')
            if latest_quarterly_sf_model and hasattr(latest_quarterly_sf_model, 'data'):
                fetched_reports_json_data['latest_any_for_others'] = latest_quarterly_sf_model.data
            prev_year_sf_model = reports_from_crud_models.get('previous_year_same_quarter_report')
            if prev_year_sf_model and hasattr(prev_year_sf_model, 'data'):
                fetched_reports_json_data['prev_year_same_q_for_yoy'] = prev_year_sf_model.data
            last_3_annual_sf_models_list = reports_from_crud_models.get('last_3_annual_reports', [])
            for sf_model in last_3_annual_sf_models_list:
                if sf_model and hasattr(sf_model, 'data'):
                    fetched_reports_json_data['annual_reports_for_3y_check'].append(sf_model.data)
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] CRUD ({crud.get_financial_reports_for_analyzer.__name__}) for '{report_type_db_key}' summary:")
            logger.debug(f"  Latest Annual: {'Found' if fetched_reports_json_data['latest_annual_for_pe'] else 'Not Found'}")
            logger.debug(f"  Latest Any: {'Found' if fetched_reports_json_data['latest_any_for_others'] else 'Not Found'}")
            logger.debug(f"  Prev Year Same Q: {'Found' if fetched_reports_json_data['prev_year_same_q_for_yoy'] else 'Not Found'}")
            logger.debug(f"  Past 3Y Annuals Count: {len(fetched_reports_json_data['annual_reports_for_3y_check'])}")
        except AttributeError as e:
            logger.error(f"AttrError processing reports from crud ({symbol} @ {signal_date}, type {report_type_db_key}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error calling crud.get_financial_reports_for_analyzer ({symbol} @ {signal_date}, type {report_type_db_key}): {e}", exc_info=True)
        return fetched_reports_json_data

    def _calculate_pb_ratio(self, symbol: str, market_cap: Optional[float],
                            balance_sheet_financial_datas: Dict[str, Any],
                            signal_date: date) -> Optional[float]:
        """计算市净率 (PB Ratio)"""
        pb_ratio = np.nan
        if pd.isna(market_cap) or market_cap is None or market_cap <= 1e-9:
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] PB计算跳过：市值无效 ({market_cap})。")
            return np.nan

        # 净资产通常来自最新的资产负债表 (可能是年报或季报)
        # 我们使用 _fetch_financial_reports 获取的 'latest_any_for_others'，
        # 假设当 report_type_db_key 指向资产负债表时，这个字段会填充最新的资产负债表数据。
        latest_balance_sheet_json = balance_sheet_financial_datas.get('latest_any_for_others')

        if not latest_balance_sheet_json:
            logger.warning(f"[{symbol}@{signal_date.isoformat()}] PB计算失败：未找到最新的资产负债表数据。")
            return np.nan

        net_assets = self._safe_extract_json_value(latest_balance_sheet_json, EQUITY_FIELD)

        if net_assets is not None and net_assets > 1e-9: # 净资产应为正数才有意义
            pb_ratio = market_cap / net_assets
            logger.info(f"[{symbol}@{signal_date.isoformat()}] PB 计算: 市值={market_cap:.2f}, 净资产={net_assets:.2f}, PB={pb_ratio:.4f}")
            return pb_ratio
        else:
            logger.warning(f"[{symbol}@{signal_date.isoformat()}] PB计算失败：净资产无效或非正 ({net_assets})。")
            if net_assets is not None and net_assets <= 0 and market_cap > 0 : # 市值存在但净资产为0或负
                 return np.inf if net_assets == 0 else pb_ratio # 如果净资产为0，PB为无穷大；如果为负，PB为负
            return np.nan


    def analyze_stock(self, symbol: str, signal_date: date) -> Dict[str, Any]:
        """
        Performs fundamental analysis for a stock on a given signal date.
        Price for market cap calculation is fetched internally using signal_date.
        """
        results = {
            'net_profit_positive_3y_latest': None, 'pe': np.nan, 'pe_lt_30': None,
            'pb': np.nan, 'market_cap': np.nan, # 初始化 pb 和 market_cap
            'revenue_growth_yoy': np.nan, 'profit_growth_yoy': np.nan, 'growth_positive': None,
            'peg_like_ratio': np.nan, 'peg_like_lt_1': None, 'error_reason': None
        }
        error_messages = []

        # --- 修改点 2: 在方法内部获取当前价格 ---
        current_price: Optional[float] = None
        # 假设 StockDaily 模型中使用 'symbol' 作为股票代码字段
        # 并且 crud.get_stock_daily_for_date 的股票代码参数名为 'symbol'
        daily_data_for_signal_date = crud.get_stock_daily_for_date(
            self.db_session,
            symbol=symbol, # 传递给 crud 函数的股票代码参数名
            trade_date=signal_date
        )
        if daily_data_for_signal_date and daily_data_for_signal_date.close is not None: # 使用 .close (根据您的模型)
            current_price = daily_data_for_signal_date.close
            logger.info(f"[{symbol}@{signal_date.isoformat()}] 获取到信号日价格: {current_price:.2f}")
        else:
            # 尝试获取信号日之前最近一个交易日的价格作为备用
            logger.warning(f"[{symbol}@{signal_date.isoformat()}] 未找到信号日当天价格，尝试回溯查找...")
            temp_date = signal_date - timedelta(days=1)
            for _ in range(7): # 最多回溯7天
                prev_daily_data = crud.get_stock_daily_for_date(self.db_session, symbol=symbol, trade_date=temp_date)
                if prev_daily_data and prev_daily_data.close is not None:
                    current_price = prev_daily_data.close
                    logger.warning(f"[{symbol}@{signal_date.isoformat()}] 使用回溯日期 {temp_date} 的价格: {current_price:.2f}")
                    break
                temp_date -= timedelta(days=1)
            if current_price is None:
                error_messages.append(f"信号日 {signal_date} 及之前7天内均无法获取有效价格.")
                logger.error(f"[{symbol}@{signal_date.isoformat()}] 无法获取有效价格用于分析。")
        # --- 价格获取结束 ---

        # 1. 获取总股本并计算市值
        total_shares = self._get_total_shares(symbol, signal_date)
        market_cap = np.nan
        if total_shares is not None and current_price is not None and pd.notna(current_price):
            if total_shares > 1e-9:
                market_cap = total_shares * current_price
            else:
                error_messages.append(f"总股本计算为非正数或过小 ({total_shares}).")
        else:
            if total_shares is None: error_messages.append("总股本无法从数据库获取或无效.")
            if current_price is None or pd.isna(current_price): error_messages.append("当前价格未提供或无效，无法计算市值.")

        if pd.notna(market_cap):
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] 市值计算: {market_cap:.2f} (总股本: {total_shares:.0f}, 价格: {current_price:.2f})")
        else:
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] 市值无法计算。总股本: {total_shares}, 价格: {current_price}")

        # 2. 获取财务报告数据 (利润表相关)
        # report_type_db_key 应与您在 StockFinancial 表中存储利润表时使用的 report_type 值对应
        # 常见的可能是 '利润表' 或 'income_statement'，这里使用您提供的 'benefit'
        income_statement_datas = self._fetch_financial_reports(symbol, signal_date,
                                                               report_type_db_key='benefit')
        latest_annual_income_json = income_statement_datas['latest_annual_for_pe']
        latest_any_income_json = income_statement_datas['latest_any_for_others'] # 用于同比增长的当期利润表
        prev_year_same_q_income_json = income_statement_datas['prev_year_same_q_for_yoy'] # 用于同比增长的去年同期利润表
        annual_reports_for_3y_list = income_statement_datas['annual_reports_for_3y_check']


        # 2.1 获取财务报告数据 (资产负债表相关 - 为 PB 计算)
        # report_type_db_key 应与您在 StockFinancial 表中存储资产负债表时使用的 report_type 值对应
        # 常见的可能是 '资产负债表' 或 'balance_sheet'。您需要确认这个键名。
        # 假设键名为 'balance_sheet' (如果不同，请修改)
        balance_sheet_datas = self._fetch_financial_reports(symbol, signal_date,
                                                            report_type_db_key='debt') # <--- 假设的资产负债表类型键

        # 3. 计算 PE
        if pd.notna(market_cap) and latest_annual_income_json:
            net_profit_for_pe = self._safe_extract_json_value(latest_annual_income_json, NET_PROFIT_FIELD)
            if net_profit_for_pe is not None and net_profit_for_pe > 1e-6: # 净利润应为正
                results['pe'] = market_cap / net_profit_for_pe
                results['pe_lt_30'] = results['pe'] < PE_THRESHOLD
                logger.debug(f"[{symbol}@{signal_date.isoformat()}] PE 计算: {results['pe']:.4f} (市值: {market_cap:.2f}, 年净利润: {net_profit_for_pe:.2f})")
            else:
                error_messages.append(f"PE计算失败：年度净利润无效或非正 ({net_profit_for_pe}).")
        else:
            if pd.isna(market_cap): error_messages.append("PE计算失败：市值是 NaN.")
            if not latest_annual_income_json: error_messages.append("PE计算失败：未找到最新的年度利润表.")

        # 3.1 计算 PB
        # 使用上面获取的 balance_sheet_datas
        pb_value = self._calculate_pb_ratio(symbol, market_cap, balance_sheet_datas, signal_date)
        results['pb'] = pb_value # pb_value 可能是 np.nan


        # 4. 计算同比增长率 (营收、利润) - 使用利润表数据
        rev_growth = np.nan
        prof_growth = np.nan
        # latest_any_income_json 和 prev_year_same_q_income_json 来自上面对 'benefit' 的调用
        if latest_any_income_json and prev_year_same_q_income_json:
            current_revenue = self._safe_extract_json_value(latest_any_income_json, REVENUE_FIELD)
            prev_revenue = self._safe_extract_json_value(prev_year_same_q_income_json, REVENUE_FIELD)
            current_profit = self._safe_extract_json_value(latest_any_income_json, NET_PROFIT_FIELD)
            prev_profit = self._safe_extract_json_value(prev_year_same_q_income_json, NET_PROFIT_FIELD)

            logger.debug(f"[{symbol}@{signal_date.isoformat()}] YoY 数据: CurrRev={current_revenue}, PrevRev={prev_revenue}, CurrProf={current_profit}, PrevProf={prev_profit}")

            if current_revenue is not None and prev_revenue is not None:
                if abs(prev_revenue) > 1e-9:
                    rev_growth = (current_revenue - prev_revenue) / abs(prev_revenue)
                    results['revenue_growth_yoy'] = rev_growth
                else: error_messages.append("营收同比增长计算失败：上一期营收接近零.")
            else: error_messages.append("营收同比增长计算失败：缺少当前或上一期营收数据.")

            if current_profit is not None and prev_profit is not None:
                if abs(prev_profit) > 1e-9:
                    prof_growth = (current_profit - prev_profit) / abs(prev_profit)
                    results['profit_growth_yoy'] = prof_growth
                elif current_profit > 1e-9 and (prev_profit is None or abs(prev_profit) < 1e-9):
                    prof_growth = np.inf
                    results['profit_growth_yoy'] = prof_growth
                else: error_messages.append(f"利润同比增长计算失败：无效的利润数据 (当前:{current_profit}, 上一期:{prev_profit}).")
            else: error_messages.append("利润同比增长计算失败：缺少当前或上一期利润数据.")

            if pd.notna(rev_growth) and pd.notna(prof_growth):
                if np.isinf(prof_growth) and prof_growth > 0:
                    results['growth_positive'] = (rev_growth > 1e-6)
                elif not np.isinf(prof_growth):
                    results['growth_positive'] = (rev_growth > 1e-6) and (prof_growth > 1e-6)
            if results['growth_positive'] is not None:
                logger.debug(f"[{symbol}@{signal_date.isoformat()}] Growth Positive Check: RevGrowth={rev_growth:.4f}, ProfGrowth={prof_growth:.4f}, Result={results['growth_positive']}")
        else:
            error_messages.append("同比增长率计算失败：缺少当期利润表或去年同期利润表.")


        # 5. 计算 PEG-like Ratio
        current_pe_val = results.get('pe')
        finite_prof_growth_for_peg = prof_growth if pd.notna(prof_growth) and np.isfinite(prof_growth) else np.nan
        if pd.notna(current_pe_val) and pd.notna(finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6:
            peg_like = current_pe_val / (finite_prof_growth_for_peg * 100.0)
            results['peg_like_ratio'] = peg_like
            results['peg_like_lt_1'] = peg_like < PEG_LIKE_THRESHOLD
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] PEG-like 计算: {peg_like:.4f} (PE: {current_pe_val:.4f}, 利润增长率: {finite_prof_growth_for_peg:.4f})")
        else:
            if pd.isna(current_pe_val): error_messages.append("PEG近似值计算失败：PE 是 NaN.")
            if not (pd.notna(finite_prof_growth_for_peg) and finite_prof_growth_for_peg > 1e-6):
                error_messages.append(f"PEG近似值计算失败：利润增长率无效或非正 ({finite_prof_growth_for_peg}).")

        # 6. 计算最近3年净利润是否为正 (Net Profit Positive 3 Years Latest)
        # 使用 latest_any_income_json (最新的任何类型利润表) 和 annual_reports_for_3y_list (年利润表列表)
        num_annual_reports_for_check = len(annual_reports_for_3y_list)
        all_checked_profits_positive = True

        if latest_any_income_json: # 检查最新一期财报（可能是季报）的利润
            latest_profit = self._safe_extract_json_value(latest_any_income_json, NET_PROFIT_FIELD)
            if latest_profit is None or latest_profit <= 1e-6:
                all_checked_profits_positive = False
                logger.debug(f"[{symbol}@{signal_date.isoformat()}] 3年盈利检查：最新报告期利润非正 ({latest_profit}).")
        else:
            all_checked_profits_positive = False
            error_messages.append("3年盈利检查失败：缺少最新的利润表数据.")
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] 3年盈利检查：缺少最新利润表数据。")

        if all_checked_profits_positive: # 如果最新一期利润为正，再检查过去几年的年报
            if num_annual_reports_for_check < 3: # 严格要求3份年报
                results['net_profit_positive_3y_latest'] = None # 数据不足，无法判断
                error_messages.append(f"3年盈利检查存疑：年度利润表数量不足 ({num_annual_reports_for_check}/3).")
            else:
                annual_profits_log = []
                for i in range(min(3, num_annual_reports_for_check)):
                    annual_report_json_data = annual_reports_for_3y_list[i]
                    annual_profit = self._safe_extract_json_value(annual_report_json_data, NET_PROFIT_FIELD)
                    annual_profits_log.append(annual_profit)
                    if annual_profit is None or annual_profit <= 1e-6:
                        all_checked_profits_positive = False
                        break
                results['net_profit_positive_3y_latest'] = all_checked_profits_positive
                logger.debug(f"[{symbol}@{signal_date.isoformat()}] 3年盈利检查：年利润检查: {annual_profits_log}. 结果: {results['net_profit_positive_3y_latest']}")
        else: # 如果最新一期利润已非正或缺失
            results['net_profit_positive_3y_latest'] = False


        if error_messages:
            results['error_reason'] = "; ".join(sorted(list(set(error_messages))))
            logger.debug(f"[{symbol}@{signal_date.isoformat()}] 基本面分析错误: {results['error_reason']}")

        results['market_cap'] = market_cap  # 确保在任何可能的分支后，results['market_cap'] 都被赋予计算出的 market_cap 值

        # Round relevant float results
        for key in ['pe', 'pb', 'market_cap', 'revenue_growth_yoy', 'profit_growth_yoy',
                    'peg_like_ratio']:  # <-- *** 修正点：加入 'market_cap' ***
            if key in results and pd.notna(results[key]):  # pd.notna 会正确处理 np.nan
                if np.isinf(results[key]):
                    results[key] = str(results[key])  # 将 Inf 转为字符串 "inf" 或 "-inf"
                elif np.isfinite(results[key]):  # 确保是有限的浮点数才进行 round
                    results[key] = round(results[key], 4)  # 对 market_cap 也保留4位小数，或按需调整
                # else NaN (pd.notna(results[key]) is False) remains np.nan, NpAndDateEncoder 会处理为 null
        return results


# if __name__ == '__main__': 测试代码块保持不变 (来自您提供的版本)
# ... (您的 __main__ 测试代码块) ...
if __name__ == '__main__':
    import logging
    import json
    # from datetime import date, datetime, timedelta # 确保 timedelta 也导入了 (已在顶部导入)
    # import numpy as np # 已在顶部导入
    # import pandas as pd # 已在顶部导入

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger(__name__) # 为 main 代码块定义 logger

    from db.database import get_db_session # 确保路径正确
    # FundamentalAnalyzer 内部会导入 db.crud

    logger_main.info("--- 开始 FundamentalAnalyzer 针对特定股票的真实数据测试 (使用 get_stock_daily_data_period 获取价格) ---")

    db_sess = None
    try:
        with get_db_session() as session:
            if session is None:
                logger_main.error("未能获取数据库会话，测试中止。请检查数据库配置和连接。")
                exit(1)
            db_sess = session

            analyzer = FundamentalAnalyzer(db_session=db_sess)

            stock_to_test = "000001"
            test_signal_date = date(2025, 5, 30)
            test_current_price = None

            start_date_for_price_fetch = test_signal_date - timedelta(days=7)
            logger_main.info(f"为股票 {stock_to_test} 从 {start_date_for_price_fetch} 到 {test_signal_date} 获取日线数据以确定当前价格...")

            daily_data_df = crud.get_stock_daily_data_period(
                db=db_sess,
                symbol=stock_to_test,
                start_date=start_date_for_price_fetch,
                end_date=test_signal_date
            )

            if daily_data_df is not None and not daily_data_df.empty:
                if not pd.api.types.is_datetime64_any_dtype(daily_data_df['date']):
                    try:
                        daily_data_df['date'] = pd.to_datetime(daily_data_df['date']).dt.date
                    except Exception as e_date_conv:
                        logger_main.error(f"DataFrame 中的 'date' 列无法转换为日期对象: {e_date_conv}")
                        daily_data_df = None

                if daily_data_df is not None:
                    signal_day_data = daily_data_df[daily_data_df['date'] == test_signal_date]
                    if not signal_day_data.empty:
                        if 'close' in signal_day_data.columns: # 使用 'close'
                            test_current_price = signal_day_data['close'].iloc[0]
                            logger_main.info(f"在 {test_signal_date} 找到收盘价: {test_current_price}")
                        else:
                            logger_main.warning(f"DataFrame 中未找到 'close' 列。列名: {daily_data_df.columns.tolist()}")
                    else:
                        daily_data_df_sorted = daily_data_df.sort_values(by='date', ascending=False)
                        if not daily_data_df_sorted.empty:
                            if 'close' in daily_data_df_sorted.columns: # 使用 'close'
                                latest_available_date = daily_data_df_sorted['date'].iloc[0]
                                test_current_price = daily_data_df_sorted['close'].iloc[0]
                                logger_main.warning(f"信号日 {test_signal_date} 无数据，使用最近可用日期 {latest_available_date} 的收盘价: {test_current_price}")
                            else:
                                logger_main.warning(f"DataFrame 中未找到 'close' 列 (回溯查找时)。列名: {daily_data_df.columns.tolist()}")
                        else:
                            logger_main.warning(f"在 {start_date_for_price_fetch} 到 {test_signal_date} 期间未找到 {stock_to_test} 的任何日线数据。")
            else:
                logger_main.warning(f"未能从 crud.get_stock_daily_data_period 获取 {stock_to_test} 的日线数据。")


            if test_current_price is None:
                logger_main.error(f"未能为股票 {stock_to_test} 在日期 {test_signal_date} 附近确定测试价格，中止对该股票的 analyze_stock 测试。")
            else:
                logger_main.info(f"\n--- 测试 analyze_stock: 股票代码 {stock_to_test}, 日期 {test_signal_date}, 获取到的价格 {float(test_current_price):.2f} ---")
                analysis_results = analyzer.analyze_stock(
                    symbol=stock_to_test,
                    signal_date=test_signal_date,

                )

                class NpAndDateEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating):
                            if np.isnan(obj): return None
                            if np.isinf(obj): return str(obj)
                            return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        if isinstance(obj, (datetime, date)): return obj.isoformat()
                        if pd.isna(obj): return None
                        return super(NpAndDateEncoder, self).default(obj)

                results_str = json.dumps(analysis_results, indent=4, ensure_ascii=False, cls=NpAndDateEncoder)
                print("\n分析结果:")
                print(results_str)

                if analysis_results.get('error_reason'):
                    logger_main.warning(f"股票 {stock_to_test} 的错误原因: {analysis_results.get('error_reason')}")

    except Exception as e:
        logger_main.error(f"运行 FundamentalAnalyzer 测试时发生严重错误: {e}", exc_info=True)
    finally:
        logger_main.info("\n--- FundamentalAnalyzer 真实数据测试结束 ---")