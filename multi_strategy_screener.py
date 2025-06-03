# multi_strategy_screener.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta, time  # 确保 time 被导入
from sqlalchemy.orm import Session
from typing import List, Dict, Type, Optional, Any
import os

# --- 项目模块导入 ---

from config.settings import settings
from strategies.base_strategy import BaseStrategy, StrategyResult, StrategyContext
from strategies.ma_pullback_strategy import AdaptedMAPullbackStrategy
from strategies.monthly_ma_pullback_strategy import MonthlyMAPullbackStrategy
from strategies.multi_level_cross_refactored_strategy import RefactoredMultiLevelCrossStrategy
from analysis.fundamental_analyzer import FundamentalAnalyzer
from strategies.breakout_strategy import BreakoutStrategy
from db.database import SessionLocal
from db import crud
from db.models import StockList
from strategies.weekly_ma_pullback_strategy import WeeklyMAPullbackStrategy

# --- 日志配置 ---
# 假设日志已在外部配置，如 main.py。如果单独运行此脚本，请取消注释或配置。
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CSV 输出列顺序 (使用英文键名，与 fundamental_analyzer 返回结果对应) ---
CSV_COLUMN_ORDER = [
    'symbol',  # 股票代码 (来自 tech_signal)
    'stock_name',  # 股票简称 (来自 stock_info)
    'signal_date',  # 信号日期 (来自 tech_signal)
    'strategy_name',  # 信号来源 (来自 tech_signal)
    'signal_level',  # 信号级别 (由 screener 判断)
    # 基本面指标 (键名与 fundamental_analyzer 返回的字典一致)
    'market_cap_formatted',  # <--- 新增：格式化后的市值（带单位）
    'pe',
    'pb',
    'revenue_growth_yoy',
    'profit_growth_yoy',
    'peg_like_ratio',
    'net_profit_positive_3y_latest',
    'growth_positive',
    'pe_lt_30',
    'peg_like_lt_1',
    'tech_notes',  # 技术备注 (来自 tech_signal.notes)
    'error_reason'  # 基本面分析的错误原因 (来自 fundamental_analyzer)
    # 'market_cap' # 如果也想输出原始市值的数值，可以加这一列
]

# --- 筛选器配置 ---
CONFIG = {
    "data_lookback_days": 750,
    "output_dir": settings.REPORT_SAVE_PATH,  # 使用 settings 中的路径
    "strategies_to_run": [  # 策略类列表
        AdaptedMAPullbackStrategy,
        RefactoredMultiLevelCrossStrategy,
        BreakoutStrategy,
        WeeklyMAPullbackStrategy,
        MonthlyMAPullbackStrategy,
    ],
    "strategy_params": {  # 特定策略参数 (如果需要覆盖默认值)
        "AdaptedMAPullbackStrategy": {},
        "RefactoredMultiLevelCrossStrategy": {},
        "ChanBreakoutStrategy": {  # 这是一个示例，如果您的策略列表中没有它，可以移除
            'pullback_depth_pct': 0.2,
            'max_price_vs_3year_low_ratio': 2,
            'volume_ratio_threshold': 2.0,
        }
    }
}


# --- 辅助函数：格式化大数字为亿/万单位 ---
def format_large_number_to_unit_str(num: Optional[float], default_decimals: int = 2) -> str:
    if pd.isna(num) or num is None: return ''
    if np.isinf(num): return str(num)  # 'inf' 或 '-inf'
    num_abs = abs(num)
    sign = "-" if num < 0 else ""
    if num_abs >= 1_0000_0000:
        return f"{sign}{num_abs / 1_0000_0000:.{default_decimals}f}亿"
    if num_abs >= 1_0000:
        return f"{sign}{num_abs / 1_0000:.{default_decimals}f}万"
    return f"{sign}{num_abs:.{default_decimals}f}"


class MultiStrategyScreener:
    def __init__(self, db_session: Session, strategies_classes: List[Type[BaseStrategy]],
                 fundamental_analyzer: Optional[FundamentalAnalyzer] = None):
        self.db_session = db_session
        self.strategies_classes = strategies_classes  # 保存策略类
        self.fundamental_analyzer = fundamental_analyzer
        if self.fundamental_analyzer is None and FundamentalAnalyzer:
            logger.info("MultiStrategyScreener 初始化时未提供 fundamental_analyzer，将尝试自行实例化。")
            self.fundamental_analyzer = FundamentalAnalyzer(db_session)

    def _get_stock_list(self) -> List[Dict[str, Any]]:
        # crud.get_all_stocks 返回 List[StockList] (ORM对象列表)
        stocks_db_objects = crud.get_all_stocks(self.db_session)
        stock_list_for_processing = []
        for stock_obj in stocks_db_objects:
            if stock_obj and stock_obj.code and stock_obj.name:
                stock_list_for_processing.append({'code': stock_obj.code, 'name': stock_obj.name})
        return stock_list_for_processing

    def run_screening(self, analysis_date: date,
                      context_params: Optional[Dict] = None):  # context_params from caller is not used now
        logger.info(f"开始执行多策略筛选（技术信号优先），分析日期: {analysis_date.isoformat()}")
        # if context_params is None: context_params = {} # Not currently used for StrategyContext

        all_final_output_data = []
        stocks_to_process = self._get_stock_list()

        if not stocks_to_process:
            logger.warning("未能获取到股票列表，筛选中止。")
            return  # db_session will be closed in __main__ or by the caller

        total_stocks = len(stocks_to_process)
        logger.info(f"将对 {total_stocks} 只股票进行技术信号扫描。")

        for i, stock_info in enumerate(stocks_to_process):
            current_processing_symbol = stock_info['code']
            stock_name = stock_info['name']

            logger.debug(
                f"--- 扫描技术信号 ({i + 1}/{total_stocks}): {current_processing_symbol} ({stock_name}) ---")

            # 1. Load data for strategies
            data_for_strategies: Dict[str, pd.DataFrame] = {}
            lookback_days = CONFIG.get("data_lookback_days", 750)
            lookback_start_date = analysis_date - timedelta(days=lookback_days)

            daily_data_df = crud.get_stock_daily_data_period(
                self.db_session, symbol=current_processing_symbol,
                start_date=lookback_start_date, end_date=analysis_date
            )
            if daily_data_df is not None and not daily_data_df.empty:
                data_for_strategies["daily"] = daily_data_df
            else:
                logger.warning(f"股票 {current_processing_symbol} 无日线数据。跳过此股票的技术策略执行。")
                continue

            weekly_data_df = crud.get_stock_weekly_data_period(
                self.db_session, symbol=current_processing_symbol,
                start_date=lookback_start_date, end_date=analysis_date
            )
            if weekly_data_df is not None and not weekly_data_df.empty:
                data_for_strategies["weekly"] = weekly_data_df
            else:
                logger.info(f"股票 {current_processing_symbol} 在指定期间无周线数据。")

            monthly_data_df = crud.get_stock_monthly_data_period(
                self.db_session, symbol=current_processing_symbol,
                start_date=lookback_start_date, end_date=analysis_date
            )
            if monthly_data_df is not None and not monthly_data_df.empty:
                data_for_strategies["monthly"] = monthly_data_df
            else:
                logger.info(f"股票 {current_processing_symbol} 在指定期间无月线数据。")

            # 2. Execute technical strategies
            generated_technical_signals_on_analysis_date: List[StrategyResult] = []
            for StrategyClass in self.strategies_classes:
                # ***MODIFICATION START***
                # Create StrategyContext first
                strategy_specific_params = CONFIG.get("strategy_params", {}).get(StrategyClass.__name__, {})
                strategy_context_obj = StrategyContext(
                    db_session=self.db_session,  # Pass the db_session from MultiStrategyScreener
                    current_date=analysis_date,
                    strategy_params=strategy_specific_params
                )
                # Instantiate strategy by passing the context object
                strategy_instance = StrategyClass(context=strategy_context_obj)
                # ***MODIFICATION END***

                try:
                    logger.info(f"  执行策略: {strategy_instance.strategy_name} for {current_processing_symbol}")
                    signals = strategy_instance.run_for_stock(
                        symbol=current_processing_symbol,
                        current_date=analysis_date,  # All strategies now accept current_date
                        data=data_for_strategies
                    )
                    if signals:
                        for sig in signals:
                            if sig.signal_date == analysis_date and sig.signal_type != "NO_SIGNAL":
                                generated_technical_signals_on_analysis_date.append(sig)
                except Exception as e_strat:
                    logger.error(
                        f"策略 {StrategyClass.__name__} 在股票 {current_processing_symbol} 上执行失败: {e_strat}",
                        exc_info=True)

            if not generated_technical_signals_on_analysis_date:
                logger.info(
                    f"股票 {current_processing_symbol} ({stock_name}) 在 {analysis_date.isoformat()} 无有效技术信号。")
                continue

            # 3. Fundamental Analysis if technical signals exist
            logger.info(f"股票 {current_processing_symbol} 发现技术信号，准备进行基本面分析...")
            fundamental_metrics: Dict[str, Any] = {}
            if self.fundamental_analyzer:
                try:
                    fundamental_metrics = self.fundamental_analyzer.analyze_stock(
                        symbol=current_processing_symbol,
                        signal_date=analysis_date
                    )
                except Exception as e_fund:
                    logger.error(f"对股票 {current_processing_symbol} 进行基本面分析时出错: {e_fund}",
                                 exc_info=True)
                    fundamental_metrics = {'error_reason': f"基本面分析失败: {str(e_fund)}"}

            # 4. Combine and store results
            for tech_signal in generated_technical_signals_on_analysis_date:
                signal_level = "Daily"  # Default
                if tech_signal.strategy_name == "WeeklyMAPullbackStrategy":
                    signal_level = "Weekly"
                elif tech_signal.strategy_name == "MonthlyMAPullbackStrategy":
                    signal_level = "Monthly"

                output_row = {
                    'symbol': tech_signal.symbol,
                    'stock_name': stock_name,
                    'signal_date': tech_signal.signal_date.isoformat(),
                    'strategy_name': tech_signal.strategy_name,
                    'signal_level': signal_level,
                    'tech_notes': tech_signal.notes if tech_signal.notes and tech_signal.notes != "No specific notes" else None,
                }
                for key_in_csv_order in CSV_COLUMN_ORDER:
                    if key_in_csv_order in fundamental_metrics:
                        output_row[key_in_csv_order] = fundamental_metrics[key_in_csv_order]
                    elif key_in_csv_order == 'market_cap_formatted':
                        output_row[key_in_csv_order] = fundamental_metrics.get('market_cap')

                fm_error = fundamental_metrics.get('error_reason')
                if fm_error:
                    output_row['error_reason'] = f"{output_row.get('error_reason', '') or ''}; {fm_error}".strip('; ')
                if not output_row.get('error_reason'): output_row['error_reason'] = None

                all_final_output_data.append(output_row)

        # self.db_session.close() # Session should be closed by the caller (__main__)

        if not all_final_output_data:
            logger.info("没有生成任何满足技术信号的最终结果。")
            return  # db_session will be closed by the caller

        final_df = pd.DataFrame(all_final_output_data)
        for col_name in CSV_COLUMN_ORDER:
            if col_name not in final_df.columns:
                final_df[col_name] = np.nan
        final_df = final_df[CSV_COLUMN_ORDER]

        if 'market_cap_formatted' in final_df.columns:
            final_df['market_cap_formatted'] = final_df['market_cap_formatted'].apply(
                lambda x: format_large_number_to_unit_str(x, default_decimals=2)
            )

        float_ratio_cols_in_df = ['pe', 'pb', 'revenue_growth_yoy',
                                  'profit_growth_yoy', 'peg_like_ratio']
        for col in float_ratio_cols_in_df:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) and isinstance(x, (float, int))
                    else ('Infinity' if x == np.inf else (
                        '-Infinity' if x == -np.inf else (str(x) if pd.notna(x) else '')))
                )

        bool_cols_in_df = [
            'net_profit_positive_3y_latest', 'growth_positive',
            'pe_lt_30', 'peg_like_lt_1'
        ]
        for col in bool_cols_in_df:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(
                    lambda x: "是" if x is True else (
                        "否" if x is False else ("不适用" if pd.isna(x) or x is None else str(x)))
                )

        output_filename = f"signals_selected_{analysis_date.isoformat()}.csv"
        output_dir_path = settings.REPORT_SAVE_PATH

        if not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path, exist_ok=True)
            except OSError as e_dir:
                logger.error(f"创建输出目录 {output_dir_path} 失败: {e_dir}"); return
        output_path = os.path.join(output_dir_path, output_filename)
        try:
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"最终选股结果已保存到: {output_path}")
        except Exception as e_csv:
            logger.error(f"保存CSV文件失败 {output_path}: {e_csv}", exc_info=True)

# (get_analysis_date 函数保持不变)
def get_analysis_date() -> date:
    now = datetime.now()
    today = now.date()
    trading_close_time = time(16, 0)
    if today.weekday() >= 5:
        return today - timedelta(days=today.weekday() - 4)
    else:
        if now.time() >= trading_close_time:
            return today
        else:
            return today - timedelta(days=1 if today.weekday() != 0 else 3)


if __name__ == "__main__":
    logger.info("开始执行 MultiStrategyScreener 独立测试...")
    db_sess_main: Optional[Session] = None
    try:
        db_sess_main = SessionLocal()
        if db_sess_main is None:
            logger.error("无法获取数据库会话，测试中止。")
            exit(1)

        active_strategies_classes = CONFIG.get("strategies_to_run", [])
        if not active_strategies_classes:
            logger.error("CONFIG 中未配置任何策略 (strategies_to_run 为空)。")
            if db_sess_main: db_sess_main.close()
            exit(1)

        # *** 修改点：通过实例调用 run_screening ***
        screener_instance = MultiStrategyScreener(
            db_session=db_sess_main,
            strategies_classes=active_strategies_classes  # 传递策略类列表
            # fundamental_analyzer 会在 __init__ 中自动创建
        )

        analysis_target_date = get_analysis_date()
        # analysis_target_date = date(2025, 5, 30) # 固定日期测试

        screener_instance.run_screening(analysis_date=analysis_target_date)  # *** 调用实例方法 ***

    except Exception as e_main_run:
        logger.error(f"运行筛选器时发生严重错误: {e_main_run}", exc_info=True)
    finally:
        if db_sess_main:
            db_sess_main.close()  # 在 MultiStrategyScreener 内部已关闭，这里是 __main__ 级别的会话
            logger.info("主脚本数据库会话已关闭。MultiStrategyScreener 独立测试结束。")