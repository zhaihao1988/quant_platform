# multi_strategy_screener.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from typing import List, Dict, Type, Optional

# --- 项目模块导入 ---
# 假设的更高级的 BaseStrategy, StrategyResult, StrategyContext
# 您需要确保这个 BaseStrategy 文件在您的项目中是这样的，或者更新它
from strategies.base_strategy import BaseStrategy, StrategyResult, StrategyContext

# 导入具体的策略类
# (确保这些策略文件在 quant_platform/strategies/ 目录下，并且类名正确)
from strategies.ma_pullback_strategy import AdaptedMAPullbackStrategy
from strategies.multi_level_cross_refactored_strategy import RefactoredMultiLevelCrossStrategy

# 导入基本面分析器
from analysis.fundamental_analyzer import FundamentalAnalyzer
from strategies.breakout_strategy import BreakoutStrategy
# 数据库相关
from db.database import SessionLocal # 假设 SessionLocal 和 init_db
from db import crud
from db.models import StockList  # 用于获取股票代码和名称



# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CSV 输出列顺序 ---
# 与用户在 multi_level_cross_strategy_new.py 中定义的保持一致
CSV_COLUMN_ORDER = [
    '股票代码', '股票简称', '信号日期', '信号来源', '信号级别',
    'net_profit_positive_3y_latest', 'pe', 'pe_lt_30',
    'revenue_growth_yoy', 'profit_growth_yoy', 'growth_positive',
    'peg_like_ratio', 'peg_like_lt_1', 'error_reason'
]
# 可以加入策略本身返回的 'details' 中的一些关键信息，如果需要的话
# 例如：'reference_high_price', 'current_close_tech' (技术信号时的收盘价)

# --- 筛选器配置 ---
# 您可以根据需要调整这些配置
CONFIG = {
    "data_lookback_days": 750,  # 为策略加载大约3年的日线数据 (250 * 3)
    "weekly_resample_lookback_factor": 5,  # 用于周线MA的额外因子 (针对MA回调策略)
    "monthly_resample_lookback_factor": 20,  # 用于月线MA的额外因子 (针对MA回调策略)
    "output_dir": "output",  # CSV输出目录
    "strategies_to_run": [
        AdaptedMAPullbackStrategy,
        RefactoredMultiLevelCrossStrategy,
        BreakoutStrategy,
    ],
    "strategy_params": {  # 可选：为特定策略传递参数
        "AdaptedMAPullbackStrategy": {
            # 'ma_short': 5, # 如果要覆盖策略中的默认值
        },
        "RefactoredMultiLevelCrossStrategy": {
            # "ma_list_map": { "daily": [5,10,20,60], ... } # 覆盖默认MA列表
        },
        "ChanBreakoutStrategy": { # <--- 为新策略配置参数 (可选)
            'pullback_depth_pct': 0.2,
            'max_price_vs_3year_low_ratio': 2,
            'volume_ratio_threshold': 2.0,
            # 'pattern_identification_lookback_days': 250, # 默认250天
        }
    }
}


def get_analysis_date() -> date:
    """
    确定用于分析的日期。
    通常是当前日期前的一个交易日，或者用户指定的日期。
    对于周日运行的脚本，可能是上周五。
    """
    today = date.today()  # 当前日期是 2025-05-11 (周日)
    # 如果今天是周日，分析上周五 (T-2)
    if today.weekday() == 6:  # 周日
        return today - timedelta(days=2)
    # 如果今天是周一，分析上周五 (T-3)
    elif today.weekday() == 0:  # 周一
        return today - timedelta(days=3)
    # 其他工作日，分析前一天 (T-1)
    else:
        return today - timedelta(days=1)


def run_screener(analysis_date_str: Optional[str] = None):
    """
    执行多策略选股流程。
    """
    if analysis_date_str:
        try:
            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"提供的分析日期格式无效: {analysis_date_str}. 请使用 YYYY-MM-DD 格式。")
            return
    else:
        analysis_date = get_analysis_date()

    logger.info(f"========== 多策略选股器开始运行 - 分析日期: {analysis_date.isoformat()} ==========")

    db: Session = SessionLocal()  # type: ignore
    # 确保数据库已初始化 (如果需要创建表等)
    # init_db() # 通常在项目首次运行时执行，或由Alembic管理

    # 1. 获取股票列表
    logger.info("步骤 1: 获取股票列表...")
    try:
        stock_entities = crud.get_all_stocks(db)  # 假设 crud 中有 get_all_stocks() -> List[StockList]
        if not stock_entities:
            logger.warning("数据库中未找到股票列表 (stock_list 表为空或查询失败)。")
            return
        # stock_codes_to_scan = [s.code for s in stock_entities if s.code.startswith(('0','3','6'))] # 示例：只扫描A股主板/创业板/科创板
        stock_codes_to_scan = {s.code: s.name for s in stock_entities}  # {代码: 名称}
        logger.info(f"将扫描 {len(stock_codes_to_scan)} 只股票。")
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}", exc_info=True)
        db.close()
        return

    # 2. 初始化数据加载器和基本面分析器
    # DataLoader的初始化可能需要配置，这里假设一个默认的或从context获取
    # 对于这个脚本，我们可能直接调用其静态方法或实例化一个简单的版本
    # 这里简化为在循环中按需加载，但更优化的方式是预加载或使用更高级的DataLoader
    try:
        # 假设 DataLoader 可以直接实例化或其加载函数是静态的/可直接调用
        # 如果 DataLoader 需要 db_session 或 engine，请确保传递
        # data_loader = DataLoader(db_session=db) # 或 DataLoader(engine=engine_instance)
        # 这里我们先不在外部实例化，策略内部或加载函数会处理
        pass
    except Exception as e:
        logger.error(f"初始化数据加载器失败: {e}", exc_info=True)
        db.close()
        return

    fundamental_analyzer = FundamentalAnalyzer(db_session=db)

    all_final_signals_data = []
    processed_stocks_count = 0

    # 3. 遍历股票列表，执行策略和分析
    for stock_code, stock_name in stock_codes_to_scan.items():
        processed_stocks_count += 1
        logger.info(
            f"--- 开始处理股票 ({processed_stocks_count}/{len(stock_codes_to_scan)}): {stock_code} ({stock_name}) ---")

        # a. 加载数据 (日线、周线、月线)
        #    策略需要足够的回溯期来计算MA和识别模式
        #    DataLoader应能提供一个包含 'daily', 'weekly', 'monthly' DataFrame 的字典
        #    每个DataFrame的index应为日期，列包含 open, high, low, close, volume
        try:
            # 假设 crud 中有 load_stock_data_for_strategy 返回所需格式
            # 或者 DataLoader 有类似方法
            # data_start_date = analysis_date - timedelta(days=CONFIG["data_lookback_days"])
            # raw_data_dict = data_loader.load_data_for_stock(
            #     stock_code,
            #     start_date=data_start_date, # 开始日期
            #     end_date=analysis_date,     # 结束日期 (分析日)
            #     timeframes=['daily', 'weekly', 'monthly'] # 需要的时间级别
            # )

            # 模拟数据加载 - 您需要替换为实际的 DataLoader 调用
            # 确保您的 DataLoader 返回的 DataFrame 包含 'date', 'open', 'high', 'low', 'close', 'volume' 列
            # 并且日期已是 datetime 类型或可以转换
            lookback_start_date = analysis_date - timedelta(days=CONFIG["data_lookback_days"])
            daily_data_df = crud.get_stock_daily_data_period(db, symbol=stock_code, start_date=lookback_start_date,
                                                             end_date=analysis_date)

            if daily_data_df is None or daily_data_df.empty:
                logger.warning(
                    f"股票 {stock_code} 在日期范围 {lookback_start_date} 到 {analysis_date} 无日线数据。跳过。")
                continue

            # BaseStrategyApp 通常会处理周线和月线的生成，如果策略本身不处理
            # 但我们的 RefactoredMultiLevelCrossStrategy 内部会从日线重采样
            # AdaptedMAPullbackStrategy 也需要周线数据。
            # 这里，我们只传递日线，并假设策略或其调用者会处理周/月转换
            # 或者，如果 DataLoader 能直接提供多级别数据，则更好。
            # 为简化，我们传递日线，并构建一个基础的周线和月线（如果适用）

            data_for_strategies: Dict[str, pd.DataFrame] = {"daily": daily_data_df}

            # 生成周线数据 (如果 AdapatedMAPullbackStrategy 需要)
            # 确保日期索引用 'date' 列，并按日期升序排序
            if not daily_data_df.empty:
                daily_df_sorted = daily_data_df.sort_values(by='date').copy()
                daily_df_sorted['date'] = pd.to_datetime(daily_df_sorted['date'])
                daily_df_sorted = daily_df_sorted.set_index('date')

                try:
                    weekly_df = daily_df_sorted.resample('W-FRI').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna(how='all').reset_index()
                    if not weekly_df.empty: data_for_strategies["weekly"] = weekly_df
                except Exception as e_resample_w:
                    logger.warning(f"为 {stock_code} 生成周线数据失败: {e_resample_w}")

                try:
                    monthly_df = daily_df_sorted.resample('ME').agg({  # ME for Month End
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna(how='all').reset_index()
                    if not monthly_df.empty: data_for_strategies["monthly"] = monthly_df
                except Exception as e_resample_m:
                    logger.warning(f"为 {stock_code} 生成月线数据失败: {e_resample_m}")

        except Exception as e_load:
            logger.error(f"为股票 {stock_code} 加载数据失败: {e_load}", exc_info=True)
            continue

        # b. 运行策略
        technical_signals: List[StrategyResult] = []
        for StrategyClass in CONFIG["strategies_to_run"]:
            # StrategyClass 是策略的类本身，例如 AdaptedMAPullbackStrategy

            # 1. 首先，创建完整的 StrategyContext 实例
            strategy_context = StrategyContext(
                db_session=db,
                current_date=analysis_date,  # <--- 确保 analysis_date 在这里被传递
                strategy_params=CONFIG["strategy_params"]
            )

            try:
                # 2. 然后，使用这个完整的 context 来实例化策略
                strategy_instance = StrategyClass(context=strategy_context)

                # 3. 现在可以安全地从实例获取 strategy_name
                strategy_name_from_instance = strategy_instance.strategy_name
                logger.info(f"  执行策略: {strategy_name_from_instance} for {stock_code}")

                signals_from_strategy = strategy_instance.run_for_stock(
                    stock_code=stock_code,
                    current_date=analysis_date,  # run_for_stock 也需要 current_date
                    data=data_for_strategies
                )
                if signals_from_strategy:
                    technical_signals.extend(signals_from_strategy)
                    for sig in signals_from_strategy:
                        # 使用从实例获取的名称，而不是类名直接作为字符串
                        logger.info(
                            f"技术信号: {stock_code} ({stock_name}) on {sig.signal_date.isoformat()} by {sig.strategy_name} ({sig.details.get('level', 'N/A')})")  # sig.strategy_name 来自 StrategyResult
            except Exception as e_strat:
                # 如果想在错误日志中包含策略类名，可以这样做：
                logger.error(f"策略 {StrategyClass.__name__} 在股票 {stock_code} 上执行失败: {e_strat}",
                             exc_info=True)

        # c. 对每个技术信号进行基本面分析
        if not technical_signals:
            logger.info(f"股票 {stock_code} ({stock_name}) 在 {analysis_date.isoformat()} 无任何策略产生技术信号。")
            continue

        for tech_signal in technical_signals:
            if tech_signal.signal_date != analysis_date:  # 确保只处理当天的信号
                continue

            logger.info(
                f"对信号进行基本面分析: {stock_code} by {tech_signal.strategy_name} on {tech_signal.signal_date.isoformat()}")

            # 获取信号日当天的收盘价用于PE计算等
            # 假设 daily_data_df 是按日期升序排列的，并且包含 analysis_date
            signal_day_price_data = daily_data_df[
                pd.to_datetime(daily_data_df['date']).dt.date == tech_signal.signal_date]
            current_close_price = None
            if not signal_day_price_data.empty:
                current_close_price = signal_day_price_data['close'].iloc[0]
            else:
                logger.warning(f"无法获取股票 {stock_code} 在信号日 {tech_signal.signal_date.isoformat()} 的收盘价。")
                # 可以选择跳过此信号的基本面分析，或用None价格（会导致PE等无法计算）
                # continue # 如果严格要求价格

            if current_close_price is None:
                logger.warning(f"由于缺少信号日收盘价，跳过 {stock_code} 的基本面分析。")
                # 创建一个只包含技术部分和错误信息的条目
                signal_output_data = {
                    "股票代码": tech_signal.stock_code,
                    "股票简称": stock_name,
                    "信号日期": tech_signal.signal_date.isoformat(),
                    "信号来源": tech_signal.strategy_name,
                    "信号级别": tech_signal.details.get("level", "Daily"),  # MA回调策略可能没有level
                    "error_reason": "Missing closing price for fundamental analysis"
                }
                all_final_signals_data.append(signal_output_data)
                continue

            try:
                fundamental_metrics = fundamental_analyzer.analyze_stock(
                    stock_code=tech_signal.stock_code,
                    signal_date=tech_signal.signal_date,
                    current_price=current_close_price
                )

                # 合并技术信号和基本面数据
                signal_output_data = {
                    "股票代码": tech_signal.stock_code,
                    "股票简称": stock_name,
                    "信号日期": tech_signal.signal_date.isoformat(),
                    "信号来源": tech_signal.strategy_name,
                    "信号级别": tech_signal.details.get("level", "Daily"),  # MA回调策略的level可以设为"Daily"
                    # 添加基本面指标
                    **fundamental_metrics  # 解包字典
                }
                # 如果需要，可以从 tech_signal.details 中提取更多技术指标加入
                # signal_output_data["tech_detail_example"] = tech_signal.details.get("some_key")

                all_final_signals_data.append(signal_output_data)
                logger.info(
                    f"基本面分析完成: {stock_code}. PE: {fundamental_metrics.get('pe', 'N/A')}, GrowthPositive: {fundamental_metrics.get('growth_positive', 'N/A')}")

            except Exception as e_fund:
                logger.error(f"对股票 {stock_code} 进行基本面分析失败: {e_fund}", exc_info=True)
                # 记录一个包含错误信息的条目
                signal_output_data = {
                    "股票代码": tech_signal.stock_code,
                    "股票简称": stock_name,
                    "信号日期": tech_signal.signal_date.isoformat(),
                    "信号来源": tech_signal.strategy_name,
                    "信号级别": tech_signal.details.get("level", "Daily"),
                    "error_reason": f"Fundamental analysis failed: {str(e_fund)}"
                }
                all_final_signals_data.append(signal_output_data)

    db.close()  # 关闭数据库会话

    # 4. 保存结果到CSV
    if not all_final_signals_data:
        logger.info("没有生成任何最终信号。")
    else:
        logger.info(f"总共生成 {len(all_final_signals_data)} 条最终信号。准备保存到CSV...")
        final_df = pd.DataFrame(all_final_signals_data)

        # 确保所有预期的列都存在，如果某条记录缺少则填充NaN
        for col in CSV_COLUMN_ORDER:
            if col not in final_df.columns:
                final_df[col] = np.nan

        # 按照指定的顺序排列列，并只选择这些列
        final_df = final_df[CSV_COLUMN_ORDER]

        # 格式化浮点数列为字符串，保留4位小数，处理np.inf
        float_cols_to_format = ['pe', 'revenue_growth_yoy', 'profit_growth_yoy', 'peg_like_ratio']
        for col in float_cols_to_format:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) and isinstance(x, (int, float))
                    else ('+Inf' if x == np.inf else ('-Inf' if x == -np.inf else (str(x) if pd.notna(x) else None)))
                )

        output_filename = f"signals_selected_{analysis_date.isoformat()}.csv"
        output_path = f"{CONFIG['output_dir']}/{output_filename}"
        try:
            # 确保输出目录存在
            import os
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"最终选股结果已保存到: {output_path}")
        except Exception as e_csv:
            logger.error(f"保存CSV文件失败 {output_path}: {e_csv}", exc_info=True)

    logger.info(f"========== 多策略选股器运行结束 - 分析日期: {analysis_date.isoformat()} ==========")


if __name__ == "__main__":
    # 可以接受命令行参数来指定分析日期
    # import argparse
    # parser = argparse.ArgumentParser(description="多策略每日选股器")
    # parser.add_argument("--date", type=str, help="指定分析日期 (YYYY-MM-DD)。默认为最近交易日。")
    # args = parser.parse_args()
    # run_screener(analysis_date_str=args.date)

    run_screener()  # 默认运行