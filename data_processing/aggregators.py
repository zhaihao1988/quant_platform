# data_processing/aggregators.py
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta, datetime
import logging
from typing import List, Optional, Dict, Any
import numpy as np  # 确保导入 numpy

# 确保导入路径正确
from db import crud
from db.models import StockDaily, StockWeekly, StockMonthly
from utils import data_loader as dl  # 假设这是您用于加载数据的模块

logger = logging.getLogger(__name__)


def _prepare_daily_df_for_aggregation(
        daily_df: Optional[pd.DataFrame],
        stock_symbol: str  # 改为 stock_symbol 以清晰
) -> Optional[pd.DataFrame]:
    """
    预处理日线DataFrame，确保列名、日期索引和数据类型正确。
    返回的 DataFrame 的索引是 DatetimeIndex ('date')。
    """
    if daily_df is None or daily_df.empty:
        logger.warning(f"[{stock_symbol}] 输入的日线数据为空，无法预处理。")
        return None

    df = daily_df.copy()

    # 确保 'date' 列存在且为 datetime 类型
    if 'date' not in df.columns:
        logger.error(f"[{stock_symbol}] 日线数据中缺少 'date' 列。")
        return None
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            logger.error(f"[{stock_symbol}] 转换日线数据的 'date' 列为 datetime 失败: {e}")
            return None

    # 将 'date' 列设为索引
    df = df.set_index('date', drop=False)  # 保留 date 列，并将 date 设置为 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):  # 双重检查
        df.index = pd.to_datetime(df.index)

    # 确保股票代码列存在并统一为 'symbol' (与 StockWeekly/Monthly 模型一致)
    if 'stock_code' in df.columns and 'symbol' not in df.columns:
        df.rename(columns={'stock_code': 'symbol'}, inplace=True)
    elif 'symbol' not in df.columns and 'stock_code' in df.columns:  # 如果 StockDaily 用 stock_code
        df['symbol'] = df['stock_code']  # 确保输出有 'symbol' 列
    elif 'symbol' not in df.columns and 'stock_code' not in df.columns:
        logger.warning(f"[{stock_symbol}] 日线数据缺少股票代码列 ('symbol' 或 'stock_code')")
        # 可以考虑从函数参数 stock_symbol 填充，但这假设了df中所有行都是该股票
        df['symbol'] = stock_symbol

    # 确保必要的OHLCV列存在 (根据 StockDaily 模型调整)
    # StockDaily 模型使用的是 open, high, low, close, volume, turnover, amount
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amount']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"[{stock_symbol}] 日线数据中缺少必要列 '{col}'，将用 NaN 填充。")
            df[col] = np.nan  # 用 NaN 填充缺失的必要列

    return df


def _aggregate_single_period_dynamically(
        daily_df_segment: pd.DataFrame,
        stock_symbol: str,
        processing_date: date,  # 最新日线数据的日期，将作为动态K线的日期
        period_type: str,  # 'weekly' 或 'monthly'
        target_model_columns: List[str]
) -> Optional[Dict[str, Any]]:
    """
    为一个特定周期（截至 processing_date）动态合成K线数据。
    K线的 'date' 字段将被设置为 processing_date。
    """
    if daily_df_segment.empty:
        return None

    # daily_df_segment 的索引应该是已经处理好的 DatetimeIndex
    # 并且它只包含属于当前动态周/月并且不晚于 processing_date 的日线数据

    # 确保只使用截至 processing_date 的数据
    segment_for_agg = daily_df_segment[daily_df_segment['date'].dt.date <= processing_date]
    if segment_for_agg.empty:
        return None

    # 聚合规则，换手率按您的要求 sum
    agg_values = {
        'symbol': stock_symbol,  # 使用 symbol
        'date': processing_date,  # 动态K线的日期是当前处理日
        'open': segment_for_agg['open'].iloc[0] if not segment_for_agg['open'].empty else None,
        'high': segment_for_agg['high'].max() if not segment_for_agg['high'].empty else None,
        'low': segment_for_agg['low'].min() if not segment_for_agg['low'].empty else None,
        'close': segment_for_agg['close'].iloc[-1] if not segment_for_agg['close'].empty else None,
        'volume': segment_for_agg['volume'].sum() if not segment_for_agg['volume'].empty else None,
        'turnover': segment_for_agg['turnover'].sum(),  # 按要求 sum
        'amount': segment_for_agg['amount'].sum() if not segment_for_agg['amount'].empty else None,
    }

    if pd.isna(agg_values['open']) or pd.isna(agg_values['close']):
        logger.debug(f"[{stock_symbol}] 动态聚合周期 {processing_date} ({period_type}) 缺少开盘或收盘价，记录无效。")
        return None

    return {k: v for k, v in agg_values.items() if k in target_model_columns and pd.notna(v)}


def _resample_completed_periods(
        daily_df: pd.DataFrame,
        period_type: str,
        stock_symbol: str,
        target_model_columns: List[str],
        boundary_date_for_resample: date  # 只聚合此日期之前（不含）的完整周期
) -> List[Dict[str, Any]]:
    """
    使用标准 resample 聚合历史上已完成的周期。
    boundary_date_for_resample: 例如，当前动态周的开始日，或当前动态月的开始日。
                                 我们只 resample 早于这个边界的日线数据所形成的完整周期。
    """
    if daily_df.empty: return []

    # 筛选出用于 resample 的历史数据部分 (严格早于边界日期)
    df_for_resample = daily_df[daily_df['date'].dt.date < boundary_date_for_resample].copy()

    if df_for_resample.empty:
        logger.debug(
            f"[{stock_symbol}] 没有早于 {boundary_date_for_resample} 的日线数据用于 {period_type} 完整周期聚合。")
        return []

    # 确保索引是 DatetimeIndex
    if not isinstance(df_for_resample.index, pd.DatetimeIndex):
        if 'date' in df_for_resample.columns and pd.api.types.is_datetime64_any_dtype(df_for_resample['date']):
            df_for_resample = df_for_resample.set_index(pd.to_datetime(df_for_resample['date']))
        else:
            logger.error(f"[{stock_symbol}] resample 的日线数据缺少可用的日期索引。")
            return []

    agg_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'turnover': 'sum', 'amount': 'sum'
    }
    rule = ''
    if period_type == 'weekly':
        rule = 'W-FRI'
    elif period_type == 'monthly':
        rule = 'BME'
    else:
        raise ValueError("period_type 必须是 'weekly' 或 'monthly'")

    period_df_resampled = df_for_resample.resample(rule).agg(agg_rules)
    period_df_resampled = period_df_resampled.dropna(subset=['open', 'close'])

    records_list = []
    if not period_df_resampled.empty:
        # 重要：resample 产生的日期是周期的结束日。我们需要确保这些结束日也小于 boundary_date_for_resample
        # 因为 resample 可能会包含部分跨越 boundary_date_for_resample 的周期，但我们只想要完全在此之前的
        period_df_resampled = period_df_resampled[period_df_resampled.index.date < boundary_date_for_resample]

        if not period_df_resampled.empty:
            period_df_resampled_reset = period_df_resampled.reset_index()
            period_df_resampled_reset['symbol'] = stock_symbol  # 使用 symbol

            # pct_chg 将在所有记录（历史+动态）合并后计算
            for _, row in period_df_resampled_reset.iterrows():
                record = row.to_dict()
                record['date'] = pd.to_datetime(row['date']).date()
                record_clean = {k: v for k, v in record.items() if k in target_model_columns and pd.notna(v)}
                if 'open' in record_clean: records_list.append(record_clean)

    logger.debug(
        f"[{stock_symbol}] 为 {period_type} 聚合了 {len(records_list)} 条历史完整周期数据 (截至 {boundary_date_for_resample}之前)。")
    return records_list


def _calculate_pct_chg_and_finalize_records(
        records: List[Dict[str, Any]],
        target_model_columns: List[str]
) -> List[Dict[str, Any]]:
    """对记录列表按日期排序，计算pct_chg，并最终清理字段。"""
    if not records:
        return []

    df = pd.DataFrame(records)
    if 'date' not in df.columns or 'close' not in df.columns:
        logger.warning("记录中缺少 'date' 或 'close' 列，无法计算涨跌幅。")
        # 仅做清理返回
        final_records = []
        for record in records:
            final_records.append({k: v for k, v in record.items() if k in target_model_columns and pd.notna(v)})
        return final_records

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').drop_duplicates(subset=['symbol', 'date'], keep='last')  # 按日期排序并去重，保留最新的动态记录

    df['pct_chg'] = df['close'].pct_change() * 100

    final_records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        record['date'] = record['date'].date()  # 转回 date 对象
        # 再次清理，确保所有字段都有效且在模型中
        final_records.append({k: v for k, v in record.items() if k in target_model_columns and pd.notna(v)})

    return final_records


def calculate_and_store_historical_periods(db: Session, stock_symbol: str):
    logger.info(f"开始为股票 {stock_symbol} 计算和存储历史周线及月线数据...")
    first_day_obj: Optional[date] = None
    last_day_obj: Optional[date] = None  # 这是数据库中日线的最后日期 (processing_date)

    try:
        first_day_obj = db.query(func.min(StockDaily.date)).filter(StockDaily.symbol == stock_symbol).scalar()
        last_day_obj = db.query(func.max(StockDaily.date)).filter(StockDaily.symbol == stock_symbol).scalar()
        if not first_day_obj or not last_day_obj:
            logger.warning(f"股票 {stock_symbol} 在 stock_daily 中没有找到日线数据，无法进行历史聚合。")
            return
    except Exception as e:
        logger.error(f"为 {stock_symbol} 查询日线数据范围失败: {e}", exc_info=True)
        return

    logger.debug(f"为 {stock_symbol} 获取全部日线数据范围: {first_day_obj} 到 {last_day_obj}")
    daily_df_all = dl.load_daily_data(
        symbol=stock_symbol,
        start_date=first_day_obj.strftime('%Y-%m-%d'),
        end_date=last_day_obj.strftime('%Y-%m-%d'),
    )
    daily_df_prepared = _prepare_daily_df_for_aggregation(daily_df_all, stock_symbol)
    if daily_df_prepared is None:
        logger.warning(f"未能加载或预处理股票 {stock_symbol} 的日线数据，无法进行历史聚合。")
        return

    # --- 处理周线 ---
    logger.info(f"为 {stock_symbol} 计算历史周线数据 (含最后一个动态周)...")
    # 1. 确定最后一个动态周的范围和名义结束日
    last_day_dt = pd.Timestamp(last_day_obj)
    # last_week_friday_dt = last_day_dt + pd.offsets.Week(weekday=4) # 最后一天所在周的周五
    # start_of_last_week = (last_week_friday_dt - pd.Timedelta(days=6)).date()

    # 历史完整周的聚合边界应为最后一个动态周的开始日 (的再前一天)
    # 我们用 last_day_obj 作为动态周的日期，它的周数据包含从周一到 last_day_obj
    start_of_last_week_period = last_day_dt.to_period('W').start_time.date()

    historical_weekly_records = _resample_completed_periods(
        daily_df_prepared, 'weekly', stock_symbol,
        StockWeekly.__table__.columns.keys(),
        boundary_date_for_resample=start_of_last_week_period  # 只聚合此日期前的完整周
    )

    # 2. 动态合成最后一个周的K线 (截至 last_day_obj)
    last_week_daily_data = daily_df_prepared[daily_df_prepared['date'].dt.date >= start_of_last_week_period]
    dynamic_last_week_record = _aggregate_single_period_dynamically(
        last_week_daily_data, stock_symbol, last_day_obj, 'weekly',  # date 使用 last_day_obj
        StockWeekly.__table__.columns.keys()
    )

    all_weekly_records = historical_weekly_records
    if dynamic_last_week_record:
        all_weekly_records.append(dynamic_last_week_record)

    final_weekly_to_upsert = _calculate_pct_chg_and_finalize_records(all_weekly_records,
                                                                     StockWeekly.__table__.columns.keys())
    if final_weekly_to_upsert:
        crud.bulk_upsert_stock_weekly(db, final_weekly_to_upsert)
        logger.info(f"成功为 {stock_symbol} 存储/更新 {len(final_weekly_to_upsert)} 条历史周线数据。")

    # --- 处理月线 ---
    logger.info(f"为 {stock_symbol} 计算历史月线数据 (含最后一个动态月)...")
    # last_month_end_dt = last_day_dt + pd.offsets.MonthEnd(0) # 最后一天所在月的月末
    start_of_last_month_period = last_day_dt.to_period('M').start_time.date()

    historical_monthly_records = _resample_completed_periods(
        daily_df_prepared, 'monthly', stock_symbol,
        StockMonthly.__table__.columns.keys(),
        boundary_date_for_resample=start_of_last_month_period
    )
    last_month_daily_data = daily_df_prepared[daily_df_prepared['date'].dt.date >= start_of_last_month_period]
    dynamic_last_month_record = _aggregate_single_period_dynamically(
        last_month_daily_data, stock_symbol, last_day_obj, 'monthly',  # date 使用 last_day_obj
        StockMonthly.__table__.columns.keys()
    )
    all_monthly_records = historical_monthly_records
    if dynamic_last_month_record:
        all_monthly_records.append(dynamic_last_month_record)

    final_monthly_to_upsert = _calculate_pct_chg_and_finalize_records(all_monthly_records,
                                                                      StockMonthly.__table__.columns.keys())
    if final_monthly_to_upsert:
        crud.bulk_upsert_stock_monthly(db, final_monthly_to_upsert)
        logger.info(f"成功为 {stock_symbol} 存储/更新 {len(final_monthly_to_upsert)} 条历史月线数据。")

    logger.info(f"股票 {stock_symbol} 的历史周线和月线数据计算存储完成。")


def update_recent_periods_data(db: Session, stock_symbol: str, lookback_calendar_days: int = 90,
                               processing_date_override: Optional[date] = None):
    """
    计算并更新/插入指定股票最近一段时间的周线和月线数据。
    包含对当前未完成周/月的动态K线合成，其 date 字段为 processing_date。
    """
    processing_date = processing_date_override if processing_date_override else datetime.now().date()
    logger.info(
        f"开始为股票 {stock_symbol} 更新最近 {lookback_calendar_days} 日历日的周/月聚合数据 (处理日期: {processing_date})...")

    start_date_for_fetch = processing_date - timedelta(days=lookback_calendar_days)

    daily_df_recent_raw = dl.load_daily_data(
        symbol=stock_symbol,
        start_date=start_date_for_fetch.strftime('%Y-%m-%d'),
        end_date=processing_date.strftime('%Y-%m-%d'),
    )
    daily_df_recent_prepared = _prepare_daily_df_for_aggregation(daily_df_recent_raw, stock_symbol)

    if daily_df_recent_prepared is None:
        logger.info(f"股票 {stock_symbol} 在最近 {lookback_calendar_days} 天内没有有效日线数据，跳过近期周/月更新。")
        return

    # --- 处理周线 ---
    logger.info(f"为 {stock_symbol} 计算近期周线数据 (含动态周)...")
    # 1. 确定当前动态周的范围和名义结束日 (这里动态周的date字段就是processing_date)
    processing_date_dt = pd.Timestamp(processing_date)
    start_of_current_processing_week = (processing_date_dt.to_period('W').start_time).date()

    # 2. 获取回溯期内已完成的周 (早于当前动态周的开始)
    completed_weekly_records = _resample_completed_periods(
        daily_df_recent_prepared, 'weekly', stock_symbol,
        StockWeekly.__table__.columns.keys(),
        boundary_date_for_resample=start_of_current_processing_week
    )

    # 3. 动态合成当前周的K线数据 (截至 processing_date)
    current_week_daily_data = daily_df_recent_prepared[
        daily_df_recent_prepared['date'].dt.date >= start_of_current_processing_week
        ]  # daily_df_recent_prepared 的date已经是datetime对象，可以直接用dt.date
    dynamic_current_week_record = _aggregate_single_period_dynamically(
        current_week_daily_data, stock_symbol, processing_date, 'weekly',  # date 使用 processing_date
        StockWeekly.__table__.columns.keys()
    )

    all_weekly_records_to_process = completed_weekly_records
    if dynamic_current_week_record:
        all_weekly_records_to_process.append(dynamic_current_week_record)

    final_weekly_to_upsert = _calculate_pct_chg_and_finalize_records(all_weekly_records_to_process,
                                                                     StockWeekly.__table__.columns.keys())
    if final_weekly_to_upsert:
        crud.bulk_upsert_stock_weekly(db, final_weekly_to_upsert)
        logger.info(f"成功为 {stock_symbol} Upsert {len(final_weekly_to_upsert)} 条近期周线数据 (含动态周)。")
    else:
        logger.info(f"股票 {stock_symbol} 最近 {lookback_calendar_days} 天没有生成有效的周线数据进行更新。")

    # --- 处理月线 ---
    logger.info(f"为 {stock_symbol} 计算近期月线数据 (含动态月)...")
    start_of_current_processing_month = processing_date.replace(day=1)

    completed_monthly_records = _resample_completed_periods(
        daily_df_recent_prepared, 'monthly', stock_symbol,
        StockMonthly.__table__.columns.keys(),
        boundary_date_for_resample=start_of_current_processing_month
    )
    current_month_daily_data = daily_df_recent_prepared[
        daily_df_recent_prepared['date'].dt.date >= start_of_current_processing_month
        ]
    dynamic_current_month_record = _aggregate_single_period_dynamically(
        current_month_daily_data, stock_symbol, processing_date, 'monthly',  # date 使用 processing_date
        StockMonthly.__table__.columns.keys()
    )
    all_monthly_records_to_process = completed_monthly_records
    if dynamic_current_month_record:
        all_monthly_records_to_process.append(dynamic_current_month_record)

    final_monthly_to_upsert = _calculate_pct_chg_and_finalize_records(all_monthly_records_to_process,
                                                                      StockMonthly.__table__.columns.keys())
    if final_monthly_to_upsert:
        crud.bulk_upsert_stock_monthly(db, final_monthly_to_upsert)
        logger.info(f"成功为 {stock_symbol} Upsert {len(final_monthly_to_upsert)} 条近期月线数据 (含动态月)。")
    else:
        logger.info(f"股票 {stock_symbol} 最近 {lookback_calendar_days} 天没有生成有效的月线数据进行更新。")

    logger.info(f"股票 {stock_symbol} 的近期周线和月线数据更新完成。")


# --- Main函数用于测试 ---
if __name__ == '__main__':
    # (保持您之前的 __main__ 测试块，确保导入路径和函数调用正确)
    # ... (您的测试代码，调用 calculate_and_store_historical_periods 和 update_recent_periods_data) ...
    # 例如:
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_dir)
    # actual_project_root_for_imports = os.path.dirname(project_root)
    # if actual_project_root_for_imports not in sys.path:
    #     sys.path.insert(0, actual_project_root_for_imports)

    from db.database import SessionLocal  # 确保这个导入在调整 sys.path 后能工作

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,  # 调低级别以便看到更多日志
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]  # 输出到控制台
        )
    logger_main_test = logging.getLogger(__name__ + "_main_test")  # 使用不同的logger名以区分

    logger_main_test.info("开始 aggregators.py 直接调用测试...")
    db_test_session: Optional[Session] = None
    try:
        db_test_session = SessionLocal()  # type: ignore
        if db_test_session is None:
            logger_main_test.error("无法获取数据库会话，测试中止。")
            exit()

        test_stock_symbol = "000001"  # 使用 symbol

        # 测试全量历史回填
        logger_main_test.info(f"\n--- 测试历史数据回填 for {test_stock_symbol} ---")
        calculate_and_store_historical_periods(db_test_session, test_stock_symbol)
        logger_main_test.info(f"历史数据回填测试完成 for {test_stock_symbol}.")

        # 测试最近周期数据更新 (假设今天是 2025-06-03, 数据库中 000001 最新日线是 2025-06-03)
        # 为了可重复测试，可以覆盖 processing_date
        test_processing_date = date(2025, 6, 3)  # 周二
        logger_main_test.info(
            f"\n--- 测试最近周期数据更新 for {test_stock_symbol} (处理日期: {test_processing_date}) ---")
        update_recent_periods_data(db_test_session, test_stock_symbol, lookback_calendar_days=90,
                                   processing_date_override=test_processing_date)
        logger_main_test.info(f"最近周期数据更新测试完成 for {test_stock_symbol}.")

        logger_main_test.info(f"从数据库查询最近5条周线数据 for {test_stock_symbol}:")
        df_w = dl.load_weekly_data(symbol=test_stock_symbol, tail_limit=5,
                                   db_session=db_test_session)  # dl.load_weekly_data 也需要 symbol
        print(df_w if df_w is not None else "无周线数据")

        logger_main_test.info(f"从数据库查询最近5条月线数据 for {test_stock_symbol}:")
        df_m = dl.load_monthly_data(symbol=test_stock_symbol, tail_limit=5,
                                    db_session=db_test_session)  # dl.load_monthly_data 也需要 symbol
        print(df_m if df_m is not None else "无月线数据")

    except Exception as e_test:
        logger_main_test.error(f"aggregators.py 测试过程中发生错误: {e_test}", exc_info=True)
        if db_test_session:
            db_test_session.rollback()
    finally:
        if db_test_session:
            db_test_session.close()
    logger_main_test.info("aggregators.py 直接调用测试结束。")