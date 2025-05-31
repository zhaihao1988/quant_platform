# data_processing/aggregators.py
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta, datetime
import logging
from typing import List, Optional, Dict, Any
from db import crud
from db.models import StockDaily, StockWeekly, StockMonthly  # 确保导入 StockWeekly, StockMonthly
from utils import data_loader as dl

logger = logging.getLogger(__name__)


# resample_to_period_df 和 calculate_and_store_historical_periods 函数保持不变 (如上一轮所示)
# ... (resample_to_period_df from previous response) ...
# ... (calculate_and_store_historical_periods from previous response, ensuring it uses dl.load_daily_data for all daily data) ...

def resample_to_period_df(daily_df: pd.DataFrame, period_type: str) -> pd.DataFrame:
    # (保持上一轮提供的 resample_to_period_df 函数实现)
    if daily_df.empty:
        logger.warning(f"输入给 resample_to_period_df 的 daily_df (period: {period_type}) 为空。")
        return pd.DataFrame()
    df = daily_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    if df.index.name != 'date':
        df = df.set_index('date')
    agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'turnover': 'sum'}
    if period_type == 'weekly':
        period_df = df.resample('W-FRI').agg(agg_rules)
    elif period_type == 'monthly':
        period_df = df.resample('BME').agg(agg_rules)
    else:
        logger.error(f"无效的 period_type: {period_type}")
        raise ValueError("period_type 必须是 'weekly' 或 'monthly'")
    period_df = period_df.dropna(subset=['open'])
    logger.debug(f"聚合了 {len(period_df)} 条 {period_type} 数据。")
    return period_df.reset_index()


def calculate_and_store_historical_periods(db: Session, stock_code: str):
    # (保持上一轮提供的 calculate_and_store_historical_periods 函数实现)
    logger.info(f"开始为股票 {stock_code} 计算和存储历史周线及月线数据...")
    first_day_obj: Optional[date] = None
    last_day_obj: Optional[date] = None
    daily_df: Optional[pd.DataFrame] = None
    try:
        first_day_obj = db.query(func.min(StockDaily.date)).filter(StockDaily.symbol == stock_code).scalar()
        last_day_obj = db.query(func.max(StockDaily.date)).filter(StockDaily.symbol == stock_code).scalar()
        if not first_day_obj or not last_day_obj:
            logger.warning(f"股票 {stock_code} 在 stock_daily 中没有找到日线数据，无法进行历史聚合。")
            return
        logger.debug(f"为 {stock_code} 获取日线数据范围: {first_day_obj} 到 {last_day_obj}")
        daily_df = dl.load_daily_data(
            symbol=stock_code,
            start_date=first_day_obj.strftime('%Y-%m-%d'),
            end_date=last_day_obj.strftime('%Y-%m-%d'),
        )
    except Exception as e:
        logger.error(f"为 {stock_code} 获取全部日线数据失败: {e}", exc_info=True)
        return
    if daily_df is None or daily_df.empty:
        logger.warning(f"未能加载股票 {stock_code} 的日线数据，无法进行历史聚合。")
        return
    if 'symbol' in daily_df.columns and 'stock_code' not in daily_df.columns:
        daily_df.rename(columns={'symbol': 'stock_code'}, inplace=True)
    elif 'stock_code' not in daily_df.columns and 'symbol' in daily_df.columns:
        daily_df['stock_code'] = daily_df['symbol']
    if not pd.api.types.is_datetime64_any_dtype(daily_df['date']):
        daily_df['date'] = pd.to_datetime(daily_df['date'])

    logger.info(f"为 {stock_code} 计算历史周线数据...")
    weekly_agg_df = resample_to_period_df(daily_df, 'weekly')
    if not weekly_agg_df.empty:
        weekly_agg_df['stock_code'] = stock_code
        weekly_agg_df['pct_chg_calc'] = weekly_agg_df['close'].pct_change() * 100
        weekly_records = []
        for _, row in weekly_agg_df.iterrows():
            record = row.to_dict()
            record['pct_chg'] = record.pop('pct_chg_calc', None)
            record = {k: v for k, v in record.items() if k in StockWeekly.__table__.columns.keys() and pd.notna(v)}
            if 'open' in record: weekly_records.append(record)  # Ensure essential data
        if weekly_records: crud.bulk_upsert_stock_weekly(db, weekly_records)
        logger.info(f"成功为 {stock_code} 存储/更新 {len(weekly_records)} 条历史周线数据。")

    logger.info(f"为 {stock_code} 计算历史月线数据...")
    monthly_agg_df = resample_to_period_df(daily_df, 'monthly')
    if not monthly_agg_df.empty:
        monthly_agg_df['stock_code'] = stock_code
        monthly_agg_df['pct_chg_calc'] = monthly_agg_df['close'].pct_change() * 100
        monthly_records = []
        for _, row in monthly_agg_df.iterrows():
            record = row.to_dict()
            record['pct_chg'] = record.pop('pct_chg_calc', None)
            record = {k: v for k, v in record.items() if k in StockMonthly.__table__.columns.keys() and pd.notna(v)}
            if 'open' in record: monthly_records.append(record)
        if monthly_records: crud.bulk_upsert_stock_monthly(db, monthly_records)
        logger.info(f"成功为 {stock_code} 存储/更新 {len(monthly_records)} 条历史月线数据。")
    logger.info(f"股票 {stock_code} 的历史周线和月线数据计算存储完成。")


def update_recent_periods_data(db: Session, stock_code: str, lookback_calendar_days: int = 90):
    """
    计算并更新/插入指定股票最近一段时间的周线和月线数据。
    用于每日例行更新，确保最近完成的周期和当前动态周期的数据是最新的。
    :param db: SQLAlchemy Session.
    :param stock_code: 股票代码.
    :param lookback_calendar_days: 从今天回溯多少个日历日的数据用于重新计算。
                                   应足够覆盖至少2-3个完整周期（例如，月线则至少60-90天）。
    """
    logger.info(f"开始为股票 {stock_code} 更新最近 {lookback_calendar_days} 日历日的周/月聚合数据...")

    # 1. 确定获取日线数据的日期范围
    # processing_date 可以是今天的日期，或者如果您在特定日期后运行批处理，则是那个批处理的业务日期
    processing_date = datetime.now().date()
    start_date_for_fetch = processing_date - timedelta(days=lookback_calendar_days)

    # 2. 获取最近一段时间的日线数据
    logger.debug(f"为 {stock_code} 获取最近日线数据范围: {start_date_for_fetch} 到 {processing_date}")
    # 假设 dl.load_daily_data 使用 stock_code/symbol, 并能处理 session
    # 并且假设 StockDaily 中股票代码字段为 symbol
    daily_df_recent = dl.load_daily_data(
        symbol=stock_code,  # 确保与 load_daily_data 的参数名一致
        start_date=start_date_for_fetch.strftime('%Y-%m-%d'),
        end_date=processing_date.strftime('%Y-%m-%d'),
        # db_session=db # 如果 load_daily_data 支持传入session
    )

    if daily_df_recent is None or daily_df_recent.empty:
        logger.info(f"股票 {stock_code} 在最近 {lookback_calendar_days} 天内没有日线数据，跳过近期周/月更新。")
        return

    # 确保列名和数据类型正确
    if 'symbol' in daily_df_recent.columns and 'stock_code' not in daily_df_recent.columns:
        daily_df_recent.rename(columns={'symbol': 'stock_code'}, inplace=True)
    elif 'stock_code' not in daily_df_recent.columns and 'symbol' in daily_df_recent.columns:
        daily_df_recent['stock_code'] = daily_df_recent['symbol']

    if not pd.api.types.is_datetime64_any_dtype(daily_df_recent['date']):
        daily_df_recent['date'] = pd.to_datetime(daily_df_recent['date'])

    # 3. 重新聚合这些日线数据为周线，并 Upsert
    logger.info(f"为 {stock_code} 计算近期周线数据...")
    weekly_recent_agg_df = resample_to_period_df(daily_df_recent, 'weekly')
    if not weekly_recent_agg_df.empty:
        weekly_recent_agg_df['stock_code'] = stock_code
        weekly_recent_agg_df['pct_chg_calc'] = weekly_recent_agg_df['close'].pct_change() * 100

        weekly_records = []
        for _, row in weekly_recent_agg_df.iterrows():
            record = row.to_dict()
            record['pct_chg'] = record.pop('pct_chg_calc', None)
            # 过滤掉NaN值并确保键在模型中，以避免插入错误
            record_clean = {k: v for k, v in record.items() if
                            k in StockWeekly.__table__.columns.keys() and pd.notna(v)}
            if 'open' in record_clean and 'close' in record_clean:  # 确保核心数据存在
                weekly_records.append(record_clean)

        if weekly_records:
            crud.bulk_upsert_stock_weekly(db, weekly_records)
            logger.info(f"成功为 {stock_code} Upsert {len(weekly_records)} 条近期周线数据。")
    else:
        logger.info(f"股票 {stock_code} 最近 {lookback_calendar_days} 天没有生成有效的近期周线数据。")

    # 4. 重新聚合这些日线数据为月线，并 Upsert
    logger.info(f"为 {stock_code} 计算近期月线数据...")
    monthly_recent_agg_df = resample_to_period_df(daily_df_recent, 'monthly')
    if not monthly_recent_agg_df.empty:
        monthly_recent_agg_df['stock_code'] = stock_code
        monthly_recent_agg_df['pct_chg_calc'] = monthly_recent_agg_df['close'].pct_change() * 100

        monthly_records = []
        for _, row in monthly_recent_agg_df.iterrows():
            record = row.to_dict()
            record['pct_chg'] = record.pop('pct_chg_calc', None)
            record_clean = {k: v for k, v in record.items() if
                            k in StockMonthly.__table__.columns.keys() and pd.notna(v)}
            if 'open' in record_clean and 'close' in record_clean:  # 确保核心数据存在
                monthly_records.append(record_clean)

        if monthly_records:
            crud.bulk_upsert_stock_monthly(db, monthly_records)
            logger.info(f"成功为 {stock_code} Upsert {len(monthly_records)} 条近期月线数据。")
    else:
        logger.info(f"股票 {stock_code} 最近 {lookback_calendar_days} 天没有生成有效的近期月线数据。")

    logger.info(f"股票 {stock_code} 的近期周线和月线数据更新完成。")


# --- Main函数用于测试 (无Mock版本，修正了导入问题) ---
if __name__ == '__main__':
    import os
    import sys
    import logging  # 确保 logging 在这里也被正确配置或导入
    from sqlalchemy.orm import Session  # 移到这里，因为 SessionLocal 会在下面导入
    from sqlalchemy import func  # 移到这里

    # 将项目根目录添加到 sys.path
    # aggregators.py 位于 quant_platform/data_processing/aggregators.py
    # 所以父目录是 data_processing, 父父目录是 quant_platform (我们假设这是项目根目录)
    # os.path.abspath(__file__) 获取当前文件的绝对路径
    # os.path.dirname() 获取目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # data_processing 目录
    project_root = os.path.dirname(current_dir)  # quant_platform 目录

    # 如果 quant_platform 的上一级才是真正的项目根 (例如 D:\project\ )
    # 并且您希望从 D:\project\quant_platform 这样引用，那么 project_root 就是正确的
    # 如果您的顶级包是 quant_platform，那么这里应该添加包含 quant_platform 的目录到 sys.path
    # 假设 quant_platform 是可以被导入的顶级包名，那么包含它的目录 (D:\project) 应该在 sys.path 中
    # 为了能找到 quant_platform.db 等，我们需要 D:\project 在 sys.path
    actual_project_root_for_imports = os.path.dirname(project_root)  # D:\project
    if actual_project_root_for_imports not in sys.path:
        sys.path.insert(0, actual_project_root_for_imports)  # 插入到最前面

    # 现在因为 D:\project 在 sys.path 中, 我们可以使用绝对路径从 quant_platform 开始导入
    from db import crud
    from db.models import StockDaily, StockWeekly, StockMonthly
    from utils import data_loader as dl
    from db.database import SessionLocal  # 假设 SessionLocal 是你的会话工厂

    # 配置日志记录 (确保在导入其他模块后，如果它们也配置日志，这里的配置可能被覆盖或需要协调)
    # 或者在脚本最开始就配置全局日志
    if not logging.getLogger().hasHandlers():  # 避免重复配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    logger = logging.getLogger(__name__)  # 获取本模块的logger实例

    logger.info("开始 aggregators.py 直接调用测试 (无Mock, sys.path 已调整)...")

    # !!! 警告: 此处直接操作数据库，请在测试环境或确保数据安全的情况下运行 !!!

    db: Session = SessionLocal()
    try:
        test_stock_code = "000001"  # 用于从 stock_daily 读取的 symbol/code

        # 1. 测试历史数据回填
        # logger.info(f"\n--- 测试历史数据回填 for {test_stock_code} ---")
        # aggregators.calculate_and_store_historical_periods(db, test_stock_code) # 注意：aggregators.py 内部函数调用不需要前缀
        # calculate_and_store_historical_periods(db, test_stock_code)
        # logger.info(f"历史数据回填测试完成 for {test_stock_code}.")

        # 2. 测试最近周期数据更新
        actual_latest_daily_date = db.query(func.max(StockDaily.date)).filter(
            StockDaily.symbol == test_stock_code).scalar()

        if actual_latest_daily_date:
            logger.info(
                f"\n--- 测试最近周期数据更新 for {test_stock_code} (基于日线最新到: {actual_latest_daily_date}) ---")
            update_recent_periods_data(db, test_stock_code, lookback_calendar_days=90)  # 调用本文件内的函数
            logger.info(f"最近周期数据更新测试完成 for {test_stock_code}.")

            logger.info(f"从数据库查询最近5条周线数据 for {test_stock_code}:")
            # 这里的 dl.load_weekly_data 仍然需要 symbol 参数名，而我们内部用 stock_code
            # 确保 dl.load_weekly_data 使用的列名与 stock_weekly 表中一致 (我们之前定义为 stock_code)
            # 如果 dl.load_weekly_data 内部查询 stock_weekly.stock_code，那么传入的 symbol 参数需要对应
            df_w = dl.load_weekly_data(symbol=test_stock_code, tail_limit=5, db_session=db)
            print(df_w)

            logger.info(f"从数据库查询最近5条月线数据 for {test_stock_code}:")
            df_m = dl.load_monthly_data(symbol=test_stock_code, tail_limit=5, db_session=db)
            print(df_m)
        else:
            logger.warning(f"无法获取 {test_stock_code} 的最新日线交易日，跳过最新周期数据更新测试。")

    except Exception as e:
        logger.error(f"aggregators.py 测试过程中发生错误: {e}", exc_info=True)
        if db:
            db.rollback()
    finally:
        if db:
            db.close()
    logger.info("aggregators.py 直接调用测试结束。")