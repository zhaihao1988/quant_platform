# data_processing/loader.py
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import text, or_, and_, not_
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date

# Use correct path for models and database session
from db.models import StockDisclosure, StockDaily, StockFinancial
from db.database import get_engine_instance  # Use engine instance for pandas

logger = logging.getLogger(__name__)
engine = get_engine_instance()
try:
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning("python-dateutil not installed. Using timedelta(days=3*365) for 3-year calculation, which might be less accurate.")

# Keywords for filtering disclosures
DISCLOSURE_KEYWORDS = ['年度报告', '半年度报告', '调研', '股权激励', '回购']

def load_price_data(symbol: str, window: int = 90) -> Optional[pd.DataFrame]:
    """Loads recent daily price data for a given stock symbol."""
    if engine is None:
        logger.error("Database engine not available for loading price data.")
        return None
    logger.info(f"Loading price data for {symbol}, window={window} days.")
    sql = text("""
      SELECT date, open, close, high, low, volume, pct_change, amount, turnover
      FROM stock_daily
      WHERE symbol = :symbol
      ORDER BY date DESC
      LIMIT :limit
    """)
    try:
        # Ensure correct parameter binding
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection, params={"symbol": symbol, "limit": window})
        if df.empty:
            logger.warning(f"No price data found for symbol {symbol} within the last {window} days.")
            return None
        # Reverse to have chronological order for analysis
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}", exc_info=True)
        return None

def load_financial_data(symbol: str, report_type: str) -> Optional[Dict[str, Any]]:
    """Loads the latest financial report data (JSONB) for a given type."""
    # Valid report types might be 'benefit', 'balance', 'cashflow' etc.
    if engine is None:
        logger.error("Database engine not available for loading financial data.")
        return None
    logger.info(f"Loading latest '{report_type}' financial data for {symbol}.")
    sql = text("""
      SELECT data
      FROM stock_financial
      WHERE symbol = :symbol AND report_type = :rtype
      ORDER BY report_date DESC
      LIMIT 1
    """)
    try:
        # Ensure correct parameter binding
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection, params={"symbol": symbol, "rtype": report_type})
        if df.empty or 'data' not in df.columns or df['data'].iloc[0] is None:
            logger.warning(f"No financial data found for symbol {symbol}, report_type '{report_type}'.")
            return None
        # Assuming the 'data' column stores JSON
        return df["data"].iloc[0]
    except Exception as e:
        logger.error(f"Error loading financial data for {symbol} (type: {report_type}): {e}", exc_info=True)
        return None
def load_multiple_financial_reports(symbol: str, report_type: str, num_years: int = 3) -> List[Dict[str, Any]]:
    """
    加载指定股票和报告类型的财务数据：
    1. 最新一期报告。
    2. 最近 num_years 个年度报告 (截止日期为 12-31)。
    返回包含报告日期和数据的字典列表，按日期降序排列。
    """
    if engine is None:
        logger.error("Database engine not available for loading financial data.")
        return []
    logger.info(f"Loading multiple '{report_type}' reports for {symbol} (latest + last {num_years} annual).")

    # 计算 N 年前的年份
    current_year = datetime.now().year
    start_year = current_year - num_years
    # 从 N 年前年初开始查，确保覆盖年报
    start_date_limit = date(start_year, 1, 1)

    sql = text("""
        SELECT report_date, data
        FROM stock_financial
        WHERE symbol = :symbol
          AND report_type = :rtype
          AND report_date >= :start_date
        ORDER BY report_date DESC
    """)

    reports_data = []
    try:
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection, params={
                "symbol": symbol,
                "rtype": report_type,
                "start_date": start_date_limit
            })

        if df.empty:
            logger.warning(f"No recent financial data found for {symbol}, type '{report_type}' since {start_date_limit}.")
            return []

        df['report_date'] = pd.to_datetime(df['report_date']).dt.date
        selected_reports = {}

        # 1. 添加最新报告
        latest_report_row = df.iloc[0]
        latest_date = latest_report_row['report_date']
        if pd.notna(latest_date) and latest_report_row['data'] is not None: # 检查非空
             selected_reports[latest_date] = latest_report_row['data']
             logger.debug(f"Added latest report: {latest_date}")
        else:
             logger.warning(f"Latest report has null date or data for {symbol}, type {report_type}")


        # 2. 添加最近 num_years 的年报 (12-31)
        annual_report_count = 0
        for _, row in df.iterrows():
            report_date = row['report_date']
            if pd.notna(report_date) and row['data'] is not None: # 检查非空
                 # 检查是否是年报且尚未添加
                 if report_date.month == 12 and report_date.day == 31 and report_date not in selected_reports:
                     if annual_report_count < num_years:
                         selected_reports[report_date] = row['data']
                         annual_report_count += 1
                         logger.debug(f"Added annual report: {report_date}")
                         if annual_report_count >= num_years:
                             break # 找到足够数量

        # 转换并排序
        reports_data = [{'report_date': dt, 'data': data} for dt, data in selected_reports.items()]
        reports_data.sort(key=lambda x: x['report_date'], reverse=True)

        logger.info(f"Loaded {len(reports_data)} financial reports for {symbol}, type '{report_type}'.")
        return reports_data

    except Exception as e:
        logger.error(f"Error loading multiple financial reports for {symbol} (type: {report_type}): {e}", exc_info=True)
        return []
def get_disclosures_needing_content(db: Session, symbol: str) -> List[StockDisclosure]:
    """
    查找需要处理内容的公告。
    - 年度报告/半年度报告：仅查找最近3年的。
    - 调研/股权激励/回购：仅查找最近1年的。
    - 排除标题含 "摘要" 的。
    - 只查找 raw_content 尚为空 (NULL) 的记录。
    """
    logger.info(f"Querying database for disclosures needing content for symbol: {symbol}")
    try:
        # --- 计算日期阈值 ---
        now = datetime.now()
        one_year_ago = now - timedelta(days=365) # 其他关键词使用1年限制

        if DATEUTIL_AVAILABLE:
            three_years_ago = now - relativedelta(years=3)
        else:
            three_years_ago = now - timedelta(days=3*365 + 1) # 粗略计算

        logger.debug(f"Filtering Annual/Semi-Annual Reports >= {three_years_ago.date()}")
        logger.debug(f"Filtering Other Keywords >= {one_year_ago.date()}")

        # --- 定义关键词 ---
        annual_semi_keywords = ['年度报告', '半年度报告']
        other_time_keywords = ['调研', '股权激励', '回购']

        # --- 构建过滤器列表 ---
        filter_conditions = []

        # 条件组1：最近3年的年报/半年报
        annual_semi_filters = [
            and_(
                StockDisclosure.title.ilike(f'%{kw}%'),
                StockDisclosure.ann_date >= three_years_ago.date() # **修正：应用3年日期限制**
            )
            for kw in annual_semi_keywords
        ]
        if annual_semi_filters:
            filter_conditions.append(or_(*annual_semi_filters))

        # 条件组2：最近1年的其他关键词
        other_time_filters = [
            and_(
                StockDisclosure.title.ilike(f'%{kw}%'),
                StockDisclosure.ann_date >= one_year_ago.date() # **保持1年日期限制**
            )
            for kw in other_time_keywords
        ]
        if other_time_filters:
            filter_conditions.append(or_(*other_time_filters))

        # --- 组合所有条件组 ---
        if not filter_conditions:
            logger.warning(f"No keyword filters active for {symbol}.")
            return []
        combined_keyword_date_filters = or_(*filter_conditions)

        # --- 执行最终查询 ---
        disclosures = db.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            StockDisclosure.raw_content == None,             # 只找还没内容的
            not_(StockDisclosure.title.ilike('%摘要%')),     # 排除摘要
            combined_keyword_date_filters                   # 应用组合后的关键词和日期限制
        ).order_by(StockDisclosure.ann_date.desc()).all()   # 优先处理最近的

        logger.info(f"Found {len(disclosures)} disclosures needing content for {symbol} (within time limits and unprocessed).")
        return disclosures
    except Exception as e:
        logger.error(f"Error querying disclosures needing content for {symbol}: {e}", exc_info=True)
        return []
# Removed the faulty load_announcements function which duplicated orchestration logic.
# Orchestration will now happen in main.py or a dedicated processing script.