# quant_platform/utils/data_loader.py
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, or_, and_, not_, desc  # 从 data_processing/loader.py 移动
from datetime import datetime, timedelta, date  # 从 data_processing/loader.py 移动
from typing import List, Dict, Optional, Any  # 从 data_processing/loader.py 移动
import logging  # 从 data_processing/loader.py 移动

# 数据库模型和引擎实例 (确保路径正确)
from db.models import StockDaily, StockFinancial, StockDisclosure  # 从 data_processing/loader.py 移动, StockList 已存在
from db.database import get_engine_instance, SessionLocal  # SessionLocal 可能用于需要传入db session的函数

logger = logging.getLogger(__name__)
engine = get_engine_instance()  # 保持 engine 实例的获取方式

try:
    from dateutil.relativedelta import relativedelta

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning(
        "python-dateutil not installed. Using timedelta(days=3*365) for 3-year calculation, which might be less accurate.")


def load_daily_data(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        tail_limit: Optional[int] = None,
        db_session: Optional[Session] = None  # 可选参数，如果需要用传入的session
) -> pd.DataFrame:
    """
    从数据库加载日线数据。
    可按日期范围加载，或加载最近N条数据。

    :param symbol: 股票代码，如 '000001'
    :param start_date: 起始日期 'YYYY-MM-DD' (与 tail_limit互斥)
    :param end_date: 结束日期 'YYYY-MM-DD' (与 tail_limit互斥)
    :param fields: 需要的字段列表，默认全字段
    :param tail_limit: 获取最近N条数据 (与 start_date/end_date 互斥)
    :param db_session: 可选的数据库会话
    :return: Pandas DataFrame
    """
    if tail_limit is not None and (start_date is not None or end_date is not None):
        raise ValueError("tail_limit cannot be used with start_date or end_date.")

    selected_fields = "*"
    if fields:
        selected_fields = ", ".join(fields)

    # 使用 with engine.connect() 来确保连接在使用后关闭，或者使用传入的session
    conn_or_session = db_session if db_session else engine

    if tail_limit is not None:
        # 实现 load_price_data 的功能
        logger.info(f"Loading last {tail_limit} daily price data for {symbol}.")
        # 注意：stock_daily 表中的字段名可能与 load_price_data 中写死的不完全一致，比如 pct_change vs pct_chg
        # 这里我们假设 StockDaily 模型定义的字段是标准，从 stock_daily 表读取
        # 如果原始 load_price_data 中的字段名如 amount, pct_change 与 StockDaily 不同，需要做映射或确认
        sql_query = text(f"""
          SELECT {selected_fields} 
          FROM stock_daily
          WHERE symbol = :symbol
          ORDER BY date DESC
          LIMIT :limit
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "limit": tail_limit}, parse_dates=["date"])
            else:  # engine
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection, params={"symbol": symbol, "limit": tail_limit},
                                     parse_dates=["date"])
            if df.empty:
                logger.warning(f"No price data found for symbol {symbol} within the last {tail_limit} days.")
                return pd.DataFrame()  # 返回空DataFrame
            return df.iloc[::-1].reset_index(drop=True)  # 保持时间升序
        except Exception as e:
            logger.error(f"Error loading last {tail_limit} price data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    elif start_date and end_date:
        # 保留原有的 load_daily_data 功能
        logger.info(f"Loading daily data for {symbol} from {start_date} to {end_date}.")
        sql_query = text(f"""
            SELECT {selected_fields} FROM stock_daily
            WHERE symbol = :symbol
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                 parse_dates=["date"])
            else:  # engine
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection,
                                     params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                     parse_dates=["date"])
            return df
        except Exception as e:
            logger.error(f"Error loading daily data for {symbol} between {start_date}-{end_date}: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.error("Either date range (start_date, end_date) or tail_limit must be provided.")
        return pd.DataFrame()


def load_financial_data(symbol: str, report_type: str, db_session: Optional[Session] = None) -> Optional[
    Dict[str, Any]]:
    """
    加载最新的财务报告数据 (JSONB)。
    与 quant_platform/data_processing/loader.py::load_financial_data 类似。
    """
    conn_or_session = db_session if db_session else engine
    logger.info(f"Loading latest '{report_type}' financial data for {symbol}.")
    # SQL 和逻辑与原 data_processing/loader.py::load_financial_data 中的相同
    sql = text("""
      SELECT data
      FROM stock_financial
      WHERE symbol = :symbol AND report_type = :rtype
      ORDER BY report_date DESC
      LIMIT 1
    """)
    try:
        if isinstance(conn_or_session, Session):
            df = pd.read_sql(sql, conn_or_session.connection(), params={"symbol": symbol, "rtype": report_type})
        else:  # engine
            with conn_or_session.connect() as connection:
                df = pd.read_sql(sql, connection, params={"symbol": symbol, "rtype": report_type})

        if df.empty or 'data' not in df.columns or df['data'].iloc[0] is None:
            logger.warning(f"No financial data found for symbol {symbol}, report_type '{report_type}'.")
            return None
        return df["data"].iloc[0]
    except Exception as e:
        logger.error(f"Error loading financial data for {symbol} (type: {report_type}): {e}", exc_info=True)
        return None


def load_multiple_financial_reports(symbol: str, report_type: str, num_years: int = 3,
                                    db_session: Optional[Session] = None) -> List[Dict[str, Any]]:
    """
    加载指定股票和报告类型的财务数据：最新一期 + 最近N年年报。
    与 quant_platform/data_processing/loader.py::load_multiple_financial_reports 类似。
    """
    conn_or_session = db_session if db_session else engine
    logger.info(f"Loading multiple '{report_type}' reports for {symbol} (latest + last {num_years} annual).")
    # SQL 和逻辑与原 data_processing/loader.py::load_multiple_financial_reports 中的相同
    current_year = datetime.now().year
    start_year = current_year - num_years
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
        if isinstance(conn_or_session, Session):
            df = pd.read_sql(sql, conn_or_session.connection(), params={
                "symbol": symbol, "rtype": report_type, "start_date": start_date_limit
            })
        else:  # engine
            with conn_or_session.connect() as connection:
                df = pd.read_sql(sql, connection, params={
                    "symbol": symbol, "rtype": report_type, "start_date": start_date_limit
                })

        if df.empty:
            logger.warning(
                f"No recent financial data found for {symbol}, type '{report_type}' since {start_date_limit}.")
            return []

        df['report_date'] = pd.to_datetime(df['report_date']).dt.date
        selected_reports = {}

        latest_report_row = df.iloc[0]
        latest_date = latest_report_row['report_date']
        if pd.notna(latest_date) and latest_report_row['data'] is not None:
            selected_reports[latest_date] = latest_report_row['data']
            logger.debug(f"Added latest report: {latest_date}")
        else:
            logger.warning(f"Latest report has null date or data for {symbol}, type {report_type}")

        annual_report_count = 0
        for _, row in df.iterrows():
            report_date_val = row['report_date']  # 重命名以避免与外部report_date冲突
            if pd.notna(report_date_val) and row['data'] is not None:
                if report_date_val.month == 12 and report_date_val.day == 31 and report_date_val not in selected_reports:
                    if annual_report_count < num_years:
                        selected_reports[report_date_val] = row['data']
                        annual_report_count += 1
                        logger.debug(f"Added annual report: {report_date_val}")
                        if annual_report_count >= num_years:
                            break

        reports_data = [{'report_date': dt, 'data': data} for dt, data in selected_reports.items()]
        reports_data.sort(key=lambda x: x['report_date'], reverse=True)

        logger.info(f"Loaded {len(reports_data)} financial reports for {symbol}, type '{report_type}'.")
        return reports_data
    except Exception as e:
        logger.error(f"Error loading multiple financial reports for {symbol} (type: {report_type}): {e}", exc_info=True)
        return []


def get_disclosures_needing_content(db: Session, symbol: str) -> List[StockDisclosure]:
    """
    查找需要处理原始内容的公司公告。
    与 quant_platform/data_processing/loader.py::get_disclosures_needing_content 逻辑相同。
    """
    logger.info(f"Querying database for disclosures needing content for symbol: {symbol}")
    # 此处直接复制粘贴 quant_platform/data_processing/loader.py 中 get_disclosures_needing_content 的完整逻辑
    # ... (逻辑完全一致) ...
    try:
        now = datetime.now()
        one_year_ago = now - timedelta(days=365)
        if DATEUTIL_AVAILABLE:
            three_years_ago = now - relativedelta(years=3)
        else:
            three_years_ago = now - timedelta(days=3 * 365 + 1)

        annual_semi_keywords = ['年度报告', '半年度报告']
        other_time_keywords = ['调研', '股权激励', '回购']
        filter_conditions = []

        annual_semi_filters = [
            and_(StockDisclosure.title.ilike(f'%{kw}%'), StockDisclosure.ann_date >= three_years_ago.date())
            for kw in annual_semi_keywords
        ]
        if annual_semi_filters:
            filter_conditions.append(or_(*annual_semi_filters))

        other_time_filters = [
            and_(StockDisclosure.title.ilike(f'%{kw}%'), StockDisclosure.ann_date >= one_year_ago.date())
            for kw in other_time_keywords
        ]
        if other_time_filters:
            filter_conditions.append(or_(*other_time_filters))

        if not filter_conditions:
            logger.warning(f"No keyword filters active for {symbol}.")
            return []
        combined_keyword_date_filters = or_(*filter_conditions)

        disclosures = db.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            StockDisclosure.raw_content == None,
            not_(StockDisclosure.title.ilike('%摘要%')),
            combined_keyword_date_filters
        ).order_by(StockDisclosure.ann_date.desc()).all()

        logger.info(
            f"Found {len(disclosures)} disclosures needing content for {symbol} (within time limits and unprocessed).")
        return disclosures
    except Exception as e:
        logger.error(f"Error querying disclosures needing content for {symbol}: {e}", exc_info=True)
        return []


# === 新增：加载周线和月线数据的函数 ===
def load_weekly_data(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        tail_limit: Optional[int] = None,
        db_session: Optional[Session] = None
) -> pd.DataFrame:
    """
    从数据库加载周线数据 (stock_weekly 表)。
    可按日期范围加载，或加载最近N条数据。
    :param symbol: 股票代码
    :param start_date: 起始日期 'YYYY-MM-DD' (周五)
    :param end_date: 结束日期 'YYYY-MM-DD' (周五)
    :param fields: 需要的字段列表，默认全字段
    :param tail_limit: 获取最近N条周线数据
    :param db_session: 可选的数据库会话
    :return: Pandas DataFrame
    """
    if tail_limit is not None and (start_date is not None or end_date is not None):
        raise ValueError("tail_limit cannot be used with start_date or end_date.")

    table_name = "stock_weekly"
    selected_fields = "*"
    if fields:
        # 确保 'date' 和 'symbol' 字段总是被选择，如果用户没有指定的话，因为它们常用于后续处理
        if 'date' not in fields: fields.append('date')
        if 'symbol' not in fields: fields.append('symbol')
        selected_fields = ", ".join(list(set(fields)))  # 去重并转为字符串

    conn_or_session = db_session if db_session else engine

    if tail_limit is not None:
        logger.info(f"Loading last {tail_limit} weekly data for {symbol} from {table_name}.")
        sql_query = text(f"""
          SELECT {selected_fields} 
          FROM {table_name}
          WHERE symbol = :symbol  -- 注意：新表中 symbol 而非 symbol
          ORDER BY date DESC
          LIMIT :limit
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "limit": tail_limit}, parse_dates=["date"])
            else:
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection, params={"symbol": symbol, "limit": tail_limit},
                                     parse_dates=["date"])
            if df.empty:
                logger.warning(f"No weekly data found for symbol {symbol} within the last {tail_limit} weeks.")
                return pd.DataFrame()
            return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error loading last {tail_limit} weekly data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    elif start_date and end_date:
        logger.info(f"Loading weekly data for {symbol} from {start_date} to {end_date} from {table_name}.")
        sql_query = text(f"""
            SELECT {selected_fields} FROM {table_name}
            WHERE symbol = :symbol -- 注意：新表中 symbol 而非 symbol
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                 parse_dates=["date"])
            else:
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection,
                                     params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                     parse_dates=["date"])
            return df
        except Exception as e:
            logger.error(f"Error loading weekly data for {symbol} between {start_date}-{end_date}: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.error("For weekly data, either date range (start_date, end_date) or tail_limit must be provided.")
        return pd.DataFrame()


def load_monthly_data(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
        tail_limit: Optional[int] = None,
        db_session: Optional[Session] = None
) -> pd.DataFrame:
    """
    从数据库加载月线数据 (stock_monthly 表)。
    与 load_weekly_data 类似，只是表名不同。
    :param symbol: 股票代码
    :param start_date: 起始日期 'YYYY-MM-DD' (月末最后交易日)
    :param end_date: 结束日期 'YYYY-MM-DD' (月末最后交易日)
    :param fields: 需要的字段列表，默认全字段
    :param tail_limit: 获取最近N条月线数据
    :param db_session: 可选的数据库会话
    :return: Pandas DataFrame
    """
    if tail_limit is not None and (start_date is not None or end_date is not None):
        raise ValueError("tail_limit cannot be used with start_date or end_date.")

    table_name = "stock_monthly"
    selected_fields = "*"
    if fields:
        if 'date' not in fields: fields.append('date')
        if 'symbol' not in fields: fields.append('symbol')
        selected_fields = ", ".join(list(set(fields)))

    conn_or_session = db_session if db_session else engine

    if tail_limit is not None:
        logger.info(f"Loading last {tail_limit} monthly data for {symbol} from {table_name}.")
        sql_query = text(f"""
          SELECT {selected_fields} 
          FROM {table_name}
          WHERE symbol = :symbol -- 注意：新表中 symbol 而非 symbol
          ORDER BY date DESC
          LIMIT :limit
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "limit": tail_limit}, parse_dates=["date"])
            else:
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection, params={"symbol": symbol, "limit": tail_limit},
                                     parse_dates=["date"])
            if df.empty:
                logger.warning(f"No monthly data found for symbol {symbol} within the last {tail_limit} months.")
                return pd.DataFrame()
            return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error loading last {tail_limit} monthly data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    elif start_date and end_date:
        logger.info(f"Loading monthly data for {symbol} from {start_date} to {end_date} from {table_name}.")
        sql_query = text(f"""
            SELECT {selected_fields} FROM {table_name}
            WHERE symbol = :symbol -- 注意：新表中 symbol 而非 symbol
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date
        """)
        try:
            if isinstance(conn_or_session, Session):
                df = pd.read_sql(sql_query, conn_or_session.connection(),
                                 params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                 parse_dates=["date"])
            else:
                with conn_or_session.connect() as connection:
                    df = pd.read_sql(sql_query, connection,
                                     params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                                     parse_dates=["date"])
            return df
        except Exception as e:
            logger.error(f"Error loading monthly data for {symbol} between {start_date}-{end_date}: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.error("For monthly data, either date range (start_date, end_date) or tail_limit must be provided.")
        return pd.DataFrame()