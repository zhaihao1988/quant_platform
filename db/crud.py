# db/crud.py
import logging
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import asc, text, func, extract, desc
from typing import List, Optional, Dict, Any
from datetime import date # <--- *** 在这里添加这一行 ***
import pandas as pd # <--- *** 确保 pandas 也已导入，因为函数返回 Optional[pd.DataFrame] ***
from sqlalchemy.exc import SQLAlchemyError

from db.database import SessionLocal
# Use correct path for models
from .models import (
    StockDisclosure,
    StockList,
    StockFinancial, # 在您的 models.py 中，表名是 stock_financial_reports，类名是 StockFinancial
    StockDaily,
    StockDisclosureChunk,
    StockWeekly,
    StockMonthly,
    StockShareDetail  # <-- 1. 导入新的 StockShareDetail 模型
)
# Import the centralized embedding function
from core.vectorizer import get_embedding
from sqlalchemy.dialects.postgresql import insert as pg_insert
logger = logging.getLogger(__name__)

# Remove redundant embedding model loading from here

def get_stock_list_info(db: Session, symbol: str) -> Optional[StockList]:
    """Gets basic stock information (name, industry) from the StockList table."""
    logger.debug(f"Querying StockList for symbol: {symbol}")
    try:
        stock_info = db.query(StockList).filter(StockList.code == symbol).first()
        if not stock_info:
            logger.warning(f"No entry found in stock_list for code: {symbol}")
        return stock_info
    except Exception as e:
        logger.error(f"Error getting stock list info for {symbol}: {e}", exc_info=True)
        return None
def save_disclosure_chunk(db: Session, disclosure_id: int, chunk_order: int, chunk_text: str, vector: List[float]) -> bool:
    """Saves a single disclosure chunk and its vector."""
    try:
        db_chunk = StockDisclosureChunk(
            disclosure_id=disclosure_id,
            chunk_order=chunk_order,
            chunk_text=chunk_text,
            chunk_vector=vector
        )
        db.add(db_chunk)
        # db.commit() # 通常在调用者那里统一 commit
        logger.debug(f"Added chunk {chunk_order} for disclosure {disclosure_id} to session.")
        return True
    except Exception as e:
        # db.rollback() # 通常在调用者那里统一 rollback
        logger.error(f"Error saving disclosure chunk (Disclosure ID: {disclosure_id}, Order: {chunk_order}): {e}", exc_info=True)
        return False
def retrieve_relevant_disclosures(db: Session, symbol: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves relevant disclosure CHUNKS from the knowledge base using vector similarity search.
    Returns a list of dictionaries containing chunk info and original disclosure metadata.
    """
    logger.info(f"Retrieving relevant disclosure chunks for symbol '{symbol}' with query: '{query_text[:50]}...'")
    query_embedding = get_embedding(query_text, is_query=True)
    if not query_embedding:
        logger.error("Failed to generate query embedding.")
        return []

    try:
        # 查询 chunk 表，并通过 relationship 加载关联的 disclosure 信息
        similar_chunks = db.query(StockDisclosureChunk).options(
            joinedload(StockDisclosureChunk.disclosure) # 预加载关联的公告信息
        ).join(StockDisclosure).filter( # 确保只查询指定 symbol 的公告块
             StockDisclosure.symbol == symbol,
             StockDisclosureChunk.chunk_vector != None
        ).order_by(
            StockDisclosureChunk.chunk_vector.cosine_distance(query_embedding)
        ).limit(top_k).all()

        results = []
        if similar_chunks:
             logger.info(f"Found {len(similar_chunks)} relevant disclosure chunks in KB for query.")
             for chunk in similar_chunks:
                  results.append({
                       "chunk_text": chunk.chunk_text,
                       "chunk_order": chunk.chunk_order,
                       "disclosure_id": chunk.disclosure_id,
                       "title": chunk.disclosure.title, # 从关联对象获取
                       "ann_date": chunk.disclosure.ann_date # 从关联对象获取
                  })
        else:
             logger.info("No relevant disclosure chunks found.")

        return results

    except Exception as e:
        logger.error(f"Error during chunk vector search for {symbol}: {e}", exc_info=True)
        db.rollback() # <--- 发生查询错误时回滚
        return []


def get_all_stocks(db: Session) -> List[StockList]:
    """获取 stock_list 表中的所有股票基本信息。"""
    try:
        return db.query(StockList).all()
    except Exception as e:
        logger.error(f"获取所有股票列表失败: {e}", exc_info=True)
        return []

# --- 新增函数 ---
def get_all_symbols_from_stocklist(db: Session) -> List[str]:
    """
    从 StockList 表获取所有有效的股票代码字符串列表。
    """
    logger.debug("正在从 StockList 表查询所有股票代码...")
    try:
        # 查询 StockList 表的 code 列
        # 过滤掉 code 为 None 或空字符串的情况，确保代码有效
        results = db.query(StockList.code).filter(StockList.code.isnot(None), StockList.code != '').all()
        # 从结果元组中提取股票代码字符串
        symbols = [result[0] for result in results if result[0]] # 确保 result[0] (即code) 不是None或空
        logger.info(f"从 StockList 表获取到 {len(symbols)} 个有效股票代码。")
        return symbols
    except SQLAlchemyError as e:
        logger.error(f"查询 StockList 所有股票代码时发生数据库错误: {e}", exc_info=True)
        return [] # 发生错误时返回空列表
    except Exception as e: # 捕获其他可能的未知错误
        logger.error(f"查询 StockList 所有股票代码时发生未知错误: {e}", exc_info=True)
        return []
# --- 新增函数结束 ---
def get_stock_daily_data_period(db: Session, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    获取指定股票在给定日期范围内的日线行情数据。
    返回一个 Pandas DataFrame，列应包含 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover'。
    数据按日期升序排列。
    """
    logger.debug(f"正在为股票 {symbol} 查询从 {start_date.isoformat()} 到 {end_date.isoformat()} 的日线数据...")
    try:
        query = (
            db.query(
                StockDaily.date,
                StockDaily.open,  # <--- 确认模型字段名
                StockDaily.high,  # <--- 确认模型字段名
                StockDaily.low,  # <--- 确认模型字段名
                StockDaily.close,  # <--- 确认模型字段名
                StockDaily.volume,
                StockDaily.turnover,
                StockDaily.amount,
                StockDaily.amplitude,  # <--- 新增 (根据您的模型)
                StockDaily.pct_change,  # <--- 新增 (根据您的模型)
                StockDaily.price_change  # <--- 新增 (根据您的模型)
            )
            .filter(StockDaily.symbol == symbol)  # <--- 使用 StockDaily.symbol
            .filter(StockDaily.date >= start_date)
            .filter(StockDaily.date <= end_date)
            .order_by(asc(StockDaily.date))
        )

        df = pd.read_sql_query(query.statement, db.bind)  # type: ignore

        if df.empty:
            logger.warning(f"股票 {symbol} 在日期范围 {start_date.isoformat()} 到 {end_date.isoformat()} 内无日线数据。")
            return None

        # 确保 'date' 列是 datetime 类型 (如果从数据库取出时不是)
        # pd.read_sql_query 通常会自动处理日期类型，但可以加一道保险
        df['date'] = pd.to_datetime(df['date'])

        logger.debug(f"成功获取股票 {symbol} 的 {len(df)} 条日线数据。")
        return df

    except Exception as e:
        logger.error(f"为股票 {symbol} 获取日线数据时出错: {e}", exc_info=True)
        return None


def get_stock_daily_for_date(db: Session, symbol: str, trade_date: date) -> Optional[StockDaily]:
    """获取特定股票在特定日期的日线数据记录 (StockDaily 对象)。"""
    logger.debug(f"正在为股票 {symbol} 查询日期为 {trade_date.isoformat()} 的日线数据...")
    try:
        return db.query(StockDaily).filter_by(symbol=symbol, date=trade_date).first()
    except Exception as e:
        logger.error(f"为股票 {symbol} 在日期 {trade_date.isoformat()} 获取单日日线数据时出错: {e}", exc_info=True)
        return None


# --- FundamentalAnalyzer 所需的更复杂的财务数据获取逻辑 ---
# FundamentalAnalyzer 需要获取特定类型的财报（例如年报、最新季报、去年同期季报）。
# 您现有的 data_processing/loader.py 中的 load_multiple_financial_reports 函数
# 包含了这类逻辑。理想情况下，这类逻辑也应该封装在 crud.py 中，或者
# FundamentalAnalyzer 直接调用 loader.py 中的函数（确保它能接收 db_session）。

def get_financial_reports_for_analyzer(
    db: Session,
    symbol: str,
    report_type_db_key: str, # 例如 'benefit' (利润表), 'balance' (资产负债表)
    current_signal_date: date
) -> Dict[str, Optional[StockFinancial] | List[StockFinancial]]:
    """
    为 FundamentalAnalyzer 获取特定需求的财务报告。

    返回一个字典，包含:
    - 'latest_annual_report': 最新年度报告 (StockFinancial 对象或 None)。
    - 'latest_quarterly_report': 最新一期（任何类型，包括年报、季报、中报）报告 (StockFinancial 对象或 None)。
    - 'previous_year_same_quarter_report': 对应最新一期的去年同期报告 (StockFinancial 对象或 None)。
    - 'last_3_annual_reports': 最近三个年度报告的列表 (List[StockFinancial]，可能少于3个如果数据不足)。
    """
    reports = {
        'latest_annual_report': None,
        'latest_quarterly_report': None,
        'previous_year_same_quarter_report': None,
        'last_3_annual_reports': [],
    }
    logger.debug(f"Fetching financial reports for analyzer: {symbol}, type: {report_type_db_key}, signal_date: {current_signal_date}")

    try:
        # 1. 获取所有相关的财务报告记录 (按报告日期降序)
        all_reports_query = db.query(StockFinancial).filter(
            StockFinancial.symbol == symbol,
            StockFinancial.report_type == report_type_db_key,
            StockFinancial.report_date <= current_signal_date # 确保报告日期不晚于信号日
        ).order_by(desc(StockFinancial.report_date))

        all_stock_reports: List[StockFinancial] = all_reports_query.all()

        if not all_stock_reports:
            logger.warning(f"No financial reports found for {symbol} with type {report_type_db_key} up to {current_signal_date}.")
            return reports

        # 2. 查找最新年度报告
        for report in all_stock_reports:
            if report.report_date.month == 12 and report.report_date.day == 31:
                reports['latest_annual_report'] = report
                logger.debug(f"Found latest annual report: {report.report_date} for {symbol}")
                break

        # 3. 查找最新一期报告 (可能是年报、中报或季报)
        if all_stock_reports:
            reports['latest_quarterly_report'] = all_stock_reports[0]
            latest_quarterly_report = all_stock_reports[0]
            logger.debug(f"Found latest quarterly/any report: {latest_quarterly_report.report_date} for {symbol}")

            # 4. 查找对应最新一期的去年同期报告
            if latest_quarterly_report:
                target_prev_year = latest_quarterly_report.report_date.year - 1
                target_month = latest_quarterly_report.report_date.month
                target_day = latest_quarterly_report.report_date.day # 通常季报的日期是固定的

                # 筛选去年同期的报告
                # 为了更精确匹配，如果目标日期不存在（例如闰年2月29日），可以稍微放宽日期匹配
                # 但通常财务报告日期是固定的（如03-31, 06-30, 09-30, 12-31）
                for report in all_stock_reports:
                    if report.report_date.year == target_prev_year and \
                       report.report_date.month == target_month and \
                       report.report_date.day == target_day:
                        reports['previous_year_same_quarter_report'] = report
                        logger.debug(f"Found previous year same quarter report: {report.report_date} for {symbol}")
                        break
                if not reports['previous_year_same_quarter_report']:
                    logger.warning(f"Could not find previous year same quarter report for latest: {latest_quarterly_report.report_date} for {symbol}")


        # 5. 查找最近三个年度报告
        annual_reports_found = []
        for report in all_stock_reports:
            if report.report_date.month == 12 and report.report_date.day == 31:
                annual_reports_found.append(report)
                if len(annual_reports_found) >= 3:
                    break
        reports['last_3_annual_reports'] = annual_reports_found
        logger.debug(f"Found {len(annual_reports_found)} annual reports for 3-year profit check for {symbol}")
        if len(annual_reports_found) < 3:
             logger.warning(f"Found only {len(annual_reports_found)} annual reports (less than 3) for {symbol}")


    except Exception as e:
        logger.error(f"Error fetching financial reports for analyzer ({symbol}, {report_type_db_key}): {e}", exc_info=True)
        # 返回部分获取的数据或空数据
        return reports

    return reports

def get_all_symbols_from_stocklist(db: Session) -> List[str]: # 新增的函数
    """从 StockList 表获取所有有效的股票代码字符串列表."""
    logger.debug("正在从 StockList 表查询所有股票代码...")
    try:
        results = db.query(StockList.code).filter(StockList.code.isnot(None), StockList.code != '').all()
        symbols = [result[0] for result in results if result[0]]
        logger.info(f"从 StockList 表获取到 {len(symbols)} 个有效股票代码。")
        return symbols
    except SQLAlchemyError as e:
        logger.error(f"查询 StockList 所有股票代码时发生数据库错误: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"查询 StockList 所有股票代码时发生未知错误: {e}", exc_info=True)
        return []

def get_stock_daily_data_period(db: Session, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """获取指定股票在给定日期范围内的日线行情数据。"""
    logger.debug(f"正在为股票 {symbol} 查询从 {start_date.isoformat()} 到 {end_date.isoformat()} 的日线数据...")
    try:
        query = (
            db.query(
                StockDaily.date, StockDaily.open, StockDaily.high, StockDaily.low,
                StockDaily.close, StockDaily.volume, StockDaily.turnover, StockDaily.amount,
                StockDaily.amplitude, StockDaily.pct_change, StockDaily.price_change
            )
            .filter(StockDaily.symbol == symbol)
            .filter(StockDaily.date >= start_date)
            .filter(StockDaily.date <= end_date)
            .order_by(asc(StockDaily.date))
        )
        df = pd.read_sql_query(query.statement, db.bind)
        if df.empty:
            logger.warning(f"股票 {symbol} 在日期范围 {start_date.isoformat()} 到 {end_date.isoformat()} 内无日线数据。")
            return None
        df['date'] = pd.to_datetime(df['date']).dt.date # 确保是 date 对象
        return df
    except Exception as e:
        logger.error(f"为股票 {symbol} 获取日线数据时出错: {e}", exc_info=True)
        return None

# --- 新增或完善 get_stock_weekly_data_period ---
def get_stock_weekly_data_period(db: Session, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """获取指定股票在给定日期范围内的周线行情数据。"""
    logger.debug(f"正在为股票 {symbol} 查询从 {start_date.isoformat()} 到 {end_date.isoformat()} 的周线数据...")
    try:
        # 假设 StockWeekly 模型有 open, high, low, close, volume, turnover, date, symbol 字段
        query = (
            db.query(
                StockWeekly.date, StockWeekly.open, StockWeekly.high, StockWeekly.low,
                StockWeekly.close, StockWeekly.volume, StockWeekly.turnover
                # 如果还有 amount 或其他 StockWeekly 特有字段，也应加入
            )
            .filter(StockWeekly.symbol == symbol) # 假设 StockWeekly 使用 symbol 字段
            .filter(StockWeekly.date >= start_date)
            .filter(StockWeekly.date <= end_date)
            .order_by(asc(StockWeekly.date))
        )
        df = pd.read_sql_query(query.statement, db.bind)
        if df.empty:
            logger.warning(f"股票 {symbol} 在日期范围 {start_date.isoformat()} 到 {end_date.isoformat()} 内无周线数据。")
            return None
        df['date'] = pd.to_datetime(df['date']).dt.date # 确保是 date 对象
        return df
    except Exception as e:
        logger.error(f"为股票 {symbol} 获取周线数据时出错: {e}", exc_info=True)
        return None

# --- 完善 get_stock_monthly_data_period (基于您已有的 get_stock_monthly_data) ---
def get_stock_monthly_data_period(db: Session, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """获取指定股票在给定日期范围内的月线行情数据。"""
    logger.debug(f"正在为股票 {symbol} 查询从 {start_date.isoformat()} 到 {end_date.isoformat()} 的月线数据...")
    try:
        # 假设 StockMonthly 模型有 open, high, low, close, volume, turnover, date, symbol 字段
        query = (
            db.query(
                StockMonthly.date, StockMonthly.open, StockMonthly.high, StockMonthly.low,
                StockMonthly.close, StockMonthly.volume, StockMonthly.turnover
                # 如果还有 amount 或其他 StockMonthly 特有字段，也应加入
            )
            .filter(StockMonthly.symbol == symbol) # 假设 StockMonthly 使用 symbol 字段
            .filter(StockMonthly.date >= start_date)
            .filter(StockMonthly.date <= end_date)
            .order_by(asc(StockMonthly.date))
        )
        df = pd.read_sql_query(query.statement, db.bind)
        if df.empty:
            logger.warning(f"股票 {symbol} 在日期范围 {start_date.isoformat()} 到 {end_date.isoformat()} 内无月线数据。")
            return None
        df['date'] = pd.to_datetime(df['date']).dt.date # 确保是 date 对象
        return df
    except Exception as e:
        logger.error(f"为股票 {symbol} 获取月线数据时出错: {e}", exc_info=True)
        return None
def bulk_upsert_stock_weekly(db: Session, weekly_data: List[Dict[str, Any]]):
    """
    批量插入或更新 StockWeekly 数据 (使用 PostgreSQL 的 ON CONFLICT DO UPDATE)。
    如果基于 symbol 和 date 的记录已存在，则更新；否则插入新记录。
    """
    if not weekly_data:
        logger.info("没有周线数据需要更新/插入。")
        return

    table = StockWeekly.__table__
    logger.info(f"准备批量 Upsert {len(weekly_data)} 条周线数据到 {table.name}...")

    try:
        # 构建 insert 语句
        stmt = pg_insert(table).values(weekly_data)

        # 定义冲突时的更新操作
        # index_elements 指定了唯一约束的列 (在您的模型中是 'symbol' 和 'date')
        # set_ 中的键是要更新的列，值是使用 excluded 对象来引用导致冲突的待插入数据的值
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in table.c  # table.c 包含了表的所有列对象
            if col.name not in ['id', 'symbol', 'date']  # 不更新主键和唯一约束键本身
        }

        if not update_dict:
            # 如果除了唯一键和主键外没有其他列了（不太可能，但作为保险）
            # 或者您明确只想插入不想更新已存在的，可以使用 on_conflict_do_nothing()
            # on_conflict_stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'date'])
            logger.warning(f"表 {table.name} 除了唯一键外没有其他列可更新，将仅尝试插入。")
            # 如果确实是这种情况，可能需要调整模型或这里的逻辑
            # 为了安全，我们还是执行一个空的更新，或者可以改成 do_nothing
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],
                set_={}  # 空更新，效果类似 DO NOTHING 但会返回影响行数
            )
        else:
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],  # 确保这些列上有唯一索引
                set_=update_dict
            )

        db.execute(on_conflict_stmt)
        db.commit()
        logger.info(f"StockWeekly 批量 Upsert 操作完成。")

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"批量 Upsert StockWeekly 数据失败: {e}", exc_info=True)
    except Exception as e:  # 捕获其他潜在错误
        db.rollback()
        logger.error(f"StockWeekly 批量 Upsert 发生未知错误: {e}", exc_info=True)
def get_latest_stock_weekly_date(db: Session, symbol: str) -> Optional[date]:
    """获取某支股票在 StockWeekly 表中最新的日期"""
    try:
        latest_date = db.query(func.max(StockWeekly.date)).filter(StockWeekly.symbol == symbol).scalar()
        return latest_date
    except Exception as e:
        logger.error(f"获取 StockWeekly 最新日期失败 (symbol={symbol}): {e}", exc_info=True)
        return None

# --- CRUD for StockMonthly (与 StockWeekly 非常类似) ---

def bulk_upsert_stock_monthly(db: Session, monthly_data: List[Dict[str, Any]]):
    """
    批量插入或更新 StockMonthly 数据 (使用 PostgreSQL 的 ON CONFLICT DO UPDATE)。
    """
    if not monthly_data:
        logger.info("没有月线数据需要更新/插入。")
        return

    table = StockMonthly.__table__
    logger.info(f"准备批量 Upsert {len(monthly_data)} 条月线数据到 {table.name}...")

    try:
        stmt = pg_insert(table).values(monthly_data)

        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in table.c
            if col.name not in ['id', 'symbol', 'date']
        }

        if not update_dict:
            logger.warning(f"表 {table.name} 除了唯一键外没有其他列可更新，将仅尝试插入。")
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],
                set_={}
            )
        else:
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],  # 确保这些列上有唯一索引
                set_=update_dict
            )

        db.execute(on_conflict_stmt)
        db.commit()
        logger.info(f"StockMonthly 批量 Upsert 操作完成。")

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"批量 Upsert StockMonthly 数据失败: {e}", exc_info=True)
    except Exception as e:
        db.rollback()
        logger.error(f"StockMonthly 批量 Upsert 发生未知错误: {e}", exc_info=True)

def get_stock_monthly_data(db: Session, symbol: str, start_date: Optional[date] = None, end_date: Optional[date] = None, tail_limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    获取指定股票的月线数据。
    """
    logger.debug(f"查询 StockMonthly: symbol={symbol}, start={start_date}, end={end_date}, limit={tail_limit}")
    try:
        query = db.query(StockMonthly).filter(StockMonthly.symbol == symbol)
        if start_date:
            query = query.filter(StockMonthly.date >= start_date)
        if end_date:
            query = query.filter(StockMonthly.date <= end_date)

        if tail_limit is not None:
            query = query.order_by(desc(StockMonthly.date)).limit(tail_limit)
            df = pd.read_sql_query(query.statement, db.bind, parse_dates=["date"])
            if not df.empty:
                return df.iloc[::-1].reset_index(drop=True) # 保持时间升序
            return df # 返回空DataFrame
        else:
            query = query.order_by(asc(StockMonthly.date))
            df = pd.read_sql_query(query.statement, db.bind, parse_dates=["date"])
            return df

    except Exception as e:
        logger.error(f"获取 StockMonthly 数据失败 (symbol={symbol}): {e}", exc_info=True)
        return None

def get_latest_stock_monthly_date(db: Session, symbol: str) -> Optional[date]:
    """获取某支股票在 StockMonthly 表中最新的日期"""
    try:
        latest_date = db.query(func.max(StockMonthly.date)).filter(StockMonthly.symbol == symbol).scalar()
        return latest_date
    except Exception as e:
        logger.error(f"获取 StockMonthly 最新日期失败 (symbol={symbol}): {e}", exc_info=True)
        return None


def get_stock_share_detail(db: Session, symbol: str) -> Optional[StockShareDetail]:
    """
    从 stock_share_details 表获取特定股票的最新股本明细信息。
    """
    logger.debug(f"正在查询股票 {symbol} 的股本明细 (StockShareDetail)...")
    try:
        # 由于 symbol 在 StockShareDetail 中是唯一的，直接查询即可
        share_detail = db.query(StockShareDetail).filter(StockShareDetail.symbol == symbol).first()
        if not share_detail:
            logger.warning(f"在 stock_share_details 表中未找到股票代码 {symbol} 的记录。")
        return share_detail
    except SQLAlchemyError as e:
        logger.error(f"查询 StockShareDetail (symbol={symbol}) 时发生数据库错误: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"查询 StockShareDetail (symbol={symbol}) 时发生未知错误: {e}", exc_info=True)
        return None


def update_stock_share_detail(db: Session, symbol: str, data: Dict[str, Any]) -> Optional[StockShareDetail]:
    """
    更新或插入股票的股本明细信息到 stock_share_details 表。
    'data' 参数是一个字典，包含了从 akshare 解析后的字段和值。
    例如: {'stock_name': '平安银行', 'total_shares': 19405918198, ...}
    """
    logger.debug(f"准备为股票 {symbol} 更新/插入股本明细数据: {data}")

    # 准备要插入或更新的值
    # 我们需要确保 data 字典中的键名与 StockShareDetail 模型的列名匹配
    # 并且处理好数据类型转换 (akshare 返回的可能是字符串或需要清洗的 float)

    # 从传入的 data 字典中提取并转换值
    # 这个 data 字典应该是调用此函数前，已经从 akshare 返回的 DataFrame 解析好的
    insert_data = {
        "symbol": symbol,
        "stock_name": data.get("stock_name"),
        "total_shares": data.get("total_shares"),
        "float_shares": data.get("float_shares"),
        "total_market_cap": data.get("total_market_cap"),
        "float_market_cap": data.get("float_market_cap"),
        "industry": data.get("industry"),
        "listing_date": data.get("listing_date"),  # 应为 date 对象
        "data_source_date": data.get("data_source_date", date.today())  # 如果未提供，默认为今天
        # updated_at 会自动处理
    }

    # 移除值为 None 的键，以便数据库可以使用其默认值（如果定义了）
    # 或者让 SQLAlchemy 处理空值如何映射到数据库的 NULL
    # insert_data = {k: v for k, v in insert_data.items() if v is not None}

    # 使用 PostgreSQL的 ON CONFLICT DO UPDATE (upsert) 功能
    # statement 中的键应该是数据库表的列名
    stmt = pg_insert(StockShareDetail).values(**insert_data)

    # 定义在冲突时要更新的字段
    # 这里我们假设如果 symbol 已存在，则更新所有传入的字段
    # 确保这里的键名与 StockShareDetail 模型的属性名一致
    update_columns = {
        name: getattr(stmt.excluded, name)
        for name, column in StockShareDetail.__table__.columns.items()
        if name not in ["id", "symbol"]  # id 和 symbol (冲突目标) 不应在 update 部分
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=[StockShareDetail.symbol],  # 冲突目标
        set_=update_columns
    ).returning(StockShareDetail)  # 返回插入或更新后的行

    try:
        result = db.execute(stmt)
        db.commit()
        updated_row = result.fetchone()
        if updated_row:
            logger.info(f"股票 {symbol} 的股本明细已成功更新/插入。")
            # 将 RowProxy 转换为 ORM 对象 (如果需要与其他 ORM 对象交互)
            # 或者直接使用 updated_row 中的值
            # 这里我们假设直接返回 ORM 对象更方便
            return db.query(StockShareDetail).filter(StockShareDetail.symbol == symbol).first()
        else:
            logger.error(f"执行 upsert 后未能获取股票 {symbol} 的返回行。")
            return None

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"更新/插入 StockShareDetail (symbol={symbol}) 时发生数据库错误: {e}", exc_info=True)
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"更新/插入 StockShareDetail (symbol={symbol}) 时发生未知错误: {e}", exc_info=True)
        return None
def main():
    """
    主函数，用于测试 get_stock_daily_for_date 功能。
    """
    db: Session = SessionLocal() # 获取数据库会话

    symbol_to_test = "601083"
    date_to_test = date(2025, 5, 30)

    logger.info(f"开始测试 get_stock_daily_for_date 函数，股票代码: {symbol_to_test}, 日期: {date_to_test.isoformat()}")

    try:
        stock_daily_data = get_stock_daily_for_date(db, symbol_to_test, date_to_test)

        if stock_daily_data:
            logger.info(f"成功获取到股票 {symbol_to_test} 在 {date_to_test.isoformat()} 的日线数据:")
            logger.info(f"  股票代码 (Symbol): {stock_daily_data.symbol}")
            logger.info(f"  日期 (Date): {stock_daily_data.date}")
            logger.info(f"  开盘价 (Open): {stock_daily_data.open}")
            logger.info(f"  收盘价 (Close): {stock_daily_data.close}")
            logger.info(f"  最高价 (High): {stock_daily_data.high}")
            logger.info(f"  最低价 (Low): {stock_daily_data.low}")
            logger.info(f"  成交量 (Volume): {stock_daily_data.volume}")
            # 您可以根据 StockDaily模型的实际字段打印更多信息
        else:
            logger.warning(f"未能获取到股票 {symbol_to_test} 在 {date_to_test.isoformat()} 的日线数据。请检查数据库中是否存在该记录。")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    finally:
        logger.info("测试完成。关闭数据库会话。")
        db.close() # 关闭会话

if __name__ == "__main__":
    main()