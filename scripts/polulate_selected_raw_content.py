# scripts/populate_selected_raw_content.py
import pandas as pd
import logging
import time
import random
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, not_  # 导入 sqlalchemy 的查询构造器
from typing import List, Optional
from datetime import datetime, timedelta

# 确保相对导入路径正确
import sys
import os

from db.database import get_db_session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import models  # 直接从 models 导入，因为我们将在这里构建查询
from data_processing import scraper  # scraper 用于抓取内容
from config.settings import settings  # 用于获取配置，如延迟时间

# 日志配置
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# dateutil 用于更精确的日期计算，如果不可用则使用 timedelta
try:
    from dateutil.relativedelta import relativedelta

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning(
        "python-dateutil not installed. Using timedelta for date calculations, which might be less accurate for 'years'.")


def get_target_disclosures_for_processing(db: Session, symbol: str) -> List[models.StockDisclosure]:
    """
    查找指定股票的、符合特定业务规则且 raw_content 为空的公告。
    (此函数包含了原 loader.get_disclosures_needing_content 的筛选逻辑)
    """
    logger.info(f"为股票 {symbol} 查询需要处理的公告 (筛选逻辑已内置)")
    try:
        now = datetime.now()
        one_year_ago_date = (now - timedelta(days=365)).date()

        if DATEUTIL_AVAILABLE:
            three_years_ago_date = (now - relativedelta(years=3)).date()
        else:
            three_years_ago_date = (now - timedelta(days=3 * 365 + 1)).date()  # 粗略计算

        annual_semi_keywords = ['年度报告', '半年度报告']
        other_relevant_keywords = ['调研', '股权激励', '回购']  # 您可以按需调整这些关键词

        filter_conditions = []

        # 条件组1：最近3年的年报/半年报
        for kw in annual_semi_keywords:
            filter_conditions.append(
                and_(
                    models.StockDisclosure.title.ilike(f'%{kw}%'),
                    models.StockDisclosure.ann_date >= three_years_ago_date
                )
            )

        # 条件组2：最近1年的其他相关关键词公告
        for kw in other_relevant_keywords:
            filter_conditions.append(
                and_(
                    models.StockDisclosure.title.ilike(f'%{kw}%'),
                    models.StockDisclosure.ann_date >= one_year_ago_date
                )
            )

        if not filter_conditions:
            logger.warning(f"股票 {symbol}: 未配置有效的关键词筛选条件。")
            return []

        combined_keyword_filters = or_(*filter_conditions)

        disclosures = db.query(models.StockDisclosure).filter(
            models.StockDisclosure.symbol == symbol,
            models.StockDisclosure.raw_content.is_(None),  # 核心：raw_content 为空
            not_(models.StockDisclosure.title.ilike('%摘要%')),  # 排除摘要
            combined_keyword_filters  # 应用关键词和日期组合筛选
        ).order_by(models.StockDisclosure.ann_date.desc()).all()

        logger.info(f"股票 {symbol}: 找到 {len(disclosures)} 条符合筛选条件且 raw_content 为空的公告。")
        return disclosures
    except Exception as e:
        logger.error(f"为股票 {symbol} 查询待处理公告时出错: {e}", exc_info=True)
        return []


def populate_raw_content_for_stock(db: Session, stock_symbol: str):
    """
    为指定股票获取并填充符合筛选条件的公告的 raw_content。
    """
    logger.info(f"开始为股票 {stock_symbol} 填充筛选后公告的 raw_content。")

    # 使用此脚本内部定义的筛选函数获取公告列表
    disclosures_to_process = get_target_disclosures_for_processing(db, symbol=stock_symbol)

    if not disclosures_to_process:
        logger.info(f"股票 {stock_symbol}: 无需填充 raw_content 的公告。")
        return

    processed_count = 0
    errors_count = 0

    for disclosure_obj in disclosures_to_process:  # disclosure_obj 是 StockDisclosure SQLAlchemy 对象
        logger.info(f"处理公告 ID {disclosure_obj.id}: {disclosure_obj.title} (URL: {disclosure_obj.url})")

        if not disclosure_obj.url:
            logger.warning(f"跳过公告 ID {disclosure_obj.id}，因为缺少 URL。")
            continue

        try:
            content = scraper.fetch_announcement_text(
                detail_url=disclosure_obj.url,
                title=disclosure_obj.title,
                tag=disclosure_obj.tag  # <--- 新增传递 tag 参数
            )

            if content:
                logger.debug(f"成功抓取公告 ID {disclosure_obj.id} 的内容，长度: {len(content)}。")
                disclosure_obj.raw_content = content  # 直接更新 SQLAlchemy 对象的属性
                db.add(disclosure_obj)  # 将已更改的对象添加到 session，以便后续 commit
                processed_count += 1
                logger.info(f"公告 ID {disclosure_obj.id} 的 raw_content 已在会话中准备更新。")
            else:
                logger.warning(f"未能为公告 ID {disclosure_obj.id} 提取内容 (URL: {disclosure_obj.url})。")
                errors_count += 1

        except Exception as e:
            errors_count += 1
            logger.error(f"处理公告 ID {disclosure_obj.id} (URL: {disclosure_obj.url}) 时发生严重错误: {e}",
                         exc_info=True)
            # 发生错误时，最好不要立即回滚，因为可能其他公告已成功处理。
            # 可以在整个股票处理完毕后，根据是否有错误决定是否回滚，或部分提交。
            # 当前设计是在外层（例如 sync_disclosure_data.py）处理整体事务。
            # 或者这里可以标记此条公告处理失败，但不影响其他。

        scrape_delay_min = getattr(settings, "SCRAPE_DELAY_MIN", 0.5)
        scrape_delay_max = getattr(settings, "SCRAPE_DELAY_MAX", 1.5)
        time.sleep(random.uniform(scrape_delay_min, scrape_delay_max))

    # 提交事务的责任交给调用者（例如 sync_disclosure_data.py 中的主循环）
    # 这样可以确保元数据写入和 raw_content 填充在同一个事务中，或者按需分开。
    # 如果这里要独立提交，可以取消注释下一行，但要注意与调用方的事务管理协调。
    # try:
    #     db.commit()
    #     logger.info(f"股票 {stock_symbol}: raw_content 填充事务已提交。成功: {processed_count}, 失败: {errors_count}")
    # except Exception as e_commit:
    #     logger.error(f"股票 {stock_symbol}: 提交 raw_content 填充事务时发生错误: {e_commit}", exc_info=True)
    #     db.rollback()
    #     logger.info(f"股票 {stock_symbol}: raw_content 填充事务已回滚。")

    if errors_count > 0:
        logger.warning(
            f"股票 {stock_symbol}: 完成 raw_content 填充尝试，其中有 {errors_count} 条公告处理失败或未获取到内容。")
    if processed_count > 0:
        logger.info(f"股票 {stock_symbol}: {processed_count} 条公告的 raw_content 已在会话中更新，等待提交。")


if __name__ == "__main__":
    logger.info("开始独立运行 populate_selected_raw_content.py 脚本...")

    # 使用 with 语句管理数据库会话
    try:
        with get_db_session() as db_s:  # <--- 修改此处
            # 1. 获取所有股票列表
            try:
                all_stocks_query = db_s.query(models.StockList.code.label("symbol"))
                stock_list_df = pd.read_sql_query(all_stocks_query.statement, db_s.bind)
                if stock_list_df.empty:
                    logger.warning("股票列表为空，无法进行批量填充。请先同步股票列表。")
                    sys.exit(0)
                logger.info(f"获取到 {len(stock_list_df)} 只股票，将逐个处理。")
            except Exception as e_stock_list:
                logger.error(f"获取股票列表失败: {e_stock_list}", exc_info=True)
                sys.exit(1)

            # 2. 循环处理每只股票
            total_stocks = len(stock_list_df)
            for index, row in stock_list_df.iterrows():
                current_symbol = row["symbol"]
                logger.info(f"--- 开始处理股票: {current_symbol} ({index + 1}/{total_stocks}) ---")

                populate_raw_content_for_stock(db_s, stock_symbol=current_symbol)

                try:
                    db_s.commit()
                    logger.info(f"股票 {current_symbol} 的 raw_content 更新已提交。")
                except Exception as e_commit_stock:
                    logger.error(f"提交股票 {current_symbol} 的 raw_content 更新时发生错误: {e_commit_stock}",
                                 exc_info=True)
                    db_s.rollback()
                    logger.info(f"股票 {current_symbol} 的 raw_content 更新已回滚。")

            logger.info("所有股票的 raw_content 填充处理完毕。")

    except Exception as e_main:
        logger.error(f"populate_selected_raw_content.py 独立运行过程中发生未处理的错误: {e_main}", exc_info=True)
        # 如果 with 块内部发生异常，通常 SQLAlchemy 的 session 在退出 with 块时会自动回滚 (如果配置如此)
        # 并且 session 会被关闭。

    logger.info("populate_selected_raw_content.py 脚本执行结束。")  # 移动到 try/except 外部，表示脚本最终结束