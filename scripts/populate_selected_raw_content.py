import sys
import os
import logging
import time
import random
import json
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import and_, or_, not_
from datetime import datetime, timedelta

# --- Start of permanent fix ---
# Goal: Make this script runnable from anywhere by adding project root to sys.path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db import models, get_db_session
from data_processing import scraper
from config import settings
# --- End of permanent fix ---


try:
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

def populate_raw_content_for_stock(db: Session, stock_symbol: str):
    """
    为指定股票获取并填充符合筛选条件的公告的 raw_content。
    根据公告类型（年报/半年报、调研活动）应用不同的解析逻辑。
    """
    logger.info(f"开始为股票 {stock_symbol} 填充筛选后公告的 raw_content。")

    disclosures_to_process = get_target_disclosures_for_processing(db, symbol=stock_symbol)

    if not disclosures_to_process:
        logger.info(f"股票 {stock_symbol}: 无需填充 raw_content 的公告。")
        return

    processed_count = 0
    errors_count = 0

    for disclosure_obj in disclosures_to_process:
        logger.info(f"处理公告 ID {disclosure_obj.id}: {disclosure_obj.title} (URL: {disclosure_obj.url})")

        if not disclosure_obj.url:
            logger.warning(f"跳过公告 ID {disclosure_obj.id}，因为缺少 URL。")
            continue

        try:
            # 步骤1：先获取公告全文
            full_text = scraper.fetch_announcement_text(
                detail_url=disclosure_obj.url,
                title=disclosure_obj.title,
                tag=disclosure_obj.tag
            )

            if not full_text:
                logger.warning(f"未能为公告 ID {disclosure_obj.id} 抓取到任何文本。")
                errors_count += 1
                continue

            # 步骤2：根据类型应用不同解析逻辑
            content_to_store = None
            title = disclosure_obj.title
            tag = disclosure_obj.tag if disclosure_obj.tag else "" # 确保 tag 是字符串

            if '调研' in tag or '调研' in title:
                logger.info("检测到调研活动，使用AI提取Q&A...")
                # 使用我们确认好用的AI Q&A提取
                qa_list = scraper.extract_qa_with_ai(full_text)
                if qa_list:
                    # 将Q&A列表格式化为JSON字符串进行存储
                    content_to_store = json.dumps(qa_list, ensure_ascii=False, indent=4)
                    logger.info(f"成功提取到 {len(qa_list)} 条Q&A。")
                else:
                    logger.warning("AI未能提取到Q&A内容，将使用全文作为备用。")
                    content_to_store = full_text
            
            elif '年度报告' in title or '半年度报告' in title:
                logger.info("检测到年报/半年报，提取【管理层讨论与分析】...")
                # 使用我们之前确认好用的传统方法
                narrative_section = scraper.extract_section_from_text(full_text, "管理层讨论与分析")
                if narrative_section:
                    logger.info("成功提取章节，开始清理表格...")
                    content_to_store = scraper.remove_tables(narrative_section)
                else:
                    logger.warning("未能提取到'管理层讨论与分析'，将使用全文作为备用。")
                    content_to_store = full_text
            
            else: # 其他类型
                logger.info(f"常规公告 '{title}'，不提取内容，将跳过。")
                # content_to_store 保持为 None，因此不会更新数据库
                pass

            # 步骤3：更新数据库对象
            if content_to_store:
                logger.debug(f"公告 ID {disclosure_obj.id} 的内容已处理，最终长度: {len(content_to_store)}。")
                disclosure_obj.raw_content = content_to_store
                db.add(disclosure_obj)
                processed_count += 1
                logger.info(f"公告 ID {disclosure_obj.id} 的 raw_content 已在会话中准备更新。")
            else:
                logger.error(f"公告 ID {disclosure_obj.id} 未能生成任何可存储内容。")
                errors_count += 1

        except Exception as e:
            errors_count += 1
            logger.error(f"处理公告 ID {disclosure_obj.id} (URL: {disclosure_obj.url}) 时发生严重错误: {e}", exc_info=True)

        # 添加延迟，防止请求过于频繁
        scrape_delay_min = getattr(settings, "SCRAPE_DELAY_MIN", 0.5)
        scrape_delay_max = getattr(settings, "SCRAPE_DELAY_MAX", 1.5)
        time.sleep(random.uniform(scrape_delay_min, scrape_delay_max))

    if errors_count > 0:
        logger.warning(f"股票 {stock_symbol}: 完成 raw_content 填充尝试，其中有 {errors_count} 条公告处理失败或未获取到内容。")
    if processed_count > 0:
        logger.info(f"股票 {stock_symbol}: {processed_count} 条公告的 raw_content 已在会话中更新，等待提交。")

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
        other_relevant_keywords = ['调研'] # 只处理调研活动，不再捞取然后丢弃其他类型

        # 定义不想要的公告标题关键词
        exclusion_keywords = [
            '摘要', '自愿', '取消', '监管', '意见',
            '函', '督导', '提示', '审核'
        ]

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

        # 基础查询
        query = db.query(models.StockDisclosure).filter(
            models.StockDisclosure.symbol == symbol,
            # models.StockDisclosure.raw_content.is_(None),
            combined_keyword_filters
        )

        # 循环添加排除条件
        for keyword in exclusion_keywords:
            query = query.filter(not_(models.StockDisclosure.title.ilike(f'%{keyword}%')))

        disclosures = query.order_by(models.StockDisclosure.ann_date.desc()).all()

        logger.info(f"股票 {symbol}: 找到 {len(disclosures)} 条符合筛选条件且 raw_content 为空的公告。")
        return disclosures
    except Exception as e:
        logger.error(f"为股票 {symbol} 查询待处理公告时出错: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("脚本启动，开始全量刷新公告内容...")

    # 使用 'with' 语句来正确管理数据库会话
    # 它会自动处理 session 的创建、提交、回滚和关闭
    with get_db_session() as db:
        if db is None:
            logger.error("无法获取数据库会话，脚本中止。")
            exit() # or return

        # 从数据库动态获取所有不重复的股票代码
        stocks_to_process = []
        try:
            logger.info("正在从数据库查询所有不重复的股票代码...")
            all_stocks_query = db.query(models.StockDisclosure.symbol).distinct()
            stocks_to_process = [item[0] for item in all_stocks_query.all()]
            logger.info(f"查询到 {len(stocks_to_process)} 只不重复的股票，将开始处理。")
        except Exception as e:
            logger.error(f"从数据库查询股票列表时出错，脚本中止: {e}", exc_info=True)
            exit()

        if not stocks_to_process:
            logger.warning("数据库中没有找到任何股票，脚本结束。")
            exit()

        for i, symbol in enumerate(stocks_to_process, 1):
            try:
                logger.info(f"--- 处理进度: {i}/{len(stocks_to_process)}，股票代码: {symbol} ---")
                populate_raw_content_for_stock(db, stock_symbol=symbol)
                db.commit() # 每处理完一只股票就提交一次，避免单个错误导致全部回滚
                logger.info(f"股票 {symbol} 处理成功并已提交事务。")
            except Exception as e:
                logger.error(f"处理股票 {symbol} 时发生严重错误，正在回滚该股票的更改并继续处理下一只股票: {e}", exc_info=True)
                db.rollback() # 回滚当前失败股票的事务

    logger.info("脚本执行完毕。") 