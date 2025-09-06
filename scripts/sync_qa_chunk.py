# scripts/sync_qa_chunk.py
import logging
import time
import io
import pdfplumber
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

# 设置工作目录为项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from sqlalchemy.dialects.postgresql import insert

from db.database import SessionLocal
from db.models import StockDisclosure, StockDisclosureChunk, StockList
from data_processing.scraper import (
    stealth_download_pdf,
    identify_common_lines,
    clean_text_with_blacklist,
    extract_qa_with_ai,
    try_construct_pdf_url
)
from core.vectorizer import get_embedding

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_qa_chunk.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_last_qa_sync_date(db: Session) -> Optional[datetime.date]:
    """获取 StockDisclosureChunk 表中最新的 QA 类型记录的公告日期"""
    latest_date = db.query(
        func.max(StockDisclosureChunk.ann_date)
    ).filter(
        StockDisclosureChunk.document_type == 'QA',
        StockDisclosureChunk.ann_date.isnot(None)
    ).scalar()
    
    # SQLAlchemy 1.4+ .scalar() returns the first element of the first result or None
    if latest_date:
        logger.info(f"找到数据库中最新的QA记录日期: {latest_date}")
        return latest_date
    else:
        logger.info("数据库中没有找到任何QA类型的记录，将进行全量同步。")
        return None

def get_target_disclosures(db: Session, after_date: Optional[datetime.date]) -> List[StockDisclosure]:
    """获取需要处理的调研公告列表 (仅限最近3年)"""
    # 1. 计算3年前的日期
    three_years_ago = datetime.now().date() - timedelta(days=3*365)
    
    # 2. 确定查询的起始日期：应为'增量更新日期'和'3年前'两者中较晚的那个
    start_date = three_years_ago
    if after_date and after_date > three_years_ago:
        start_date = after_date
    
    logger.info(f"将获取公告日期在 {start_date} 之后及当日的Q&A公告。")

    query = db.query(StockDisclosure).filter(
        or_(
            StockDisclosure.tag.ilike('%调研%'),
            StockDisclosure.tag.ilike('%投资者关系%')
        ),
        # 3. 将日期过滤条件统一放在这里
        StockDisclosure.ann_date >= start_date
    ).order_by(StockDisclosure.ann_date.asc())

    return query.all()


def process_and_store_qa(db: Session, disclosure: StockDisclosure):
    """
    处理单个调研公告：爬取PDF -> AI提取QA -> 向量化 -> 存储
    """
    logger.info(f"--- 开始处理公告: 《{disclosure.title}》 ({disclosure.ann_date}) ---")
    
    # 1. 获取PDF内容
    # 步骤 1a: 从数据库中旧的、可能已失效的详情页URL，构造出当前有效的直接PDF下载链接。
    direct_pdf_url = try_construct_pdf_url(disclosure.url)
    if not direct_pdf_url:
        logger.error(f"无法从详情页URL构造PDF链接: {disclosure.url}")
        return

    # 步骤 1b: 使用新构造的直接链接下载PDF，并将原始URL作为Referer以提高成功率。
    pdf_content_stream = stealth_download_pdf(direct_pdf_url, referer=disclosure.url)
    if not pdf_content_stream:
        logger.error(f"无法下载PDF: {direct_pdf_url} (源自: {disclosure.url})")
        return

    # 2. 提取并清理PDF文本
    try:
        common_lines = identify_common_lines(pdf_content_stream)
        pdf_content_stream.seek(0)
        with pdfplumber.open(pdf_content_stream) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        if not full_text:
            logger.warning("PDF文本提取为空。")
            return
            
        cleaned_text = clean_text_with_blacklist(full_text, common_lines)
        logger.info(f"PDF文本提取并清理完成，准备送入AI...")
    except Exception as e:
        logger.error(f"处理PDF时出错: {e}", exc_info=True)
        return

    # 3. AI提取Q&A
    qa_pairs = extract_qa_with_ai(cleaned_text)
    if not qa_pairs:
        logger.warning(f"AI未能从《{disclosure.title}》中提取到任何Q&A对。")
        return
        
    logger.info(f"AI成功提取到 {len(qa_pairs)} 个Q&A对，准备进行向量化...")

    # 4. 获取股票简称
    stock_info = db.query(StockList).filter(StockList.code == disclosure.symbol).first()
    short_name = stock_info.name if stock_info else disclosure.short_name

    chunks_to_upsert = []
    
    # 5. 准备数据以便批量插入
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question", "").strip()
        answer = qa_pair.get("answer", "").strip()
        
        if not question or not answer:
            continue
            
        chunk_text = f"问题：{question}\n\n回答：{answer}"
        
        # 向量化
        embedding = get_embedding(chunk_text, is_query=False)
        if not embedding:
            logger.warning(f"无法为Q&A chunk {i} 生成向量，已跳过。")
            continue

        chunk_data = {
            "symbol": disclosure.symbol,
            "short_name": short_name,
            "ann_date": disclosure.ann_date,
            "document_type": 'QA',
            "document_title": disclosure.title,
            "chunk_order": i,
            "chunk_text": chunk_text,
            "chunk_vector": embedding
        }
        chunks_to_upsert.append(chunk_data)
        
    if not chunks_to_upsert:
        logger.info("没有生成任何有效的Q&A文本块可供存储。")
        return

    # 6. 批量写入数据库 (Upsert)
    try:
        stmt = insert(StockDisclosureChunk).values(chunks_to_upsert)
        update_dict = {
            'chunk_text': stmt.excluded.chunk_text,
            'chunk_vector': stmt.excluded.chunk_vector,
            'short_name': stmt.excluded.short_name
        }
        final_stmt = stmt.on_conflict_do_update(
            index_elements=['symbol', 'document_type', 'document_title', 'chunk_order'],
            set_=update_dict
        )
        db.execute(final_stmt)
        db.commit()
        logger.info(f"✅ 成功将 {len(chunks_to_upsert)} 个Q&A块写入/更新到数据库。")
    except Exception as e:
        logger.error(f"数据库写入Q&A块时失败: {e}", exc_info=True)
        db.rollback()


def sync_all_qa(since_date: Optional[str] = None):
    """
    主函数：对数据库中所有新的调研公告，执行Q&A提取和向量化。
    :param since_date: 如果提供，则从该日期开始同步 (格式 YYYY-MM-DD)。
    """
    logger.info("--- 启动Q&A同步任务 ---")
    db_session = SessionLocal()
    start_time = time.time()

    try:
        # 1. 确定增量更新的起始日期
        start_date_to_sync = None
        if since_date:
            try:
                start_date_to_sync = datetime.strptime(since_date, "%Y-%m-%d").date()
                logger.info(f"将使用命令行指定的起始日期: {start_date_to_sync}")
            except ValueError:
                logger.error(f"指定的日期格式无效: '{since_date}'. 请使用 YYYY-MM-DD 格式。")
                return
        else:
            logger.info("未指定起始日期，将自动从数据库中查找上次同步的位置。")
            start_date_to_sync = get_last_qa_sync_date(db_session)

        # 2. 获取待处理的公告列表
        target_disclosures = get_target_disclosures(db_session, start_date_to_sync)

        total_disclosures = len(target_disclosures)
        if not total_disclosures:
            logger.info("没有新的调研公告需要处理。任务结束。")
            return

        logger.info(f"发现 {total_disclosures} 份调研公告需要处理。")

        # 3. 依次处理每个公告
        for i, disclosure in enumerate(target_disclosures):
            logger.info(f"--- 处理进度: [{i+1}/{total_disclosures}] ---")
            process_and_store_qa(db_session, disclosure)
            # time.sleep(2) # 可以在此加入延时，避免AI或爬虫请求过于频繁

    except Exception as e:
        logger.error(f"在Q&A同步主循环中发生严重错误: {e}", exc_info=True)
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- Q&A同步任务结束，总耗时: {duration:.2f} 秒 ---")
        db_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步投资者问答(Q&A)数据。")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="提供一个起始日期 (格式: YYYY-MM-DD)，脚本将从该日期之后开始同步数据。"
             "如果省略此参数，脚本会自动从上次同步的断点处继续。"
    )
    args = parser.parse_args()
    sync_all_qa(since_date=args.since) 