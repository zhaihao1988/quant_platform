# data_processing/process_mda.py
import logging
import argparse
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

# 本地模块导入
from db.database import SessionLocal, get_db_session
from db.models import StockDisclosureChunk, StockList
from core.vectorizer import split_text_into_chunks, get_embedding
from scripts.scrape_10jqka import get_management_discussion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_and_store_mda(db: Session, symbol: str):
    """
    获取、处理并存储指定股票代码的董事会经营评述(MDA)。
    - 从 10jqka 获取数据。
    - 对每个报告期的文本进行分块和向量化。
    - 将数据存入 StockDisclosureChunk 表，并处理重复数据。
    """
    logger.info(f"--- 开始为股票 {symbol} 处理董事会经营评述 ---")

    # 1. 获取股票简称
    stock_info = db.query(StockList).filter(StockList.code == symbol).first()
    if not stock_info:
        logger.error(f"在 StockList 中未找到股票代码 {symbol}，处理中止。")
        return
    short_name = stock_info.name
    logger.info(f"找到股票简称: {short_name}")

    # 2. 从 10jqka 获取所有日期的经营评述
    discussion_data = get_management_discussion(symbol)
    if not discussion_data:
        logger.warning(f"未能从 10jqka 获取到 {symbol} 的经营评述数据。")
        return

    logger.info(f"从 10jqka 获取到 {len(discussion_data)} 份经营评述，将逐一处理。")

    chunks_to_upsert = []
    
    # 3. 遍历所有报告期的数据 (确保所有日期都保存)
    for report_date_str, text in discussion_data.items():
        if not text or not text.strip():
            logger.warning(f"报告期 {report_date_str} 的文本为空，已跳过。")
            continue

        document_title = f"{short_name} {report_date_str} 董事会经营评述"
        logger.info(f"正在处理: {document_title}")
        
        # 4. 文本分块
        chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            logger.warning(f"文本无法分块: {document_title}")
            continue

        logger.info(f"文本被分为 {len(chunks)} 个块，正在生成向量...")
        
        # 5. 向量化
        try:
            embeddings = get_embedding(chunks) # 批量处理
            if not embeddings or len(embeddings) != len(chunks):
                 logger.error(f"向量化失败或返回数量不匹配: {document_title}")
                 continue
        except Exception as e:
            logger.error(f"向量化过程中出现严重错误: {e}", exc_info=True)
            continue
            
        # 6. 准备要插入/更新的数据
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "symbol": symbol,
                "short_name": short_name,
                "ann_date": datetime.strptime(report_date_str, '%Y-%m-%d').date(),
                "document_type": 'MDA',
                "document_title": document_title,
                "chunk_order": i,
                "chunk_text": chunk_text,
                "chunk_vector": embeddings[i]
            }
            chunks_to_upsert.append(chunk_data)

    if not chunks_to_upsert:
        logger.info("没有生成任何新的文本块可供存储。")
        return
        
    logger.info(f"共生成 {len(chunks_to_upsert)} 个文本块，准备写入数据库...")

    # 7. 批量写入数据库 (Upsert)
    try:
        # 基于唯一索引 'idx_chunk_identity' 进行冲突处理
        stmt = insert(StockDisclosureChunk).values(chunks_to_upsert)
        
        # 定义在冲突时要更新的字段
        update_dict = {
            'chunk_text': stmt.excluded.chunk_text,
            'chunk_vector': stmt.excluded.chunk_vector,
            'short_name': stmt.excluded.short_name, # 同时更新简称，以防股票改名
        }
        
        # ON CONFLICT DO UPDATE
        final_stmt = stmt.on_conflict_do_update(
            index_elements=['symbol', 'document_type', 'document_title', 'chunk_order'],
            set_=update_dict
        )
        
        db.execute(final_stmt)
        db.commit()
        logger.info(f"✅ 成功将 {len(chunks_to_upsert)} 个文本块写入/更新到数据库。")

    except Exception as e:
        logger.error(f"数据库写入失败: {e}", exc_info=True)
        db.rollback()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理并存储来自10jqka的董事会经营评述。")
    parser.add_argument("symbol", type=str, help="要处理的股票代码，例如 '000887'。")
    args = parser.parse_args()

    db_session = SessionLocal()
    try:
        process_and_store_mda(db_session, args.symbol)
    finally:
        db_session.close()
        logger.info("数据库会话已关闭。") 