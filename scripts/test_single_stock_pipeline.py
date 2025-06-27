# --- 关键修复：将日志配置移到所有项目导入之前 ---
import logging
# 配置必须在其他模块（它们可能会记录日志）被导入之前完成
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # 获取当前模块的 logger

import argparse
import os
import json
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, not_
from typing import List
# 移除akshare导入
# import akshare as ak

# --- 使用现有数据库 ---
from db.database import SessionLocal
from db.models import Base, StockDisclosure, StockDisclosureChunk

# --- 项目模块 ---
from data_processing.scraper import fetch_announcement_text, extract_and_clean_narrative_section, extract_qa_with_ai, extract_section_from_text, remove_tables
from scripts.to_chunks import embed_and_store_disclosure_chunks
from config.settings import settings

# --- 日志配置 (旧位置) ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # 已移动到顶部

# --- 测试输出目录 ---
TEST_OUTPUT_DIR = "tests/output_test"
if not os.path.exists(TEST_OUTPUT_DIR):
    os.makedirs(TEST_OUTPUT_DIR)

try:
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

def get_target_disclosures_for_processing(db_session, symbol: str) -> List[StockDisclosure]:
    """
    查找指定股票的、符合特定业务规则的公告。
    复用 populate_selected_raw_content.py 中的筛选逻辑。
    """
    logger.info(f"为股票 {symbol} 查询需要处理的公告 (使用筛选逻辑)")
    try:
        now = datetime.now()
        one_year_ago_date = (now - timedelta(days=365)).date()

        if DATEUTIL_AVAILABLE:
            three_years_ago_date = (now - relativedelta(years=3)).date()
        else:
            three_years_ago_date = (now - timedelta(days=3 * 365 + 1)).date()  # 粗略计算

        annual_semi_keywords = ['年度报告', '半年度报告']
        # 保留"调研"关键词，因为它是我们关注的重点
        other_relevant_keywords = ['调研'] 

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
                    StockDisclosure.title.ilike(f'%{kw}%'),
                    StockDisclosure.ann_date >= three_years_ago_date
                )
            )

        # 条件组2：最近1年的调研活动 (根据用户最新反馈，仅基于tag)
        # --- 关键修改：仅在 tag 中搜索，并严格限定在1年内 ---
        filter_conditions.append(
            and_(
                StockDisclosure.tag.ilike(f'%调研%'),
                StockDisclosure.ann_date >= one_year_ago_date 
            )
        )

        if not filter_conditions:
            logger.warning(f"股票 {symbol}: 未配置有效的关键词筛选条件。")
            return []

        combined_keyword_filters = or_(*filter_conditions)

        # 基础查询
        query = db_session.query(StockDisclosure).filter(
            StockDisclosure.symbol == symbol,
            combined_keyword_filters
        )

        # 循环添加排除条件
        for keyword in exclusion_keywords:
            query = query.filter(not_(StockDisclosure.title.ilike(f'%{keyword}%')))

        disclosures = query.order_by(StockDisclosure.ann_date.desc()).all()

        logger.info(f"股票 {symbol}: 找到 {len(disclosures)} 条符合筛选条件的公告。")
        return disclosures
    except Exception as e:
        logger.error(f"为股票 {symbol} 查询待处理公告时出错: {e}", exc_info=True)
        return []

def process_disclosure_content(disclosure_obj: StockDisclosure) -> str:
    """
    根据公告类型处理内容，复用 populate_selected_raw_content.py 中的处理逻辑。
    """
    title = disclosure_obj.title
    tag = disclosure_obj.tag if disclosure_obj.tag else ""
    
    # # 根据用户测试要求，注释掉此部分，总是重新获取
    # if disclosure_obj.raw_content and disclosure_obj.raw_content.strip():
    #     logger.info("数据库中已有 raw_content，直接使用。")
    #     return disclosure_obj.raw_content
    
    # 总是从URL获取并处理
    logger.info("根据测试要求，无论是否存在旧内容，都将重新获取 raw_content...")
    full_text = fetch_announcement_text(
        detail_url=disclosure_obj.url,
        title=title,
        tag=tag
    )
    
    if not full_text:
        logger.error("无法从公告中提取任何文本内容。")
        return None
    
    # 根据类型应用不同解析逻辑
    content_to_store = None
    
    if '调研' in tag :
        logger.info("检测到调研活动，使用AI提取Q&A...")
        # 强制使用指定的模型进行测试
        logger.info("使用硅基流动模型 'Qwen/Qwen3-8B' 进行处理...")
        qa_list = extract_qa_with_ai(full_text, model_override="Qwen/Qwen3-8B")
        if qa_list:
            # 将Q&A列表格式化为JSON字符串进行存储
            content_to_store = json.dumps(qa_list, ensure_ascii=False, indent=4)
            logger.info(f"成功提取到 {len(qa_list)} 条Q&A。")
        else:
            logger.warning("AI未能提取到Q&A内容，将使用全文作为备用。")
            content_to_store = full_text
    
    elif '年度报告' in title or '半年度报告' in title:
        logger.info("检测到年报/半年报，提取【管理层讨论与分析】...")
        narrative_section = extract_section_from_text(full_text, "管理层讨论与分析")
        if narrative_section:
            logger.info("成功提取章节，开始清理表格...")
            content_to_store = remove_tables(narrative_section)
        else:
            logger.warning("未能提取到'管理层讨论与分析'，将使用清理后的全文作为备用。")
            # 直接对全文进行表格清理，而不是调用extract_and_clean_narrative_section
            content_to_store = remove_tables(full_text)
    
    else: # 其他类型
        logger.info(f"常规公告 '{title}'，使用清理后的全文。")
        # 对全文进行基本的表格清理
        content_to_store = remove_tables(full_text)
    
    return content_to_store

# --- V2: 新增独立验证函数 ---
def verify_data_in_new_session(disclosure_id: int) -> bool:
    """
    开启一个全新的、独立的数据库会话来验证数据是否被永久提交。
    这模拟了外部工具的查询行为。
    """
    logger.info("--- Verification with NEW session ---")
    new_session = None
    try:
        new_session = SessionLocal()
        logger.info(f"New session created. Querying for disclosure_id: {disclosure_id}")
        count = new_session.query(StockDisclosureChunk).filter(StockDisclosureChunk.disclosure_id == disclosure_id).count()
        
        if count > 0:
            logger.info(f"✅✅✅ ULTIMATE SUCCESS: Found {count} chunks in a new, independent session. Data is permanently stored.")
            return True
        else:
            logger.error(f"❌❌❌ ULTIMATE FAILURE: Found 0 chunks in a new, independent session. Data was NOT committed.")
            return False
            
    except Exception as e:
        logger.error(f"Error during new session verification: {e}", exc_info=True)
        return False
    finally:
        if new_session:
            new_session.close()
            logger.info("New verification session closed.")


def run_batch_processing_for_stock(stock_code: str):
    """
    针对单个股票从数据库获取所有符合条件的公告，进行批量处理、向量化和存储。
    """
    db_session = SessionLocal()

    try:
        # --- 关键修复：在测试开始前，强制重建表结构以匹配最新模型 ---
        # 这个操作在整个批次开始前只执行一次
        logger.info("--- [BATCH PROCESSING] Forcibly resetting 'stock_disclosure_chunk' table schema ---")
        StockDisclosureChunk.__table__.drop(db_session.bind, checkfirst=True)
        logger.info("Dropped old 'stock_disclosure_chunk' table (if it existed).")
        Base.metadata.create_all(db_session.bind)
        logger.info("Recreated table(s) based on current model definitions. Schema is now up-to-date.")
        db_session.commit()
        logger.info("--- Schema reset complete. Proceeding with batch processing. ---")

        # 1. 从数据库获取符合条件的公告列表
        logger.info(f"--- 步骤 1: 从数据库获取股票代码 {stock_code} 所有符合条件的公告... ---")
        target_disclosures = get_target_disclosures_for_processing(db_session, stock_code)
        if not target_disclosures:
            logger.error(f"数据库中未找到股票 {stock_code} 符合条件的任何公告。")
            return
        
        logger.info(f"找到 {len(target_disclosures)} 条符合条件的公告，将全部处理。")
        
        # --- 主循环：遍历并处理每一条公告 ---
        for i, disclosure in enumerate(target_disclosures, 1):
            disclosure_id = disclosure.id
            title = disclosure.title
            url = disclosure.url
            ann_date = disclosure.ann_date

            logger.info(f"\n{'='*20}>> 开始处理第 {i}/{len(target_disclosures)} 条公告 <<{'='*20}")
            logger.info(f"公告: '{title}' (ID: {disclosure_id}, 发布于 {ann_date})")

            # 2. 处理文本内容
            logger.info("--- 步骤 2/4: 处理公告内容... ---")
            processed_content = process_disclosure_content(disclosure)
            if not processed_content or not processed_content.strip():
                logger.error(f"公告ID {disclosure_id} 处理后内容为空，跳过此公告。")
                continue # 跳到下一个循环

            # 将处理好的内容保存到文件，方便检查 (文件名包含ID以示区分)
            output_file = os.path.join(TEST_OUTPUT_DIR, f"{stock_code}_{disclosure_id}_processed.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(processed_content)
            logger.info(f"✅ 处理后的内容已保存至: {output_file}")

            # 3. 直接进行分块和向量化存储
            logger.info("--- 步骤 3/4: 对处理的内容进行分块和向量化存储... ---")
            chunks_stored = embed_and_store_disclosure_chunks(
                db=db_session,
                disclosure_id=disclosure_id,
                raw_text=processed_content,
                ann_date=disclosure.ann_date
            )
            logger.info(f"✅ 公告ID {disclosure_id}: 向量化和存储流程完成。共存入 {chunks_stored} 个数据块。")

            # 4. 验证结果 (在同一个会话内)
            logger.info("--- 步骤 4/4: (内部验证) 在当前会话中检查存储结果... ---")
            chunk_count = db_session.query(StockDisclosureChunk).filter(
                StockDisclosureChunk.disclosure_id == disclosure_id
            ).count()
            
            if chunk_count == chunks_stored:
                logger.info(f"✅ 公告ID {disclosure_id}: 内部验证成功！数量匹配 ({chunk_count})。")
            else:
                logger.error(f"❌ 公告ID {disclosure_id}: 内部验证失败！存入数({chunks_stored})与查询数({chunk_count})不匹配。")
            
            # 每次循环后提交一次，确保单个公告的处理是原子性的
            logger.info(f"Committing changes for disclosure ID {disclosure_id}...")
            db_session.commit()
            logger.info(f"{'='*20}>> 公告ID {disclosure_id} 处理完毕 <<{'='*20}\n")
        
        logger.info("🎉🎉🎉 所有公告处理完毕! 🎉🎉🎉")
        
        # 最终验证可以被移除或简化，因为每次循环都已经验证
        total_chunk_count = db_session.query(StockDisclosureChunk).count()
        logger.info(f"数据库中现在总共有 {total_chunk_count} 个数据块。")

    except Exception as e:
        logger.error(f"批量处理过程中发生意外错误: {e}", exc_info=True)
        logger.error("正在回滚主会话中的所有未提交更改...")
        db_session.rollback()
    finally:
        logger.info("正在关闭主数据库会话。")
        db_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对单个股票的符合条件公告进行端到端处理测试。")
    parser.add_argument("--stock_code", type=str, default="000887", help="要测试的股票代码")
    args = parser.parse_args()

    # 确保Embedder可以被初始化
    if not hasattr(settings, 'EMBEDDING_MODEL_NAME') or not settings.EMBEDDING_MODEL_NAME:
        logger.critical("EMBEDDING_MODEL_NAME 未在 settings.py 中配置。脚本无法执行。")
        exit(1)

    # 运行批量处理
    run_batch_processing_for_stock(args.stock_code)

    logger.info("="*20 + " 批量处理流程结束 " + "="*20) 