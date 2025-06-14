#scripts/to_chunks.py
import re
import logging
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from collections import Counter

# Langchain for recursive chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目导入
from db.database import SessionLocal
from db.models import StockDisclosure, StockDisclosureChunk # 确保 StockDisclosureChunk 有 disclosure_ann_date 字段
from config.settings import settings # CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIM, etc.
from rag.embeddings import Embedder # 更新为直接使用 Embedder 类

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 全局 Embedder 实例 ---
embedder_instance = None
EMBEDDING_DIM_TO_USE = settings.EMBEDDING_DIM # 默认为 settings 中的值

try:
    logger.info(f"Initializing Embedder (model from settings: {settings.EMBEDDING_MODEL_NAME})...")
    embedder_instance = Embedder() # Embedder 内部会从 settings 加载模型名称

    # 从加载的模型实例中获取真实的维度并验证
    if hasattr(embedder_instance, 'model') and \
       embedder_instance.model is not None and \
       hasattr(embedder_instance.model, 'get_sentence_embedding_dimension'):
        MODEL_ACTUAL_DIM = embedder_instance.model.get_sentence_embedding_dimension()
        if MODEL_ACTUAL_DIM != settings.EMBEDDING_DIM:
            logger.warning(
                f"CRITICAL MISMATCH: Actual model output dimension ({MODEL_ACTUAL_DIM}) "
                f"from '{settings.EMBEDDING_MODEL_NAME}' "
                f"does NOT match settings.EMBEDDING_DIM ({settings.EMBEDDING_DIM}). "
                f"Using actual model dimension: {MODEL_ACTUAL_DIM}. "
                "PLEASE ENSURE 'EMBEDDING_DIM' IN settings.py IS CORRECT FOR THE CHOSEN MODEL."
            )
            EMBEDDING_DIM_TO_USE = MODEL_ACTUAL_DIM # 优先使用模型真实维度
        else:
            EMBEDDING_DIM_TO_USE = settings.EMBEDDING_DIM # 一致
            logger.info(f"Embedder initialized. Model: '{settings.EMBEDDING_MODEL_NAME}'. "
                        f"Effective dimension for storage: {EMBEDDING_DIM_TO_USE} (matches settings).")
    else:
        logger.warning(
            "Could not reliably determine embedding dimension from the loaded model instance. "
            f"Falling back to settings.EMBEDDING_DIM ({settings.EMBEDDING_DIM}). "
            "Ensure this matches the actual output dimension of the model: "
            f"'{settings.EMBEDDING_MODEL_NAME}'."
        )
        # EMBEDDING_DIM_TO_USE 保持为 settings.EMBEDDING_DIM
        # 这种情况可能发生在 Embedder.__init__ 中模型加载失败但未抛出异常（如果修改了错误处理）

except Exception as e:
    logger.error(f"Fatal: Failed to initialize Embedder or determine model dimension: {e}", exc_info=True)
    # embedder_instance 保持为 None, 后续依赖此实例的操作将失败或被跳过


# --- 文本预处理函数 ---
def remove_headers_footers_by_repetition(text: str, min_repeats: int, max_line_len: int) -> str:
    """
    通过查找文档内重复的短文本行来移除页眉和页脚。
    """
    lines = text.splitlines()
    if not lines:
        return ""

    stripped_lines = [line.strip() for line in lines]
    line_counts = Counter(s_line for s_line in stripped_lines if s_line)

    lines_to_remove_content = set()
    for line_content, count in line_counts.items():
        if count >= min_repeats and len(line_content) <= max_line_len:
            lines_to_remove_content.add(line_content)
            logger.debug(f"Identified potential header/footer by repetition: '{line_content}' (count: {count})")

    if not lines_to_remove_content:
        return text

    cleaned_lines = [original_line for original_line in lines if original_line.strip() not in lines_to_remove_content]
    return "\n".join(cleaned_lines)

def normalize_whitespace(text: str) -> str:
    """标准化文本中的空白符。"""
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

# --- 核心处理逻辑 ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type((IOError, IntegrityError)), # 可根据需要调整重试的异常类型
    reraise=True
)
def embed_and_store_disclosure_chunks(db: Session, disclosure_id: int, raw_text: str, ann_date: date):
    """对单个公告进行预处理、递归分块、向量化并存储。"""
    if not raw_text or not raw_text.strip():
        logger.info(f"Disclosure ID {disclosure_id} has empty raw_content. Skipping.")
        return 0

    # 1. 预处理
    logger.debug(f"Preprocessing disclosure ID {disclosure_id}...")
    text = remove_headers_footers_by_repetition(
        raw_text,
        min_repeats=settings.HEADER_FOOTER_MIN_REPEATS,
        max_line_len=settings.HEADER_FOOTER_MAX_LINE_LEN
    )
    text = normalize_whitespace(text)

    if not text.strip():
        logger.info(f"Disclosure ID {disclosure_id} content became empty after preprocessing. Skipping.")
        return 0

    # 2. 递归分块
    logger.info(f"Recursively chunking disclosure ID {disclosure_id} (ann_date: {ann_date}). "
                f"Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=settings.TEXT_SPLITTER_SEPARATORS, # 从 settings 读取分隔符列表
        keep_separator=settings.TEXT_SPLITTER_KEEP_SEPARATOR
    )
    chunks_text = text_splitter.split_text(text)

    if not chunks_text:
        logger.info(f"No text chunks generated by RecursiveCharacterTextSplitter for disclosure ID {disclosure_id}. Skipping.")
        return 0
    logger.info(f"Generated {len(chunks_text)} raw chunks using RecursiveCharacterTextSplitter for disclosure ID {disclosure_id}.")

    # 3. 向量化
    if not embedder_instance or not hasattr(embedder_instance, 'embed'):
        logger.error("Embedder instance is not available (failed to initialize). Cannot vectorize chunks.")
        raise EnvironmentError("Embedder not initialized, unable to proceed with vectorization.")

    logger.info(f"Generating embeddings for {len(chunks_text)} chunks for disclosure ID {disclosure_id}...")
    try:
        embeddings = embedder_instance.embed(chunks_text) # embedder 内部处理 batching
    except Exception as e:
        logger.error(f"Error during embedding generation for disclosure {disclosure_id}: {e}", exc_info=True)
        raise

    if len(embeddings) != len(chunks_text):
        logger.error(f"Mismatch between number of chunks ({len(chunks_text)}) and "
                     f"embeddings ({len(embeddings)}) for disclosure {disclosure_id}. Skipping this disclosure.")
        return 0

    # 4. 准备并存储数据块对象
    chunk_objects_to_store = []
    for i, (text_chunk, embedding_vector) in enumerate(zip(chunks_text, embeddings)):
        if not text_chunk.strip():
            logger.warning(f"Skipping empty text_chunk at index {i} for disclosure {disclosure_id} before storage.")
            continue

        # embed 方法应已返回 list of lists of float
        embedding_list = embedding_vector # 直接使用，假设 embedder_instance.embed 已返回正确格式

        if len(embedding_list) != EMBEDDING_DIM_TO_USE: # 使用在启动时确定的维度
            logger.error(
                f"Embedding dimension mismatch for chunk {i} of disclosure {disclosure_id}. "
                f"Expected {EMBEDDING_DIM_TO_USE}, got {len(embedding_list)}. "
                "Skipping this entire disclosure to prevent data corruption."
            )
            raise ValueError(f"Embedding dimension mismatch for disclosure {disclosure_id}")

        chunk_obj = StockDisclosureChunk(
            disclosure_id=disclosure_id,
            chunk_order=i,
            chunk_text=text_chunk,
            chunk_vector=embedding_list,
            disclosure_ann_date=ann_date
        )
        chunk_objects_to_store.append(chunk_obj)

    if not chunk_objects_to_store:
        logger.info(f"No valid chunks to store for disclosure ID {disclosure_id} after processing.")
        return 0

    logger.info(f"Deleting existing chunks for disclosure ID {disclosure_id} before inserting new ones...")
    db.query(StockDisclosureChunk).filter(StockDisclosureChunk.disclosure_id == disclosure_id).delete(synchronize_session='fetch')

    db.add_all(chunk_objects_to_store)
    logger.info(f"Prepared {len(chunk_objects_to_store)} chunks for disclosure ID {disclosure_id} for database commit.")
    return len(chunk_objects_to_store)


def process_disclosures_batch(db: Session, batch_size: int, re_process_all: bool = False, specific_disclosure_ids: list[int] = None):
    """分批处理所有符合条件的股票公告。"""
    if not embedder_instance: # 检查 embedder_instance 是否成功初始化
        logger.critical("Embedder instance not available. Aborting main processing loop.")
        return

    processed_count = 0
    total_chunks_count = 0
    offset = 0

    while True:
        logger.info(f"Fetching next batch of disclosures (offset: {offset}, batch_size: {batch_size})...")
        query = db.query(StockDisclosure.id, StockDisclosure.raw_content, StockDisclosure.ann_date)

        if specific_disclosure_ids:
            logger.info(f"Processing specific disclosure IDs: {specific_disclosure_ids}")
            query = query.filter(StockDisclosure.id.in_(specific_disclosure_ids))
        elif not re_process_all:
            processed_ids_subquery = db.query(StockDisclosureChunk.disclosure_id).distinct().subquery()
            query = query.filter(StockDisclosure.id.notin_(processed_ids_subquery))
            logger.info("Querying for disclosures not yet processed...")
        else:
            logger.info("Configured to re-process all disclosures.")

        disclosures_batch = query.order_by(StockDisclosure.id).offset(offset).limit(batch_size).all()

        if not disclosures_batch:
            if specific_disclosure_ids and offset == 0:
                 logger.info(f"No disclosures found for the specified IDs: {specific_disclosure_ids}.")
            elif not specific_disclosure_ids:
                 logger.info("No more disclosures to process in this run.")
            break

        logger.info(f"Processing batch of {len(disclosures_batch)} disclosures.")
        for disclosure_data in disclosures_batch:
            disclosure_id, raw_content, ann_date = disclosure_data

            if raw_content is None or not raw_content.strip():
                logger.warning(f"Disclosure ID {disclosure_id} has no raw_content or it's empty. Skipping.")
                continue
            if ann_date is None:
                logger.warning(f"Disclosure ID {disclosure_id} has no 'ann_date'. Skipping as it's crucial for time-stamping.")
                continue

            logger.info(f"Starting processing for disclosure ID: {disclosure_id} (Published: {ann_date})")
            try:
                chunks_generated = embed_and_store_disclosure_chunks(db, disclosure_id, raw_content, ann_date)
                db.commit()
                logger.info(f"Successfully processed and committed disclosure ID {disclosure_id}. Chunks stored: {chunks_generated}.")
                processed_count += 1
                total_chunks_count += chunks_generated
            except ValueError as ve: # 例如维度不匹配
                logger.error(f"ValueError during processing of disclosure {disclosure_id}: {ve}. Rolling back changes for this disclosure.")
                db.rollback()
            except EnvironmentError as ee: # 例如 Embedder 初始化失败
                logger.critical(f"EnvironmentError processing disclosure {disclosure_id}: {ee}. This may affect further processing. Rolling back.")
                db.rollback()
                # 根据严重性，可能需要终止整个批处理
                # raise #  可以选择重新抛出，终止整个脚本
            except Exception as e:
                logger.error(f"Unhandled error processing disclosure ID {disclosure_id}: {e}", exc_info=True)
                db.rollback()

        if specific_disclosure_ids:
            logger.info(f"Finished processing specific disclosure IDs.")
            break
        offset += batch_size

    logger.info(f"Batch processing finished. Processed {processed_count} disclosures in this run. Total chunks stored/updated: {total_chunks_count}.")


if __name__ == "__main__":
    if not embedder_instance: # 在主执行前再次检查
        logger.critical("Embedder instance failed to initialize. Script will not run.")
        exit(1) # 以错误码退出

    # 从 settings 中获取参数，并提供默认值
    # 这些参数应该在您的 settings.py 中定义，否则这里会使用默认值
    # 例如，在 settings.py 中添加：
    # HEADER_FOOTER_MIN_REPEATS: int = 3
    # HEADER_FOOTER_MAX_LINE_LEN: int = 100
    # TEXT_SPLITTER_SEPARATORS: list[str] = ["\n\n\n", "\n\n", "\n", "。", "！", ...]
    # TEXT_SPLITTER_KEEP_SEPARATOR: bool = True
    # PROCESSING_BATCH_SIZE: int = 50
    # EMBEDDING_BATCH_SIZE: int = 32 # 这个参数目前在 Embedder.embed 中没有直接使用，但可以保留用于其他地方

    # 为 settings 中可能缺失的参数提供脚本级默认值或确保它们存在
    if not hasattr(settings, 'HEADER_FOOTER_MIN_REPEATS'):
        settings.HEADER_FOOTER_MIN_REPEATS = 3
        logger.warning(f"settings.HEADER_FOOTER_MIN_REPEATS not found, using default: {settings.HEADER_FOOTER_MIN_REPEATS}")
    if not hasattr(settings, 'HEADER_FOOTER_MAX_LINE_LEN'):
        settings.HEADER_FOOTER_MAX_LINE_LEN = 100
        logger.warning(f"settings.HEADER_FOOTER_MAX_LINE_LEN not found, using default: {settings.HEADER_FOOTER_MAX_LINE_LEN}")
    if not hasattr(settings, 'TEXT_SPLITTER_SEPARATORS'):
        settings.TEXT_SPLITTER_SEPARATORS = ["\n\n\n", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        logger.warning(f"settings.TEXT_SPLITTER_SEPARATORS not found, using default list.")
    if not hasattr(settings, 'TEXT_SPLITTER_KEEP_SEPARATOR'):
        settings.TEXT_SPLITTER_KEEP_SEPARATOR = True # Langchain 默认为 True
        logger.warning(f"settings.TEXT_SPLITTER_KEEP_SEPARATOR not found, using default: {settings.TEXT_SPLITTER_KEEP_SEPARATOR}")
    if not hasattr(settings, 'PROCESSING_BATCH_SIZE'):
        settings.PROCESSING_BATCH_SIZE = 50
        logger.warning(f"settings.PROCESSING_BATCH_SIZE not found, using default: {settings.PROCESSING_BATCH_SIZE}")


    db_session: Session = SessionLocal()
    try:
        logger.info("Starting disclosure chunking and vectorization script (recursive chunking, settings-driven embedder)...")
        process_disclosures_batch(db_session, batch_size=settings.PROCESSING_BATCH_SIZE, re_process_all=False)
        logger.info("Disclosure chunking and vectorization script finished successfully.")
    except Exception as e:
        logger.error(f"An unhandled error occurred in the main execution block: {e}", exc_info=True)
    finally:
        db_session.close()
        logger.info("Database session closed.")