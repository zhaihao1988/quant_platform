# rag/vectorizer.py
import requests
import time

import random
import logging
from io import BytesIO
from pypdf import PdfReader
from bs4 import BeautifulSoup
import re
from sqlalchemy.orm import Session

from data_processing.scraper import fetch_announcement_text
# 使用您正确的模型路径
from db.models import StockDisclosure
# from db.database import SessionLocal # 如果需要独立运行测试

# SentenceTransformer 用于嵌入
from sentence_transformers import SentenceTransformer
embedding_model_name = 'shibing624/text2vec-base-chinese' # 或从配置读取
embedding_dim = 768 # 确保与模型和数据库列一致
try:
    embedding_model = SentenceTransformer(embedding_model_name)
    loaded_dim = embedding_model.get_sentence_embedding_dimension()
    if loaded_dim != embedding_dim:
         logger.warning(f"Model '{embedding_model_name}' dimension ({loaded_dim}) does not match expected dimension ({embedding_dim})!")
         # 可以选择抛出错误或继续（可能导致后续问题）
    logger.info(f"Loaded embedding model '{embedding_model_name}' with dimension {loaded_dim}.")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model '{embedding_model_name}': {e}")
    embedding_model = None

def get_text_embedding(text: str) -> list[float] | None:
    """生成文本嵌入"""
    if not embedding_model or not text:
        return None
    try:
        # 考虑分块！当前为简化直接嵌入（可能截断）
        # 模型通常有输入长度限制，例如 512 tokens
        # 简单的按字符截断可能不是最优
        max_model_input_length = 510 # 示例，应根据模型调整
        # 粗略估计字符数，一个中文token约占1-2字符，英文token更长
        # 需要更精确的tokenizer来计算长度，或者使用text-splitter库进行分块
        estimated_char_limit = max_model_input_length * 2 # 非常粗略的估计
        truncated_text = text[:estimated_char_limit]
        if len(text) > estimated_char_limit:
             logger.warning(f"Text longer than estimated limit ({estimated_char_limit} chars), truncating for embedding.")

        embedding = embedding_model.encode(truncated_text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# --- 存储与向量化 (使用 raw_content, content_vector) ---
def scrape_and_store_announcements(db: Session, announcements: list[StockDisclosure]):
    """遍历公告列表，爬取内容存入 raw_content，生成向量存入 content_vector"""
    count_scraped = 0
    count_embedded = 0
    for announcement in announcements:
        # 双重检查 raw_content 是否为空
        if announcement.raw_content is not None:
            logger.info(f"Skipping {announcement.title}, raw_content already exists.")
            # 检查是否需要生成向量
            if announcement.content_vector is None:
                logger.info(f"Raw content exists but vector is missing for {announcement.title}. Attempting to embed.")
                embed_existing_content(db, announcement)
                count_embedded +=1 # 计数嵌入操作
            continue # 跳过爬取

        # 爬取内容 (使用 url 字段)
        extracted_text = fetch_announcement_text(announcement.url, announcement.title)

        if extracted_text:
            try:
                # 存储原文到 raw_content
                announcement.raw_content = extracted_text
                db.flush() # 先将 raw_content 刷入，确保 ID 可用或内容已保存

                # 生成并存储向量到 content_vector
                embedding = get_text_embedding(extracted_text)
                if embedding:
                    announcement.content_vector = embedding
                    logger.info(f"Generated and storing vector for: {announcement.title}")
                    count_embedded += 1
                else:
                    logger.error(f"Failed to generate embedding for: {announcement.title}. Vector will be null.")

                db.commit() # 提交本次公告的所有更改
                logger.info(f"Successfully scraped, stored content and vector (if generated) for: {announcement.title}")
                count_scraped += 1
            except Exception as e:
                db.rollback()
                logger.error(f"Error saving content/vector for {announcement.title} to DB: {e}")
        else:
            logger.warning(f"Failed to fetch or extract content for: {announcement.title}")
            # 可以在这里做标记，避免下次重复尝试失败的链接

    logger.info(f"Finished scraping task. Scraped content for {count_scraped} announcements. Generated/updated vectors for {count_embedded} announcements.")

def embed_existing_content(db: Session, announcement: StockDisclosure):
     """为已存在 raw_content 但缺少向量的公告生成并存储向量"""
     if not announcement.raw_content:
          logger.warning(f"Cannot embed {announcement.title}, raw_content is empty.")
          return
     if announcement.content_vector is not None:
          logger.info(f"Vector already exists for {announcement.title}")
          return

     embedding = get_text_embedding(announcement.raw_content)
     if embedding:
         try:
             announcement.content_vector = embedding
             db.commit() # 单独提交这条记录的向量更新
             logger.info(f"Successfully generated and stored vector for existing content: {announcement.title}")
         except Exception as e:
             db.rollback()
             logger.error(f"Error saving vector for existing content {announcement.title}: {e}")
     else:
          logger.error(f"Failed to generate embedding for existing content: {announcement.title}")
