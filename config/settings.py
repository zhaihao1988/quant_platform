# config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Optional
import logging

# --- V3: 增强的 .env 诊断 ---
# 在所有操作之前配置日志，以捕获所有信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- V6: 更换为1024维模型以解决数据库限制问题 ---
CORRECT_DIMENSION_1024 = 1024

class Settings(BaseSettings):
    # ---------- 数据库 ----------
    DB_URL: str = "postgresql://postgres:postgres@localhost:5432/postgres"

    # ---------- LLM Provider Settings (V2) ----------
    LLM_PROVIDER: str = "siliconflow" # 可选 "ollama" 或 "siliconflow"
    
    # ---------- SiliconFlow 相关 ----------
    SILICONFLOW_API_KEY: Optional[str] = None
    SILICONFLOW_MODEL: str = "Qwen/Qwen3-8B" 

    # ---------- Ollama 相关 ----------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b-q4_K_M" 
    # 默认模型，可以在调用时覆盖

    # ---------- 邮件设置 ----------
    EMAIL_USER: str = "zhaihao_n@126.com"
    EMAIL_PASS: str = "glmdfpoaA8" 
    # **IMPORTANT: Use 126 Mail Authorization Code**
    EMAIL_SMTP_SERVER: str = "smtp.126.com"
    EMAIL_SMTP_PORT: int = 465

    # ---------- Google Custom Search Engine ----------
    GOOGLE_API_KEY: Optional[str] = "AIzaSyB0Kv14UpjEDv59HEOV4ducTqaPk8633L8"
    GOOGLE_CX: Optional[str] = "533a067c36f9d48f1"
    AKSHARE_REQUEST_DELAY: float = 0.3
    
    # ---------- Embedding Model (关键修改) ----------
    # 硬编码为本地路径，确保加载正确的低维模型
    EMBEDDING_MODEL_NAME: str = "D:/project/quant_platform/models/Qwen3-Embedding-0.6B"
    # 我们将不再直接使用这个值，而是使用上面的 CORRECT_DIMENSION_1024
    # EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_LENGTH: int = 8192

    # ---------- PGVector HNSW Index Settings ----------
    PGVECTOR_HNSW_M: int = 16
    PGVECTOR_HNSW_EF_CONSTRUCTION: int = 64
    PGVECTOR_IVFFLAT_LISTS: int = 100

    # ---------- Report Saving ----------
    REPORT_SAVE_PATH: str = "output"

    # --- RAG Pipeline: Chunking & Preprocessing ---
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 100
    HEADER_FOOTER_MIN_REPEATS: int = 3
    HEADER_FOOTER_MAX_LINE_LEN: int = 100
    TEXT_SPLITTER_SEPARATORS: list[str] = [
        "\n\n\n", "\n\n", "\n", "。\n", "。",
        "！", "？", "；", "，", " ", ""
    ]
    TEXT_SPLITTER_KEEP_SEPARATOR: bool = True

    # --- RAG Pipeline: Processing Batch Sizes ---
    PROCESSING_BATCH_SIZE: int = 20
    EMBEDDING_BATCH_SIZE: int = 32

    class Config:
        # env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'ignore'

settings = Settings()

# --- 关键诊断日志 ---
# 这段代码将在模块首次导入时执行，打印出最终生效的配置。
# 在basicConfig之后执行，确保日志能被捕获
logger.info("--- V6: Effective Application Settings Loaded (1024 Dim Model) ---")
logger.info(f"DB_URL (last 5 chars): ...{settings.DB_URL[-5:]}")
logger.info(f"EMBEDDING_MODEL_NAME: {settings.EMBEDDING_MODEL_NAME}")
# 日志也直接打印这个强制值
logger.info(f"EMBEDDING_DIM (FORCED & RENAMED): {CORRECT_DIMENSION_1024}")
logger.info(f"SILICONFLOW_API_KEY is set: {settings.SILICONFLOW_API_KEY is not None}")
logger.info("---------------------------------------------")

# 创建报告输出目录的逻辑
if settings.REPORT_SAVE_PATH and not os.path.exists(settings.REPORT_SAVE_PATH):
    try:
        os.makedirs(settings.REPORT_SAVE_PATH)
        logger.info(f"Report save directory created: {settings.REPORT_SAVE_PATH}")
    except OSError as e:
        logger.error(f"Error creating report save directory {settings.REPORT_SAVE_PATH}: {e}")

