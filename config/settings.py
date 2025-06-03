# config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ---------- 数据库 ----------
    DB_URL: str = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

    # ---------- Ollama 相关 ----------
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:14b") # Example model

    # ---------- 邮件设置 ----------
    EMAIL_USER: str = os.getenv("EMAIL_USER", "zhaihao_n@126.com") # Replace with your actual email
    EMAIL_PASS: str = os.getenv("EMAIL_PASS", "glmdfpoaA8") # **IMPORTANT: Use 126 Mail Authorization Code**
    EMAIL_SMTP_SERVER: str = os.getenv("EMAIL_SMTP_SERVER", "smtp.126.com")
    EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "465")) # Use 465 for SSL by default for 126/163

    # ---------- Google Custom Search Engine ----------
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "AIzaSyB0Kv14UpjEDv59HEOV4ducTqaPk8633L8")
    GOOGLE_CX: Optional[str] = os.getenv("GOOGLE_CX", "533a067c36f9d48f1")
    AKSHARE_REQUEST_DELAY: float = float(os.getenv("AKSHARE_REQUEST_DELAY", "0.3"))  # <-- 新增或确保存在这一行
    # ---------- Embedding Model ----------
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", 'shibing624/text2vec-base-chinese')
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "510")) # Max tokens for embedding model

    # ---------- PGVector HNSW Index Settings ---------- # <--- 新增配置段落
    PGVECTOR_HNSW_M: int = int(os.getenv("PGVECTOR_HNSW_M", "16")) # HNSW M参数
    PGVECTOR_HNSW_EF_CONSTRUCTION: int = int(os.getenv("PGVECTOR_HNSW_EF_CONSTRUCTION", "64")) # HNSW ef_construction参数

    # ---------- Report Saving ----------
    REPORT_SAVE_PATH: str = os.getenv("REPORT_SAVE_PATH", "output")  # <--- ENSURE THIS LINE EXISTS AND IS CORRECT
    # --- Pydantic-Settings Config (应嵌套在 Settings 类内部) ---
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'ignore' # 忽略 .env 文件中多余的变量，避免报错 (可选但推荐)

settings = Settings()

# 创建报告输出目录的逻辑 (在实例化 settings之后)
# 这部分逻辑通常放在应用启动时或首次使用前，放在这里也可以，但要注意导入副作用
if settings.REPORT_SAVE_PATH and not os.path.exists(settings.REPORT_SAVE_PATH):
    try:
        os.makedirs(settings.REPORT_SAVE_PATH)
        print(f"Report save directory created: {settings.REPORT_SAVE_PATH}")
    except OSError as e:
        print(f"Error creating report save directory {settings.REPORT_SAVE_PATH}: {e}")

