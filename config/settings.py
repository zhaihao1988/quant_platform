# config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ---------- 数据库 ----------
    DB_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

    # ---------- Ollama 相关 ----------
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:14b") # Example model

    # ---------- 邮件设置 ----------
    EMAIL_USER: str = os.getenv("EMAIL_USER", "zhaihao_n@126.com") # Replace with your actual email
    EMAIL_PASS: str = os.getenv("EMAIL_PASS", "glmdfpoaA8") # **IMPORTANT: Use 126 Mail Authorization Code**
    EMAIL_SMTP_SERVER: str = os.getenv("EMAIL_SMTP_SERVER", "smtp.126.com")
    EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "465")) # Use 465 for SSL by default for 126/163

    # ---------- Google Custom Search Engine ----------
    # Add these to your .env file or environment variables
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "AIzaSyB0Kv14UpjEDv59HEOV4ducTqaPk8633L8")
    GOOGLE_CX: Optional[str] = os.getenv("GOOGLE_CX", "533a067c36f9d48f1")

    # ---------- Embedding Model ----------
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", 'shibing624/text2vec-base-chinese')
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "510")) # Max tokens for embedding model

    # ---------- Report Saving ----------
    REPORT_SAVE_PATH: str = os.getenv("REPORT_SAVE_PATH", "reports") # Directory to save reports

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Create reports directory if it doesn't exist
if not os.path.exists(settings.REPORT_SAVE_PATH):
    os.makedirs(settings.REPORT_SAVE_PATH)