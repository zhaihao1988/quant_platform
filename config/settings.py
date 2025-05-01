import os
from pydantic_settings import BaseSettings
from pydantic import Field, RedisDsn


class Settings(BaseSettings):
    # ---------- 数据库 ----------
    DB_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    # ---------- Ollama 相关 ----------
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3-14b")
    # ---------- 邮件设置 ----------
    EMAIL_USER: str = os.getenv("EMAIL_USER", "zhaihao_n@126.com")
    EMAIL_PASS: str = os.getenv("EMAIL_PASS", "glmdfpoaA8")
    EMAIL_SMTP_SERVER: str = os.getenv("EMAIL_SMTP_SERVER", "smtp.126.com")
    EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "25"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
