# db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

def get_engine():
    """创建并返回 PostgreSQL 引擎（连接字符串可通过环境变量或配置）。"""
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME", "postgres")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url, echo=False)
    return engine

def get_session():
    """创建并返回 SQLAlchemy 会话。"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
