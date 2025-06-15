# scripts/init_db.py
import logging
from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError

# 导入 Base 和所有需要创建表的模型
from db.models import (
    Base,
    StockList,
    News,
    StockDaily,
    StockWeekly,
    StockMonthly,
    StockFinancial,
    StockDisclosure,
    StockDisclosureChunk,
    StockShareDetail
    # 如果还有其他模型，请确保也在这里导入
)
from db.database import get_engine_instance
from config.settings import settings  # 导入 settings 对象

logger = logging.getLogger(__name__)


def create_hnsw_vector_indexes(engine):
    """
    专门为 StockDisclosureChunk 表的 chunk_vector 列创建 HNSW 向量索引。
    """
    inspector = inspect(engine)
    table_name = StockDisclosureChunk.__tablename__
    column_name = 'chunk_vector'
    # 索引名称应与 models.py 中定义的名称完全一致
    index_name = 'idx_chunk_vector_hnsw_cosine'

    if table_name not in inspector.get_table_names():
        logger.warning(f"表 {table_name} 不存在，无法为其创建 HNSW 向量索引。")
        return

    vector_ops_class = 'vector_cosine_ops'

    # 使用 settings.py 中定义的 HNSW 参数
    sql = text(f"""
    CREATE INDEX IF NOT EXISTS {index_name}
    ON {table_name}
    USING hnsw ({column_name} {vector_ops_class})
    WITH (m = {settings.PGVECTOR_HNSW_M}, ef_construction = {settings.PGVECTOR_HNSW_EF_CONSTRUCTION});
    """)

    with engine.begin() as conn:
        try:
            logger.info(
                f"尝试在 {table_name}({column_name}) 上创建或确认 HNSW 向量索引 ({index_name}) 使用 {vector_ops_class} ...")
            conn.execute(sql)
            logger.info(f"✅ HNSW 向量索引 {index_name} 已检查或创建。")
        except OperationalError as oe:
            logger.error(f"创建 HNSW 向量索引 {index_name} 失败 (OperationalError): {oe}", exc_info=True)
            logger.error("请确保 PostgreSQL 已安装并启用了 'vector' 扩展，并且指定的向量操作符类可用。")
        except Exception as e:
            logger.error(f"创建 HNSW 向量索引 {index_name} 失败: {e}", exc_info=True)


def init_db_main():
    engine = get_engine_instance()
    if engine is None:
        logger.critical("数据库引擎未初始化，无法执行 init_db。")
        return

    logger.info("--- 开始数据库初始化 (HNSW 索引版) ---")

    # 步骤 1: 基于 SQLAlchemy 模型创建/检查所有表
    logger.info("步骤 1: 基于 SQLAlchemy 模型创建/检查所有表...")
    try:
        # create_all 会自动创建 models.py 中通过 Index() 定义的索引
        Base.metadata.create_all(bind=engine)
        logger.info("✅ 所有模型定义的表及其常规索引已检查/创建。")

    except Exception as e:
        logger.error(f"执行 Base.metadata.create_all 时出错: {e}", exc_info=True)
        return

    # 步骤 2: 再次确认自定义的 HNSW 向量索引已创建
    # 这一步提供了双重保障
    logger.info("步骤 2: 再次确认自定义的 HNSW 向量索引...")
    create_hnsw_vector_indexes(engine)

    logger.info("--- ✅ 数据库初始化完成 (HNSW 索引版) ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("脚本启动，db.models 中的模型已通过顶部 import 加载。")

    init_db_main()