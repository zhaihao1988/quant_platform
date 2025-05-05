# scripts/init_db.py
import logging
import logging
from sqlalchemy import text, inspect, Index
from db.models import Base, StockDisclosureChunk, StockDisclosure # 从 db.models 导入共享的 Base 和模型

from db.database import get_engine_instance
from config.settings import settings

logger = logging.getLogger(__name__)

def add_missing_columns(engine):
    """检查并添加缺失的列"""
    inspector = inspect(engine)

    if 'stock_disclosure' in inspector.get_table_names():
        existing_columns = [col['name'] for col in inspector.get_columns('stock_disclosure')]

        # 使用 SQLAlchemy 的 text() 函数包装 SQL 语句
        if 'raw_content' not in existing_columns:
            with engine.begin() as conn:  # 使用 begin() 自动提交事务
                conn.execute(text("ALTER TABLE stock_disclosure ADD COLUMN raw_content TEXT"))
                print("✅ 已添加 raw_content 列到 stock_disclosure 表")

        if 'content_vector' not in existing_columns:
            with engine.begin() as conn:
                # 对于 pgvector 类型，需要使用正确的语法
                conn.execute(
                    text(f"ALTER TABLE stock_disclosure ADD COLUMN content_vector vector({settings.EMBEDDING_DIM})"))
                print(f"✅ 已添加 content_vector 列到 stock_disclosure 表 (维度: {settings.EMBEDDING_DIM})")

def add_or_modify_columns(engine):
    """检查并调整表结构，适应新的 Schema"""
    inspector = inspect(engine)
    with engine.begin() as conn: # 使用事务
        # --- 处理 stock_disclosure 表 ---
        if 'stock_disclosure' in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns('stock_disclosure')]
            # 确保 raw_content 列存在
            if 'raw_content' not in existing_columns:
                try:
                    conn.execute(text("ALTER TABLE stock_disclosure ADD COLUMN raw_content TEXT"))
                    logger.info("✅ 已添加 raw_content 列到 stock_disclosure 表")
                except Exception as e:
                    logger.error(f"添加 raw_content 列失败: {e}") # 添加错误处理

            # 如果 content_vector 列存在，则删除它 (迁移逻辑)
            if 'content_vector' in existing_columns:
                logger.warning("发现旧的 content_vector 列，正在尝试删除...")
                try:
                    conn.execute(text("ALTER TABLE stock_disclosure DROP COLUMN content_vector"))
                    logger.info("✅ 已从 stock_disclosure 表删除旧的 content_vector 列")
                except Exception as e:
                    logger.error(f"删除 stock_disclosure.content_vector 列失败: {e}. 可能需要手动处理。")
        else:
             logger.warning("stock_disclosure 表不存在，将由 create_all 创建。")

        # --- 检查 stock_disclosure_chunk 表存在性 ---
        # 这里的日志只是确认表是否会被 create_all 处理，实际操作在 create_all 完成
        if 'stock_disclosure_chunk' not in inspector.get_table_names():
             logger.warning("stock_disclosure_chunk 表不存在，将由 create_all 创建。")
        else:
            logger.info("stock_disclosure_chunk 表已存在。")


def create_vector_index(engine):
    """在 stock_disclosure_chunk 表上创建 pgvector 索引"""
    # 检查表是否存在，避免在表不存在时尝试创建索引
    inspector = inspect(engine)
    if StockDisclosureChunk.__tablename__ not in inspector.get_table_names():
        logger.error(f"表 {StockDisclosureChunk.__tablename__} 不存在，无法创建索引。请先确保表已创建。")
        return

    index_name = 'idx_chunk_vector_cosine' # 为索引命名
    table_name = StockDisclosureChunk.__tablename__
    column_name = 'chunk_vector'

    # 使用 HNSW 索引
    sql = text(f"""
    CREATE INDEX IF NOT EXISTS {index_name}
    ON {table_name}
    USING hnsw ({column_name} vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    """)

    with engine.begin() as conn:
        try:
            logger.info(f"尝试在 {table_name}({column_name}) 上创建或确认 HNSW 向量索引 ({index_name})...")
            conn.execute(sql)
            logger.info(f"✅ HNSW 向量索引 {index_name} 已存在或已创建。")
        except Exception as e:
            logger.error(f"创建向量索引 {index_name} 失败: {e}", exc_info=True) # 添加 exc_info=True

    # 创建其他需要的常规索引
    try:
         disclosure_id_index = Index('idx_chunk_disclosure_id', StockDisclosureChunk.disclosure_id)
         # 使用 checkfirst=True 避免重复创建时出错
         disclosure_id_index.create(bind=engine, checkfirst=True)
         logger.info(f"✅ {table_name}(disclosure_id) 索引已存在或已创建。")
    except Exception as e:
         logger.error(f"创建 disclosure_id 索引失败: {e}", exc_info=True) # 添加 exc_info=True

def init_db():
    engine = get_engine_instance()
    if engine is None:
        logger.critical("数据库引擎未初始化，无法执行 init_db。")
        return

    logger.info("--- 开始数据库初始化 ---")

    # 1. (可选) 运行列修改/迁移逻辑
    logger.info("步骤 1: 检查并调整现有表结构...")
    add_or_modify_columns(engine)

    # 2. 创建所有在 Base 中定义的表
    logger.info("步骤 2: 创建或更新 SQLAlchemy 模型定义的表...")
    try:
        # 确保 db/models.py 中所有模型都已加载并继承自正确的 Base
        # 现在调用 create_all 时，它应该知道 StockDisclosureChunk 了
        Base.metadata.create_all(engine)
        logger.info("✅ SQLAlchemy 模型对应的表已检查/创建。")
    except Exception as e:
        logger.error(f"执行 Base.metadata.create_all 时出错: {e}", exc_info=True)
        return # Stop if table creation fails

    # 3. 创建必要的索引
    logger.info("步骤 3: 创建或确认必要的索引 (特别是向量索引)...")
    create_vector_index(engine) # 调用索引创建函数

    logger.info("--- ✅ 数据库初始化完成 ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 确保模型模块被加载
    try:
        import db.models
        logger.info("已加载 db.models 模块。")
    except ImportError as e:
        logger.error(f"无法导入 db.models: {e}")
        # 根据情况决定是否退出

    init_db()
