# scripts/sync_mda_chunk.py
import logging
import time
import argparse  # 导入 argparse 模块
from typing import Optional

from db.database import SessionLocal
from db.crud import get_all_symbols_from_stocklist
from data_processing.process_mda import process_and_store_mda

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_mda_chunk.log", mode='a'), # 保存到日志文件
        logging.StreamHandler() # 同时在控制台输出
    ]
)
logger = logging.getLogger(__name__)


def sync_all_mda(start_symbol: Optional[str] = None):
    """
    对数据库中 stock_list 表里的所有股票，执行董事会经营评述(MDA)的同步和向量化。
    :param start_symbol: 如果提供，将从该股票代码开始处理。
    """
    logger.info("--- 启动全量董事会经营评述(MDA)同步任务 ---")
    if start_symbol:
        logger.info(f"将从股票代码 {start_symbol} 继续执行任务。")
    
    db_session = SessionLocal()
    start_time = time.time()
    
    try:
        # 1. 获取所有股票代码
        all_symbols = get_all_symbols_from_stocklist(db_session)
        if not all_symbols:
            logger.warning("未能从数据库获取到任何股票代码，任务中止。")
            return
            
        total_stocks = len(all_symbols)
        logger.info(f"将要处理 {total_stocks} 只股票。")
        
        # --- 断点续传逻辑 ---
        start_index = 0
        if start_symbol:
            try:
                start_index = all_symbols.index(start_symbol)
                logger.info(f"找到起始股票 {start_symbol} 在列表中的位置: {start_index}。将从该位置开始处理。")
            except ValueError:
                logger.error(f"指定的起始股票代码 {start_symbol} 不在股票列表中！任务将从头开始。")
        
        symbols_to_process = all_symbols[start_index:]
        # --- 断点续传逻辑结束 ---

        # 2. 依次处理每只股票
        # 使用 enumerate 的 start 参数，确保日志中的进度是正确的
        for i, symbol in enumerate(symbols_to_process, start=start_index):
            logger.info(f"--- [{i + 1}/{total_stocks}] 正在处理股票: {symbol} ---")
            try:
                process_and_store_mda(db_session, symbol)
                logger.info(f"--- 股票 {symbol} 处理完成。 ---")
            except Exception as e:
                # 即使单个股票处理失败，也记录错误并继续处理下一个
                logger.error(f"处理股票 {symbol} 时发生未预料的错误: {e}", exc_info=True)
                # 重新创建 session 可能是个好主意，以防 session 状态损坏
                db_session.rollback()
            
            # 可以在这里加入一个小的延时，避免请求过于频繁
            # time.sleep(1) 

    except Exception as e:
        logger.error(f"在同步任务主循环中发生严重错误: {e}", exc_info=True)
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- 全量同步任务结束，总耗时: {duration:.2f} 秒 ---")
        db_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步所有股票的董事会经营评述(MDA)并进行向量化。")
    parser.add_argument(
        '--start_symbol', 
        type=str, 
        nargs='?', # 使其成为可选的位置参数
        const=None,
        default=None,
        help="（可选）指定一个股票代码，脚本将从该股票开始继续执行同步任务。"
    )
    args = parser.parse_args()

    sync_all_mda(start_symbol=args.start_symbol) 