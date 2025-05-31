# scripts/update_aggregated_data.py
import logging
import argparse
from sqlalchemy.orm import Session
from datetime import datetime
import os  # For path manipulation if needed
import sys  # For path manipulation if needed
from typing import List, Optional, Dict, Any
# --- 将项目根目录添加到 sys.path 以确保模块可被找到 ---
# 假设此脚本位于 quant_platform/scripts/ 目录下
# 项目根目录是 quant_platform/
# 则其父目录是 quant_platform 所在的目录
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_parent = os.path.dirname(os.path.dirname(current_dir))  # D:\project
    if project_root_parent not in sys.path:
        sys.path.insert(0, project_root_parent)
except NameError:  # __file__ is not defined, e.g. in interactive environment
    # 如果在交互式环境或特殊执行上下文中，可能需要手动设置项目路径
    # 或者确保PYTHONPATH已包含项目根目录的父目录
    pass

# 根据您的项目结构调整导入路径
# 假设 quant_platform 是您的顶级包名
from db.database import SessionLocal
from db import crud
from data_processing import aggregators  # 导入我们刚修改的 aggregators 模块
from utils import data_loader as dl  # 如果需要获取股票列表
from db.models import StockList  # 用于获取股票列表时指定返回类型

# 配置日志
# 建议在项目根目录的 __init__.py 或 main.py 中进行全局日志配置
# 这里为了脚本独立运行，也配置一下
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 确保日志输出到控制台
        # logging.FileHandler("update_aggregated_data.log") # 可选：输出到文件
    ]
)
logger = logging.getLogger(__name__)


def main(stock_codes_str: Optional[str] = None, full_rebuild: bool = True, lookback_days: int = 90):
    """
    主函数，用于更新股票的周线和月线聚合数据。

    :param stock_codes_str: 以逗号分隔的股票代码字符串，例如 "000001,600000"。如果为 None，则处理所有股票。
    :param full_rebuild: 是否对指定股票执行完全历史数据回填。
    :param lookback_days: 在非 full_rebuild 模式下，每日更新时回溯多少个日历日的数据进行重算。
    """
    logger.info("开始执行周/月聚合数据更新任务...")
    db: Session = SessionLocal()
    try:
        target_stocks_info: List[StockList] = []  # 用于存储 StockList 对象

        if stock_codes_str:
            stock_codes_list = [code.strip() for code in stock_codes_str.split(',') if code.strip()]
            logger.info(f"将处理指定的股票列表: {stock_codes_list}")
            # 如果只给代码，可能需要从数据库获取完整的 StockList 对象（如果后续需要name等信息）
            # 但 aggregators 函数主要需要 stock_code，所以列表也可以
            # 为了与下面获取所有股票的逻辑保持一致，我们也尝试获取 StockList 对象
            for code in stock_codes_list:
                stock_info = crud.get_stock_list_info(db, code)  # 假设 crud 中有此函数返回 StockList 对象
                if stock_info:
                    target_stocks_info.append(stock_info)
                else:
                    logger.warning(f"指定的股票代码 {code} 在 stock_list 中未找到，将跳过。")
        else:
            logger.info("未指定股票列表，将尝试获取所有股票代码...")
            # 从 stock_list 表获取所有股票对象
            target_stocks_info = crud.get_all_stocks(db)  # 假设 crud 中有此函数返回 List[StockList]
            if target_stocks_info:
                logger.info(f"获取到 {len(target_stocks_info)} 只股票进行处理。")
            else:
                logger.error("无法从数据库获取股票列表，任务中止。")
                return

        if not target_stocks_info:
            logger.info("没有需要处理的股票，任务结束。")
            return

        total_stocks = len(target_stocks_info)
        for i, stock_info_obj in enumerate(target_stocks_info):
            stock_code = stock_info_obj.code  # 从 StockList 对象获取股票代码
            logger.info(
                f"--- 开始处理股票: {stock_code} ({getattr(stock_info_obj, 'name', '')}) ({i + 1}/{total_stocks}) ---")

            if full_rebuild:
                logger.info(f"对 {stock_code} 执行历史数据完全回填...")
                aggregators.calculate_and_store_historical_periods(db, stock_code)
            else:
                # 每日例行更新：更新最近N天的数据
                logger.info(f"对 {stock_code} 执行最近 {lookback_days} 天数据的例行更新...")
                aggregators.update_recent_periods_data(db, stock_code, lookback_calendar_days=lookback_days)

            logger.info(f"--- 完成处理股票: {stock_code} ---")

        logger.info("所有指定股票的周/月聚合数据更新任务完成。")

    except Exception as e:
        logger.error(f"周/月聚合数据更新任务执行失败: {e}", exc_info=True)
        if db:  # 确保db已初始化
            db.rollback()  # 发生任何错误时回滚
    finally:
        if db:  # 确保db已初始化
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新股票的周线和月线聚合数据。")
    parser.add_argument(
        "--stocks",
        type=str,
        help="可选，以逗号分隔的股票代码字符串。如果未提供，则处理所有股票。"
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",  # 指定此参数即为True，不指定为False
        help="可选，如果指定，则对目标股票执行历史数据完全回填，而不是每日增量更新。"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,  # 默认回溯90个日历日
        help="可选，在每日更新模式下，回溯多少个日历日的数据进行重算。默认90天。"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="可选，如果指定，在开始聚合前先尝试运行数据库初始化/建表逻辑。"
    )

    args = parser.parse_args()

    if args.init_db:
        logger.info("请求执行数据库初始化...")
        try:
            # 假设 init_db_command 是一个可以调用的函数
            # 或者您直接调用 init_db() 函数
            from quant_platform.scripts.init_db import init_db as run_db_init_main_func

            logger.info("正在运行 init_db()...")
            run_db_init_main_func()  # 调用 init_db.py 中的主函数
            logger.info("数据库初始化完成。")
        except ImportError:
            logger.error("无法导入 init_db 函数。请确保 quant_platform.scripts.init_db.py 可访问且包含 init_db 函数。")
        except Exception as e_init:
            logger.error(f"数据库初始化过程中出错: {e_init}", exc_info=True)

    main(stock_codes_str=args.stocks, full_rebuild=args.full_rebuild, lookback_days=args.lookback)