# D:\project\quant_platform\test_crud_function.py

import logging
from datetime import date
from sqlalchemy.orm import Session

# 确保这里的导入路径相对于项目根目录 quant_platform 是正确的
# 如果 quant_platform 本身就是根目录下的一个包名
try:
    from quant_platform.db.database import SessionLocal  # 假设 SessionLocal 用于获取数据库会话
    from quant_platform.db.models import StockDaily      # 导入 StockDaily 模型
    from quant_platform.db.crud import get_stock_daily_for_date # 导入要测试的函数
except ModuleNotFoundError:
    # 如果您直接在 quant_platform 目录内运行，可能需要调整上一级目录到 sys.path
    # 或者确保您的 PYTHONPATH 设置正确。
    # 更推荐的结构是将 quant_platform 视为一个可安装的包或项目主目录。
    # 以下是假设 quant_platform 就是项目的根目录的情况：
    import sys
    import os
    # 将当前脚本的父目录（即 D:\project\quant_platform）添加到Python路径
    # 使得 quant_platform.db.crud 等可以被找到
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #如果test_crud_function.py在tests子目录
    # 如果 test_crud_function.py 就在 D:\project\quant_platform\ 下，则下面的导入应该能工作
    # 当 D:\project\ 作为PYTHONPATH的一部分，或者 D:\project\quant_platform 本身是PYTHONPATH的一部分时

    # 假设您是从 D:\project\quant_platform 目录运行此脚本
    # 并且 D:\project\quant_platform 目录就是您的项目主包
    # 或者，如果 D:\project 是您的项目根，而 quant_platform 是其下的一个包
    # 那么导入应该是 from quant_platform.db...

    # 根据您提供的路径 D:\project\quant_platform\db\crud.py，
    # 最可能的情况是 D:\project\quant_platform 是您的主包或者PYTHONPATH的一部分
    from db.database import SessionLocal
    from db.models import StockDaily
    from db.crud import get_stock_daily_for_date


# 配置日志，以便看到 crud 函数中的日志输出
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """
    主函数，用于测试 get_stock_daily_for_date 功能。
    """
    db: Session = SessionLocal() # 获取数据库会话

    symbol_to_test = "601083"
    date_to_test = date(2025, 5, 30) # 确保这个日期在您的数据库中有数据

    logger.info(f"开始测试 get_stock_daily_for_date 函数，股票代码: {symbol_to_test}, 日期: {date_to_test.isoformat()}")

    try:
        stock_daily_data = get_stock_daily_for_date(db, symbol_to_test, date_to_test)

        if stock_daily_data:
            logger.info(f"成功获取到股票 {symbol_to_test} 在 {date_to_test.isoformat()} 的日线数据:")
            logger.info(f"  股票代码 (Symbol): {stock_daily_data.symbol}") # 假设模型字段为 symbol
            logger.info(f"  日期 (Date): {stock_daily_data.date}")
            logger.info(f"  开盘价 (Open): {getattr(stock_daily_data, 'open', 'N/A')}") # 使用 getattr 以防字段名不同
            logger.info(f"  收盘价 (Close): {getattr(stock_daily_data, 'close', 'N/A')}")
            logger.info(f"  最高价 (High): {getattr(stock_daily_data, 'high', 'N/A')}")
            logger.info(f"  最低价 (Low): {getattr(stock_daily_data, 'low', 'N/A')}")
            logger.info(f"  成交量 (Volume): {getattr(stock_daily_data, 'volume', 'N/A')}")
            # 您可以根据 StockDaily 模型的实际字段打印更多信息
            # 例如: logger.info(f"  昨收 (Prev Close): {stock_daily_data.prev_close}")
        else:
            logger.warning(f"未能获取到股票 {symbol_to_test} 在 {date_to_test.isoformat()} 的日线数据。请检查数据库中是否存在该记录。")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    finally:
        logger.info("测试完成。关闭数据库会话。")
        db.close() # 关闭会话

if __name__ == "__main__":
    # 确保您的项目根目录 (D:\project\quant_platform) 在 PYTHONPATH 中，
    # 或者您从该目录运行此脚本。
    # 如果您直接从 D:\project\quant_platform 运行 `python test_crud_function.py`
    # 并且您的 db 目录就在 D:\project\quant_platform\db
    # 那么 `from db.crud import ...` 这样的导入应该是有效的。
    main()