# scripts/sync_stock_share_data.py

import logging
# import argparse # 不再需要命令行参数解析
from datetime import date, datetime, timedelta  # timedelta 可能用于增量更新逻辑，暂时保留
import pandas as pd
import akshare as ak  # 确保 akshare 已安装
from sqlalchemy.orm import Session
import time  # 用于可能的API请求延时
from typing import Optional, List, Dict, Any
# 调整导入路径
import sys
import os

try:
    from db.database import SessionLocal
    from db import crud
    from db.models import StockList  # 需要 StockList 来获取股票列表
    from config.settings import settings
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from db.database import SessionLocal
    from db import crud
    from db.models import StockList  # 需要 StockList
    from config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# parse_akshare_individual_info_df 函数保持不变 (与上一版本相同)
def parse_akshare_individual_info_df(symbol_from_arg: str, df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    将 akshare.stock_individual_info_em 返回的 DataFrame 解析为适合
    crud.update_stock_share_detail 函数的字典。
    symbol_from_arg: 从参数传入的股票代码，用于验证和填充。
    """
    if df is None or df.empty or 'item' not in df.columns or 'value' not in df.columns:
        logger.warning(f"[{symbol_from_arg}] 输入的 AkShare DataFrame 为空或格式不正确。")
        return None

    parsed_data = {}
    try:
        df_indexed = df.set_index('item')
    except Exception as e:
        logger.error(f"[{symbol_from_arg}] 设置 DataFrame 索引失败: {e}")
        return None

    def safe_get_value(key, data_type=str, current_symbol=""):
        try:
            val = df_indexed.loc[key, 'value']
            if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() in ['none', 'nan', '-', '--']:
                return None

            if data_type == int:
                val_str = str(val)
                if val_str.endswith(".0"):  # 处理 "123.0" 形式的整数
                    val_str = val_str[:-2]
                return int(val_str)
            if data_type == float:
                return float(val)
            if data_type == date:
                if isinstance(val, (int, float)):
                    val_str = str(int(val))
                    if len(val_str) == 8:  # YYYYMMDD 格式
                        return datetime.strptime(val_str, "%Y%m%d").date()
                elif isinstance(val, str):
                    val_str_cleaned = val.replace('-', '').replace('/', '')
                    if len(val_str_cleaned) == 8 and val_str_cleaned.isdigit():
                        return datetime.strptime(val_str_cleaned, "%Y%m%d").date()
                logger.warning(f"[{current_symbol}] 字段 '{key}' 的日期值 '{val}' 格式无法解析。")
                return None
            return str(val)
        except (KeyError, ValueError, TypeError) as e:
            if isinstance(e, KeyError):
                logger.debug(f"[{current_symbol}] 字段 '{key}' 未在 AkShare DataFrame 中找到。")
            else:
                logger.warning(f"[{current_symbol}] 解析字段 '{key}' (期望类型: {data_type.__name__}) 时出错: {e}. "
                               f"原始值: {df_indexed.loc[key, 'value'] if key in df_indexed.index else 'N/A'}")
            return None

    ak_symbol = safe_get_value('股票代码', current_symbol=symbol_from_arg)
    if not ak_symbol:
        logger.error(f"未能从 AkShare DataFrame 中为预期代码 '{symbol_from_arg}' 解析出'股票代码'字段。")
        return None
    if ak_symbol != symbol_from_arg:
        logger.warning(
            f"参数传入的股票代码 '{symbol_from_arg}' 与 AkShare 返回的股票代码 '{ak_symbol}' 不匹配。将使用参数传入的代码。")

    parsed_data['symbol'] = symbol_from_arg

    parsed_data['stock_name'] = safe_get_value('股票简称', current_symbol=symbol_from_arg)
    parsed_data['total_shares'] = safe_get_value('总股本', int, current_symbol=symbol_from_arg)
    parsed_data['float_shares'] = safe_get_value('流通股', int, current_symbol=symbol_from_arg)
    parsed_data['total_market_cap'] = safe_get_value('总市值', float, current_symbol=symbol_from_arg)
    parsed_data['float_market_cap'] = safe_get_value('流通市值', float, current_symbol=symbol_from_arg)
    parsed_data['industry'] = safe_get_value('行业', current_symbol=symbol_from_arg)
    parsed_data['listing_date'] = safe_get_value('上市时间', date, current_symbol=symbol_from_arg)
    parsed_data['data_source_date'] = date.today()

    if parsed_data.get('total_shares') is None:
        logger.warning(f"[{symbol_from_arg}] 未能解析出有效的'总股本'数据。可能影响依赖此数据的分析。")
        # 即使总股本为空，其他信息也可能有用，所以我们仍然返回字典
        # 但调用者可能需要根据总股本是否存在来做进一步判断

    return parsed_data


# sync_single_stock_share_detail 函数保持不变 (与上一版本相同)
def sync_single_stock_share_detail(db: Session, symbol: str) -> bool:
    """
    获取单个股票的股本信息并更新到数据库。
    """
    logger.info(f"开始为股票代码: {symbol} 同步股本详情...")
    try:
        ak_df = ak.stock_individual_info_em(symbol=symbol)
        if ak_df is None or ak_df.empty:
            logger.warning(f"[{symbol}] AkShare 未能返回数据。")
            return False
    except Exception as e:
        logger.error(f"[{symbol}] 调用 AkShare stock_individual_info_em 时发生错误: {e}", exc_info=True)
        return False

    parsed_data = parse_akshare_individual_info_df(symbol, ak_df)
    if not parsed_data:  # 如果解析后没有任何有效数据 (例如股票代码都无法解析)
        logger.error(f"[{symbol}] 解析 AkShare 数据失败或未获得有效数据。")
        return False

    # 即使解析出部分数据，也要确保 symbol 存在
    if not parsed_data.get('symbol'):
        logger.error(f"[{symbol}] 解析后的数据中缺少 symbol 字段。")
        return False

    updated_record = crud.update_stock_share_detail(db=db, symbol=parsed_data['symbol'], data=parsed_data)
    if updated_record:
        logger.info(f"[{symbol}] 股本详情已成功同步到数据库。")
        return True
    else:
        logger.error(f"[{symbol}] 同步股本详情到数据库失败。")
        return False


def main():
    logger.info("启动每日股本详情同步任务...")

    db: Optional[Session] = None
    try:
        db = SessionLocal()
        if db is None:
            logger.error("无法获取数据库会话。同步任务中止。")
            return

        logger.info("正在从 StockList 表获取所有股票代码...")
        symbols_to_sync = crud.get_all_symbols_from_stocklist(db)

        if not symbols_to_sync:
            logger.info("StockList 表中没有股票代码可供同步，或查询失败。任务结束。")
            return

        logger.info(f"准备同步 {len(symbols_to_sync)} 只股票的股本详情...")

        total_stocks = len(symbols_to_sync)
        success_count = 0
        failure_count = 0

        for i, code in enumerate(symbols_to_sync):
            logger.info(f"正在处理第 {i + 1}/{total_stocks} 只股票: {code}")
            if sync_single_stock_share_detail(db, code):
                success_count += 1
            else:
                failure_count += 1

            # --- 修正点 ---
            # 直接通过属性访问 settings 对象
            time.sleep(settings.AKSHARE_REQUEST_DELAY)
            # --- 修正点结束 ---

        logger.info(f"所有股票同步完成。总计: {total_stocks}, 成功: {success_count}, 失败: {failure_count}。")

    except Exception as e:
        logger.error(f"同步脚本主流程发生严重错误: {e}", exc_info=True)
    finally:
        if db:
            db.close()
            logger.info("数据库会话已关闭。")


if __name__ == "__main__":
    main()