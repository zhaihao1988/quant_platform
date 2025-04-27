# utils/db_utils.py
from db.database import get_engine
import pandas as pd

def insert_signal_results(df: pd.DataFrame):
    """
    批量将选股信号写入 signal_results 表。
    要求 df 包含列：signal_date, strategy, symbol
    """
    engine = get_engine()
    # 如果表不存在，可先手动运行建表脚本
    df.to_sql("signal_results", engine, if_exists="append", index=False)
