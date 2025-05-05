# utils/data_loader.py
import pandas as pd
from db.database import get_engine_instance

def load_daily_data(symbol: str, start_date: str, end_date: str, fields=None) -> pd.DataFrame:
    """
    从数据库加载指定股票和时间段的日线数据。
    :param symbol: 股票代码，如 '000001'
    :param start_date: 起始日期 '2023-01-01'
    :param end_date: 结束日期 '2023-12-31'
    :param fields: 需要的字段列表，默认全字段
    """
    engine = get_engine_instance()
    if not fields:
        fields = "*"
    else:
        fields = ", ".join(fields)

    query = f"""
    SELECT {fields} FROM stock_daily
    WHERE symbol = '{symbol}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    df = pd.read_sql(query, con=engine, parse_dates=["date"])
    return df
