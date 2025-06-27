# scripts/sync_stock_list.py
import akshare as ak
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db.database import get_engine_instance
from db.models import StockList, Base
from sqlalchemy import text

def sync_stock_list():
    print("📥 开始同步全 A 股股票列表...")
    df = ak.stock_info_a_code_name()
    df.rename(columns={"code": "code", "name": "name"}, inplace=True)

    # 扩展字段（可选）
    df["area"] = None
    df["industry"] = None
    df["list_date"] = None

    engine = get_engine_instance()
    Base.metadata.create_all(engine)  # 确保表存在
    with engine.connect() as connection:
        # 使用 TRUNCATE ... CASCADE 清空 stock_list 表及相关依赖
        # 这比 DROP TABLE 更安全，因为它只删除数据，不删除表结构
        connection.execute(text("TRUNCATE TABLE stock_list RESTART IDENTITY CASCADE"))
        connection.commit()
    df.to_sql("stock_list", con=engine, if_exists="append", index=False)
    print(f"✅ 成功同步 {len(df)} 支 A 股股票到 stock_list 表")

if __name__ == "__main__":
    sync_stock_list()
