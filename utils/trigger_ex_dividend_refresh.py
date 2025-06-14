# maintenance/trigger_ex_dividend_refresh.py
import time
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# 导入您项目中的模块
from db.database import get_engine_instance
from db.models import StockDaily, StockWeekly, StockMonthly

# --- 配置 ---
try:
    from WindPy import w

    if not w.isconnected():
        w.start()
    print("✅ Wind API 连接成功。")
except ImportError:
    print("❌ 错误: 未找到 WindPy 库。")
    exit()

engine = get_engine_instance()
Session = sessionmaker(bind=engine)


# === 以下是从您同步脚本中复制并集成的核心数据获取函数 ===

def fetch_data_by_period(symbol_with_suffix: str, start_date: str, end_date: str, period: str) -> pd.DataFrame:
    """一个通用的数据获取函数，根据周期参数获取数据。"""
    period_map = {'D': '日线', 'W': '周线', 'M': '月线'}
    wind_fields = "open,high,low,close,volume,amt,swing,pct_chg,chg,turn"
    try:
        wind_data = w.wsd(symbol_with_suffix, wind_fields, start_date, end_date, f"adj=F;Period={period}")
        if wind_data.ErrorCode != 0 or not wind_data.Data:
            return pd.DataFrame()

        df = pd.DataFrame(wind_data.Data, index=wind_data.Fields).T
        df['date'] = wind_data.Times
        df.columns = df.columns.str.lower()

        df.rename(columns={"amt": "amount", "swing": "amplitude", "pct_chg": "pct_change", "chg": "price_change",
                           "turn": "turnover"}, inplace=True)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce') / 100
        df["symbol"] = symbol_with_suffix.split('.')[0]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        print(f"❌ 拉取 {symbol_with_suffix} [{period_map.get(period, '')}]数据时失败: {e}")
        return pd.DataFrame()


# ==========================================================

def run_ex_dividend_refresh():
    """
    主执行函数：检查并刷新因除权除息需要更新数据的股票。
    """
    db_session = Session()
    try:
        # 1. 获取上一个交易日
        today_str = datetime.now().strftime("%Y-%m-%d")
        # w.tdaysoffset(-1, ...) 表示获取T-1日
        last_trade_day = w.tdaysoffset(-1, today_str, "").Data[0][0]
        last_trade_day_str = last_trade_day.strftime("%Y-%m-%d")
        print(f"检查 {last_trade_day_str} 的除权除息事件...")

        # 2. 从Wind获取在该日期发生除权除息的A股列表
        # reportdate: 公告日期; ex_date: 除权除息日
        # 我们基于 ex_date 来判断
        ex_dividend_data = w.weqr("a_shares_ex_dividend",
                                  f"startdate={last_trade_day_str};enddate={last_trade_day_str}")

        if ex_dividend_data.ErrorCode != 0 or not ex_dividend_data.Data[0]:
            print("ℹ️ 上一个交易日无股票发生除权除息，无需刷新。")
            return

        stocks_to_refresh = list(
            pd.DataFrame(ex_dividend_data.Data, index=ex_dividend_data.Fields).T['wind_code'].unique())

        if not stocks_to_refresh:
            print("ℹ️ 上一个交易日无股票发生除权除息，无需刷新。")
            return

        print(f"🔥 检测到 {len(stocks_to_refresh)} 只股票需要刷新历史数据: {stocks_to_refresh}")

        # 3. 遍历列表，逐一刷新
        for i, symbol_with_suffix in enumerate(stocks_to_refresh):
            symbol = symbol_with_suffix.split('.')[0]
            print(f"\n--- [{i + 1}/{len(stocks_to_refresh)}] 开始刷新 {symbol} ---")

            # 3.1 删除该股票的所有历史行情数据
            print(f"🧹 正在删除 {symbol} 的本地日线、周线、月线旧数据...")
            try:
                db_session.execute(text(f"DELETE FROM stock_daily WHERE symbol = '{symbol}'"))
                db_session.execute(text(f"DELETE FROM stock_weekly WHERE symbol = '{symbol}'"))
                db_session.execute(text(f"DELETE FROM stock_monthly WHERE symbol = '{symbol}'"))
                db_session.commit()
                print(f"✅ {symbol} 旧数据删除成功。")
            except Exception as e:
                db_session.rollback()
                print(f"❌ 删除 {symbol} 旧数据时失败，跳过此股票: {e}")
                continue

            # 3.2 重新拉取并写入完整历史数据
            start_date = "19900101"
            end_date = datetime.now().strftime("%Y%m%d")

            for period, table_name, model in [('D', 'stock_daily', StockDaily),
                                              ('W', 'stock_weekly', StockWeekly),
                                              ('M', 'stock_monthly', StockMonthly)]:
                print(f"⬇️ 正在为 {symbol} 重新拉取完整的{model.__tablename__}数据...")
                df = fetch_data_by_period(symbol_with_suffix, start_date, end_date, period)
                if not df.empty:
                    try:
                        df.to_sql(table_name, con=engine, index=False, if_exists="append")
                        print(f"✅ {symbol} 成功写入 {len(df)} 条 {model.__tablename__} 数据。")
                    except Exception as e:
                        print(f"❌ {symbol} 写入 {table_name} 时失败: {e}")
                time.sleep(1)  # API限速

    except Exception as e:
        print(f"❌ 维护脚本运行时发生未知错误: {e}")
    finally:
        db_session.close()
        print("\n数据库会话已关闭。")


if __name__ == "__main__":
    print("=" * 50)
    print("  启动数据维护脚本：检查并刷新除权除息股票数据  ")
    print("=" * 50)

    run_ex_dividend_refresh()

    print("\n=" * 50)
    print("  维护脚本执行完毕。  ")
    print("=" * 50)