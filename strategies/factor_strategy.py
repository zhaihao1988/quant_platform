# strategies/factor_strategy.py
import akshare as ak
import pandas as pd

def multi_factor_select(date):
    """
    多因子选股示例：综合动量和市值因子选股。
    - 动量因子：过去6个月收益率
    - 规模因子：市值（取小市值）
    """
    # 获取沪深300成分股列表
    symbols = ak.index_zh_a_cons("000300.XSHG")["const_code"].tolist()
    # 获取过去一年的价格，用于计算收益率
    start = (pd.to_datetime(date) - pd.DateOffset(months=6)).strftime("%Y%m%d")
    end = date
    returns = []
    for sym in symbols:
        df = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=start, end_date=end, adjust="")
        if df is None or df.empty: continue
        df = df.sort_values("日期")
        price_start = df.iloc[0]["收盘"]
        price_end = df.iloc[-1]["收盘"]
        ret = (price_end - price_start) / price_start if price_start!=0 else 0
        returns.append((sym, ret))
    df_ret = pd.DataFrame(returns, columns=["symbol", "6m_return"])
    # 选取收益率前30%的股票
    top_quantile = df_ret["6m_return"].quantile(0.7)
    selected = df_ret[df_ret["6m_return"] >= top_quantile]["symbol"].tolist()
    print("Momentum top quantile symbols:", selected[:10])
    return selected

if __name__ == "__main__":
    # 示例：在指定日期运行因子选股
    picks = multi_factor_select("20230401")
    print("Selected stocks:", picks)
