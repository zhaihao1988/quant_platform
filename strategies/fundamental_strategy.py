# strategies/fundamental_strategy.py
import akshare as ak

def fundamental_filter():
    """
    基本面选股示例：筛选市盈率（PE）较低且净利润增长率较高的股票。
    """
    # 获取所有A股列表
    stock_list = ak.stock_info_a_code_name()["code"].tolist()
    filtered = []
    for sym in stock_list[:200]:  # 示例只处理前200只
        pe = ak.stock_zh_a_spot(symbol=sym).at[0, "市盈率"]
        profit = ak.stock_financial_report_sina(symbol=sym, report_type="A", year=2022, quarter=4)
        net_profit = profit["净利润"][0] if not profit.empty else None
        if pe and pe < 20 and net_profit and net_profit > 1e8:
            filtered.append(sym)
    print(f"Filtered {len(filtered)} stocks by fundamentals:", filtered[:5])
    return filtered

if __name__ == "__main__":
    picks = fundamental_filter()
