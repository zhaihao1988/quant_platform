import akshare as ak

# 设置股票代码，例如 '600000' 表示浦发银行
stock_code = "600000"

# 获取资产负债表数据
balance_sheet_df = ak.stock_financial_report_sina(
    stock=stock_code, symbol="资产负债表"
)
print("资产负债表：")
print(balance_sheet_df)

# 获取利润表数据
income_statement_df = ak.stock_financial_report_sina(
    stock=stock_code, symbol="利润表"
)
print("\n利润表：")
print(income_statement_df)

# 获取现金流量表数据
cash_flow_df = ak.stock_financial_report_sina(
    stock=stock_code, symbol="现金流量表"
)
print("\n现金流量表：")
print(cash_flow_df)
