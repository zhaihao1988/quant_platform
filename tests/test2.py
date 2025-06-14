import akshare as ak
import pandas as pd

symbol_to_test = "000004"
date_to_test = "20250605" # 今天 (根据模拟时间)

print(f"--- 开始直接测试 AkShare 功能 ---")
print(f"股票代码: {symbol_to_test}")
print(f"查询日期: 从 {date_to_test} 到 {date_to_test}")

try:
    df_disclosure = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=symbol_to_test,
        market="沪深京",
        category="",
        start_date=date_to_test,
        end_date=date_to_test,
    )

    if df_disclosure is None:
        print("结果: AkShare 返回了 None")
    elif df_disclosure.empty:
        print("结果: AkShare 返回了一个空的 DataFrame")
        print("列名: (空DataFrame没有列名，或根据情况打印)")
    else:
        print("结果: AkShare 成功返回了 DataFrame")
        print("DataFrame 的列名是:")
        print(df_disclosure.columns.tolist())
        print("DataFrame 的前几行数据是:")
        print(df_disclosure.head())
        print("DataFrame 的数据类型是:")
        print(df_disclosure.dtypes)

except Exception as e:
    print(f"直接调用 AkShare 时发生错误: {e}")
    import traceback
    traceback.print_exc() # 打印完整的错误堆栈

print(f"--- AkShare 功能直接测试结束 ---")