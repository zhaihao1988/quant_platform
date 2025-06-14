# tests/test_snapshot_fields.py
import pandas as pd
from datetime import datetime

try:
    from WindPy import w

    if not w.isconnected():
        w.start()
    print("✅ Wind API 连接成功。")
except ImportError:
    print("❌ 错误: 未找到 WindPy 库，请先安装 `pip install windpy`。")
    exit()

# --- 测试参数 ---
# 您可以添加更多股票代码，用逗号隔开
STOCK_CODES = "000887.SZ,000001.SZ,600519.SH"
# 您需要查询的字段，用逗号隔开
FIELDS_TO_GET = "latestincentivedate,dividendyield2"

print("=" * 60)
print(f"开始测试，股票: {STOCK_CODES}")
print(f"目标字段: {FIELDS_TO_GET}")
print("=" * 60)

# 使用 w.wss 函数获取最新截面数据
# w.wss(codes, fields, options)
# options可以留空，默认获取最新的数据
try:
    print(f"正在使用 w.wss 函数请求数据...")
    wss_data = w.wss(STOCK_CODES, FIELDS_TO_GET)

    # --- 处理并打印结果 ---
    if wss_data.ErrorCode != 0:
        print(f"  ❌ 请求失败, ErrorCode: {wss_data.ErrorCode}")
        if wss_data.Data:
            print(f"     错误信息: {wss_data.Data[0][0]}")
    elif not wss_data.Data or not wss_data.Data[0]:
        print("  ℹ️ 请求成功，但未返回任何数据。")
    else:
        # wss 返回的数据结构适合直接转为 DataFrame
        # .Codes 是股票代码列表, .Fields 是字段列表, .Data 是一个二维列表
        # 我们将它转置(.T)一下，让股票代码作为行索引，更便于查看
        df = pd.DataFrame(wss_data.Data, index=wss_data.Fields, columns=wss_data.Codes).T

        print("  ✅ 请求成功，获取到以下数据:")
        print(df)

except Exception as e:
    print(f"  ❌ 执行时发生错误: {e}")

# --- 关闭API连接 ---
w.stop()
print("\n" + "=" * 60)
print("测试脚本执行完毕。")