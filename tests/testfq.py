# test_adj_params.py
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
stock_code = "000026.SZ"
test_date = "2025-05-23"
field_to_get = "close"

print("\n" + "=" * 50)
print(f"开始测试，股票: {stock_code}, 日期: {test_date}, 字段: {field_to_get}")
print("=" * 50)

price_adj_f = None
adj_f = None

# --- 方式一：使用 "adj=F" ---
try:
    print(f"\n1. 正在使用 'adj=F' 参数请求数据...")
    wind_data_1 = w.wsd(stock_code, field_to_get, test_date, test_date, "adj=F")

    if wind_data_1.ErrorCode != 0:
        print(f"   ❌ 请求失败, ErrorCode: {wind_data_1.ErrorCode}")
    elif not wind_data_1.Data or not wind_data_1.Data[0]:
        print("   ℹ️ 请求成功，但未返回数据。")
    else:
        adj_f = wind_data_1.Data[0][0]
        print(f"   ✅ 请求成功, 'adj=F' 返回值: {adj_f}")

except Exception as e:
    print(f"   ❌ 执行时发生错误: {e}")

# --- 方式二：使用 "PriceAdj=F" ---
try:
    print(f"\n2. 正在使用 'PriceAdj=F' 参数请求数据...")
    wind_data_2 = w.wsd(stock_code, field_to_get, test_date, test_date, "PriceAdj=F")

    if wind_data_2.ErrorCode != 0:
        print(f"   ❌ 请求失败, ErrorCode: {wind_data_2.ErrorCode}")
    elif not wind_data_2.Data or not wind_data_2.Data[0]:
        print("   ℹ️ 请求成功，但未返回数据。")
    else:
        price_adj_f = wind_data_2.Data[0][0]
        print(f"   ✅ 请求成功, 'PriceAdj=F' 返回值: {price_adj_f}")

except Exception as e:
    print(f"   ❌ 执行时发生错误: {e}")

# --- 3. 对比结果 ---
print("\n" + "=" * 50)
print("             测试结论")
print("=" * 50)

if adj_f is not None and price_adj_f is not None:
    if adj_f == price_adj_f:
        print("✅ 结论：两种参数写法的返回结果【完全相同】。")
    else:
        print("❌ 结论：两种参数写法的返回结果【不相同】！")
else:
    print("ℹ️ 未能获取到足够的数据进行比较。")

# --- 关闭API连接 ---
w.stop()