import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# 设置 Pandas 显示选项，以便能完整显示内容
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 全局测试参数 ---
STOCK_CODE = "002780"  # 测试股票代码 (三夫户外)
COMPANY_NAME = "三夫户外"
DAYS_TO_TEST = 90      # 将测试时间范围扩大到90天，以增加找到“调研”公告的可能性

def test_stock_disclosure_report():
    """
    测试一：获取常规报告公告 (年报、半年报、季报、日常经营等)
    使用接口: ak.stock_zh_a_disclosure_report_cninfo
    """
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - timedelta(days=DAYS_TO_TEST)
    end_date_str = end_date_obj.strftime('%Y%m%d')
    start_date_str = start_date_obj.strftime('%Y%m%d')

    print(f"【测试 1/2】正在测试接口：ak.stock_zh_a_disclosure_report_cninfo (常规报告)")
    print(f"股票代码: {STOCK_CODE} ({COMPANY_NAME})")
    print(f"查询时间范围: {start_date_str} to {end_date_str}\n")

    try:
        # 调用 AkShare 接口
        disclosure_df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=STOCK_CODE,
            start_date=start_date_str,
            end_date=end_date_str
        )

        if disclosure_df is not None and not disclosure_df.empty:
            print("✅ [常规报告] 接口调用成功，返回数据如下：")
            print("-" * 80)
            print(disclosure_df)
            print("-" * 80)
            print(f"\n返回数据共有 {len(disclosure_df)} 条记录。")
            print(f"列名 (Columns): {disclosure_df.columns.tolist()}")
        elif disclosure_df is not None and disclosure_df.empty:
            print("⚪️ [常规报告] 接口调用成功，但在指定期间内没有找到相关公告。")
        else:
            print("⚠️ [常规报告] 接口调用返回 None。")

    except Exception as e:
        print(f"❌ [常规报告] 调用接口时发生错误: {e}")

def test_stock_disclosure_relation():
    """
    测试二：获取关系型公告 (投资者关系、调研、问询函等)
    使用接口: ak.stock_zh_a_disclosure_relation_cninfo
    """
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - timedelta(days=DAYS_TO_TEST)
    end_date_str = end_date_obj.strftime('%Y%m%d')
    start_date_str = start_date_obj.strftime('%Y%m%d')

    print(f"【测试 2/2】正在测试接口：ak.stock_zh_a_disclosure_relation_cninfo (调研等关系公告)")
    print(f"股票代码: {STOCK_CODE} ({COMPANY_NAME})")
    print(f"查询时间范围: {start_date_str} to {end_date_str}\n")

    try:
        # 调用 AkShare 获取“调研”等关系公告的接口
        disclosure_df = ak.stock_zh_a_disclosure_relation_cninfo(
            symbol=STOCK_CODE,
            start_date=start_date_str,
            end_date=end_date_str
        )

        if disclosure_df is not None and not disclosure_df.empty:
            print("✅ [调研公告] 接口调用成功，返回数据如下：")
            print("-" * 80)
            print(disclosure_df)
            print("-" * 80)
            print(f"\n返回数据共有 {len(disclosure_df)} 条记录。")
            print(f"列名 (Columns): {disclosure_df.columns.tolist()}")
        elif disclosure_df is not None and disclosure_df.empty:
            print("⚪️ [调研公告] 接口调用成功，但在指定期间内没有找到相关公告。")
        else:
            print("⚠️ [调研公告] 接口调用返回 None。")

    except Exception as e:
        print(f"❌ [调研公告] 调用接口时发生错误: {e}")


if __name__ == "__main__":
    test_stock_disclosure_report()
    print("\n" + "="*80 + "\n")  # 添加分割线，让输出更清晰
    test_stock_disclosure_relation()