# probe.py (专用诊断脚本)
import akshare as ak
import pandas as pd

# --- 配置诊断参数 ---
SYMBOL_TO_PROBE = "000001"
START_DATE = "20250605"
END_DATE = "20250610"

print("=" * 80)
print("              AkShare 接口返回内容诊断探针")
print("=" * 80)
print(f"将诊断股票: {SYMBOL_TO_PROBE}，时间范围: {START_DATE} to {END_DATE}\n")


def probe_report_interface():
    """诊断接口 1: stock_zh_a_disclosure_report_cninfo"""
    print("\n--- 正在诊断 [常规报告] 接口... ---")
    try:
        # 直接调用接口
        result = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=SYMBOL_TO_PROBE,
            start_date=START_DATE,
            end_date=END_DATE
        )

        # 打印返回结果的详细信息
        print(f"返回结果的类型: {type(result)}")

        if isinstance(result, pd.DataFrame):
            print(f"DataFrame 是否为空: {result.empty}")
            if not result.empty:
                print("DataFrame 的列名 (columns):")
                print(result.columns)
                print("\nDataFrame 的前5行内容 (head):")
                print(result.head())
        else:
            print("返回的原始结果:")
            print(result)

    except Exception as e:
        print(f"❌ 调用接口时直接发生异常: {e}")


def probe_relation_interface():
    """诊断接口 2: stock_zh_a_disclosure_relation_cninfo"""
    print("\n--- 正在诊断 [调研公告] 接口... ---")
    try:
        # 直接调用接口
        result = ak.stock_zh_a_disclosure_relation_cninfo(
            symbol=SYMBOL_TO_PROBE,
            start_date=START_DATE,
            end_date=END_DATE
        )

        # 打印返回结果的详细信息
        print(f"返回结果的类型: {type(result)}")

        if isinstance(result, pd.DataFrame):
            print(f"DataFrame 是否为空: {result.empty}")
            if not result.empty:
                print("DataFrame 的列名 (columns):")
                print(result.columns)
                print("\nDataFrame 的前5行内容 (head):")
                print(result.head())
        else:
            print("返回的原始结果:")
            print(result)

    except Exception as e:
        print(f"❌ 调用接口时直接发生异常: {e}")


if __name__ == "__main__":
    probe_report_interface()
    probe_relation_interface()
    print("\n" + "=" * 80)
    print("诊断结束。请将以上全部输出结果发给我进行分析。")
    print("=" * 80)