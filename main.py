# main.py

from rag.loader import load_price_data, load_financial_data, load_announcements
from rag.generator import generate_report
from utils.emailer import send_report_email

def run_all(symbols: list[str]):
    for sym in symbols:
        # 加载数据
        price_df = load_price_data(sym)
        fin_data = load_financial_data(sym, report_type="benefit")
        # 生成报告
        report_md = generate_report(sym, price_df, fin_data)
        # 发送邮件
        subject = f"个股报告：{sym}"
        send_report_email("zhaihao_n@126.com", subject, report_md)
        print(f"✅ 已发送 {sym} 报告")

if __name__ == "__main__":
    # 如需全市场，可先从 db 拉所有 code，再传入
    symbols = ["000001", "600519"]  # 示例
    run_all(symbols)
