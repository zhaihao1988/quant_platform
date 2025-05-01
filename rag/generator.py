# rag/generator.py

from datetime import datetime
from typing import List, Dict

import pandas as pd
from langchain.chains import LLMChain  # 用于链式调用 LLMChain :contentReference[oaicite:0]{index=0}
from langchain.llms import Ollama  # 本地 Ollama 接口 :contentReference[oaicite:1]{index=1}

from config.settings import settings  # 全局配置，包括模型名、邮件等
from rag.loader import load_price_data, load_financial_data, load_announcements  # 数据加载模块
from rag.retriever import retrieve_context  # 向量检索模块（Chroma） :contentReference[oaicite:2]{index=2}
from rag.prompting import deep_template  # 自定义深度分析 PromptTemplate

def summarize_price(df: pd.DataFrame) -> str:
    """
    将交易 DataFrame 汇总为文本，包括最近 5 日平均收盘价、最高、最低等信息。
    """
    if df.empty:
        return "无可用交易数据。"
    recent = df.head(5)
    avg_close = recent["close"].mean()
    high = recent["high"].max()
    low = recent["low"].min()
    return f"最近5日平均收盘价 {avg_close:.2f} 元，最高 {high:.2f} 元，最低 {low:.2f} 元。"

def summarize_financial(data: Dict) -> str:
    """
    将财务 JSONB 数据转换为关键信息段落，列出主要字段及数值。
    """
    if not data:
        return "无可用财务数据。"
    parts = [f"{k}：{v}" for k, v in data.items()]
    return "；".join(parts)

def generate_deep_report(
    company_name: str,
    symbol: str,
    peer_companies: List[str] = None
) -> str:
    """
    1. 加载交易、财务与公告数据
    2. 使用 RAG 检索公告上下文
    3. 基于深度分析模板调用 Ollama 本地模型生成 Markdown 报告
    """
    # 1) 数据加载
    price_df = load_price_data(symbol)  # 从 PostgreSQL 读取日线数据
    fin_data = load_financial_data(symbol)  # 读取最新一期利润表 JSONB
    announcements = load_announcements(symbol, top_n=5)  # 抓取最近公告正文

    # 2) RAG 检索公告上下文
    # 合并公告文本用于检索
    query = " ".join(announcements)
    news_context = "\n".join(retrieve_context(query, k=3))  # Chroma 相似度搜索 :contentReference[oaicite:3]{index=3}

    # 3) 汇总交易与财务文本
    price_summary = summarize_price(price_df)
    fin_summary = summarize_financial(fin_data)

    # 4) 准备模板参数
    today = datetime.now().strftime("%Y-%m-%d")
    params = {
        "company_name": company_name,
        "symbol": symbol,
        "date": today,
        "core_business": "请补充核心业务描述",
        "product_1": "产品A",
        "prod1_share": "XX",
        "prod1_barrier": "XXX",
        "product_2": "产品B",
        "prod2_share": "YY",
        "prod2_advantage": "YYY",
        "bus2b2c": "70/30",
        "dist_prop": "60/40",
        "top5_client_share": "40",
        "rev_2022": "100",
        "rev_2023": "120",
        "rev_2024": "150",
        "rev_cause1": "市场扩张",
        "rev_cause2": "价格上涨",
        "profit_2022": "10",
        "profit_2023": "12",
        "profit_2024": "15",
        "profit_cause1": "成本控制",
        "profit_cause2": "产品结构优化",
        "gm_2022": "25",
        "gm_2023": "27",
        "gm_2024": "30",
        "gm_cause": "原材料成本下降",
        "rd_2022": "5",
        "rd_2023": "6",
        "rd_2024": "7",
        "rd_proj_progress": "按计划进行",
        "recent_prod": "新产品C获批",
        "recent_prod_adv": "性能领先",
        "dom_hosp": "10",
        "oversea_markets": "欧洲, 北美",
        "mna_partners": "公司X",
        "mna_synergy": "渠道整合",
        "comp_cagr": "12",
        "comp_gm": "28",
        "comp_rd": "6",
        "comp_price_return": "50",
        "comp_diff": "专利壁垒",
        "peer1": peer_companies[0] if peer_companies else "对标A",
        "peer1_cagr": "10",
        "peer1_gm": "26",
        "peer1_rd": "5",
        "peer1_price_return": "45",
        "peer1_diff": "规模效应",
        "peer2": peer_companies[1] if peer_companies else "对标B",
        "peer2_cagr": "8",
        "peer2_gm": "24",
        "peer2_rd": "4",
        "peer2_price_return": "40",
        "peer2_diff": "渠道覆盖",
        "peer3": peer_companies[2] if peer_companies else "对标C",
        "peer3_cagr": "9",
        "peer3_gm": "25",
        "peer3_rd": "5",
        "peer3_price_return": "42",
        "peer3_diff": "品牌优势",
        "industry_policy": "集采豁免",
        "industry_tech": "AI+医疗设备",
        "industry_demand": "老龄化率 18%",
        "company_moat": "专利20项/市占率15%",
        "mgmt_capacity": "新建生产线",
        "mgmt_oversea": "东南亚",
        "exp_rev_cagr": "15",
        "exp_profit_cagr": "18",
        "risk_ar_days": "60",
        "risk_ar_avg": "45",
        "risk_inv_share": "20",
        "risk_inv_delta": "5",
        "risk_competitor": "竞品Y",
        "risk_price_drop": "10",
        "risk_mkt_share_gain": "3",
        "risk_channel_drop": "2",
        "risk_policy1": "DRG 改革风险",
        "risk_policy2": "审批延迟风险",
        "cata_tech": "首个国产三类证",
        "cata_model": "设备+耗材+服务",
        "cata_short_term": "产品D 5月上市",
        "inc_rev_target": "200",
        "inc_rev_yoy": "20",
        "inc_profit_target": "25",
        "inc_grant_price": "10",
        "inc_discount": "5",
        "inc_reserved_share": "10",
        "inc_staff_core": "5",
        "inc_staff_core_pct": "10",
        "inc_staff_exec": "3",
        "inc_staff_exec_pct": "5",
        "ir_focus1": "渠道库存消化",
        "ir_focus2": "新品放量节奏",
        "ir_guidance1": "智能化领域",
        "ir_guidance2": "10%",
        "val_pe": "30",
        "val_pe_avg": "25",
        "val_pe_cause": "成长溢价",
        "val_peg": "2.0",
        "val_peg_avg": "1.8",
        "val_wacc": "10",
        "val_terminal_growth": "3",
        "val_price_low": "50",
        "val_price_high": "70",
        "val_margin": "20",
        "val_pe_pct": "80",
        "cata_industry_factors": "医保覆盖扩容",
        "last_trading_move": "涨幅2%，受研报影响"
    }

    # 5) 调用 Ollama 本地模型生成报告
    llm = Ollama(model=settings.OLLAMA_MODEL)  # 本地 Ollama 模型 :contentReference[oaicite:4]{index=4}
    chain = LLMChain(llm=llm, prompt=deep_template)  # LLMChain 链式调用 :contentReference[oaicite:5]{index=5}
    report_md = chain.run(**params)
    return report_md

if __name__ == "__main__":
    # 本地测试示例
    report = generate_deep_report(company_name="示例公司", symbol="000001", peer_companies=["600000","600004","600006"])
    print(report)
