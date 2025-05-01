# rag/prompting.py

from langchain.prompts import PromptTemplate

DEEP_ANALYSIS_TEMPLATE = """
公司名称：{company_name}  
股票代码：{symbol}  
分析日期：{date}  

1. 主业分析  
核心业务：{core_business}  
产品矩阵：  
① {product_1}（收入占比{prod1_share}%，技术壁垒：{prod1_barrier}）  
② {product_2}（收入占比{prod2_share}%，差异化优势：{prod2_advantage}）  
商业模式：2B/2C占比{bus2b2c}%，经销/直销比例{dist_prop}，前五大客户占比{top5_client_share}%。

2. 近三年财务归因（2022-2024）  
指标|2022|2023|2024|变动主因  
---|---|---|---|---  
营收（亿）|{rev_2022}|{rev_2023}|{rev_2024}|①{rev_cause1} ②{rev_cause2}  
归母净利（亿）|{profit_2022}|{profit_2023}|{profit_2024}|①{profit_cause1} ②{profit_cause2}  
毛利率|{gm_2022}%|{gm_2023}%|{gm_2024}%|原材料/定价权变化：{gm_cause}  
研发费率|{rd_2022}%|{rd_2023}%|{rd_2024}%|关键项目：（进度：{rd_proj_progress}）

3. 近期积极变化（2024Q1至今）  
① 产品：（新取证/升级：{recent_prod}，竞品对比优势：{recent_prod_adv}）  
② 渠道：（国内新增{dom_hosp}家医院，海外进入{oversea_markets}市场）  
③ 战略：（并购/合作方：{mna_partners}，协同效应：{mna_synergy}）

4. 同业对比（Top3可比公司）  
公司|营收CAGR(3年)|2024毛利率|研发费率|股价3年涨跌幅|核心差异点  
---|---|---|---|---|---  
本公司|{comp_cagr}|{comp_gm}|{comp_rd}|{comp_price_return}|{comp_diff}  
{peer1}|{peer1_cagr}|{peer1_gm}|{peer1_rd}|{peer1_price_return}|{peer1_diff}  
{peer2}|{peer2_cagr}|{peer2_gm}|{peer2_rd}|{peer2_price_return}|{peer2_diff}  
{peer3}|{peer3_cagr}|{peer3_gm}|{peer3_rd}|{peer3_price_return}|{peer3_diff}

5. 行业与公司增长逻辑  
行业β：  
① 政策：{industry_policy}  
② 技术：{industry_tech}  
③ 需求：{industry_demand}  
公司α：  
① 护城河：{company_moat}  
② 管理层：扩产规划：{mgmt_capacity}，出海进度：{mgmt_oversea}  
预期增速：2025-2027年营收CAGR{exp_rev_cagr}%，利润CAGR{exp_profit_cagr}%。

6. 风险因素  
财务风险：  
① 应收账款周转{risk_ar_days}天（行业平均{risk_ar_avg}天）  
② 存货占比{risk_inv_share}%（同比变动{risk_inv_delta}%）  
市场风险：  
① 竞品：{risk_competitor}（价格降幅{risk_price_drop}%/市占率提升{risk_mkt_share_gain}%）  
② 渠道库存：经销商数量减少{risk_channel_drop}家  
政策风险：  
① {risk_policy1}  
② {risk_policy2}

7. 业务亮点与催化剂  
技术突破：{cata_tech}  
模式创新：{cata_model}  
短期催化剂：{cata_short_term}

8. 股权激励分析  
考核目标：2025营收≥{inc_rev_target}亿（同比+{inc_rev_yoy}%），2026净利润≥{inc_profit_target}亿  
授予价：{inc_grant_price}元（现价折扣率{inc_discount}%），预留股占比{inc_reserved_share}%  
激励对象：核心技术人员{inc_staff_core}人（占比{inc_staff_core_pct}%），高管{inc_staff_exec}人（占比{inc_staff_exec_pct}%）

9. 近期调研访谈摘要  
机构关注点：  
① {ir_focus1}  
② {ir_focus2}  
管理层指引：  
① 未来3年重点投入{ir_guidance1}领域  
② 海外营收目标：2025年占比提升至{ir_guidance2}%

10. 估值分析  
相对估值：  
① PE(TTM)：{val_pe}倍（行业平均{val_pe_avg}倍，溢价/折价主因：{val_pe_cause}）  
② PEG：{val_peg}倍（行业平均{val_peg_avg}倍）  
绝对估值（DCF）：  
① 假设：WACC={val_wacc}%、永续增长={val_terminal_growth}% → 合理股价区间{val_price_low}-{val_price_high}元  
安全边际：现价较DCF下限折让{val_margin}%、处于近3年PE分位数{val_pe_pct}%

11. 近期相关产业积极因素：{cata_industry_factors}

12. 最近一个交易日股价变动及归因：{last_trading_move}

"""

deep_template = PromptTemplate(
    input_variables=[
        "company_name","symbol","date","core_business","product_1","prod1_share","prod1_barrier",
        "product_2","prod2_share","prod2_advantage","bus2b2c","dist_prop","top5_client_share",
        "rev_2022","rev_2023","rev_2024","rev_cause1","rev_cause2",
        "profit_2022","profit_2023","profit_2024","profit_cause1","profit_cause2",
        "gm_2022","gm_2023","gm_2024","gm_cause","rd_2022","rd_2023","rd_2024","rd_proj_progress",
        "recent_prod","recent_prod_adv","dom_hosp","oversea_markets","mna_partners","mna_synergy",
        "comp_cagr","comp_gm","comp_rd","comp_price_return","comp_diff",
        "peer1","peer1_cagr","peer1_gm","peer1_rd","peer1_price_return","peer1_diff",
        "peer2","peer2_cagr","peer2_gm","peer2_rd","peer2_price_return","peer2_diff",
        "peer3","peer3_cagr","peer3_gm","peer3_rd","peer3_price_return","peer3_diff",
        "industry_policy","industry_tech","industry_demand","company_moat","mgmt_capacity","mgmt_oversea",
        "exp_rev_cagr","exp_profit_cagr","risk_ar_days","risk_ar_avg","risk_inv_share","risk_inv_delta",
        "risk_competitor","risk_price_drop","risk_mkt_share_gain","risk_channel_drop",
        "risk_policy1","risk_policy2","cata_tech","cata_model","cata_short_term","inc_rev_target",
        "inc_rev_yoy","inc_profit_target","inc_grant_price","inc_discount","inc_reserved_share",
        "inc_staff_core","inc_staff_core_pct","inc_staff_exec","inc_staff_exec_pct",
        "ir_focus1","ir_focus2","ir_guidance1","ir_guidance2","val_pe","val_pe_avg","val_pe_cause",
        "val_peg","val_peg_avg","val_wacc","val_terminal_growth","val_price_low","val_price_high",
        "val_margin","val_pe_pct","cata_industry_factors","last_trading_move"
    ],
    template=DEEP_ANALYSIS_TEMPLATE
)
