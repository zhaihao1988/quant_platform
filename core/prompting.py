# core/prompting.py
import logging
import os

import pandas as pd
from sqlalchemy.orm import Session
import ollama
from typing import Optional, Dict, Any, List
import json
import time

# Import settings and utility functions/classes
from config.settings import settings
from db.models import StockDisclosure
from db.crud import get_stock_list_info, retrieve_relevant_disclosures
from data_processing.loader import load_price_data, load_financial_data, load_multiple_financial_reports
from integrations.web_search import get_web_search_results

logger = logging.getLogger(__name__)


# --- Ollama LLM Interaction ---
def call_local_llm(prompt: str) -> str:
    """
    Calls the configured local Ollama model to generate text based on the prompt.
    """
    model_name = settings.OLLAMA_MODEL
    logger.info(f"Calling local Ollama model: '{model_name}'. Prompt length: {len(prompt)} chars.")
    if not model_name:
        logger.error("OLLAMA_MODEL not configured in settings.")
        return "Error: LLM model name not configured."

    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
        )
        generated_text = response.get('response', '').strip()
        logger.info(f"Ollama generation complete. Response length: {len(generated_text)} chars.")
        if not generated_text:
            logger.warning("Ollama model returned an empty response.")
            return "Error: LLM returned an empty response."
        return generated_text

    except ollama.ResponseError as e:
        logger.error(f"Ollama API Error: Status Code {e.status_code}, Error: {e.error}", exc_info=True)
        return (f"Error: Ollama API request failed ({e.status_code} - {e.error}). "
                f"Ensure Ollama server is running and model '{model_name}' is available/pulled.")
    except Exception as e:
        logger.error(f"Error calling local Ollama model '{model_name}': {e}", exc_info=True)
        return f"Error: Failed to generate response from LLM '{model_name}'. Check logs for details."


def format_kb_results(chunk_results: List[Dict[str, Any]]) -> str:
    """Formats retrieved knowledge base disclosure chunks for the prompt."""
    if not chunk_results:
        return "本地知识库: 未找到相关历史公告内容块。\n"

    formatted = "--- 本地知识库 (相关历史公告内容片段) ---\n"
    # 可以按原公告ID分组显示，或者直接列表
    for i, chunk_info in enumerate(chunk_results):
        content_snippet = chunk_info.get('chunk_text', '')[:800] + ("..." if len(chunk_info.get('chunk_text', '')) > 800 else "") # 显示更长的片段
        publish_date_str = chunk_info.get('ann_date').strftime('%Y-%m-%d') if chunk_info.get('ann_date') else "N/A"
        title = chunk_info.get('title', 'N/A')
        disclosure_id = chunk_info.get('disclosure_id', 'N/A')
        chunk_order = chunk_info.get('chunk_order', 'N/A')

        formatted += f"{i + 1}. 来源公告: 《{title}》 ({publish_date_str}) (ID: {disclosure_id}, 块: {chunk_order})\n"
        formatted += f"   内容片段: {content_snippet}\n\n"

    formatted += "--- 知识库信息结束 ---\n"
    return formatted


def format_web_results(results: List[dict]) -> str:
    """Formats web search results (snippets) for the prompt."""
    if not results:
        return "网络搜索: 未找到近期相关信息。\n"

    formatted = "--- 近期网络信息 (搜索结果摘要) ---\n"
    for i, res in enumerate(results):
        formatted += f"{i + 1}. {res.get('title', 'N/A')}\n"
        formatted += f"   摘要: {res.get('snippet', 'N/A')}\n"
        formatted += "\n"
    formatted += "--- 网络信息结束 ---\n"
    return formatted


def format_price_data(df_price: Optional[pd.DataFrame]) -> str:
    """Formats recent price data for the prompt."""
    if df_price is None or df_price.empty:
        return "近期股价: [数据库中无可用数据]\n"

    formatted = "--- 近期股价表现 (最近 {} 个交易日) ---\n".format(len(df_price))
    try:
        df_price['Date'] = pd.to_datetime(df_price['date']).dt.strftime('%Y-%m-%d')
        first_day = df_price.iloc[0]
        last_day = df_price.iloc[-1]
        overall_change = ((last_day['close'] - first_day['close']) / first_day['close']) * 100 if first_day[
            'close'] else 0

        formatted += f"时间范围: {first_day['Date']} 至 {last_day['Date']}\n"
        formatted += f"期间收盘价变动: 从 {first_day['close']:.2f} 到 {last_day['close']:.2f} (涨跌幅: {overall_change:.2f}%)\n"
        formatted += "每日概览:\n"
        indices_to_show = [0, len(df_price) // 2, len(df_price) - 1] if len(df_price) > 3 else range(len(df_price))
        for i in sorted(list(set(indices_to_show))):
            row = df_price.iloc[i]
            change_str = f"{row.get('pct_change', float('nan')):.2f}%"
            volume_str = f"{int(row.get('volume', 0))}手"
            turnover_str = f"{row.get('turnover', float('nan')):.2f}%"
            formatted += f"  - {row['Date']}: 收盘价 {row['close']:.2f} (涨跌幅 {change_str}), 成交量 {volume_str}, 换手率 {turnover_str}\n"
        formatted += "---\n"
    except Exception as e:
        logger.error(f"Error formatting price data: {e}", exc_info=True)
        formatted += "[格式化股价数据时出错]\n"
    return formatted


# --- 新增：格式化多期财务数据 ---
def format_multiple_financial_reports(reports_list: List[Dict[str, Any]], report_type_key: str) -> str:
    """
    Formats multiple financial reports (e.g., latest + last 3 annual) for the prompt.

    Args:
        reports_list: List of dictionaries, each containing 'report_date' and 'data'.
                      Expected to be sorted by date descending.
        report_type_key: The key for the report type (e.g., 'benefit', 'debt', 'cash').

    Returns:
        A formatted string ready for the LLM prompt.
    """
    if not reports_list:
        return f"{report_type_key.capitalize()} 表数据: [数据库中无可用数据]\n"

    report_type_names = {'benefit': '利润表', 'debt': '资产负债表', 'cash': '现金流量表'}
    display_name = report_type_names.get(report_type_key, report_type_key.capitalize())

    full_formatted_string = ""

    # Helper function (与旧函数类似)
    def convert_and_format_value(value):
        if isinstance(value, str):
            # 使用 try...except 处理可能的转换错误
            if '亿' in value:
                try:
                    num = float(value.replace('亿', '').strip())
                    return f"{(num * 1e8):,.2f}"  # 添加逗号和两位小数格式化
                except ValueError:
                    pass  # 转换失败则忽略，继续到函数末尾返回原始字符串
            elif '万' in value:
                try:
                    num = float(value.replace('万', '').strip())
                    return f"{(num * 1e4):,.2f}"  # 添加逗号和两位小数格式化
                except ValueError:
                    pass  # 转换失败则忽略
        elif isinstance(value, float):
            return f"{value:,.2f}"  # 对浮点数格式化
        elif isinstance(value, int):
            return f"{value:,}"  # 对整数格式化
        # 对于其他类型或转换失败的情况，返回原始值的字符串形式
        return str(value)

    for report in reports_list:
        report_date = report.get('report_date')
        financial_data = report.get('data')

        if not report_date or not financial_data: continue

        date_str = report_date.strftime('%Y-%m-%d')
        formatted_section = f"--- {display_name} ({date_str}) ---\n"

        if not isinstance(financial_data, dict):
            try: financial_data = json.loads(financial_data)
            except (TypeError, json.JSONDecodeError):
                formatted_section += "[数据格式错误]\n"
                full_formatted_string += formatted_section + "\n"
                continue

        if not financial_data:
            formatted_section += "[数据为空]\n"
        else:
            # 提取关键字段并格式化 (可以根据需要选择性显示，或全部显示)
            # 这里示例显示所有字段
            for key, value in financial_data.items():
                 # 跳过空值或非数值/字符串（如果需要）
                 # if value is None or value == '': continue
                 formatted_value = convert_and_format_value(value)
                 formatted_section += f"  {key}: {formatted_value}\n"

        formatted_section += "---\n"
        full_formatted_string += formatted_section + "\n" # 在不同报告期之间加空行

    return full_formatted_string if full_formatted_string else f"{display_name}: [无有效数据格式化]\n"


# --- Prompt Generation ---
def generate_stock_report_prompt(
        symbol: str,
        company_name: str,
        kb_context: str,
        web_context: str,
        price_context: str,
        financial_context: str, # <--- 这个现在是包含多期数据的字符串
        industry: str
) -> str:
    """
      构建用于生成股票深度研究报告初稿的完整 Prompt。
      """
    prompt = f"""
    角色：你是一位拥有多年经验的资深股票分析师，专长于中国A股市场。你的分析风格审慎、客观、注重数据和逻辑，并具备一定前瞻性。请严格根据下方提供的结构化信息，撰写一份专业的初步研究报告。

    目标：为股票代码 {symbol} ({company_name})，所属行业：{industry}，生成一份深度研究报告初稿。

    报告撰写要求：
    * **严格基于信息:** 所有分析和结论必须直接基于以下提供的【可用信息】，不得臆测或引入外部知识。
    * **说明信息来源:** 在引用关键数据或观点时，简要标注来源（如“据年报摘要”、“网络研报预览显示”、“最新财报数据表明”）。
    * **量化分析:** 尽可能使用数据支撑观点（如财务指标、市场数据）。
    * **逻辑清晰:** 确保各部分分析逻辑连贯。
    * **风险提示:** 充分揭示潜在风险。
    * **客观中立:** 避免使用强烈主观或情感化词语，保持专业判断。
    * **信息不足处理:** 如果某方面信息缺失或不足以形成判断，请明确指出“信息不足，无法评估”。

    --- 可用信息 ---


{kb_context}
{web_context}
{price_context}
{financial_context} # <--- 包含多期数据

--- 可用信息结束 ---


--- 请按以下结构和要求生成报告 ---

**========================= 股票深度研究报告（初稿）：{symbol} {company_name} =========================**

**报告日期:** {time.strftime('%Y-%m-%d')}

**1. 公司基本情况与发展历程**
   * 主营业务与核心产品：(结合知识库、网络信息描述，说明收入占比、技术壁垒、差异化优势，如信息不足请指明)
   * 商业模式：(简述 2B/2C 占比、经销/直销比例、主要客户集中度，如信息不足请指明)
   * 公司发展简史：(基于知识库或网络信息，概述关键发展节点，如无信息可省略或简述)

**2. 行业分析与竞争格局**
   * 行业概况与增长逻辑：(结合网络信息和知识库，分析行业政策、技术趋势、市场需求驱动因素，如老龄化、AI+、国产替代等)
   * 竞争格局与同业对比：(基于网络信息或知识库，列出主要竞争对手，对比其营收增速、毛利率、研发投入等关键指标。分析本公司与同业的核心差异点)

**3. 财务表现分析 (基于多期数据)**
   * 关键财务指标展示与趋势分析：(基于【财务数据】部分提供的**不同报告期**数据，展示关键指标如营收、净利、毛利率、费用率等的变化趋势，并进行简要描述)。
   * 财务变动归因：(结合知识库中的“管理层讨论与分析”摘要或网络信息，分析**近几年**财务指标变动的主要原因，如成本、定价、费用、产品结构等)。

**4. 近期股价表现与事件驱动分析 (基于近期股价与信息)**
   * 股价趋势：(描述【近期股价表现】数据反映的趋势和波动性)
   * 事件归因：(结合【近期网络信息】和【本地知识库】中的公告、新闻、研报预览，分析可能影响近期股价的事件，如业绩发布、重大合同、行业政策、市场情绪等。说明关联性强度)

**5. 核心竞争力与护城河分析**
   * 竞争力来源：(基于知识库、网络信息，提炼公司的核心优势，如技术专利、品牌形象、客户粘性、成本控制、管理效率等。引用信息来源)
   * 护城河强度评估：(初步评估这些竞争优势的可持续性)

**6. 主要风险因素评估**
   * 财务风险：(结合【财务数据】分析应收账款、存货周转、负债率等潜在风险。如数据不全请指明)
   * 经营风险：(基于知识库和网络信息，分析市场竞争加剧、客户流失、供应链、管理层变动等风险)
   * 政策与市场风险：(分析行业监管、集采、技术迭代、宏观经济等外部风险)

**7. 业务亮点、催化剂与未来展望**
   * 近期积极变化：(总结知识库和网络信息中的新产品获批/升级、新市场拓展、战略合作/并购等积极信号)
   * 潜在催化剂：(识别短期内可能驱动股价或基本面改善的事件，如新品上市、产能释放、重要订单落地、机器人业务进展等)
   * 管理层指引与规划：(参考调研纪要或年报摘要，总结管理层对未来发展方向、重点投入领域的规划和目标)

**8. 公司治理与股东回报**
   * 股权激励：(分析知识库中股权激励计划的核心条款：考核目标、授予价格、激励对象范围等。如无信息请说明)
   * 股票回购：(分析知识库中股票回购计划的进展和影响。如无信息请说明)

**9. 估值分析讨论 (初步)**
   * 相对估值：(结合已知数据和网络信息，讨论 PE、PB、PEG 等指标相对于行业平均或历史水平的状况。分析估值差异原因)
   * 绝对估值探讨：(如信息允许，可简要提及 DCF 等方法的关键假设和可能区间，强调其不确定性。如无信息则说明无法进行绝对估值)
   * 安全边际：(结合当前股价和估值讨论，初步评估安全边际)

**10. 投资逻辑总结**
    * 核心投资看点：(提炼 2-3 个支持看好该公司的核心逻辑)
    * 关键风险监控点：(指出需要密切关注的 2-3 个核心风险)
    * 初步结论：(基于以上分析，给出一个审慎、客观的总结性观点，不包含直接买卖评级)

**免责声明:** 本报告基于公开信息和有限数据进行初步分析，不构成任何投资建议。

--- 报告结束 ---

请严格按照上述结构和要求完成报告。
"""
    return prompt

# --- Main Report Generation Function ---
def generate_stock_report(db: Session, symbol: str) -> Optional[str]:
    """
    Orchestrates the process of gathering data, building the prompt,
    calling the LLM, and returning the generated stock report.
    """
    logger.info(f"--- Starting Report Generation for: {symbol} ---")

    # 1. Get Basic Info
    stock_info = get_stock_list_info(db, symbol)
    company_name = stock_info.name if stock_info else f"股票代码 {symbol}"
    industry = stock_info.industry if stock_info else "未知行业"
    logger.info(f"Processing: {company_name} ({symbol}), Industry: {industry}")

    # 2. Load Price and Financial Data (调用新函数)
    price_df = load_price_data(symbol, window=5)
    # --- 修改：调用 load_multiple_financial_reports ---
    financial_benefit_reports = load_multiple_financial_reports(symbol, report_type='benefit', num_years=3)
    financial_debt_reports = load_multiple_financial_reports(symbol, report_type='debt', num_years=3)
    financial_cash_reports = load_multiple_financial_reports(symbol, report_type='cash', num_years=3)

    # --- 修改：调用新的格式化函数 ---
    price_context = format_price_data(price_df)
    financial_context = (
            format_multiple_financial_reports(financial_benefit_reports, 'benefit') +
            format_multiple_financial_reports(financial_debt_reports, 'debt') +
            format_multiple_financial_reports(financial_cash_reports, 'cash')
    )
    # ---------------------------------------------

    # 3. Retrieve from Knowledge Base (KB) (假设使用分块逻辑)
    kb_queries = { # 可以根据需要调整或增加查询
        "业务与产品": f"{company_name} 主营业务 核心产品 最新进展",
        "竞争力与风险": f"{company_name} 竞争优势 风险因素 护城河",
        "财务与讨论": f"{company_name} 财务分析 管理层讨论", # 这个查询现在更重要
        "近期动态(公告)": f"{company_name} 股权激励 回购 调研 纪要 最新公告",
        "机器人布局": f"{company_name} 机器人 进展 合作", # 增加特定主题查询
    }
    kb_context_parts = []
    retrieved_chunk_ids = set() # 用于对 KB 结果去重（如果需要）
    for query_desc, query_text in kb_queries.items():
        logger.debug(f"Retrieving KB context for: {query_desc}")
        # 假设 retrieve_relevant_disclosures 返回的是块信息列表
        chunk_results = retrieve_relevant_disclosures(db, symbol, query_text, top_k=3) # top_k 可调整
        # 添加去重逻辑（可选，基于 chunk_id 或 disclosure_id+chunk_order）
        # unique_chunks = []
        # for chunk in chunk_results: ... add if unique ...
        if chunk_results:
            kb_context_parts.append(format_kb_results(chunk_results)) # 使用处理块的格式化函数
        else:
            logger.info(f"No KB results found for query: '{query_text[:50]}...'")

    full_kb_context = "\n".join(kb_context_parts) if kb_context_parts else "本地知识库: 未检索到相关历史公告内容。\n"


    # 4. Search Web for Recent News/Analysis (调用最新的 web search 接口)
    logger.info("Retrieving recent web information...")
    # web_results = search_financial_news_google(symbol, company_name, ...) # 旧接口
    web_results = get_web_search_results(symbol) # <--- 调用新的 web search 接口
    # format_web_results 可能需要调整以更好地显示抓取到的 content
    web_context = format_web_results(web_results) # 假设此函数能处理新格式

    # 5. Build the Final Prompt
    logger.debug("Building final prompt for LLM.")
    final_prompt = generate_stock_report_prompt(
        symbol=symbol,
        company_name=company_name,
        industry=industry, # 传入行业
        kb_context=full_kb_context,
        web_context=web_context,
        price_context=price_context,
        financial_context=financial_context # 传入包含多期数据的 context
    )

    # --- 保存 Prompt 代码 (保持不变) ---
    try:
        debug_prompt_dir = "debug_prompts"
        if not os.path.exists(debug_prompt_dir): os.makedirs(debug_prompt_dir)
        prompt_filename = f"prompt_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        prompt_save_path = os.path.join(debug_prompt_dir, prompt_filename)
        with open(prompt_save_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Prompt for {symbol} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Prompt Length (chars): {len(final_prompt)}\n")
            f.write("="*40 + "\n\n")
            f.write(final_prompt)
        logger.info(f"Full prompt saved for debugging to: {prompt_save_path}")
        logger.info(f"Prompt character count: {len(final_prompt)}")
    except Exception as e:
        logger.error(f"Error saving debug prompt: {e}", exc_info=True)
    # --- 保存结束 ---

    # 6. Call Local LLM
    logger.info("Calling local LLM to generate the report...")
    generated_report = call_local_llm(final_prompt)


    # 7. Validate and Return Report
    if generated_report.startswith("Error:"):
        logger.error(f"Report generation failed for {symbol}. LLM Error: {generated_report}")
        return None
    elif not generated_report:
        logger.error(f"Report generation failed for {symbol}. LLM returned empty string.")
        return None
    else:
        logger.info(f"Successfully generated report for {symbol}.")
        report_header = f"**股票分析报告 ({time.strftime('%Y-%m-%d %H:%M:%S')})**\n**标的:** {company_name} ({symbol})\n\n"
        return report_header + generated_report