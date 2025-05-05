# core/prompting.py
import logging
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
from data_processing.loader import load_price_data, load_financial_data
from integrations.web_search import search_financial_news_google

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


# --- Context Formatting Functions ---
def format_kb_results(disclosures: List[StockDisclosure]) -> str:
    """Formats retrieved knowledge base disclosures for the prompt."""
    if not disclosures:
        return "本地知识库: 未找到相关历史公告。\n"

    formatted = "--- 本地知识库 (相关历史公告摘要) ---\n"
    for i, disc in enumerate(disclosures):
        content_snippet = disc.raw_content[:600] + "..." if disc.raw_content and len(
            disc.raw_content) > 600 else disc.raw_content
        publish_date_str = disc.ann_date.strftime('%Y-%m-%d') if disc.ann_date else "N/A"
        formatted += f"{i + 1}. 《{disc.title}》 ({publish_date_str})\n"
        formatted += f"   摘要: {content_snippet}\n\n"
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


def format_financial_data(financials: Optional[Dict[str, Any]], report_type: str) -> str:
    """Formats latest financial data (JSONB) for the prompt, dynamically extracting all fields."""
    if not financials:
        return f"最新{report_type}财务数据: [数据库中无可用数据]\n"

    # Validate report type
    valid_report_types = ['benefit', 'debt', 'cash']
    if report_type not in valid_report_types:
        logger.warning(f"Invalid report type: {report_type}. Using raw data display.")
        report_type = "财务"

    # Map report types to Chinese names for display
    report_type_names = {
        'benefit': '利润表',
        'debt': '资产负债表',
        'cash': '现金流量表'
    }
    display_name = report_type_names.get(report_type, report_type)

    formatted = f"--- 最新{display_name}财务数据 (完整字段) ---\n"

    def convert_chinese_number(value):
        """Convert numbers with Chinese units (亿/万) to standard numeric format"""
        if isinstance(value, str):
            if '亿' in value:
                try:
                    num = float(value.replace('亿', '').strip())
                    return num * 100000000
                except ValueError:
                    return value
            elif '万' in value:
                try:
                    num = float(value.replace('万', '').strip())
                    return num * 10000
                except ValueError:
                    return value
        return value

    try:
        if not isinstance(financials, dict):
            # If data is not already a dict, try to parse it from JSON string
            try:
                financials = json.loads(financials)
            except (TypeError, json.JSONDecodeError):
                logger.error(f"Financial data is not in expected format (dict or JSON string): {type(financials)}")
                return f"最新{display_name}财务数据: [数据格式错误]\n"

        if not financials:
            return f"最新{display_name}财务数据: [数据为空]\n"

        # Dynamically extract all fields from the JSON data
        for key, value in financials.items():
            # First convert any Chinese number units
            converted_value = convert_chinese_number(value)

            # Format the value based on its type
            if isinstance(converted_value, (int, float)):
                # For numbers, format with commas for thousands and 2 decimal places
                formatted_value = f"{converted_value:,.2f}" if isinstance(converted_value,
                                                                          float) else f"{converted_value:,}"
            elif isinstance(converted_value, str):
                # For strings, just use as-is
                formatted_value = converted_value
            else:
                # For other types (lists, dicts), convert to string
                formatted_value = str(converted_value)

            formatted += f"  {key}: {formatted_value}\n"

        formatted += "---\n"

    except Exception as e:
        logger.error(f"Error formatting financial data ({report_type}): {e}", exc_info=True)
        formatted += f"[格式化{display_name}财务数据时出错]\n"

    return formatted


# --- Prompt Generation ---
def generate_stock_report_prompt(
        symbol: str,
        company_name: str,
        kb_context: str,
        web_context: str,
        price_context: str,
        financial_context: str
) -> str:
    """
    Builds the final structured prompt string for the LLM.
    """
    prompt = f"""
角色：你是一位经验丰富、严谨客观的初级股票分析师。你的任务是基于提供的结构化信息，生成一份对中国A股上市公司的初步研究报告。请严格根据下方提供的信息进行分析，避免编造数据或信息之外的内容。如果信息不足，请明确指出。

目标：为股票代码 {symbol} ({company_name}) 生成一份研究报告。

可用信息源：
1.  本地知识库：包含公司历史公告（如年报、半年报、调研、回购、股权激励）的原文摘要。
2.  网络搜索：近期（约6个月内）来自主流财经网站的新闻摘要和分析片段。
3.  近期股价：最近几个交易日的收盘价、涨跌幅、成交量等数据。
4.  最新财务数据：最新一期财务报表（如利润表、资产负债表）的关键指标摘要。

报告结构要求：请严格按照以下序号和标题组织报告内容。

--- 可用信息 ---

{kb_context}
{web_context}
{price_context}
{financial_context}

--- 可用信息结束 ---


--- 请生成以下结构的报告 ---

**========================= 股票研究报告：{symbol} {company_name} =========================**

**1. 公司概览与主营业务**
   * 基于已知信息（如来自StockList的行业、地域信息，或知识库/网络信息中提及的），简述公司主营业务和主要产品/服务。指出信息来源。

**2. 近期市场表现分析**
   * 结合【近期股价表现】数据，描述近期的股价趋势（涨跌幅度、成交量变化）。
   * 尝试结合【近期网络信息】或【本地知识库】中的事件（如公告发布、行业新闻），对股价波动进行初步归因分析。如果无法找到明确原因，请说明。

**3. 近期关键基本面信息**
   * 结合【最新财务数据】摘要，展示关键财务指标。
   * 结合【本地知识库】和【近期网络信息】，总结近期（6个月内）是否有重要的公告（如重大合同、业绩预告、回购、股权激励、高管变动、重要调研纪要）。列出关键信息点。
   * 结合【近期网络信息】，总结是否有行业层面的积极或消极动态与公司相关。

**4. 核心竞争力与风险点（基于现有信息总结）**
   * 根据【本地知识库】（年报讨论部分摘要）或【近期网络信息】（分析片段），尝试总结公司可能的核心竞争力（如品牌、技术、市场地位等）。
   * 根据【本地知识库】（年报风险章节摘要）或【近期网络信息】，尝试总结公司面临的主要风险点。
   * **重要：** 如果信息不足以判断，请明确指出信息缺乏。

**5. 初步总结**
   * 基于以上信息的汇总，给出一个简短、客观的总结陈述，说明当前观察到的公司状况、近期动态和潜在关注点。避免给出明确的买卖建议。

--- 报告结束 ---

请确保语言专业、客观、简洁，严格依据提供的信息。
"""
    return prompt


# --- Main Report Generation Function ---
def generate_stock_report(db: Session, symbol: str) -> Optional[str]:
    """
    Orchestrates the process of gathering data, building the prompt,
    calling the LLM, and returning the generated stock report.
    """
    logger.info(f"--- Starting Report Generation for: {symbol} ---")

    # 1. Get Basic Info (Name, Industry)
    stock_info = get_stock_list_info(db, symbol)
    company_name = stock_info.name if stock_info else f"股票代码 {symbol}"
    industry = stock_info.industry if stock_info else "未知行业"
    logger.info(f"Processing: {company_name} ({symbol}), Industry: {industry}")

    # 2. Load Price and Financial Data
    price_df = load_price_data(symbol, window=5)
    # Load all three types of financial reports
    financial_benefit = load_financial_data(symbol, report_type='benefit')
    financial_debt = load_financial_data(symbol, report_type='debt')
    financial_cash = load_financial_data(symbol, report_type='cash')

    # Format the loaded data for the prompt
    price_context = format_price_data(price_df)
    financial_context = (
            format_financial_data(financial_benefit, 'benefit') +
            format_financial_data(financial_debt, 'debt') +
            format_financial_data(financial_cash, 'cash')
    )

    # 3. Retrieve from Knowledge Base (KB)
    kb_queries = {
        "业务与产品": f"{company_name} 主营业务 核心产品 最新进展",
        "竞争力与风险": f"{company_name} 竞争优势 风险因素 护城河",
        "财务与讨论": f"{company_name} 财务分析 管理层讨论",
        "近期动态(公告)": f"{company_name} 股权激励 回购 调研 纪要 最新公告",
    }
    kb_context_parts = []
    for query_desc, query_text in kb_queries.items():
        logger.debug(f"Retrieving KB context for: {query_desc}")
        disclosures = retrieve_relevant_disclosures(db, symbol, query_text, top_k=3)
        if disclosures:
            kb_context_parts.append(format_kb_results(disclosures))
        else:
            logger.info(f"No KB results found for query: '{query_text[:50]}...'")

    full_kb_context = "\n".join(kb_context_parts) if kb_context_parts else "本地知识库: 未检索到相关历史公告。\n"

    # 4. Search Web for Recent News/Analysis
    logger.info("Retrieving recent web information...")
    web_results = search_financial_news_google(symbol, company_name, num_results_per_site=2, total_general_results=3)
    web_context = format_web_results(web_results)

    # 5. Build the Final Prompt
    logger.debug("Building final prompt for LLM.")
    final_prompt = generate_stock_report_prompt(
        symbol=symbol,
        company_name=company_name,
        kb_context=full_kb_context,
        web_context=web_context,
        price_context=price_context,
        financial_context=financial_context
    )

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