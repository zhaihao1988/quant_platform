# core/prompting.py
import logging
import os
import json
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

# Import settings and utility functions/classes
from config.settings import settings
from db.database import SessionLocal, get_db_session
from db.models import StockDisclosure, StockDisclosureChunk, StockList
from data_processing.loader import load_price_data, load_multiple_financial_reports
from integrations.web_search import get_web_search_results
from core.llm_provider import LLMProviderFactory
from rag.embeddings import Embedder
# --- V3: Import the new high-value disclosure getter ---
from db.crud import get_high_value_disclosures_for_summary

logger = logging.getLogger(__name__)

# --- 全局 Embedder 实例 ---
embedder_instance = None
def get_embedder():
    """单例模式获取 embedder 实例"""
    global embedder_instance
    if embedder_instance is None:
        logger.info("为核心报告生成流程初始化全局 Embedder 实例...")
        embedder_instance = Embedder()
        logger.info("全局 Embedder 实例已初始化。")
    return embedder_instance

# --- 核心数据检索函数 ---
def _retrieve_relevant_chunks(
    db: Session,
    symbol: str,
    query_text: str,
    top_k: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[Dict]:
    """
    使用向量相似度搜索，从数据库中检索最相关的公告文本块。
    可按日期范围过滤。
    """
    if not query_text:
        logger.warning("查询文本为空，无法执行语义搜索。")
        return []

    try:
        embedder = get_embedder()
        query_vector = embedder.embed([query_text])[0]
    except Exception as e:
        logger.error(f"为查询 '{query_text[:50]}...' 生成向量时失败: {e}", exc_info=True)
        return []

    # 基础查询
    from sqlalchemy import func
    query = db.query(
        StockDisclosureChunk.id,
        StockDisclosureChunk.chunk_text,
        StockDisclosure.title,
        StockDisclosure.ann_date,
        StockDisclosureChunk.disclosure_id,
        StockDisclosureChunk.chunk_order,
        StockDisclosureChunk.chunk_vector.cosine_distance(query_vector).label('distance')
    ).join(
        StockDisclosure, StockDisclosureChunk.disclosure_id == StockDisclosure.id
    ).filter(
        StockDisclosure.symbol == symbol
    )

    if start_date:
        query = query.filter(StockDisclosure.ann_date >= start_date)
    if end_date:
        query = query.filter(StockDisclosure.ann_date <= end_date)

    results = query.order_by('distance').limit(top_k).all()

    formatted_results = [dict(row._mapping) for row in results]
    logger.info(f"为查询 '{query_text[:50]}...' 检索到 {len(formatted_results)} 个文本块 (top_k={top_k}, symbol={symbol})")
    return formatted_results

# --- V3: This function is being deprecated in favor of the one from crud.py ---
# def _get_high_value_disclosures(db: Session, symbol: str, num_reports: int = 3) -> List[str]:
#     """获取近期最高价值的公告内容用于主题发现。"""
#     logger.info(f"正在为 {symbol} 提取近期高价值公告...")
#     one_year_ago = date.today() - timedelta(days=365)
#     
#     # 优先查找年报和半年报
#     keywords = ['年度报告', '半年度报告']
#     clauses = [StockDisclosure.title.ilike(f'%{kw}%') for kw in keywords]
#     
#     recent_reports = db.query(
#         StockDisclosure.raw_content
#     ).filter(
#         StockDisclosure.symbol == symbol,
#         StockDisclosure.ann_date >= one_year_ago,
#         and_(StockDisclosure.raw_content.isnot(None), StockDisclosure.raw_content != ''),
#         or_(*clauses)
#     ).order_by(
#         desc(StockDisclosure.ann_date)
#     ).limit(num_reports).all()
#     
#     contents = [report.raw_content for report in recent_reports]
#     logger.info(f"为主题发现提取了 {len(contents)} 份高价值报告。")
#     return contents

# --- LLM 交互与内容格式化 ---
def _call_llm(prompt: str, model_override: Optional[str] = None, json_mode: bool = False) -> str:
    """
    封装 LLM 调用，使用 LLMProviderFactory。
    新增 json_mode 支持，如果 provider 支持，会使用 JSON 模式。
    """
    provider = LLMProviderFactory.get_provider()
    model_to_use = model_override if model_override else None
    
    logger.info(f"正在调用 LLM Provider: {type(provider).__name__} (模型: {model_to_use or '默认'}, JSON Mode: {json_mode})")
    try:
        # 假设 provider 的 generate 方法接受一个 json_mode 参数
        response = provider.generate(prompt, model=model_to_use, json_mode=json_mode)
        if not response or not response.strip():
            logger.warning("LLM 返回了空响应。")
            return "错误: LLM 返回空响应。"
        return response
    except Exception as e:
        logger.error(f"调用 LLM 时发生错误: {e}", exc_info=True)
        return f"错误: LLM 调用失败 - {e}"

def _create_research_plan(db: Session, symbol: str, company_name: str) -> Optional[Dict[str, List[str]]]:
    """
    【V3 - 核心变革】第一阶段：使用LLM创建一个包含内部检索主题和外部搜索查询的"研究计划"。
    """
    logger.info(f"--- RAG V3 Stage 1: Creating Research Plan for {company_name} ({symbol}) ---")

    # 1. 使用新的 crud 函数获取高价值公告内容
    # 我们只关心内容，所以将它们提取出来
    disclosures = get_high_value_disclosures_for_summary(db, symbol, num_reports=3)
    if not disclosures:
        logger.warning("No high-value reports found for research plan creation. Aborting.")
        return None
    
    # 将公告内容和标题格式化后合并，为LLM提供更丰富的上下文
    report_contexts = []
    for d in disclosures:
        content_snippet = d.raw_content[:4000] # 限制每个公告的长度
        report_contexts.append(f"公告标题:《{d.title}》(发布于 {d.ann_date})\n内容摘要:\n{content_snippet}")
    combined_text = "\n\n---\n\n".join(report_contexts)

    # 2. 构建新的"研究计划" Prompt
    prompt = f"""
    你是顶尖的金融分析师，任务是为深入研究「{company_name}」制定一份周密的"研究计划"。
    请仔细阅读下方提供的该公司近期核心公告摘要。

    你的任务是输出一个JSON对象，其中包含两个关键部分：
    1.  `topics_for_internal_retrieval`: 一个包含3-5个核心【主题】的列表。这些主题将用于从我们【内部的历史公告数据库】中进行深入的语义搜索。主题应具有代表性，能覆盖公司的核心业务、战略、风险和机遇。
    2.  `queries_for_web_search`: 一个包含3-5个精准【搜索查询】的列表。这些查询将用于在【外部互联网】上获取最新的新闻、行业动态或竞争信息，以补充内部数据的不足。查询应具体、可搜索，旨在发现最新的、可能未在历史公告中体现的信息。

    **输出格式要求:**
    - 必须严格返回一个有效的JSON对象。
    - JSON对象必须包含 `topics_for_internal_retrieval` 和 `queries_for_web_search` 两个键，其值均为字符串列表。
    - 不要添加任何解释或无关文字，直接输出JSON对象。

    **示例输出:**
    {{
        "topics_for_internal_retrieval": [
            "空气悬挂系统的技术研发与产能布局",
            "海外市场特别是北美地区的业务拓展情况",
            "机器人核心部件（如谐波减速器）的业务进展与挑战",
            "原材料价格波动对毛利率的影响及应对措施"
        ],
        "queries_for_web_search": [
            "中鼎股份 空气悬挂系统 新能源车企客户 2024",
            "AMK德国公司整合最新进展及协同效应",
            "汽车零部件行业海外并购风险与机遇",
            "人形机器人行业对谐波减速器的最新需求"
        ]
    }}

    --- 需要分析的公告摘要 ---
    {combined_text}
    --- 公告摘要结束 ---

    请严格按要求输出JSON对象。
    """

    # 3. 调用 LLM (使用支持JSON模式的调用)
    logger.info("Calling LLM to create the research plan...")
    response_str = _call_llm(prompt, model_override="Qwen/Qwen3-8B", json_mode=True)

    if response_str.startswith("错误:"):
        logger.error(f"Research plan creation failed: {response_str}.")
        return None

    # 4. 解析 LLM 返回的 JSON
    try:
        # 尝试从返回的文本中找到json对象
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in the response.", response_str, 0)
        
        plan = json.loads(json_match.group(0))

        if isinstance(plan, dict) and \
           'topics_for_internal_retrieval' in plan and \
           'queries_for_web_search' in plan and \
           isinstance(plan['topics_for_internal_retrieval'], list) and \
           isinstance(plan['queries_for_web_search'], list):
            logger.info(f"Successfully created research plan: {plan}")
            return plan
        else:
            logger.error(f"Parsed JSON does not match the required structure. Parsed data: {plan}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse research plan from LLM response: {e}. Raw response: '{response_str}'")
        return None


def format_kb_results(chunk_results: List[Dict[str, Any]]) -> str:
    """格式化检索到的知识库文本块用于 Prompt。"""
    if not chunk_results:
        return "" # 如果没有，返回空字符串

    # 按公告 ID 分组
    grouped_results = {}
    for chunk in chunk_results:
        disclosure_id = chunk.get('disclosure_id')
        if disclosure_id not in grouped_results:
            grouped_results[disclosure_id] = {
                'title': chunk.get('title', 'N/A'),
                'ann_date': chunk.get('ann_date').strftime('%Y-%m-%d') if chunk.get('ann_date') else "N/A",
                'chunks': []
            }
        grouped_results[disclosure_id]['chunks'].append(chunk)

    # 格式化输出
    formatted_str = ""
    for disc_id, data in grouped_results.items():
        formatted_str += f"来源公告: 《{data['title']}》 (日期: {data['ann_date']})\n"
        # 对文本块按 chunk_order 排序
        sorted_chunks = sorted(data['chunks'], key=lambda x: x.get('chunk_order', 0))
        for chunk_data in sorted_chunks:
            content_snippet = chunk_data.get('chunk_text', '').strip()
            formatted_str += f"  - 片段内容: {content_snippet}\n"
        formatted_str += "\n"
        
    return formatted_str

# format_price_data 和 format_multiple_financial_reports 函数基本不变，但需要确保它们在这里定义或被正确导入
# 为了自包含，这里直接定义它们。

def format_price_data(df_price: Optional[pd.DataFrame]) -> str:
    """格式化近期股价数据用于 Prompt。"""
    if df_price is None or df_price.empty:
        return "近期股价: [数据库中无可用数据]\n"

    formatted = "--- 近期股价表现 (最近 {} 个交易日) ---\n".format(len(df_price))
    try:
        # 确保 'date' 列是 datetime 对象
        if not pd.api.types.is_datetime64_any_dtype(df_price['date']):
             df_price['date'] = pd.to_datetime(df_price['date'])

        df_price['DateStr'] = df_price['date'].dt.strftime('%Y-%m-%d')
        first_day = df_price.iloc[0]
        last_day = df_price.iloc[-1]
        
        if first_day['close'] and last_day['close']:
            overall_change = ((last_day['close'] - first_day['close']) / first_day['close']) * 100
            formatted += f"时间范围: {first_day['DateStr']} 至 {last_day['DateStr']}\n"
            formatted += f"期间收盘价变动: 从 {first_day['close']:.2f} 到 {last_day['close']:.2f} (涨跌幅: {overall_change:.2f}%)\n"
        
        highest = df_price['high'].max()
        lowest = df_price['low'].min()
        avg_vol = df_price['volume'].mean()
        formatted += f"期间最高价: {highest:.2f}, 最低价: {lowest:.2f}, 平均成交量: {avg_vol:,.0f} 手\n"
        
        formatted += "---\n"
    except Exception as e:
        logger.error(f"格式化股价数据时出错: {e}", exc_info=True)
        formatted = "近期股价: [格式化数据时出错]\n"
    return formatted


def format_multiple_financial_reports(reports_list: List[Dict[str, Any]], report_type_key: str) -> str:
    """格式化多期财务报告用于 Prompt。"""
    if not reports_list:
        return "" # 返回空字符串而不是提示

    report_type_names = {'benefit': '利润表', 'debt': '资产负债表', 'cash': '现金流量表'}
    display_name = report_type_names.get(report_type_key, report_type_key.capitalize())
    full_formatted_string = f"--- {display_name} (多期) ---\n"
    
    # 定义关键指标，可以按报表类型区分
    key_metrics = {
        'benefit': ['营业总收入', '营业总成本', '研发费用', '销售费用', '管理费用', '财务费用', '营业利润', '净利润', '归属于母公司所有者的净利润', '基本每股收益'],
        'debt': ['资产总计', '负债合计', '所有者权益合计', '归属于母公司所有者权益合计', '流动资产合计', '流动负债合计', '非流动资产合计', '非流动负债合计'],
        'cash': ['经营活动产生的现金流量净额', '投资活动产生的现金流量净额', '筹资活动产生的现金流量净额', '现金及现金等价物净增加额']
    }
    
    # Helper function
    def _safe_format(value):
        if value is None: return "N/A"
        try:
            num = float(value)
            if abs(num) > 1e8: return f"{num/1e8:.2f} 亿"
            if abs(num) > 1e4: return f"{num/1e4:.2f} 万"
            return f"{num:,.2f}"
        except (ValueError, TypeError):
            return str(value).strip()

    # Transpose data: rows as metrics, columns as report dates
    data_table = {}
    dates = []
    for report in reports_list:
        report_date = report.get('report_date')
        if not report_date: continue
        dates.append(report_date.strftime('%Y-%m-%d'))
        
        financial_data = report.get('data', {})
        if not isinstance(financial_data, dict):
            try: financial_data = json.loads(financial_data)
            except (TypeError, json.JSONDecodeError): financial_data = {}

        for key, value in financial_data.items():
            if key not in data_table:
                data_table[key] = {}
            data_table[key][dates[-1]] = value

    # Format the transposed table
    for metric in key_metrics.get(report_type_key, sorted(data_table.keys())):
        if metric in data_table:
            full_formatted_string += f"{metric:<20}" # Left-align metric name
            for report_date_str in dates:
                value = data_table[metric].get(report_date_str)
                full_formatted_string += f" | {report_date_str}: {_safe_format(value)}"
            full_formatted_string += "\n"
            
    return full_formatted_string + "---\n"


def format_web_results(results: List[dict]) -> str:
    """格式化网络搜索结果用于 Prompt。"""
    if not results:
        return "网络搜索: 未找到近期相关信息。\n"
    
    formatted = "--- 近期网络信息 (搜索结果摘要) ---\n"
    for i, res in enumerate(results):
        formatted += f"{i + 1}. {res.get('title', 'N/A')} (来源: {res.get('displayLink', 'N/A')})\n"
        formatted += f"   摘要: {res.get('snippet', 'N/A')}\n\n"
    formatted += "--- 网络信息结束 ---\n"
    return formatted


def generate_stock_report_prompt(
        symbol: str,
        company_name: str,
        kb_context: str,
        web_context: str,
        price_context: str,
        financial_context: str,
        industry: str
) -> str:
    """
      构建用于生成股票深度研究报告初稿的完整 Prompt。
    """
    prompt = f"""
    角色：你是一位拥有超过15年经验的顶尖A股策略分析师，以其深度、审慎、数据驱动的分析风格而闻名。你的任务是基于下方提供的、严格限定的信息，为机构客户撰写一份专业的、结构化的股票初步研究报告。

    目标：为股票代码 {symbol} ({company_name})，所属行业：{industry}，生成一份深度研究报告初稿。

    **核心指令与要求:**
    1.  **绝对信息依赖**: 你的所有分析、判断和结论【必须】严格、直接地源自下方【可用信息】部分提供的数据和文本片段。严禁使用任何外部知识或个人数据库。
    2.  **明确标注来源**: 在引用关键数据或观点时，必须清晰标注其来源。例如："根据[公告: 2023-04-25 年度报告]中的片段..."、"[网络信息]显示..."、"最新的[财务数据]表明..."。
    3.  **量化与归因**: 尽可能使用数据进行量化分析。在分析财务和业务变化时，要结合[知识库]中的管理层讨论等内容进行归因。
    4.  **识别信息缺失**: 如果某个分析要点（如下方报告结构中要求的）在【可用信息】中找不到对应内容，你【必须】明确指出"相关信息缺失，无法评估"。
    5.  **专业与中立**: 保持客观、中立的分析师立场，避免使用"看好"、"强烈推荐"等情感化或诱导性词语。
    6.  **结构化输出**: 严格按照下面定义的报告结构进行撰写，确保每个章节都得到回应（即使是指出信息缺失）。

    ---
    ### **可用信息**
    ---

    #### **1. 本地知识库 (历史公告语义检索结果)**
    {kb_context}

    #### **2. 外部网络信息 (近期新闻与摘要)**
    {web_context}

    #### **3. 财务数据 (多期财务报表关键指标)**
    {financial_context}

    #### **4. 交易数据 (近期股价与成交量)**
    {price_context}

    ---
    ### **请严格按以下结构和要求生成报告**
    ---

    **========================= 股票深度研究报告（初稿）：{symbol} {company_name} =========================**

    **报告日期:** {time.strftime('%Y-%m-%d')}

    **1. 核心观点 (Executive Summary)**
       * (综合所有信息，提炼出2-3个最核心的投资逻辑和关键风险点作为开篇摘要。)

    **2. 公司基本面与业务分析**
       * **主营业务与商业模式**: (结合[知识库]和[网络信息]，描述公司的核心业务、主要产品及其在行业中的定位。如信息充足，请分析其商业模式，如客户类型、销售模式等。若信息不足，请明确指出。)
       * **近期经营动态**: (基于[知识库]中的最新公告（如季报、调研纪要）和[网络信息]，总结公司近期的关键经营活动、项目进展或战略调整。)

    **3. 行业分析与竞争格局**
       * **行业趋势与驱动力**: (根据[网络信息]和[知识库]内容，分析公司所处行业的主要发展趋势、政策影响和市场需求变化。)
       * **竞争定位**: (基于可用信息，分析公司的市场地位和核心竞争力，如技术壁垒、品牌优势、成本控制等。如果信息允许，与潜在竞争对手进行简要对比。)

    **4. 财务表现深入分析**
       * **盈利能力分析**: (基于[财务数据]中的多期利润表，分析公司营收和净利润的增长趋势、驱动因素及利润率（如毛利率、净利率）的变化，并结合[知识库]内容进行归因。)
       * **资产与负债结构**: (基于[财务数据]中的资产负债表，评估公司的资产结构、偿债能力和财务健康状况。关注关键比率的变化。)
       * **现金流量分析**: (基于[财务数据]中的现金流量表，评估公司经营活动、投资和融资的现金流状况，判断其造血能力和资金链健康度。)

    **5. 主要风险因素评估**
       * (综合所有信息，分类列举并详细阐述公司面临的主要风险。例如：市场竞争风险、技术迭代风险、客户集中度风险、财务风险（如应收账款、商誉）、宏观经济与政策风险等。每一个风险点都需说明判断依据。)

    **6. 未来展望与潜在催化剂**
       * **增长点与催化剂**: (基于[知识库]和[网络信息]，识别公司未来的潜在增长点，如新产品、新技术、新市场或行业拐点。列出可能驱动价值重估的短期和中期催化剂。)
       * **管理层规划**: (如果[知识库]中有相关内容，总结管理层在年报或调研中披露的未来发展战略和规划。)

    **7. 估值讨论**
       * (基于[财务数据]和[交易数据]，进行初步的估值水平讨论。例如，可以提及历史估值区间或与行业对比，但强调这仅为初步观察。明确指出，由于信息有限，无法进行精确估值。)

    **免责声明:** 本报告完全基于提供的有限信息生成，不构成任何投资建议。所有信息未经独立核实，投资者应自行承担风险。
    """
    return prompt

# --- 主报告生成函数 ---
def generate_stock_report(db: Session, symbol: str) -> Optional[str]:
    """
    【V3 - 研究计划驱动版】
    编排整个 RAG 流程，包括：
    1.  LLM生成研究计划（内部主题 + 外部查询）。
    2.  根据计划并行执行信息检索。
    3.  聚合所有信息，生成最终报告。
    """
    logger.info(f"--- 开始为股票代码 {symbol} 生成【V3 研究计划驱动版】分析报告 ---")

    # 1. 获取基本信息
    stock_info = db.query(StockList).filter(StockList.code == symbol).first()
    company_name = stock_info.name if stock_info else f"股票代码 {symbol}"
    industry = stock_info.industry if stock_info else "未知行业"
    logger.info(f"标的: {company_name} ({symbol}), 行业: {industry}")

    # 2. RAG V3 第一阶段：创建研究计划
    research_plan = _create_research_plan(db, symbol, company_name)
    if not research_plan:
        logger.error("无法创建研究计划，报告生成中止。")
        return "错误：未能创建研究计划，无法生成报告。"

    internal_topics = research_plan.get("topics_for_internal_retrieval", [])
    web_queries = research_plan.get("queries_for_web_search", [])

    # 3. RAG V3 第二阶段：根据计划执行信息挖掘
    logger.info(f"--- RAG V3 Stage 2: Executing Research Plan ---")

    # 3a. 内部知识库检索
    full_kb_context = ""
    if internal_topics:
        today = date.today()
        time_buckets = {
            "近期 (1年内)": {'start': today - timedelta(days=365), 'end': today, 'k': 3},
            "中期 (1-3年前)": {'start': today - timedelta(days=3*365), 'end': today - timedelta(days=365), 'k': 2},
            "远期 (3年前)": {'start': None, 'end': today - timedelta(days=3*365), 'k': 1},
        }
        kb_context_parts = {}
        retrieved_chunk_ids = set()

        for topic in internal_topics:
            logger.info(f"正在为内部主题 '{topic}' 检索相关信息...")
            topic_context = ""
            for bucket_name, params in time_buckets.items():
                chunks = _retrieve_relevant_chunks(db, symbol, topic, params['k'], params['start'], params['end'])
                unique_chunks = [c for c in chunks if c['id'] not in retrieved_chunk_ids]
                for c in unique_chunks:
                    retrieved_chunk_ids.add(c['id'])
                
                if unique_chunks:
                    topic_context += f"--- {bucket_name} ---\n"
                    topic_context += format_kb_results(unique_chunks)
            
            if topic_context:
                kb_context_parts[topic] = topic_context

        for topic, context in kb_context_parts.items():
            full_kb_context += f"\n**分析主题: {topic}**\n{context}"
    
    if not full_kb_context:
        full_kb_context = "本地知识库: 未检索到相关历史公告内容。\n"

    # 3b. 外部网络搜索
    all_web_results = []
    if web_queries:
        for query in web_queries:
            logger.info(f"正在为外部查询 '{query}' 执行网络搜索...")
            # 限制每个查询返回的结果数量，例如 top_n=3
            web_results = get_web_search_results(query, top_n=3)
            if web_results:
                all_web_results.extend(web_results)
    web_context = format_web_results(all_web_results)


    # 4. 加载其他固定上下文数据 (股价和财务)
    price_df = load_price_data(symbol, window=30)
    price_context = format_price_data(price_df)
    
    financial_benefit_reports = load_multiple_financial_reports(symbol, report_type='benefit', num_years=3)
    financial_context = format_multiple_financial_reports(financial_benefit_reports, 'benefit')
    
    # 5. 构建并保存最终的 Prompt
    final_prompt = generate_stock_report_prompt(
        symbol=symbol, company_name=company_name, industry=industry,
        kb_context=full_kb_context, web_context=web_context,
        price_context=price_context, financial_context=financial_context
    )
    
    # 保存 Prompt 到文件
    try:
        debug_dir = "debug_prompts"
        os.makedirs(debug_dir, exist_ok=True)
        filename = f"prompt_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(debug_dir, filename), 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        logger.info(f"最终 Prompt 已保存至: {os.path.join(debug_dir, filename)}")
    except Exception as e:
        logger.error(f"保存调试 Prompt 时出错: {e}", exc_info=True)

    # 6. 调用 LLM 生成最终报告
    logger.info("调用 LLM 生成最终分析报告...")
    generated_report = _call_llm(final_prompt)
    
    if generated_report.startswith("错误:"):
        return f"报告生成失败: {generated_report}"
    
    return generated_report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="生成个股的智能研究报告")
    parser.add_argument("symbol", type=str, help="股票代码，例如 '000887'")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    db_session: Session = SessionLocal()
    try:
        final_report = generate_stock_report(db_session, args.symbol)
        if final_report:
            report_dir = "generated_reports"
            os.makedirs(report_dir, exist_ok=True)
            report_filename = f"report_{args.symbol}_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(os.path.join(report_dir, report_filename), 'w', encoding='utf-8') as f:
                f.write(final_report)
            logger.info(f"报告已成功生成并保存到: {os.path.join(report_dir, report_filename)}")
        else:
            logger.error(f"未能为股票 {args.symbol} 生成报告。")
    except Exception as e:
        logger.error(f"在主流程中发生未处理的错误: {e}", exc_info=True)
    finally:
        db_session.close()
        logger.info("数据库会话已关闭。")