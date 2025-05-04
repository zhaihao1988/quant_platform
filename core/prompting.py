# rag/prompting.py

# prompting.py
import logging
from sqlalchemy.orm import Session
import ollama # 导入 ollama 库
from db.models import StockDisclosure
# 导入您的配置
from config import settings
# 使用正确的 crud 函数和 web_search 函数
from db.crud import retrieve_relevant_disclosures, get_stock_list_info # 使用 get_stock_list_info
from integrations.web_search import search_financial_news_google # 使用 Google CSE 搜索函数

logger = logging.getLogger(__name__)

# --- Local LLM Interaction using Ollama ---
def call_local_llm(prompt: str, max_tokens: int = 4096) -> str:
    """
    调用本地 Ollama 模型生成文本。
    使用 config.settings 中的 OLLAMA_MODEL。
    """
    model_name = settings.OLLAMA_MODEL
    logger.info(f"Calling local Ollama model: '{model_name}'. Prompt length: {len(prompt)}")
    try:
        # 确保 Ollama 服务正在运行
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            # stream=False, # 等待完整响应
            # options={"num_predict": max_tokens} # Ollama 可能使用 num_predict 或其他参数控制长度
        )
        logger.info(f"Ollama generation complete. Response length: {len(response.get('response', ''))}")
        return response.get('response', '') # 提取生成的文本

    except ollama.ResponseError as e:
         logger.error(f"Ollama API Error: {e.status_code} - {e.error}")
         return f"Error: Ollama API request failed ({e.status_code} - {e.error}). Please ensure Ollama server is running and the model '{model_name}' is available."
    except Exception as e:
        logger.error(f"Error calling local Ollama model '{model_name}': {e}")
        # 返回更详细的错误信息
        return f"Error generating response from local LLM '{model_name}': {e}. Check connection and model availability."


# --- 格式化函数 (使用 raw_content, ann_date) ---
def format_kb_results(disclosures: list[StockDisclosure]) -> str: # Type hint 使用 StockDisclosure
    """格式化知识库检索结果"""
    if not disclosures:
        return "本地知识库未找到直接相关的历史公告。\n"

    formatted = "来自本地知识库的相关信息摘要：\n\n"
    for i, disc in enumerate(disclosures):
        # 使用 raw_content
        content_snippet = disc.raw_content[:500] + "..." if disc.raw_content and len(disc.raw_content) > 500 else disc.raw_content
        # 使用 ann_date
        publish_date_str = disc.ann_date.strftime('%Y-%m-%d') if disc.ann_date else "N/A"
        formatted += f"{i+1}. **《{disc.title}》 ({publish_date_str})**\n"
        formatted += f"   摘要: {content_snippet}\n\n" # 这里是原始内容摘要，非LLM总结
    return formatted

def format_web_results(results: list[dict]) -> str:
    """格式化网络搜索结果 (来自 Google CSE 的 snippet)"""
    if not results:
        return "网络搜索未找到近期的相关信息。\n"

    formatted = "近期网络信息摘要 (来源 Google Search / 指定财经网站):\n\n"
    for i, res in enumerate(results):
        formatted += f"{i+1}. **{res.get('title', 'N/A')}**\n"
        # 使用 snippet
        formatted += f"   摘要 (Snippet): {res.get('snippet', 'N/A')}\n"
        # formatted += f"   链接: {res.get('link', '#')}\n\n" # 可选
        formatted += "\n"
    return formatted

# --- Prompt 构建函数 (保持不变，但调用处使用 company_name) ---
def generate_stock_report_prompt(symbol: str, company_name: str, kb_context: str, web_context: str) -> str:
    # ... (之前的 generate_stock_report_prompt 函数代码不变, 确保占位符 {symbol} {company_name} 正确) ...
    prompt = f"""
请扮演一位资深的股票分析师，基于以下提供的背景信息和实时数据，为股票代码 {symbol} ({company_name}) 生成一份深度研究报告。

**报告要求：**
1.  **客观全面：** 结合本地知识库（历史公告、报告原始文本摘要）和最新的网络信息（搜索结果摘要）进行分析。
2.  **逻辑清晰：** 报告结构需包含以下部分，并确保各部分内容连贯。
3.  **重点突出：** 明确公司的核心竞争力、增长逻辑、潜在风险及投资看点。
4.  **数据支撑：** 财务数据部分需准确展示，并进行简要归因（依赖LLM基于上下文信息生成）。
5.  **价值导向：** 估值分析需结合多种方法，并与同业进行对比（依赖LLM基于上下文信息生成）。

**背景信息：**

--- 本地知识库信息 (历史公告原文摘要) ---
{kb_context}
--- 本地知识庫信息结束 ---

--- 近期网络信息 (搜索结果摘要) ---
{web_context}
--- 近期网络信息结束 ---

**请按照以下结构撰写报告：**

**========================= 股票研究报告：{symbol} {company_name} =========================**

**1. 公司概览与发展历程**
   * 公司基本情况介绍 (来自 StockList 或 LLM 知识)。
   * 关键发展里程碑或转折点。

**2. 主营业务与核心产品**
   * 当前主营业务构成分析（例如，按产品、按地区）。
   * 核心产品/服务矩阵介绍及其市场地位。
   * 商业模式分析。

**3. 核心竞争力与技术壁垒**
   * 公司的主要竞争优势（如品牌、技术、成本、渠道、牌照等）。
   * 技术研发实力与专利情况（基于知识库信息或网络搜索）。
   * 行业门槛及公司的护城河。

**4. 同业对比分析**
   * 选取 1-2 家主要竞争对手 (来自 StockList 的 industry 或 LLM 知识)。
   * 在市场份额、盈利能力、产品、技术等方面进行简要对比，突出 {company_name} 的相对优势与劣势。

**5. 近三年财务数据概览与分析**
   * 展示近三个完整年度的关键财务指标（营收、净利润、毛利率、净利率、ROE等，依赖知识库中年报/半年报讨论或LLM知识）。
   * 对近三年财务表现进行简要归因分析。

**6. 近期积极变化与潜在催化剂**
   * 公司近期（过去半年内）发生的积极变化（基于网络搜索信息）。
   * 可能驱动股价上涨的潜在催化剂。

**7. 行业趋势与公司增长逻辑**
   * 公司所处行业的发展趋势、市场空间和竞争格局 (来自 StockList 的 industry 或 LLM 知识)。
   * 支撑公司未来增长的核心逻辑。

**8. 风险因素分析**
   * 公司面临的主要经营风险、财务风险、市场风险、政策风险等 (基于知识库或网络信息)。

**9. 股权激励与回购分析 (基于知识库信息)**
   * 分析近期（如有）股权激励计划的关键条款及其可能影响。
   * 分析近期（如有）股票回购计划的规模、价格区间及其信号意义。

**10. 近期机构调研摘要 (基于知识库信息)**
    * 总结近期（如有）机构调研活动中，公司管理层传递的关键信息或市场关注的焦点。

**11. 估值分析**
    * 结合历史数据和同业情况，进行 PE, PB 分析 (依赖LLM结合上下文)。
    * 如有信息，可进行 PEG 分析或简要提及 DCF 思路。
    * 给出当前估值水平的判断。

**12. 近期市场表现与归因**
    * 描述最近 5 个交易日股价的主要变动趋势 (此信息需外部获取，如下方所述)。
    * 结合近期市场情绪、公司新闻或行业动态，简要分析股价变动的原因。

**13. 近期产业积极因素**
    * 提及近期（过去1-3个月）与公司所处产业相关的宏观、政策或技术方面的积极动态 (基于网络搜索信息)。

**14. 总结与投资建议（可选）**
    * 对公司的整体看法和潜在投资价值进行总结。

**请基于上述信息和结构，生成报告内容。确保语言专业、客观。分析未知信息时，请说明信息来源或缺失。**
"""
    return prompt

# --- 主生成函数 (使用 get_stock_list_info, search_financial_news_google) ---
def generate_stock_report(db: Session, symbol: str) -> str | None:
    """
    生成单只股票的分析报告。
    """
    logger.info(f"Starting report generation for symbol: {symbol}")

    # 1. 获取股票名称 (从 StockList)
    stock_info = get_stock_list_info(db, symbol) # 使用正确的函数
    if not stock_info:
        logger.error(f"Stock list info not found for symbol {symbol}. Cannot generate report.")
        # 可以考虑让 LLM 在没有公司名称的情况下尝试生成，但信息会缺失
        # return None
        company_name = "该公司" # 使用占位符名称
        logger.warning(f"Stock list info not found for {symbol}, using placeholder name.")
    else:
        company_name = stock_info.name
        logger.info(f"Found stock: {symbol} - {company_name}")

    # 2. 从知识库检索 (使用 raw_content, ann_date)
    kb_queries = {
        "主业与核心产品": f"{company_name} 主营业务 核心产品",
        "竞争力与技术": f"{company_name} 竞争优势 技术壁垒",
        "财务分析": f"{company_name} 财务报告摘要",
        "风险因素": f"{company_name} 风险因素",
        "股权激励": f"{company_name} 股权激励计划",
        "回购": f"{company_name} 回购股份",
        "调研纪要": f"{company_name} 调研 纪要",
    }
    kb_context_parts = []
    for section, query in kb_queries.items():
        logger.info(f"Retrieving KB context for: {section}")
        # 检索时使用 symbol
        disclosures = retrieve_relevant_disclosures(db, symbol, query, top_k=2)
        if disclosures:
             # 格式化时使用 raw_content 和 ann_date
            kb_context_parts.append(f"--- 关于【{section}】的知识库信息 ---\n{format_kb_results(disclosures)}")

    full_kb_context = "\n".join(kb_context_parts) if kb_context_parts else "本地知识库中未检索到与特定章节高度相关的历史公告信息。\n"

    # 3. 从网络搜索 (使用 Google CSE 函数)
    logger.info("Retrieving recent web information using Google CSE...")
    web_results = search_financial_news_google(symbol, company_name, num_results_per_site=2, total_general_results=3) # 调整数量
    # 格式化时使用 snippet
    web_context = format_web_results(web_results)

    # --- 待办: 获取最近5日股价变动 ---
    # 这部分数据通常需要从另一个数据源获取 (如您项目中的 StockDaily 表或实时行情API)
    # 这里暂时留空，让 LLM 基于已有信息推断或说明缺失
    price_change_context = "最近5个交易日股价变动及归因：[此部分信息当前缺失，请基于其他上下文进行分析或说明]\n"
    # 您可以在 main.py 中查询 StockDaily 表获取数据，然后传递给 generate_stock_report 函数
    # 或者修改 generate_stock_report_prompt 将 price_change_context 作为参数传入

    # 4. 构建 Prompt
    final_prompt = generate_stock_report_prompt(symbol, company_name, full_kb_context + "\n" + price_change_context, web_context)

    # 5. 调用本地 LLM 生成报告
    generated_report = call_local_llm(final_prompt, max_tokens=3500) # 增加 token 预留

    # 检查返回是否为错误信息
    if generated_report.startswith("Error:"):
        logger.error(f"Failed to generate report for {symbol} due to LLM error.")
        return f"报告生成失败：{generated_report}" # 返回错误信息给调用者
    elif not generated_report.strip():
         logger.error(f"LLM returned empty response for {symbol}.")
         return "报告生成失败：LLM 返回为空。"

    logger.info(f"Successfully generated report for {symbol}")
    return generated_report

# 示例用法保持不变