# analysis/dynamic_analyst.py
import logging
import json
import re
import os
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple

# --- Project Imports ---
from db.database import get_db_session
from db import crud
from core.llm_provider import LLMProviderFactory
from core.prompting import CATALYST_IDENTIFICATION_PROMPT, DYNAMIC_ANALYSIS_PROMPT_V2
from analysis.fundamental_analyzer import FundamentalAnalyzer

# Import the tool directly, as this script is the orchestrator
from __main__ import web_search

logger = logging.getLogger(__name__)

def perform_web_search_for_analyst(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    A wrapper for the web_search tool to be used by the DynamicAnalyst.
    """
    logger.info(f"Performing web search for: {query}")
    try:
        # The web_search tool is assumed to be available in the execution context
        search_results = web_search.call(search_term=query)
        # The tool output needs to be parsed into the desired format.
        # This is a placeholder for the actual parsing logic based on tool's output format.
        # Assuming the tool returns a list of dictionaries with 'title', 'link', 'snippet'.
        if isinstance(search_results, list):
             return search_results[:num_results]
        else:
             # Add logic here to parse search_results if it's a string or other format
             logger.warning(f"Web search returned unexpected format. Type: {type(search_results)}")
             return []

    except Exception as e:
        logger.error(f"Web search tool failed for query '{query}': {e}", exc_info=True)
        return []


class DynamicAnalyst:
    """
    An intelligent agent that performs a dynamic, two-stage analysis of a stock,
    combining internal database information with external, real-time web search data.
    The process is decoupled: it first prepares data and identifies search queries,
    then consumes external data to generate the final report.
    """
    def __init__(self, stock_code: str, stock_name: str):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.db_session = next(get_db_session())
        self.llm_provider = LLMProviderFactory.get_provider()
        self.fundamental_analyzer = FundamentalAnalyzer(self.db_session)
        
        # Data stores
        self.internal_data_str: Optional[str] = None

    def _gather_internal_data(self) -> str:
        """
        Gathers all relevant information about the stock from the local database
        and returns a formatted string for prompts.
        """
        logger.info(f"[{self.stock_code}] Gathering internal data...")
        
        today = date.today()
        financial_summary = self.fundamental_analyzer.analyze_stock(self.stock_code, today)
        company_info_obj = crud.get_stock_list_info(self.db_session, self.stock_code)
        
        company_info = {
            "name": company_info_obj.name,
            "industry": company_info_obj.industry,
            "area": company_info_obj.area,
            "list_date": company_info_obj.list_date.isoformat()
        } if company_info_obj else {}

        price_data_text = crud.get_recent_daily_data_as_text(self.db_session, self.stock_code, days_back=10)

        internal_data_str = f"""
        公司基本信息: {json.dumps(company_info, ensure_ascii=False)}
        
        最新财务摘要: {json.dumps(financial_summary, ensure_ascii=False, default=str)}
        
        最近10日股价数据:
        {price_data_text}
        """
        logger.info(f"[{self.stock_code}] Finished gathering internal data.")
        self.internal_data_str = internal_data_str.strip()
        return self.internal_data_str

    def _identify_catalysts(self) -> List[str]:
        """
        Uses an LLM to identify company-specific catalysts based on internal data.
        Returns a list of search query strings.
        """
        if not self.internal_data_str:
            logger.error(f"[{self.stock_code}] Cannot identify catalysts because internal data has not been gathered.")
            return []

        logger.info(f"[{self.stock_code}] Identifying company-specific catalysts...")
        prompt = CATALYST_IDENTIFICATION_PROMPT.format(internal_data=self.internal_data_str)
        
        try:
            response_text = self.llm_provider.generate(prompt)
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"[{self.stock_code}] Failed to find a valid JSON array in the catalyst response. Response: {response_text}")
                return []

            catalysts = json.loads(json_match.group(0))
            logger.info(f"[{self.stock_code}] Successfully identified {len(catalysts)} catalyst types: {catalysts}")
            return catalysts
        except Exception as e:
            logger.error(f"[{self.stock_code}] An unexpected error occurred during catalyst identification: {e}", exc_info=True)
            return []

    def prepare_analysis_data(self) -> Tuple[str, List[str]]:
        """
        Orchestrates the first stage of analysis: data gathering and catalyst identification.
        
        Returns:
            A tuple containing:
            - The formatted string of internal data.
            - A list of search queries (catalysts) to be executed externally.
        """
        logger.info(f"--- Stage 1: Preparing analysis data for {self.stock_name} ---")
        internal_data = self._gather_internal_data()
        catalysts = self._identify_catalysts()
        return internal_data, catalysts

    def generate_final_report(self, external_data: List[Dict[str, str]]) -> str:
        """
        Orchestrates the final stage of analysis after external data has been provided.
        """
        if not self.internal_data_str:
            return "错误: 内部数据未准备。请先运行 `prepare_analysis_data`。"

        logger.info(f"--- Stage 2: Generating final report for {self.stock_name} ---")
        
        if not external_data:
            external_data_str = "网络搜索未找到与已识别的刺激源直接相关的近期新闻。"
        else:
            external_data_str = "\n".join([
                f"- {item.get('title', 'N/A')}\n  摘要: {item.get('snippet', 'N/A')}\n  链接: {item.get('link', 'N/A')}" 
                for item in external_data
            ])
        
        logger.info(f"[{self.stock_code}] Synthesizing all data for final report...")
        final_prompt = DYNAMIC_ANALYSIS_PROMPT_V2.format(
            stock_code=self.stock_code,
            stock_name=self.stock_name,
            internal_data=self.internal_data_str,
            external_data=external_data_str
        )

        try:
            final_report = self.llm_provider.generate(final_prompt)
            logger.info(f"--- Finished Dynamic Analysis for {self.stock_name} ({self.stock_code}) ---")
            return final_report
        except Exception as e:
            logger.error(f"[{self.stock_code}] Failed to generate final report: {e}", exc_info=True)
            return "错误：在生成最终报告时发生意外错误。"
        finally:
            self.db_session.close()

def run_dynamic_analysis_pipeline(stock_code: str, stock_name: str, web_search_tool: Any):
    """
    This function demonstrates the decoupled, agent-driven workflow.
    It simulates how an external agent (like this script) would use the DynamicAnalyst.
    """
    logger.info(f"--- Starting Full Dynamic Analysis Pipeline for {stock_name} ---")
    
    # 1. Initialize the analyst
    analyst = DynamicAnalyst(stock_code=stock_code, stock_name=stock_name)
    
    # 2. Prepare internal data and get search queries
    internal_data, search_queries = analyst.prepare_analysis_data()
    
    if not search_queries:
        logger.warning("No search queries were generated. Proceeding without external data.")
        web_search_results = []
    else:
        # 3. The "agent" (this script) executes the web searches
        logger.info("Agent is now performing web searches...")
        all_results = []
        for query in search_queries:
            logger.info(f"Searching for: '{stock_name} {query}'")
            try:
                # This is where the actual tool call happens
                results = web_search_tool.call(search_term=f"{stock_name} {query}")
                if results and isinstance(results, list):
                    all_results.extend(results[:2]) # Take top 2 results per query
            except Exception as e:
                logger.error(f"Failed to execute web search for query '{query}': {e}")
        web_search_results = all_results

    # 4. Generate the final report with the gathered external data
    final_report = analyst.generate_final_report(external_data=web_search_results)
    
    # 5. Print and save the report
    print("\n" + "="*50 + "\n")
    print(final_report)
    print("\n" + "="*50 + "\n")

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    report_path = os.path.join(output_dir, f"dynamic_report_{stock_code}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    logger.info(f"Report saved to {report_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # +++ 用户输入区 +++
    # 请在这里修改您想分析的股票代码和名称
    # =======================================
    USER_STOCK_CODE = "000887"  # 例如: "000887" 代表中鼎股份
    USER_STOCK_NAME = "中鼎股份"   # 例如: "中鼎股份"
    # =======================================

    # This block requires the script to be run in a context where 'web_search' tool is defined.
    # For standalone testing, you'd mock this tool.
    
    # Mock web_search tool for standalone execution
    class MockWebSearch:
        def call(self, search_term: str):
            logger.info(f"[MOCK] Web searching for: {search_term}")
            return [
                {"title": f"Mock Result for {search_term}", "snippet": "This is a mock search result snippet.", "link": "http://mock.com"}
            ]
    
    mock_search_tool = MockWebSearch()
    
    run_dynamic_analysis_pipeline(
        stock_code=USER_STOCK_CODE, 
        stock_name=USER_STOCK_NAME,
        web_search_tool=mock_search_tool
    )