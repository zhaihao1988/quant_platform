# integrations/web_search.py
import logging
import time
import random
import requests
from data_processing.scraper import fetch_announcement_text  # re-use PDF/HTML parser
from config import settings
API_KEY = "AIzaSyB0Kv14UpjEDv59HEOV4ducTqaPk8633L8"
CX = "533a067c36f9d48f1"

def search_stock_news(symbol: str, num: int = 3) -> list[str]:
    """Return top news article texts for a given stock symbol."""
    url = ("https://www.googleapis.com/customsearch/v1"
           f"?key={API_KEY}&cx={CX}&q={symbol}+stock+news&num={num}")
    resp = requests.get(url)
    articles = resp.json().get("items", [])
    snippets = []
    for item in articles:
        link = item.get("link")
        if link:
            text = fetch_announcement_text(link)
            if text:
                snippets.append(text[:800])
    return snippets
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用您配置文件中的 API Key 和 CX
# 注意：API 密钥不应硬编码在此处，从 settings 中读取是好的实践
GOOGLE_API_KEY = settings.GOOGLE_API_KEY # 假设您在 Settings 中添加了 GOOGLE_API_KEY
GOOGLE_CX = settings.GOOGLE_CX         # 假设您在 Settings 中添加了 GOOGLE_CX

# 目标财经网站列表
FINANCIAL_SITES = ["eastmoney.com", "10jqka.com.cn", "xueqiu.com", "finance.sina.com.cn"]

def search_google_cse(query: str, num: int = 3, site_search: str | None = None) -> list[dict]:
    """
    调用 Google Custom Search Engine API。
    返回结果列表，每个结果包含 'title', 'link', 'snippet'。
    """
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num,
        # 可以添加其他参数，如 dateRestrict (e.g., "d7" for last 7 days, "m6" for last 6 months)
        # "dateRestrict": "m6" # 限制结果为最近6个月
    }
    if site_search:
        params["siteSearch"] = site_search
        params["siteSearchFilter"] = "i" # Include results from this site

    logger.info(f"Querying Google CSE: q='{query}', num={num}" + (f", siteSearch={site_search}" if site_search else ""))

    results = []
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)
        data = response.json()
        items = data.get("items", [])

        for item in items:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet", "").replace("\n", " ").strip() # 使用 snippet
            })
        logger.info(f"Received {len(results)} results from Google CSE.")
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Google CSE request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Error processing Google CSE response: {e}")
        return []

def search_financial_news_google(symbol: str, company_name: str | None, num_results_per_site: int = 2, total_general_results: int = 4) -> list[dict]:
    """
    使用 Google CSE 搜索指定财经网站和通用新闻。
    优先使用 Google 返回的 snippet。
    """
    all_results = []
    display_name = f"{symbol} {company_name}" if company_name else symbol

    # 1. 通用新闻搜索 (最近6个月)
    general_query = f"{display_name} 股票 新闻 分析 研究报告"
    # 在 params 中添加 dateRestrict="m6"
    # general_results = search_google_cse(general_query, num=total_general_results, dateRestrict="m6") # 需要确认 dateRestrict 格式
    general_results = search_google_cse(general_query, num=total_general_results) # 暂时不加日期限制，依赖 Google 排序
    all_results.extend(general_results)

    # 2. 特定网站搜索
    for site in FINANCIAL_SITES:
        site_query = f"{display_name}" # 查询词可以简化，因为已限定网站
        time.sleep(random.uniform(0.5, 1.0)) # 稍微增加延迟
        site_results = search_google_cse(site_query, num=num_results_per_site, site_search=site)
        all_results.extend(site_results)

    # 3. 去重 (基于链接)
    unique_results = []
    seen_links = set()
    for res in all_results:
        link = res.get('link')
        if link and link not in seen_links:
            unique_results.append(res)
            seen_links.add(link)

    logger.info(f"Collected {len(unique_results)} unique web results using Google CSE for {display_name}.")
    # 返回最多 N 条结果，例如 10 条
    return unique_results[:10]

# 示例用法 (需要确保 Settings 中配置了 GOOGLE_API_KEY 和 GOOGLE_CX)
# if __name__ == "__main__":
#     # 需要先设置环境变量或创建 .env 文件
#     # export GOOGLE_API_KEY='YourApiKey'
#     # export GOOGLE_CX='YourCxId'
#     try:
#         results = search_financial_news_google("600519", "贵州茅台")
#         if results:
#             for r in results:
#                 print(f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}\n---")
#         else:
#             print("No results found or API keys missing/invalid.")
#     except NameError:
#          print("Ensure GOOGLE_API_KEY and GOOGLE_CX are set in config/settings.")
