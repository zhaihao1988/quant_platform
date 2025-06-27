# integrations/web_search.py
import logging
import requests
from config.settings import settings # 导入 settings
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# --- 主函数接口 (V3 - Google Search API 版) ---
def get_web_search_results(query: str, top_n: int = 5) -> List[Dict]:
    """
    【V3】使用 Google Custom Search API 执行网络搜索。
    根据 settings.py 中的 GOOGLE_API_KEY 和 GOOGLE_CX 配置。
    """
    logger.info(f"Executing web search with Google API for query: '{query}' (top_n={top_n})")

    # 1. 检查配置
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CX:
        logger.error("Google API Key or CX is not configured in settings.py. Web search is disabled.")
        return []

    # 2. 构建请求参数
    # https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    params = {
        'key': settings.GOOGLE_API_KEY,
        'cx': settings.GOOGLE_CX,
        'q': query,
        'num': top_n, # 指定返回结果数量
        'sort': 'date:r:d:w', # 按日期排序，限制在最近一周内
        'lr': 'lang_zh-CN', # 限制为中文简体
        'gl': 'cn', # 地理位置限制为中国
    }

    # 3. 发送API请求
    try:
        # 使用 Google 的官方 API 端点
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=20
        )
        response.raise_for_status()  # 如果请求失败 (4xx or 5xx), 抛出异常

    except requests.exceptions.Timeout:
        logger.error(f"Google Search API request timed out for query: '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during Google Search API request: {e}", exc_info=True)
        # 打印部分响应内容以帮助调试
        if e.response is not None:
            logger.error(f"Google API Error Response: {e.response.status_code} - {e.response.text}")
        return []

    # 4. 解析和格式化结果
    try:
        data = response.json()
        items = data.get('items', [])

        if not items:
            logger.info(f"No web search results found for query: '{query}'")
            return []

        # 将返回的条目格式化为我们需要的字典结构
        results = [
            {
                "title": item.get('title'),
                "link": item.get('link'),
                "snippet": item.get('snippet'),
                "displayLink": item.get('displayLink'),
                # 可以添加其他需要的字段，例如 'pagemap' 中的 'metatags'
            }
            for item in items
        ]
        logger.info(f"Successfully retrieved {len(results)} web search results for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"Failed to parse JSON response from Google Search API: {e}", exc_info=True)
        logger.error(f"Raw response text: {response.text}")
        return []