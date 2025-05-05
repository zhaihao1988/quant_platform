# integrations/web_search.py
import logging
import time
import random
import requests
from typing import List, Dict, Optional

# Import settings correctly
from config.settings import settings

logger = logging.getLogger(__name__)

# Use keys from settings
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
GOOGLE_CX = settings.GOOGLE_CX

# Target financial sites
FINANCIAL_SITES = ["eastmoney.com", "10jqka.com.cn", "xueqiu.com", "finance.sina.com.cn"]

# Remove the old search_stock_news function as it was flawed

def search_google_cse(query: str, num: int = 3, site_search: Optional[str] = None, **kwargs) -> List[Dict]:
    """
    Calls the Google Custom Search Engine API with specified parameters.

    Args:
        query: The search query.
        num: Number of results to request (max 10 per API call).
        site_search: Restrict search to a specific site (e.g., "eastmoney.com").
        **kwargs: Additional parameters for the API (e.g., dateRestrict="m6").

    Returns:
        A list of dictionaries, each containing 'title', 'link', 'snippet'.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logger.error("Google API Key or CX ID is missing in settings. Cannot perform web search.")
        return []

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": min(num, 10), # API limit is 10 per request
        **kwargs # Include additional parameters like dateRestrict
    }
    if site_search:
        params["siteSearch"] = site_search
        # siteSearchFilter='i' includes results, 'e' excludes. 'i' is default.

    log_params = params.copy()
    log_params["key"] = "[REDACTED]" # Don't log the API key
    logger.info(f"Querying Google CSE with params: {log_params}")

    results = []
    try:
        response = requests.get(base_url, params=params, timeout=20) # Increased timeout
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        items = data.get("items", [])

        for item in items:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet", "").replace("\n", " ").strip() # Clean snippet
            })
        logger.info(f"Received {len(results)} results from Google CSE for query: '{query}'" + (f", site: {site_search}" if site_search else ""))
        return results

    except requests.exceptions.Timeout:
         logger.error(f"Google CSE request timed out for query: '{query}'")
         return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Google CSE request failed for query '{query}': {e}")
        # Check for specific status codes if possible (e.g., 403 for key issues, 429 for quota)
        if e.response is not None:
             logger.error(f"Response status: {e.response.status_code}, Response body: {e.response.text[:200]}")
        return []
    except Exception as e:
        logger.error(f"Error processing Google CSE response for query '{query}': {e}", exc_info=True)
        return []

def search_financial_news_google(symbol: str, company_name: Optional[str], num_results_per_site: int = 2, total_general_results: int = 3) -> List[Dict]:
    """
    Uses Google CSE to search specified financial sites and general news for a stock.

    Args:
        symbol: The stock symbol.
        company_name: The company name (optional but helpful).
        num_results_per_site: Number of results to fetch from each specific site.
        total_general_results: Number of general news results to fetch.

    Returns:
        A list of unique search result dictionaries.
    """
    all_results = []
    display_name = f"{symbol} {company_name}" if company_name else symbol
    search_term_base = f"{company_name} {symbol}" if company_name else symbol

    # 1. General news search (add specific keywords, consider date restrict)
    general_query = f"{search_term_base} 股票分析 OR 公司新闻 OR 最新研究报告"
    # Add date restriction for recent news (e.g., last 6 months)
    general_results = search_google_cse(general_query, num=total_general_results, dateRestrict="m6")
    all_results.extend(general_results)

    # 2. Specific financial site searches
    for site in FINANCIAL_SITES:
        # Query can be simpler when site is specified
        site_query = f"{search_term_base}"
        # Random delay between site searches
        time.sleep(random.uniform(0.3, 0.8))
        site_results = search_google_cse(site_query, num=num_results_per_site, site_search=site, dateRestrict="m6") # Also restrict date here
        all_results.extend(site_results)

    # 3. Deduplicate results based on the 'link'
    unique_results_dict = {}
    for res in all_results:
        link = res.get('link')
        if link and link not in unique_results_dict: # Use dict for faster lookup
            unique_results_dict[link] = res

    unique_results = list(unique_results_dict.values())
    logger.info(f"Collected {len(unique_results)} unique web results using Google CSE for {display_name}.")

    # Return a limited number of top results (e.g., max 10-15)
    return unique_results[:15]