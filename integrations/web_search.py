# integrations/web_search.py
import logging
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urljoin
import re
import os # 导入 os

try:
    from dateutil.parser import parse as date_parse
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__) # Ensure logger is defined before use
    logger.warning("python-dateutil not installed. Date parsing might be less robust. pip install python-dateutil")


# --- Selenium Imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    # 导入之前 scraper.py 中的 setup_stealth_options 函数 (如果它在那里定义)
    # 或者在此处重新定义它
    from data_processing.scraper import setup_stealth_options # 假设 setup_stealth_options 在 scraper.py
    SELENIUM_AVAILABLE = True
except ImportError as e:
    # 捕获具体的导入错误
    logger = logging.getLogger(__name__)
    logger.error(f"Selenium or related modules import failed: {e}. Selenium features disabled.")
    SELENIUM_AVAILABLE = False
    # Define Selenium types as None if not available
    Options = None
    WebDriverWait = None
    EC = None
    By = None
    webdriver = None
    setup_stealth_options = None # 确保定义为 None

# 导入或定义 get_random_user_agent (来自之前的回答)
# (Definition code omitted for brevity - ensure it's present in the file)
import random
def get_random_user_agent() -> str:
    chrome_versions = ['123.0.6312.86', '124.0.6367.60', '122.0.6261.94']
    platforms = ['Windows NT 10.0; Win64; x64', 'Macintosh; Intel Mac OS X 10_15_7', 'X11; Linux x86_64']
    chosen_version = random.choice(chrome_versions)
    chosen_platform = random.choice(platforms)
    return ( f"Mozilla/5.0 ({chosen_platform}) AppleWebKit/537.36 "
             f"(KHTML, like Gecko) Chrome/{chosen_version} Safari/537.36" )

logger = logging.getLogger(__name__)

# --- 全局设置 ---
MAX_REPORTS_TO_CHECK = 5
TIME_LIMIT_MONTHS = 3
DEFAULT_TIMEOUT = 25
MAX_CONTENT_LENGTH = 5000

# --- 日期处理辅助函数 ---
def parse_date_string(date_str: str) -> Optional[datetime]:
    # ... (代码与上次回复相同) ...
    if not date_str: return None
    date_str = date_str.strip()
    try: return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: pass
    if DATEUTIL_AVAILABLE:
        try: return date_parse(date_str)
        except Exception: pass
    logger.warning(f"Failed to parse date string: {date_str}")
    return None


# --- 网页内容获取 (fetch_report_detail_content) ---
def fetch_report_detail_content(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    # ... (代码与上次回复相同) ...
    logger.info(f"Fetching report detail content for: {url}")
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        try:
            html_content = response.content.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            html_content = response.content.decode(response.apparent_encoding, errors='replace')

        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type: return None # 只处理 HTML

        soup = BeautifulSoup(html_content, 'lxml')
        content_div = soup.select_one('div#ctx-content.ctx-content, div#ctx-content') # 尝试两种选择器
        if not content_div: return None

        text = content_div.get_text(separator='\n', strip=True)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', text).strip()
        cleaned_text = re.sub(r'[ \t]{2,}', ' ', cleaned_text)
        cleaned_text = re.sub(r'^\s*[（\(]?\d{6}\.SZ|SH[）\)]?\s*[\(（]?[\s\S]*?[\)）]?\s*\n?', '', cleaned_text, count=1).strip()
        disclaimer_keywords = ["免责声明", "重要声明", "风险提示", "分析师声明", "评级说明"]
        for keyword in disclaimer_keywords:
             match = re.search(rf'^\s*{keyword}.*$', cleaned_text, re.MULTILINE | re.IGNORECASE)
             if match: cleaned_text = cleaned_text[:match.start()].strip(); break

        if not cleaned_text: return None
        logger.info(f"Successfully extracted text (approx {len(cleaned_text)} chars) from #ctx-content of {url}")
        return cleaned_text[:MAX_CONTENT_LENGTH] + ('...' if len(cleaned_text) > MAX_CONTENT_LENGTH else '')

    except Exception as e: logger.error(f"Error processing content from {url}: {e}", exc_info=True); return None


# --- 修改：东方财富 F10 研报抓取 (强制使用 Selenium) ---
def scrape_eastmoney_f10_reports(symbol: str) -> List[Dict]:
    """
    从东方财富 F10 页面抓取最近3个月的研报摘要信息，
    并获取符合条件的研报全文内容。(使用 Selenium)
    """
    if not SELENIUM_AVAILABLE or not setup_stealth_options: # 检查 Selenium 可用性 和 辅助函数
        logger.error("Selenium is required for scraping Eastmoney F10 reports but is not available or setup_stealth_options is missing.")
        return []

    # 1. 确定股票代码市场前缀 (SH/SZ)
    if symbol.startswith(('6', '9')): market_code = f"SH{symbol}"
    elif symbol.startswith(('0', '3', '2')): market_code = f"SZ{symbol}"
    else:
        logger.error(f"无法确定股票代码 {symbol} 的市场前缀 (SH/SZ)")
        return []

    f10_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/ResearchReport/Index?type=web&code={market_code}"
    logger.info(f"Scraping Eastmoney F10 Reports page (using Selenium): {f10_url}")

    results = []
    # 计算3个月前的日期
    if DATEUTIL_AVAILABLE:
        three_months_ago = datetime.now() - relativedelta(months=TIME_LIMIT_MONTHS)
    else:
        three_months_ago = datetime.now() - timedelta(days=TIME_LIMIT_MONTHS * 30)

    driver = None
    try:
        # --- 设置并启动 Selenium ---
        chrome_opts = setup_stealth_options()
        if not chrome_opts or not webdriver:
            logger.error("Failed to setup Selenium options or import webdriver.")
            return []
        driver = webdriver.Chrome(options=chrome_opts)

        # --- （可选）注入 Stealth Script ---
        stealth_script_path = 'stealth.min.js' # 确保路径正确
        if not os.path.exists(stealth_script_path):
            logger.warning(f"Stealth script not found at {stealth_script_path}.")
        else:
            try:
                with open(stealth_script_path, encoding='utf-8') as f:
                    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": f.read()})
                logger.info("Stealth script injected.")
            except Exception as e:
                logger.warning(f"Error executing stealth script: {e}")
        # --------------------------

        # --- 加载页面并等待动态内容 ---
        driver.get(f10_url)
        # 等待研报条目出现 (使用更具体的选择器)
        report_item_selector = "div#templateDiv div.section.first div.report a.tips-color" # 等待链接出现可能更可靠
        logger.info(f"Waiting up to {DEFAULT_TIMEOUT}s for report items ('{report_item_selector}') to load...")
        wait = WebDriverWait(driver, DEFAULT_TIMEOUT)
        try:
            # 等待至少一个匹配的元素出现
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, report_item_selector)))
            logger.info("Report items appear to be loaded.")
            # 短暂等待以确保 JS 完全执行完毕（可选）
            time.sleep(2)
        except Exception:
            logger.warning(f"Timed out waiting for report items ('{report_item_selector}'). Page might be empty or structure changed.")
            # 可以在这里尝试截屏 driver.save_screenshot(f'debug_em_{symbol}.png')
            return [] # 如果等待超时，直接返回空列表

        # --- 使用 BeautifulSoup 解析渲染后的页面源代码 ---
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # --- 提取数据 ---
        report_section = soup.select_one('div#templateDiv div.section.first')
        if not report_section:
            logger.warning("Could not find '研报摘要' section even after waiting.")
            return []

        report_items = report_section.select('div.report')[:MAX_REPORTS_TO_CHECK] # 限制检查数量
        if not report_items:
            logger.warning(f"Could not find any report items (div.report) within the first section after wait.")
            return []

        logger.info(f"Found {len(report_items)} potential report items to check.")
        processed_links = set()

        for item in report_items:
            title_tag = item.select_one('a.tips-color')
            date_tag = item.select_one('span > samp')

            if not title_tag or not date_tag or not title_tag.get('href'):
                logger.warning("Skipping item: Missing title, date, or link.")
                continue

            title = title_tag.get_text(strip=True)
            link = urljoin(f10_url, title_tag['href']) # 获取完整链接
            date_str = date_tag.get_text(strip=True)
            pub_date = parse_date_string(date_str)

            if link in processed_links: continue
            processed_links.add(link)

            # 检查日期
            if pub_date and pub_date >= three_months_ago:
                logger.info(f"Processing recent report: '{title}' ({pub_date.strftime('%Y-%m-%d')}) - Link: {link}")
                # 获取详情页内容
                full_content = fetch_report_detail_content(link) # 复用之前的函数
                if full_content:
                    results.append({
                        'source': 'Eastmoney F10 Report',
                        'title': title,
                        'link': link,
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'content': full_content
                    })
                else:
                     logger.warning(f"Failed to fetch content for recent report link: {link}")
                # 控制请求频率
                time.sleep(random.uniform(0.8, 1.8))
            elif pub_date:
                logger.info(f"Skipping report (older than {TIME_LIMIT_MONTHS} months): '{title}' ({pub_date.strftime('%Y-%m-%d')})")
            else:
                logger.warning(f"Skipping report (could not parse date): '{title}' ({date_str})")

        logger.info(f"Finished scraping Eastmoney F10. Got {len(results)} results within time limit.")
        return results

    except Exception as e:
        logger.error(f"Error scraping Eastmoney F10 page ({f10_url}) with Selenium: {e}", exc_info=True)
        return []
    finally:
        if driver:
            try:
                driver.quit()
                logger.debug("Selenium driver quit.")
            except Exception as e_quit:
                logger.error(f"Error quitting selenium driver: {e_quit}")


# --- 主函数接口 (保持不变) ---
def get_web_search_results(symbol: str, company_name: Optional[str] = None) -> List[Dict]:
    """
    获取指定股票的网络搜索结果，目前仅从东方财富 F10 研报页面获取。
    """
    return scrape_eastmoney_f10_reports(symbol)