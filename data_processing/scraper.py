# data_processing/scraper.py
# Keep your existing scraper.py content here as it focuses solely on scraping.
# Ensure it has necessary imports and the 'stealth.min.js' file is accessible.
# (Your provided scraper.py code seems appropriate for this file's responsibility)

import logging
import re
import time
import random
import requests
import io
from typing import List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
# Selenium imports (consider making optional if not always needed)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    Options = None # Define as None if Selenium not installed
    WebDriverWait = None
    EC = None
    By = None

# PyPDF2 or pypdf (choose one, pypdf recommended)
# from PyPDF2 import PdfReader # Old
from pypdf import PdfReader # New

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. 随机 User-Agent
# ------------------------------------------------------------------------------
def get_random_user_agent() -> str:
    chrome_versions = [
        '90.0.4430.212', '91.0.4472.124', '92.0.4515.131',
        '93.0.4577.63', '94.0.4606.71', '95.0.4638.54',
        '96.0.4664.45', '97.0.4692.71', '98.0.4758.102'
    ]
    platform = random.choice([
        'Windows NT 10.0; Win64; x64',
        'Macintosh; Intel Mac OS X 12_4'
    ])
    return (
        f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{random.choice(chrome_versions)} Safari/537.36"
    )

# ------------------------------------------------------------------------------
# 2. Stealth 浏览器配置
# ------------------------------------------------------------------------------
def setup_stealth_options() -> Options:
    if not SELENIUM_AVAILABLE:
        logger.warning("Selenium is not installed. Stealth options cannot be configured.")
        return None
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    # 随机视口尺寸
    viewports = [(1920, 1080), (1366, 768), (1536, 864)]
    w, h = random.choice(viewports)
    chrome_options.add_argument(f"--window-size={w},{h}")
    return chrome_options

# ------------------------------------------------------------------------------
# 3. 动态等待
# ------------------------------------------------------------------------------
def dynamic_wait(driver, selector_type: str, selector: str, timeout: int = 15):
    if not SELENIUM_AVAILABLE or not driver:
        logger.warning("Selenium not available or driver not provided. Cannot perform dynamic wait.")
        return None
    wait = WebDriverWait(driver, timeout)

    try:
        if selector_type == 'css':
            return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        elif selector_type == 'xpath':
            return wait.until(EC.presence_of_element_located((By.XPATH, selector)))
        elif selector_type == 'tag':
            return wait.until(EC.presence_of_element_located((By.TAG_NAME, selector)))
    except Exception as e:
        logger.error(f"Dynamic wait failed for {selector_type}='{selector}': {e}")
        return None

# ------------------------------------------------------------------------------
# 4. 中文数字转阿拉伯数字
# ------------------------------------------------------------------------------
def chinese_to_number(cn: str) -> int:
    cn_dict = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }
    return cn_dict.get(cn, 0)

# ------------------------------------------------------------------------------
# 5. 增强版章节提取
# ------------------------------------------------------------------------------
def extract_section_from_text(text: str, section_title: str) -> Optional[str]:
    numbered_pattern = re.compile(
        rf'第([一二三四五六七八九十]+)节\s*{re.escape(section_title)}'
        r'(?![^\n]*?\.{4,}\s*\d+)',
        re.IGNORECASE
    )
    unnumbered_pattern = re.compile(
        rf'(?<=\n)\s*{re.escape(section_title)}\s*\n'
        r'(?![^\n]*?\.{4,}\s*\d+)',
        re.IGNORECASE
    )
    m_num = numbered_pattern.search(text)
    m_unn = unnumbered_pattern.search(text)

    if m_num:
        current_sec = chinese_to_number(m_num.group(1))
        start_idx = m_num.end()
    elif m_unn:
        current_sec = None
        start_idx = m_unn.end()
    else:
        return None

    # 跳过空行或点号行
    tail = text[start_idx:]
    m_non = re.search(r'\S', tail)
    if m_non:
        start_idx += m_non.start()

    # 小标题偏移
    subtitle_pat = re.compile(r'[一二三四五六七八九十]+、\s*\S+')
    m_sub = subtitle_pat.search(text[start_idx:start_idx + 3000])
    if m_sub:
        start_idx += m_sub.start()

    # 后续章节查找
    sec_pat = re.compile(r'第([一二三四五六七八九十]+)节', re.IGNORECASE)
    end_positions = []
    for m in sec_pat.finditer(text):
        num = chinese_to_number(m.group(1))
        if current_sec is None or num > current_sec:
            if m.start() > start_idx:
                end_positions.append(m.start())
    end_idx = min(end_positions) if end_positions else len(text)

    content = text[start_idx:end_idx].strip()
    # 清理残留目录行及多余空行
    content = re.sub(rf'^{re.escape(section_title)}.*?\.{{4,}}\s*\d+$',
                     '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content or None

# ------------------------------------------------------------------------------
# 6. 提取 PDF 链接
# ------------------------------------------------------------------------------
def extract_pdf_links_enhanced(driver, base_url: str) -> List[str]:
    try:
        driver.get(base_url)
        time.sleep(random.uniform(1.5, 3.5))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        time.sleep(random.uniform(0.8, 1.5))
        driver.execute_script("window.scrollTo(document.body.scrollHeight/3, document.body.scrollHeight*2/3);")
        time.sleep(random.uniform(1.0, 2.0))

        dynamic_wait(driver, 'css', 'div.main-content', timeout=20)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                full = urljoin(base_url, href.split('#')[0])
                links.append(full)
        return list(set(links))
    except Exception:
        return []

# ------------------------------------------------------------------------------
# 7. 隐蔽下载 PDF
# ------------------------------------------------------------------------------
def stealth_download_pdf(url: str, referer: str) -> Optional[io.BytesIO]:
    headers = {
        'Referer': referer,
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20,
                            stream=True, allow_redirects=True)
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=1024):
            buf.extend(chunk)
            time.sleep(random.uniform(0.01, 0.1))
        return io.BytesIO(buf)
    except Exception:
        return None

# ------------------------------------------------------------------------------
# 8. 主函数：抓取全文或章节
# ------------------------------------------------------------------------------
def fetch_announcement_text(detail_url: str, title: str) -> Optional[str]:
    """
    Fetches announcement from a URL (detail_url), attempts to find and download
    a PDF, extracts text, and optionally extracts a specific section based on title.

    Args:
        detail_url: The URL pointing to the announcement page or directly to the PDF.
        title: The title of the announcement (used to decide section extraction).

    Returns:
        The extracted text (full or section), or None if fetching/parsing fails.
    """
    logger.info(f"Fetching announcement: '{title}' from {detail_url}")

    # Determine if Selenium is needed (e.g., for complex sites like cninfo)
    # You might refine this condition based on the URL patterns
    use_selenium = SELENIUM_AVAILABLE and "cninfo.com.cn" in detail_url

    pdf_url = detail_url # Assume detail_url might be the direct PDF link initially
    pdf_stream = None

    if use_selenium:
        driver = None
        try:
            chrome_opts = setup_stealth_options()
            if not chrome_opts: return None # Exit if Selenium setup failed

            # Consider using webdriver_manager or specifying driver path
            driver = webdriver.Chrome(options=chrome_opts)

            # Inject stealth.js if needed (ensure file exists)
            try:
                with open('stealth.min.js', encoding='utf-8') as f:
                    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": f.read()})
            except FileNotFoundError:
                logger.warning("stealth.min.js not found. Proceeding without stealth script injection.")
            except Exception as e:
                 logger.warning(f"Error executing stealth script: {e}")


            # Establish session (optional, might help with cookies)
            # driver.get("https://www.cninfo.com.cn/")
            # time.sleep(random.uniform(1.0, 2.0))

            # Extract actual PDF links from the detail page
            pdf_links = extract_pdf_links_enhanced(driver, detail_url)
            if not pdf_links:
                logger.warning(f"No PDF links found on page {detail_url} using Selenium.")
                # Maybe try direct download as fallback?
            else:
                pdf_url = pdf_links[0] # Use the first PDF link found
                logger.info(f"Found PDF link via Selenium: {pdf_url}")
                # Download using stealth requests, using detail_url as referer
                pdf_stream = stealth_download_pdf(pdf_url, detail_url)

        except Exception as e:
            logger.error(f"Error during Selenium operation for {detail_url}: {e}", exc_info=True)
            # Fallback or return None
        finally:
            if driver:
                driver.quit()
    else:
        # If not using Selenium, attempt direct download or basic link finding
        logger.info(f"Attempting direct download or basic extraction for {detail_url} (Selenium not used/needed).")
        # Try direct download first
        pdf_stream = stealth_download_pdf(detail_url, detail_url) # Use URL as its own referer? Or a generic one?

        # If direct download fails or it's HTML, try basic link finding
        if pdf_stream is None or b'html' in pdf_stream.getvalue()[:100].lower(): # Quick check if it looks like HTML
             try:
                 headers = {'User-Agent': get_random_user_agent()}
                 response = requests.get(detail_url, headers=headers, timeout=15)
                 response.raise_for_status()
                 soup = BeautifulSoup(response.text, 'lxml')
                 found_links = []
                 for a in soup.find_all('a', href=True):
                     href = a['href']
                     if href.lower().endswith('.pdf'):
                         found_links.append(urljoin(detail_url, href.split('#')[0]))

                 if found_links:
                     pdf_url = found_links[0]
                     logger.info(f"Found PDF link via basic parsing: {pdf_url}")
                     pdf_stream = stealth_download_pdf(pdf_url, detail_url)
                 else:
                      logger.warning(f"No PDF link found via basic parsing on {detail_url}")
                      pdf_stream = None # Ensure stream is None if no PDF found

             except Exception as e:
                 logger.error(f"Error during basic request/parsing for {detail_url}: {e}")
                 pdf_stream = None


    # --- Process the PDF stream if obtained ---
    if not pdf_stream:
        logger.error(f"Failed to obtain PDF stream for {title} from {detail_url} (or derived URL {pdf_url}).")
        return None

    try:
        reader = PdfReader(pdf_stream)
        if len(reader.pages) == 0:
            logger.warning(f"PDF appears to be empty or corrupted: {pdf_url}")
            return None

        full_text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            except Exception as page_err:
                logger.warning(f"Error extracting text from page {i+1} of {pdf_url}: {page_err}")
                continue # Skip problematic pages

        if not full_text.strip():
             logger.warning(f"Text extraction yielded empty result for {pdf_url}.")
             return None

        logger.info(f"Successfully extracted text (length: {len(full_text)}) from {pdf_url}")

        # Section extraction based on title
        if '年度报告' in title or '半年度报告' in title:
            section = extract_section_from_text(full_text, "管理层讨论与分析")
            if section:
                logger.info("Extracted '管理层讨论与分析' section.")
                return section
            else:
                logger.warning("Could not extract '管理层讨论与分析', returning full text.")
                return full_text # Fallback to full text

        # Other announcement types: return full text
        return full_text.strip()

    except Exception as e:
        logger.error(f"Error processing PDF stream from {pdf_url}: {e}", exc_info=True)
        return None
