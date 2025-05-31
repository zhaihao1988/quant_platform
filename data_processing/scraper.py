# data_processing/scraper.py
import logging
import re
import time
import random # <--- 确保导入 random 模块
import requests
import io
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs

# --- Selenium Imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    # Define Selenium types as None if not available
    Options = None
    WebDriverWait = None
    EC = None
    By = None
    webdriver = None # 添加这个

# --- PDF Parsing ---
from pypdf import PdfReader # 使用 pypdf

logger = logging.getLogger(__name__)

# --- 保持不变的辅助函数 ---
# get_random_user_agent, setup_stealth_options, chinese_to_number,
# extract_section_from_text, stealth_download_pdf, try_construct_pdf_url
# ... (这些函数的代码如上次回复所示) ...

def dynamic_wait(driver, selector_type: str, selector: str, timeout: int = 15):
    """智能等待页面元素加载"""
    if not SELENIUM_AVAILABLE or not driver or not WebDriverWait or not EC or not By: # 检查所有依赖
        logger.warning("Selenium 相关模块未完全导入或 driver 未提供。无法执行 dynamic_wait。")
        return None
    wait = WebDriverWait(driver, timeout)
    by_map = {'css': By.CSS_SELECTOR, 'xpath': By.XPATH, 'tag': By.TAG_NAME, 'id': By.ID}
    if selector_type not in by_map:
         logger.error(f"无效的选择器类型: {selector_type}")
         return None

    try:
        # 等待元素变得可见可能更可靠
        logger.debug(f"Waiting for element ({selector_type}='{selector}') to be visible...")
        element = wait.until(EC.visibility_of_element_located((by_map[selector_type], selector)))
        logger.debug(f"Element ({selector_type}='{selector}') is visible.")
        return element
    except Exception as e:
        # 超时是常见情况，记录为 warning 而不是 error
        logger.warning(f"Dynamic wait timed out for ({selector_type}='{selector}'): {e}")
        return None
# ==================== 新增/整合的函数 ====================
def get_random_user_agent() -> str:
    """生成一个随机的、看起来比较真实的 User-Agent 字符串"""
    # 基于常见的现代浏览器版本
    chrome_versions = [
        '110.0.5481.177', '111.0.5563.64', '112.0.5615.49',
        '113.0.5672.92', '114.0.5735.198', '115.0.5790.170',
        '116.0.5845.180', '117.0.5938.149', '118.0.5993.88',
        '119.0.6045.159', '120.0.6099.129', '121.0.6167.85',
        '122.0.6261.94', '123.0.6312.86', '124.0.6367.60' # 保持更新
    ]
    # 常见的操作系统平台
    platforms = [
        'Windows NT 10.0; Win64; x64',      # Windows 10/11
        'Macintosh; Intel Mac OS X 10_15_7', # macOS Catalina/Big Sur/Monterey etc.
        'X11; Linux x86_64',               # Linux
        # '(Linux; Android 13; Pixel 7)', # 可选：添加移动端UA
    ]
    # 随机选择版本和平台
    chosen_version = random.choice(chrome_versions)
    chosen_platform = random.choice(platforms)

    # 构造 User-Agent 字符串
    user_agent = (
        f"Mozilla/5.0 ({chosen_platform}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{chosen_version} Safari/537.36"
    )
    logger.debug(f"Generated User-Agent: {user_agent}")
    return user_agent
def try_construct_pdf_url(detail_url: str) -> Optional[str]:
    """尝试根据详情页 URL 构造直接的 PDF 下载链接"""
    try:
        parsed_url = urlparse(detail_url)
        # 使用 parse_qs 解析查询参数，它返回一个字典，值为列表
        query_params = parse_qs(parsed_url.query)

        # 从列表中获取第一个元素，如果键不存在或列表为空，则默认为 None
        announcement_id = query_params.get('announcementId', [None])[0]
        announcement_time_str = query_params.get('announcementTime', [None])[0]

        if not announcement_id or not announcement_time_str:
            logger.warning(f"无法从详情页 URL 中提取 announcementId 或 announcementTime: {detail_url}")
            return None

        # 解析日期，格式化为 yyyy-MM-DD
        try:
            # 时间字符串格式如 '2025-04-29 00:00:00'
            ann_date = datetime.strptime(announcement_time_str.split(' ')[0], '%Y-%m-%d')
            date_path = ann_date.strftime('%Y-%m-%d')
        except ValueError:
            logger.warning(f"无法解析 announcementTime 格式: {announcement_time_str}")
            return None
        except Exception as date_e:
             logger.error(f"解析日期时发生意外错误: {date_e}")
             return None


        # 构造 PDF URL (使用 https)
        pdf_url = f"https://static.cninfo.com.cn/finalpage/{date_path}/{announcement_id}.PDF"
        logger.info(f"尝试构造的 PDF URL: {pdf_url}")
        return pdf_url

    except Exception as e:
        logger.error(f"构造 PDF URL 时出错: {e}", exc_info=True)
        return None

def chinese_to_number(cn_num: str) -> int:
    """将简单的中文数字（一到十）转换为阿拉伯数字"""
    # 也可以扩展这个函数以支持更复杂的中文数字转换，如果需要的话
    num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }
    # 查找字符串中所有的中文数字字符并转换第一个找到的
    # (基于 "第X节" 的模式，通常只有一个数字)
    num = 0
    for char in cn_num:
         if char in num_map:
              num = num_map[char]
              break # 找到第一个就停止
    return num
def extract_section_from_text(text: str, section_title: str) -> Optional[str]:
    """
    增强版章节内容提取（基于 test5.py 中的逻辑）。
    尝试从 PDF 全文中提取指定标题的章节内容。

    Args:
        text: PDF 的完整文本内容。
        section_title: 要提取的章节标题 (例如 "管理层讨论与分析")。

    Returns:
        提取到的章节文本内容，如果未找到则返回 None。
    """
    if not text or not section_title:
        return None

    logger.debug(f"Attempting to extract section: '{section_title}'")

    # 1. 构造正则表达式
    # 匹配 "第[一二三四五六七八九十]+节<空格或换行符>*章节标题"
    # 排除像目录页那样的行 "... <数字>"
    numbered_pattern = re.compile(
        r'第\s*([一二三四五六七八九十]+)\s*(?:节|章)'  # <--- 已修改
        r'\s*'rf'{re.escape(section_title)}'
        r'(?![^\n]*?\s*\.{3,}\s*\d+)'r'\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    unnumbered_pattern = re.compile(  # 这个通常不变
        r'^\s*'rf'{re.escape(section_title)}'
        r'(?![^\n]*?\s*\.{3,}\s*\d+)'r'\s*$',
        re.IGNORECASE | re.MULTILINE
    )

    # 2. 查找起始位置
    start_match = numbered_pattern.search(text)
    current_section_num = None
    if start_match:
        current_section_num = chinese_to_number(start_match.group(1))
        start_idx = start_match.end()
        logger.debug(f"Found numbered section start (Section {current_section_num}) at index {start_idx}")
    else:
        start_match = unnumbered_pattern.search(text)
        if start_match:
            start_idx = start_match.end()
            logger.debug(f"Found unnumbered section start at index {start_idx}")
        else:
            logger.warning(f"Section title '{section_title}' not found.")
            return None # 未找到起始标题

    # 跳过起始标题后的空行或特殊字符行，找到实际内容的开始
    content_start_match = re.search(r'\S', text[start_idx:], re.MULTILINE)
    if content_start_match:
        start_idx += content_start_match.start()
        logger.debug(f"Actual content start index adjusted to: {start_idx}")
    else:
        logger.warning("No actual content found after the section title.")
        return "" # 标题找到了，但后面没内容

    # 3. 查找结束位置
    end_idx = len(text)
    next_section_pattern = re.compile(r'^\s*第\s*([一二三四五六七八九十]+)\s*(?:节|章)',
                                      re.IGNORECASE | re.MULTILINE)  # <--- 已修改
    next_section_match = next_section_pattern.search(text, start_idx)

    if next_section_match:
        # 如果当前章节有编号，确保找到的下一个章节编号更大（防止错误匹配文档内的引用）
        if current_section_num is not None:
            next_num = chinese_to_number(next_section_match.group(1))
            if next_num > current_section_num:
                end_idx = next_section_match.start()
                logger.debug(f"Found next numbered section (Section {next_num}) at index {end_idx}")
            else:
                # 找到的 "第X节" 编号不大于当前编号，可能不是真正的下一章节标题，继续搜索
                logger.debug(f"Found potential next section (Section {next_num}) but number is not greater than current ({current_section_num}), searching further...")
                next_section_match_further = next_section_pattern.search(text, next_section_match.end())
                while next_section_match_further:
                     next_num_further = chinese_to_number(next_section_match_further.group(1))
                     if next_num_further > current_section_num:
                          end_idx = next_section_match_further.start()
                          logger.debug(f"Found correct next numbered section (Section {next_num_further}) further down at index {end_idx}")
                          break
                     next_section_match_further = next_section_pattern.search(text, next_section_match_further.end())
                else: # while 循环正常结束，没有找到更大的编号
                     logger.debug("No subsequent section with a greater number found, using end of text.")
                     end_idx = len(text)

        # 如果当前章节没有编号，找到的第一个 "第X节" 就是结束标志
        else:
            end_idx = next_section_match.start()
            logger.debug(f"Found next numbered section at index {end_idx}")
    else:
         logger.debug("No subsequent '第X节' found, using end of text as boundary.")


    # 4. 提取并初步清理
    content = text[start_idx:end_idx].strip()
    logger.info(f"Extracted section content length: {len(content)}")

    # 进一步清理：移除可能的页眉页脚（简单规则，可能需要优化）
    # 例如，移除看起来像页码的行 (^\s*\d+\s*$)
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    # 合并多个空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content if content else None
def stealth_download_pdf(url: str, referer: Optional[str] = None) -> Optional[io.BytesIO]:
    """
    使用 requests 隐蔽地下载 PDF 文件内容。

    Args:
        url: 要下载的 PDF URL。
        referer: 可选的 Referer 请求头。

    Returns:
        包含 PDF 内容的 BytesIO 对象，如果下载失败则返回 None。
    """
    headers = {
        'User-Agent': get_random_user_agent(), # 使用随机 User-Agent
        'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1', # 尝试升级到 HTTPS
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin' if referer and urlparse(url).netloc == urlparse(referer).netloc else 'cross-site', # 根据 referer 判断
        'Sec-Fetch-User': '?1',
        # 添加 CH-UA 头 (可选，模拟现代浏览器)
        'Sec-CH-UA': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
        'Sec-CH-UA-Mobile': '?0',
        'Sec-CH-UA-Platform': '"Windows"', # 或 "macOS", "Linux" 等
    }
    # 仅在 referer 有效时添加
    if referer:
        headers['Referer'] = referer

    try:
        logger.debug(f"Requesting PDF from {url} with headers: {headers}")
        # 增加 stream=True 和 timeout，允许重定向
        resp = requests.get(url, headers=headers, timeout=45, stream=True, allow_redirects=True) # 增加超时
        resp.raise_for_status() # 检查 HTTP 错误 (4xx, 5xx)

        # 检查 Content-Type 是否真的是 PDF
        content_type = resp.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type:
             logger.warning(f"URL {url} Content-Type is not PDF: {content_type}. Headers: {resp.headers}")
             # 可以选择返回 None 或者尝试处理（如果网站用错误类型返回PDF）
             # return None # 严格模式
             pass # 宽松模式，继续尝试读取内容

        # 使用 iter_content 读取内容
        pdf_content = b''.join(chunk for chunk in resp.iter_content(chunk_size=8192))

        if not pdf_content:
            logger.warning(f"Downloaded empty content from {url}")
            return None

        logger.info(f"Successfully downloaded PDF content from {url} ({len(pdf_content)} bytes)")
        return io.BytesIO(pdf_content) # 返回 BytesIO 对象

    except requests.exceptions.Timeout:
         logger.error(f"Stealth download timed out for {url}")
         return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Stealth download request failed for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during stealth download for {url}: {e}", exc_info=True)
        return None

# ========================================================
# ==================== 新增 setup_stealth_options 函数 ====================
def setup_stealth_options() -> Optional[Options]:
    """配置反反爬虫的 Selenium Chrome 浏览器选项"""
    if not SELENIUM_AVAILABLE or not Options: # 检查 Options 是否导入成功
        logger.error("Selenium Options class is not available. Cannot setup stealth options.")
        return None

    chrome_options = Options()
    # 基础伪装
    # chrome_options.add_argument("--headless") # <--- 注意：为了调试方便，可以先注释掉 headless
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}") # 使用随机 UA
    chrome_options.add_argument("--disable-blink-features=AutomationControlled") # 禁用 webdriver 标志

    # 实验性选项
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # 禁用自动化特征
    chrome_options.add_argument("--disable-infobars")       # 禁用 "Chrome is being controlled by automated test software"
    chrome_options.add_argument("--disable-dev-shm-usage") # 解决 Docker 或某些 Linux 环境下的问题
    chrome_options.add_argument("--no-sandbox")            # 禁用沙箱模式

    # 随机化视口大小 (可选)
    viewports = [(1920, 1080), (1366, 768), (1536, 864)]
    w, h = random.choice(viewports)
    chrome_options.add_argument(f"--window-size={w},{h}")

    # 其他可能的选项 (根据需要添加)
    # chrome_options.add_argument('--ignore-certificate-errors')
    # chrome_options.add_argument('--allow-running-insecure-content')
    # chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--proxy-server='direct://'")
    # chrome_options.add_argument("--proxy-bypass-list=*")
    # chrome_options.add_argument('--log-level=3') # 减少日志输出

    logger.debug("Stealth Chrome options configured.")
    return chrome_options
# ======================================================================
# --- 主函数：抓取全文或章节 ---
def fetch_announcement_text(detail_url: str, title: str) -> Optional[str]:
    """
    获取公告文本。优先尝试直接构造 PDF URL 下载，
    如果失败，则使用 Selenium 查找 embed 标签获取 URL。
    """
    logger.info(f"Fetching announcement: '{title}' from {detail_url}")
    if "摘要" in title:
        logger.info(f"Skipping processing for abstract announcement: {title}")
        return None

    pdf_stream = None
    pdf_url_to_process = None

    # --- 1. 尝试直接构造 PDF URL (优先) ---
    constructed_pdf_url = try_construct_pdf_url(detail_url)
    if constructed_pdf_url:
        logger.info(f"Attempting direct download using constructed URL: {constructed_pdf_url}")
        pdf_stream = stealth_download_pdf(constructed_pdf_url, referer=detail_url) # 使用 detail_url 作为 Referer
        if pdf_stream:
            pdf_url_to_process = constructed_pdf_url
            logger.info("Direct download successful using constructed URL.")
        else:
            logger.warning("Direct download failed using constructed URL, falling back to Selenium...")

    # --- 2. 如果直接构造/下载失败，尝试 Selenium (查找 embed 标签) ---
    if not pdf_stream and SELENIUM_AVAILABLE:
        logger.info("Attempting Selenium fallback to find PDF link in <embed> tag...")
        driver = None
        selenium_pdf_url = None # 初始化变量
        try:
            chrome_opts = setup_stealth_options()
            if not chrome_opts: return None

            # 确保 webdriver 导入成功
            if not webdriver:
                 logger.error("Selenium is available but webdriver could not be imported.")
                 return None
            driver = webdriver.Chrome(options=chrome_opts)

            # --- 注入 Stealth Script ---
            stealth_script_path = 'stealth.min.js' # 假设在项目根目录
            if not os.path.exists(stealth_script_path):
                 logger.warning(f"Stealth script not found at {stealth_script_path}. Proceeding without it.")
            else:
                 try:
                     with open(stealth_script_path, encoding='utf-8') as f:
                         driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": f.read()})
                     logger.info("Stealth script injected.")
                 except Exception as e:
                     logger.warning(f"Error executing stealth script: {e}")
            # --------------------------

            driver.get(detail_url)
            # 不需要太复杂的滚动，因为 embed 通常在主框架中
            time.sleep(random.uniform(1.0, 2.0))

            # --- 修改等待逻辑：等待 embed 标签出现并可见 ---
            embed_selector = 'embed.pdfobject[src*=".pdf"], embed.pdfobject[src*=".PDF"]' # CSS Selector
            logger.info(f"Waiting up to 35s for embed element to be visible: {embed_selector}")
            # 使用 dynamic_wait 等待元素可见
            wait_element = dynamic_wait(driver, 'css', embed_selector, timeout=35)

            if wait_element:
                logger.info("Selenium found visible embed PDF element. Extracting src...")
                try:
                    selenium_pdf_url_raw = wait_element.get_attribute('src')
                    if selenium_pdf_url_raw:
                         # 清理 URL，移除 # 后面的部分，并确保是绝对 URL
                         selenium_pdf_url = urljoin(detail_url, selenium_pdf_url_raw.split('#')[0])
                         logger.info(f"Extracted PDF URL from embed src: {selenium_pdf_url}")
                    else:
                         logger.warning("Found embed element but could not extract src attribute.")
                except Exception as e_get_attr:
                     logger.error(f"Error getting src attribute from embed element: {e_get_attr}")
            else:
                logger.warning(f"Selenium timed out waiting for embed PDF element on {detail_url}.")
                # （可选）即使等待失败，也可以尝试最后解析一次页面源码
                # soup = BeautifulSoup(driver.page_source, 'lxml')
                # embed_tag = soup.select_one(embed_selector)
                # if embed_tag and embed_tag.get('src'): ...

            # --- 如果通过 Selenium 找到了 URL，尝试下载 ---
            if selenium_pdf_url:
                 logger.info(f"Attempting download using URL from Selenium: {selenium_pdf_url}")
                 pdf_stream = stealth_download_pdf(selenium_pdf_url, referer=detail_url)
                 if pdf_stream:
                     pdf_url_to_process = selenium_pdf_url
                     logger.info("Successfully downloaded PDF using URL found via Selenium.")
                 else:
                     logger.warning(f"Download failed for URL found via Selenium: {selenium_pdf_url}")
            else:
                 logger.warning("Could not extract a valid PDF URL using Selenium method.")


        except Exception as e:
            logger.error(f"General Selenium operation error for {detail_url}: {e}", exc_info=True)
        finally:
            if driver:
                driver.quit()
                logger.debug("Selenium driver quit.")

    # --- 处理获取到的 PDF 流 ---
    if not pdf_stream:
        logger.error(f"Ultimately failed to obtain PDF stream for '{title}' from {detail_url}")
        return None

    logger.info(f"Processing PDF stream obtained from URL: {pdf_url_to_process}")
    try:
        reader = PdfReader(pdf_stream)
        if reader.is_encrypted:
             logger.warning(f"PDF is encrypted: {pdf_url_to_process}")
             return None
        if len(reader.pages) == 0:
            logger.warning(f"PDF appears to be empty: {pdf_url_to_process}")
            return None

        full_text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            except Exception as page_err:
                # 记录更详细的页码错误
                logger.warning(f"Error extracting text from page {i+1}/{len(reader.pages)} of {pdf_url_to_process}: {page_err}")
                continue # 继续处理下一页

        if not full_text.strip():
             logger.warning(f"Text extraction yielded empty result for {pdf_url_to_process}.")
             # 即使提取为空，也可能返回一个空字符串或 None，取决于下游是否需要区分
             return None # 或者 return ""

        logger.info(f"Successfully extracted text (length: {len(full_text)}) from {pdf_url_to_process}")

        # 章节提取逻辑
        if '年度报告' in title or '半年度报告' in title:
            if '摘要' not in title:
                 logger.debug(f"Attempting to extract '管理层讨论与分析' from non-abstract report: {title}")
                 section = extract_section_from_text(full_text, "管理层讨论与分析")
                 if section:
                     logger.info("Extracted '管理层讨论与分析' section.")
                     return section
                 else:
                     logger.warning("Could not extract '管理层讨论与分析', returning full text for non-abstract report.")
                     return full_text.strip()
            else:
                 logger.info("Returning full text for abstract report.")
                 return full_text.strip()
        else:
            logger.info("Returning full text for other announcement types.")
            return full_text.strip()

    except ImportError as e_pypdf:
         # 特别处理 pypdf 可能的导入错误
         logger.critical(f"pypdf library might be missing or corrupted: {e_pypdf}. Please install/reinstall it.")
         return None
    except Exception as e:
        logger.error(f"Error processing PDF stream from {pdf_url_to_process}: {e}", exc_info=True)
        return None