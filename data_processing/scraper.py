# data_processing/scraper.py
import collections
import logging
import re
import time
import random # <--- 确保导入 random 模块
import requests
import io
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
import pdfplumber

from core.llm_provider import SiliconFlowProvider
# --- 关键修改：从新的 prompting 文件导入 Prompts ---
from core.prompting import QA_EXTRACTION_PROMPT_V1, NARRATIVE_CLEANUP_PROMPT

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


def identify_common_lines(pdf_stream: io.BytesIO, threshold: int = 2) -> set:
    """
    仅识别PDF中的通用行（页眉/页脚），返回一个待删除行的集合。
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("需要 pdfplumber 库，请先安装: pip install pdfplumber")
        return set()

    all_lines = []
    with pdfplumber.open(pdf_stream) as pdf:
        if not pdf.pages:
            return set()
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines_on_page = [line.strip() for line in page_text.split('\n')]
                all_lines.extend(lines_on_page)

    if not all_lines:
        return set()

    line_counts = collections.Counter(line for line in all_lines if line)
    common_lines = {line for line, count in line_counts.items() if count > threshold}

    logger.info(f"识别到 {len(common_lines)} 条通用行（页眉/页脚）。")
    return common_lines


def clean_text_with_blacklist(text: str, blacklist: set) -> str:
    """
    使用一个黑名单（集合）来清理文本中的所有通用行。
    """
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip() not in blacklist]

    return "\n".join(cleaned_lines)

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


def reconstruct_paragraphs(text_with_soft_breaks: str) -> str:
    """
    智能重构段落，将因页面宽度导致的软换行合并。
    """
    if not text_with_soft_breaks:
        return ""

    logger.info("开始进行段落重构，合并段内换行...")

    # 按换行符分割成独立的行
    lines = text_with_soft_breaks.split('\n')
    reconstructed_text = ""

    for i, line in enumerate(lines):
        current_line = line.strip()
        if not current_line:
            continue

        reconstructed_text += current_line

        # 判断当前行是否是一个段落的真正结尾
        # 规则：如果行尾是结束性标点，或者是整个文本的最后一行，则保留换行
        # 增加了对表格中数字结尾的特殊判断，如果行尾是数字，也倾向于保留换行
        if (current_line.endswith(('。', '！', '？', '"',"：",";",")","]")) or
                (i + 1 == len(lines)) or
                (re.search(r'\d$', current_line))):
            reconstructed_text += "\n"
        else:
            # 如果不是段落结尾，则不加换行符，下一行的内容会直接拼接上来
            pass

    # 将可能出现的多个换行符合并为两个（段间空一行）
    final_text = re.sub(r'\n{2,}', '\n\n', reconstructed_text)

    logger.info("段落重构完成。")
    return final_text


# scraper.py (请替换此函数)

def remove_tables(text: str) -> str:
    """
    【V4 用户反馈优化版】使用更保守和精确的启发式规则，移除文本化表格，核心是防止误删正文。
    """
    logger.info("开始最终清理：使用【V4 用户反馈优化版】规则移除表格...")
    lines = text.split('\n')
    cleaned_lines = []

    # --- Pre-compiled Regex Patterns for efficiency ---
    # 规则 1: 匹配含有至少3个大间距空格的行，这是表格对齐的强特征。
    large_space_pattern = re.compile(r'\s{3,}')

    # 规则 2: 匹配包含多个独立数字/百分比的行
    numeric_pattern_str = r'\s*\(?-?[\d,.]+\)?%?\s*'
    multi_number_pattern = re.compile(f'({numeric_pattern_str}){{3,}}')  # 匹配连续出现3次以上的数字模式
    
    # 规则 3: 匹配标题行, 覆盖"数字."、"中文数字、"、"(一)"、"①"等多种格式
    title_pattern = re.compile(r'^\s*([（\(]\s*[一二三四五六七八九十\d]+\s*[）\)]|[①-⑩]|[一二三四五六七八九十\d]+[\.、．]|\■|\●|\◆)')

    removed_count = 0
    for line in lines:
        stripped_line = line.strip()

        # --- 步骤 1: 绝对保留规则 (如果匹配任意一条，则保留该行并跳过后续检查) ---
        # 1a. 保留空行 (它们通常是段落分隔符)
        if not stripped_line:
            cleaned_lines.append(line)
            continue
        
        # 1b. 保留看起来像完整句子的行
        if stripped_line.endswith('。'):
            cleaned_lines.append(line)
            continue
            
        # 1c. 保留被识别为标题的行
        if title_pattern.match(stripped_line):
            cleaned_lines.append(line)
            continue

        # --- 步骤 2: 移除规则 (通过了步骤1的行，现在是候选移除对象) ---
        
        # --- 规则 A (来自您的反馈): 移除短的、且有大空格的行 ---
        # "去掉非标题行...充斥超过多个空格...总字数小于15"
        # "非标题行" 的部分已经在上面的 1c 规则中处理了。
        if large_space_pattern.search(line) and len(stripped_line) < 15:
            logger.debug(f"Removing line (reason: user rule - short with large space): {stripped_line[:100]}...")
            removed_count += 1
            continue # 移除并处理下一行

        # --- 规则 B: 移除包含多个数字的行 (表格行的强特征) ---
        if multi_number_pattern.search(stripped_line):
            logger.debug(f"Removing line (reason: multiple numbers): {stripped_line[:100]}...")
            removed_count += 1
            continue # 移除并处理下一行
            
        # --- 规则 C (原有逻辑的优化): 移除有大空格但本身不短的行，
        # **仅当**它不包含太多文本内容时。这可以防止误删包含特殊格式的长句子。
        if large_space_pattern.search(line) and len(re.findall(r'[\u4e00-\u9fa5]', stripped_line)) < 10:
            logger.debug(f"Removing line (reason: large space with few Chinese chars): {stripped_line[:100]}...")
            removed_count += 1
            continue

        # --- 步骤 3: 如果没有任何移除规则匹配，则保留该行 ---
        cleaned_lines.append(line)

    logger.info(f"使用【V4 用户反馈优化版】规则移除了 {removed_count} 行疑似表格的行。")
    final_text = "\n".join(cleaned_lines)
    # 最后再规整一下可能因移除表格而产生的多余空行
    return re.sub(r'\n{3,}', '\n\n', final_text)

def extract_and_clean_narrative_section(full_text: str, section_title: str) -> Optional[str]:
    """
    【V4 封装流程】
    封装了提取并清理叙述性章节（如"管理层讨论与分析"）的完整传统流程。
    1. 按标题提取章节 (extract_section_from_text)
    2. 移除表格内容 (remove_tables)
    3. 重构段落，修复软换行 (reconstruct_paragraphs)
    """
    logger.info(f"执行传统方法流水线，提取并清理章节: '{section_title}'")

    # 步骤 1: 提取章节
    narrative_section = extract_section_from_text(full_text, section_title)
    if not narrative_section:
        # extract_section_from_text 内部已有日志，此处不再重复
        return None
    logger.info(f"✅ 步骤 1/3: 章节提取成功，初步长度: {len(narrative_section)}")

    # 步骤 2: 清理表格
    text_without_tables = remove_tables(narrative_section)
    logger.info(f"✅ 步骤 2/3: 表格清理完成，内容长度: {len(text_without_tables)}")

    # 步骤 3: 重构段落
    final_text = reconstruct_paragraphs(text_without_tables)
    logger.info(f"✅ 步骤 3/3: 段落重构完成，最终长度: {len(final_text)}")

    return final_text

def extract_section_from_text(text: str, section_title: str) -> Optional[str]:
    """
    【最终版】章节内容提取。
    核心思想：
    1. 使用灵活的模式定位起始标题。
    2. 使用一个"界碑"列表（包含所有可能的下一章节标题）来准确定位结束边界。
    """
    if not text or not section_title:
        return None

    logger.info(f"开始使用最终版逻辑提取章节: '{section_title}'")

    # --- 1. 定位起始位置 (此部分逻辑已经验证有效，予以保留) ---
    start_match = None
    patterns_to_try = [
        re.compile(r"^\s*" + re.escape(section_title) + r"\s*$", re.MULTILINE),
        re.compile(r"^\s*" + re.escape(section_title) + r"[^\w\n]*$", re.MULTILINE),
        re.compile(
            r"^\s*(?:第\s*[一二三四五六七八九十]+\s*[节章]|\d{1,2}(?:\.\d{1,2})?\s*[\.、]?)\s*"
            rf"{re.escape(section_title)}[^\w\n]*$",
            re.MULTILINE
        )
    ]
    for i, pattern in enumerate(patterns_to_try):
        start_match = pattern.search(text)
        if start_match:
            logger.info(f"通过模式 {i + 1} 找到章节标题 '{section_title}'，起始索引: {start_match.end()}")
            break

    if not start_match:
        logger.error(f"所有模式都未能找到章节标题: '{section_title}'")
        return None

    start_idx = start_match.end()

    # --- 2. 查找结束位置 (采用新的"界碑"策略) ---
    end_idx = len(text)

    # 定义一个列表，包含所有可能标志着我们目标章节结束的、下一个主章节的标题
    # 这些标题来自您PDF的目录页
    next_section_markers = [
        "公司治理报告",  # 第8章
        "董事、监事、高级管理人员",  # 第9章，用关键词即可
        "环境与社会责任",  # 第10章
        "重要事项",  # 第11章
        "股本变动及股东情况"  # 第12章
    ]

    found_end_positions = []

    # 遍历所有可能的结束标志
    for marker in next_section_markers:
        # 构造一个简单的、只在行首匹配的正则表达式
        pattern = re.compile(r"^\s*" + re.escape(marker), re.MULTILINE)
        match = pattern.search(text, start_idx)  # 从我们找到的章节开始位置之后搜索
        if match:
            found_end_positions.append(match.start())
            logger.debug(f"找到结束标志 '{marker}' 在索引: {match.start()}")

    # 如果找到了一个或多个结束标志，取其中最靠前（最小）的那个作为最终结束点
    if found_end_positions:
        end_idx = min(found_end_positions)
        logger.info(f"最终确定章节结束边界在索引: {end_idx}")
    else:
        logger.warning("未找到任何预定义的下一章节标题，将提取至文档末尾。")

    # --- 3. 提取与清理 (此部分逻辑不变) ---
    content = text[start_idx:end_idx].strip()

    # ... (清理 TOC 行、页码行、空行的 re.sub 代码保持不变) ...
    toc_pattern = re.compile(r'^[^\n]*\.{3,}\s*\d+\s*$', re.MULTILINE)
    content = toc_pattern.sub('', content)
    page_number_pattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
    content = page_number_pattern.sub('', content)
    content = re.sub(r'\n{3,}', '\n\n', content)

    logger.info(f"最终清理后内容长度: {len(content)}")

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
def fetch_announcement_text(detail_url: str, title: str, tag: Optional[str] = None) -> Optional[str]:
    """
    【V2 精简版】获取公告的原始全文。
    此函数的职责是获取最原始、最完整的文本，不做任何章节提取。
    后续的解析、提取、清理工作由调用方根据业务需求决定。

    处理流程:
    1. 尝试从详情页URL直接构造PDF链接并下载。
    2. 如果失败，则使用Selenium模拟浏览器访问详情页，找到PDF链接并下载。
    3. 获取PDF内容后，使用pdfplumber提取所有文本。
    4. 对提取的文本进行基础清理（移除页眉页脚）。
    """
    logger.info(f"开始为公告 '{title}' 获取全文，URL: {detail_url}")

    if not detail_url:
        logger.error("公告URL为空，无法获取内容。")
        return None

    # 优先尝试直接构造PDF链接
    pdf_url = try_construct_pdf_url(detail_url)
    pdf_content_stream = None

    if pdf_url:
        # 使用带有随机User-Agent的requests下载
        headers = {'User-Agent': get_random_user_agent()}
        try:
            response = requests.get(pdf_url, headers=headers, timeout=60)
            response.raise_for_status()
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_content_stream = io.BytesIO(response.content)
                logger.info("已通过构造的URL直接下载PDF。")
        except requests.exceptions.RequestException as e:
            logger.warning(f"直接下载PDF失败: {e}，将尝试Selenium后备方案。")
            pdf_content_stream = None

    # 如果直接下载失败，使用Selenium作为后备
    if not pdf_content_stream:
        if not SELENIUM_AVAILABLE:
            logger.error("直接下载PDF失败且Selenium不可用，无法获取公告内容。")
            return None
        logger.info("正在启动Selenium以间接获取PDF...")
        pdf_content_stream = stealth_download_pdf(detail_url)

    if not pdf_content_stream:
        logger.error(f"最终未能为URL获取到PDF内容: {detail_url}")
        return None

    # --- 【V2 回退流程】 ---
    # 回退到之前稳定的、基于文本重复的页眉页脚清理逻辑。
    try:
        # 步骤 1: 识别通用行（页眉/页脚）
        common_lines = identify_common_lines(pdf_content_stream)
        
        # 步骤 2: 重置流并提取全部文本
        pdf_content_stream.seek(0)
        with pdfplumber.open(pdf_content_stream) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        if not full_text:
            logger.warning("pdfplumber 未能提取到任何文本。")
            return None

        # 步骤 3: 使用黑名单清理文本
        cleaned_text = clean_text_with_blacklist(full_text, common_lines)
        
        logger.info(f"成功提取并清理文本 (最终长度: {len(cleaned_text)})")
        return cleaned_text

    except Exception as e:
        logger.error(f"在主文本提取流程中发生未知错误: {e}", exc_info=True)
        return None



def _parse_numbered_qa(text: str) -> List[Dict[str, str]]:
    """
    【内部解析器1】处理数字编号格式的Q&A (例如 "1、问题...")
    这是我们之前的V2版本逻辑，现在封装为辅助函数。
    """
    logger.info("检测到编号格式，使用【数字编号解析器】。")
    qa_pairs = []
    # 正则表达式：查找从 "数字、" 开始，直到下一个 "数字、" 或文本末尾的整个块
    question_blocks = re.finditer(r"(\d+[\.、．]\s*.*?)(?=\d+[\.、．]|$)", text, re.DOTALL)

    for match in question_blocks:
        parts = match.group(0).strip().split('\n')
        if not parts:
            continue

        question_text = re.sub(r"^\d+[\.、．]\s*", "", parts[0]).strip()
        answer_text = "\n".join(parts[1:]).strip()

        if answer_text.startswith("答:"):
            answer_text = answer_text[len("答:"):].strip()

        if question_text and answer_text:
            qa_pairs.append({"question": question_text, "answer": answer_text})

    return qa_pairs


def _parse_qa_by_prefix(text: str) -> List[Dict[str, str]]:
    """
    【内部解析器2】处理前缀格式的Q&A (例如 "Q: 问题... A: 答案...")
    """
    logger.info("检测到前缀格式，使用【Q:/A: 前缀解析器】。")
    qa_pairs = []

    # 使用 "Q:" 作为分隔符来切分整个文本块
    # 第一个元素是 "Q:" 之前的内容，我们将其忽略
    q_blocks = text.split('Q:')[1:]

    for block in q_blocks:
        # 在每个块中，使用 "A:" 来分割问题和答案
        parts = block.split('A:', 1)  # 只分割一次
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            if question and answer:
                qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs


def extract_qa_with_ai(full_text: str, model_override: Optional[str] = None) -> list[dict[str, str]]:
    """
    【AI封装版 V3.3 - 使用Prompt模块】
    使用 LLMProviderFactory 获取提供商，并允许在调用时覆盖模型。
    使用从 core.prompting 导入的标准化Prompt。
    """
    logger.info("开始使用【工厂模式+Prompt模块】的AI Provider解析Q&A...")
    if model_override:
        logger.info(f"此次调用将使用覆盖模型: {model_override}")

    # --- 使用从模块导入的 "黄金Prompt V1" ---
    prompt = QA_EXTRACTION_PROMPT_V1.format(full_text=full_text)

    try:
        # --- 使用工厂获取 Provider ---
        from core.llm_provider import LLMProviderFactory
        llm_provider = LLMProviderFactory.get_provider()

        ai_response_text = llm_provider.generate(prompt, model=model_override)

        if not ai_response_text:
            logger.error("AI Provider 未能返回任何内容。")
            return []

        logger.debug(f"AI Provider 返回的原始文本: {ai_response_text}")

        # 解析AI返回的JSON字符串
        json_match = re.search(r'\[.*\]', ai_response_text, re.DOTALL)
        if not json_match:
            logger.error(f"AI未能返回有效的JSON数组格式。返回内容: {ai_response_text}")
            return []

        parsed_json = json.loads(json_match.group(0))
        logger.info(f"AI成功解析出 {len(parsed_json)} 个问答对。")
        return parsed_json

    except ImportError as e:
        logger.error(f"导入模块时发生错误，请检查文件路径和名称：{e}")
        return []
    except Exception as e:
        logger.error(f"在AI解析流程中发生错误: {e}", exc_info=True)
        return []


def _chunk_text(text: str, max_chunk_size: int = 10000) -> list[str]:
    """
    一个简单的文本分块函数，会尽量在段落末尾分割。
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
        else:
            current_chunk += '\n' + p
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def extract_narrative_with_ai(full_text: str) -> Optional[str]:
    """
    【AI混合策略V4.1 - 工厂模式+Prompt模块】
    使用 LLMProviderFactory，并从 core.prompting 导入清理提示。
    """
    logger.info("开始使用【AI混合策略V4.1-工厂模式+Prompt模块】提取年报/半年报核心章节...")

    # 步骤1：先用传统方法进行"粗剪"，找到大概范围
    logger.info("步骤1: 使用传统方法进行初步章节定位...")
    rough_section = extract_section_from_text(full_text, "管理层讨论与分析")

    if not rough_section or len(rough_section) < 200:
        logger.warning("传统方法未能定位到有效的'管理层讨论与分析'章节，无法进行AI精修。")
        return rough_section

    logger.info(f"步骤1完成: 初步定位到章节文本，长度: {len(rough_section)}，准备进行分块AI精修。")

    # 步骤2：将大文本块分割成小块
    chunks = _chunk_text(rough_section)
    logger.info(f"步骤2完成: 文本被分割成 {len(chunks)} 个块，准备逐一处理。")

    # 步骤3：逐个处理每个文本块
    cleaned_chunks = []
    # 这个提示词现在只专注于清理，因为边界已经由传统方法确定
    # --- 关键修改：直接使用导入的 NARRATIVE_CLEANUP_PROMPT ---
    chunk_prompt_template = NARRATIVE_CLEANUP_PROMPT

    # --- 关键修改：使用工厂获取 Provider ---
    from core.llm_provider import LLMProviderFactory
    ai_provider = LLMProviderFactory.get_provider()

    for i, chunk in enumerate(chunks):
        logger.info(f"步骤3: 正在处理块 {i+1}/{len(chunks)}...")
        prompt = chunk_prompt_template.format(text_chunk=chunk)
        
        try:
            # 注意：这里我们让它使用提供商的默认模型，不进行覆盖
            cleaned_chunk = ai_provider.generate(prompt)
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
                logger.info(f"块 {i+1} 清理成功。")
            else:
                logger.warning(f"AI未能处理块 {i+1}，将使用原始块作为备用。")
                cleaned_chunks.append(chunk) # Fallback to original
        except Exception as e:
            logger.error(f"处理块 {i+1} 时发生严重错误: {e}，将使用原始块作为备用。")
            cleaned_chunks.append(chunk)


    # 步骤4：合并所有清理过的块
    final_text = '\n'.join(cleaned_chunks)
    logger.info(f"步骤4完成: 所有块已处理完毕并合并。最终文本长度: {len(final_text)}")
    return final_text


def run_qa_test():
    """测试函数1：专门用于解析"投资者关系活动记录表"(Q&A)"""
    TEST_PDF_PATH = "D:/project/quant_platform/data_processing/reports/test2.pdf"

    print("=" * 60)
    print("      ▶️ 开始测试【Q&A公告】的AI解析功能...")
    print(f"      测试文件: {TEST_PDF_PATH}")
    print("=" * 60)

    if not os.path.exists(TEST_PDF_PATH):
        logger.error(f"测试失败：找不到Q&A测试文件 -> {TEST_PDF_PATH}")
        return

    try:
        with open(TEST_PDF_PATH, 'rb') as f:
            full_text = "\n".join(
                [page.extract_text() for page in pdfplumber.open(io.BytesIO(f.read())).pages if page.extract_text()])

        if not full_text:
            raise ValueError("pdfplumber 未能从PDF中提取任何文本。")
        logger.info("PDF原始文本提取成功。")

        # 调用Q&A提取AI
        qa_results = extract_qa_with_ai(full_text)

        print("-" * 60)
        if qa_results:
            logger.info(f"AI成功解析出 {len(qa_results)} 个问答对！")
            print(json.dumps(qa_results, ensure_ascii=False, indent=4))
        else:
            logger.warning("AI未能从Q&A公告中解析出任何问答对。")
    except Exception as e:
        logger.error(f"处理Q&A公告时发生错误: {e}", exc_info=True)


def run_narrative_test():
    """
    测试【年报/半年报】的核心章节提取功能（传统方法版）。
    根据用户要求，暂时回退到稳定可靠的非AI方法。
    """
    print_separator("▶️ 开始测试【年报/半年报】的核心章节提取功能...")
    test_file_path = os.path.join(os.path.dirname(__file__), "reports", "test.pdf")
    print(f"      测试文件: {test_file_path}")
    print_separator()

    full_text = extract_pdf_text(test_file_path)
    if not full_text:
        logger.error("无法从PDF中提取文本，测试终止。")
        return

    logger.info("PDF原始文本提取成功。现在调用封装好的流水线函数...")

    # 调用新的、封装好的流水线函数
    final_text = extract_and_clean_narrative_section(full_text, "管理层讨论与分析")

    if final_text:
        save_path = os.path.join(os.path.dirname(__file__), "extracted_narrative_content.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        logger.info(f"✅ 提取的完整内容已成功保存到文件: {save_path}")
    else:
        logger.error("❌ 未能使用封装流水线提取到核心章节。")


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- 执行Q&A公告解析测试 ---
    run_qa_test()

    # --- 添加清晰的分割线 ---
    print("\n\n" + "#" * 25 + "  切换测试任务  " + "#" * 25 + "\n\n")

    # --- 执行年报/半年报解析测试 ---
    run_narrative_test()

    print("\n" + "=" * 60)
    print("所有AI解析测试结束。")