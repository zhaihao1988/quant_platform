# utils/scraper.py


import re
import time
import random
import requests
import io
from typing import List, Optional
from urllib.parse import urljoin
from functools import wraps

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PyPDF2 import PdfReader

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
    wait = WebDriverWait(driver, timeout)
    try:
        if selector_type == 'css':
            return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        elif selector_type == 'xpath':
            return wait.until(EC.presence_of_element_located((By.XPATH, selector)))
        elif selector_type == 'tag':
            return wait.until(EC.presence_of_element_located((By.TAG_NAME, selector)))
    except Exception:
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
    - 年报/中报（title 包含 '年度报告' 或 '半年度报告'）：
        提取“管理层讨论与分析”章节；
    - 其他公告：完整转换全文并返回。
    """
    chrome_opts = setup_stealth_options()
    driver = webdriver.Chrome(options=chrome_opts)
    # 注入 stealth.js（请确保路径正确）
    with open('stealth.min.js', encoding='utf-8') as f:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument", {"source": f.read()}
        )

    try:
        # 建立会话
        driver.get("https://www.cninfo.com.cn/")
        time.sleep(random.uniform(2.0, 4.0))

        # 提取 PDF 链接
        pdfs = extract_pdf_links_enhanced(driver, detail_url)
        if not pdfs:
            return None

        # 下载第一个 PDF
        pdf_stream = stealth_download_pdf(pdfs[0], detail_url)
        if not pdf_stream:
            return None

        # 解析文本
        reader = PdfReader(pdf_stream)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"
            time.sleep(0.05)

        # 区分年报/中报与其他公告
        if '年度报告' in title or '半年度报告' in title:
            section = extract_section_from_text(full_text, "管理层讨论与分析")
            return section or full_text

        # 其他公告，返回全文
        return full_text

    finally:
        driver.quit()
