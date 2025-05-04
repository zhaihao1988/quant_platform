import re
import random
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import io
from PyPDF2 import PdfReader


def get_random_user_agent():
    """生成随机现代版User-Agent"""
    chrome_versions = [
        '90.0.4430.212', '91.0.4472.124', '92.0.4515.131',
        '93.0.4577.63', '94.0.4606.71', '95.0.4638.54',
        '96.0.4664.45', '97.0.4692.71', '98.0.4758.102'
    ]
    platform = random.choice(['Windows NT 10.0; Win64; x64', 'Macintosh; Intel Mac OS X 12_4'])
    return f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.choice(chrome_versions)} Safari/537.36"


def setup_stealth_options():
    """配置反反爬浏览器选项"""
    chrome_options = Options()

    # 基础伪装
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # 实验性选项
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    # 禁用自动化特征
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")

    # 随机化视口大小
    viewports = [(1920, 1080), (1366, 768), (1536, 864)]
    w, h = random.choice(viewports)
    chrome_options.add_argument(f"--window-size={w},{h}")

    return chrome_options


def dynamic_wait(driver, selector_type, selector, timeout=15):
    """智能等待页面元素加载"""
    wait = WebDriverWait(driver, timeout)
    try:
        if selector_type == 'css':
            return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        elif selector_type == 'xpath':
            return wait.until(EC.presence_of_element_located((By.XPATH, selector)))
        elif selector_type == 'tag':
            return wait.until(EC.presence_of_element_located((By.TAG_NAME, selector)))
    except Exception as e:
        print(f"等待元素超时: {selector}")
        return None


def chinese_to_number(cn):
    """中文数字转阿拉伯数字（扩展支持）"""
    cn_dict = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
               '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    return cn_dict.get(cn, 0)


def extract_section_from_text(text, section_title):
    """增强版章节内容提取（修正目录干扰问题）"""
    # 匹配带章节编号的标题（排除目录条目）
    numbered_pattern = re.compile(
        rf'第([一二三四五六七八九十]+)节\s*{re.escape(section_title)}(?![^\n]*?\.{{4,}}\s*\d+)',
        re.IGNORECASE
    )
    numbered_match = numbered_pattern.search(text)

    # 匹配不带编号的标题（排除目录条目）
    unnumbered_pattern = re.compile(
        rf'(?<=\n)\s*{re.escape(section_title)}\s*\n(?![^\n]*?\.{{4,}}\s*\d+)',
        re.IGNORECASE
    )
    unnumbered_match = unnumbered_pattern.search(text)

    # 确定匹配情况
    if numbered_match:
        match = numbered_match
        cn_num = match.group(1)
        current_section_num = chinese_to_number(cn_num)
        start_idx = match.end()
    elif unnumbered_match:
        match = unnumbered_match
        current_section_num = None
        start_idx = match.end()
    else:
        return None

    # 跳过可能存在的点号行或空行
    next_text = text[start_idx:]
    non_empty_match = re.search(r'\S', next_text)
    if non_empty_match:
        start_idx += non_empty_match.start()

    # 尝试定位子标题（更灵活的匹配）
    subtitle_pattern = re.compile(r'[一二三四五六七八九十]+、\s*\S+')
    subtitle_match = subtitle_pattern.search(text[start_idx:start_idx + 3000])

    if subtitle_match:
        start_idx += subtitle_match.start()

    # 构建章节匹配模式
    section_pattern = re.compile(
        r'第([一二三四五六七八九十]+)节\s*',
        re.IGNORECASE
    )

    # 查找所有章节位置
    sections = []
    for m in section_pattern.finditer(text):
        section_num = chinese_to_number(m.group(1))
        sections.append((m.start(), section_num))

    # 确定结束位置
    end_idx = len(text)
    if current_section_num is not None:
        # 查找后续章节中第一个大于当前章节编号的
        for pos, num in sections:
            if pos > start_idx and num > current_section_num:
                end_idx = pos
                break
    else:
        # 没有章节编号时，查找下一个章节标题
        next_section_match = section_pattern.search(text[start_idx:])
        if next_section_match:
            end_idx = start_idx + next_section_match.start()

    # 提取并清理内容
    content = text[start_idx:end_idx].strip()
    # 清理残留的目录行
    content = re.sub(rf'^{re.escape(section_title)}.*?\.{{4,}}\s*\d+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{3,}', '\n\n', content)  # 合并多个空行
    return content if content else None

def extract_pdf_links_enhanced(driver, base_url):
    """动态PDF链接提取"""
    try:
        driver.get(base_url)
        time.sleep(random.uniform(1.5, 3.5))  # 初始加载等待

        # 执行滚动操作触发动态加载
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        time.sleep(random.uniform(0.8, 1.5))
        driver.execute_script("window.scrollTo(document.body.scrollHeight/3, document.body.scrollHeight*2/3);")
        time.sleep(random.uniform(1.0, 2.0))

        # 等待主要内容区域加载
        dynamic_wait(driver, 'css', 'div.main-content', timeout=20)

        # 解析处理后的DOM
        soup = BeautifulSoup(driver.page_source, 'lxml')
        pdf_links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if re.search(r'\.pdf$', href, re.I):
                full_url = urljoin(base_url, href.split('#')[0])
                pdf_links.append(full_url)

        return list(set(pdf_links))

    except Exception as e:
        print(f"链接提取异常: {str(e)}")
        return []


def stealth_download_pdf(url, referer):
    """隐蔽式PDF下载"""
    headers = {
        'Referer': referer,
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
    }

    try:
        response = requests.get(url, headers=headers,
                                timeout=20,
                                stream=True,
                                allow_redirects=True)
        response.raise_for_status()

        # 随机化下载节奏
        content = bytearray()
        for chunk in response.iter_content(chunk_size=1024):
            content.extend(chunk)
            time.sleep(random.uniform(0.01, 0.1))  # 模拟人类下载速度

        return io.BytesIO(content)
    except Exception as e:
        print(f"下载失败: {url} - {str(e)}")
        return None


if __name__ == "__main__":
    target_url = "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000887&announcementId=1223369489&orgId=gssz0000887&announcementTime=2025-04-29 00:00:00"

    # 初始化浏览器
    chrome_options = setup_stealth_options()
    driver = webdriver.Chrome(options=chrome_options)

    # 注入反检测脚本
    with open('./stealth.min.js') as f:
        stealth_script = f.read()
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": stealth_script
    })

    try:
        # 首次访问建立会话
        driver.get("https://www.cninfo.com.cn/")
        time.sleep(random.uniform(2.0, 4.0))

        # 获取目标页面
        pdf_links = extract_pdf_links_enhanced(driver, target_url)

        with open("result.txt", "w", encoding="utf-8") as f:
            f.write("提取到的PDF链接：\n")
            for link in pdf_links:
                f.write(f"{link}\n")

            for idx, link in enumerate(pdf_links):
                # 随机化处理间隔
                if idx > 0:
                    time.sleep(random.uniform(5.0, 15.0))

                f.write(f"\n\n正在处理: {link}\n")
                print(f"\n正在处理: {link}")

                pdf_file = stealth_download_pdf(link, target_url)
                if not pdf_file:
                    continue

                try:
                    pdf_reader = PdfReader(pdf_file)

                    if pdf_reader.is_encrypted:
                        print("  跳过加密PDF")
                        continue

                    full_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        full_text += page_text + "\n"
                        time.sleep(0.05)  # 模拟阅读节奏

                    section_content = extract_section_from_text(full_text, "管理层讨论与分析")

                    if section_content:
                        # 内容清理
                        cleaned = re.sub(r'([^\n])\n([^\n])', r'\1 \2', section_content)
                        cleaned = re.sub(r'\s{4,}', '   ', cleaned)

                        f.write(f"\n【管理层讨论与分析】章节内容：\n")
                        f.write(cleaned[:2000] + "..." if len(cleaned) > 2000 else cleaned)
                        f.write("\n" + "=" * 80 + "\n")

                        print(f"  成功提取章节内容（前2000字符）")
                    else:
                        f.write("  未找到该章节内容\n")

                except Exception as e:
                    print(f"  处理异常: {str(e)}")

    finally:
        driver.quit()
        print("\n处理结果已保存至 result.txt")