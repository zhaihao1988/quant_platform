import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import io
from PyPDF2 import PdfReader


def extract_pdf_links_enhanced(html, base_url, driver=None):
    """PDF链接提取增强版"""
    use_selenium = False
    if driver is None:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        use_selenium = True

    try:
        driver.get(base_url)
        rendered_html = driver.page_source
        soup = BeautifulSoup(rendered_html, 'lxml')

        pdf_urls = []
        for a_tag in soup.find_all('a', href=re.compile(r'\.pdf$', re.I)):
            href = a_tag.get('href', '')
            if href:
                full_url = urljoin(base_url, href.split('#')[0])
                pdf_urls.append(full_url)

        unique_urls = list(set(pdf_urls))
        valid_urls = [url for url in unique_urls if re.search(r'\.pdf$', url, re.I)]
        return valid_urls

    finally:
        if use_selenium:
            driver.quit()


def extract_section_from_text(text, section_title):
    """从PDF文本中提取指定章节内容"""
    # 匹配带章节编号的标题（如"第三节 管理层讨论与分析"）
    pattern = re.compile(rf'第[一二三四五六七八九十]+节\s*{re.escape(section_title)}', re.IGNORECASE)
    match = pattern.search(text)

    if not match:
        # 尝试匹配不带章节编号的标题
        alt_pattern = re.compile(rf'\n\s*{re.escape(section_title)}\s*\n', re.IGNORECASE)
        match = alt_pattern.search(text)

    if not match:
        return None

    start_idx = match.end()

    # 查找下一个章节开始的位置
    next_section_pattern = re.compile(r'第[一二三四五六七八九十]+节\s*')
    end_match = next_section_pattern.search(text[start_idx:])

    if end_match:
        end_idx = start_idx + end_match.start()
    else:
        end_idx = len(text)

    return text[start_idx:end_idx].strip()


if __name__ == "__main__":
    # 替换为实际目标URL（链接1）
    target_url = "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000887&announcementId=1223369508&orgId=gssz0000887&announcementTime=2024-04-30 00:00:00"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        pdf_links = extract_pdf_links_enhanced(None, target_url, driver)
        print("\n提取到的PDF链接：")
        for link in pdf_links:
            print(link)

        # 处理每个PDF链接
        for link in pdf_links:
            print(f"\n正在处理: {link}")
            try:
                # 下载PDF
                response = requests.get(link, timeout=30)
                response.raise_for_status()

                # 读取PDF内容
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PdfReader(pdf_file)

                if pdf_reader.is_encrypted:
                    print("  跳过加密PDF")
                    continue

                # 提取全部文本
                full_text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

                # 提取目标章节
                section_title = "管理层讨论与分析"
                section_content = extract_section_from_text(full_text, section_title)

                print(f"\n【{section_title}】章节内容：")
                if section_content:
                    print(section_content)
                else:
                    print("  未找到该章节内容")
                print("=" * 80)

            except Exception as e:
                print(f"  处理失败: {str(e)}")

    finally:
        driver.quit()