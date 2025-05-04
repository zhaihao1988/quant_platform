import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import io
from PyPDF2 import PdfReader

def extract_pdf_links_enhanced(html, base_url, driver=None):
    """PDF链接提取增强版（带方法标记）"""
    # 初始化浏览器用于动态渲染（可选）
    use_selenium = False
    if driver is None:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        use_selenium = True

    try:
        # 动态渲染获取完整HTML
        driver.get(base_url)
        rendered_html = driver.page_source
        soup = BeautifulSoup(rendered_html, 'lxml')

        pdf_urls = []

        # 方式4：常规链接（优先级1）
        for a_tag in soup.find_all('a', href=re.compile(r'\.pdf$', re.I)):
            href = a_tag.get('href', '')
            if href:
                full_url = urljoin(base_url, href.split('#')[0])
                pdf_urls.append(full_url)

        for embed in soup.find_all('embed', type=re.compile(r'application/pdf|text/pdf', re.I)):
            src = embed.get('src', '')
            if src:
                full_url = urljoin(base_url, src.split('#')[0])
                pdf_urls.append(full_url)
                print(f"✅ [Method 1 - Embed标签] 发现PDF链接: {full_url}")

        # 结果处理
        unique_urls = list(set(pdf_urls))
        valid_urls = [url for url in unique_urls if re.search(r'\.pdf$', url, re.I)]

        return valid_urls

    finally:
        if use_selenium:
            driver.quit()

# 使用示例（保持不变）
if __name__ == "__main__":
    target_url = "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000887&announcementId=1223369489&orgId=gssz0000887&announcementTime=2025-04-29 00:00:00"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        pdf_links = extract_pdf_links_enhanced(None, target_url, driver)
        print("\n最终提取的PDF链接：")
        for link in pdf_links:
            print(link)

        # 新增PDF转文本逻辑
        for link in pdf_links:
            print(f"\n正在处理: {link}")
            try:
                # 下载PDF
                response = requests.get(link, timeout=10)
                response.raise_for_status()

                # 创建内存文件对象
                pdf_file = io.BytesIO(response.content)

                # 提取文本
                text = ""
                pdf_reader = PdfReader(pdf_file)

                if pdf_reader.is_encrypted:
                    print("  跳过加密PDF")
                    continue

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                print("提取文本内容:")
                print(text.strip() or "  未找到文本内容")
                print("=" * 80)

            except Exception as e:
                print(f"  处理失败: {str(e)}")

    finally:
        driver.quit()