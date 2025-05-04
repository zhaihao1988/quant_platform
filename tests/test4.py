import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import io
from PyPDF2 import PdfReader


def chinese_to_number(cn):
    """中文数字转阿拉伯数字"""
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


if __name__ == "__main__":
    target_url = "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000887&announcementId=1223369489&orgId=gssz0000887&announcementTime=2025-04-29 00:00:00"

    chrome_options = Options()
    chrome_options.add_argument("--headnew")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        pdf_links = extract_pdf_links_enhanced(None, target_url, driver)

        # 创建结果文件
        with open("../result.txt", "w", encoding="utf-8") as f:
            f.write("提取到的PDF链接：\n")
            for link in pdf_links:
                f.write(f"{link}\n")

            for link in pdf_links:
                f.write(f"\n\n正在处理: {link}\n")
                print(f"\n正在处理: {link}")

                try:
                    response = requests.get(link, timeout=30)
                    response.raise_for_status()

                    pdf_file = io.BytesIO(response.content)
                    pdf_reader = PdfReader(pdf_file)

                    if pdf_reader.is_encrypted:
                        f.write("  跳过加密PDF\n")
                        print("  跳过加密PDF")
                        continue

                    full_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"

                    section_title = "管理层讨论与分析"
                    section_content = extract_section_from_text(full_text, section_title)

                    if section_content:
                        # 清理内容并写入文件
                        cleaned_content = re.sub(r'([^\n])\n([^\n])', r'\1 \2', section_content)
                        f.write(f"\n【{section_title}】章节内容：\n")
                        f.write(cleaned_content + "\n")
                        f.write("=" * 80 + "\n")

                        # 控制台输出（保留前2000字符）
                        print(f"\n【{section_title}】章节内容：")
                        print(cleaned_content[:2000] + "..." if len(cleaned_content) > 2000 else cleaned_content)
                        print("=" * 80)
                    else:
                        f.write("  未找到该章节内容\n")
                        print("  未找到该章节内容")

                except Exception as e:
                    error_msg = f"  处理失败: {str(e)}"
                    print(error_msg)
                    f.write(error_msg + "\n")

    finally:
        driver.quit()
        print("\n处理结果已保存至 result.txt")