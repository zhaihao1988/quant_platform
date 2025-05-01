# utils/scraper.py

from newspaper import Article
import logging

def fetch_announcement_text(url: str) -> str:
    """下载并解析公告正文，失败时返回空字符串"""
    try:
        art = Article(url, language="zh")
        art.download()
        art.parse()
        return art.text.replace("\n", " ")
    except Exception as e:
        logging.warning(f"抓取公告失败 {url}: {e}")
        return ""
