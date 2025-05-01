# utils/scraper.py

import logging
import requests
from newspaper import Article
from io import BytesIO
from PyPDF2 import PdfReader

def fetch_announcement_text(url: str) -> str:
    """Download and parse HTML or PDF announcements."""
    try:
        if url.lower().endswith(".pdf"):
            # 1) Download PDF
            resp = requests.get(url)
            resp.raise_for_status()
            # 2) Read PDF pages
            reader = PdfReader(BytesIO(resp.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            art = Article(url, language="zh")
            art.download()
            art.parse()
            text = art.text
        return text.replace("\n", " ")
    except Exception as e:
        logging.warning(f"抓取公告失败 {url}: {e}")
        return ""
