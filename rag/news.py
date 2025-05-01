# rag/news.py

import requests
from utils.scraper import fetch_announcement_text  # re-use PDF/HTML parser

API_KEY = "<YOUR_GOOGLE_API_KEY>"
CX = "533a067c36f9d48f1"

def search_stock_news(symbol: str, num: int = 3) -> list[str]:
    """Return top news article texts for a given stock symbol."""
    url = ("https://www.googleapis.com/customsearch/v1"
           f"?key={API_KEY}&cx={CX}&q={symbol}+stock+news&num={num}")
    resp = requests.get(url)
    articles = resp.json().get("items", [])
    snippets = []
    for item in articles:
        link = item.get("link")
        if link:
            text = fetch_announcement_text(link)
            if text:
                snippets.append(text[:800])
    return snippets
