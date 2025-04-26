# news/news_scraper.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from db.database import get_engine
from db.models import News
from sqlalchemy.orm import sessionmaker

def scrape_sina_news(symbol):
    """
    简单爬取新浪财经个股新闻（示例）。
    """
    url = f"http://finance.sina.com.cn/realstock/company/{symbol}/nc.shtml"
    res = requests.get(url)
    res.encoding = 'gbk'
    soup = BeautifulSoup(res.text, "html.parser")
    titles = [tag.text for tag in soup.select(".datelist ul li span a")]
    dates = [tag.text for tag in soup.select(".datelist ul li span.date")]
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    for d, t in zip(dates, titles):
        date = datetime.strptime(d, "%Y年%m月%d日").date()
        content = ""  # 可进一步爬取详情页
        news = News(date=date, symbol=symbol, title=t, content=content, sentiment=None)
        session.add(news)
    session.commit()
    session.close()
    print(f"Inserted {len(titles)} news items for {symbol}")

if __name__ == "__main__":
    # 示例：抓取000001 (平安银行) 新闻
    scrape_sina_news("sz000001")
