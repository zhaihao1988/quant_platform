# news/nlp_analysis.py
import jieba
from collections import Counter

def analyze_news_text(text):
    """
    对新闻/公告文本做分词并简单统计高频词（去除常见停用词）。
    """
    stopwords = set(['的', '和', '是', '了', '在', '上', '公司', '投资'])
    words = [w for w in jieba.cut(text) if w.strip() and w not in stopwords]
    freq = Counter(words)
    common = freq.most_common(5)
    print("Top words:", common)
    return common

if __name__ == "__main__":
    sample = "公司主营业务实现增长，同比增长超过20%。董事会批准了新的投资计划。"
    analyze_news_text(sample)
