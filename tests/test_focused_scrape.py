# test_eastmoney_f10_scrape.py
import logging
import json
import os

# 确保能正确导入修改后的 web_search 模块
# 可能需要调整 Python 路径或将测试脚本移到合适位置
try:
    # 假设测试脚本在项目根目录，模块在 integrations 子目录
    from integrations.web_search import get_web_search_results
except ImportError:
    print("Error: Could not import 'get_web_search_results'. Make sure the script path and PYTHONPATH are correct.")
    exit()

# 配置日志
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
# logging.getLogger('integrations.web_search').setLevel(logging.DEBUG) # 开启 DEBUG 获取详细信息

def test_f10_scrape():
    symbol = "000887" # 测试股票代码
    print(f"\n--- Testing Eastmoney F10 Report Scraping for symbol: {symbol} ---")

    # 调用新的抓取函数
    results = get_web_search_results(symbol) # 不再需要 company_name

    print(f"\n--- Found {len(results)} results ---")
    if not results:
        print("No results found within the last 3 months.")
        return

    output_data = []
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Source: {res.get('source', 'N/A')}")
        print(f"  Date: {res.get('date', 'N/A')}")
        print(f"  Title: {res.get('title', 'N/A')}")
        print(f"  Link: {res.get('link', 'N/A')}")
        content = res.get('content', 'N/A')
        print(f"  Content (first 300 chars):")
        print(f"    {content[:300]}{'...' if len(content) > 300 else ''}")
        print("-" * 40)
        output_data.append(res) # 添加到列表以便保存

    # (可选) 将结果保存到 JSON 文件以便查看完整内容
    output_dir = "output_test" # 可以指定输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"f10_scrape_results_{symbol}.json")

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nFull results saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")


if __name__ == "__main__":
    # 确保 .env 文件被加载 (如果 settings.py 依赖它)
    try:
      from dotenv import load_dotenv
      load_dotenv()
      print("Attempted to load .env file (if needed by dependencies).")
    except ImportError:
      print(".env file loading skipped (dotenv not installed?).")

    test_f10_scrape()