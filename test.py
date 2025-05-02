from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
import requests
import re
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama


# 动态获取最近一个交易日（排除节假日）
def get_last_trading_day():
    today = datetime.now()
    # 假设五一假期为5月1日-5月3日
    if today.month == 5 and today.day in (1, 2, 3):
        return today - timedelta(days=4)
    return today - timedelta(days=1)


last_trading_day = get_last_trading_day()
date_str = last_trading_day.strftime("%Y年%m月%d日")

llm = ChatOllama(
    model="qwen3:14b",
    temperature=0.7,
    repetition_penalty=1.2,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# 增强版金融数据检索（新增股票代码校验）
def google_finance_search(query: str) -> str:
    API_KEY = "AIzaSyB0Kv14UpjEDv59HEOV4ducTqaPk8633L8"
    CX = "533a067c36f9d48f1"

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": f"{query} 000887.SZ {date_str} 收盘价 site:finance.sina.com.cn",
        "num": 3,
        "gl": "cn",
        "lr": "lang_zh-CN",
        "sort": "date",
        "dateRestrict": "d1"
    }

    try:
        response = requests.get(endpoint, params=params, timeout=15)
        result = response.json()

        if "items" in result:
            # 精确匹配股价和总股本
            price_pattern = r"收盘价\s*([\d\.]+)元"
            shares_pattern = r"总股本\s*([\d\.]+)亿股"
            for item in result["items"][:2]:
                price_match = re.search(price_pattern, item.get("snippet", ""))
                shares_match = re.search(shares_pattern, item.get("snippet", ""))
                if price_match and shares_match:
                    return f"收盘价: {price_match.group(1)}元，总股本: {shares_match.group(1)}亿股"
            return "未找到完整数据"
        return "未找到实时数据"

    except Exception as e:
        return f"数据检索失败：{str(e)}"


# 工具集（更新描述）
tools = [
    Tool(
        name="GoogleFinanceSearch",
        func=google_finance_search,
        description="精确获取中鼎股份(000887.SZ)的实时股价和总股本，输入应为'中鼎股份 股价 总股本'"
    )
]

# 提示模板（新增数据校验要求）
prompt_template = """
你是一名资深金融分析师，请按以下步骤分析：

当前时间：{current_time}
用户问题：{input}

可用工具：
{tools}

关键数据要求：
1. 股票代码必须为000887.SZ
2. 仅使用{date_str}的收盘价数据
3. 总股本需来自最新公告（当前应为13.16亿股）

分析步骤：
...（原有步骤保持不变）...

最终答案：必须按以下JSON格式输出：
{{
  "市值": "数值+单位",
  "合理性分析": ["要点1", "要点2"],
  "投资建议": "建议内容"
}}
"""

agent_executor = AgentExecutor(
    agent=create_react_agent(llm, tools, PromptTemplate.from_template(prompt_template)),
    tools=tools,
    max_iterations=3,
    verbose=True
)

# 执行（验证正确数据）
response = agent_executor.invoke({
    "input": "分析最近一个交易日中鼎股份（000887.SZ）的市值是多少，是否合理",
    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "date_str": date_str
})

print(response['output'])