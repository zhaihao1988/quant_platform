import os
import pandas as pd
import requests
import time

# 设置 DeepSeek 模型的 API 端点
DEEPSEEK_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"  # 根据实际使用的模型名称进行调整

# 读取 CSV 文件
csv_file = "output/final_signals_2025-04-25.csv"
df = pd.read_csv(csv_file)

# 确保输出目录存在
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个股票代码
for index, row in df.iterrows():
    symbol = row["symbol"]
    signal_date = row["signal_date"]
    strategy = row["strategy"]
    timeframe = row["timeframe"]

    # 构建提示词
    prompt = (
        f"请分析股票代码 {symbol} 在 {signal_date} 使用策略 {strategy} 和时间框架 {timeframe} 下的表现，"
        "包括基本面和技术面的分析，并提供投资建议。"
    )

    # 构建请求数据
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt
    }

    try:
        # 发送请求到 DeepSeek 模型
        response = requests.post(DEEPSEEK_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        analysis = result.get("response", "").strip()

        # 保存分析报告到文件
        output_file = os.path.join(output_dir, f"{symbol}_analysis.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"股票代码: {symbol}\n")
            f.write(f"信号日期: {signal_date}\n")
            f.write(f"策略: {strategy}\n")
            f.write(f"时间框架: {timeframe}\n\n")
            f.write("分析报告:\n")
            f.write(analysis)

        print(f"已生成 {symbol} 的分析报告。")

        # 为了避免对模型服务的请求过于频繁，添加延时
        time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"分析 {symbol} 时发生错误: {e}")
