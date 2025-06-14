# quant_platform/tests/test_llm.py

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 采用正确的导入方式
from core.llm_provider import SiliconFlowProvider


def test_api_connection():
    """
    测试与 SiliconFlow API 的连接和基本调用。
    """
    print("\n" + "=" * 50)
    print("🚀 开始测试 SiliconFlow API 连接...")
    print("=" * 50)

    try:
        provider = SiliconFlowProvider()
        prompt = "你好，请用中文说一句话，证明你是一个AI模型。"
        test_model = "Qwen/Qwen3-8B"
        print(f"正在调用模型: {test_model}...")
        response = provider.generate(prompt, model=test_model)

        if response:
            print("\n🎉🎉🎉 连接成功！🎉🎉🎉")
            print(f"AI回复: “{response}”")
        else:
            print("\n❌❌❌ 连接失败 ❌❌❌")
            print("未能从AI获取到有效的回复，请检查上方日志中的详细错误信息。")

    except Exception as e:
        print(f"\n❌ 测试过程中发生严重错误: {e}")

    print("\n" + "=" * 50)
    print("测试结束。")


if __name__ == "__main__":
    test_api_connection()