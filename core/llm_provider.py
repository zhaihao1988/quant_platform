# quant_platform/core/llm_provider.py

import requests
import logging
from typing import Optional

# 采用正确的导入方式
from config.settings import settings

logger = logging.getLogger(__name__)


class SiliconFlowProvider:
    """
    用于与 SiliconFlow API 交互的提供商类。
    """

    def __init__(self):
        # 直接从导入的 settings 实例获取配置
        self.api_key = settings.SILICONFLOW_API_KEY
        self.base_url = "https://api.siliconflow.cn/v1"

        if not self.api_key:
            raise ValueError("API key not found! Please check your .env file and ensure SILICONFLOW_API_KEY is set.")

    def generate(self, prompt: str, model: str = "Qwen/Qwen3-8B", **kwargs) -> Optional[str]:
        # ... generate 函数的内部逻辑保持不变 ...
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
            "stream": False
        }
        try:
            logger.info(f"正在向 SiliconFlow API 发送请求，使用模型: {model}...")
            proxies = {
               "http": None,
               "https": None,
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=300, proxies=proxies)
            response.raise_for_status()
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            logger.info("成功从 SiliconFlow API 获取到响应。")
            return content
        except Exception as e:
            logger.error(f"调用AI时发生错误: {e}", exc_info=True)
            if 'response' in locals():
                logger.error(f"错误的API响应内容: {response.text}")
            return None