# quant_platform/core/llm_provider.py

import requests
import logging
from typing import Optional, Protocol, Dict, Any

# 采用正确的导入方式
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """
    定义语言模型提供商必须遵循的接口协议 (Interface)。
    """
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        生成文本。
        - prompt: 发送给模型的提示。
        - model: (可选) 指定要使用的模型名称。如果为 None，则使用提供商的默认模型。
        - kwargs: 其他特定于提供商的参数。
        """
        ...


class SiliconFlowProvider:
    """
    用于与 SiliconFlow API 交互的提供商类。
    """
    def __init__(self):
        self.api_key = settings.SILICONFLOW_API_KEY
        self.base_url = "https://api.siliconflow.cn/v1"
        self.default_model = settings.SILICONFLOW_MODEL
        if not self.api_key:
            raise ValueError("API key not found for SiliconFlow! Please check your settings.")

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[str]:
        """使用 SiliconFlow API 生成文本。"""
        target_model = model if model is not None else self.default_model
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
            "stream": False
        }
        try:
            logger.info(f"Sending request to SiliconFlow API with model: {target_model}...")
            proxies = {"http": None, "https": None}
            response = requests.post(endpoint, headers=headers, json=payload, timeout=300, proxies=proxies)
            response.raise_for_status()
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            logger.info("Successfully received response from SiliconFlow API.")
            return content
        except Exception as e:
            logger.error(f"Error calling SiliconFlow API: {e}", exc_info=True)
            if 'response' in locals():
                logger.error(f"Error API response content: {response.text}")
            return None


class OllamaProvider:
    """
    用于与本地 Ollama 服务交互的提供商类。
    """
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_model = settings.OLLAMA_MODEL
        logger.info(f"Ollama provider initialized. Base URL: {self.base_url}, Default Model: {self.default_model}")

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[str]:
        """使用 Ollama API 生成文本。"""
        target_model = model if model is not None else self.default_model
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_predict": kwargs.get("max_tokens", 4096),
            }
        }
        try:
            logger.info(f"Sending request to Ollama API with model: {target_model}...")
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            response_data = response.json()
            content = response_data.get('response')
            logger.info("Successfully received response from Ollama API.")
            return content
        except requests.exceptions.ConnectionError as e:
             logger.error(f"Connection to Ollama failed at {self.base_url}. Is Ollama running?", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}", exc_info=True)
            if 'response' in locals():
                logger.error(f"Error API response content: {response.text}")
            return None


class LLMProviderFactory:
    """
    【工厂类】根据配置创建并返回相应的 LLM 提供商实例。
    """
    _provider_instance: Optional[LLMProvider] = None

    @classmethod
    def get_provider(cls) -> LLMProvider:
        """
        获取一个单例的 LLM Provider。
        """
        if cls._provider_instance is None:
            provider_name = settings.LLM_PROVIDER.lower()
            logger.info(f"LLM_PROVIDER setting is '{provider_name}'. Creating provider instance...")
            if provider_name == "ollama":
                cls._provider_instance = OllamaProvider()
            elif provider_name == "siliconflow":
                cls._provider_instance = SiliconFlowProvider()
            else:
                raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Must be 'ollama' or 'siliconflow'.")
        return cls._provider_instance