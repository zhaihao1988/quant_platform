#!/usr/bin/env python3
import requests
import time
from typing import Optional


def check_ollama_service(base_url: str = "http://localhost:11434", timeout: int = 5) -> bool:
    """
    检查Ollama服务是否正在运行
    """
    try:
        response = requests.get(f"{base_url}/", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ollama服务连接失败: {str(e)}")
        return False


def check_model_availability(model: str = "qwen3:14b", base_url: str = "http://localhost:11434") -> Optional[dict]:
    """
    检查指定模型是否可用
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return {"available": model in models, "installed_models": models}
    except Exception as e:
        print(f"❌ 模型检查失败: {str(e)}")
    return None


def test_model_inference(model: str = "qwen3:14b", base_url: str = "http://localhost:11434") -> bool:
    """
    测试模型推理功能
    """
    test_prompt = "请用中文回答：1+1等于几？"
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            answer = response.json()["message"]["content"]
            print(f"✅ 模型测试成功\n提问: {test_prompt}\n回答: {answer}")
            return True
        else:
            print(f"❌ 模型推理失败 (HTTP {response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ 推理测试异常: {str(e)}")
    return False


def full_ollama_check():
    print("\n🔍 开始Ollama服务健康检查...")

    # 检查1: 服务是否运行
    if not check_ollama_service():
        print("\n💡 解决方案建议:")
        print("1. 请确认Ollama服务已启动 (命令行运行: ollama serve)")
        print("2. 检查防火墙是否放行11434端口")
        print("3. 如果是远程服务，请确认base_url参数正确")
        return False

    print("✅ Ollama服务正在运行")

    # 检查2: 模型是否可用
    model_status = check_model_availability()
    if not model_status:
        print("\n💡 解决方案建议:")
        print("1. 运行: ollama pull qwen3:14b")
        print("2. 运行: ollama list 确认模型存在")
        return False

    if not model_status["available"]:
        print(f"❌ 模型 qwen3:14b 未安装")
        print(f"已安装模型: {', '.join(model_status['installed_models']) or '无'}")
        print("\n💡 解决方案建议:")
        print("1. 运行: ollama pull qwen3:14b")
        print("2. 检查模型名称拼写是否正确")
        return False

    print(f"✅ 模型 qwen3:14b 已安装")

    # 检查3: 实际推理测试
    if not test_model_inference():
        print("\n💡 解决方案建议:")
        print("1. 检查GPU内存是否足够 (运行: nvidia-smi)")
        print("2. 尝试更简单的模型测试: ollama run llama2")
        print("3. 查看Ollama日志获取更多信息")
        return False

    print("\n🎉 所有检查通过，Ollama服务运行正常！")
    return True


if __name__ == "__main__":
    if full_ollama_check():
        # 获取系统信息
        try:
            sys_info = requests.get("http://localhost:11434/api/version", timeout=5).json()
            print("\n🖥️ 系统信息:")
            print(f"Ollama版本: {sys_info.get('version')}")
            print(f"API版本: {sys_info.get('api_version', 'N/A')}")
        except:
            pass
    else:
        print("\n❌ Ollama服务检查未通过，请根据上述建议解决问题")