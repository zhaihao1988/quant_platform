from anythingllm import LLM

# 本地模型示例：使用 Ollama 提供的 deepseek-r1:14b
model = LLM(provider='Ollama', model_name='deepseek-r1:14b')
# 也可以配置 HTTP API endpoint
# model = LLM(api_url='http://localhost:8000/api', api_key='YOUR_API_KEY')

response = model.chat("1 + 1 = ?")
print(response)
