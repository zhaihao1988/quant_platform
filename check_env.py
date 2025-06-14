# quant_platform/check_env.py

from dotenv import load_dotenv
import os

# 手动加载位于当前目录的 .env 文件
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"正在尝试从以下路径加载 .env 文件: {dotenv_path}")

# load_dotenv 会自动寻找 .env 文件
success = load_dotenv()

if success:
    print("✅ .env 文件被成功找到并加载！")
else:
    print("❌ 警告: 未能找到 .env 文件。")

# 尝试读取密钥
api_key = os.getenv("SILICONFLOW_API_KEY")

print("-" * 30)
if api_key:
    print(f"🎉 成功从环境变量中读取到密钥！")
    print(f"   密钥的前5位是: {api_key[:5]}...")
else:
    print(f"🔥 失败! 在环境变量中找不到 'SILICONFLOW_API_KEY'。")
    print("   请再次检查 .env 文件的位置、文件名和文件内容。")