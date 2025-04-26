import subprocess

# 清华 PyPI 镜像地址
tsinghua_url = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 所需依赖包列表
packages = [
    "akshare",
    "pandas",
    "numpy",
    "sqlalchemy",
    "psycopg2-binary",
    "scikit-learn",
    "jieba",
    "requests",
    "beautifulsoup4",
    "rqalpha",
    "streamlit",
]

# 安装函数
def install_packages():
    for pkg in packages:
        print(f"Installing {pkg} from 清华源...")
        subprocess.check_call(["pip", "install", pkg, "-i", tsinghua_url])

if __name__ == "__main__":
    install_packages()
