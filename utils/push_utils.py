# utils/push_utils.py
import requests

def pushplus_send_message(content: str):
    """
    使用 pushplus 微信推送。
    用户已给出 token 与接口。
    """
    token = "fa964c8140ad4bf4802595f958469c6d"
    title = "买点提醒"
    api_url = f"https://www.pushplus.plus/send?token={token}&title={title}&content={content}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            print("✅ 微信推送成功")
        else:
            print(f"❌ 推送失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"❌ 推送异常：{e}")
