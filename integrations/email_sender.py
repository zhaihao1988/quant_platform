# integrations/email_sender.py
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# 导入您的配置
from config import settings

logger = logging.getLogger(__name__)

def send_email(subject: str, body: str, receiver_email: str):
    """
    使用配置文件中的 126 邮箱设置发送邮件。
    注意：端口 25 通常使用 STARTTLS 加密。
    强烈建议 EMAIL_PASS 使用 126 邮箱生成的授权码而非登录密码。
    """
    sender_email = settings.EMAIL_USER
    sender_password = settings.EMAIL_PASS # 应为授权码
    smtp_server = settings.EMAIL_SMTP_SERVER
    smtp_port = settings.EMAIL_SMTP_PORT # 您的配置是 25

    if not all([sender_email, sender_password, smtp_server, smtp_port]):
        logger.error("Email settings are incomplete. Cannot send email.")
        return False

    if not sender_password or sender_password == "glmdfpoaA8": # 避免使用示例密码
         logger.warning("EMAIL_PASS seems to be using the example password or is empty. Please use a 126 Mail authorization code.")
         # return False # 可以选择在这里阻止发送

    message = MIMEMultipart()
    # 使用 Header 对象处理中文，避免乱码
    message['From'] = Header(f"Quant Platform <{sender_email}>", 'utf-8')
    message['To'] = Header(receiver_email, 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    # 添加邮件正文
    message.attach(MIMEText(body, 'plain', 'utf-8')) # 发送纯文本，如果需要 HTML 改为 'html'

    try:
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        # 连接到 SMTP 服务器 (端口 25 通常不用 SSL)
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=20) # 设置超时

        # 启用 TLS 加密 (对于端口 25/587)
        server.ehlo() # 确认服务器身份
        server.starttls() # 启动 TLS
        server.ehlo() # 再次确认身份 (TLS 后)
        logger.info("TLS encryption started.")

        # 登录邮箱 (使用授权码)
        logger.info(f"Logging in as {sender_email}...")
        server.login(sender_email, sender_password)
        logger.info("SMTP login successful.")

        # 发送邮件
        logger.info(f"Sending email to {receiver_email} with subject: {subject}")
        server.sendmail(sender_email, [receiver_email], message.as_string())
        logger.info("Email sent successfully.")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Error: {e}. Check username/password(authorization code) and 126 Mail settings.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error sending email: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during email sending: {e}")
        return False
    finally:
        # 关闭连接
        if 'server' in locals() and server:
            try:
                server.quit()
                logger.info("SMTP connection closed.")
            except Exception as e:
                logger.error(f"Error closing SMTP connection: {e}")

# 示例用法 (可以在 main.py 中调用)
# if __name__ == "__main__":
#     # 需要设置环境变量或 .env 文件
#     test_subject = "测试邮件 - Quant Platform"
#     test_body = "这是一封来自您的量化平台项目的测试邮件。\n祝您使用愉快！"
#     test_receiver = "your_personal_email@example.com" # 替换为您的接收邮箱
#     success = send_email(test_subject, test_body, test_receiver)
#     if success:
#         print("测试邮件发送成功！")
#     else:
#         print("测试邮件发送失败，请检查日志。")