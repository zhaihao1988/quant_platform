# utils/emailer.py

import yagmail
from config.settings import settings

def send_report_email(to: str, subject: str, body: str):
    yag = yagmail.SMTP(user=settings.EMAIL_USER, password=settings.EMAIL_PASS,
                       host=settings.EMAIL_SMTP_SERVER, port=settings.EMAIL_SMTP_PORT)
    yag.send(to=to, subject=subject, contents=body)
