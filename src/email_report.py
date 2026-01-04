import os
import smtplib
from email.message import EmailMessage

def send_gmail(subject: str, body_text: str, body_html: str | None = None):
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    to_email = os.environ.get("EMAIL_TO")

    if not gmail_user or not gmail_pass or not to_email:
        print("Email skipped: missing Gmail environment variables")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_email

    # Always include plain text fallback
    msg.set_content(body_text)

    # Optional HTML version
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.send_message(msg)
