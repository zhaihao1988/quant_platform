# integrations/email_sender.py
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# Import settings
from config import settings

logger = logging.getLogger(__name__)

def send_email(subject: str, body: str, receiver_email: str) -> bool:
    """
    Sends an email using configuration from settings.
    Uses SSL for port 465 or STARTTLS for port 587/25.
    """
    sender_email = settings.EMAIL_USER
    # **CRITICAL SECURITY WARNING:** EMAIL_PASS MUST be the Authorization Code from 126/163 Mail, NOT the login password.
    sender_password = settings.EMAIL_PASS
    smtp_server = settings.EMAIL_SMTP_SERVER
    smtp_port = settings.EMAIL_SMTP_PORT

    # Validate settings
    if not all([sender_email, sender_password, smtp_server, smtp_port]):
        logger.error("Email settings are incomplete in config/settings.py or .env file. Cannot send email.")
        return False
    if "@" not in sender_email: # Basic validation
        logger.error(f"Invalid sender email format: {sender_email}")
        return False

    # Log a warning if the default/example password might be in use
    if sender_password == "YOUR_AUTHORIZATION_CODE" or sender_password == "glmdfpoaA8":
        logger.critical("EMAIL_PASS seems to be using a placeholder/example value. "
                        "Please generate and set the 126/163 Mail Authorization Code in your .env file.")
        # return False # Optionally prevent sending with placeholder password

    # Construct email message
    message = MIMEMultipart()
    message['From'] = Header(f"Stock Analysis Platform <{sender_email}>", 'utf-8')
    message['To'] = Header(receiver_email, 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    # Attach the body as plain text (can be changed to 'html' if needed)
    message.attach(MIMEText(body, 'plain', 'utf-8'))

    server = None # Initialize server to None
    try:
        logger.info(f"Attempting to send email via {smtp_server}:{smtp_port} to {receiver_email}")

        # Choose connection type based on port
        if smtp_port == 465:
            logger.debug("Connecting using SMTP_SSL (Port 465)")
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) # Use SMTP_SSL for port 465
        elif smtp_port == 587 or smtp_port == 25:
            logger.debug("Connecting using SMTP and STARTTLS (Port 587/25)")
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.ehlo() # Identify client to server
            server.starttls() # Upgrade connection to TLS
            server.ehlo() # Re-identify after TLS
            logger.info("STARTTLS encryption established.")
        else:
            logger.error(f"Unsupported SMTP port: {smtp_port}. Use 465 (SSL) or 587/25 (STARTTLS).")
            return False

        # Login using sender email and authorization code
        logger.info(f"Logging in as {sender_email}...")
        server.login(sender_email, sender_password)
        logger.info("SMTP login successful.")

        # Send the email
        logger.info(f"Sending email with subject: '{subject}'")
        server.sendmail(sender_email, [receiver_email], message.as_string())
        logger.info("Email sent successfully.")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Failed: {e}. Verify EMAIL_USER and EMAIL_PASS (Authorization Code). "
                     "Ensure 'SMTP Service' is enabled in your 126/163 Mail settings.", exc_info=True)
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error occurred: {e}", exc_info=True)
        return False
    except ConnectionRefusedError:
         logger.error(f"Connection refused by SMTP server {smtp_server}:{smtp_port}. Check server/port and firewall.")
         return False
    except TimeoutError:
         logger.error(f"Connection timed out to SMTP server {smtp_server}:{smtp_port}.")
         return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during email sending: {e}", exc_info=True)
        return False
    finally:
        # Ensure the connection is closed
        if server:
            try:
                server.quit()
                logger.debug("SMTP connection closed.")
            except Exception as e_quit:
                logger.error(f"Error closing SMTP connection: {e_quit}")