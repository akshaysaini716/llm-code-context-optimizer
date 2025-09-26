"""
Email service module for sending notifications
Contains several bugs for testing RAG system debugging capabilities
"""

import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime
import logging
from config import get_config

# BUG 1: Missing import for email.mime.base - will cause runtime error
# from email.mime.base import MIMEBase

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails with various notification types"""
    
    def __init__(self):
        self.config = get_config()
        # BUG 2: Hardcoded credentials - security issue
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = "admin@company.com"  
        self.password = "hardcoded_password123!"  # This is terrible!
        self.connection = None
        
    def connect(self):
        """Establish SMTP connection"""
        try:
            self.connection = smtplib.SMTP(self.smtp_host, self.smtp_port)
            self.connection.starttls()
            self.connection.login(self.username, self.password)
            logger.info("SMTP connection established")
        except Exception as e:
            # BUG 3: Generic exception handling - too broad
            logger.error(f"Failed to connect: {e}")
            # BUG 4: Not re-raising exception, caller won't know about failure
            pass
    
    def disconnect(self):
        """Close SMTP connection"""
        if self.connection:
            self.connection.quit()
            # BUG 5: Not setting connection to None after closing
            logger.info("SMTP connection closed")
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        # BUG 6: Weak email validation regex - allows invalid emails
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def send_welcome_email(self, user_email: str, username: str):
        """Send welcome email to new user"""
        if not self.validate_email(user_email):
            raise ValueError("Invalid email address")
        
        # BUG 7: String formatting vulnerability - could be XSS in HTML emails
        html_body = f"""
        <html>
            <body>
                <h1>Welcome {username}!</h1>
                <p>Thanks for joining our platform.</p>
            </body>
        </html>
        """
        
        self.send_email(
            to_email=user_email,
            subject="Welcome to Our Platform!",
            body=html_body,
            is_html=True
        )
    
    def send_notification_batch(self, notifications: List[Dict[str, str]]):
        """Send batch notifications"""
        # BUG 8: No connection management - doesn't call connect()
        successful_sends = 0
        
        for notification in notifications:
            try:
                self.send_email(
                    to_email=notification['email'],
                    subject=notification['subject'],
                    body=notification['body']
                )
                successful_sends += 1
            except Exception as e:
                # BUG 9: Continues processing even if connection is broken
                logger.error(f"Failed to send to {notification['email']}: {e}")
                continue
        
        logger.info(f"Sent {successful_sends} out of {len(notifications)} emails")
        # BUG 10: Not disconnecting after batch send - resource leak
        
    def send_email(self, to_email: str, subject: str, body: str, is_html: bool = False):
        """Send individual email"""
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # BUG 11: No check if connection exists before using it
        text = msg.as_string()
        self.connection.sendmail(self.username, to_email, text)
        
        logger.info(f"Email sent to {to_email}")
    
    def send_password_reset(self, user_email: str, reset_token: str):
        """Send password reset email"""
        # BUG 12: Reset link uses HTTP instead of HTTPS - security issue
        reset_link = f"http://ourplatform.com/reset?token={reset_token}"
        
        body = f"""
        Hello,
        
        You requested a password reset. Click the link below:
        {reset_link}
        
        This link expires in 24 hours.
        
        Best regards,
        The Team
        """
        
        try:
            self.send_email(
                to_email=user_email,
                subject="Password Reset Request",
                body=body
            )
        except Exception:
            # BUG 13: Silent failure - user won't know reset email failed
            pass
    
    def get_email_stats(self) -> Dict[str, int]:
        """Get email sending statistics"""
        # BUG 14: This method doesn't actually track any stats
        # Returns fake data
        return {
            "sent_today": 42,  # Hardcoded values
            "failed_today": 3,
            "total_sent": 1337
        }
    
    def cleanup_old_logs(self, days_old: int = 30):
        """Clean up old email logs"""
        # BUG 15: This method doesn't actually do anything
        # but pretends to clean logs
        logger.info(f"Cleaned up logs older than {days_old} days")
        # No actual cleanup logic implemented
        pass

# BUG 16: Global instance creation - not thread-safe
email_service = EmailService()

def send_notification(email: str, message: str):
    """Utility function to send quick notifications"""
    # BUG 17: Creates new connection for each call - inefficient
    service = EmailService()
    service.connect()
    
    try:
        service.send_email(
            to_email=email,
            subject="Notification",
            body=message
        )
    finally:
        # BUG 18: Might not disconnect if send_email raises exception before this
        service.disconnect()
