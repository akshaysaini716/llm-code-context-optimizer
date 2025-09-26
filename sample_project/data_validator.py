"""
Data validation module for user inputs and system data
Contains various validation bugs for testing RAG debugging capabilities
"""

import re
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates various types of data input"""
    
    def __init__(self):
        # BUG 1: Hardcoded limits instead of configurable ones
        self.max_string_length = 255
        self.min_password_length = 8
        self.max_file_size = 1024 * 1024  # 1MB
        
    def validate_user_data(self, user_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate user registration data"""
        errors = {}
        
        # Validate username
        username = user_data.get('username')
        if not username:
            errors['username'] = ['Username is required']
        # BUG 2: Username validation too strict - doesn't allow valid characters
        elif not re.match(r'^[a-zA-Z]+$', username):  # Only letters, no numbers/underscores
            errors['username'] = ['Username can only contain letters']
        elif len(username) < 3:
            errors['username'] = ['Username must be at least 3 characters']
        # BUG 3: Off-by-one error in length check
        elif len(username) >= 20:  # Should be > 20, not >= 20
            errors['username'] = ['Username must be less than 20 characters']
            
        # Validate email
        email = user_data.get('email')
        if not email:
            errors['email'] = ['Email is required']
        # BUG 4: Case sensitive email check
        elif not self._is_valid_email(email):
            errors['email'] = ['Invalid email format']
        # BUG 5: No check for email length limits
            
        # Validate password
        password = user_data.get('password')
        if not password:
            errors['password'] = ['Password is required']
        else:
            password_errors = self._validate_password(password)
            if password_errors:
                errors['password'] = password_errors
                
        # Validate age
        age = user_data.get('age')
        if age is not None:
            # BUG 6: No type checking - could crash if age is string
            if age < 18:
                errors['age'] = ['Must be at least 18 years old']
            elif age > 120:
                errors['age'] = ['Invalid age']
                
        return errors
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email format is valid"""
        # BUG 7: Regex doesn't handle all valid email formats
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        # BUG 8: Case sensitive matching
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> List[str]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.min_password_length:
            errors.append(f'Password must be at least {self.min_password_length} characters')
            
        # BUG 9: Password validation logic error - uses 'or' instead of 'and'
        if (not any(c.isupper() for c in password) or
            not any(c.islower() for c in password) or  # Should be 'and'
            not any(c.isdigit() for c in password)):
            errors.append('Password must contain uppercase, lowercase, and numbers')
            
        # BUG 10: No check for special characters even though error message suggests it
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            pass  # Should add error but doesn't
            
        return errors
    
    def validate_phone_number(self, phone: str) -> bool:
        """Validate phone number format"""
        if not phone:
            return False
            
        # BUG 11: Only accepts US format, not international
        # BUG 12: Regex too strict - doesn't allow spaces or different formatting
        pattern = r'^\d{3}-\d{3}-\d{4}$'  # Only 123-456-7890 format
        return re.match(pattern, phone) is not None
    
    def validate_credit_card(self, card_number: str) -> Dict[str, bool]:
        """Validate credit card number"""
        result = {"valid": False, "card_type": None}
        
        if not card_number:
            return result
            
        # Remove spaces and hyphens
        card_number = re.sub(r'[- ]', '', card_number)
        
        # BUG 13: No check for non-digit characters after cleaning
        
        # BUG 14: Luhn algorithm implementation has off-by-one error
        if self._luhn_check(card_number):
            # Detect card type
            if card_number.startswith('4'):
                result["card_type"] = "Visa"
            elif card_number.startswith('5'):
                result["card_type"] = "MasterCard"
            # BUG 15: Missing other card types (Amex, Discover, etc.)
            
            result["valid"] = True
            
        return result
    
    def _luhn_check(self, card_number: str) -> bool:
        """Implement Luhn algorithm for credit card validation"""
        if not card_number.isdigit():
            return False
            
        total = 0
        reverse_digits = card_number[::-1]
        
        # BUG 16: Luhn algorithm implementation error
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit from right
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
            
        # BUG 17: Wrong modulo check
        return total % 10 == 0
    
    def validate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Validate date range inputs"""
        errors = []
        
        try:
            # BUG 18: Assumes specific date format without documenting it
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # BUG 19: No check for dates in the future when they shouldn't be
            if start > end:
                errors.append('Start date must be before end date')
                
            # BUG 20: Logic error - allows same start and end date when it shouldn't
            if start == end:
                pass  # Should probably add an error
                
        except ValueError:
            errors.append('Invalid date format. Use YYYY-MM-DD')
            
        return errors
    
    def validate_json_data(self, json_string: str) -> Dict[str, Any]:
        """Validate and parse JSON data"""
        result = {"valid": False, "data": None, "error": None}
        
        try:
            data = json.loads(json_string)
            result["valid"] = True
            result["data"] = data
        except json.JSONDecodeError as e:
            # BUG 21: Exposing too much error detail to user
            result["error"] = f"JSON Error: {str(e)}"
        except Exception as e:
            # BUG 22: Too broad exception handling
            result["error"] = f"Unexpected error: {str(e)}"
            
        return result
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not user_input:
            return ""
            
        # BUG 23: Incomplete XSS protection - only removes script tags
        sanitized = re.sub(r'<script.*?</script>', '', user_input, flags=re.DOTALL)
        
        # BUG 24: Doesn't handle other dangerous tags like <iframe>, <object>, etc.
        # BUG 25: Case insensitive matching not applied
        
        return sanitized
    
    def validate_file_upload(self, filename: str, file_size: int, file_type: str) -> List[str]:
        """Validate file upload parameters"""
        errors = []
        
        # Check filename
        if not filename:
            errors.append('Filename is required')
        # BUG 26: Path traversal vulnerability - doesn't check for ../
        elif len(filename) > self.max_string_length:
            errors.append('Filename too long')
            
        # Check file size
        # BUG 27: No check for negative file sizes
        if file_size > self.max_file_size:
            errors.append(f'File size exceeds {self.max_file_size} bytes')
            
        # Check file type
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'text/plain']
        # BUG 28: Case sensitive file type checking
        if file_type not in allowed_types:
            errors.append('File type not allowed')
            
        return errors
    
    def calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)"""
        if not password:
            return 0
            
        score = 0
        
        # Length bonus
        # BUG 29: Integer overflow possible with very long passwords
        score += len(password) * 2
        
        # Character variety bonuses
        if any(c.islower() for c in password):
            score += 5
        if any(c.isupper() for c in password):
            score += 5
        if any(c.isdigit() for c in password):
            score += 5
        if any(c in '!@#$%^&*()' for c in password):
            score += 10
            
        # BUG 30: No upper limit - score can exceed 100
        return score

def batch_validate_emails(email_list: List[str]) -> Dict[str, bool]:
    """Validate a list of emails"""
    validator = DataValidator()
    results = {}
    
    # BUG 31: No input validation - could crash on None input
    for email in email_list:
        results[email] = validator._is_valid_email(email)
        
    return results

# BUG 32: Function creates validator instance every time - inefficient
def quick_validate_user(username: str, email: str, password: str) -> bool:
    """Quick validation for user data"""
    validator = DataValidator()
    user_data = {
        'username': username,
        'email': email,
        'password': password
    }
    
    errors = validator.validate_user_data(user_data)
    # BUG 33: Returns True even if there are warnings (not just errors)
    return len(errors) == 0
