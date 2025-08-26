"""
User Management System
Handles user authentication, authorization, and profile management
"""

import hashlib
import jwt
import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

class UserRole(Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    created_at: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    is_active: bool = True
    preferences: Dict = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

class PasswordManager:
    """Handles password hashing and verification"""
    
    def __init__(self, salt_length: int = 32):
        self.salt_length = salt_length
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash a password with salt
        
        Args:
            password: Plain text password
            salt: Optional salt, generates new if None
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = self._generate_salt()
        
        # Combine password and salt
        salted_password = password + salt
        
        # Hash using SHA-256
        hashed = hashlib.sha256(salted_password.encode()).hexdigest()
        
        return hashed, salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password to verify
            hashed_password: Stored hash
            salt: Salt used for hashing
            
        Returns:
            True if password matches
        """
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == hashed_password
    
    def _generate_salt(self) -> str:
        """Generate a random salt"""
        import secrets
        return secrets.token_hex(self.salt_length)

class TokenManager:
    """Handles JWT token creation and validation"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = datetime.timedelta(hours=24)
    
    def create_token(self, user_profile: UserProfile) -> str:
        """
        Create a JWT token for user
        
        Args:
            user_profile: User profile to encode in token
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_profile.user_id,
            'username': user_profile.username,
            'role': user_profile.role.value,
            'exp': datetime.datetime.utcnow() + self.token_expiry,
            'iat': datetime.datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate and decode JWT token
        
        Args:
            token: JWT token to validate
            
        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh an existing token
        
        Args:
            token: Current token to refresh
            
        Returns:
            New token or None if current token invalid
        """
        payload = self.validate_token(token)
        if not payload:
            return None
        
        # Create new token with updated expiry
        payload['exp'] = datetime.datetime.utcnow() + self.token_expiry
        payload['iat'] = datetime.datetime.utcnow()
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

class UserManager:
    """Main user management class"""
    
    def __init__(self, secret_key: str):
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager(secret_key)
        self.users: Dict[str, UserProfile] = {}
        self.user_credentials: Dict[str, Dict] = {}  # username -> {hash, salt}
    
    def register_user(self, username: str, password: str, email: str, 
                     full_name: str, role: UserRole = UserRole.USER) -> Optional[UserProfile]:
        """
        Register a new user
        
        Args:
            username: Unique username
            password: Plain text password
            email: User email
            full_name: User's full name
            role: User role (default: USER)
            
        Returns:
            UserProfile if successful, None if username exists
        """
        if username in self.users:
            return None
        
        # Generate user ID
        user_id = self._generate_user_id()
        
        # Hash password
        hashed_password, salt = self.password_manager.hash_password(password)
        
        # Create user profile
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            created_at=datetime.datetime.utcnow()
        )
        
        # Store user data
        self.users[username] = user_profile
        self.user_credentials[username] = {
            'hash': hashed_password,
            'salt': salt
        }
        
        return user_profile
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and return token
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        if username not in self.users or username not in self.user_credentials:
            return None
        
        credentials = self.user_credentials[username]
        if not self.password_manager.verify_password(
            password, credentials['hash'], credentials['salt']
        ):
            return None
        
        # Update last login
        user_profile = self.users[username]
        user_profile.last_login = datetime.datetime.utcnow()
        
        # Create and return token
        return self.token_manager.create_token(user_profile)
    
    def get_user_from_token(self, token: str) -> Optional[UserProfile]:
        """
        Get user profile from JWT token
        
        Args:
            token: JWT token
            
        Returns:
            UserProfile if token valid, None otherwise
        """
        payload = self.token_manager.validate_token(token)
        if not payload:
            return None
        
        username = payload.get('username')
        return self.users.get(username)
    
    def update_user_profile(self, username: str, **kwargs) -> bool:
        """
        Update user profile fields
        
        Args:
            username: Username to update
            **kwargs: Fields to update
            
        Returns:
            True if successful, False if user not found
        """
        if username not in self.users:
            return False
        
        user_profile = self.users[username]
        
        # Update allowed fields
        allowed_fields = {'email', 'full_name', 'preferences', 'is_active'}
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user_profile, field, value)
        
        return True
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.user_credentials:
            return False
        
        credentials = self.user_credentials[username]
        
        # Verify old password
        if not self.password_manager.verify_password(
            old_password, credentials['hash'], credentials['salt']
        ):
            return False
        
        # Hash new password
        new_hash, new_salt = self.password_manager.hash_password(new_password)
        
        # Update credentials
        self.user_credentials[username] = {
            'hash': new_hash,
            'salt': new_salt
        }
        
        return True
    
    def get_users_by_role(self, role: UserRole) -> List[UserProfile]:
        """
        Get all users with specific role
        
        Args:
            role: User role to filter by
            
        Returns:
            List of user profiles with the specified role
        """
        return [user for user in self.users.values() if user.role == role]
    
    def deactivate_user(self, username: str) -> bool:
        """
        Deactivate a user account
        
        Args:
            username: Username to deactivate
            
        Returns:
            True if successful, False if user not found
        """
        if username not in self.users:
            return False
        
        self.users[username].is_active = False
        return True
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        import uuid
        return str(uuid.uuid4())

# Authorization decorators and utilities
def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    def decorator(func):
        def wrapper(self, token: str, *args, **kwargs):
            user = self.get_user_from_token(token)
            if not user or not user.is_active:
                raise PermissionError("Invalid or inactive user")
            
            # Check role hierarchy
            role_hierarchy = {
                UserRole.GUEST: 0,
                UserRole.USER: 1,
                UserRole.MODERATOR: 2,
                UserRole.ADMIN: 3
            }
            
            if role_hierarchy.get(user.role, 0) < role_hierarchy.get(required_role, 0):
                raise PermissionError(f"Insufficient permissions. Required: {required_role.value}")
            
            return func(self, token, *args, **kwargs)
        return wrapper
    return decorator

class AuthorizedUserManager(UserManager):
    """User manager with authorization decorators"""
    
    @require_role(UserRole.ADMIN)
    def delete_user(self, token: str, username: str) -> bool:
        """Delete a user (admin only)"""
        if username in self.users:
            del self.users[username]
        if username in self.user_credentials:
            del self.user_credentials[username]
        return True
    
    @require_role(UserRole.MODERATOR)
    def ban_user(self, token: str, username: str) -> bool:
        """Ban a user (moderator or admin)"""
        return self.deactivate_user(username)
    
    @require_role(UserRole.USER)
    def view_profile(self, token: str, username: str) -> Optional[UserProfile]:
        """View user profile (authenticated users only)"""
        return self.users.get(username)
