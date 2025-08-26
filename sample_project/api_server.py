"""
REST API Server
FastAPI-based web server with authentication and CRUD operations
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import logging

from user_management import UserManager, UserRole, UserProfile
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sample Project API",
    description="A comprehensive REST API for user management and data operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
user_manager = UserManager("your-secret-key-here")
db_manager = DatabaseManager("api_server.db")

# Pydantic models for API
class UserRegistration(BaseModel):
    """User registration request model"""
    username: str
    password: str
    email: EmailStr
    full_name: str
    role: Optional[UserRole] = UserRole.USER

class UserLogin(BaseModel):
    """User login request model"""
    username: str
    password: str

class UserUpdate(BaseModel):
    """User update request model"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

class PasswordChange(BaseModel):
    """Password change request model"""
    old_password: str
    new_password: str

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Any] = None

class UserResponse(BaseModel):
    """User profile response model"""
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
    """
    Get current user from JWT token
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        UserProfile of authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    user = user_manager.get_user_from_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    return user

async def get_admin_user(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """
    Ensure current user has admin privileges
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserProfile if user is admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Authentication endpoints
@app.post("/auth/register", response_model=APIResponse)
async def register_user(user_data: UserRegistration):
    """
    Register a new user
    
    Args:
        user_data: User registration data
        
    Returns:
        API response with success status
    """
    try:
        user_profile = user_manager.register_user(
            username=user_data.username,
            password=user_data.password,
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role
        )
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        return APIResponse(
            success=True,
            message="User registered successfully",
            data={"user_id": user_profile.user_id}
        )
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login", response_model=APIResponse)
async def login_user(login_data: UserLogin):
    """
    Authenticate user and return JWT token
    
    Args:
        login_data: User login credentials
        
    Returns:
        API response with JWT token
    """
    try:
        token = user_manager.authenticate_user(
            username=login_data.username,
            password=login_data.password
        )
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        return APIResponse(
            success=True,
            message="Login successful",
            data={"token": token, "token_type": "bearer"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_profile(current_user: UserProfile = Depends(get_current_user)):
    """
    Get current user's profile
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User profile data
    """
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        is_active=current_user.is_active
    )

# User management endpoints
@app.put("/users/profile", response_model=APIResponse)
async def update_user_profile(
    update_data: UserUpdate,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Update current user's profile
    
    Args:
        update_data: Profile update data
        current_user: Current authenticated user
        
    Returns:
        API response with success status
    """
    try:
        # Convert update data to dict, excluding None values
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        
        success = user_manager.update_user_profile(current_user.username, **update_dict)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Profile update failed"
            )
        
        return APIResponse(
            success=True,
            message="Profile updated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@app.put("/users/password", response_model=APIResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Change user password
    
    Args:
        password_data: Password change data
        current_user: Current authenticated user
        
    Returns:
        API response with success status
    """
    try:
        success = user_manager.change_password(
            username=current_user.username,
            old_password=password_data.old_password,
            new_password=password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )
        
        return APIResponse(
            success=True,
            message="Password changed successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
async def list_all_users(admin_user: UserProfile = Depends(get_admin_user)):
    """
    Get list of all users (admin only)
    
    Args:
        admin_user: Current admin user
        
    Returns:
        List of all user profiles
    """
    try:
        users = list(user_manager.users.values())
        return [
            UserResponse(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                created_at=user.created_at,
                last_login=user.last_login,
                is_active=user.is_active
            )
            for user in users
        ]
    
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@app.put("/admin/users/{username}/deactivate", response_model=APIResponse)
async def deactivate_user(
    username: str,
    admin_user: UserProfile = Depends(get_admin_user)
):
    """
    Deactivate a user account (admin only)
    
    Args:
        username: Username to deactivate
        admin_user: Current admin user
        
    Returns:
        API response with success status
    """
    try:
        success = user_manager.deactivate_user(username)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return APIResponse(
            success=True,
            message=f"User {username} deactivated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deactivation failed"
        )

# Database endpoints
@app.get("/admin/database/stats", response_model=APIResponse)
async def get_database_stats(admin_user: UserProfile = Depends(get_admin_user)):
    """
    Get database statistics (admin only)
    
    Args:
        admin_user: Current admin user
        
    Returns:
        API response with database statistics
    """
    try:
        stats = db_manager.get_statistics()
        return APIResponse(
            success=True,
            message="Database statistics retrieved",
            data=stats
        )
    
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database statistics"
        )

# Health check endpoints
@app.get("/health", response_model=APIResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        API response with health status
    """
    return APIResponse(
        success=True,
        message="API server is healthy",
        data={
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    )

@app.get("/health/database", response_model=APIResponse)
async def database_health_check():
    """
    Database health check endpoint
    
    Returns:
        API response with database health status
    """
    try:
        # Simple query to test database connection
        result = db_manager.execute_query("SELECT 1 as test")
        
        if result.success:
            return APIResponse(
                success=True,
                message="Database is healthy",
                data={"execution_time": result.execution_time}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is not responding"
            )
    
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database health check failed"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return APIResponse(
        success=False,
        message=exc.detail
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return APIResponse(
        success=False,
        message="Internal server error"
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting API server...")
    
    # Create default admin user if not exists
    admin_user = user_manager.register_user(
        username="admin",
        password="admin123",  # In production, use secure password
        email="admin@example.com",
        full_name="System Administrator",
        role=UserRole.ADMIN
    )
    
    if admin_user:
        logger.info("Default admin user created")
    else:
        logger.info("Admin user already exists")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down API server...")
    db_manager.close()

# Main function to run the server
def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    run_server(debug=True)
