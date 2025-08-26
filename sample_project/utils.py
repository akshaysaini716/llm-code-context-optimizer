"""
Utility Functions
Common utility functions used across the application
"""

import os
import json
import yaml
import hashlib
import uuid
import time
import functools
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configuration utilities
def load_config(config_path: str, config_type: str = "auto") -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        config_type: Type of config file (json, yaml, auto)
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect file type
    if config_type == "auto":
        if path.suffix.lower() in ['.yml', '.yaml']:
            config_type = "yaml"
        elif path.suffix.lower() == '.json':
            config_type = "json"
        else:
            raise ValueError(f"Cannot auto-detect config type for {path.suffix}")
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            if config_type == "yaml":
                return yaml.safe_load(file)
            elif config_type == "json":
                return json.load(file)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")
    
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")

def save_config(config: Dict[str, Any], config_path: str, config_type: str = "auto") -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        config_type: Type of config file (json, yaml, auto)
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(config_path)
    
    # Auto-detect file type
    if config_type == "auto":
        if path.suffix.lower() in ['.yml', '.yaml']:
            config_type = "yaml"
        elif path.suffix.lower() == '.json':
            config_type = "json"
        else:
            config_type = "json"  # Default to JSON
    
    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as file:
            if config_type == "yaml":
                yaml.dump(config, file, default_flow_style=False, indent=2)
            elif config_type == "json":
                json.dump(config, file, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        return False

# File utilities
def ensure_directory(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_size(file_path: str) -> Optional[int]:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        return None

def get_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    """
    Calculate file hash
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        File hash or None if error
    """
    try:
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return None

def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copy file from source to destination
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import shutil
        
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source}")
            return False
        
        if dest_path.exists() and not overwrite:
            logger.error(f"Destination file exists and overwrite is False: {destination}")
            return False
        
        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source, destination)
        return True
    
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False

# String utilities
def generate_random_string(length: int = 10, include_numbers: bool = True, 
                         include_symbols: bool = False) -> str:
    """
    Generate random string
    
    Args:
        length: Length of string
        include_numbers: Include numbers in string
        include_symbols: Include symbols in string
        
    Returns:
        Random string
    """
    import random
    import string
    
    chars = string.ascii_letters
    
    if include_numbers:
        chars += string.digits
    
    if include_symbols:
        chars += "!@#$%^&*"
    
    return ''.join(random.choice(chars) for _ in range(length))

def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to URL-friendly slug
    
    Args:
        text: Text to slugify
        max_length: Maximum length of slug
        
    Returns:
        Slugified text
    """
    import re
    
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    
    # Trim to max length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

# Date/time utilities
def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object to string
    
    Args:
        dt: Datetime object
        format_string: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_string)

def parse_datetime(date_string: str, format_string: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse datetime string to datetime object
    
    Args:
        date_string: Datetime string
        format_string: Format string
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(date_string, format_string)
    except ValueError:
        return None

def get_timestamp() -> int:
    """Get current timestamp in seconds"""
    return int(time.time())

def get_iso_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + 'Z'

def time_ago(dt: datetime) -> str:
    """
    Get human-readable time difference
    
    Args:
        dt: Datetime to compare with now
        
    Returns:
        Human-readable time difference
    """
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"

# Validation utilities
def is_valid_email(email: str) -> bool:
    """
    Validate email address
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_url(url: str) -> bool:
    """
    Validate URL
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    import re
    pattern = r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

def is_valid_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID, False otherwise
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

# Decorators
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for functions
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator

def timing(func: Callable) -> Callable:
    """
    Timing decorator to measure function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper

def cache_result(ttl: int = 300):
    """
    Cache function result for specified time
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            return result
        
        return wrapper
    return decorator

# Context managers
@contextmanager
def temporary_directory():
    """
    Context manager for temporary directory
    
    Yields:
        Path to temporary directory
    """
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@contextmanager
def change_directory(path: str):
    """
    Context manager to temporarily change directory
    
    Args:
        path: Directory to change to
    """
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

# Data utilities
def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result

# Environment utilities
def get_env_var(var_name: str, default: Any = None, var_type: type = str) -> Any:
    """
    Get environment variable with type conversion
    
    Args:
        var_name: Environment variable name
        default: Default value if not found
        var_type: Type to convert to
        
    Returns:
        Environment variable value or default
    """
    value = os.getenv(var_name)
    
    if value is None:
        return default
    
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return var_type(value)
    except (ValueError, TypeError):
        return default

def set_env_var(var_name: str, value: Any) -> None:
    """
    Set environment variable
    
    Args:
        var_name: Environment variable name
        value: Value to set
    """
    os.environ[var_name] = str(value)

# Async utilities
async def run_in_executor(func: Callable, *args, executor=None) -> Any:
    """
    Run synchronous function in executor
    
    Args:
        func: Function to run
        *args: Function arguments
        executor: Executor to use
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def gather_with_concurrency(n: int, *tasks):
    """
    Run tasks with limited concurrency
    
    Args:
        n: Maximum concurrent tasks
        *tasks: Tasks to run
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))

# Logging utilities
def setup_logger(name: str, level: str = "INFO", 
                format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    """
    Setup logger with specified configuration
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Log format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
