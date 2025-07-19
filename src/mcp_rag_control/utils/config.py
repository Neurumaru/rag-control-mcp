"""Configuration management for MCP-RAG-Control system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="mcp_rag_control", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"], description="Allowed hosts")
    max_request_size: int = Field(default=10485760, description="Max request size in bytes (10MB)")
    rate_limit_requests: int = Field(default=1000, description="Rate limit: requests per hour")
    
    @validator('allowed_hosts')
    def validate_hosts(cls, v):
        for host in v:
            if not host or len(host.strip()) == 0:
                raise ValueError('Host cannot be empty')
        return v


class MCPConfig(BaseModel):
    """MCP protocol configuration."""
    
    protocol_version: str = Field(default="1.0", description="MCP protocol version")
    timeout_seconds: float = Field(default=30.0, description="Request timeout in seconds")
    max_connections: int = Field(default=10, description="Maximum connections per adapter")
    health_check_interval: int = Field(default=60, description="Health check interval in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v
    
    @validator('max_connections')
    def validate_max_connections(cls, v):
        if v <= 0:
            raise ValueError('Max connections must be positive')
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


def validate_url(url: str) -> bool:
    """Validate MCP server URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check hostname
        if not parsed.hostname:
            return False
        
        # Check for malicious patterns
        malicious_patterns = [
            'javascript:', 'data:', 'file:', 'ftp:',
            '..', '<script', 'localhost:22', ':3389'
        ]
        
        url_lower = url.lower()
        for pattern in malicious_patterns:
            if pattern in url_lower:
                return False
        
        return True
    except Exception:
        return False


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables.
    
    Returns:
        AppConfig instance with values from environment
    """
    config_dict = {
        'debug': os.getenv('MCP_DEBUG', 'false').lower() == 'true',
        'environment': os.getenv('MCP_ENVIRONMENT', 'development'),
        'host': os.getenv('MCP_HOST', '0.0.0.0'),
        'port': int(os.getenv('MCP_PORT', '8000')),
        'workers': int(os.getenv('MCP_WORKERS', '1')),
        
        'database': {
            'host': os.getenv('MCP_DB_HOST', 'localhost'),
            'port': int(os.getenv('MCP_DB_PORT', '5432')),
            'name': os.getenv('MCP_DB_NAME', 'mcp_rag_control'),
            'user': os.getenv('MCP_DB_USER', 'postgres'),
            'password': os.getenv('MCP_DB_PASSWORD', ''),
        },
        
        'security': {
            'api_key_header': os.getenv('MCP_API_KEY_HEADER', 'X-API-Key'),
            'api_key': os.getenv('MCP_API_KEY'),
            'allowed_hosts': os.getenv('MCP_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            'max_request_size': int(os.getenv('MCP_MAX_REQUEST_SIZE', '10485760')),
            'rate_limit_requests': int(os.getenv('MCP_RATE_LIMIT', '1000')),
        },
        
        'mcp': {
            'protocol_version': os.getenv('MCP_PROTOCOL_VERSION', '1.0'),
            'timeout_seconds': float(os.getenv('MCP_TIMEOUT', '30.0')),
            'max_connections': int(os.getenv('MCP_MAX_CONNECTIONS', '10')),
            'health_check_interval': int(os.getenv('MCP_HEALTH_CHECK_INTERVAL', '60')),
            'retry_attempts': int(os.getenv('MCP_RETRY_ATTEMPTS', '3')),
            'retry_delay': float(os.getenv('MCP_RETRY_DELAY', '1.0')),
        }
    }
    
    return AppConfig(**config_dict)


def load_config_from_file(config_path: str) -> AppConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        AppConfig instance
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_file.suffix == '.json':
        import json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
    elif config_file.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    return AppConfig(**config_dict)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global configuration instance.
    
    Returns:
        AppConfig instance
    """
    global _config
    if _config is None:
        # Try to load from file first, fall back to environment
        config_file = os.getenv('MCP_CONFIG_FILE')
        if config_file and Path(config_file).exists():
            _config = load_config_from_file(config_file)
        else:
            _config = load_config_from_env()
    
    return _config


def set_config(config: AppConfig) -> None:
    """Set global configuration instance.
    
    Args:
        config: AppConfig instance to set as global
    """
    global _config
    _config = config